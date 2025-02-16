#!/usr/bin/env python3
"""
Example: Real-time STT with AssemblyAI,
DeepSeek R1 model via Ollama,
TTS with ElevenLabs,
interruptible "barge-in" when the user starts speaking again.
"""

import os
import threading
import time
import assemblyai as aai
from elevenlabs.client import ElevenLabs
from elevenlabs import stream
import ollama
import sounddevice as sd
import subprocess
from typing import Iterator


def read_env():
    """Load key-value pairs from .env into os.environ."""
    try:
        with open(".env", "r") as f:
            for line in f:
                if "=" in line:
                    key, val = line.split("=", 1)
                    os.environ[key.strip()] = val.strip()
    except Exception:
        print("Error reading .env file. Make sure it exists.")

class AIVoiceAgent:
    def __init__(self):
        # Load environment variables
        read_env()
        self.mpv_process = None

        # Set AssemblyAI API key
        aai.settings.api_key = os.getenv("ASSEMBLYAI_API_KEY")

        # ElevenLabs client
        self.client = ElevenLabs(api_key=os.getenv("ELEVENLABS_API_KEY"))

        # We'll store conversation history here
        self.full_transcript = [
            {
                "role": "system",
                "content": (
                    "You are a language model called R1 created by DeepSeek. "
                    "Answer the questions being asked in fewer than 300 characters."
                )
            }
        ]

        # The real-time transcriber object
        self.transcriber = None

        # For TTS concurrency
        self.tts_thread = None
        self.stop_tts_event = threading.Event()
        self.threads = []

    def m_stream(self, audio_stream: Iterator[bytes]) -> bytes:

        mpv_command = ["mpv", "--no-cache", "--no-terminal", "--", "fd://0"]
        self.mpv_process = subprocess.Popen(
            mpv_command,
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

        audio = b""

        for chunk in audio_stream:
            if self.stop_tts_event.is_set():
                print("[inside m_stream: AI speech interrupted!]")
                self.mpv_process.terminate()
                return
            if chunk is not None and self.mpv_process:
                self.mpv_process.stdin.write(chunk)  # type: ignore
                self.mpv_process.stdin.flush()  # type: ignore
                audio += chunk
        if self.mpv_process.stdin:
            self.mpv_process.stdin.close()
        self.mpv_process.wait()

        return audio


    def start_transcription(self):
        """
        Start real-time transcription and keep it running indefinitely
        so we can detect when user speaks again (interrupt).
        """
        print("\nReal-time transcription started.\n")
        self.transcriber = aai.RealtimeTranscriber(
            sample_rate=16_000,
            on_data=self.on_data,
            on_error=self.on_error,
            on_open=self.on_open,
            on_close=self.on_close,
        )
        self.transcriber.connect()

        # Default microphone stream from AssemblyAI
        microphone_stream = aai.extras.MicrophoneStream(sample_rate=16_000)
        self.transcriber.stream(microphone_stream)

    def on_open(self, session_opened: aai.RealtimeSessionOpened):
        print("AssemblyAI session opened:", session_opened.session_id)

    def on_data(self, transcript: aai.RealtimeTranscript):
        """
        Callback whenever AssemblyAI has new transcript data (partial or final).
        We'll use final transcripts to trigger AI responses.
        Also, if TTS is currently speaking, we'll interrupt if user has said something new.
        """
        if not transcript.text:
            return  # Nothing to do if empty text

        # If TTS is playing and user is speaking, barge-in:
        if self.tts_thread and self.tts_thread.is_alive():
            print("[inside on_data: TTS is playing and user is speaking, barge-in]")
            # Set the stop flag immediately without waiting
            self.stop_tts_event.set()
            if self.mpv_process:
                self.mpv_process.terminate()
            # Don't wait for join here - let the audio stream abort
            self.tts_thread = None  # Allow new thread to be created

        if isinstance(transcript, aai.RealtimeFinalTranscript):
            # This final transcript is what we send to the AI
            print("[User final]:", transcript.text)
            self.generate_ai_response(transcript.text)
        else:
            # Partial transcript: display if you like
            # (commented out to avoid spamming console)
            # print("[User partial]:", transcript.text, end="\r")
            pass

    def on_error(self, error: aai.RealtimeError):
        print("An error occurred:", error)

    def on_close(self):
        print("AssemblyAI session closed.")

    def generate_ai_response(self, user_text: str):
        """
        Send user's final transcript to R1 (via Ollama),
        stream the response in TTS, while still listening for barge-in.
        """
        # 1. Append user message to conversation
        self.full_transcript.append({"role": "user", "content": user_text})

        print("\nUser:", user_text)

        # 2. Call Ollama in streaming mode
        ollama_stream = ollama.chat(
            model="llama3.2:1b",  # or your model name
            messages=self.full_transcript,
            stream=True,
        )

        # 3. Use a separate thread to speak out the AI response
        self.stop_tts_event.clear()  # reset the stop flag
        self.tts_thread = None
        if self.mpv_process:
            self.mpv_process.terminate()
        
        self.tts_thread = threading.Thread(
            target=lambda: self.m_stream(self.play_response(ollama_stream))  # Add stream() call here
        )
        self.tts_thread.start()

    def play_response(self, ollama_stream):
        """
        Receives streaming data from Ollama,
        breaks on periods, sends them to ElevenLabs TTS,
        can be interrupted if self.stop_tts_event is set.
        """
        text_buffer = ""
        full_text = ""
        print("initating conv...\n")
        print(ollama_stream)

        for chunk in ollama_stream:
            # Accumulate chunk text
            if self.stop_tts_event.is_set():
                print("[AI speech interrupted!]")
                sd.stop()
                return
            text_buffer += chunk["message"]["content"]

            # Split on any sentence-ending punctuation for faster response
            if any(text_buffer.endswith(p) for p in ('.', '!', '?', '...')):
                sentence = text_buffer
                text_buffer = ""

                # TTS stream with more frequent interruption checks
                print("[AI partial]:", sentence)
                audio_stream = self.client.generate(
                    text=sentence,
                    model="eleven_turbo_v2",
                    stream=True
                )
                
                # Stream with immediate interruption support
                try:
                    for audio_chunk in audio_stream:
                        if self.stop_tts_event.is_set():
                            raise StopTTSException()
                        yield audio_chunk  # Directly yield audio chunks
                except StopTTSException:
                    print("[AI speech stopped mid-chunk]")
                    break

                full_text += sentence

        # If there's leftover text after the loop ends (no trailing period),
        # speak that as well, unless interrupted
        if text_buffer and not self.stop_tts_event.is_set():
            print("[AI partial (no period)]:", text_buffer)
            audio_stream = self.client.generate(
                text=text_buffer,
                model="eleven_turbo_v2",
                stream=True
            )
            try:
                for audio_chunk in audio_stream:
                    if self.stop_tts_event.is_set():
                        raise StopTTSException()
                        break
                    yield audio_chunk
            except StopTTSException:
                print("[AI speech stopped mid-chunk]")
            full_text += text_buffer

        # Add the AI's final text to conversation if not fully interrupted
        if full_text.strip():
            self.full_transcript.append({"role": "assistant", "content": full_text})

        print("[AI finished or interrupted]")

    def _tts_on_chunk(self, chunk_bytes: bytes):
        """
        Callback for each chunk of TTS audio.  
        We can interrupt mid-chunk by raising an exception.
        """
        if self.stop_tts_event.is_set():
            raise StopTTSException("Stop TTS requested by user barge-in.")

class StopTTSException(Exception):
    """Custom exception to break from ElevenLabs streaming mid-chunk."""
    pass

if __name__ == "__main__":
    agent = AIVoiceAgent()
    agent.start_transcription()

    # Keep main thread alive. STT runs in background threads internally.
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("Exiting...")
