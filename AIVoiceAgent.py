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
        microphone_stream = aai.extras.MicrophoneStream(sample_rate=16_000, device_index=1)
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
            # This means the AI is talking. Let's interrupt it.
            self.stop_tts_event.set()
            # Wait for TTS thread to exit so we don't talk over ourselves
            self.tts_thread.join()

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
        print("DeepSeek R1 is thinking...\n")
        ollama_stream = ollama.chat(
            model="llama3.2:1b",  # or your model name
            messages=self.full_transcript,
            stream=True,
        )

        # 3. Use a separate thread to speak out the AI response
        self.stop_tts_event.clear()  # reset the stop flag
        self.tts_thread = threading.Thread(
            target=self.play_response,
            args=(ollama_stream,)
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

        for chunk in ollama_stream:
            # Check if we should stop because user spoke again
            if self.stop_tts_event.is_set():
                print("[AI speech interrupted!]")
                break

            # Accumulate chunk text
            text_buffer += chunk["message"]["content"]

            # Whenever we hit a period, speak that sentence
            if text_buffer.endswith("."):
                sentence = text_buffer
                text_buffer = ""

                # TTS stream
                print("[AI partial]:", sentence)
                audio_stream = self.client.generate(
                    text=sentence,
                    model="eleven_turbo_v2",
                    stream=True
                )
                # If user interrupts mid-stream, we can only break between chunks:
                try:
                    stream(audio_stream)
                except StopTTSException:
                    print("[AI speech forcibly stopped mid-chunk]")
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
                stream(audio_stream, on_chunk=self._tts_on_chunk)
            except StopTTSException:
                print("[AI speech forcibly stopped mid-chunk]")
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
