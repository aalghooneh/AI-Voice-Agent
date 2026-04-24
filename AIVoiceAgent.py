#!/usr/bin/env python3
import os
import re
import threading
from typing import Callable, Optional

import ollama
from elevenlabs.client import ElevenLabs
from elevenlabs import stream

import assemblyai as aai
from assemblyai.streaming.v3 import (
    StreamingClient,
    StreamingClientOptions,
    StreamingParameters,
    StreamingEvents,
    TurnEvent,
    SpeechModel,
)
from assemblyai.extras import MicrophoneStream


class MutableMicStream(MicrophoneStream):
    """MicrophoneStream that sends silence while muted, draining PyAudio so it never overflows."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.muted = False
        self._silence = bytes(self._chunk_size * 2)  # paInt16 = 2 bytes/sample

    def __next__(self):
        chunk = super().__next__()          # always drain the real buffer
        return self._silence if self.muted else chunk


def read_env():
    try:
        with open(".env", "r") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, _, value = line.partition("=")
                os.environ[key.strip()] = value.strip()
    except FileNotFoundError:
        print("No .env file found — using existing environment variables.")


class AIVoiceAgent:
    def __init__(self, on_state_change: Optional[Callable] = None):
        read_env()
        self._api_key = os.getenv("ASSEMBLYAI_API_KEY")
        self.elevenlabs = ElevenLabs(api_key=os.getenv("ELEVENLABS_API_KEY"))
        self.on_state_change = on_state_change or (lambda state, text="": None)
        self._busy = False
        self._mic: Optional[MutableMicStream] = None

        self.full_transcript = [
            {
                "role": "system",
                "content": (
                    "You are a helpful, concise AI voice assistant. "
                    "Answer clearly and naturally in under 100 words."
                ),
            }
        ]

    def _emit(self, state: str, text: str = ""):
        self.on_state_change(state, text)

    def start_transcription(self):
        self._emit("listening")
        print("\n[Listening...]")

        client = StreamingClient(StreamingClientOptions(api_key=self._api_key))
        client.on(StreamingEvents.Turn, self._on_turn)
        client.on(StreamingEvents.Error, self._on_error)
        client.connect(StreamingParameters(
            sample_rate=16_000,
            speech_model=SpeechModel.universal_streaming_english,
            inactivity_timeout=300,
        ))

        self._mic = MutableMicStream(sample_rate=16_000)
        client.stream(self._mic)  # blocks until mic is closed

    def _on_turn(self, client: StreamingClient, event: TurnEvent):
        if not event.transcript or self._busy:
            return
        if event.end_of_turn:
            print(f"\nUser: {event.transcript}")
            self._emit("user", event.transcript)
            self._busy = True
            # Handle in a new thread so we don't block the read loop
            t = threading.Thread(target=self._handle_response, args=(client, event.transcript))
            t.daemon = True
            t.start()
        else:
            print(event.transcript, end="\r")

    def _on_error(self, client: StreamingClient, error):
        print(f"[Error] {error}")
        self._emit("error", str(error))

    def _handle_response(self, client: StreamingClient, user_text: str):
        # Mute mic immediately so the speaker output isn't transcribed
        if self._mic:
            self._mic.muted = True

        self._emit("thinking", user_text)
        self.full_transcript.append({"role": "user", "content": user_text})

        ollama_stream = ollama.chat(
            model="deepseek-r1:7b",
            messages=self.full_transcript,
            stream=True,
        )

        print("Agent: ", end="", flush=True)
        text_buffer = ""
        full_text = ""

        for chunk in ollama_stream:
            text_buffer += chunk["message"]["content"]
            cleaned = re.sub(r"<think>.*?</think>", "", text_buffer, flags=re.DOTALL)

            if cleaned.endswith((".", "!", "?", ":")):
                sentence = cleaned.strip()
                if sentence:
                    self._emit("speaking", sentence)
                    print(sentence, flush=True)
                    audio = self.elevenlabs.text_to_speech.stream(
                        text=sentence,
                        voice_id="EXAVITQu4vr4xnSDxMaL",
                        model_id="eleven_turbo_v2",
                    )
                    stream(audio)
                    full_text += sentence + " "
                text_buffer = ""

        remaining = re.sub(r"<think>.*?</think>", "", text_buffer, flags=re.DOTALL).strip()
        if remaining:
            self._emit("speaking", remaining)
            print(remaining, flush=True)
            audio = self.elevenlabs.text_to_speech.stream(
                text=remaining,
                voice_id="EXAVITQu4vr4xnSDxMaL",
                model_id="eleven_turbo_v2",
            )
            stream(audio)
            full_text += remaining

        self.full_transcript.append({"role": "assistant", "content": full_text.strip()})

        # Wait for room echo to decay, flush AssemblyAI's buffer, then unmute
        threading.Timer(1.2, self._resume_listening, args=(client,)).start()

    def _resume_listening(self, client: StreamingClient):
        try:
            client.force_endpoint()   # discard anything AssemblyAI buffered while muted
        except Exception:
            pass
        if self._mic:
            self._mic.muted = False
        self._busy = False
        self._emit("listening")
        print("\n[Listening...]")


if __name__ == "__main__":
    agent = AIVoiceAgent()
    agent.start_transcription()
