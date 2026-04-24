# AI Voice Agent

A real-time AI voice agent with an animated web UI. Speak into your mic, get an intelligent response spoken back — with a reactive orb that visualizes what the agent is doing.

**Stack:** AssemblyAI (speech-to-text) · DeepSeek R1 7B via Ollama (LLM) · ElevenLabs (text-to-speech) · FastAPI + WebSocket (server) · Three.js (orb UI)

---

## Features

- Real-time speech-to-text using AssemblyAI's streaming v3 API
- AI responses from DeepSeek R1 (7B) running locally via Ollama
- Natural text-to-speech via ElevenLabs
- Animated orb UI with ocean-wave fluid shader and floating particles that reacts to agent state (listening / thinking / speaking)
- Scrollable conversation history panel alongside the orb
- Feedback loop prevention — mic is muted while the agent speaks

---

## Requirements

- Python 3.10+
- macOS or Ubuntu/Debian
- An [AssemblyAI](https://www.assemblyai.com) account with billing enabled (required for real-time streaming)
- An [ElevenLabs](https://elevenlabs.io) account

---

## Setup

### 1. Install system dependencies

**macOS**
```bash
brew install portaudio mpv ollama
```

**Ubuntu / Debian**
```bash
apt install portaudio19-dev mpv
# Install Ollama separately: https://ollama.com
```

### 2. Pull the model

```bash
ollama serve &
ollama pull deepseek-r1:7b
```

### 3. Install Python dependencies

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 4. Configure API keys

Create a `.env` file in the project root:

```
ASSEMBLYAI_API_KEY=your_assemblyai_key
ELEVENLABS_API_KEY=your_elevenlabs_key
```

> **Note:** Never commit `.env` to version control. It is already listed in `.gitignore`.

### 5. Run

```bash
uvicorn server:app --host 0.0.0.0 --port 8000
```

Open `http://localhost:8000` in your browser and start talking.

---

## Project structure

```
AIVoiceAgent.py   # Core agent — STT → LLM → TTS pipeline with state callbacks
server.py         # FastAPI server — runs agent in background thread, streams state over WebSocket
frontend/
  index.html      # Single-file UI — Three.js orb + conversation history
requirements.txt
install.sh        # Automated setup script (macOS / Ubuntu)
```

---

## How it works

1. AssemblyAI streams audio from your microphone and fires a `TurnEvent` when you finish a sentence
2. The agent mutes the mic, sends your text to DeepSeek R1 via Ollama
3. Responses are streamed sentence by sentence to ElevenLabs for TTS playback
4. Each state change (`listening` → `thinking` → `speaking`) is broadcast over WebSocket to the browser
5. The orb shifts color, wave intensity, and particle speed to match the current state
6. After speaking, the mic is unmuted and AssemblyAI's buffer is flushed to prevent echo feedback

---

## Terminal-only mode

If you don't need the browser UI:

```bash
python AIVoiceAgent.py
```
