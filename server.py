#!/usr/bin/env python3
"""
FastAPI WebSocket server that runs the AI Voice Agent in a background thread
and broadcasts state events to connected browser clients.

Run with:
    uvicorn server:app --host 0.0.0.0 --port 8000
"""
import asyncio
import json
import threading
from pathlib import Path
from typing import Set

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles

from AIVoiceAgent import AIVoiceAgent

app = FastAPI(title="AI Voice Agent")

# ── WebSocket connection manager ──────────────────────────────────────────────

class ConnectionManager:
    def __init__(self):
        self.active: Set[WebSocket] = set()
        self._loop: asyncio.AbstractEventLoop | None = None

    def set_loop(self, loop: asyncio.AbstractEventLoop):
        self._loop = loop

    async def connect(self, ws: WebSocket):
        await ws.accept()
        self.active.add(ws)

    def disconnect(self, ws: WebSocket):
        self.active.discard(ws)

    async def broadcast(self, data: dict):
        dead = set()
        for ws in self.active:
            try:
                await ws.send_text(json.dumps(data))
            except Exception:
                dead.add(ws)
        self.active -= dead

    def broadcast_from_thread(self, data: dict):
        """Thread-safe broadcast called from the agent's background thread."""
        if self._loop and not self._loop.is_closed():
            asyncio.run_coroutine_threadsafe(self.broadcast(data), self._loop)


manager = ConnectionManager()

# ── Agent lifecycle ───────────────────────────────────────────────────────────

agent_thread: threading.Thread | None = None


def state_callback(state: str, text: str = ""):
    manager.broadcast_from_thread({"state": state, "text": text})


def run_agent():
    agent = AIVoiceAgent(on_state_change=state_callback)
    try:
        agent.start_transcription()
    except KeyboardInterrupt:
        pass
    except Exception as e:
        manager.broadcast_from_thread({"state": "error", "text": str(e)})


@app.on_event("startup")
async def startup():
    global agent_thread
    manager.set_loop(asyncio.get_event_loop())
    agent_thread = threading.Thread(target=run_agent, daemon=True)
    agent_thread.start()


# ── Routes ────────────────────────────────────────────────────────────────────

frontend_path = Path(__file__).parent / "frontend"


@app.get("/")
async def serve_ui():
    index = frontend_path / "index.html"
    if index.exists():
        return FileResponse(index)
    return HTMLResponse("<h1>Frontend not found — place index.html in ./frontend/</h1>")


@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await manager.connect(ws)
    await ws.send_text(json.dumps({"state": "idle", "text": ""}))
    try:
        while True:
            await ws.receive_text()
    except (WebSocketDisconnect, Exception):
        manager.disconnect(ws)
