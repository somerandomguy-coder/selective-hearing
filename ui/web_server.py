import asyncio
import os
from typing import Optional
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from engine.orchestrator import AcousticLensEngine
from engine.state import AcousticState, AudioMode


class ModeRequest(BaseModel):
    mode: str


class SoloRequest(BaseModel):
    speaker_id: int
    solo: bool


class MuteRequest(BaseModel):
    speaker_id: int
    mute: bool


class ThresholdRequest(BaseModel):
    threshold: float


class EnrollRequest(BaseModel):
    duration_s: float = 5.0


def create_web_app(engine: AcousticLensEngine, state: AcousticState) -> FastAPI:
    """Creates and configures the FastAPI web dashboard application."""
    app = FastAPI(title="AcousticLens Control Dashboard")

    static_dir = os.path.join(os.path.dirname(__file__), "web_static")
    if os.path.exists(static_dir):
        app.mount("/static", StaticFiles(directory=static_dir), name="static")

    @app.get("/")
    async def index():
        index_file = os.path.join(static_dir, "index.html")
        if os.path.exists(index_file):
            return FileResponse(index_file)
        return {"app": "AcousticLens", "status": "running"}

    @app.get("/api/status")
    async def get_status():
        return state.to_dict()

    @app.post("/api/mode")
    async def set_mode(req: ModeRequest):
        try:
            mode_enum = AudioMode(req.mode)
            engine.switch_mode(mode_enum)
            return {"success": True, "mode": mode_enum.value}
        except ValueError:
            return {"success": False, "error": f"Invalid mode: {req.mode}"}

    @app.post("/api/solo")
    async def set_solo(req: SoloRequest):
        state.set_solo(req.speaker_id, req.solo)
        return {"success": True, "speaker_id": req.speaker_id, "solo": req.solo}

    @app.post("/api/mute")
    async def set_mute(req: MuteRequest):
        state.set_mute(req.speaker_id, req.mute)
        return {"success": True, "speaker_id": req.speaker_id, "mute": req.mute}

    @app.post("/api/threshold")
    async def set_threshold(req: ThresholdRequest):
        state.update_mode1(
            similarity=state.similarity_score, threshold=req.threshold
        )
        return {"success": True, "threshold": req.threshold}

    @app.post("/api/enroll")
    async def trigger_enroll(req: EnrollRequest):
        engine.trigger_enrollment(duration_s=req.duration_s)
        return {"success": True, "duration_s": req.duration_s}

    @app.websocket("/ws/telemetry")
    async def websocket_telemetry(websocket: WebSocket):
        await websocket.accept()
        try:
            # Send immediate initial state
            await websocket.send_json(state.to_dict())

            while True:
                await asyncio.sleep(0.04)  # 25 Hz telemetry rate
                payload = state.to_dict()
                await websocket.send_json(payload)
        except WebSocketDisconnect:
            pass
        except Exception:
            pass

    return app
