import pytest
from starlette.testclient import TestClient
from engine.orchestrator import AcousticLensEngine
from engine.state import AcousticState, AudioMode
from ui.web_server import create_web_app


def test_web_server_endpoints():
    state = AcousticState()
    engine = AcousticLensEngine(
        state=state,
        sample_rate=16000,
        block_size=1280,
        use_mock_audio=True,
        use_mock_models=True,
    )
    app = create_web_app(engine=engine, state=state)
    client = TestClient(app)

    # 1. GET /
    res = client.get("/")
    assert res.status_code == 200
    assert "AcousticLens" in res.text

    # 2. GET /api/status
    res = client.get("/api/status")
    assert res.status_code == 200
    data = res.json()
    assert data["active_mode"] == "enrolled_gate"

    # 3. POST /api/mode
    res = client.post("/api/mode", json={"mode": "diarization_mixer"})
    assert res.status_code == 200
    assert state.active_mode == AudioMode.DIARIZATION_MIXER

    # 4. POST /api/solo
    res = client.post("/api/solo", json={"speaker_id": 2, "solo": True})
    assert res.status_code == 200
    assert state.speaker_channels[2].solo is True

    # 5. POST /api/mute
    res = client.post("/api/mute", json={"speaker_id": 2, "mute": True})
    assert res.status_code == 200
    assert state.speaker_channels[2].mute is True

    # 6. POST /api/threshold
    res = client.post("/api/threshold", json={"threshold": 0.85})
    assert res.status_code == 200
    assert state.similarity_threshold == 0.85

    # 7. POST /api/enroll
    res = client.post("/api/enroll", json={"duration_s": 1.0})
    assert res.status_code == 200
    assert state.is_enrolling is True


def test_websocket_telemetry():
    state = AcousticState()
    engine = AcousticLensEngine(
        state=state,
        sample_rate=16000,
        block_size=1280,
        use_mock_audio=True,
        use_mock_models=True,
    )
    app = create_web_app(engine=engine, state=state)
    client = TestClient(app)

    with client.websocket_connect("/ws/telemetry") as ws:
        data = ws.receive_json()
        assert "active_mode" in data
        assert "rms_level_db" in data
        assert "speaker_channels" in data
