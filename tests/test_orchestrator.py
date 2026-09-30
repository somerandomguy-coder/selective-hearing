import time
import numpy as np
import pytest
from engine.orchestrator import AcousticLensEngine
from engine.state import AcousticState, AudioMode


def test_orchestrator_initialization():
    state = AcousticState()
    engine = AcousticLensEngine(
        state=state,
        sample_rate=16000,
        block_size=1280,
        use_mock_audio=True,
        use_mock_models=True,
    )
    assert not engine.is_running()
    inputs, outputs = engine.list_devices()
    assert isinstance(inputs, list)
    assert isinstance(outputs, list)


def test_orchestrator_mock_lifecycle_and_telemetry():
    state = AcousticState()
    engine = AcousticLensEngine(
        state=state,
        sample_rate=16000,
        block_size=1280,
        use_mock_audio=True,
        use_mock_models=True,
    )

    engine.start()
    assert engine.is_running()

    # Wait for a couple of audio/inference ticks (e.g. 200ms)
    time.sleep(0.3)

    # Telemetry should be updating
    assert state.rms_level_db > -100.0

    # Test dynamic mode switching without stopping the stream
    engine.switch_mode(AudioMode.DIARIZATION_MIXER)
    assert state.active_mode == AudioMode.DIARIZATION_MIXER
    time.sleep(0.1)
    assert engine.is_running()

    engine.switch_mode(AudioMode.PASSTHROUGH)
    assert state.active_mode == AudioMode.PASSTHROUGH
    time.sleep(0.1)

    engine.stop()
    assert not engine.is_running()


def test_orchestrator_enrollment_trigger():
    state = AcousticState()
    engine = AcousticLensEngine(
        state=state,
        sample_rate=16000,
        block_size=1280,
        use_mock_audio=True,
        use_mock_models=True,
    )
    engine.start()

    # Trigger quick 0.3s enrollment
    engine.trigger_enrollment(duration_s=0.3)
    assert state.is_enrolling is True

    # Wait for enrollment to complete
    time.sleep(0.5)
    assert state.is_enrolling is False
    assert engine.enrolled_gate.reference_embedding is not None

    engine.stop()
