import pytest
from engine.orchestrator import AcousticLensEngine
from engine.state import AcousticState, AudioMode
from ui.tui_app import AcousticLensTUI


@pytest.mark.asyncio
async def test_tui_app_lifecycle():
    state = AcousticState()
    engine = AcousticLensEngine(
        state=state,
        sample_rate=16000,
        block_size=1280,
        use_mock_audio=True,
        use_mock_models=True,
    )
    app = AcousticLensTUI(engine=engine, state=state)

    async with app.run_test() as pilot:
        # Check initial mode
        assert state.active_mode == AudioMode.ENROLLED_GATE

        # Press '2' to switch to Diarization Mixer
        await pilot.press("2")
        assert state.active_mode == AudioMode.DIARIZATION_MIXER

        # Press '3' to switch to Auto-Solo
        await pilot.press("3")
        assert state.active_mode == AudioMode.AUTO_SOLO

        # Press '4' to switch to Passthrough
        await pilot.press("4")
        assert state.active_mode == AudioMode.PASSTHROUGH

        # Press '1' to switch back to Enrolled Gate
        await pilot.press("1")
        assert state.active_mode == AudioMode.ENROLLED_GATE

        # Press 'e' to trigger enrollment
        await pilot.press("e")
        assert state.is_enrolling is True

        # Press 'q' to quit
        await pilot.press("q")
