import pytest
from engine.state import AcousticState, AudioMode, SpeakerChannelState


def test_initial_state():
    state = AcousticState()
    assert state.active_mode == AudioMode.ENROLLED_GATE
    assert state.rms_level_db == -100.0
    assert state.smoothed_gain == 0.0
    assert state.gate_status == "MUTED"
    assert len(state.speaker_channels) == 8
    assert state.similarity_threshold == 0.72


def test_mode_and_telemetry_update():
    state = AcousticState()
    notifications = []
    state.register_listener(lambda s: notifications.append(s.to_dict()))

    state.set_mode(AudioMode.DIARIZATION_MIXER)
    assert state.active_mode == AudioMode.DIARIZATION_MIXER
    assert len(notifications) == 1

    state.update_telemetry(rms_db=-18.5, gain=0.98, gate_status="OPEN")
    assert state.rms_level_db == -18.5
    assert state.smoothed_gain == 0.98
    assert state.gate_status == "OPEN"
    assert len(notifications) == 2


def test_mode1_state_updates():
    state = AcousticState()
    state.update_mode1(similarity=0.85, threshold=0.75, is_enrolling=True, enroll_progress=0.5)
    assert state.similarity_score == 0.85
    assert state.similarity_threshold == 0.75
    assert state.is_enrolling is True
    assert state.enroll_progress == 0.5


def test_speaker_channels_solo_and_mute():
    state = AcousticState()
    # Speaker 0 defaults to solo=True
    assert state.speaker_channels[0].solo is True
    assert state.speaker_channels[1].solo is False

    state.set_solo(speaker_id=2, solo=True)
    assert state.speaker_channels[2].solo is True

    state.set_mute(speaker_id=2, mute=True)
    assert state.speaker_channels[2].mute is True

    # Test updating probabilities and activities
    channels = [
        SpeakerChannelState(id=i, active=(i == 2), prob=(0.92 if i == 2 else 0.05), solo=(i == 2), mute=False)
        for i in range(8)
    ]
    state.update_mode2_speakers(channels, overlap_detected=False)
    assert state.speaker_channels[2].prob == 0.92
    assert state.speaker_channels[2].active is True
    assert state.overlap_detected is False


def test_json_serialization():
    state = AcousticState()
    d = state.to_dict()
    assert d["active_mode"] == "enrolled_gate"
    assert "rms_level_db" in d
    assert "speaker_channels" in d
    assert len(d["speaker_channels"]) == 8
