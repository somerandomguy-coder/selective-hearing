import numpy as np
import pytest
from modes.diarization_mixer import StreamingDiarizationMixer


def test_channel_solo_and_mute_control():
    mixer = StreamingDiarizationMixer(use_mock=True)
    assert mixer.solo_mask[0] is True
    assert mixer.solo_mask[1] is False
    assert mixer.mute_mask[0] is False

    mixer.set_solo(1, True)
    assert mixer.solo_mask[1] is True

    mixer.set_mute(1, True)
    assert mixer.mute_mask[1] is True


def test_diarization_arbitration_solo_active():
    mixer = StreamingDiarizationMixer(use_mock=True)
    # Solo speaker 0 only
    mixer.reset_channels()
    mixer.set_solo(0, True)

    # Simulated probs: speaker 0 active (0.9), others inactive
    fake_probs = [0.9, 0.05, 0.02, 0.01, 0.0, 0.0, 0.0, 0.0]
    gate_open, probs, overlap = mixer.evaluate_channel_activities(fake_probs)
    
    assert gate_open is True
    assert overlap is False
    assert probs == fake_probs


def test_diarization_arbitration_non_solo_active():
    mixer = StreamingDiarizationMixer(use_mock=True)
    # Solo speaker 0 only
    mixer.reset_channels()
    mixer.set_solo(0, True)

    # Simulated probs: speaker 1 active (0.85), speaker 0 inactive (0.1)
    fake_probs = [0.1, 0.85, 0.02, 0.01, 0.0, 0.0, 0.0, 0.0]
    gate_open, probs, overlap = mixer.evaluate_channel_activities(fake_probs)

    # Gate should close / duck because only non-soloed speaker is speaking!
    assert gate_open is False
    assert overlap is False


def test_diarization_arbitration_overlap():
    mixer = StreamingDiarizationMixer(use_mock=True)
    mixer.reset_channels()
    mixer.set_solo(0, True)

    # Both speaker 0 (soloed) and speaker 1 (non-soloed) speaking simultaneously
    fake_probs = [0.88, 0.82, 0.02, 0.01, 0.0, 0.0, 0.0, 0.0]
    gate_open, probs, overlap = mixer.evaluate_channel_activities(fake_probs)

    # Gate must stay open for soloed speaker, but overlap flag must be set!
    assert gate_open is True
    assert overlap is True


def test_diarization_arbitration_mute_soloed_speaker():
    mixer = StreamingDiarizationMixer(use_mock=True)
    mixer.reset_channels()
    mixer.set_solo(0, True)
    mixer.set_mute(0, True)  # Mute speaker 0

    fake_probs = [0.95, 0.05, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    gate_open, probs, overlap = mixer.evaluate_channel_activities(fake_probs)

    # Muted speaker 0 must not open gate
    assert gate_open is False


def test_diarization_process_chunk():
    mixer = StreamingDiarizationMixer(use_mock=True)
    chunk = np.random.randn(1280).astype(np.float32)
    gate_open, probs, overlap = mixer.process_chunk(chunk)
    assert len(probs) == 8
    assert isinstance(gate_open, bool)
    assert isinstance(overlap, bool)
