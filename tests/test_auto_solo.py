import numpy as np
import pytest
from modes.auto_solo import AutoSoloMatcher
from modes.enrolled_gate import EnrolledSpeakerGate


def test_auto_solo_matcher_initialization():
    enrolled_gate = EnrolledSpeakerGate(use_mock=True)
    matcher = AutoSoloMatcher(enrolled_gate=enrolled_gate, min_samples=3200)
    assert matcher.get_locked_speaker_id() is None
    assert matcher.is_locked() is False


def test_auto_solo_successful_lock():
    enrolled_gate = EnrolledSpeakerGate(use_mock=True)
    # Generate reference audio and enroll
    audio = np.random.randn(16000 * 2).astype(np.float32)
    enrolled_gate.enroll_from_audio(audio)

    # Use smaller min_samples for fast testing
    matcher = AutoSoloMatcher(enrolled_gate=enrolled_gate, min_samples=1600, correlation_threshold=0.70)

    # Feed chunks where Speaker 2 is exclusively active with the enrolled audio!
    chunk = audio[:1600]
    probs = [0.05, 0.02, 0.95, 0.01, 0.0, 0.0, 0.0, 0.0]

    locked_id = matcher.update_observation(chunk, probs)
    assert locked_id == 2
    assert matcher.get_locked_speaker_id() == 2
    assert matcher.is_locked() is True

    # Calling reset unlocks
    matcher.reset()
    assert matcher.is_locked() is False
    assert matcher.get_locked_speaker_id() is None


def test_auto_solo_does_not_lock_on_dissimilar_voice():
    enrolled_gate = EnrolledSpeakerGate(use_mock=True)
    ref_vector = np.zeros(192, dtype=np.float32)
    ref_vector[0] = 1.0
    enrolled_gate.reference_embedding = ref_vector

    matcher = AutoSoloMatcher(enrolled_gate=enrolled_gate, min_samples=1600, correlation_threshold=0.70)

    # Audio chunk that has high energy at high frequencies (dissimilar in mock fft)
    t = np.linspace(0, 0.1, 1600, endpoint=False)
    high_freq_audio = np.sin(2 * np.pi * 7000 * t).astype(np.float32)

    probs = [0.0, 0.90, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    locked_id = matcher.update_observation(high_freq_audio, probs)

    # Should not lock onto speaker 1 because similarity is low
    assert locked_id is None
    assert matcher.is_locked() is False
