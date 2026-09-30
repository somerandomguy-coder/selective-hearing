import os
import numpy as np
import pytest
from modes.enrolled_gate import EnrolledSpeakerGate


def test_mock_enrollment_and_similarity():
    gate = EnrolledSpeakerGate(use_mock=True)
    
    # 5 seconds calibration sample of test audio
    audio = np.random.randn(16000 * 5).astype(np.float32)
    ref_emb = gate.enroll_from_audio(audio)
    
    assert ref_emb is not None
    # Verify L2 normalization
    norm = np.linalg.norm(ref_emb)
    assert norm == pytest.approx(1.0, abs=1e-4)

    # Identical audio in mock mode should yield high similarity
    sim = gate.compute_similarity(audio[:16000])
    assert sim >= 0.85

    # Evaluation with default threshold 0.72
    gate_open, score = gate.evaluate_gate(audio[:16000], threshold=0.72)
    assert gate_open is True
    assert score == sim


def test_gate_threshold_logic():
    gate = EnrolledSpeakerGate(use_mock=True)
    ref_vector = np.zeros(192, dtype=np.float32)
    ref_vector[0] = 1.0
    gate.reference_embedding = ref_vector

    # Chunk with orthogonal vector (dot product = 0.0)
    ortho_vector = np.zeros(192, dtype=np.float32)
    ortho_vector[1] = 1.0
    
    # Custom similarity computation directly from embedding
    sim = gate.cosine_similarity(ref_vector, ortho_vector)
    assert sim == pytest.approx(0.0, abs=1e-4)

    # Below threshold: gate closes
    gate_open = sim >= 0.72
    assert gate_open is False


def test_profile_persistence(tmp_path):
    gate = EnrolledSpeakerGate(use_mock=True)
    audio = np.random.randn(16000).astype(np.float32)
    emb = gate.enroll_from_audio(audio)

    save_path = str(tmp_path / "test_profile.npy")
    gate.save_profile(save_path)
    assert os.path.exists(save_path)

    # Load into fresh instance
    new_gate = EnrolledSpeakerGate(use_mock=True)
    assert new_gate.reference_embedding is None
    loaded = new_gate.load_profile(save_path)
    assert loaded is True
    assert new_gate.reference_embedding is not None
    assert np.allclose(new_gate.reference_embedding, emb)
