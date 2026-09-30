import numpy as np
import pytest
from engine.dsp_gate import SoftKneeDSPGate


def test_gate_initial_state():
    gate = SoftKneeDSPGate(sample_rate=16000, ducking_floor_db=-24.0)
    assert gate.get_current_gain() == pytest.approx(10 ** (-24.0 / 20.0), rel=1e-3)
    assert gate.get_gate_status() in ("DUCKED", "MUTED")


def test_gate_attack_smoothness_and_timing():
    sr = 16000
    attack_ms = 20.0
    gate = SoftKneeDSPGate(sample_rate=sr, attack_ms=attack_ms, ducking_floor_db=-24.0)
    
    # Trigger gate open
    gate.set_target_gain(1.0)
    
    # Process 20ms of silence (320 samples)
    num_samples = int(sr * (attack_ms / 1000.0))
    chunk = np.ones(num_samples, dtype=np.float32)
    processed = gate.process(chunk)
    
    # Current gain should have risen substantially toward 1.0
    final_gain = gate.get_current_gain()
    assert final_gain > 0.60
    assert final_gain <= 1.0
    assert gate.get_gate_status() == "OPEN"
    
    # Verify no sample-to-sample jump exceeds 0.01 (click-free test)
    gains = processed / (chunk + 1e-12)
    deltas = np.abs(np.diff(gains))
    assert np.max(deltas) < 0.01


def test_gate_hold_and_release():
    sr = 16000
    hold_ms = 150.0
    release_ms = 250.0
    gate = SoftKneeDSPGate(sample_rate=sr, attack_ms=20.0, release_ms=release_ms, hold_ms=hold_ms)
    
    # Open gate fully
    gate.set_target_gain(1.0)
    gate.process(np.ones(int(sr * 0.1), dtype=np.float32)) # 100ms
    assert gate.get_current_gain() > 0.98
    
    # Close gate (duck)
    gate.set_target_gain(0.0)
    
    # Process within hold window (e.g. 100ms < 150ms hold)
    hold_chunk = np.ones(int(sr * 0.1), dtype=np.float32)
    gate.process(hold_chunk)
    
    # Gain should still remain high because of hold time!
    assert gate.get_current_gain() > 0.95
    
    # Now process 300ms more (exceeding hold and progressing through release)
    release_chunk = np.ones(int(sr * 0.3), dtype=np.float32)
    gate.process(release_chunk)
    
    # Gain should have dropped towards ducking floor
    assert gate.get_current_gain() < 0.20


def test_gate_full_mute():
    gate = SoftKneeDSPGate(sample_rate=16000, ducking_floor_db=-100.0)
    gate.set_target_gain(0.0)
    # Drain any hold/release
    gate.process(np.ones(16000, dtype=np.float32))
    assert gate.get_gate_status() == "MUTED"
    assert gate.get_current_gain() == pytest.approx(0.0, abs=1e-4)
