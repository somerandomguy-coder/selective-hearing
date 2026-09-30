import numpy as np
import pytest
from engine.audio_buffer import AudioBufferManager


def test_buffer_initialization():
    buf = AudioBufferManager(sample_rate=16000, capacity_s=2.0, lookahead_ms=250.0)
    assert buf.sample_rate == 16000
    assert buf.lookahead_samples == 4000
    assert buf.capacity_samples == 32000
    assert buf.get_current_rms_db() == -100.0


def test_buffer_delay_playback():
    sr = 16000
    lookahead_ms = 100.0  # 1600 samples
    chunk_size = 800  # 50ms
    buf = AudioBufferManager(sample_rate=sr, capacity_s=2.0, lookahead_ms=lookahead_ms)

    # Push 1st chunk of ones
    chunk1 = np.ones(chunk_size, dtype=np.float32)
    buf.push_input(chunk1)
    
    # Pop 1st chunk: should be silence because delay is 1600 samples (> 800 samples)
    out1 = buf.pop_delayed(chunk_size)
    assert np.allclose(out1, 0.0)

    # Push 2nd chunk of ones
    chunk2 = np.ones(chunk_size, dtype=np.float32)
    buf.push_input(chunk2)
    
    # Pop 2nd chunk: still silence (total pushed 1600, delay 1600)
    out2 = buf.pop_delayed(chunk_size)
    assert np.allclose(out2, 0.0)

    # Push 3rd chunk with recognizable ramp [1, 2, ..., 800]
    chunk3 = np.arange(1, chunk_size + 1, dtype=np.float32)
    buf.push_input(chunk3)

    # Pop 3rd chunk: should now start producing chunk1 (ones)
    out3 = buf.pop_delayed(chunk_size)
    assert np.allclose(out3, 1.0)


def test_context_window_and_wraparound():
    sr = 16000
    # Small capacity to force quick wraparound: 0.5s = 8000 samples
    buf = AudioBufferManager(sample_rate=sr, capacity_s=0.5, lookahead_ms=100.0)

    # Feed sequential samples to track exact ordering across wraparound
    total_samples = 12000  # 1.5x capacity
    seq = np.arange(total_samples, dtype=np.float32)
    
    chunk_len = 1000
    for i in range(0, total_samples, chunk_len):
        buf.push_input(seq[i : i + chunk_len])

    # Get context window of 0.2s = 3200 samples
    ctx = buf.get_context_window(window_duration_s=0.2)
    assert len(ctx) == 3200
    # Expected: the most recent 3200 samples of seq
    expected = seq[-3200:]
    assert np.allclose(ctx, expected)


def test_rms_db_calculation():
    buf = AudioBufferManager(sample_rate=16000, capacity_s=1.0, lookahead_ms=100.0)
    # Silence
    buf.push_input(np.zeros(1600, dtype=np.float32))
    assert buf.get_current_rms_db() <= -90.0

    # Sine wave with amplitude 0.5 (RMS = 0.5 / sqrt(2) ~ 0.35355, dBFS ~ -9.03 dB)
    t = np.linspace(0, 0.1, 1600, endpoint=False)
    sine = (0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    buf.push_input(sine)
    rms_db = buf.get_current_rms_db()
    assert rms_db == pytest.approx(-9.03, abs=0.5)
