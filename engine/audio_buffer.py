import threading
import numpy as np


class AudioBufferManager:
    """Thread-safe circular ring buffer for real-time audio delay and neural context extraction.
    
    Maintains a circular buffer of 16kHz audio samples. Audio pushed from the
    hardware input stream can be simultaneously read with a configurable lookahead delay
    (to give neural models time to infer gating decisions) and sliced into recent
    sliding context windows (0.32s - 1.04s) for neural workers.
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        capacity_s: float = 3.0,
        lookahead_ms: float = 250.0,
    ) -> None:
        self.sample_rate = sample_rate
        self.capacity_samples = max(1600, int(sample_rate * capacity_s))
        self.lookahead_samples = int(sample_rate * (lookahead_ms / 1000.0))

        self.buffer = np.zeros(self.capacity_samples, dtype=np.float32)
        self.write_pos: int = 0  # Total cumulative samples written
        self.lock = threading.Lock()
        self.latest_rms_db: float = -100.0

    def push_input(self, chunk: np.ndarray) -> None:
        """Pushes a chunk of incoming mono audio into the circular buffer."""
        chunk = np.asarray(chunk, dtype=np.float32).ravel()
        n = len(chunk)
        if n == 0:
            return

        # Compute RMS and dBFS
        rms = float(np.sqrt(np.mean(chunk**2) + 1e-12))
        db = 20.0 * float(np.log10(rms))
        self.latest_rms_db = float(np.clip(db, -100.0, 0.0))

        with self.lock:
            start = self.write_pos % self.capacity_samples
            if start + n <= self.capacity_samples:
                self.buffer[start : start + n] = chunk
            else:
                part1 = self.capacity_samples - start
                self.buffer[start:] = chunk[:part1]
                part2 = n - part1
                # If chunk is larger than capacity, only write trailing portion
                if part2 > self.capacity_samples:
                    self.buffer[:] = chunk[-self.capacity_samples :]
                else:
                    self.buffer[:part2] = chunk[part1:]

            self.write_pos += n

    def pop_delayed(self, chunk_size: int) -> np.ndarray:
        """Pops a chunk of audio delayed by lookahead_samples relative to current write_pos.
        
        Args:
            chunk_size: Number of samples to retrieve.
            
        Returns:
            Delayed 1D numpy array of float32 samples.
        """
        out = np.zeros(chunk_size, dtype=np.float32)
        with self.lock:
            read_start = self.write_pos - self.lookahead_samples - chunk_size
            read_end = read_start + chunk_size

            # If delayed read position is before the beginning of the stream, return zeros for that portion
            if read_end <= 0:
                return out

            fill_start_idx = 0
            if read_start < 0:
                fill_start_idx = -read_start
                read_start = 0

            samples_to_read = read_end - read_start
            if samples_to_read > self.capacity_samples:
                read_start = read_end - self.capacity_samples
                samples_to_read = self.capacity_samples

            buf_start = read_start % self.capacity_samples
            if buf_start + samples_to_read <= self.capacity_samples:
                out[fill_start_idx : fill_start_idx + samples_to_read] = self.buffer[
                    buf_start : buf_start + samples_to_read
                ]
            else:
                part1 = self.capacity_samples - buf_start
                out[fill_start_idx : fill_start_idx + part1] = self.buffer[buf_start:]
                part2 = samples_to_read - part1
                out[fill_start_idx + part1 : fill_start_idx + samples_to_read] = self.buffer[:part2]

        return out

    def get_context_window(self, window_duration_s: float) -> np.ndarray:
        """Retrieves the most recent window of audio up to the current write position.
        
        Args:
            window_duration_s: Duration in seconds (e.g. 0.32s, 0.8s, 1.04s).
            
        Returns:
            Contiguous 1D numpy array of float32 samples.
        """
        req_samples = int(self.sample_rate * window_duration_s)
        out = np.zeros(req_samples, dtype=np.float32)

        with self.lock:
            available = min(self.write_pos, self.capacity_samples, req_samples)
            if available <= 0:
                return out

            read_start = self.write_pos - available
            fill_start = req_samples - available

            buf_start = read_start % self.capacity_samples
            if buf_start + available <= self.capacity_samples:
                out[fill_start:] = self.buffer[buf_start : buf_start + available]
            else:
                part1 = self.capacity_samples - buf_start
                out[fill_start : fill_start + part1] = self.buffer[buf_start:]
                part2 = available - part1
                out[fill_start + part1 :] = self.buffer[:part2]

        return out

    def get_current_rms_db(self) -> float:
        """Returns the latest input signal RMS level in dBFS."""
        return self.latest_rms_db
