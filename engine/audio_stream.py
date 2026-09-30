import threading
import time
from typing import Callable, List, Optional, Tuple
import numpy as np

try:
    import sounddevice as sd
except Exception:
    sd = None


class AudioStreamManager:
    """Manages low-latency audio I/O streaming using sounddevice or synthetic mock."""

    def __init__(
        self,
        process_callback: Callable[[np.ndarray], np.ndarray],
        sample_rate: int = 16000,
        block_size: int = 1280,
        input_device: Optional[int] = None,
        output_device: Optional[int] = None,
        use_mock: bool = False,
    ) -> None:
        self.process_callback = process_callback
        self.sample_rate = sample_rate
        self.block_size = block_size
        self.input_device = input_device
        self.output_device = output_device
        self.use_mock = use_mock or (sd is None)

        self._running = False
        self._stream = None
        self._mock_thread: Optional[threading.Thread] = None

    @staticmethod
    def list_devices() -> Tuple[List[dict], List[dict]]:
        """Queries available audio input and output devices."""
        input_devs = []
        output_devs = []
        if sd is None:
            return (
                [{"id": 0, "name": "Mock Virtual Microphone", "channels": 1}],
                [{"id": 0, "name": "Mock Virtual Speaker", "channels": 1}],
            )

        try:
            devs = sd.query_devices()
            for idx, dev in enumerate(devs):
                if dev.get("max_input_channels", 0) > 0:
                    input_devs.append({"id": idx, "name": dev["name"], "channels": dev["max_input_channels"]})
                if dev.get("max_output_channels", 0) > 0:
                    output_devs.append({"id": idx, "name": dev["name"], "channels": dev["max_output_channels"]})
        except Exception as e:
            print(f"[AcousticLens] Warning querying audio devices: {e}")
            input_devs = [{"id": 0, "name": "System Default Mic", "channels": 1}]
            output_devs = [{"id": 0, "name": "System Default Output", "channels": 1}]

        return input_devs, output_devs

    def _audio_callback(self, indata, outdata, frames, time_info, status):
        """High-priority sounddevice duplex stream callback."""
        if status:
            pass  # Buffer underrun or overflow occurred
        try:
            in_mono = indata[:, 0]
            out_mono = self.process_callback(in_mono)
            outdata[:, 0] = out_mono
        except Exception:
            outdata.fill(0.0)

    def _mock_worker(self):
        """Simulates real-time audio block streaming for headless/testing environments."""
        dt = self.block_size / self.sample_rate
        t_phase = 0.0
        while self._running:
            t0 = time.perf_counter()
            # Generate speech-like modulated synthetic audio block
            t = np.linspace(t_phase, t_phase + dt, self.block_size, endpoint=False)
            t_phase += dt
            # 220Hz harmonic with random amplitude bursts
            burst = 0.6 if (int(t_phase * 2) % 2 == 0) else 0.02
            synthetic = (burst * np.sin(2 * np.pi * 220 * t) + 0.01 * np.random.randn(self.block_size)).astype(
                np.float32
            )
            self.process_callback(synthetic)
            elapsed = time.perf_counter() - t0
            sleep_time = dt - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

    def start(self) -> None:
        """Starts the audio stream."""
        if self._running:
            return
        self._running = True

        if self.use_mock or sd is None:
            self._mock_thread = threading.Thread(target=self._mock_worker, daemon=True)
            self._mock_thread.start()
            return

        try:
            self._stream = sd.Stream(
                samplerate=self.sample_rate,
                blocksize=self.block_size,
                device=(self.input_device, self.output_device),
                channels=1,
                dtype="float32",
                callback=self._audio_callback,
            )
            self._stream.start()
        except Exception as e:
            print(f"[AcousticLens] Failed to open physical audio stream: {e}. Falling back to mock audio.")
            self.use_mock = True
            self._mock_thread = threading.Thread(target=self._mock_worker, daemon=True)
            self._mock_thread.start()

    def stop(self) -> None:
        """Stops the audio stream."""
        if not self._running:
            return
        self._running = False

        if self._stream is not None:
            try:
                self._stream.stop()
                self._stream.close()
            except Exception:
                pass
            self._stream = None

        if self._mock_thread is not None:
            self._mock_thread.join(timeout=1.0)
            self._mock_thread = None

    def is_running(self) -> bool:
        """Returns True if the stream is currently active."""
        return self._running
