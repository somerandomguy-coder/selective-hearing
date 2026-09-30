from dataclasses import asdict, dataclass, field
from enum import Enum
import threading
from typing import Callable, List, Optional


class AudioMode(str, Enum):
    ENROLLED_GATE = "enrolled_gate"
    DIARIZATION_MIXER = "diarization_mixer"
    AUTO_SOLO = "auto_solo"
    PASSTHROUGH = "passthrough"


@dataclass
class SpeakerChannelState:
    id: int
    active: bool = False
    prob: float = 0.0
    solo: bool = False
    mute: bool = False


class AcousticState:
    """Centralized thread-safe observable state for AcousticLens."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._listeners: List[Callable[["AcousticState"], None]] = []

        self.active_mode: AudioMode = AudioMode.ENROLLED_GATE
        self.rms_level_db: float = -100.0
        self.smoothed_gain: float = 0.0
        self.gate_status: str = "MUTED"

        # Mode 1 State
        self.similarity_score: float = 0.0
        self.similarity_threshold: float = 0.72
        self.is_enrolling: bool = False
        self.enroll_progress: float = 0.0

        # Mode 2 & Hybrid State (8 slots)
        self.speaker_channels: List[SpeakerChannelState] = [
            SpeakerChannelState(id=i, solo=(i == 0)) for i in range(8)
        ]
        self.overlap_detected: bool = False

        # Hardware & Devices
        self.input_device_name: str = "Default Input"
        self.output_device_name: str = "Default Output"

    def register_listener(self, callback: Callable[["AcousticState"], None]) -> None:
        """Subscribes a listener callback to state updates."""
        with self._lock:
            self._listeners.append(callback)

    def _notify(self) -> None:
        """Notifies all registered listeners of state modification."""
        # Call outside lock to prevent deadlocks
        for listener in self._listeners:
            try:
                listener(self)
            except Exception:
                pass

    def set_mode(self, mode: AudioMode) -> None:
        """Switches the active operating mode."""
        with self._lock:
            if self.active_mode != mode:
                self.active_mode = mode
        self._notify()

    def update_telemetry(self, rms_db: float, gain: float, gate_status: str) -> None:
        """Updates fast audio telemetry."""
        with self._lock:
            self.rms_level_db = round(float(rms_db), 1)
            self.smoothed_gain = round(float(gain), 3)
            self.gate_status = gate_status
        self._notify()

    def update_mode1(
        self,
        similarity: float,
        threshold: Optional[float] = None,
        is_enrolling: Optional[bool] = None,
        enroll_progress: Optional[float] = None,
    ) -> None:
        """Updates Mode 1 speaker verification telemetry."""
        with self._lock:
            self.similarity_score = round(float(similarity), 3)
            if threshold is not None:
                self.similarity_threshold = round(float(threshold), 3)
            if is_enrolling is not None:
                self.is_enrolling = bool(is_enrolling)
            if enroll_progress is not None:
                self.enroll_progress = round(float(enroll_progress), 2)
        self._notify()

    def set_solo(self, speaker_id: int, solo: bool) -> None:
        """Toggles or sets the solo status of a specific speaker slot."""
        with self._lock:
            if 0 <= speaker_id < len(self.speaker_channels):
                self.speaker_channels[speaker_id].solo = solo
        self._notify()

    def set_mute(self, speaker_id: int, mute: bool) -> None:
        """Toggles or sets the mute status of a specific speaker slot."""
        with self._lock:
            if 0 <= speaker_id < len(self.speaker_channels):
                self.speaker_channels[speaker_id].mute = mute
        self._notify()

    def update_mode2_speakers(
        self,
        speakers: List[SpeakerChannelState],
        overlap_detected: bool,
    ) -> None:
        """Updates speaker activity probabilities and overlap detection."""
        with self._lock:
            # Retain solo/mute states while updating prob & active
            for s in speakers:
                if 0 <= s.id < len(self.speaker_channels):
                    curr = self.speaker_channels[s.id]
                    curr.prob = round(float(s.prob), 3)
                    curr.active = bool(s.active)
                    curr.solo = s.solo
                    curr.mute = s.mute
            self.overlap_detected = bool(overlap_detected)
        self._notify()

    def to_dict(self) -> dict:
        """Exports thread-safe snapshot dictionary for serialization."""
        with self._lock:
            return {
                "active_mode": self.active_mode.value,
                "rms_level_db": self.rms_level_db,
                "smoothed_gain": self.smoothed_gain,
                "gate_status": self.gate_status,
                "similarity_score": self.similarity_score,
                "similarity_threshold": self.similarity_threshold,
                "is_enrolling": self.is_enrolling,
                "enroll_progress": self.enroll_progress,
                "speaker_channels": [asdict(ch) for ch in self.speaker_channels],
                "overlap_detected": self.overlap_detected,
                "input_device_name": self.input_device_name,
                "output_device_name": self.output_device_name,
            }
