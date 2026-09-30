from typing import Dict, List, Optional
import numpy as np
from modes.enrolled_gate import EnrolledSpeakerGate


class AutoSoloMatcher:
    """Hybrid Auto-Solo matcher.
    
    Observes speech activity across the 8 diarization slots and compares isolated
    utterances of each active speaker against the host's enrolled reference voice.
    Automatically locks onto and solos the matching speaker slot.
    """

    def __init__(
        self,
        enrolled_gate: EnrolledSpeakerGate,
        min_samples: int = 8000,
        correlation_threshold: float = 0.70,
    ) -> None:
        self.enrolled_gate = enrolled_gate
        self.min_samples = min_samples
        self.correlation_threshold = correlation_threshold

        self.locked_speaker_id: Optional[int] = None
        self._accumulators: Dict[int, List[np.ndarray]] = {i: [] for i in range(8)}
        self._sample_counts: Dict[int, int] = {i: 0 for i in range(8)}

    def is_locked(self) -> bool:
        """Returns True if a diarization channel has been locked onto the enrolled voice."""
        return self.locked_speaker_id is not None

    def get_locked_speaker_id(self) -> Optional[int]:
        """Returns the locked speaker ID (0-7) or None if still searching."""
        return self.locked_speaker_id

    def reset(self) -> None:
        """Resets the auto-solo lock and clears all accumulated voice buffers."""
        self.locked_speaker_id = None
        for i in range(8):
            self._accumulators[i].clear()
            self._sample_counts[i] = 0

    def update_observation(
        self, audio_chunk: np.ndarray, speaker_probs: List[float]
    ) -> Optional[int]:
        """Processes an audio chunk and diarization probabilities to detect speaker identity.
        
        Args:
            audio_chunk: 1D numpy array of 16kHz audio samples.
            speaker_probs: 8-element list of speaker activity probabilities.
            
        Returns:
            The locked speaker ID (0-7) if locked, or None.
        """
        if self.locked_speaker_id is not None:
            return self.locked_speaker_id

        if self.enrolled_gate.reference_embedding is None:
            return None

        # Find the dominant speaker in this frame
        probs = np.array(speaker_probs, dtype=np.float32)
        top_idx = int(np.argmax(probs))
        top_val = float(probs[top_idx])

        # Find second highest probability to verify isolation
        other_probs = [probs[i] for i in range(8) if i != top_idx]
        other_max = float(max(other_probs)) if other_probs else 0.0

        # Isolated speech condition: strong activity for top speaker and low for others
        if top_val >= 0.65 and other_max < 0.30:
            self._accumulators[top_idx].append(audio_chunk)
            self._sample_counts[top_idx] += len(audio_chunk)

            if self._sample_counts[top_idx] >= self.min_samples:
                # Concatenate accumulated audio and compare against enrolled profile
                combined = np.concatenate(self._accumulators[top_idx])
                sim = self.enrolled_gate.compute_similarity(combined)

                if sim >= self.correlation_threshold:
                    self.locked_speaker_id = top_idx
                    return self.locked_speaker_id
                else:
                    # Clear accumulator to avoid infinite buffer growth
                    self._accumulators[top_idx].clear()
                    self._sample_counts[top_idx] = 0

        return None
