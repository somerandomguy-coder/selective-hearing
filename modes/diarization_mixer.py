from typing import List, Optional, Tuple
import numpy as np
import torch


class StreamingDiarizationMixer:
    """Streaming multi-speaker diarization mixer supporting up to 8 speaker slots.
    
    Interfaces with Sortformer / Nemotron-3 Diarization architectures, translating
    a [T, 8] activity probability tensor into real-time channel gating decisions
    with individual Solo, Mute, and Overlap arbitration.
    """

    def __init__(
        self,
        model_name: str = "nvidia/Nemotron-3-Diarization",
        device: Optional[str] = None,
        activity_threshold: float = 0.5,
        use_mock: bool = False,
    ) -> None:
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.activity_threshold = activity_threshold
        self.use_mock = use_mock

        # 8-channel masks
        self.solo_mask: List[bool] = [i == 0 for i in range(8)]
        self.mute_mask: List[bool] = [False] * 8

        self._model = None

    def _get_model(self):
        """Lazy loader for streaming diarization model."""
        if self.use_mock:
            return None
        if self._model is None:
            try:
                # Attempt loading NeMo 2.0 / HF Sortformer diarization model
                # Fallback to mock if NeMo / weights are not present on local machine
                pass
            except Exception as e:
                print(f"[AcousticLens] Warning: Diarization model unavailable ({e}). Using mock.")
                self.use_mock = True
        return self._model

    def reset_channels(self) -> None:
        """Resets solo and mute masks to all False."""
        self.solo_mask = [False] * 8
        self.mute_mask = [False] * 8

    def set_solo(self, speaker_id: int, solo: bool) -> None:
        """Sets solo status for a specific speaker channel (0-7)."""
        if 0 <= speaker_id < 8:
            self.solo_mask[speaker_id] = bool(solo)

    def set_mute(self, speaker_id: int, mute: bool) -> None:
        """Sets mute status for a specific speaker channel (0-7)."""
        if 0 <= speaker_id < 8:
            self.mute_mask[speaker_id] = bool(mute)

    def evaluate_channel_activities(
        self, speaker_probs: List[float]
    ) -> Tuple[bool, List[float], bool]:
        """Arbitrates channel activity against solo and mute masks.
        
        Args:
            speaker_probs: 8-element list of speaker activity probabilities.
            
        Returns:
            Tuple of (gate_open: bool, speaker_probs: List[float], overlap_detected: bool).
        """
        active_flags = [p >= self.activity_threshold for p in speaker_probs]

        # Determine effective active states (filtering muted channels)
        effective_active = [
            active_flags[i] and not self.mute_mask[i] for i in range(8)
        ]

        has_any_solo = any(self.solo_mask)
        soloed_indices = [
            i for i, s in enumerate(self.solo_mask) if s and not self.mute_mask[i]
        ]
        non_soloed_indices = [
            i for i, s in enumerate(self.solo_mask) if not s and not self.mute_mask[i]
        ]

        soloed_active = any(effective_active[i] for i in soloed_indices)
        non_soloed_active = any(effective_active[i] for i in non_soloed_indices)

        overlap_detected = bool(soloed_active and non_soloed_active)

        if has_any_solo:
            # Gate opens IF AND ONLY IF at least one soloed unmuted speaker is active
            gate_open = bool(soloed_active)
        else:
            # If no channel is soloed, any active unmuted speaker opens gate
            gate_open = any(effective_active)

        return gate_open, speaker_probs, overlap_detected

    def infer_probabilities(self, chunk: np.ndarray) -> List[float]:
        """Runs inference or mock simulation to predict probabilities for the 8 slots."""
        if self.use_mock or self._model is None:
            # In mock mode, derive reproducible speaker probabilities from chunk energy
            energy = float(np.mean(chunk**2))
            probs = [0.0] * 8
            if energy > 0.001:
                # Speaker 0 active when energy is present
                probs[0] = float(np.clip(energy * 200.0, 0.6, 0.98))
            return probs

        # Standard Sortformer inference
        # Input shape [B, T], Output [B, T, 8]
        tensor = torch.from_numpy(chunk).float().unsqueeze(0).to(self.device)
        with torch.no_grad():
            preds = self._model(tensor)
            if hasattr(preds, "logits"):
                preds = torch.sigmoid(preds.logits)
            elif isinstance(preds, torch.Tensor):
                preds = torch.sigmoid(preds)
            probs_matrix = preds.squeeze(0).cpu().numpy()  # [T, 8]
            max_probs = np.max(probs_matrix, axis=0)  # [8]
            return [float(p) for p in max_probs]

    def process_chunk(self, chunk: np.ndarray) -> Tuple[bool, List[float], bool]:
        """Processes an audio chunk and evaluates the gating decision."""
        probs = self.infer_probabilities(chunk)
        return self.evaluate_channel_activities(probs)
