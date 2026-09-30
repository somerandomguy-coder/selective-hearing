import os
from typing import Optional, Tuple
import numpy as np
import torch


class EnrolledSpeakerGate:
    """Enrolled speaker verification gate using ECAPA-TDNN embeddings.
    
    Extracts speaker embeddings from a reference audio calibration sample
    and performs real-time cosine similarity verification on sliding audio windows.
    """

    def __init__(
        self,
        model_source: str = "speechbrain/spkrec-ecapa-voxceleb",
        device: Optional[str] = None,
        use_mock: bool = False,
    ) -> None:
        self.model_source = model_source
        self.use_mock = use_mock
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.reference_embedding: Optional[np.ndarray] = None
        self._model = None

    def _get_model(self):
        """Lazy loader for SpeechBrain ECAPA-TDNN model."""
        if self.use_mock:
            return None
        if self._model is None:
            try:
                from speechbrain.inference.speaker import SpeakerRecognition

                self._model = SpeakerRecognition.from_hparams(
                    source=self.model_source,
                    run_opts={"device": self.device},
                    savedir=os.path.join("data", "cache", "spkrec-ecapa-voxceleb"),
                )
            except Exception as e:
                # Fallback to mock if offline or weights unavailable
                print(f"[AcousticLens] Warning: Could not load ECAPA-TDNN ({e}). Using mock mode.")
                self.use_mock = True
        return self._model

    def _extract_embedding(self, audio: np.ndarray, sample_rate: int = 16000) -> np.ndarray:
        """Extracts and L2-normalizes a speaker embedding vector."""
        if self.use_mock:
            # Deterministic projection for mock / test mode
            # Hash chunk energy and spectral characteristics into 192-dim vector
            if len(audio) < 160:
                audio = np.pad(audio, (0, 160 - len(audio)))
            fft = np.abs(np.fft.rfft(audio[:1024]))
            vec = np.zeros(192, dtype=np.float32)
            n_bins = min(len(fft), 192)
            vec[:n_bins] = fft[:n_bins]
            norm = np.linalg.norm(vec) + 1e-12
            return (vec / norm).astype(np.float32)

        model = self._get_model()
        tensor = torch.from_numpy(audio).float().unsqueeze(0).to(self.device)
        with torch.no_grad():
            emb = model.encode_batch(tensor)
            emb = emb.squeeze().cpu().numpy()
            norm = np.linalg.norm(emb) + 1e-12
            return (emb / norm).astype(np.float32)

    def cosine_similarity(self, v1: np.ndarray, v2: np.ndarray) -> float:
        """Computes cosine similarity between two vectors."""
        norm1 = np.linalg.norm(v1) + 1e-12
        norm2 = np.linalg.norm(v2) + 1e-12
        dot = float(np.dot(v1, v2))
        sim = dot / (norm1 * norm2)
        return float(np.clip(sim, -1.0, 1.0))

    def enroll_from_audio(self, audio: np.ndarray, sample_rate: int = 16000) -> np.ndarray:
        """Enrolls a speaker from a calibration audio recording (e.g. 5 seconds)."""
        emb = self._extract_embedding(audio, sample_rate=sample_rate)
        self.reference_embedding = emb
        return emb

    def compute_similarity(self, chunk: np.ndarray) -> float:
        """Computes cosine similarity of chunk against reference embedding."""
        if self.reference_embedding is None:
            return 0.0
        chunk_emb = self._extract_embedding(chunk)
        return self.cosine_similarity(chunk_emb, self.reference_embedding)

    def evaluate_gate(self, chunk: np.ndarray, threshold: float = 0.72) -> Tuple[bool, float]:
        """Evaluates whether gate should be open based on cosine similarity."""
        sim = self.compute_similarity(chunk)
        gate_open = sim >= threshold
        return gate_open, sim

    def save_profile(self, filepath: str) -> None:
        """Persists the reference speaker embedding to disk."""
        if self.reference_embedding is None:
            raise ValueError("No reference embedding to save.")
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        np.save(filepath, self.reference_embedding)

    def load_profile(self, filepath: str) -> bool:
        """Loads a reference speaker embedding from disk."""
        if not os.path.exists(filepath):
            return False
        try:
            self.reference_embedding = np.load(filepath)
            return True
        except Exception:
            return False
