import numpy as np


class SoftKneeDSPGate:
    """Sample-accurate soft-knee DSP gate with exponential smoothing and hold time.
    
    Prevents clicks and pops by smoothing gain transitions using an Exponential
    Moving Average (EMA) envelope with asymmetric attack and release characteristics.
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        attack_ms: float = 20.0,
        release_ms: float = 250.0,
        hold_ms: float = 150.0,
        ducking_floor_db: float = -24.0,
    ) -> None:
        self.sample_rate = sample_rate
        self.attack_ms = max(1.0, attack_ms)
        self.release_ms = max(1.0, release_ms)
        self.hold_ms = max(0.0, hold_ms)
        self.ducking_floor_db = ducking_floor_db

        self.ducking_floor = (
            10.0 ** (ducking_floor_db / 20.0) if ducking_floor_db > -90.0 else 0.0
        )

        # Standard 3*tau factor: envelope reaches ~95% of target within specified ms
        decay_factor = 3.0
        self.alpha_attack = 1.0 - float(
            np.exp(-decay_factor / (sample_rate * (self.attack_ms / 1000.0)))
        )
        self.alpha_release = 1.0 - float(
            np.exp(-decay_factor / (sample_rate * (self.release_ms / 1000.0)))
        )

        self.hold_samples_total = int(sample_rate * (self.hold_ms / 1000.0))
        self.hold_counter = 0

        self.current_gain: float = self.ducking_floor
        self.target_gain: float = self.ducking_floor

    def set_target_gain(self, target: float) -> None:
        """Sets the desired gain target (clamped between ducking_floor and 1.0)."""
        clamped_target = float(np.clip(target, self.ducking_floor, 1.0))
        if clamped_target > self.current_gain or clamped_target >= 0.99:
            # Trigger attack and refresh hold timer
            self.hold_counter = self.hold_samples_total
        self.target_gain = clamped_target

    def process(self, chunk: np.ndarray) -> np.ndarray:
        """Processes an incoming audio chunk, applying the smoothed gain envelope.
        
        Args:
            chunk: 1D numpy array of float32 samples.
            
        Returns:
            Processed 1D numpy array of float32 samples.
        """
        if len(chunk) == 0:
            return chunk

        out = np.empty_like(chunk, dtype=np.float32)
        gain = self.current_gain
        target = self.target_gain
        alpha_att = self.alpha_attack
        alpha_rel = self.alpha_release
        floor = self.ducking_floor

        # Sample-accurate gain follower
        for i in range(len(chunk)):
            if target > gain:
                # Attack phase: rapidly approach higher target
                gain += alpha_att * (target - gain)
            else:
                # Release phase: respect hold counter before decaying
                if self.hold_counter > 0:
                    self.hold_counter -= 1
                else:
                    gain += alpha_rel * (target - gain)

            # Clamp within floor and 1.0
            if gain < floor:
                gain = floor
            elif gain > 1.0:
                gain = 1.0

            out[i] = chunk[i] * gain

        self.current_gain = gain
        return out

    def get_current_gain(self) -> float:
        """Returns the instantaneous gain (linear scalar)."""
        return self.current_gain

    def get_gate_status(self) -> str:
        """Returns gate status string: 'OPEN', 'DUCKED', or 'MUTED'."""
        if self.current_gain > 0.5:
            return "OPEN"
        elif self.current_gain > 0.001:
            return "DUCKED"
        else:
            return "MUTED"
