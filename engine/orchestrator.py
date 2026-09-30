import os
import threading
import time
from typing import List, Optional, Tuple
import numpy as np

from engine.audio_buffer import AudioBufferManager
from engine.audio_stream import AudioStreamManager
from engine.dsp_gate import SoftKneeDSPGate
from engine.state import AcousticState, AudioMode, SpeakerChannelState
from modes.auto_solo import AutoSoloMatcher
from modes.diarization_mixer import StreamingDiarizationMixer
from modes.enrolled_gate import EnrolledSpeakerGate


class AcousticLensEngine:
    """Master orchestrator integrating audio streaming, DSP gating, neural workers, and state."""

    def __init__(
        self,
        state: AcousticState,
        sample_rate: int = 16000,
        block_size: int = 1280,
        lookahead_ms: float = 250.0,
        input_device: Optional[int] = None,
        output_device: Optional[int] = None,
        use_mock_audio: bool = False,
        use_mock_models: bool = False,
    ) -> None:
        self.state = state
        self.sample_rate = sample_rate
        self.block_size = block_size
        self.lookahead_ms = lookahead_ms

        self.buffer_manager = AudioBufferManager(
            sample_rate=sample_rate, capacity_s=3.0, lookahead_ms=lookahead_ms
        )
        self.dsp_gate = SoftKneeDSPGate(
            sample_rate=sample_rate,
            attack_ms=20.0,
            release_ms=250.0,
            hold_ms=150.0,
            ducking_floor_db=-24.0,
        )

        self.enrolled_gate = EnrolledSpeakerGate(use_mock=use_mock_models)
        # Attempt loading existing profile if available
        profile_path = os.path.join("data", "profiles", "default_profile.npy")
        if os.path.exists(profile_path):
            self.enrolled_gate.load_profile(profile_path)

        self.diarization_mixer = StreamingDiarizationMixer(use_mock=use_mock_models)
        self.auto_solo = AutoSoloMatcher(enrolled_gate=self.enrolled_gate)

        self.stream_manager = AudioStreamManager(
            process_callback=self._process_audio_block,
            sample_rate=sample_rate,
            block_size=block_size,
            input_device=input_device,
            output_device=output_device,
            use_mock=use_mock_audio,
        )

        # Inference loop control
        self._running = False
        self._inference_thread: Optional[threading.Thread] = None

        # Live enrollment state
        self._is_enrolling = False
        self._enroll_samples_target = 0
        self._enroll_buffer: List[np.ndarray] = []
        self._enroll_lock = threading.Lock()

    def list_devices(self) -> Tuple[List[dict], List[dict]]:
        """Returns lists of available input and output devices."""
        return self.stream_manager.list_devices()

    def _process_audio_block(self, in_chunk: np.ndarray) -> np.ndarray:
        """Audio I/O block callback: buffers, applies DSP gate, returns delayed output."""
        self.buffer_manager.push_input(in_chunk)

        with self._enroll_lock:
            if self._is_enrolling:
                self._enroll_buffer.append(in_chunk.copy())

        # Retrieve lookahead-delayed audio
        delayed_audio = self.buffer_manager.pop_delayed(len(in_chunk))

        # Gate delayed audio with current smooth envelope
        gated_audio = self.dsp_gate.process(delayed_audio)

        # Update real-time telemetry
        self.state.update_telemetry(
            rms_db=self.buffer_manager.get_current_rms_db(),
            gain=self.dsp_gate.get_current_gain(),
            gate_status=self.dsp_gate.get_gate_status(),
        )

        return gated_audio

    def trigger_enrollment(self, duration_s: float = 5.0) -> None:
        """Triggers a 5-second calibration utterance recording to enroll the host voice."""
        with self._enroll_lock:
            self._enroll_samples_target = int(self.sample_rate * max(0.2, duration_s))
            self._enroll_buffer.clear()
            self._is_enrolling = True
            self.state.update_mode1(
                similarity=0.0, is_enrolling=True, enroll_progress=0.0
            )

    def _inference_worker(self) -> None:
        """Background inference loop executing neural models on sliding context windows."""
        while self._running:
            time.sleep(0.04)  # ~25 Hz loop rate

            # 1. Handle Active Enrollment
            with self._enroll_lock:
                enrolling = self._is_enrolling
                if enrolling:
                    curr_samples = sum(len(c) for c in self._enroll_buffer)
                    progress = min(1.0, curr_samples / max(1, self._enroll_samples_target))
                    self.state.update_mode1(similarity=0.0, enroll_progress=progress)

                    if curr_samples >= self._enroll_samples_target:
                        combined = np.concatenate(self._enroll_buffer)
                        self.enrolled_gate.enroll_from_audio(combined)
                        os.makedirs(os.path.join("data", "profiles"), exist_ok=True)
                        profile_path = os.path.join("data", "profiles", "default_profile.npy")
                        self.enrolled_gate.save_profile(profile_path)

                        self._is_enrolling = False
                        self._enroll_buffer.clear()
                        self.state.update_mode1(
                            similarity=1.0, is_enrolling=False, enroll_progress=1.0
                        )
                    continue

            # 2. Extract recent sliding context window for neural evaluation
            ctx = self.buffer_manager.get_context_window(0.8)
            mode = self.state.active_mode

            if mode == AudioMode.ENROLLED_GATE:
                gate_open, sim = self.enrolled_gate.evaluate_gate(
                    ctx, threshold=self.state.similarity_threshold
                )
                self.state.update_mode1(similarity=sim)
                target_gain = 1.0 if gate_open else self.dsp_gate.ducking_floor
                self.dsp_gate.set_target_gain(target_gain)

            elif mode == AudioMode.DIARIZATION_MIXER:
                # Sync solo and mute masks from centralized state
                for i, ch in enumerate(self.state.speaker_channels):
                    self.diarization_mixer.solo_mask[i] = ch.solo
                    self.diarization_mixer.mute_mask[i] = ch.mute

                gate_open, probs, overlap = self.diarization_mixer.process_chunk(ctx)

                updated_channels = [
                    SpeakerChannelState(
                        id=i,
                        active=(probs[i] >= self.diarization_mixer.activity_threshold),
                        prob=probs[i],
                        solo=self.state.speaker_channels[i].solo,
                        mute=self.state.speaker_channels[i].mute,
                    )
                    for i in range(8)
                ]
                self.state.update_mode2_speakers(updated_channels, overlap_detected=overlap)
                target_gain = 1.0 if gate_open else self.dsp_gate.ducking_floor
                self.dsp_gate.set_target_gain(target_gain)

            elif mode == AudioMode.AUTO_SOLO:
                gate_open, probs, overlap = self.diarization_mixer.process_chunk(ctx)
                locked_id = self.auto_solo.update_observation(ctx, probs)
                if locked_id is not None:
                    for i in range(8):
                        self.state.set_solo(i, i == locked_id)
                        self.diarization_mixer.set_solo(i, i == locked_id)

                gate_open, probs, overlap = self.diarization_mixer.evaluate_channel_activities(probs)
                updated_channels = [
                    SpeakerChannelState(
                        id=i,
                        active=(probs[i] >= self.diarization_mixer.activity_threshold),
                        prob=probs[i],
                        solo=self.state.speaker_channels[i].solo,
                        mute=self.state.speaker_channels[i].mute,
                    )
                    for i in range(8)
                ]
                self.state.update_mode2_speakers(updated_channels, overlap_detected=overlap)
                target_gain = 1.0 if gate_open else self.dsp_gate.ducking_floor
                self.dsp_gate.set_target_gain(target_gain)

            elif mode == AudioMode.PASSTHROUGH:
                self.dsp_gate.set_target_gain(1.0)

    def switch_mode(self, mode: AudioMode) -> None:
        """Dynamically switches mode without restarting the audio stream."""
        self.state.set_mode(mode)
        if mode == AudioMode.AUTO_SOLO:
            self.auto_solo.reset()

    def start(self) -> None:
        """Starts audio streaming and neural inference worker."""
        if self._running:
            return
        self._running = True

        self.stream_manager.start()
        self._inference_thread = threading.Thread(target=self._inference_worker, daemon=True)
        self._inference_thread.start()

    def stop(self) -> None:
        """Safely stops audio streams and worker threads."""
        if not self._running:
            return
        self._running = False

        self.stream_manager.stop()
        if self._inference_thread is not None:
            self._inference_thread.join(timeout=1.0)
            self._inference_thread = None

    def is_running(self) -> bool:
        """Returns True if the engine is running."""
        return self._running
