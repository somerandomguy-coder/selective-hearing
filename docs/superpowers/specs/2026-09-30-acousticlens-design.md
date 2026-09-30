# Design Specification: AcousticLens (Real-Time Selective Hearing & Audio Gater)

## 1. Executive Summary
AcousticLens is a low-latency, real-time audio processing desktop application designed to isolate or gate human voices from microphone inputs and route the processed signal to virtual audio sinks (e.g., Discord, OBS, headphones, or virtual audio cables).

The system supports three primary gating modes:
1. **Mode 1: Enrolled Voice Gate (Personalized VAD)**: Tracks a pre-enrolled user voice embedding (using ECAPA-TDNN / TitaNet) and opens the audio gate when cosine similarity exceeds a defined threshold (default: 0.72).
2. **Mode 2: Streaming Diarization Mixer**: Employs real-time multi-speaker diarization (NVIDIA Nemotron-3 Diarization / Sortformer) to track up to 8 arrival-time-ordered speaker channels with independent Solo/Mute controls and overlap handling.
3. **Hybrid Mode: Auto-Solo**: Automatically matches the enrolled voice against the 8 diarization slots to identify and lock onto the host speaker's channel without manual channel selection.

AcousticLens is designed with a decoupled architecture featuring both an interactive **Textual TUI** for local terminal usage and an optional **FastAPI + WebSocket** server for remote web/mobile control.

---

## 2. System Architecture & Component Design

```
+---------------------------------------------------------------------------------+
|                                 AcousticLens                                    |
|                                                                                 |
|  +--------------------+                                                         |
|  |  Audio Input       |                                                         |
|  |  (sounddevice 16k) |                                                         |
|  +---------+----------+                                                         |
|            | (Blocks: 80ms / 1280 samples)                                      |
|            v                                                                    |
|  +----------------------------------------------------+                         |
|  | AudioBufferManager                                 |                         |
|  | - Circular Delay Ring Buffer (200-350ms Lookahead) |                         |
|  | - Sliding Context Window (0.32s - 1.04s)           |                         |
|  +---------+------------------------------------------+                         |
|            |                                |                                   |
|   (Delayed Audio Stream)          (Inference Context Window)                    |
|            |                                v                                   |
|            |                  +-----------------------------+                   |
|            |                  | Neural Inference Dispatcher |                   |
|            |                  | - Mode 1: ECAPA-TDNN Gate   |                   |
|            |                  | - Mode 2: Sortformer Mixer  |                   |
|            |                  | - Mode 3: Auto-Solo Matcher |                   |
|            |                  +--------------+--------------+                   |
|            |                                 |                                  |
|            |                      (Target Gate Gain: 0.0 - 1.0)                 |
|            v                                 v                                  |
|  +----------------------------------------------------+                         |
|  | SoftKneeDSPGate                                    |                         |
|  | - 20ms Attack EMA, 250ms Release EMA + Hold        |                         |
|  | - Ducking Floor (-24 dB or full mute)              |                         |
|  | - Click-free C1 continuous gain smoothing          |                         |
|  +--------------------+-------------------------------+                         |
|                       |                                                         |
|                       v                                                         |
|            +--------------------+                                               |
|            | Audio Output Sink  |                                               |
|            | (VB-Cable / Phones)|                                               |
|            +--------------------+                                               |
+---------------------------------------------------------------------------------+
                       ^
                       | (Thread-safe State & Telemetry)
+----------------------+-----------------------+
| AcousticState Hub                            |
+----------------------+-----------------------+
|                      |                       |
v                      v                       v
Textual TUI            FastAPI Web/WebSocket   Headless / CLI
(Local Terminal)       (Remote / Mobile LAN)   (Daemon / Automation)
```

---

## 3. Detailed Module Specifications

### 3.1 Audio Buffer Manager (`engine/audio_buffer.py`)
- **Sample Rate**: 16,000 Hz, single channel (mono float32).
- **Chunk Size**: Default 1,280 samples (80ms) or 2,560 samples (160ms).
- **Delay Ring Buffer**:
  - Capacity: Configurable (default: 16,000 samples = 1.0 second).
  - Lookahead delay: Configurable (default: 250ms = 4,000 samples).
  - Guarantees that neural inference latency (~40–100ms) finishes before the corresponding audio slice reaches the output gate, eliminating clipping on speech onset.
- **Sliding Context Window**:
  - Maintains the recent 0.32s to 1.04s of uncorrupted input audio for neural embeddings and diarization frames.

### 3.2 Soft-Knee DSP Gate (`engine/dsp_gate.py`)
- **Exponential Moving Average (EMA)** smoothing prevents step discontinuities that generate audible clicks.
- **Attack Time**: $\tau_{\text{attack}} = 20\,\text{ms}$, $\alpha_{\text{attack}} = 1 - \exp\left(-\frac{1}{f_s \cdot 0.020}\right)$.
- **Release Time**: $\tau_{\text{release}} = 250\,\text{ms}$, $\alpha_{\text{release}} = 1 - \exp\left(-\frac{1}{f_s \cdot 0.250}\right)$.
- **Hold Time**: 150ms hold counter to bridge micro-pauses between words.
- **Ducking Floor**: Configurable linear gain $g_{\text{duck}}$ (default: $-24\,\text{dB} \approx 0.0631$ or $0.0$).
- **Sample-Accurate Processing**:
  $$\text{gain}[n] = \alpha \cdot \text{target\_gain} + (1 - \alpha) \cdot \text{gain}[n-1]$$
  $$\text{output}[n] = \text{input}[n] \cdot \text{gain}[n]$$

### 3.3 Mode 1: Enrolled Speaker Verifier (`modes/enrolled_gate.py`)
- **Model**: `speechbrain/spkrec-ecapa-voxceleb` (fallback/custom: NVIDIA `titanet_large`).
- **Enrollment Phase**:
  - Ingests a 5-second calibration utterance from host.
  - Extracts reference embedding $\mathbf{E}_{\text{ref}} \in \mathbb{R}^{D}$, applies $L_2$-normalization.
  - Persists embedding to disk (`data/profiles/default_profile.npy`).
- **Streaming Verification**:
  - For each sliding context window $W$, computes $\mathbf{E}_{\text{chunk}} = \text{Normalize}(\text{Model}(W))$.
  - Cosine similarity: $S = \mathbf{E}_{\text{chunk}} \cdot \mathbf{E}_{\text{ref}}$.
  - If $S \ge \text{threshold}$ (default: 0.72): `target_gain = 1.0`.
  - If $S < \text{threshold}$: `target_gain = ducking_floor`.

### 3.4 Mode 2: Streaming Diarization Mixer (`modes/diarization_mixer.py`)
- **Model**: NVIDIA Sortformer architecture (`nvidia/Nemotron-3-Diarization` / NeMo streaming diarizer).
- **Output**: Matrix of shape $[T, 8]$ representing frame-level probabilities for up to 8 arrival-time-ordered speaker slots.
- **Channel State Vector**:
  - `solo_mask`: `[bool] * 8` (default: `[True, False, False, False, False, False, False, False]`).
  - `mute_mask`: `[bool] * 8` (default: all `False`).
  - `activity_threshold`: $\tau_{\text{diar}} = 0.5$.
- **Arbitration Logic**:
  - Speaker $k$ is considered active if $\max_t P_k(t) \ge \tau_{\text{diar}}$.
  - If any soloed speaker is active: `target_gain = 1.0`.
  - If non-soloed speakers are active and NO soloed speaker is active: `target_gain = ducking_floor`.
  - Overlap handling: If both soloed and non-soloed speakers are active simultaneously, `target_gain = 1.0` and `overlap_flag = True`.

### 3.5 Hybrid Mode: Auto-Solo (`modes/auto_solo.py`)
- During active dialogue, segments where an individual diarization slot $k$ is exclusively active are extracted.
- An embedding is computed for the segment and compared with $\mathbf{E}_{\text{ref}}$.
- When similarity exceeds correlation threshold ($\ge 0.70$), slot $k$ is automatically assigned to the Host and set to Solo.

---

## 4. Control Interfaces & User Experience

### 4.1 Centralized Observable State (`engine/state.py`)
A thread-safe state container tracking:
- `active_mode`: `ENROLLED_GATE` | `DIARIZATION_MIXER` | `AUTO_SOLO` | `PASSTHROUGH`
- `rms_level_db`: Current input level in dBFS
- `smoothed_gain`: Current DSP envelope gain (0.0 to 1.0)
- `gate_status`: `OPEN` | `DUCKED` | `MUTED`
- `similarity_score`: 0.0 to 1.0 (Mode 1)
- `similarity_threshold`: Tunable threshold (0.50 to 0.95)
- `speaker_channels`: 8-element list of objects: `{ id: int, active: bool, prob: float, solo: bool, mute: bool }`
- `overlap_detected`: bool
- `is_enrolling`: bool, progress: float (0.0 to 1.0)

### 4.2 Textual TUI (`ui/tui_app.py`)
- Header with app branding, audio devices, and latency readout.
- Mode panel with instant hotkey toggling (`[1]`, `[2]`, `[3]`, `[4]`).
- Live horizontal VU-meter and Gain bar.
- Mode 1: Cosine similarity indicator with visual threshold marker and `[E]` enrollment trigger.
- Mode 2: 8 speaker slots with LED indicators, Solo (`[S]`) and Mute (`[M]`) buttons.
- Global status bar with active warnings (e.g., underrun, overlap).

### 4.3 Remote Web Control (`ui/web_server.py` & `ui/web_static/`)
- Lightweight FastAPI app with WebSocket endpoint `/ws/telemetry`.
- Serves an ultra-fast Vanilla HTML/CSS/JS dark-mode dashboard.
- Broadcasts JSON telemetry at 25Hz.
- REST/WS endpoints for toggling solo/mute, changing thresholds, and triggering calibration.

---

## 5. Hardware Acceleration & Fallbacks
- **CUDA Acceleration**: Detects PyTorch CUDA; enables `torch.cuda.is_available()` with `fp16` inference.
- **CPU Fallback**: Gracefully executes with multi-threaded CPU inference (`torch.set_num_threads(4)`).
- **Mock / Offline Mode**:
  - `--mock-audio`: Generates synthetic test tone signals without requiring physical microphone/speakers.
  - `--mock-models`: Generates synthetic speaker diarization and embedding similarities for development, CI, and testing on systems without GPU or pre-downloaded weights.

---

## 6. Testing & Quality Assurance
- `tests/test_dsp_gate.py`:
  - Validates attack time ($\sim 20\,\text{ms}$) and release time ($\sim 250\,\text{ms}$).
  - Verifies click prevention: Max first-order difference $\max |\Delta g| < 0.01$ per sample.
- `tests/test_audio_buffer.py`:
  - Ring buffer lookahead delay correctness and wraparound stability.
- `tests/test_modes.py`:
  - Mode 1 cosine similarity logic and boundary triggers.
  - Mode 2 solo/mute/overlap truth table.
- `tests/test_state.py`:
  - Thread-safe updates and subscriber callbacks.
