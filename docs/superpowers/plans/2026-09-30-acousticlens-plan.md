# AcousticLens Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a real-time, low-latency selective hearing and audio gater desktop application in Python with Enrolled Voice Gate (ECAPA-TDNN), Streaming Diarization Mixer (Nemotron/Sortformer), Hybrid Auto-Solo, Textual TUI, and optional FastAPI/WebSocket remote web control.

**Architecture:** An asynchronous audio stream pipeline decouples real-time audio I/O (sounddevice/numpy) through a circular delay buffer (~250ms lookahead) and a soft-knee exponential moving average (EMA) DSP gate. Neural inference workers evaluate speaker embeddings and 8-channel diarization probabilities on sliding context windows, updating a thread-safe centralized state hub that drives both a Textual terminal dashboard and a WebSocket web client.

**Tech Stack:** Python 3.11, sounddevice, numpy, scipy, PyTorch, SpeechBrain, Textual, Rich, FastAPI, Uvicorn, WebSockets, Pytest.

## Global Constraints
- Sample rate: 16,000 Hz, mono float32.
- Audio block size: 80ms (1,280 samples) or 160ms (2,560 samples).
- Click-free gating: Sample-accurate exponential gain ramps (20ms attack, 250ms release, 150ms hold).
- Zero audio I/O blocking: Neural inference must execute in separate worker threads.
- Graceful fallbacks: Automatic GPU (CUDA fp16) vs CPU detection and built-in mock modes (`--mock`) for offline testing without audio hardware or weights.
- Always use git commit for every task deliverable.

---

### Task 1: Project Scaffolding & Dependencies

**Files:**
- Create: `pyproject.toml`
- Create: `.gitignore`
- Create: `engine/__init__.py`
- Create: `modes/__init__.py`
- Create: `ui/__init__.py`
- Create: `tests/__init__.py`

**Interfaces:**
- Produces: Virtual environment and installable project package `acousticlens` with all core dependencies.

- [ ] **Step 1: Write `pyproject.toml` and `.gitignore`**
Define package dependencies: `sounddevice`, `numpy`, `scipy`, `torch`, `speechbrain`, `textual`, `rich`, `fastapi`, `uvicorn`, `websockets`, `pytest`, `pytest-asyncio`, `requests`.

- [ ] **Step 2: Initialize virtual environment using uv**
Run: `uv venv --python 3.11 .venv`
Expected: Virtual environment created.

- [ ] **Step 3: Install dependencies**
Run: `uv pip install -e .` (or install dependencies into `.venv`)
Expected: Packages installed successfully.

- [ ] **Step 4: Verify test suite runner**
Run: `uv run pytest -v`
Expected: 0 passed, no errors (empty suite).

- [ ] **Step 5: Commit**
```bash
git add pyproject.toml .gitignore engine/__init__.py modes/__init__.py ui/__init__.py tests/__init__.py
git commit -m "chore: scaffold project structure and dependencies"
```

---

### Task 2: Click-Free Soft-Knee DSP Gate (`engine/dsp_gate.py`)

**Files:**
- Create: `engine/dsp_gate.py`
- Test: `tests/test_dsp_gate.py`

**Interfaces:**
- Consumes: `sample_rate: int = 16000`, `attack_ms: float = 20.0`, `release_ms: float = 250.0`, `hold_ms: float = 150.0`, `ducking_floor_db: float = -24.0`
- Produces: `SoftKneeDSPGate` class:
  - `set_target_gain(target: float) -> None`
  - `process(chunk: np.ndarray) -> np.ndarray`
  - `get_current_gain() -> float`
  - `get_gate_status() -> str` ("OPEN", "DUCKED", "MUTED")

- [ ] **Step 1: Write failing test in `tests/test_dsp_gate.py`**
Test that:
1. When target gain transitions from 0.0 to 1.0, the gain smoothly rises over ~20ms without any step discontinuity (max delta per sample < 0.01).
2. When target gain transitions from 1.0 to 0.0, the hold time keeps gain at 1.0 for 150ms before releasing smoothly over ~250ms.
3. Audio processed with ducking floor matches expected dB attenuation.

- [ ] **Step 2: Run test to verify it fails**
Run: `uv run pytest tests/test_dsp_gate.py -v`
Expected: FAIL (ModuleNotFoundError)

- [ ] **Step 3: Implement `SoftKneeDSPGate` in `engine/dsp_gate.py`**
Implement sample-by-sample or vectorized exponential moving average:
$$\alpha_{\text{attack}} = 1 - \exp(-1 / (f_s \cdot (\tau_{\text{attack}}/1000)))$$
$$\alpha_{\text{release}} = 1 - \exp(-1 / (f_s \cdot (\tau_{\text{release}}/1000)))$$
Includes hold counter (in samples) and ducking floor conversion ($10^{\text{floor\_db}/20}$).

- [ ] **Step 4: Run test to verify it passes**
Run: `uv run pytest tests/test_dsp_gate.py -v`
Expected: PASS

- [ ] **Step 5: Commit**
```bash
git add engine/dsp_gate.py tests/test_dsp_gate.py
git commit -m "feat(engine): implement click-free soft-knee DSP gate"
```

---

### Task 3: Audio Buffer Manager & Circular Delay (`engine/audio_buffer.py`)

**Files:**
- Create: `engine/audio_buffer.py`
- Test: `tests/test_audio_buffer.py`

**Interfaces:**
- Consumes: Raw 16kHz audio chunks from input callback.
- Produces: `AudioBufferManager` class:
  - `push_input(chunk: np.ndarray) -> None`
  - `pop_delayed(chunk_size: int) -> np.ndarray` (delayed by `lookahead_ms`, e.g. 250ms)
  - `get_context_window(window_duration_s: float) -> np.ndarray` (sliding recent window for neural models)
  - `get_current_rms_db() -> float`

- [ ] **Step 1: Write failing test in `tests/test_audio_buffer.py`**
Test that:
1. Pushing audio chunks stores them in ring buffer.
2. Popping delayed chunk with 250ms delay returns silence for the first 250ms, then reproduces the input stream accurately.
3. `get_context_window(0.8)` returns the most recent 0.8s (12,800 samples) of audio.
4. Wraparound across the ring buffer boundary produces contiguous data without corruption.

- [ ] **Step 2: Run test to verify it fails**
Run: `uv run pytest tests/test_audio_buffer.py -v`
Expected: FAIL

- [ ] **Step 3: Implement `AudioBufferManager` in `engine/audio_buffer.py`**
Implement thread-safe circular ring buffer with numpy array storage, atomic write/read indices, and lookahead offset.

- [ ] **Step 4: Run test to verify it passes**
Run: `uv run pytest tests/test_audio_buffer.py -v`
Expected: PASS

- [ ] **Step 5: Commit**
```bash
git add engine/audio_buffer.py tests/test_audio_buffer.py
git commit -m "feat(engine): implement circular delay ring buffer and context window manager"
```

---

### Task 4: Centralized Observable State Hub (`engine/state.py`)

**Files:**
- Create: `engine/state.py`
- Test: `tests/test_state.py`

**Interfaces:**
- Produces: `AudioMode` (Enum: `ENROLLED_GATE`, `DIARIZATION_MIXER`, `AUTO_SOLO`, `PASSTHROUGH`), `SpeakerState` (dataclass), and `AcousticState` observable hub:
  - `update_telemetry(rms_db, gain, gate_status)`
  - `update_mode1(similarity, threshold, is_enrolling, enroll_progress)`
  - `update_mode2_speakers(speakers_list, overlap_detected)`
  - `set_mode(mode: AudioMode)`
  - `set_solo(speaker_id: int, solo: bool)`
  - `set_mute(speaker_id: int, mute: bool)`
  - `register_listener(callback)`
  - `to_dict() -> dict` (JSON serializable for WebSocket/TUI)

- [ ] **Step 1: Write failing test in `tests/test_state.py`**
Verify thread-safe updates, listener notifications, and JSON serialization.

- [ ] **Step 2: Run test to verify it fails**
Run: `uv run pytest tests/test_state.py -v`
Expected: FAIL

- [ ] **Step 3: Implement `AcousticState` in `engine/state.py`**
Thread-safe dataclass with mutex locks, listener callbacks, and snapshot export.

- [ ] **Step 4: Run test to verify it passes**
Run: `uv run pytest tests/test_state.py -v`
Expected: PASS

- [ ] **Step 5: Commit**
```bash
git add engine/state.py tests/test_state.py
git commit -m "feat(engine): implement centralized observable state hub"
```

---

### Task 5: Mode 1 - Enrolled Speaker Verifier (`modes/enrolled_gate.py`)

**Files:**
- Create: `modes/enrolled_gate.py`
- Test: `tests/test_enrolled_gate.py`

**Interfaces:**
- Consumes: Audio context window (numpy float32), enrolled reference embedding vector.
- Produces: `EnrolledSpeakerGate`:
  - `enroll_from_audio(audio: np.ndarray, sample_rate: int = 16000) -> np.ndarray`
  - `save_profile(filepath: str) -> None`
  - `load_profile(filepath: str) -> bool`
  - `compute_similarity(chunk: np.ndarray) -> float`
  - `evaluate_gate(chunk: np.ndarray, threshold: float = 0.72) -> Tuple[bool, float]`
  - Supports `--mock` mode with synthetic cosine similarity generation.

- [ ] **Step 1: Write failing test in `tests/test_enrolled_gate.py`**
Test:
1. Enrollment normalizes reference embedding vector ($L_2 = 1.0$).
2. Identical audio produces similarity $\approx 1.0$. Orthogonal/dissimilar vectors produce low similarity.
3. Gate opens ($True$) when similarity $\ge threshold$ and closes ($False$) otherwise.
4. Profile persistence: saving to and loading from `.npy` or `.pt`.

- [ ] **Step 2: Run test to verify it fails**
Run: `uv run pytest tests/test_enrolled_gate.py -v`
Expected: FAIL

- [ ] **Step 3: Implement `EnrolledSpeakerGate` in `modes/enrolled_gate.py`**
Integrate `speechbrain/spkrec-ecapa-voxceleb` embedding extractor with fallback mock extractor. Handles device placement (`cuda` / `cpu`), $L_2$-normalization, and dot product cosine similarity.

- [ ] **Step 4: Run test to verify it passes**
Run: `uv run pytest tests/test_enrolled_gate.py -v`
Expected: PASS

- [ ] **Step 5: Commit**
```bash
git add modes/enrolled_gate.py tests/test_enrolled_gate.py
git commit -m "feat(modes): implement enrolled speaker verifier with ECAPA-TDNN"
```

---

### Task 6: Mode 2 - Streaming Diarization Mixer (`modes/diarization_mixer.py`)

**Files:**
- Create: `modes/diarization_mixer.py`
- Test: `tests/test_diarization_mixer.py`

**Interfaces:**
- Consumes: Streaming audio chunk (0.32s - 1.04s), `solo_mask`, `mute_mask`.
- Produces: `StreamingDiarizationMixer`:
  - `process_chunk(chunk: np.ndarray) -> Tuple[bool, List[float], bool]` (gate_open, speaker_probs[8], overlap_detected)
  - `set_solo(speaker_id: int, solo: bool) -> None`
  - `set_mute(speaker_id: int, mute: bool) -> None`
  - `reset_channels() -> None`
  - Supports streaming Sortformer / Nemotron inference and mock simulation fallback.

- [ ] **Step 1: Write failing test in `tests/test_diarization_mixer.py`**
Test:
1. With `Speaker 0` soloed: when Speaker 0 probability > 0.5, gate is open.
2. When only `Speaker 1` is active while Speaker 0 is silent, gate is closed (ducked).
3. Overlap test: when both soloed and non-soloed speakers speak simultaneously, gate remains open and `overlap_detected = True`.
4. Mute test: muting an active soloed speaker ducks the output.

- [ ] **Step 2: Run test to verify it fails**
Run: `uv run pytest tests/test_diarization_mixer.py -v`
Expected: FAIL

- [ ] **Step 3: Implement `StreamingDiarizationMixer` in `modes/diarization_mixer.py`**
Implements Sortformer/Nemotron streaming wrapper and multi-speaker channel arbitration matrix with 8 slots.

- [ ] **Step 4: Run test to verify it passes**
Run: `uv run pytest tests/test_diarization_mixer.py -v`
Expected: PASS

- [ ] **Step 5: Commit**
```bash
git add modes/diarization_mixer.py tests/test_diarization_mixer.py
git commit -m "feat(modes): implement streaming diarization mixer with 8-speaker solo/mute matrix"
```

---

### Task 7: Hybrid Mode - Auto-Solo Matcher (`modes/auto_solo.py`)

**Files:**
- Create: `modes/auto_solo.py`
- Test: `tests/test_auto_solo.py`

**Interfaces:**
- Consumes: Enrolled speaker embedding, diarization channel activations.
- Produces: `AutoSoloMatcher`:
  - `update_observation(audio_chunk: np.ndarray, speaker_probs: List[float]) -> Optional[int]`
  - `get_locked_speaker_id() -> Optional[int]`
  - `reset() -> None`

- [ ] **Step 1: Write failing test in `tests/test_auto_solo.py`**
Test that feeding isolated audio corresponding to the enrolled user locks onto that specific speaker slot ID and updates `locked_speaker_id`.

- [ ] **Step 2: Run test to verify it fails**
Run: `uv run pytest tests/test_auto_solo.py -v`
Expected: FAIL

- [ ] **Step 3: Implement `AutoSoloMatcher` in `modes/auto_solo.py`**
Accumulates exclusive voice activity per channel slot, computes speaker embedding, compares with reference embedding, and locks the best-matching channel to Solo.

- [ ] **Step 4: Run test to verify it passes**
Run: `uv run pytest tests/test_auto_solo.py -v`
Expected: PASS

- [ ] **Step 5: Commit**
```bash
git add modes/auto_solo.py tests/test_auto_solo.py
git commit -m "feat(modes): implement hybrid auto-solo speaker matcher"
```

---

### Task 8: Audio Engine Stream Pipeline & Orchestrator (`engine/orchestrator.py`)

**Files:**
- Create: `engine/audio_stream.py`
- Create: `engine/orchestrator.py`
- Test: `tests/test_orchestrator.py`

**Interfaces:**
- Coordinates: `AudioBufferManager`, `SoftKneeDSPGate`, `EnrolledSpeakerGate`, `StreamingDiarizationMixer`, `AutoSoloMatcher`, and `AcousticState`.
- Produces: `AcousticLensEngine`:
  - `start() -> None`
  - `stop() -> None`
  - `switch_mode(mode: AudioMode) -> None`
  - `trigger_enrollment(duration_s: float = 5.0) -> None`
  - `list_devices() -> Tuple[List[dict], List[dict]]`
  - Supports `--mock-audio` and `--mock-models`.

- [ ] **Step 1: Write integration test in `tests/test_orchestrator.py`**
Test starting and stopping engine with mock audio, dynamic mode switching without dropping audio, and telemetry updates.

- [ ] **Step 2: Run test to verify it fails**
Run: `uv run pytest tests/test_orchestrator.py -v`
Expected: FAIL

- [ ] **Step 3: Implement `AudioStream` and `AcousticLensEngine`**
Creates real-time audio input/output callbacks via `sounddevice` with synthetic fallback stream. Spawns dedicated background inference worker thread that polls context window and dispatches gating decisions to the DSP gate.

- [ ] **Step 4: Run test to verify it passes**
Run: `uv run pytest tests/test_orchestrator.py -v`
Expected: PASS

- [ ] **Step 5: Commit**
```bash
git add engine/audio_stream.py engine/orchestrator.py tests/test_orchestrator.py
git commit -m "feat(engine): implement audio stream pipeline and engine orchestrator"
```

---

### Task 9: Textual Interactive Terminal UI (`ui/tui_app.py`)

**Files:**
- Create: `ui/tui_app.py`
- Test: `tests/test_ui_app.py`

**Interfaces:**
- Consumes: `AcousticLensEngine` and `AcousticState`.
- Produces: `AcousticLensTUI(App)` Textual application:
  - Visual VU meter and Gain envelope.
  - Mode selector buttons (`[1] Enrolled Gate`, `[2] Diarization Mixer`, `[3] Auto-Solo`, `[4] Pass-through`).
  - Cosine similarity progress bar with threshold indicator.
  - 8-channel speaker matrix with Solo (`[S]`) and Mute (`[M]`) buttons.
  - Keyboard shortcuts: `1`-`4`, `0`-`7`, `E` (enroll), `Q` (quit).

- [ ] **Step 1: Write test in `tests/test_ui_app.py`**
Test TUI initialization, widget composition, and key bindings handling without crashing.

- [ ] **Step 2: Run test to verify it fails**
Run: `uv run pytest tests/test_ui_app.py -v`
Expected: FAIL

- [ ] **Step 3: Implement `AcousticLensTUI` in `ui/tui_app.py`**
Build full Textual TUI dashboard with live telemetry polling, color-coded VU meters, responsive widgets, and keyboard bindings.

- [ ] **Step 4: Run test to verify it passes**
Run: `uv run pytest tests/test_ui_app.py -v`
Expected: PASS

- [ ] **Step 5: Commit**
```bash
git add ui/tui_app.py tests/test_ui_app.py
git commit -m "feat(ui): implement interactive Textual TUI dashboard"
```

---

### Task 10: Remote Web Control Surface (`ui/web_server.py` & Web Assets)

**Files:**
- Create: `ui/web_server.py`
- Create: `ui/web_static/index.html`
- Create: `ui/web_static/style.css`
- Create: `ui/web_static/app.js`
- Test: `tests/test_web_server.py`

**Interfaces:**
- Consumes: `AcousticState` and `AcousticLensEngine`.
- Produces:
  - FastAPI server with endpoints:
    - `GET /` (Serves responsive web app)
    - `WS /ws/telemetry` (25Hz live state telemetry)
    - `POST /api/mode` (Switch mode)
    - `POST /api/solo` (Toggle solo)
    - `POST /api/mute` (Toggle mute)
    - `POST /api/threshold` (Set threshold)
    - `POST /api/enroll` (Trigger calibration)
  - Sleek dark-mode glassmorphic web dashboard.

- [ ] **Step 1: Write failing test in `tests/test_web_server.py`**
Test FastAPI endpoints and WebSocket telemetry connection using `starlette.testclient.TestClient`.

- [ ] **Step 2: Run test to verify it fails**
Run: `uv run pytest tests/test_web_server.py -v`
Expected: FAIL

- [ ] **Step 3: Implement `web_server.py` and static assets**
Build FastAPI routes, WebSocket broadcast worker, and premium dark-mode responsive HTML/CSS/JS frontend.

- [ ] **Step 4: Run test to verify it passes**
Run: `uv run pytest tests/test_web_server.py -v`
Expected: PASS

- [ ] **Step 5: Commit**
```bash
git add ui/web_server.py ui/web_static/ tests/test_web_server.py
git commit -m "feat(ui): implement remote FastAPI WebSocket control dashboard"
```

---

### Task 11: Main CLI Entry Point & End-to-End Verification (`main.py`, `README.md`)

**Files:**
- Create: `main.py`
- Create: `README.md`
- Test: Full test suite `pytest tests/`

**Interfaces:**
- Command-line arguments:
  - `--mode`: Initial mode (`enrolled`, `diarization`, `autosolo`, `passthrough`)
  - `--web`: Enable remote web server (default port 8000)
  - `--tui`: Launch Textual TUI (default True)
  - `--mock`: Run in mock mode for instant demonstration without audio hardware or neural weights
  - `--list-devices`: List available microphone and speaker devices.

- [ ] **Step 1: Implement `main.py` CLI parser and lifecycle runner**
Parses CLI flags, initializes `AcousticLensEngine`, conditionally launches FastAPI web server in a background thread, and runs the Textual TUI on main thread (or headless loop).

- [ ] **Step 2: Write comprehensive `README.md`**
Document installation, hardware requirements, virtual audio cable routing (VB-Cable/BlackHole), CLI flags, shortcuts, and usage examples.

- [ ] **Step 3: Run full automated test suite**
Run: `uv run pytest tests/ -v`
Expected: ALL tests PASS.

- [ ] **Step 4: Test mock run of `main.py`**
Run: `uv run python main.py --mock --help`
Verify argument parsing and help output.

- [ ] **Step 5: Commit**
```bash
git add main.py README.md
git commit -m "feat: complete AcousticLens application entry point and documentation"
```
