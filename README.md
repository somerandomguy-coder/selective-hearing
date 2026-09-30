# AcousticLens (Real-Time Selective Hearing & Audio Gater)

A low-latency, real-time audio processing desktop application in Python that selectively gates or isolates human voices from microphone input and outputs to a virtual audio device (for Discord, OBS, headphones, or virtual cables).

---

## 🌟 Key Features

1. **Mode 1: Enrolled Voice Gate (Personalized VAD)**
   - Calibrates a 5-second voice profile of the host using `speechbrain/spkrec-ecapa-voxceleb` (ECAPA-TDNN).
   - Real-time cosine similarity gating against the sliding audio context window.
   - Live tunable threshold (default: `0.72`).
2. **Mode 2: Zero-Shot Multi-Speaker Diarization Mixer**
   - Ingests streaming audio chunks into an 8-channel speaker activity matrix (`[T, 8]` Sortformer / Nemotron architecture).
   - Independent **Solo** and **Mute** channel controls for Speaker 0 through Speaker 7.
   - Real-time **Overlap Detection**: flags cross-talk while maintaining pass-through for the soloed speaker.
3. **Hybrid Mode: Auto-Solo**
   - Automatically cross-correlates the enrolled speaker embedding against the 8 diarization slots.
   - Locks onto the host's voice slot without manual channel configuration.
4. **Click-Free Soft-Knee DSP Gate**
   - Exponential Moving Average (EMA) gain envelope with 20ms attack, 250ms release, and 150ms hold time.
   - Prevents audible pops and clicks with $C^1$ continuous gain transitions and configurable ducking floor (-24 dB or full mute).
5. **Dual Control Interfaces**
   - **Textual TUI**: Native, responsive terminal dashboard with live VU meter, cosine similarity bar, and hotkeys.
   - **Remote Web Dashboard**: Modern FastAPI + WebSocket glassmorphic interface accessible from local PC or mobile browser on LAN (`http://localhost:8000`).

---

## 🏗️ Architecture

```
[Microphone In] ──> [AudioBufferManager] ──(Delayed Audio 250ms)──> [SoftKneeDSPGate] ──> [Virtual Cable / Out]
                            │                                              ▲
                   (Context Window 0.8s)                                   │ (Target Gain)
                            │                                              │
                            ▼                                              │
               [Neural Inference Dispatcher] ──────────────────────────────┘
               - Mode 1: ECAPA-TDNN Gate
               - Mode 2: Sortformer Diarization Mixer
               - Hybrid: Auto-Solo Matcher
                            ▲
                            │ (State Sync)
                 [AcousticState Hub]
                     /          \
      [Textual TUI]            [FastAPI Web/WebSocket]
```

---

## 🚀 Quick Start

### 1. Prerequisites & Environment
Ensure Python 3.11 and `uv` are installed:

```bash
# Clone the repository
git clone <your-repo-url>
cd selective-hearing

# Create virtual environment and install dependencies
uv venv --python 3.11 .venv
uv pip install -e .
```

### 2. Audio Routing Setup (Windows / macOS / Linux)
To route processed audio into Discord, OBS, or Zoom:
- **Windows**: Install [VB-Audio Virtual Cable](https://vb-audio.com/Cable/). Set AcousticLens output to `CABLE Input (VB-Audio Virtual Cable)`, and select `CABLE Output` as your microphone in Discord/OBS.
- **macOS**: Install [BlackHole](https://github.com/ExistentialAudio/BlackHole) virtual audio driver.
- **Linux**: Use a PulseAudio / PipeWire loopback sink.

List your local audio devices:
```bash
uv run python main.py --list-devices
```

---

## 🎮 Usage & Running

### Option A: Interactive Textual TUI (Default)
Launch the full terminal dashboard:
```bash
uv run python main.py
```

### Option B: TUI + Remote Web Dashboard (PC & Mobile)
Launch with the web control surface active on port 8000:
```bash
uv run python main.py --web
```
Open `http://localhost:8000` (or `http://<your-pc-lan-ip>:8000` on your smartphone).

### Option C: Headless Web Server (Daemon Mode)
Run as a background service without terminal TUI:
```bash
uv run python main.py --web --headless
```

### Option D: Instant Mock / Offline Mode
Run with synthetic audio and mock models (zero audio hardware or weight downloads needed):
```bash
uv run python main.py --mock --web
```

---

## ⌨️ Keyboard Shortcuts (TUI)

| Key | Action |
|:---:|:---|
| `1` | Switch to **Mode 1: Enrolled Gate** |
| `2` | Switch to **Mode 2: Diarization Mixer** |
| `3` | Switch to **Hybrid: Auto-Solo** |
| `4` | Switch to **Pass-Through** (unmuted) |
| `E` | Re-enroll 5-second host calibration voice |
| `0` - `7` | Quick-toggle Solo on Speaker 0 through 7 |
| `Q` | Safely shutdown audio stream and exit |

---

## 🧪 Running the Test Suite

Run all automated unit and integration tests:
```bash
uv run pytest -v
```

All core modules are covered:
- `test_dsp_gate.py`: Click-free envelope follower, attack/release timing, hold time.
- `test_audio_buffer.py`: Ring buffer delay, lookahead correctness, wraparound stability.
- `test_state.py`: Thread-safe observable state and listener notifications.
- `test_enrolled_gate.py`: ECAPA-TDNN feature extraction, $L_2$ normalization, profile persistence.
- `test_diarization_mixer.py`: 8-channel Sortformer arbitration, solo/mute matrix, overlap detection.
- `test_auto_solo.py`: Auto-solo matching and slot locking.
- `test_orchestrator.py`: Master stream pipeline lifecycle and dynamic mode switching.
- `test_ui_app.py`: Textual TUI key bindings and UI lifecycle.
- `test_web_server.py`: FastAPI REST API and WebSocket telemetry stream.
