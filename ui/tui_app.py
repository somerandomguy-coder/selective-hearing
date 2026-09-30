from typing import Optional
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Container, Grid, Horizontal, Vertical
from textual.widgets import Button, Footer, Header, Label, ProgressBar, Static

from engine.orchestrator import AcousticLensEngine
from engine.state import AcousticState, AudioMode


class TelemetryPanel(Static):
    """Displays live audio input level and DSP gain envelope."""

    def compose(self) -> ComposeResult:
        yield Label("INPUT LEVEL (dBFS):", classes="metric-title")
        yield Label("-100.0 dBFS", id="lbl-rms")
        yield Label("DSP GAIN ENVELOPE:", classes="metric-title")
        yield Label("0.000 [MUTED]", id="lbl-gain")


class Mode1Panel(Static):
    """Displays Mode 1 Enrolled Voice Gate similarity and calibration controls."""

    def compose(self) -> ComposeResult:
        yield Label("MODE 1: ENROLLED VOICE GATE", classes="panel-title")
        yield Label("Cosine Similarity: 0.00 / 0.72", id="lbl-similarity")
        yield ProgressBar(id="prog-similarity", total=100, show_percentage=True)
        yield Label("Enrollment: Ready (Press [E] to calibrate 5s)", id="lbl-enroll")


class SpeakerSlotWidget(Static):
    """Represents one of the 8 arrival-time-ordered speaker diarization slots."""

    def __init__(self, speaker_id: int, **kwargs) -> None:
        super().__init__(**kwargs)
        self.speaker_id = speaker_id

    def compose(self) -> ComposeResult:
        yield Label(f"Speaker {self.speaker_id}", classes="speaker-title")
        yield Label("● SILENT", id=f"spk-led-{self.speaker_id}", classes="spk-led")
        yield Label("Prob: 0.00", id=f"spk-prob-{self.speaker_id}")
        with Horizontal():
            yield Button(
                f"S{self.speaker_id}", id=f"btn-solo-{self.speaker_id}", classes="btn-solo"
            )
            yield Button(
                f"M{self.speaker_id}", id=f"btn-mute-{self.speaker_id}", classes="btn-mute"
            )


class Mode2Panel(Static):
    """Displays 8-channel diarization mixer matrix with solo/mute controls."""

    def compose(self) -> ComposeResult:
        yield Label("MODE 2: 8-SPEAKER DIARIZATION MIXER", classes="panel-title")
        yield Label("Overlap: OFF", id="lbl-overlap", classes="lbl-overlap")
        with Grid(classes="speaker-grid"):
            for i in range(8):
                yield SpeakerSlotWidget(speaker_id=i)


class AcousticLensTUI(App):
    """Interactive full-terminal dashboard for AcousticLens."""

    CSS = """
    Screen {
        background: #0f172a;
        color: #f8fafc;
    }
    .panel-title {
        text-style: bold;
        color: #38bdf8;
    }
    .metric-title {
        color: #94a3b8;
    }
    TelemetryPanel {
        border: solid #1e293b;
        background: #1e293b;
        padding: 1;
        margin-bottom: 1;
        height: auto;
    }
    Mode1Panel {
        border: solid #1e293b;
        background: #1e293b;
        padding: 1;
        margin-bottom: 1;
        height: auto;
    }
    Mode2Panel {
        border: solid #1e293b;
        background: #1e293b;
        padding: 1;
        height: auto;
    }
    .speaker-grid {
        grid-size: 4 2;
        grid-gutter: 1;
        height: auto;
    }
    SpeakerSlotWidget {
        border: round #334155;
        background: #090d16;
        padding: 1;
        height: auto;
    }
    .speaker-title {
        text-style: bold;
        color: #a855f7;
    }
    .spk-led {
        color: #64748b;
    }
    .btn-solo {
        min-width: 5;
        height: 1;
        background: #3b82f6;
        color: #ffffff;
        margin-right: 1;
    }
    .btn-mute {
        min-width: 5;
        height: 1;
        background: #ef4444;
        color: #ffffff;
    }
    """

    BINDINGS = [
        Binding("1", "mode_enrolled", "Enrolled Gate", priority=True),
        Binding("2", "mode_diarization", "Diarization Mixer", priority=True),
        Binding("3", "mode_autosolo", "Auto-Solo", priority=True),
        Binding("4", "mode_passthrough", "Passthrough", priority=True),
        Binding("e", "enroll", "Re-Enroll", priority=True),
        Binding("q", "quit", "Quit", priority=True),
    ] + [
        Binding(str(i), f"solo_{i}", f"Solo {i}", show=False, priority=True)
        for i in range(8)
    ]

    def __init__(
        self,
        engine: AcousticLensEngine,
        state: AcousticState,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.engine = engine
        self.state = state

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        yield Label("ACTIVE MODE: [ENROLLED_GATE]", id="lbl-active-mode")
        with Vertical():
            yield TelemetryPanel()
            yield Mode1Panel()
            yield Mode2Panel()
        yield Footer()

    def on_mount(self) -> None:
        """Start auto-refresh polling timer."""
        self.set_interval(0.05, self._refresh_telemetry)

    def _refresh_telemetry(self) -> None:
        """Updates UI widgets from centralized state."""
        try:
            # Active Mode
            mode_lbl = self.query_one("#lbl-active-mode", Label)
            mode_lbl.update(f"ACTIVE MODE: [{self.state.active_mode.value.upper()}]")

            # Telemetry
            rms_lbl = self.query_one("#lbl-rms", Label)
            rms_val = self.state.rms_level_db
            rms_lbl.update(f"{rms_val:0.1f} dBFS")

            gain_lbl = self.query_one("#lbl-gain", Label)
            gain_val = self.state.smoothed_gain
            status = self.state.gate_status
            gain_lbl.update(f"{gain_val:0.3f} [{status}]")

            # Mode 1
            sim_lbl = self.query_one("#lbl-similarity", Label)
            sim_val = self.state.similarity_score
            thresh_val = self.state.similarity_threshold
            sim_lbl.update(f"Cosine Similarity: {sim_val:0.2f} / {thresh_val:0.2f}")

            prog = self.query_one("#prog-similarity", ProgressBar)
            prog.update(progress=int(sim_val * 100))

            enroll_lbl = self.query_one("#lbl-enroll", Label)
            if self.state.is_enrolling:
                pct = int(self.state.enroll_progress * 100)
                enroll_lbl.update(f"Enrollment: Recording... {pct}%")
            else:
                enroll_lbl.update("Enrollment: Ready (Press [E] to calibrate 5s)")

            # Mode 2 Overlap
            overlap_lbl = self.query_one("#lbl-overlap", Label)
            overlap_lbl.update(
                "Overlap: [bold red]DETECTED[/]"
                if self.state.overlap_detected
                else "Overlap: OFF"
            )

            # 8 Speaker slots
            for i in range(8):
                ch = self.state.speaker_channels[i]
                led_lbl = self.query_one(f"#spk-led-{i}", Label)
                prob_lbl = self.query_one(f"#spk-prob-{i}", Label)

                if ch.mute:
                    led_lbl.update("● MUTED")
                elif ch.active and ch.solo:
                    led_lbl.update("● ACTIVE (SOLO)")
                elif ch.active:
                    led_lbl.update("● ACTIVE")
                else:
                    led_lbl.update("● SILENT")

                prob_lbl.update(f"Prob: {ch.prob:0.2f} | Solo: {ch.solo}")
        except Exception:
            pass

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button clicks for Solo and Mute."""
        btn_id = event.button.id or ""
        if btn_id.startswith("btn-solo-"):
            spk_id = int(btn_id.split("-")[-1])
            curr = self.state.speaker_channels[spk_id].solo
            self.state.set_solo(spk_id, not curr)
        elif btn_id.startswith("btn-mute-"):
            spk_id = int(btn_id.split("-")[-1])
            curr = self.state.speaker_channels[spk_id].mute
            self.state.set_mute(spk_id, not curr)

    def action_mode_enrolled(self) -> None:
        self.engine.switch_mode(AudioMode.ENROLLED_GATE)

    def action_mode_diarization(self) -> None:
        self.engine.switch_mode(AudioMode.DIARIZATION_MIXER)

    def action_mode_autosolo(self) -> None:
        self.engine.switch_mode(AudioMode.AUTO_SOLO)

    def action_mode_passthrough(self) -> None:
        self.engine.switch_mode(AudioMode.PASSTHROUGH)

    def action_enroll(self) -> None:
        self.engine.trigger_enrollment(duration_s=5.0)

    def action_solo_0(self) -> None: self._toggle_solo(0)
    def action_solo_1(self) -> None: self._toggle_solo(1)
    def action_solo_2(self) -> None: self._toggle_solo(2)
    def action_solo_3(self) -> None: self._toggle_solo(3)
    def action_solo_4(self) -> None: self._toggle_solo(4)
    def action_solo_5(self) -> None: self._toggle_solo(5)
    def action_solo_6(self) -> None: self._toggle_solo(6)
    def action_solo_7(self) -> None: self._toggle_solo(7)

    def _toggle_solo(self, idx: int) -> None:
        curr = self.state.speaker_channels[idx].solo
        self.state.set_solo(idx, not curr)

    def action_quit(self) -> None:
        self.engine.stop()
        self.exit()
