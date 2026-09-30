import argparse
import sys
import threading
import time
import uvicorn

from engine.audio_stream import AudioStreamManager
from engine.orchestrator import AcousticLensEngine
from engine.state import AcousticState, AudioMode
from ui.tui_app import AcousticLensTUI
from ui.web_server import create_web_app


def parse_args():
    parser = argparse.ArgumentParser(
        description="AcousticLens: Real-Time Selective Hearing & Audio Gater"
    )
    parser.add_argument(
        "--mode",
        choices=["enrolled", "diarization", "autosolo", "passthrough"],
        default="enrolled",
        help="Initial operating mode (default: enrolled)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.72,
        help="Mode 1 verification similarity threshold (default: 0.72)",
    )
    parser.add_argument(
        "--input-device",
        type=int,
        default=None,
        help="Audio input device index (run with --list-devices to view)",
    )
    parser.add_argument(
        "--output-device",
        type=int,
        default=None,
        help="Audio output device index (run with --list-devices to view)",
    )
    parser.add_argument(
        "--lookahead",
        type=float,
        default=250.0,
        help="Circular buffer lookahead delay in ms (default: 250.0)",
    )
    parser.add_argument(
        "--web",
        action="store_true",
        help="Enable remote FastAPI / WebSocket web dashboard",
    )
    parser.add_argument(
        "--web-port",
        type=int,
        default=8000,
        help="Port for remote web dashboard (default: 8000)",
    )
    parser.add_argument(
        "--web-host",
        type=str,
        default="0.0.0.0",
        help="Host interface for web dashboard (default: 0.0.0.0)",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run without Textual TUI (ideal when using remote web dashboard)",
    )
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Run in full mock mode (synthetic audio & mock neural weights)",
    )
    parser.add_argument(
        "--mock-audio",
        action="store_true",
        help="Use synthetic audio input stream",
    )
    parser.add_argument(
        "--mock-models",
        action="store_true",
        help="Use mock neural inference models",
    )
    parser.add_argument(
        "--list-devices",
        action="store_true",
        help="List available audio input and output devices and exit",
    )
    return parser.parse_args()


def print_devices():
    inputs, outputs = AudioStreamManager.list_devices()
    print("\n--- Available Input Devices (Microphones) ---")
    for dev in inputs:
        print(f"  [{dev['id']}] {dev['name']} ({dev['channels']} ch)")

    print("\n--- Available Output Devices (Headphones / Virtual Cables) ---")
    for dev in outputs:
        print(f"  [{dev['id']}] {dev['name']} ({dev['channels']} ch)")
    print()


def main():
    args = parse_args()

    if args.list_devices:
        print_devices()
        sys.exit(0)

    # Initialize state
    state = AcousticState()
    state.similarity_threshold = args.threshold

    mode_map = {
        "enrolled": AudioMode.ENROLLED_GATE,
        "diarization": AudioMode.DIARIZATION_MIXER,
        "autosolo": AudioMode.AUTO_SOLO,
        "passthrough": AudioMode.PASSTHROUGH,
    }
    state.set_mode(mode_map.get(args.mode, AudioMode.ENROLLED_GATE))

    # Initialize master engine
    engine = AcousticLensEngine(
        state=state,
        sample_rate=16000,
        block_size=1280,
        lookahead_ms=args.lookahead,
        input_device=args.input_device,
        output_device=args.output_device,
        use_mock_audio=args.mock or args.mock_audio,
        use_mock_models=args.mock or args.mock_models,
    )

    engine.start()

    # Optional background web server
    if args.web:
        web_app = create_web_app(engine=engine, state=state)

        def run_web():
            config = uvicorn.Config(
                app=web_app,
                host=args.web_host,
                port=args.web_port,
                log_level="warning",
            )
            server = uvicorn.Server(config)
            server.run()

        web_thread = threading.Thread(target=run_web, daemon=True)
        web_thread.start()
        print(f"[AcousticLens] Web dashboard running at http://{args.web_host}:{args.web_port}")

    try:
        if args.headless:
            print("[AcousticLens] Running in headless mode. Press Ctrl+C to stop.")
            while True:
                time.sleep(1.0)
        else:
            tui = AcousticLensTUI(engine=engine, state=state)
            tui.run()
    except (KeyboardInterrupt, SystemExit):
        pass
    finally:
        print("\n[AcousticLens] Shutting down audio engine...")
        engine.stop()
        print("[AcousticLens] Done.")


if __name__ == "__main__":
    main()
