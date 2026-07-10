#!/usr/bin/env python3
"""
AutoCut GUI entry point.

Launches the Tkinter interface. This is also the entry script for the
PyInstaller build (see autocut.spec), so it handles the frozen-app quirks:
windowed apps have no stdout/stderr, and logs need to go to a file the
user can find.
"""

import logging
import os
import sys
from pathlib import Path

# Add src directory to path for imports (same convention as autocut.py)
sys.path.insert(0, str(Path(__file__).parent / "src"))


def _setup_runtime() -> None:
    """Configure logging and stdio for both dev and frozen (.app) runs."""
    if getattr(sys, "frozen", False):
        # Windowed PyInstaller apps run with stdout/stderr set to None;
        # anything that prints (moviepy progress bars) would crash.
        devnull = open(os.devnull, "w")  # noqa: SIM115
        if sys.stdout is None:
            sys.stdout = devnull
        if sys.stderr is None:
            sys.stderr = devnull

        if sys.platform == "darwin":
            log_dir = Path.home() / "Library" / "Logs" / "AutoCut"
        else:
            log_dir = Path.home() / ".autocut" / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        handlers = [logging.FileHandler(log_dir / "autocut.log")]

        # Point MoviePy and imageio-ffmpeg at the bundled ffmpeg. Must be
        # set before moviepy is imported (it reads FFMPEG_BINARY at import).
        from ffmpeg_paths import get_ffmpeg_exe

        os.environ["FFMPEG_BINARY"] = get_ffmpeg_exe()
        os.environ["IMAGEIO_FFMPEG_EXE"] = get_ffmpeg_exe()
    else:
        handlers = None

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=handlers,
    )


def _selftest(args: list) -> None:
    """Headless pipeline run for verifying a frozen build from the terminal.

    Usage: AutoCut --selftest <video>... --audio <music> --output <out.mp4>
    """
    audio = args[args.index("--audio") + 1]
    output = args[args.index("--output") + 1]
    videos = args[: args.index("--audio")]

    from api import AutoCutAPI
    from ffmpeg_paths import get_ffmpeg_exe, get_ffprobe_exe

    print(f"frozen : {getattr(sys, 'frozen', False)}")
    print(f"ffmpeg : {get_ffmpeg_exe()}")
    print(f"ffprobe: {get_ffprobe_exe()}")

    def on_progress(step: str, progress: float) -> None:
        print(f"{progress:5.0%}  {step}", flush=True)

    result = AutoCutAPI().process_videos(
        video_files=videos,
        audio_file=audio,
        output_path=output,
        progress_callback=on_progress,
    )
    print(f"OK: {result}")


def main() -> None:
    _setup_runtime()

    if "--selftest" in sys.argv:
        _selftest(sys.argv[sys.argv.index("--selftest") + 1 :])
        return

    from gui import main as gui_main

    gui_main()


if __name__ == "__main__":
    main()
