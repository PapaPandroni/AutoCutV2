"""Generate tiny synthetic video + audio for end-to-end pipeline tests.

No real footage/music required: this produces small, standard H.264 clips (with
controlled scene changes, gentle motion, and fixed spatial texture so quality
scoring is non-trivial) plus a beat-detectable click track. Used by the smoke
tests to exercise the real ``assemble_clips`` pipeline on a clean checkout where
``test_media/`` is absent.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import numpy as np

# Small frames keep encode/analyze fast; still large enough for scene/quality math.
FRAME_W, FRAME_H, FPS = 320, 240, 24


def _make_frame_fn(seed: int, n_scenes: int, seg_len: float, duration: float):
    """Build a moviepy ``frame_function`` producing controlled scenes.

    - A fixed sinusoidal spatial texture -> non-zero sharpness/contrast.
    - A distinct base colour per scene -> a real cut when crossing a boundary
      (large frame diff for ``detect_scenes``), while within-scene frames differ
      only by a slow gradient drift (gentle motion, no false cuts).
    """
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:FRAME_H, 0:FRAME_W]
    texture = (np.sin(xx / 8.0) * np.cos(yy / 8.0) * 40.0).astype(np.float32)
    scene_colors = rng.integers(30, 220, size=(n_scenes, 3)).astype(np.float32)

    def frame(t: float) -> np.ndarray:
        scene = min(int(t / seg_len), n_scenes - 1)
        base = scene_colors[scene].reshape(1, 1, 3)
        shift = (t / duration) * FRAME_W
        grad = (((xx + shift) % FRAME_W) / FRAME_W * 15.0).astype(np.float32)
        img = base + (texture + grad)[..., None]
        return np.clip(img, 0, 255).astype(np.uint8)

    return frame


def generate_videos(
    dest: Path | str,
    count: int = 3,
    duration: float = 9.0,
    seg_len: float = 3.0,
) -> List[str]:
    """Write ``count`` tiny H.264 clips with ~duration/seg_len scenes each."""
    from moviepy import VideoClip

    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    n_scenes = max(1, int(round(duration / seg_len)))
    paths: List[str] = []
    for i in range(count):
        clip = VideoClip(
            frame_function=_make_frame_fn(i + 1, n_scenes, seg_len, duration),
            duration=duration,
        )
        out = dest / f"synthetic_{i}.mp4"
        clip.write_videofile(
            str(out),
            fps=FPS,
            codec="libx264",
            audio=False,
            logger=None,
            preset="ultrafast",
            ffmpeg_params=["-pix_fmt", "yuv420p"],
        )
        clip.close()
        paths.append(str(out))
    return paths


def generate_audio(
    dest: Path | str,
    duration: float = 12.0,
    bpm: int = 120,
    sr: int = 22050,
    start_offset: float = 0.0,
) -> str:
    """Write a short WAV click track with a clearly detectable beat.

    ``start_offset`` places the first click at that many seconds instead of
    0.0 (silence before it), so tests can exercise musical-intro skew (D1).
    """
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    sig = np.zeros_like(t)
    kick_len = int(0.15 * sr)
    env = np.exp(-np.linspace(0, 8, kick_len))
    kick = env * np.sin(2 * np.pi * 60 * np.linspace(0, 0.15, kick_len))
    for beat_t in np.arange(start_offset, duration, 60.0 / bpm):
        i = int(beat_t * sr)
        n = min(kick_len, len(sig) - i)
        sig[i : i + n] += kick[:n]
    sig = (sig / (np.max(np.abs(sig)) + 1e-9)).astype(np.float32)

    out = dest / "synthetic_beat.wav"
    try:
        import soundfile as sf

        sf.write(str(out), sig, sr)
    except Exception:
        from scipy.io import wavfile

        wavfile.write(str(out), sr, (sig * 32767).astype(np.int16))
    return str(out)


def generate_synthetic_media(
    dest: Path | str,
    video_count: int = 3,
) -> Tuple[List[str], str]:
    """Generate videos + audio, returning ``(video_paths, audio_path)``."""
    videos = generate_videos(dest, count=video_count)
    audio = generate_audio(dest)
    return videos, audio
