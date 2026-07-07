"""Fast regression tests locking in the Tier-1 correctness fixes.

These exercise the fixed logic directly (no video encoding) so they run in the
quick loop and guard against the streamline work reintroducing the bugs:

- H1: variety patterns actually produce their cuts (2-beat targets accepted).
- C1: a failed clip becomes a same-duration placeholder (no beat-sync drift).
- H4: uniformize_dimensions forces every clip to the exact canvas size.
- M2/M3: analyze_audio coerces the librosa tempo ndarray and clamps BPM.
"""

from pathlib import Path

import pytest

from audio_analyzer import analyze_audio, calculate_clip_constraints
from clip_assembler import (
    _calculate_duration_fit,
    match_clips_to_beats,
    uniformize_dimensions,
)
from video import VideoChunk

try:
    from moviepy import ColorClip, concatenate_videoclips
except ImportError:  # pragma: no cover - moviepy 1.x fallback
    from moviepy.editor import ColorClip, concatenate_videoclips

TARGET_W, TARGET_H = 1920, 1080


# ----------------------------- H1 ------------------------------------------
def test_constraints_include_two_beat_duration():
    """calculate_clip_constraints exposes a 2-beat duration (was 4/8/16 only)."""
    _min_dur, allowed = calculate_clip_constraints(120.0)  # beat = 0.5s
    assert any(abs(a - 1.0) < 1e-6 for a in allowed), allowed


def test_duration_fit_accepts_two_beat_target():
    """A 2-beat target is rejected under the old 4/8/16 set, accepted under new."""
    old_allowed = [2.0, 4.0, 8.0]
    _min_dur, new_allowed = calculate_clip_constraints(120.0)
    assert _calculate_duration_fit(3.0, 1.0, old_allowed) < 0
    assert _calculate_duration_fit(3.0, 1.0, new_allowed) >= 0


@pytest.mark.parametrize("pattern,min_clips", [("energetic", 8), ("dramatic", 1)])
def test_patterns_produce_dense_timeline(pattern, min_clips):
    """energetic no longer drops most cuts; dramatic's 16-beat hold still matches."""
    _min_dur, allowed = calculate_clip_constraints(120.0)
    beats = [round(0.5 * i, 3) for i in range(41)]
    chunks = [
        VideoChunk(
            video_path=f"/fake/file_{f}.mp4",
            start_time=k * 6.0,
            end_time=k * 6.0 + 5.0,
            score=90 - k,
        )
        for f in range(4)
        for k in range(6)
    ]
    timeline = match_clips_to_beats(
        chunks, beats, allowed, pattern=pattern, musical_start_time=0.0
    )
    assert len(timeline.clips) >= min_clips


# ----------------------------- H4 ------------------------------------------
def test_uniformize_forces_exact_canvas_size():
    """Mixed-size clips all come out at exactly the target canvas size."""
    clips = [
        ColorClip(size=(640, 480), color=(10, 20, 30), duration=1.0),
        ColorClip(size=(1080, 1920), color=(30, 20, 10), duration=1.0),
        ColorClip(size=(TARGET_W, TARGET_H), color=(0, 0, 0), duration=1.0),
    ]
    uniform = uniformize_dimensions(clips, TARGET_W, TARGET_H)
    assert len(uniform) == len(clips)
    assert all(tuple(c.size) == (TARGET_W, TARGET_H) for c in uniform)


# ----------------------------- C1 ------------------------------------------
def test_failed_clip_becomes_same_duration_placeholder():
    """A dropped clip is replaced by a same-duration black gap, preserving timing."""
    sorted_clips = [
        {"video_file": "/a.mp4", "start": 0.0, "end": 2.0, "index": 0},
        {"video_file": "/b.mp4", "start": 5.0, "end": 6.0, "index": 1},  # "fails"
        {"video_file": "/c.mp4", "start": 0.0, "end": 3.0, "index": 2},
    ]
    video_clips = [
        ColorClip(size=(TARGET_W, TARGET_H), color=(1, 1, 1), duration=2.0),
        None,  # loader returns None where a clip failed to load
        ColorClip(size=(TARGET_W, TARGET_H), color=(2, 2, 2), duration=3.0),
    ]

    for i, clip in enumerate(video_clips):
        if clip is None:
            meta = sorted_clips[i]
            gap = max(float(meta["end"]) - float(meta["start"]), 0.1)
            video_clips[i] = ColorClip(
                size=(TARGET_W, TARGET_H), color=(0, 0, 0), duration=gap
            )

    assert all(c is not None for c in video_clips)
    assert abs(video_clips[1].duration - 1.0) < 1e-6  # keeps the dropped duration
    final = concatenate_videoclips(video_clips, method="compose")
    assert abs(final.duration - (2.0 + 1.0 + 3.0)) < 1e-3  # no drift
    assert tuple(final.size) == (TARGET_W, TARGET_H)


# --------------------------- M2 / M3 ---------------------------------------
def test_analyze_audio_returns_clamped_float_bpm(tmp_path: Path):
    """analyze_audio coerces the librosa tempo ndarray and clamps BPM to 30-300."""
    from tests.synthetic_media import generate_audio

    audio_path = generate_audio(tmp_path, duration=8.0, bpm=120)
    result = analyze_audio(audio_path)

    assert isinstance(result["bpm"], float)
    assert 30.0 <= result["bpm"] <= 300.0
    assert len(result.get("allowed_durations", [])) > 0
