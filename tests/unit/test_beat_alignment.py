"""Beat-alignment regression tests for the D1-D3 timeline-anchoring fix.

Root cause (IMPROVEMENTS.md): the renderer just concatenates clips
sequentially, so a cut's real position in the output is the *cumulative sum
of clip durations* -- this must land on the absolute beat grid, not merely be
labeled with a ``beat_position``. These tests call ``match_clips_to_beats``
directly (planner only, no encoding) so they run fast and pinpoint the bug
without needing real audio/video.

- D1: a musical-intro skew (first effective beat != 0) must not shift every
  later cut by that same amount -- the first clip should absorb the intro.
- D2: a clip must be trimmed to *exactly* its target span, no +-slack.
- D3: per-segment target durations must come from the actual beat gap, not
  a global average -- otherwise tempo drift accumulates into timeline drift.
"""

from beat_matching import (
    _calculate_duration_fit,
    _fit_clip_to_duration,
    apply_variety_pattern,
    match_clips_to_beats,
)
from video import VideoChunk

import pytest

FRAME = 1 / 24  # matches tests/synthetic_media.py's FPS
BPM = 120
BEAT_INTERVAL = 60.0 / BPM  # 0.5s
ALLOWED_DURATIONS = [1.0, 2.0, 4.0, 8.0]  # calculate_clip_constraints(120)


def _make_uniform_beats(start_offset: float, count: int = 57) -> list:
    # 57 beats -> 56 beat-slots = exactly 2 full "balanced" pattern cycles
    # (4+4+4+8+4+4=28), so there's no irregular tail-remainder slot to
    # confuse alignment assertions with the (separate, legitimate)
    # allowed_durations tail-rejection behavior.
    return [start_offset + i * BEAT_INTERVAL for i in range(count)]


def _make_drifting_beats(count: int = 57, drift_per_beat: float = 0.006) -> list:
    """Beats with a slowly increasing interval (simulated tempo drift), mean
    interval still ~BEAT_INTERVAL so avg_beat_interval-based math looks
    plausible but diverges from the real per-segment gaps (D3)."""
    beats = [0.0]
    for i in range(count - 1):
        beats.append(beats[-1] + BEAT_INTERVAL + drift_per_beat * (i - count / 2))
    return beats


def _make_chunks(n_videos: int = 4, n_per_video: int = 8) -> list:
    """Ample, non-overlapping, generously long chunks so nothing is rejected
    or dropped for lack of footage -- isolates the alignment bug itself."""
    return [
        VideoChunk(
            video_path=f"/fake/file_{v}.mp4",
            start_time=k * 20.0,
            end_time=k * 20.0 + 18.0,
            score=95 - k,
        )
        for v in range(n_videos)
        for k in range(n_per_video)
    ]


def _expected_boundaries(beats: list, pattern: str) -> list:
    """Ground truth cumulative output-time boundary before each clip: slot 0
    is anchored to absolute 0 (video absorbs the intro, D1); every later
    slot's boundary is the actual beat timestamp reached via real beat gaps
    (D3) -- this is independent of any bug in the implementation under test.
    """
    total_beats = len(beats) - 1
    multipliers = apply_variety_pattern(pattern, total_beats)
    boundaries = [0.0]
    beat_index = 0
    for m in multipliers[:-1]:
        beat_index += m
        boundaries.append(beats[beat_index])
    return boundaries


def _assert_timeline_on_grid(timeline, expected_boundaries) -> None:
    assert len(timeline.clips) == len(expected_boundaries)
    for i, clip in enumerate(timeline.clips):
        delta = abs(clip["cumulative_start"] - expected_boundaries[i])
        assert delta <= FRAME, (
            f"clip {i}: cumulative_start={clip['cumulative_start']:.3f} "
            f"expected~={expected_boundaries[i]:.3f} (delta={delta:.3f}s)"
        )


# ------------------------------- D1 -----------------------------------------
@pytest.mark.parametrize("start_offset", [0.0, 0.23, BEAT_INTERVAL / 2])
def test_intro_skew_does_not_shift_every_cut(start_offset):
    """A song whose first effective beat isn't at t=0 must not shift every
    later cut by that constant offset (D1). offset=0.0 is the control case
    that already worked; the other two reproduce the reported symptom."""
    beats = _make_uniform_beats(start_offset)
    chunks = _make_chunks()

    timeline = match_clips_to_beats(
        chunks, beats, ALLOWED_DURATIONS, pattern="balanced", musical_start_time=0.0
    )
    expected = _expected_boundaries(beats, "balanced")
    _assert_timeline_on_grid(timeline, expected)


# ------------------------------- D3 -----------------------------------------
def test_tempo_drift_does_not_accumulate():
    """Per-segment targets must use the actual beat gap, not the song-wide
    average -- otherwise drift compounds cut after cut (D3)."""
    beats = _make_drifting_beats()
    chunks = _make_chunks()

    timeline = match_clips_to_beats(
        chunks, beats, ALLOWED_DURATIONS, pattern="balanced", musical_start_time=0.0
    )
    expected = _expected_boundaries(beats, "balanced")
    _assert_timeline_on_grid(timeline, expected)


# ------------------------------- D2 -----------------------------------------
def test_fit_clip_to_duration_trims_exact_no_slack():
    """A clip only slightly over target must still trim to exactly target,
    not the old +0.1s 'fits as-is' slack."""
    clip = VideoChunk(video_path="/f.mp4", start_time=0.0, end_time=2.08, score=90)
    _start, _end, duration = _fit_clip_to_duration(clip, target_duration=2.0)
    assert abs(duration - 2.0) < 1e-6, duration


def test_calculate_duration_fit_rejects_short_clips():
    """A clip 0.3s short of target must be unusable -- the old code accepted
    anything up to 0.5s short, letting the shortfall drift the timeline."""
    fit = _calculate_duration_fit(1.7, 2.0, ALLOWED_DURATIONS)
    assert fit < 0, fit
