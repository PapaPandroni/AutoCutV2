"""Unit tests for src/audio_analyzer.py's BPM-driven duration constraints.

Moved from the dead ``test_audio_analyzer()`` self-test that lived in
src/audio_analyzer.py (never collected by pytest, no real asserts) as part of
the IMPROVEMENTS.md Stage-3 cleanup.
"""

import pytest

from audio_analyzer import calculate_clip_constraints


@pytest.mark.parametrize("bpm", [60, 90, 120, 140])
def test_calculate_clip_constraints_accepts_valid_bpm(bpm):
    min_duration, allowed_durations = calculate_clip_constraints(bpm)
    assert min_duration > 0
    assert allowed_durations
    assert all(d > 0 for d in allowed_durations)


@pytest.mark.parametrize("bpm", [25, 350])
def test_calculate_clip_constraints_rejects_out_of_range_bpm(bpm):
    with pytest.raises(ValueError):
        calculate_clip_constraints(bpm)
