"""End-to-end smoke test for the live AutoCut pipeline.

Runs the real ``assemble_clips`` entrypoint on tiny synthetic media (no real
footage/music needed) and asserts a valid output video is produced. This is the
regression guardrail for the codebase-streamline work: it exercises audio
analysis, per-file video analysis, canvas sizing, beat matching, robust loading,
dimension uniformization, concatenation, audio attach, and encoding.

The default ``balanced`` case is the fast guard to run after every deletion step.
The other patterns are marked ``slow`` (each full run re-analyzes + encodes 1080p,
~50s) so ``-m "not slow"`` keeps the quick loop to a single pipeline run.
"""

from pathlib import Path

import pytest

from moviepy import AudioFileClip, VideoFileClip

# assemble_clips is imported from the src-on-path context (conftest adds src/),
# matching how autocut.py runs it. ``import src.clip_assembler`` is unsupported.
from clip_assembler import assemble_clips


def _assert_valid_output(output_path: str, audio_path: str) -> float:
    """Assert the rendered file exists, is non-empty, and has a sane duration."""
    out = Path(output_path)
    assert out.exists(), f"no output produced at {output_path}"
    assert out.stat().st_size > 0, "output file is empty"

    with VideoFileClip(output_path) as vid:
        duration = vid.duration
    with AudioFileClip(audio_path) as aud:
        audio_duration = aud.duration

    # Output = concatenation of the SELECTED beat-matched clips, so it must be a
    # positive length and never exceed the music (video is trimmed to audio).
    assert duration > 1.0, f"output implausibly short: {duration:.2f}s"
    assert duration <= audio_duration + 0.5, (
        f"output {duration:.2f}s exceeds audio {audio_duration:.2f}s"
    )
    return duration


def test_assemble_clips_end_to_end(synthetic_media, tmp_path):
    """The full pipeline produces a valid video with the default pattern."""
    videos, audio = synthetic_media
    output = str(tmp_path / "smoke_balanced.mp4")

    result = assemble_clips(videos, audio, output, pattern="balanced")

    assert result == output
    _assert_valid_output(result, audio)


@pytest.mark.slow
@pytest.mark.parametrize("pattern", ["energetic", "dramatic"])
def test_assemble_clips_all_patterns(synthetic_media, tmp_path, pattern):
    """Every variety pattern yields output (locks in the H1 energetic/dramatic fix)."""
    videos, audio = synthetic_media
    output = str(tmp_path / f"smoke_{pattern}.mp4")

    result = assemble_clips(videos, audio, output, pattern=pattern)

    _assert_valid_output(result, audio)
