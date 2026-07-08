"""Timeline rendering system for AutoCut V2.

This module provides timeline-based video rendering capabilities including:
- Timeline management and validation
- Clip orchestration and sequencing
- Beat synchronization timing
- Rendering coordination and delegation

Extracted from clip_assembler.py as part of system consolidation.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional


class ClipTimeline:
    """Represents the timeline of clips matched to beats."""

    def __init__(self):
        self.clips: List[Dict[str, Any]] = []
        self._cumulative_start: float = 0.0

    def add_clip(
        self,
        video_file: str,
        start: float,
        end: float,
        beat_position: float,
        score: float,
    ):
        """Add a clip to the timeline."""
        duration = end - start
        cumulative_start = self._cumulative_start
        self._cumulative_start += duration
        self.clips.append(
            {
                "video_file": video_file,
                "start": start,
                "end": end,
                "beat_position": beat_position,
                "score": score,
                "duration": duration,
                "cumulative_start": cumulative_start,
                "beat_delta": cumulative_start - beat_position,
            },
        )

    def export_json(self, file_path: str):
        """Export timeline as JSON for debugging."""
        with Path(file_path).open("w") as f:
            json.dump(self.clips, f, indent=2)

    def get_alignment_report(self) -> Dict[str, float]:
        """Summarize how well cumulative render-time positions track the beat grid.

        ``beat_delta`` (cumulative_start - beat_position) is 0 when a clip's
        position in the concatenated output lands exactly on its intended beat.
        """
        if not self.clips:
            return {"max_abs_delta": 0.0, "mean_abs_delta": 0.0}

        deltas = [abs(clip["beat_delta"]) for clip in self.clips]
        return {
            "max_abs_delta": max(deltas),
            "mean_abs_delta": sum(deltas) / len(deltas),
        }

    def get_total_duration(self) -> float:
        """Get total duration of all clips."""
        return sum(clip["duration"] for clip in self.clips)

    def get_clips_sorted_by_beat(self) -> List[Dict[str, Any]]:
        """Get clips sorted by their beat position."""
        return sorted(self.clips, key=lambda x: x["beat_position"])

    def get_unique_video_files(self) -> List[str]:
        """Get list of unique video files used in timeline."""
        unique_files = list({clip["video_file"] for clip in self.clips})
        return sorted(unique_files)  # Sort for consistent ordering

    def get_summary_stats(self) -> Dict[str, Any]:
        """Get summary statistics about the timeline."""
        if not self.clips:
            return {
                "total_clips": 0,
                "total_duration": 0.0,
                "avg_score": 0.0,
                "unique_videos": 0,
                "score_range": (0.0, 0.0),
            }

        scores = [clip["score"] for clip in self.clips]
        unique_videos = len({clip["video_file"] for clip in self.clips})

        return {
            "total_clips": len(self.clips),
            "total_duration": self.get_total_duration(),
            "avg_score": sum(scores) / len(scores),
            "unique_videos": unique_videos,
            "score_range": (min(scores), max(scores)),
            "duration_range": (
                min(clip["duration"] for clip in self.clips),
                max(clip["duration"] for clip in self.clips),
            ),
        }

    def validate_timeline(
        self, song_duration: Optional[float] = None
    ) -> Dict[str, Any]:
        """Validate timeline for common issues.

        Args:
            song_duration: Total duration of the song for coverage check

        Returns:
            Dictionary with validation results
        """
        issues = []
        warnings = []

        if not self.clips:
            issues.append("Timeline is empty")
            return {"valid": False, "issues": issues, "warnings": warnings}

        # Sort clips by beat position for analysis
        sorted_clips = self.get_clips_sorted_by_beat()

        # Check for very short clips
        short_clips = [clip for clip in self.clips if clip["duration"] < 0.5]
        if short_clips:
            warnings.append(f"{len(short_clips)} clips are very short (<0.5s)")

        # Check for very long clips
        long_clips = [clip for clip in self.clips if clip["duration"] > 8.0]
        if long_clips:
            warnings.append(f"{len(long_clips)} clips are very long (>8s)")

        # Check score distribution
        scores = [clip["score"] for clip in self.clips]
        avg_score = sum(scores) / len(scores)
        if avg_score < 50:
            warnings.append(f"Average clip quality is low: {avg_score:.1f}")

        # Check video variety
        unique_videos = len({clip["video_file"] for clip in self.clips})
        if len(self.clips) > 5 and unique_videos == 1:
            warnings.append("All clips are from the same video - low variety")

        # Check timeline coverage if song duration provided
        if song_duration:
            timeline_span = (
                sorted_clips[-1]["beat_position"] - sorted_clips[0]["beat_position"]
            )
            coverage = timeline_span / song_duration
            if coverage < 0.8:
                warnings.append(f"Timeline covers only {coverage * 100:.1f}% of song")

        return {
            "valid": len(issues) == 0,
            "issues": issues,
            "warnings": warnings,
            "stats": self.get_summary_stats(),
        }
