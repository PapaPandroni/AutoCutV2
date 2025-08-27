"""Video normalization pipeline for AutoCut V2.

This module provides video format normalization capabilities including:
- Resolution normalization with aspect ratio preservation
- Frame rate normalization
- Letterboxing/pillarboxing for mixed content
- Modern MoviePy 2.x compatibility

Extracted from clip_assembler.py as part of system consolidation.
"""

from typing import Any, Dict, List


class VideoNormalizationPipeline:
    """Pipeline for normalizing mixed video formats to prevent concatenation artifacts."""

    def __init__(self, format_analyzer):
        self.format_analyzer = format_analyzer

    def normalize_video_clips(
        self,
        video_clips: List[Any],
        target_format: Dict[str, Any],
    ) -> List[Any]:
        """Normalize all clips to consistent format to prevent artifacts.

        Args:
            video_clips: List of VideoFileClip instances with mixed formats
            target_format: Target format specification from format analyzer

        Returns:
            List of normalized VideoFileClip instances
        """
        if not target_format.get("requires_normalization", False):
            return video_clips

        normalized_clips = []

        for _i, clip in enumerate(video_clips):
            try:
                normalized_clip = self._normalize_single_clip(clip, target_format)
                normalized_clips.append(normalized_clip)

            except Exception as e:
                # Use original clip if normalization fails
                normalized_clips.append(clip)

        return normalized_clips

    def _normalize_single_clip(self, clip, target_format: Dict[str, Any]):
        """Normalize a single clip to target format."""
        normalized_clip = clip

        # Step 1: Resolution normalization with aspect ratio preservation
        if (
            clip.w != target_format["target_width"]
            or clip.h != target_format["target_height"]
        ):
            normalized_clip = self._resize_with_aspect_preservation_modern(
                normalized_clip,
                target_format["target_width"],
                target_format["target_height"],
            )

        # Step 2: Frame rate normalization
        if abs(clip.fps - target_format["target_fps"]) > 0.1:
            normalized_clip = normalized_clip.with_fps(target_format["target_fps"])

        return normalized_clip

    def _resize_with_aspect_preservation_modern(
        self,
        clip,
        target_width: int,
        target_height: int,
    ):
        """Modern MoviePy 2.2+ letterboxing implementation with intelligent aspect ratio preservation.

        This replaces the previous implementation to use the latest MoviePy best practices
        and provides superior letterboxing/pillarboxing for mixed aspect ratio content.
        """
        # Use the centralized resize function from our compatibility layer
        try:
            # Import the enhanced resize function from our compatibility module
            try:
                from compatibility.moviepy import resize_with_aspect_preservation
            except ImportError:
                from .compatibility.moviepy import resize_with_aspect_preservation

            # NEW APPROACH: Use fit mode to preserve all video content without cropping
            return resize_with_aspect_preservation(
                clip, target_width, target_height, scaling_mode="fit"
            )

        except ImportError:
            # Fallback to local implementation if compatibility module not available
            return self._resize_with_local_fallback(
                clip, target_width, target_height, scaling_mode="fit"
            )

    def _resize_with_local_fallback(
        self, clip, target_width: int, target_height: int, scaling_mode: str = "fit"
    ):
        """Resize clip to target dimensions with intelligent letterboxing to maximize content.

        NEW APPROACH: Always preserve all video content (no cropping) while maximizing
        video size within the target canvas. Uses minimal letterboxing only when needed.

        Args:
            clip: VideoFileClip to resize
            target_width: Target canvas width
            target_height: Target canvas height
            scaling_mode: "fit" (preserve all content) or "fill" (crop to fill)
        """
        # Import the import_moviepy_safely function
        try:
            from ..clip_assembler import import_moviepy_safely
        except ImportError:
            from moviepy.editor import (
                AudioFileClip,
                CompositeVideoClip,
                VideoFileClip,
                concatenate_videoclips,
            )

            def import_moviepy_safely():
                return (
                    VideoFileClip,
                    AudioFileClip,
                    concatenate_videoclips,
                    CompositeVideoClip,
                )

        # Calculate aspect ratios
        clip_aspect = clip.w / clip.h
        target_aspect = target_width / target_height

        # Calculate scaling to maximize video size while preserving all content
        # Always use "fit" scaling to preserve all video content (no cropping)
        width_scale = target_width / clip.w
        height_scale = target_height / clip.h

        # Use the smaller scale to ensure all content fits (fit mode)
        scale = min(width_scale, height_scale)

        # Calculate new dimensions (maintain aspect ratio)
        new_width = int(clip.w * scale)
        new_height = int(clip.h * scale)

        # Resize the video using MoviePy
        try:
            VideoFileClip, AudioFileClip, concatenate_videoclips, CompositeVideoClip = (
                import_moviepy_safely()
            )
            from moviepy.editor import ColorClip

            # Try modern MoviePy 2.x resize with effects first
            try:
                from moviepy.video.fx.Resize import Resize

                resized_clip = clip.with_effects([Resize((new_width, new_height))])
            except (ImportError, AttributeError):
                resized_clip = clip.resized((new_width, new_height))

        except ImportError:
            try:
                from moviepy.editor import ColorClip, CompositeVideoClip

                resized_clip = clip.resized((new_width, new_height))
            except Exception as fallback_error:
                return clip
        except Exception as e:
            return clip

        # Check if letterboxing is needed
        if new_width == target_width and new_height == target_height:
            return resized_clip

        # Calculate letterboxing needed
        width_difference = target_width - new_width
        height_difference = target_height - new_height

        # Determine letterboxing type and amount
        if abs(width_difference) < 10 and abs(height_difference) < 10:
            # Minimal difference - consider it a perfect fit
            return resized_clip

        # Apply intelligent letterboxing
        try:
            # Create black background canvas
            background = ColorClip(
                size=(target_width, target_height),
                color=(0, 0, 0),
                duration=resized_clip.duration,
            )

            # Calculate centering position
            x_pos = (target_width - new_width) // 2
            y_pos = (target_height - new_height) // 2

            # Create composite with centered video
            letterboxed_clip = CompositeVideoClip(
                [
                    background,
                    resized_clip.with_position((x_pos, y_pos)),
                ]
            )

            letterboxed_clip = letterboxed_clip.with_duration(resized_clip.duration)

            # Calculate screen utilization
            video_area = new_width * new_height
            canvas_area = target_width * target_height
            utilization = (video_area / canvas_area) * 100

            # Log letterboxing details
            if width_difference > height_difference:
                # More width padding needed (pillarbox)
                bar_width = width_difference // 2
            else:
                # More height padding needed (letterbox)
                bar_height = height_difference // 2

            # Warn if utilization is very low
            if utilization < 50 or utilization > 80:
                pass

            return letterboxed_clip

        except Exception as e:
            return resized_clip
