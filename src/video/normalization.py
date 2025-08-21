"""Video normalization pipeline for AutoCut V2.

This module provides video format normalization capabilities including:
- Resolution normalization with aspect ratio preservation
- Frame rate normalization
- Letterboxing/pillarboxing for mixed content
- Modern MoviePy 2.x compatibility

Extracted from clip_assembler.py as part of system consolidation.
"""

from typing import Dict, Any, List


class VideoNormalizationPipeline:
    """Pipeline for normalizing mixed video formats to prevent concatenation artifacts."""

    def __init__(self, format_analyzer):
        self.format_analyzer = format_analyzer

    def normalize_video_clips(
        self, video_clips: List[Any], target_format: Dict[str, Any]
    ) -> List[Any]:
        """Normalize all clips to consistent format to prevent artifacts.

        Args:
            video_clips: List of VideoFileClip instances with mixed formats
            target_format: Target format specification from format analyzer

        Returns:
            List of normalized VideoFileClip instances
        """
        if not target_format.get("requires_normalization", False):
            print(
                "Format normalization: No normalization required, formats are consistent"
            )
            return video_clips

        print(
            f"Format normalization: Normalizing {len(video_clips)} clips to {target_format['target_width']}x{target_format['target_height']} @ {target_format['target_fps']}fps"
        )

        normalized_clips = []

        for i, clip in enumerate(video_clips):
            try:
                normalized_clip = self._normalize_single_clip(clip, target_format)
                normalized_clips.append(normalized_clip)
                print(
                    f"Normalized clip {i + 1}/{len(video_clips)}: {clip.w}x{clip.h}@{clip.fps}fps -> {normalized_clip.w}x{normalized_clip.h}@{normalized_clip.fps}fps"
                )

            except Exception as e:
                print(f"Warning: Failed to normalize clip {i + 1}: {e}")
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
        self, clip, target_width: int, target_height: int
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
            
            print(f"   Using centralized resize: {clip.w}x{clip.h} → {target_width}x{target_height}")
            print(f"   LEGACY SYSTEM: Using scaling_mode='fill' for maximum screen utilization")
            # CRITICAL FIX: Add scaling_mode="fill" to match new system and maximize screen utilization
            return resize_with_aspect_preservation(clip, target_width, target_height, scaling_mode="fill")
            
        except ImportError:
            # Fallback to local implementation if compatibility module not available
            print(f"   Using local fallback resize: {clip.w}x{clip.h} → {target_width}x{target_height}")
            return self._resize_with_local_fallback(clip, target_width, target_height)
    
    def _resize_with_local_fallback(self, clip, target_width: int, target_height: int):
        """Local fallback resize implementation for when compatibility module is not available."""
        # Import the import_moviepy_safely function (this will need to be handled during import updates)
        try:
            from ..clip_assembler import import_moviepy_safely
        except ImportError:
            # Direct fallback
            from moviepy.editor import VideoFileClip, AudioFileClip, concatenate_videoclips, CompositeVideoClip
            def import_moviepy_safely():
                return VideoFileClip, AudioFileClip, concatenate_videoclips, CompositeVideoClip
        
        # Calculate scaling to fit within target dimensions
        width_scale = target_width / clip.w
        height_scale = target_height / clip.h
        scale = max(width_scale, height_scale)  # FIXED: Use max() for fill scaling instead of min() for fit scaling
        print(f"   LEGACY FALLBACK: Using fill scaling (max) for maximum screen utilization")
        print(f"   Scale factors - width: {width_scale:.3f}, height: {height_scale:.3f}, selected: {scale:.3f} (fill mode)")

        # Calculate new dimensions (maintain aspect ratio)
        new_width = int(clip.w * scale)
        new_height = int(clip.h * scale)
        
        # Use robust MoviePy compatibility with enhanced fallbacks
        try:
            # Use the existing import_moviepy_safely function
            VideoFileClip, AudioFileClip, concatenate_videoclips, CompositeVideoClip = import_moviepy_safely()
            
            # Import ColorClip using same pattern as other parts of codebase
            from moviepy.editor import ColorClip
            
            # Try modern MoviePy 2.x resize with effects first
            try:
                from moviepy.video.fx.Resize import Resize
                resized_clip = clip.with_effects([Resize((new_width, new_height))])
                print(f"   Used modern MoviePy 2.x effects: {clip.w}x{clip.h} → {new_width}x{new_height}")
            except (ImportError, AttributeError):
                # Fallback to legacy resize method
                resized_clip = clip.resized((new_width, new_height))
                print(f"   Used legacy resize method: {clip.w}x{clip.h} → {new_width}x{new_height}")
            
        except ImportError as e:
            try:
                # Direct fallback import (matches other parts of codebase)
                from moviepy.editor import ColorClip, CompositeVideoClip
                resized_clip = clip.resized((new_width, new_height))
                print(f"   Used direct import fallback: {clip.w}x{clip.h} → {new_width}x{new_height}")
            except Exception as fallback_error:
                print(f"Warning: All resize methods failed ({str(fallback_error)}), returning original clip")
                return clip
        except Exception as e:
            print(f"Warning: Could not import MoviePy components ({str(e)}), returning original clip")
            return clip
        
        # Check if letterboxing/pillarboxing is needed
        if new_width == target_width and new_height == target_height:
            # Perfect fit, no letterboxing needed
            print(f"   Perfect fit: no letterboxing needed for {new_width}x{new_height}")
            return resized_clip
            
        # Create black background for letterboxing/pillarboxing
        try:
            background = ColorClip(
                size=(target_width, target_height),
                color=(0, 0, 0),  # Black letterbox bars
                duration=resized_clip.duration
            )
            
            # Calculate centering position for perfect centering
            x_pos = (target_width - new_width) // 2
            y_pos = (target_height - new_height) // 2
            
            # Create composite with centered resized clip
            letterboxed_clip = CompositeVideoClip([
                background,
                resized_clip.with_position((x_pos, y_pos))
            ])
            
            # Ensure the composite has the correct duration and properties
            letterboxed_clip = letterboxed_clip.with_duration(resized_clip.duration)
            
            # Log the letterboxing operation for debugging
            if new_width < target_width:
                letterbox_type = "pillarbox" if new_height == target_height else "letterbox+pillarbox"
                bar_width = (target_width - new_width) // 2
                print(f"   Applied {letterbox_type}: {bar_width}px bars on sides")
            else:
                bar_height = (target_height - new_height) // 2  
                print(f"   Applied letterbox: {bar_height}px bars on top/bottom")
                
            return letterboxed_clip
            
        except Exception as e:
            # Fallback: return resized clip without letterboxing if composition fails
            print(f"Warning: Letterboxing failed ({str(e)}), returning resized clip without black bars")
            print(f"   Resized to: {new_width}x{new_height} (target: {target_width}x{target_height})")
            return resized_clip