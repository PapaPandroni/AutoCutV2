"""Video format analysis for AutoCut V2.

This module provides video format analysis capabilities including:
- Format detection and categorization
- Canvas size optimization
- Compatibility issue detection
- Quality preservation strategies

Extracted from clip_assembler.py as part of system consolidation.
"""

from typing import Dict, Any, List


class VideoFormatAnalyzer:
    """Analyzes video formats to detect inconsistencies that cause visual artifacts."""

    def __init__(self):
        self.format_cache = {}

    def analyze_video_format(self, video_clip) -> Dict[str, Any]:
        """Extract comprehensive format information from a video clip.

        Args:
            video_clip: MoviePy VideoFileClip instance

        Returns:
            Dictionary with format details
        """
        try:
            format_info = {
                "width": video_clip.w,
                "height": video_clip.h,
                "fps": video_clip.fps,
                "duration": video_clip.duration,
                "aspect_ratio": video_clip.w / video_clip.h
                if video_clip.h > 0
                else 1.0,
                "resolution_category": self._categorize_resolution(
                    video_clip.w, video_clip.h
                ),
                "fps_category": self._categorize_fps(video_clip.fps),
            }

            # Add codec information if available
            if hasattr(video_clip, "filename"):
                format_info["filename"] = video_clip.filename

            return format_info

        except Exception as e:
            print(f"Warning: Could not analyze video format: {e}")
            # Use smart fallback instead of hard-coded 1920x1080
            return self._get_smart_fallback_format()

    def _categorize_resolution(self, width: int, height: int) -> str:
        """Categorize resolution into standard formats."""
        if width >= 3840 and height >= 2160:
            return "4K"
        elif width >= 2560 and height >= 1440:
            return "1440p"
        elif width >= 1920 and height >= 1080:
            return "1080p"
        elif width >= 1280 and height >= 720:
            return "720p"
        else:
            return "SD"

    def _categorize_fps(self, fps: float) -> str:
        """Categorize frame rate into standard categories."""
        if fps >= 59.0:
            return "60fps"
        elif fps >= 29.0:
            return "30fps"
        elif fps >= 24.0:
            return "25fps"
        else:
            return "24fps"

    def _get_smart_fallback_format(self) -> Dict[str, Any]:
        """Get smart fallback format based on context instead of hard-coded 1920x1080.
        
        If we have a target format context, use that. Otherwise use safe defaults.
        This prevents always defaulting to landscape orientation.
        """
        # Try to get context from instance if available
        if hasattr(self, '_target_format') and self._target_format:
            target = self._target_format
            return {
                "width": target.get('target_width', 1920),
                "height": target.get('target_height', 1080),
                "fps": target.get('target_fps', 24.0),
                "duration": 0.0,
                "aspect_ratio": target.get('target_width', 1920) / target.get('target_height', 1080),
                "resolution_category": "1080p",
                "fps_category": "24fps",
            }
        
        # Safe default - still landscape but with context awareness
        return {
            "width": 1920,
            "height": 1080, 
            "fps": 24.0,
            "duration": 0.0,
            "aspect_ratio": 16 / 9,
            "resolution_category": "1080p", 
            "fps_category": "24fps",
        }

    def determine_optimal_canvas(self, video_clips: List[Any]) -> Dict[str, Any]:
        """Determine optimal 4:3 canvas size that preserves quality for all content.
        
        UPDATED APPROACH: Always outputs 4:3 aspect ratio (width:height = 4:3) while 
        preserving maximum quality. All content gets appropriately letterboxed/pillarboxed 
        within the 4:3 frame.
        
        Key features:
        - Always 4:3 output (1.333 aspect ratio)
        - Preserves 4K quality (no forced downscaling)
        - Landscape videos get letterboxed (black bars top/bottom)
        - Portrait/square videos get pillarboxed (black bars left/right)
        - Maximizes space usage without distortion
        
        Args:
            video_clips: List of VideoFileClip instances

        Returns:
            Target 4:3 canvas specification optimized for quality preservation
        """
        if not video_clips:
            return {
                "target_width": 1440,
                "target_height": 1080,
                "target_fps": 24.0,
                "target_aspect_ratio": 4 / 3,
                "requires_normalization": False,
                "canvas_type": "default_4_3",
            }

        # Analyze all content to find optimal 4:3 canvas
        formats = []
        max_dimension = 0  # Track the largest dimension for quality preservation
        fps_counts = {}
        content_types = []

        for clip in video_clips:
            format_info = self.analyze_video_format(clip)
            formats.append(format_info)
            
            width, height = format_info['width'], format_info['height']
            
            # Track the largest dimension across ALL clips for quality preservation
            max_dimension = max(max_dimension, width, height)
            
            # Track content type for logging
            aspect_ratio = width / height
            if aspect_ratio > 1.3:  # Landscape
                content_types.append('landscape')
            elif aspect_ratio < 0.8:  # Portrait  
                content_types.append('portrait')
            else:  # Square-ish
                content_types.append('square')
            
            # FPS analysis
            fps_key = format_info["fps_category"]
            fps_counts[fps_key] = fps_counts.get(fps_key, 0) + 1

        # Determine target FPS (most common)
        dominant_fps_category = max(fps_counts, key=fps_counts.get)
        target_fps = self._fps_category_to_value(dominant_fps_category)

        # INTELLIGENT CANVAS SELECTION - content-aware aspect ratio optimization
        # CRITICAL FIX: Pure landscape content should use 16:9 canvas to eliminate letterboxing
        unique_content_types = set(content_types)
        
        if len(unique_content_types) == 1 and 'landscape' in unique_content_types:
            # Pure landscape content - use 16:9 canvas for maximum screen utilization
            target_aspect_ratio = 16.0 / 9.0  # 1.777
            
            print(f"🎯 CANVAS OPTIMIZATION: Pure landscape content detected")
            print(f"   Using 16:9 canvas to eliminate letterboxing for landscape videos")
            
            # Determine optimal dimensions for 16:9 while preserving quality
            if max_dimension >= 3840:  # 4K content
                # For 4K, use 3840x2160 (16:9 aspect ratio, high quality)
                target_width = 3840
                target_height = 2160
            elif max_dimension >= 1920:  # HD content
                # For HD, use 1920x1080 (16:9 aspect ratio, good quality)
                target_width = 1920 
                target_height = 1080
            else:  # SD content
                # For SD, use 1280x720 (16:9 aspect ratio, standard quality)
                target_width = 1280
                target_height = 720
        else:
            # Mixed content or portrait content - use 4:3 canvas with appropriate letterboxing
            target_aspect_ratio = 4.0 / 3.0  # 1.333
            
            print(f"🎯 CANVAS OPTIMIZATION: Mixed/portrait content detected")
            print(f"   Using 4:3 canvas with letterboxing for optimal mixed content handling")
            
            # Determine optimal dimensions for 4:3 while preserving quality
            if max_dimension >= 3840:  # 4K content
                # For 4K, use 2880x2160 (4:3 aspect ratio, high quality)
                target_width = 2880
                target_height = 2160
            elif max_dimension >= 1920:  # HD content
                # For HD, use 1440x1080 (4:3 aspect ratio, good quality)
                target_width = 1440 
                target_height = 1080
            else:  # SD content
                # For SD, use 960x720 (4:3 aspect ratio, standard quality)
                target_width = 960
                target_height = 720
        
        # Verify aspect ratio is exactly as intended
        calculated_aspect = target_width / target_height
        if abs(calculated_aspect - target_aspect_ratio) > 0.01:
            # Adjust width to ensure perfect aspect ratio
            target_width = int(target_height * target_aspect_ratio)

        # Determine canvas characteristics for logging based on intelligent selection
        # Canvas type based on content mix and selected aspect ratio
        if len(unique_content_types) == 1:
            if 'landscape' in unique_content_types:
                if target_aspect_ratio > 1.6:  # 16:9 canvas
                    canvas_type = "16_9_landscape_optimized"
                    description = "Pure landscape content - 16:9 canvas with minimal letterboxing"
                else:  # 4:3 canvas (fallback case)
                    canvas_type = "4_3_landscape_optimized"
                    description = "Landscape content - 4:3 canvas with letterboxing"
            elif 'portrait' in unique_content_types:
                canvas_type = "4_3_portrait_optimized" 
                description = "Portrait content - 4:3 canvas with pillarboxing"
            else:
                canvas_type = "4_3_square_optimized"
                description = "Square content - 4:3 canvas with minimal bars"
        else:
            canvas_type = "4_3_mixed_content"
            description = "Mixed content types - 4:3 canvas with adaptive boxing"

        # Check if normalization is needed 
        resolution_diversity = len(set(f"{fmt['width']}x{fmt['height']}" for fmt in formats))
        requires_normalization = (
            resolution_diversity > 1 or 
            len(set(fps_counts.keys())) > 1 or
            len(unique_content_types) > 1 or
            True  # Always normalize for 4:3 output since most content isn't 4:3
        )

        # Quality preservation logging
        quality_info = f"4K" if max_dimension >= 3840 else f"HD" if max_dimension >= 1920 else "SD"
        print(f"Canvas Analysis: {description}")
        print(f"   - Content types: {', '.join(unique_content_types)}")
        print(f"   - Quality level: {quality_info} (max dimension: {max_dimension}px)")
        print(f"Canvas Decision: {target_width}x{target_height} @ {target_fps}fps (4:3 aspect ratio)")
        print(f"Canvas Type: {canvas_type}")
        print(f"Quality Preservation: ✅ Optimal quality for 4:3 output")

        return {
            "target_width": target_width,
            "target_height": target_height,
            "target_fps": target_fps,
            "target_aspect_ratio": target_aspect_ratio,
            "requires_normalization": requires_normalization,
            "canvas_type": canvas_type,
            "quality_level": quality_info,
            "max_dimension_preserved": max_dimension,
            "content_analysis": {
                "content_types": content_types,
                "unique_content_types": list(unique_content_types),
                "mixed_content": len(unique_content_types) > 1,
            },
            "format_diversity": {
                "resolutions": resolution_diversity,
                "frame_rates": len(fps_counts),
            },
        }

    def _fps_category_to_value(self, fps_category: str) -> float:
        """Convert FPS category back to numeric value."""
        mapping = {"60fps": 60.0, "30fps": 30.0, "25fps": 25.0, "24fps": 24.0}
        return mapping.get(fps_category, 24.0)

    def detect_format_compatibility_issues(
        self, video_clips: List[Any]
    ) -> List[Dict[str, Any]]:
        """Identify specific compatibility issues between video clips.

        Returns:
            List of issue descriptions with recommended fixes
        """
        if len(video_clips) < 2:
            return []

        issues = []
        formats = [self.analyze_video_format(clip) for clip in video_clips]

        # Check resolution variations
        resolutions = set((fmt["width"], fmt["height"]) for fmt in formats)
        if len(resolutions) > 1:
            issues.append(
                {
                    "type": "resolution_mismatch",
                    "description": f"Mixed resolutions detected: {resolutions}",
                    "severity": "high",
                    "artifacts": [
                        "flashing up/down",
                        "scaling artifacts",
                        "centering issues",
                    ],
                    "fix": "resolution_normalization",
                }
            )

        # Check frame rate variations
        fps_values = set(fmt["fps"] for fmt in formats)
        if len(fps_values) > 1:
            issues.append(
                {
                    "type": "framerate_mismatch",
                    "description": f"Mixed frame rates detected: {fps_values}",
                    "severity": "high",
                    "artifacts": [
                        "VHS-like wrap around",
                        "temporal stuttering",
                        "motion artifacts",
                    ],
                    "fix": "framerate_normalization",
                }
            )

        # Check aspect ratio variations
        aspect_ratios = set(round(fmt["aspect_ratio"], 3) for fmt in formats)
        if len(aspect_ratios) > 1:
            issues.append(
                {
                    "type": "aspect_ratio_mismatch",
                    "description": f"Mixed aspect ratios detected: {aspect_ratios}",
                    "severity": "medium",
                    "artifacts": ["letterboxing inconsistency", "stretching artifacts"],
                    "fix": "aspect_ratio_normalization",
                }
            )

        return issues