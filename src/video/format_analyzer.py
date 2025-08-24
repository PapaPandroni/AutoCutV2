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
        """Determine intelligent canvas size that maximizes video content with minimal black bars.
        
        NEW APPROACH: Intelligently selects canvas based on content analysis:
        - Single aspect ratio: Optimize canvas for that aspect ratio (minimal bars)
        - Mixed aspect ratios: Use smart default (landscape 16:9) to avoid tiny videos
        
        Key features:
        - Analyzes aspect ratio distribution of input content
        - Optimizes for single aspect ratio scenarios  
        - Uses intelligent defaults for mixed content
        - Maximizes video scaling without cropping or distortion
        - Eliminates hardcoded 4:3 canvas constraint
        
        Args:
            video_clips: List of VideoFileClip instances

        Returns:
            Intelligent canvas specification optimized for minimal letterboxing
        """
        if not video_clips:
            return {
                "target_width": 1920,
                "target_height": 1080,
                "target_fps": 24.0,
                "target_aspect_ratio": 16 / 9,
                "requires_normalization": False,
                "canvas_type": "default_16_9",
            }

        # Analyze all content for intelligent canvas selection
        formats = []
        max_dimension = 0
        fps_counts = {}
        aspect_ratios = []
        resolutions = []

        for clip in video_clips:
            format_info = self.analyze_video_format(clip)
            formats.append(format_info)
            
            width, height = format_info['width'], format_info['height']
            aspect_ratio = width / height
            
            # Collect data for analysis
            max_dimension = max(max_dimension, width, height)
            aspect_ratios.append(aspect_ratio)
            resolutions.append((width, height))
            
            # FPS analysis
            fps_key = format_info["fps_category"]
            fps_counts[fps_key] = fps_counts.get(fps_key, 0) + 1

        # Determine target FPS (most common)
        dominant_fps_category = max(fps_counts, key=fps_counts.get)
        target_fps = self._fps_category_to_value(dominant_fps_category)

        # INTELLIGENT ASPECT RATIO ANALYSIS
        # Classify videos by aspect ratio with tolerance
        landscape_count = 0  # > 1.3 (wider than 4:3)
        portrait_count = 0   # < 0.8 (taller than 4:3)
        square_count = 0     # 0.8 to 1.3 (square-ish)
        
        for ar in aspect_ratios:
            if ar > 1.3:
                landscape_count += 1
            elif ar < 0.8:
                portrait_count += 1
            else:
                square_count += 1
        
        total_videos = len(aspect_ratios)
        landscape_ratio = landscape_count / total_videos
        portrait_ratio = portrait_count / total_videos
        square_ratio = square_count / total_videos
        
        print(f"Aspect Ratio Analysis:")
        print(f"   - Landscape (>1.3): {landscape_count}/{total_videos} ({landscape_ratio:.1%})")
        print(f"   - Portrait (<0.8): {portrait_count}/{total_videos} ({portrait_ratio:.1%})")
        print(f"   - Square (0.8-1.3): {square_count}/{total_videos} ({square_ratio:.1%})")

        # INTELLIGENT CANVAS SELECTION LOGIC
        canvas_decision_threshold = 0.8  # 80% of videos must be same orientation for optimization
        
        if landscape_ratio >= canvas_decision_threshold:
            # Predominantly landscape content - optimize for landscape
            target_aspect_ratio = 16.0 / 9.0  # Standard 16:9
            canvas_type = "landscape_optimized"
            description = "Predominantly landscape content - 16:9 canvas for minimal letterboxing"
            
            # Find optimal landscape dimensions based on content
            landscape_widths = [w for w, h in resolutions if w/h > 1.3]
            if landscape_widths:
                max_landscape_width = max(landscape_widths)
                if max_landscape_width >= 3840:  # 4K
                    target_width, target_height = 3840, 2160
                elif max_landscape_width >= 1920:  # HD
                    target_width, target_height = 1920, 1080
                else:  # SD
                    target_width, target_height = 1280, 720
            else:
                target_width, target_height = 1920, 1080  # Safe default
                
        elif portrait_ratio >= canvas_decision_threshold:
            # Predominantly portrait content - optimize for portrait
            target_aspect_ratio = 9.0 / 16.0  # Portrait 9:16
            canvas_type = "portrait_optimized"
            description = "Predominantly portrait content - 9:16 canvas for minimal letterboxing"
            
            # Find optimal portrait dimensions based on content
            portrait_heights = [h for w, h in resolutions if w/h < 0.8]
            if portrait_heights:
                max_portrait_height = max(portrait_heights)
                if max_portrait_height >= 3840:  # 4K portrait
                    target_width, target_height = 2160, 3840
                elif max_portrait_height >= 1920:  # HD portrait
                    target_width, target_height = 1080, 1920
                else:  # SD portrait
                    target_width, target_height = 720, 1280
            else:
                target_width, target_height = 1080, 1920  # Safe default
                
        elif square_ratio >= canvas_decision_threshold:
            # Predominantly square content - optimize for square
            target_aspect_ratio = 1.0  # Square 1:1
            canvas_type = "square_optimized"
            description = "Predominantly square content - 1:1 canvas for minimal letterboxing"
            
            # Find optimal square dimensions based on content
            if max_dimension >= 3840:
                target_width, target_height = 2160, 2160  # Square 4K
            elif max_dimension >= 1920:
                target_width, target_height = 1080, 1080  # Square HD
            else:
                target_width, target_height = 720, 720    # Square SD
                
        else:
            # Mixed aspect ratios - use intelligent default (landscape)
            # Landscape is chosen because:
            # 1. Most playback devices are landscape oriented
            # 2. Avoids making landscape videos tiny in portrait canvas
            # 3. Portrait videos letterbox better in landscape than vice versa
            target_aspect_ratio = 16.0 / 9.0
            canvas_type = "mixed_content_default"
            description = "Mixed aspect ratios - 16:9 default canvas to avoid tiny videos"
            
            # Use high quality landscape dimensions
            if max_dimension >= 3840:
                target_width, target_height = 3840, 2160
            elif max_dimension >= 1920:
                target_width, target_height = 1920, 1080
            else:
                target_width, target_height = 1280, 720

        # Ensure aspect ratio precision
        calculated_aspect = target_width / target_height
        if abs(calculated_aspect - target_aspect_ratio) > 0.01:
            # Adjust width to ensure perfect aspect ratio
            target_width = int(target_height * target_aspect_ratio)

        # Determine if normalization is needed
        resolution_diversity = len(set(resolutions))
        fps_diversity = len(set(fps_counts.keys()))
        
        requires_normalization = (
            resolution_diversity > 1 or 
            fps_diversity > 1 or
            not all(abs(ar - target_aspect_ratio) < 0.1 for ar in aspect_ratios)
        )

        # Quality level assessment
        quality_info = "4K" if max_dimension >= 3840 else "HD" if max_dimension >= 1920 else "SD"
        
        # Enhanced logging
        print(f"Canvas Decision: {description}")
        print(f"Canvas Selection:")
        print(f"   - Resolution: {target_width}x{target_height}")
        print(f"   - Aspect ratio: {target_aspect_ratio:.3f} ({target_width}:{target_height})")
        print(f"   - Quality level: {quality_info} (max dimension: {max_dimension}px)")
        print(f"   - FPS: {target_fps}fps")
        print(f"Canvas Type: {canvas_type}")
        
        # Calculate expected letterboxing
        letterboxing_info = []
        for i, (w, h) in enumerate(resolutions):
            video_ar = w / h
            if abs(video_ar - target_aspect_ratio) > 0.05:  # Significant difference
                if video_ar > target_aspect_ratio:
                    letterboxing_info.append(f"Video {i+1}: letterbox (top/bottom bars)")
                else:
                    letterboxing_info.append(f"Video {i+1}: pillarbox (side bars)")
        
        if letterboxing_info:
            print(f"Expected letterboxing:")
            for info in letterboxing_info:
                print(f"   - {info}")
        else:
            print(f"✅ Minimal letterboxing expected - optimal aspect ratio match")

        return {
            "target_width": target_width,
            "target_height": target_height,
            "target_fps": target_fps,
            "target_aspect_ratio": target_aspect_ratio,
            "requires_normalization": requires_normalization,
            "canvas_type": canvas_type,
            "quality_level": quality_info,
            "max_dimension_preserved": max_dimension,
            "aspect_ratio_analysis": {
                "landscape_count": landscape_count,
                "portrait_count": portrait_count,
                "square_count": square_count,
                "dominant_orientation": "landscape" if landscape_ratio >= canvas_decision_threshold 
                                       else "portrait" if portrait_ratio >= canvas_decision_threshold
                                       else "square" if square_ratio >= canvas_decision_threshold
                                       else "mixed",
                "decision_rationale": description
            },
            "content_analysis": {
                "aspect_ratios": aspect_ratios,
                "resolutions": resolutions,
                "resolution_diversity": resolution_diversity,
                "fps_diversity": fps_diversity,
            },
            "letterboxing_analysis": letterboxing_info
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