"""Video composition and format normalization for rendering pipeline."""

from typing import List, Dict, Any, Tuple
try:
    from core.logging_config import get_logger
    from core.exceptions import VideoProcessingError
except ImportError:
    # Fallback for testing without proper package structure
    import logging
    def get_logger(name):
        return logging.getLogger(name)
    
    class VideoProcessingError(Exception):
        pass

logger = get_logger(__name__)


class VideoFormatAnalyzer:
    """Analyzes video formats and detects compatibility issues.
    
    Extracted from clip_assembler.py as part of Phase 3 refactoring.
    This class handles format analysis and normalization decisions.
    """
    
    def determine_optimal_canvas(self, video_clips: List) -> Dict[str, Any]:
        """Determine optimal 4:3 canvas size that preserves quality for all content.
        
        UPDATED APPROACH: Always outputs 4:3 aspect ratio (width:height = 4:3) while 
        preserving maximum quality. All content gets appropriately letterboxed/pillarboxed 
        within the 4:3 frame.
        
        Args:
            video_clips: List of video clips to analyze
            
        Returns:
            Dictionary containing optimal 4:3 canvas specifications
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
        max_dimension = 0  # Track the largest dimension for quality preservation
        fps_values = []
        content_types = []
        
        for i, clip in enumerate(video_clips):
            if clip is None:
                logger.error(f"CRITICAL: Clip {i} is None during format analysis - this indicates a problem in clip loading!")
                continue
                
            # Extract format information
            fps = getattr(clip, 'fps', 24.0)
            size = getattr(clip, 'size', (1920, 1080))
            duration = getattr(clip, 'duration', 0)
            
            fps_values.append(fps)
            width, height = size
            
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
            
            logger.debug(f"Clip {i}: {type(clip)} - duration {duration:.3f}s, format {width}x{height}@{fps}fps, type: {content_types[-1]}")

        # Calculate target FPS (average of all clips)
        target_fps = sum(fps_values) / len(fps_values) if fps_values else 24.0
        
        # ALWAYS CREATE 4:3 CANVAS - preserving maximum quality
        # Target aspect ratio: 4:3 (1.333)
        target_aspect_ratio = 4.0 / 3.0
        
        # Determine optimal dimensions for 4:3 while preserving quality
        # Use the largest dimension as a starting point for quality preservation
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
        
        # Verify aspect ratio is exactly 4:3
        calculated_aspect = target_width / target_height
        if abs(calculated_aspect - target_aspect_ratio) > 0.01:
            # Adjust width to ensure perfect 4:3 ratio
            target_width = int(target_height * target_aspect_ratio)

        # Determine canvas characteristics for logging
        unique_content_types = set(content_types)
        
        # Canvas type based on content mix
        if len(unique_content_types) == 1:
            if 'landscape' in unique_content_types:
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

        # Check if normalization is needed (always true for mixed content)
        resolution_set = set()
        fps_set = set()
        for clip in video_clips:
            if clip is not None:
                size = getattr(clip, 'size', (1920, 1080))
                fps = getattr(clip, 'fps', 24.0)
                resolution_set.add(f"{size[0]}x{size[1]}")
                fps_set.add(round(fps, 1))

        requires_normalization = (
            len(resolution_set) > 1 or 
            len(fps_set) > 1 or
            len(unique_content_types) > 1 or
            True  # Always normalize for 4:3 output since most content isn't 4:3
        )

        # Quality preservation logging
        quality_info = f"4K" if max_dimension >= 3840 else f"HD" if max_dimension >= 1920 else "SD"
        logger.info(f"Canvas Analysis: {description}")
        logger.info(f"   - Content types: {', '.join(unique_content_types)}")
        logger.info(f"   - Quality level: {quality_info} (max dimension: {max_dimension}px)")
        logger.info(f"Canvas Decision: {target_width}x{target_height} @ {target_fps:.1f}fps (4:3 aspect ratio)")
        logger.info(f"Canvas Type: {canvas_type}")
        logger.info("Quality Preservation: ✅ Optimal quality for 4:3 output")
        if requires_normalization:
            logger.info("Format normalization required for 4:3 output")
            
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
        }
    
    def detect_format_compatibility_issues(self, video_clips: List) -> List[Dict[str, Any]]:
        """Detect format compatibility issues that could cause artifacts.
        
        Args:
            video_clips: List of video clips to analyze
            
        Returns:
            List of detected issues with descriptions and potential artifacts
        """
        issues = []
        
        # Check for frame rate variations
        fps_values = []
        for clip in video_clips:
            if clip is not None:
                fps_values.append(getattr(clip, 'fps', 24.0))
        
        if fps_values:
            fps_range = max(fps_values) - min(fps_values)
            if fps_range > 5.0:  # More than 5 fps difference
                issues.append({
                    "type": "frame_rate_variance",
                    "description": f"Frame rate varies from {min(fps_values):.1f} to {max(fps_values):.1f} fps",
                    "artifacts": ["stuttering", "timing_issues", "sync_problems"]
                })
        
        # Check for resolution variations
        resolutions = set()
        for clip in video_clips:
            if clip is not None:
                size = getattr(clip, 'size', (1920, 1080))
                resolutions.add(f"{size[0]}x{size[1]}")
        
        if len(resolutions) > 1:
            issues.append({
                "type": "resolution_variance", 
                "description": f"Multiple resolutions detected: {', '.join(resolutions)}",
                "artifacts": ["scaling_artifacts", "quality_loss", "black_bars"]
            })
        
        return issues


class VideoNormalizationPipeline:
    """Pipeline for normalizing video clips to consistent format.
    
    Extracted from clip_assembler.py as part of Phase 3 refactoring.
    Handles format normalization to prevent visual artifacts.
    """
    
    def __init__(self, format_analyzer: VideoFormatAnalyzer):
        self.format_analyzer = format_analyzer
        
    def normalize_video_clips(self, video_clips: List, target_format: Dict[str, Any]) -> List:
        """Normalize video clips to target format.
        
        Args:
            video_clips: List of video clips to normalize
            target_format: Target format specification
            
        Returns:
            List of normalized video clips
        """
        if not target_format.get("requires_normalization", False):
            logger.info("No normalization required - formats are consistent")
            return video_clips
            
        logger.info("Applying format normalization to prevent artifacts")
        
        # Import compatibility functions
        try:
            from compatibility.moviepy import resize_clip_safely, set_fps_safely, check_moviepy_api_compatibility
            compatibility_info = check_moviepy_api_compatibility()
        except ImportError:
            logger.error("MoviePy compatibility layer not available - normalization may fail")
            compatibility_info = None
        
        normalized_clips = []
        target_fps = target_format["target_fps"]
        target_size = (target_format["target_width"], target_format["target_height"])
        
        for i, clip in enumerate(video_clips):
            if clip is None:
                normalized_clips.append(None)
                continue
                
            normalized_clip = clip
            
            # Normalize frame rate if needed
            current_fps = getattr(clip, 'fps', 24.0)
            if abs(current_fps - target_fps) > 0.1:
                logger.debug(f"Normalizing clip {i} fps: {current_fps:.1f} -> {target_fps:.1f}")
                try:
                    if compatibility_info:
                        normalized_clip = set_fps_safely(normalized_clip, target_fps, compatibility_info)
                    else:
                        # Fallback to direct method calls
                        try:
                            normalized_clip = normalized_clip.with_fps(target_fps)
                        except AttributeError:
                            try:
                                normalized_clip = normalized_clip.set_fps(target_fps)
                            except AttributeError:
                                logger.warning(f"Neither 'with_fps' nor 'set_fps' available for clip {i}")
                except Exception as e:
                    logger.warning(f"Failed to normalize fps for clip {i}: {e}")
            
            # Normalize resolution if needed
            current_size = getattr(clip, 'size', (1920, 1080))
            if current_size != target_size:
                logger.debug(f"Normalizing clip {i} resolution: {current_size} -> {target_size}")
                try:
                    if compatibility_info:
                        normalized_clip = resize_clip_safely(normalized_clip, newsize=target_size, compatibility_info=compatibility_info)
                    else:
                        # The resize_clip_safely function should handle all resize operations
                        logger.error(f"resize_clip_safely import failed but no fallback available for clip {i}")
                        raise RuntimeError("Cannot resize clip without resize_clip_safely function")
                except Exception as e:
                    logger.warning(f"Failed to normalize resolution for clip {i}: {e}")
            
            normalized_clips.append(normalized_clip)
        
        logger.info(f"Format normalization complete for {len(normalized_clips)} clips")
        return normalized_clips


class VideoCompositor:
    """High-level video composition orchestrator.
    
    Combines format analysis, normalization, and concatenation logic
    extracted from the original render_video function.
    """
    
    def __init__(self):
        self.format_analyzer = VideoFormatAnalyzer()
        self.normalization_pipeline = VideoNormalizationPipeline(self.format_analyzer)
    
    def compose_timeline(self, video_clips: List, timeline) -> Tuple[Any, Dict[str, Any]]:
        """Compose video clips into final timeline with format normalization.
        
        Args:
            video_clips: List of loaded video clips
            timeline: ClipTimeline with timing information
            
        Returns:
            Tuple of (final_video, format_info)
            
        Raises:
            VideoProcessingError: If composition fails
        """
        # Import MoviePy safely
        try:
            try:
                from compatibility.moviepy import import_moviepy_safely
            except ImportError:
                def import_moviepy_safely():
                    from moviepy.editor import VideoFileClip, AudioFileClip, concatenate_videoclips, CompositeVideoClip
                    return VideoFileClip, AudioFileClip, concatenate_videoclips, CompositeVideoClip
            
            VideoFileClip, AudioFileClip, concatenate_videoclips, CompositeVideoClip = import_moviepy_safely()
        except ImportError:
            raise VideoProcessingError("Failed to import MoviePy compatibility layer")
        
        # Format analysis and normalization
        logger.info(f"Analyzing format for {len(video_clips)} video clips")
        
        # CRITICAL: Debug clip validity before any processing
        none_clip_count = sum(1 for clip in video_clips if clip is None)
        valid_clip_count = len(video_clips) - none_clip_count
        logger.info(f"Input clip validation: {valid_clip_count} valid, {none_clip_count} None clips")
        
        if none_clip_count > 0:
            logger.warning(f"CRITICAL: Compositor received {none_clip_count} None clips - this should not happen!")
            for i, clip in enumerate(video_clips):
                if clip is None:
                    logger.warning(f"  Clip {i}: None")
                else:
                    logger.debug(f"  Clip {i}: {type(clip)} - {getattr(clip, 'duration', 'unknown')}s")
        
        target_format = self.format_analyzer.determine_optimal_canvas(video_clips)
        
        # Store target format in analyzer for smart fallbacks
        self.format_analyzer._target_format = target_format
        
        # Detect and log format compatibility issues
        format_issues = self.format_analyzer.detect_format_compatibility_issues(video_clips)
        if format_issues:
            logger.warning("Format issues detected:")
            for issue in format_issues:
                logger.warning(f"  - {issue['type']}: {issue['description']}")
                logger.warning(f"    Potential artifacts: {', '.join(issue['artifacts'])}")
        
        # Apply format normalization
        normalized_video_clips = self.normalization_pipeline.normalize_video_clips(
            video_clips, target_format
        )
        
        # CRITICAL: Filter out None clips before concatenation to prevent get_frame errors
        valid_clips = [clip for clip in normalized_video_clips if clip is not None]
        
        if not valid_clips:
            raise VideoProcessingError("No valid clips remaining after normalization")
        
        logger.info(f"Concatenation input: {len(valid_clips)} valid clips (filtered from {len(normalized_video_clips)} total)")
        
        # Additional validation: Check if any valid clips might have become invalid during normalization
        for i, clip in enumerate(valid_clips):
            try:
                # Test clip properties to ensure it's properly loaded
                duration = getattr(clip, 'duration', None)
                size = getattr(clip, 'size', None)
                if duration is None or size is None:
                    logger.warning(f"Valid clip {i} has invalid properties: duration={duration}, size={size}")
                else:
                    logger.debug(f"Valid clip {i}: {duration:.2f}s, {size}")
            except Exception as e:
                logger.error(f"Valid clip {i} validation failed: {e}")
        
        # Validate valid clip durations
        total_expected_duration = self._validate_valid_clip_durations(valid_clips, timeline)
        
        # Choose concatenation method based on format requirements
        concatenation_method = "compose" if target_format.get("requires_normalization", False) else "chain"
        logger.info(f"Using '{concatenation_method}' concatenation method")
        
        # Perform concatenation with enhanced null checking
        try:
            # CRITICAL FIX: Final defensive check before concatenation
            # Check for None clips that may have slipped through normalization
            pre_concat_valid_clips = []
            corrupted_clip_count = 0
            
            for i, clip in enumerate(valid_clips):
                if clip is None:
                    logger.error(f"  Clip {i}: None (should have been filtered earlier!)")
                    corrupted_clip_count += 1
                    continue
                    
                # Test clip viability by checking key attributes
                try:
                    # Check if clip has essential attributes
                    if not hasattr(clip, 'start'):
                        logger.error(f"  Clip {i}: Missing 'start' attribute (corrupted)")
                        corrupted_clip_count += 1
                        continue
                        
                    if not hasattr(clip, 'duration') or clip.duration is None:
                        logger.error(f"  Clip {i}: Missing or None duration (corrupted)")
                        corrupted_clip_count += 1
                        continue
                        
                    # Basic attribute access test
                    _ = clip.start
                    _ = clip.duration
                    
                    pre_concat_valid_clips.append(clip)
                    
                except Exception as attr_error:
                    logger.error(f"  Clip {i}: Attribute access failed: {attr_error}")
                    corrupted_clip_count += 1
                    continue
            
            if corrupted_clip_count > 0:
                logger.warning(f"🛡️  Defensive filtering removed {corrupted_clip_count} corrupted clips")
                logger.warning(f"   Final clip count: {len(pre_concat_valid_clips)}/{len(valid_clips)}")
            
            if not pre_concat_valid_clips:
                raise VideoProcessingError("No viable clips remaining after final corruption check")
            
            logger.info(f"Starting concatenation with {len(pre_concat_valid_clips)} verified clips using {concatenation_method} method")
            final_video = concatenate_videoclips(pre_concat_valid_clips, method=concatenation_method)
            logger.info(f"Concatenation successful - final video duration: {final_video.duration:.2f}s")
        except Exception as e:
            logger.error(f"Concatenation failed with {len(pre_concat_valid_clips)} verified clips")
            for i, clip in enumerate(pre_concat_valid_clips):
                try:
                    logger.error(f"  Clip {i}: {type(clip)} - duration={getattr(clip, 'duration', 'N/A')}, size={getattr(clip, 'size', 'N/A')}")
                except:
                    logger.error(f"  Clip {i}: Unable to inspect clip properties")
            raise VideoProcessingError(f"Video concatenation failed: {str(e)}")
        
        # Validate final duration
        actual_video_duration = final_video.duration
        logger.info(f"Concatenation successful, final video duration: {actual_video_duration:.6f}s")
        
        duration_error = abs(actual_video_duration - total_expected_duration)
        if duration_error > 0.05:  # More than 50ms error
            logger.warning(f"Concatenation timing error: {duration_error:.6f}s discrepancy")
        
        return final_video, target_format
    
    def _validate_valid_clip_durations(self, valid_clips: List, timeline) -> float:
        """Validate valid clip durations match timeline expectations.
        
        Args:
            valid_clips: List of valid (non-None) video clips 
            timeline: ClipTimeline with expected durations
            
        Returns:
            Total expected duration in seconds
        """
        total_expected_duration = 0
        
        # Map valid clips back to timeline positions for duration validation
        timeline_clips = [clip for clip in timeline.clips if clip is not None]
        
        for i, clip in enumerate(valid_clips):
            if i < len(timeline_clips):
                clip_duration = clip.duration
                expected_duration = timeline_clips[i]["duration"]
                
                logger.debug(f"Valid clip {i + 1}: actual={clip_duration:.6f}s, expected={expected_duration:.6f}s")
                
                # Check for significant duration discrepancies
                duration_diff = abs(clip_duration - expected_duration)
                if duration_diff > 0.1:  # More than 100ms difference
                    logger.warning(f"Valid clip {i + 1} duration mismatch: {duration_diff:.6f}s difference")
                
                total_expected_duration += clip_duration
            else:
                # If we have more valid clips than expected, just use actual duration
                total_expected_duration += clip.duration
        
        logger.info(f"Total expected video duration: {total_expected_duration:.6f}s")
        return total_expected_duration