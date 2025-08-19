"""Transition effects system for video rendering."""

from typing import List, Any
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


class TransitionEngine:
    """Handles transition effects between video clips.
    
    Extracted from clip_assembler.py add_transitions function as part of 
    Phase 3 refactoring to create focused, testable transition system.
    """
    
    def __init__(self):
        self.fade_effects = None
        self._initialize_effects()
    
    def _initialize_effects(self):
        """Initialize fade effects with version compatibility."""
        try:
            from moviepy.video.fx import fadeout, fadein
            self.fade_effects = {"fadeout": fadeout, "fadein": fadein}
            logger.debug("MoviePy fade effects loaded successfully")
        except ImportError:
            try:
                # MoviePy 2.x might have capitalized effect names
                from moviepy.video.fx import FadeOut as fadeout, FadeIn as fadein
                self.fade_effects = {"fadeout": fadeout, "fadein": fadein}
                logger.debug("MoviePy 2.x fade effects loaded successfully")
            except ImportError:
                logger.warning("Fade effects not available, transitions will be skipped")
                self.fade_effects = None
    
    def add_crossfade_transitions(
        self, 
        clips: List[Any], 
        transition_duration: float = 0.5
    ) -> Any:
        """Add crossfade transitions between clips.
        
        Args:
            clips: List of video clips
            transition_duration: Duration of crossfade in seconds
            
        Returns:
            Composite video with transitions
            
        Raises:
            VideoProcessingError: If clips list is invalid
        """
        try:
            # Import MoviePy safely with dual import pattern
            try:
                # Relative import for package execution
                from ...compatibility.moviepy import import_moviepy_safely
            except ImportError:
                try:
                    # Absolute import for direct execution
                    from compatibility.moviepy import import_moviepy_safely
                except ImportError:
                    # Fallback if compatibility module not available
                    def import_moviepy_safely():
                        from moviepy.editor import VideoFileClip, AudioFileClip, concatenate_videoclips, CompositeVideoClip
                        return VideoFileClip, AudioFileClip, concatenate_videoclips, CompositeVideoClip
            
            VideoFileClip, AudioFileClip, concatenate_videoclips, CompositeVideoClip = import_moviepy_safely()
        except ImportError:
            raise VideoProcessingError("Failed to import MoviePy compatibility layer")
        
        if not clips:
            raise VideoProcessingError("No clips provided for transitions")
        
        logger.info(f"Adding transitions to {len(clips)} clips with {transition_duration}s duration")
        
        # Handle single clip case
        if len(clips) == 1:
            return self._apply_single_clip_fades(clips[0])
        
        # Handle multiple clips with crossfades
        return self._apply_crossfade_transitions(clips, transition_duration, concatenate_videoclips)
    
    def _apply_single_clip_fades(self, clip: Any) -> Any:
        """Apply fade in/out effects to single clip.
        
        Args:
            clip: Single video clip
            
        Returns:
            Clip with fade effects applied
        """
        if not self.fade_effects:
            logger.info("No fade effects available, returning original clip")
            return clip
        
        try:
            processed_clip = clip
            # Add fade in at start (0.5s)
            processed_clip = processed_clip.fx(self.fade_effects["fadein"], 0.5)
            # Add fade out at end (0.5s)  
            processed_clip = processed_clip.fx(self.fade_effects["fadeout"], 0.5)
            logger.debug("Applied fade in/out to single clip")
            return processed_clip
        except Exception as e:
            logger.warning(f"Failed to apply fade effects to single clip: {e}")
            return clip
    
    def _apply_crossfade_transitions(
        self, 
        clips: List[Any], 
        transition_duration: float,
        concatenate_videoclips: callable
    ) -> Any:
        """Apply crossfade transitions to multiple clips.
        
        Args:
            clips: List of video clips
            transition_duration: Duration of transitions
            concatenate_videoclips: MoviePy concatenation function
            
        Returns:
            Video with crossfade transitions
        """
        if not self.fade_effects:
            logger.info("No fade effects available, using simple concatenation")
            return concatenate_videoclips(clips, method="chain")
        
        processed_clips = []
        
        for i, clip in enumerate(clips):
            current_clip = clip.copy()
            
            try:
                if i == 0:
                    # First clip: fade in at start, fade out at end for transition
                    current_clip = current_clip.fx(self.fade_effects["fadein"], 0.5)
                    if len(clips) > 1:
                        current_clip = current_clip.fx(self.fade_effects["fadeout"], transition_duration)
                
                elif i == len(clips) - 1:
                    # Last clip: fade in from previous, fade out at end
                    current_clip = current_clip.fx(self.fade_effects["fadein"], transition_duration)
                    current_clip = current_clip.fx(self.fade_effects["fadeout"], 0.5)
                
                else:
                    # Middle clips: fade in from previous, fade out to next
                    current_clip = current_clip.fx(self.fade_effects["fadein"], transition_duration)
                    current_clip = current_clip.fx(self.fade_effects["fadeout"], transition_duration)
                
                logger.debug(f"Applied transition effects to clip {i + 1}")
                
            except Exception as e:
                logger.warning(f"Failed to apply transition effects to clip {i + 1}: {e}")
                # Use original clip if effects fail
            
            processed_clips.append(current_clip)
        
        # Concatenate with overlapping transitions
        try:
            logger.info("Concatenating clips with crossfade transitions")
            final_video = concatenate_videoclips(
                processed_clips, 
                padding=-transition_duration, 
                method="compose"
            )
        except Exception as e:
            logger.warning(f"Compose method failed: {e}, falling back to chain method")
            final_video = concatenate_videoclips(processed_clips, method="chain")
        
        return final_video


def add_transitions(clips: List[Any], transition_duration: float = 0.5) -> Any:
    """Legacy function for backward compatibility.
    
    Args:
        clips: List of video clips
        transition_duration: Duration of crossfade in seconds
        
    Returns:
        Composite video with transitions
    """
    engine = TransitionEngine()
    return engine.add_crossfade_transitions(clips, transition_duration)