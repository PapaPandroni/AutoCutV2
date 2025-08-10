"""Audio synchronization system for video rendering."""

import os
from typing import Dict, Any, Tuple
try:
    from core.logging_config import get_logger
    from core.exceptions import AudioAnalysisError, VideoProcessingError
except ImportError:
    # Fallback for testing without proper package structure
    import logging
    def get_logger(name):
        return logging.getLogger(name)
    
    class AudioAnalysisError(Exception):
        pass
    
    class VideoProcessingError(Exception):
        pass

logger = get_logger(__name__)


class AudioSynchronizer:
    """Handles precise audio-video synchronization for rendering.
    
    Extracted from clip_assembler.py render_video function as part of Phase 3 refactoring.
    Provides frame-accurate audio timing to prevent sync issues.
    """
    
    def __init__(self):
        self.compatibility_info = None
    
    def load_and_sync_audio(
        self, 
        audio_file: str, 
        video_duration: float, 
        target_format: Dict[str, Any]
    ) -> Tuple[Any, float]:
        """Load audio and synchronize with video duration.
        
        Args:
            audio_file: Path to audio file
            video_duration: Duration of video in seconds
            target_format: Target format specifications
            
        Returns:
            Tuple of (synchronized_audio_clip, final_audio_duration)
            
        Raises:
            AudioAnalysisError: If audio loading fails
            VideoProcessingError: If sync calculation fails
        """
        if not os.path.exists(audio_file):
            raise AudioAnalysisError(f"Audio file not found: {audio_file}")
        
        logger.info(f"Loading audio file: {os.path.basename(audio_file)}")
        
        # Load audio with robust handling
        try:
            audio_clip = self._load_audio_robust(audio_file)
            original_audio_duration = audio_clip.duration
            logger.info(f"Original audio duration: {original_audio_duration:.6f}s")
        except Exception as e:
            raise AudioAnalysisError(f"Failed to load audio file {audio_file}: {str(e)}")
        
        # Calculate frame-accurate audio duration
        synchronized_audio, final_duration = self._calculate_frame_accurate_sync(
            audio_clip, original_audio_duration, video_duration, target_format
        )
        
        return synchronized_audio, final_duration
    
    def _load_audio_robust(self, audio_file: str) -> Any:
        """Load audio file with MoviePy compatibility handling.
        
        Args:
            audio_file: Path to audio file
            
        Returns:
            AudioFileClip instance
        """
        try:
            # Import compatibility layer
            try:
                from compatibility.moviepy import import_moviepy_safely
            except ImportError:
                def import_moviepy_safely():
                    from moviepy.editor import VideoFileClip, AudioFileClip, concatenate_videoclips, CompositeVideoClip
                    return VideoFileClip, AudioFileClip, concatenate_videoclips, CompositeVideoClip
            
            VideoFileClip, AudioFileClip, concatenate_videoclips, CompositeVideoClip = import_moviepy_safely()
            
            # Use simple AudioFileClip approach (more reliable than complex safe_load_audio_file)
            return AudioFileClip(audio_file)
            
        except ImportError as e:
            raise AudioAnalysisError(f"Failed to import MoviePy compatibility layer: {str(e)}")
    
    def _calculate_frame_accurate_sync(
        self, 
        audio_clip: Any, 
        original_duration: float, 
        video_duration: float, 
        target_format: Dict[str, Any]
    ) -> Tuple[Any, float]:
        """Calculate frame-accurate audio sync to prevent cutoff issues.
        
        Args:
            audio_clip: Original audio clip
            original_duration: Original audio duration
            video_duration: Target video duration 
            target_format: Target format specifications
            
        Returns:
            Tuple of (synchronized_audio_clip, final_duration)
        """
        # Get compatibility info for safe operations
        if not self.compatibility_info:
            try:
                try:
                    from compatibility.moviepy import check_moviepy_api_compatibility
                except ImportError:
                    def check_moviepy_api_compatibility():
                        return {"version_detected": "unknown", "method_mappings": {"subclip": "subclip", "set_audio": "set_audio"}}
                self.compatibility_info = check_moviepy_api_compatibility()
            except ImportError:
                raise VideoProcessingError("MoviePy compatibility layer not available")
        
        # Calculate frame-accurate timing parameters
        target_fps = target_format["target_fps"]
        frame_duration = 1.0 / target_fps
        sync_buffer = frame_duration * 2  # 2-frame buffer to prevent cutoff
        
        logger.info(f"Target FPS: {target_fps}, Frame duration: {frame_duration:.6f}s")
        logger.info(f"Adding sync buffer: {sync_buffer:.6f}s to prevent audio cutoff")
        
        # Prepare audio with frame-accurate timing
        if original_duration > video_duration:
            # Trim audio to match video duration, plus sync buffer
            audio_end_time = min(video_duration + sync_buffer, original_duration)
            logger.info(f"Frame-accurate audio trim: {original_duration:.6f}s -> {audio_end_time:.6f}s")
            logger.info(f"Audio buffer added: {sync_buffer:.6f}s to prevent cutoff")
            
            try:
                try:
                    from compatibility.moviepy import subclip_safely
                except ImportError:
                    def subclip_safely(clip, start_time, end_time, compatibility_info):
                        return clip.subclip(start_time, end_time)
                
                synchronized_audio = subclip_safely(
                    audio_clip, 0, audio_end_time, self.compatibility_info
                )
            except ImportError:
                raise VideoProcessingError("MoviePy subclip compatibility function not available")
        else:
            # Audio is shorter than video - use full audio
            logger.info(f"Audio ({original_duration:.6f}s) shorter than video ({video_duration:.6f}s)")
            synchronized_audio = audio_clip
        
        final_duration = synchronized_audio.duration
        logger.info(f"Final audio duration: {final_duration:.6f}s")
        
        # Validate sync quality
        sync_difference = abs(final_duration - video_duration)
        logger.info(f"Audio-video sync difference: {sync_difference:.6f}s")
        
        if sync_difference > 0.1:  # More than 100ms difference
            logger.warning(f"Significant audio-video sync difference: {sync_difference:.6f}s")
        
        return synchronized_audio, final_duration
    
    def attach_audio_to_video(self, video: Any, audio: Any) -> Any:
        """Attach audio to video using compatibility layer.
        
        Args:
            video: Video clip
            audio: Audio clip
            
        Returns:
            Video with attached audio
        """
        try:
            try:
                from compatibility.moviepy import attach_audio_safely
            except ImportError:
                def attach_audio_safely(video, audio, compatibility_info):
                    return video.set_audio(audio)
            
            if not self.compatibility_info:
                try:
                    from compatibility.moviepy import check_moviepy_api_compatibility
                except ImportError:
                    def check_moviepy_api_compatibility():
                        return {"version_detected": "unknown", "method_mappings": {"subclip": "subclip", "set_audio": "set_audio"}}
                self.compatibility_info = check_moviepy_api_compatibility()
            
            logger.info(f"Attaching audio using method: {self.compatibility_info['method_mappings']['set_audio']}")
            return attach_audio_safely(video, audio, self.compatibility_info)
            
        except ImportError:
            raise VideoProcessingError("MoviePy audio attachment compatibility function not available")


def load_audio_robust(audio_file: str) -> Any:
    """Legacy function for backward compatibility.
    
    Args:
        audio_file: Path to audio file
        
    Returns:
        AudioFileClip instance
    """
    synchronizer = AudioSynchronizer()
    return synchronizer._load_audio_robust(audio_file)