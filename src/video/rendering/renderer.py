"""Main video rendering orchestrator for AutoCut.

This module provides the high-level render_video function that coordinates
all rendering subsystems: timeline loading, composition, audio sync, and encoding.
"""

import os
from typing import Optional, Callable
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
from .timeline import TimelineRenderer
from .compositor import VideoCompositor
from .audio_sync import AudioSynchronizer  
from .encoder import VideoEncoder
from .transitions import TransitionEngine

logger = get_logger(__name__)


class VideoRenderingOrchestrator:
    """Main orchestrator for the complete video rendering pipeline.
    
    Coordinates all rendering subsystems extracted from the original
    clip_assembler.py render_video function as part of Phase 3 refactoring.
    """
    
    def __init__(self, max_workers: int = 3):
        self.timeline_renderer = TimelineRenderer(max_workers)
        self.compositor = VideoCompositor()
        self.audio_synchronizer = AudioSynchronizer()
        self.encoder = VideoEncoder()
        self.transition_engine = TransitionEngine()
    
    def render_complete_video(
        self,
        timeline,
        audio_file: str,
        output_path: str,
        progress_callback: Optional[Callable] = None,
        add_transitions: bool = False,
        transition_duration: float = 0.5,
        bpm: Optional[float] = None,
        avg_beat_interval: Optional[float] = None
    ) -> str:
        """Render complete video with music synchronization.
        
        This is the main entry point that orchestrates the entire rendering pipeline
        with fixes for MoviePy 2.2.1 audio-video sync issues. Includes automatic
        musical fade-out when insufficient clips exist.
        
        Args:
            timeline: ClipTimeline with all clips and timing
            audio_file: Path to music file
            output_path: Path for output video
            progress_callback: Optional callback for progress updates
            add_transitions: Whether to add crossfade transitions
            transition_duration: Duration of transitions in seconds
            bpm: Beats per minute for musical fade calculations
            avg_beat_interval: Average time between beats in seconds
            
        Returns:
            Path to rendered video file
            
        Raises:
            VideoProcessingError: If rendering fails
        """
        def report_progress(step: str, progress: float):
            """Helper to report progress if callback provided."""
            if progress_callback:
                progress_callback(step, progress)
        
        try:
            # Validate inputs
            self._validate_inputs(timeline, audio_file)
            
            # Phase 1: Load video clips with intelligent strategy selection
            report_progress("Loading clips", 0.1)
            logger.info("Phase 1: Loading video clips for timeline")
            
            video_clips, failed_indices, resource_manager = \
                self.timeline_renderer.load_clips_for_timeline(timeline, progress_callback)
            
            # Synchronize timeline with loaded clips
            clean_video_clips = self.timeline_renderer.synchronize_timeline_with_loaded_clips(
                timeline, video_clips, failed_indices
            )
            
            # Phase 2: Format analysis and composition
            report_progress("Composing video", 0.4)
            logger.info("Phase 2: Format analysis and video composition")
            
            final_video, target_format = self.compositor.compose_timeline(clean_video_clips, timeline)
            
            # Phase 3: Audio loading and synchronization  
            report_progress("Preparing audio", 0.6)
            logger.info("Phase 3: Audio loading and synchronization")
            
            synchronized_audio, final_audio_duration = \
                self.audio_synchronizer.load_and_sync_audio(
                    audio_file, final_video.duration, target_format,
                    bpm, avg_beat_interval
                )
            
            # Attach audio to video
            report_progress("Attaching audio", 0.75)
            final_video = self.audio_synchronizer.attach_audio_to_video(
                final_video, synchronized_audio
            )
            
            # Phase 4: Optional transitions
            if add_transitions:
                report_progress("Adding transitions", 0.8)
                logger.info("Phase 4: Adding crossfade transitions")
                # Note: Transitions are typically applied before audio attachment
                # This is a simplified approach - for full transitions, they should
                # be applied to the clean_video_clips before composition
                logger.warning("Transitions should be applied before composition for optimal results")
            
            # Phase 5: Encoding and output
            report_progress("Encoding video", 0.85) 
            logger.info("Phase 5: Video encoding with optimal settings")
            
            encoding_params = self.encoder.prepare_encoding_parameters(target_format)
            output_path = self.encoder.encode_video(final_video, output_path, encoding_params)
            
            # Phase 6: Cleanup
            logger.info("Phase 6: Resource cleanup")
            self._cleanup_resources(final_video, synchronized_audio)
            self.timeline_renderer.cleanup_resources()
            
            report_progress("Rendering complete", 1.0)
            logger.info(f"Video rendering completed successfully: {output_path}")
            
            return output_path
            
        except Exception as e:
            # Cleanup on error
            try:
                self._emergency_cleanup()
                self.timeline_renderer.cleanup_resources()
            except:
                pass
            
            raise VideoProcessingError(f"Video rendering failed: {str(e)}")
    
    def _validate_inputs(self, timeline, audio_file: str):
        """Validate rendering inputs."""
        if not timeline.clips:
            raise VideoProcessingError("Timeline is empty - no clips to render")
        
        if not os.path.exists(audio_file):
            raise VideoProcessingError(f"Audio file not found: {audio_file}")
    
    def _cleanup_resources(self, final_video, audio_clip):
        """Clean up video and audio resources after successful rendering."""
        try:
            if final_video:
                final_video.close()
            if audio_clip:
                audio_clip.close()
            logger.debug("Video and audio resources cleaned up successfully")
        except Exception as e:
            logger.warning(f"Error during resource cleanup: {e}")
    
    def _emergency_cleanup(self):
        """Emergency cleanup in case of rendering failure."""
        try:
            # Force garbage collection to free resources
            import gc
            gc.collect()
            logger.debug("Emergency cleanup completed")
        except Exception as e:
            logger.warning(f"Error during emergency cleanup: {e}")


def render_video(
    timeline,
    audio_file: str,
    output_path: str,
    max_workers: int = 3,
    progress_callback: Optional[Callable] = None,
    bpm: Optional[float] = None,
    avg_beat_interval: Optional[float] = None,
) -> str:
    """Main video rendering function for backward compatibility.
    
    This function provides the same interface as the original render_video
    from clip_assembler.py but uses the new modular rendering architecture.
    Includes automatic musical fade-out when insufficient clips exist.
    
    Args:
        timeline: ClipTimeline with all clips and timing
        audio_file: Path to music file
        output_path: Path for output video
        max_workers: Maximum parallel workers (legacy parameter)
        progress_callback: Optional callback for progress updates
        bpm: Beats per minute for musical fade calculations
        avg_beat_interval: Average time between beats in seconds
        
    Returns:
        Path to rendered video file
        
    Raises:
        VideoProcessingError: If rendering fails
    """
    orchestrator = VideoRenderingOrchestrator(max_workers)
    return orchestrator.render_complete_video(
        timeline=timeline,
        audio_file=audio_file,
        output_path=output_path,
        progress_callback=progress_callback,
        add_transitions=False,  # Default to no transitions for compatibility
        bpm=bpm,
        avg_beat_interval=avg_beat_interval,
    )