"""Timeline rendering system for video composition."""

import os
import gc
from typing import List, Optional, Callable, Dict, Any, Tuple

try:
    from ..loading.strategies import UnifiedVideoLoader, LoadingStrategyType, ClipSpec
    from ..loading.cache import VideoCache
    from ..loading.resource_manager import VideoResourceManager
except ImportError:
    # Fallback definitions if loading modules not available
    class UnifiedVideoLoader:
        def load_clips(self, clip_specs): return []
    class LoadingStrategyType:
        SEQUENTIAL = "sequential"
        ROBUST = "robust"
        AUTO = "auto"
    class ClipSpec:
        pass
    class VideoCache:
        def clear(self): pass
    class VideoResourceManager:
        def cleanup_delayed_videos(self): pass 
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


class TimelineRenderer:
    """Handles timeline-based video rendering with intelligent loading strategies."""
    
    def __init__(self, max_workers: int = 3):
        self.max_workers = max_workers
        self.video_cache = None
        self.resource_manager = None
        
    def load_clips_for_timeline(
        self, 
        timeline,
        progress_callback: Optional[Callable] = None
    ) -> Tuple[List, List[int], Optional[VideoResourceManager]]:
        """Load video clips for timeline rendering with intelligent strategy selection.
        
        Args:
            timeline: ClipTimeline with clips and timing
            progress_callback: Optional progress callback
            
        Returns:
            Tuple of (video_clips, failed_indices, resource_manager)
            
        Raises:
            VideoProcessingError: If no clips can be loaded
        """
        # Get clips sorted by beat position
        sorted_clips = timeline.get_clips_sorted_by_beat()
        unique_video_files = timeline.get_unique_video_files()
        
        logger.info(f"Timeline has {len(sorted_clips)} clips to load")
        
        # Convert timeline clips to ClipSpec format for new unified loader
        clip_specs = []
        for clip_data in sorted_clips:
            try:
                clip_spec = ClipSpec(
                    file_path=clip_data.get('video_file', ''),
                    start_time=clip_data.get('start', 0.0),
                    end_time=clip_data.get('end', clip_data.get('start', 0.0) + clip_data.get('duration', 1.0)),
                    quality_score=clip_data.get('quality_score', 0.0)
                )
                clip_specs.append(clip_spec)
            except Exception as e:
                logger.warning(f"Failed to create ClipSpec from {clip_data}: {e}")
                continue
        
        # Analyze system conditions for strategy selection
        complexity_factors = self._analyze_system_complexity(sorted_clips, unique_video_files)
        complexity_score = sum(complexity_factors.values())
        
        logger.info(f"System complexity analysis: {complexity_score}/6 factors detected")
        for factor, present in complexity_factors.items():
            if present:
                logger.debug(f"  - {factor.replace('_', ' ').title()}")
        
        # Select strategy based on complexity
        if complexity_score >= 4:
            strategy = LoadingStrategyType.ROBUST
            logger.info("Using robust error handling (maximum reliability)")
        elif complexity_score >= 2:
            strategy = LoadingStrategyType.ROBUST  # Use robust for conservative approach
            logger.info("Using robust loading (conservative mode)")
        else:
            strategy = LoadingStrategyType.SEQUENTIAL
            logger.info("Using sequential video loading (standard mode)")
        
        # Use unified video loader
        try:
            loader = UnifiedVideoLoader(
                default_strategy=strategy,
                cache=self.video_cache,
                resource_manager=self.resource_manager
            )
            
            loaded_clips = loader.load_clips(clip_specs, strategy=strategy)
            
            # Convert LoadedClip objects back to video clips and track failures
            video_clips = []
            failed_indices = []
            
            for i, loaded_clip in enumerate(loaded_clips):
                if loaded_clip and hasattr(loaded_clip, 'clip'):
                    video_clips.append(loaded_clip.clip)
                else:
                    video_clips.append(None)
                    failed_indices.append(i)
            
            # Validate loading results
            successful_clips = [clip for clip in video_clips if clip is not None]
            if not successful_clips:
                raise VideoProcessingError("No video clips could be loaded successfully")
                
            logger.info(f"Loaded {len(successful_clips)} video clips successfully")
            
            return video_clips, failed_indices, loader.resource_manager
            
        except Exception as e:
            logger.error(f"Unified loader failed: {e}")
            # Fallback to empty results
            return [], list(range(len(clip_specs))), self.resource_manager or VideoResourceManager()
    
    def synchronize_timeline_with_loaded_clips(
        self, 
        timeline, 
        video_clips: List,
        failed_indices: List[int]
    ) -> List:
        """Synchronize timeline with successfully loaded clips.
        
        Args:
            timeline: ClipTimeline to synchronize
            video_clips: List of loaded video clips (may contain None)
            failed_indices: List of indices that failed to load
            
        Returns:
            Clean list of video clips without None values
        """
        # Defensive validation: Check for None clips
        none_clip_indices = [i for i, clip in enumerate(video_clips) if clip is None]
        
        if none_clip_indices:
            logger.info(f"Found {len(none_clip_indices)} None clips at indices {none_clip_indices}")
            
            # Verify that None clips match our failed_indices  
            expected_none_indices = set(failed_indices)
            actual_none_indices = set(none_clip_indices)
            
            if expected_none_indices != actual_none_indices:
                logger.warning(f"Index mismatch: Expected None at {expected_none_indices}, found at {actual_none_indices}")
                failed_indices = sorted(list(actual_none_indices))
        
        # Synchronize timeline by removing failed clips
        if failed_indices:
            logger.info(f"Removing {len(failed_indices)} failed clips from timeline")
            original_clips = timeline.clips.copy()
            successful_clips = [
                clip for i, clip in enumerate(original_clips) 
                if i not in failed_indices
            ]
            timeline.clips = successful_clips
            logger.info(f"Timeline synchronized: {len(original_clips)} -> {len(timeline.clips)} clips")
        
        # Filter out None clips for clean rendering array
        clean_video_clips = [clip for clip in video_clips if clip is not None]
        
        # Verify clip count consistency
        if len(clean_video_clips) != len(timeline.clips):
            raise VideoProcessingError(
                f"Clip count mismatch: {len(clean_video_clips)} valid clips vs {len(timeline.clips)} timeline clips"
            )
        
        logger.info(f"Final clip count - video_clips: {len(clean_video_clips)}, timeline.clips: {len(timeline.clips)}")
        
        return clean_video_clips
    
    def _analyze_system_complexity(self, sorted_clips: List, unique_video_files: List) -> Dict[str, bool]:
        """Analyze system complexity factors to choose optimal loading strategy."""
        try:
            from memory.monitor import get_memory_info
            initial_memory = get_memory_info()
        except ImportError:
            # Fallback if memory monitoring not available
            initial_memory = {"percent": 50, "available_gb": 8.0}
        
        return {
            "high_memory_usage": initial_memory["percent"] > 70,
            "low_available_memory": initial_memory["available_gb"] < 4.0,
            "many_clips": len(sorted_clips) > 50,
            "many_files": len(unique_video_files) > 20,
            "large_clips": any(
                (clip.get("end", 0) - clip.get("start", 0)) > 10
                for clip in sorted_clips
            ),
            "mixed_formats": len(
                set(
                    clip.get("video_file", "").split(".")[-1].lower()
                    for clip in sorted_clips
                )
            ) > 2,
        }
    
    def cleanup_resources(self):
        """Clean up video loading resources."""
        if self.resource_manager:
            logger.info("Cleaning up parent videos after rendering")
            self.resource_manager.cleanup_delayed_videos()
        else:
            # Fallback cleanup
            gc.collect()
            
        if self.video_cache:
            self.video_cache.clear()
        
        logger.info("Timeline renderer resource cleanup complete")