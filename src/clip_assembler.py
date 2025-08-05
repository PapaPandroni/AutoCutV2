"""
Clip Assembly Module for AutoCut

Handles the core logic of matching video clips to musical beats,
applying variety patterns, and rendering the final video.
"""

from typing import Dict, List, Tuple, Optional, Any
import json
import os
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import defaultdict
try:
    from moviepy.editor import VideoFileClip, CompositeVideoClip, concatenate_videoclips
except ImportError:
    try:
        # Fallback for MoviePy 2.x direct imports
        from moviepy import VideoFileClip, CompositeVideoClip, concatenate_videoclips
    except ImportError:
        # Final fallback for testing without moviepy installation
        VideoFileClip = CompositeVideoClip = concatenate_videoclips = None
try:
    from .video_analyzer import VideoChunk
except ImportError:
    # Direct import for testing
    from video_analyzer import VideoChunk


# Variety patterns to prevent monotonous cutting
VARIETY_PATTERNS = {
    'energetic': [1, 1, 2, 1, 1, 4],  # Mostly fast with occasional pause
    'buildup': [4, 2, 2, 1, 1, 1],    # Start slow, increase pace
    'balanced': [2, 1, 2, 4, 2, 1],   # Mixed pacing
    'dramatic': [1, 1, 1, 1, 8],      # Fast cuts then long hold
}


class VideoCache:
    """Thread-safe cache for loaded video files to prevent duplicate loading."""
    
    def __init__(self):
        self._cache: Dict[str, VideoFileClip] = {}
        self._lock = threading.Lock()
        self._ref_counts: Dict[str, int] = defaultdict(int)
    
    def get_or_load(self, video_path: str) -> VideoFileClip:
        """Get cached video or load it if not cached.
        
        Args:
            video_path: Path to video file
            
        Returns:
            VideoFileClip instance
            
        Raises:
            FileNotFoundError: If video file doesn't exist
            RuntimeError: If video cannot be loaded
        """
        if VideoFileClip is None:
            raise RuntimeError("MoviePy not available. Please install moviepy>=1.0.3")
            
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")
        
        with self._lock:
            if video_path not in self._cache:
                try:
                    video_clip = VideoFileClip(video_path)
                    self._cache[video_path] = video_clip
                except Exception as e:
                    raise RuntimeError(f"Failed to load video {video_path}: {str(e)}")
            
            # Increment reference count
            self._ref_counts[video_path] += 1
            return self._cache[video_path]
    
    def release(self, video_path: str) -> None:
        """Release reference to cached video.
        
        Args:
            video_path: Path to video file to release
        """
        with self._lock:
            if video_path in self._ref_counts:
                self._ref_counts[video_path] -= 1
                
                # Remove from cache if no more references
                if self._ref_counts[video_path] <= 0:
                    if video_path in self._cache:
                        try:
                            self._cache[video_path].close()
                        except Exception:
                            pass  # Ignore cleanup errors
                        del self._cache[video_path]
                    del self._ref_counts[video_path]
    
    def get_cached_paths(self) -> List[str]:
        """Get list of currently cached video paths.
        
        Returns:
            List of cached video file paths
        """
        with self._lock:
            return list(self._cache.keys())
    
    def clear(self) -> None:
        """Clear all cached videos and close resources."""
        with self._lock:
            for video_clip in self._cache.values():
                try:
                    video_clip.close()
                except Exception:
                    pass  # Ignore cleanup errors
            self._cache.clear()
            self._ref_counts.clear()

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
                'width': video_clip.w,
                'height': video_clip.h,
                'fps': video_clip.fps,
                'duration': video_clip.duration,
                'aspect_ratio': video_clip.w / video_clip.h if video_clip.h > 0 else 1.0,
                'resolution_category': self._categorize_resolution(video_clip.w, video_clip.h),
                'fps_category': self._categorize_fps(video_clip.fps),
            }
            
            # Add codec information if available
            if hasattr(video_clip, 'filename'):
                format_info['filename'] = video_clip.filename
            
            return format_info
            
        except Exception as e:
            print(f"Warning: Could not analyze video format: {e}")
            return {
                'width': 1920, 'height': 1080, 'fps': 24.0, 'duration': 0.0,
                'aspect_ratio': 16/9, 'resolution_category': '1080p', 'fps_category': '24fps'
            }
    
    def _categorize_resolution(self, width: int, height: int) -> str:
        """Categorize resolution into standard formats."""
        if width >= 3840 and height >= 2160:
            return '4K'
        elif width >= 2560 and height >= 1440:
            return '1440p'
        elif width >= 1920 and height >= 1080:
            return '1080p'
        elif width >= 1280 and height >= 720:
            return '720p'
        else:
            return 'SD'
    
    def _categorize_fps(self, fps: float) -> str:
        """Categorize frame rate into standard categories."""
        if fps >= 59.0:
            return '60fps'
        elif fps >= 29.0:
            return '30fps'
        elif fps >= 24.0:
            return '25fps'
        else:
            return '24fps'
    
    def find_dominant_format(self, video_clips: List[Any]) -> Dict[str, Any]:
        """Determine the dominant format across all clips for normalization target.
        
        Args:
            video_clips: List of VideoFileClip instances
            
        Returns:
            Target format specification for normalization
        """
        if not video_clips:
            return {
                'target_width': 1920, 'target_height': 1080, 'target_fps': 24.0,
                'target_aspect_ratio': 16/9, 'requires_normalization': False
            }
        
        formats = []
        for clip in video_clips:
            format_info = self.analyze_video_format(clip)
            formats.append(format_info)
        
        # Find most common resolution
        resolution_counts = {}
        fps_counts = {}
        
        for fmt in formats:
            res_key = f"{fmt['width']}x{fmt['height']}"
            fps_key = fmt['fps_category']
            
            resolution_counts[res_key] = resolution_counts.get(res_key, 0) + 1
            fps_counts[fps_key] = fps_counts.get(fps_key, 0) + 1
        
        # Determine target resolution (prefer highest quality that's most common)
        dominant_resolution = max(resolution_counts, key=resolution_counts.get)
        width, height = map(int, dominant_resolution.split('x'))
        
        # Determine target FPS (most common)
        dominant_fps_category = max(fps_counts, key=fps_counts.get)
        target_fps = self._fps_category_to_value(dominant_fps_category)
        
        # Check if normalization is needed
        requires_normalization = len(set(resolution_counts.keys())) > 1 or len(set(fps_counts.keys())) > 1
        
        print(f"Format Analysis: Dominant {dominant_resolution} @ {target_fps}fps")
        print(f"Format Diversity: {len(resolution_counts)} resolutions, {len(fps_counts)} frame rates")
        print(f"Normalization Required: {requires_normalization}")
        
        return {
            'target_width': width,
            'target_height': height, 
            'target_fps': target_fps,
            'target_aspect_ratio': width / height,
            'requires_normalization': requires_normalization,
            'format_diversity': {
                'resolutions': len(resolution_counts),
                'frame_rates': len(fps_counts)
            }
        }
    
    def _fps_category_to_value(self, fps_category: str) -> float:
        """Convert FPS category back to numeric value."""
        mapping = {
            '60fps': 60.0,
            '30fps': 30.0, 
            '25fps': 25.0,
            '24fps': 24.0
        }
        return mapping.get(fps_category, 24.0)
    
    def detect_format_compatibility_issues(self, video_clips: List[Any]) -> List[Dict[str, Any]]:
        """Identify specific compatibility issues between video clips.
        
        Returns:
            List of issue descriptions with recommended fixes
        """
        if len(video_clips) < 2:
            return []
        
        issues = []
        formats = [self.analyze_video_format(clip) for clip in video_clips]
        
        # Check resolution variations
        resolutions = set((fmt['width'], fmt['height']) for fmt in formats)
        if len(resolutions) > 1:
            issues.append({
                'type': 'resolution_mismatch',
                'description': f"Mixed resolutions detected: {resolutions}",
                'severity': 'high',
                'artifacts': ['flashing up/down', 'scaling artifacts', 'centering issues'],
                'fix': 'resolution_normalization'
            })
        
        # Check frame rate variations  
        fps_values = set(fmt['fps'] for fmt in formats)
        if len(fps_values) > 1:
            issues.append({
                'type': 'framerate_mismatch',
                'description': f"Mixed frame rates detected: {fps_values}",
                'severity': 'high', 
                'artifacts': ['VHS-like wrap around', 'temporal stuttering', 'motion artifacts'],
                'fix': 'framerate_normalization'
            })
        
        # Check aspect ratio variations
        aspect_ratios = set(round(fmt['aspect_ratio'], 3) for fmt in formats) 
        if len(aspect_ratios) > 1:
            issues.append({
                'type': 'aspect_ratio_mismatch',
                'description': f"Mixed aspect ratios detected: {aspect_ratios}",
                'severity': 'medium',
                'artifacts': ['letterboxing inconsistency', 'stretching artifacts'],
                'fix': 'aspect_ratio_normalization'
            })
        
        return issues


class VideoNormalizationPipeline:
    """Pipeline for normalizing mixed video formats to prevent concatenation artifacts."""
    
    def __init__(self, format_analyzer: VideoFormatAnalyzer):
        self.format_analyzer = format_analyzer
    
    def normalize_video_clips(self, video_clips: List[Any], target_format: Dict[str, Any]) -> List[Any]:
        """Normalize all clips to consistent format to prevent artifacts.
        
        Args:
            video_clips: List of VideoFileClip instances with mixed formats
            target_format: Target format specification from format analyzer
            
        Returns:
            List of normalized VideoFileClip instances
        """
        if not target_format.get('requires_normalization', False):
            print("Format normalization: No normalization required, formats are consistent")
            return video_clips
        
        print(f"Format normalization: Normalizing {len(video_clips)} clips to {target_format['target_width']}x{target_format['target_height']} @ {target_format['target_fps']}fps")
        
        normalized_clips = []
        
        for i, clip in enumerate(video_clips):
            try:
                normalized_clip = self._normalize_single_clip(clip, target_format)
                normalized_clips.append(normalized_clip)
                print(f"Normalized clip {i+1}/{len(video_clips)}: {clip.w}x{clip.h}@{clip.fps}fps -> {normalized_clip.w}x{normalized_clip.h}@{normalized_clip.fps}fps")
                
            except Exception as e:
                print(f"Warning: Failed to normalize clip {i+1}: {e}")
                # Use original clip if normalization fails
                normalized_clips.append(clip)
        
        return normalized_clips
    
    def _normalize_single_clip(self, clip, target_format: Dict[str, Any]):
        """Normalize a single clip to target format."""
        normalized_clip = clip
        
        # Step 1: Resolution normalization with aspect ratio preservation
        if clip.w != target_format['target_width'] or clip.h != target_format['target_height']:
            normalized_clip = self._resize_with_aspect_preservation(
                normalized_clip, 
                target_format['target_width'], 
                target_format['target_height']
            )
        
        # Step 2: Frame rate normalization
        if abs(clip.fps - target_format['target_fps']) > 0.1:
            normalized_clip = normalized_clip.with_fps(target_format['target_fps'])
        
        return normalized_clip
    
    def _resize_with_aspect_preservation(self, clip, target_width: int, target_height: int):
        """Resize clip while preserving aspect ratio using letterbox/pillarbox."""
        # Calculate scaling to fit within target dimensions
        width_scale = target_width / clip.w
        height_scale = target_height / clip.h
        scale = min(width_scale, height_scale)
        
        # Calculate new dimensions
        new_width = int(clip.w * scale)
        new_height = int(clip.h * scale)
        
        # Resize to fit within target dimensions
        resized_clip = clip.resized((new_width, new_height))
        
        # Add padding to reach exact target dimensions (letterbox/pillarbox)
        if new_width != target_width or new_height != target_height:
            # Create a black background at target size
            from moviepy.video.VideoClip import ColorClip
            background = ColorClip(size=(target_width, target_height), color=(0,0,0), duration=resized_clip.duration)
            
            # Calculate centering position
            x_pos = (target_width - new_width) // 2
            y_pos = (target_height - new_height) // 2
            
            # Composite the resized clip onto the background
            normalized_clip = CompositeVideoClip([
                background,
                resized_clip.with_position((x_pos, y_pos))
            ])
            
            return normalized_clip
        
        return resized_clip


def load_video_segment(clip_data: Dict[str, Any], video_cache: VideoCache) -> Optional[Tuple[Dict[str, Any], Any]]:
    """Load a single video segment in parallel processing.
    
    Args:
        clip_data: Dictionary containing video_file, start, end times
        video_cache: Shared video cache instance
        
    Returns:
        Tuple of (clip_data, video_segment) or None if failed
    """
    try:
        # Get cached video (thread-safe)
        source_video = video_cache.get_or_load(clip_data['video_file'])
        
        # Extract the specific segment using safe compatibility method
        segment = subclip_safely(source_video, clip_data['start'], clip_data['end'])
        
        return (clip_data, segment)
        
    except Exception as e:
        print(f"Warning: Failed to load clip {clip_data['video_file']}: {str(e)}")
        return None


def load_video_clips_parallel(sorted_clips: List[Dict[str, Any]], 
                             progress_callback: Optional[callable] = None,
                             max_workers: int = 6) -> Tuple[List[Any], VideoCache, List[int]]:
    """Load video clips in parallel with intelligent caching.
    
    Args:
        sorted_clips: List of clip data dictionaries sorted by beat position
        progress_callback: Optional callback for progress updates
        max_workers: Maximum number of parallel workers (default: 6)
        
    Returns:
        Tuple of (video_clips_list, video_cache, failed_indices) for resource management
        
    Raises:
        RuntimeError: If no clips could be loaded successfully
    """
    def report_progress(step: str, progress: float):
        """Helper to report progress if callback provided."""
        if progress_callback:
            progress_callback(step, progress)
    
    if not sorted_clips:
        raise ValueError("No clips provided for loading")
    
    # Initialize cache and results
    video_cache = VideoCache()
    video_clips = []
    clip_mapping = {}  # Map to maintain order
    failed_indices = []  # Track which clips failed to load
    
    # Calculate optimal worker count (limit to prevent resource exhaustion)
    optimal_workers = min(max_workers, len(sorted_clips), 8)
    
    report_progress(f"Loading {len(sorted_clips)} clips with {optimal_workers} workers", 0.1)
    
    try:
        with ThreadPoolExecutor(max_workers=optimal_workers) as executor:
            # Submit all clip loading tasks
            future_to_index = {}
            for i, clip_data in enumerate(sorted_clips):
                future = executor.submit(load_video_segment, clip_data, video_cache)
                future_to_index[future] = i
            
            # Collect results as they complete
            completed_count = 0
            for future in as_completed(future_to_index):
                completed_count += 1
                index = future_to_index[future]
                
                try:
                    result = future.result()
                    if result is not None:
                        clip_data, segment = result
                        clip_mapping[index] = segment
                    else:
                        failed_indices.append(index)
                    
                except Exception as e:
                    print(f"Warning: Future failed for clip {index}: {str(e)}")
                    failed_indices.append(index)
                
                # Update progress
                progress = 0.1 + (0.6 * completed_count / len(sorted_clips))
                report_progress(f"Loaded {completed_count}/{len(sorted_clips)} clips", progress)
    
    except Exception as e:
        # Clean up cache on error
        video_cache.clear()
        raise RuntimeError(f"Parallel video loading failed: {str(e)}")
    
    # Reconstruct clips in original order, skipping failed ones
    for i in range(len(sorted_clips)):
        if i in clip_mapping:
            video_clips.append(clip_mapping[i])
    
    if not video_clips:
        video_cache.clear()
        raise RuntimeError("No video clips could be loaded successfully")
    
    # Report failed clips
    if failed_indices:
        print(f"Warning: {len(failed_indices)} clips failed to load (indices: {failed_indices})")
    
    report_progress(f"Successfully loaded {len(video_clips)} clips", 0.7)
    
    # Log cache statistics
    cached_files = video_cache.get_cached_paths()
    print(f"Video cache: {len(cached_files)} unique files loaded")
    
    return video_clips, video_cache, failed_indices


def check_moviepy_api_compatibility():
    """Comprehensive MoviePy API compatibility analysis for version 2.1.2+ changes.
    
    Returns:
        Dict with complete API mapping and compatibility information
    """
    import inspect
    
    # Handle import structure changes in MoviePy 2.1.2
    try:
        # Try new import structure first (MoviePy 2.1.2+)
        from moviepy import VideoFileClip, AudioFileClip, concatenate_videoclips
        import_pattern = 'new'  # from moviepy import ...
    except ImportError:
        try:
            # Fallback to legacy import structure (MoviePy < 2.1.2)
            from moviepy.editor import VideoFileClip, AudioFileClip, concatenate_videoclips
            import_pattern = 'legacy'  # from moviepy.editor import ...
        except ImportError:
            raise RuntimeError("Could not import MoviePy with either import pattern")
    
    # Test method availability on dummy instances
    video_dummy = VideoFileClip.__new__(VideoFileClip)
    audio_dummy = AudioFileClip.__new__(AudioFileClip)
    
    compatibility = {
        'import_pattern': import_pattern,
        'version_detected': 'new' if import_pattern == 'new' else 'legacy',
        
        # Method mappings: old_name -> new_name
        'method_mappings': {
            # Clip manipulation methods
            'subclip': 'subclipped' if hasattr(video_dummy, 'subclipped') else 'subclip',
            
            # Audio attachment methods  
            'set_audio': 'with_audio' if hasattr(video_dummy, 'with_audio') else 'set_audio',
            
            # Other common method patterns that might have changed
            'set_duration': 'with_duration' if hasattr(video_dummy, 'with_duration') else 'set_duration',
            'set_position': 'with_position' if hasattr(video_dummy, 'with_position') else 'set_position',
            'set_start': 'with_start' if hasattr(video_dummy, 'with_start') else 'set_start',
        },
        
        # Method availability matrix
        'methods': {
            'video_clip': {
                'subclip': hasattr(video_dummy, 'subclip'),
                'subclipped': hasattr(video_dummy, 'subclipped'),
                'set_audio': hasattr(video_dummy, 'set_audio'),
                'with_audio': hasattr(video_dummy, 'with_audio'),
                'write_videofile': hasattr(video_dummy, 'write_videofile'),
            },
            'audio_clip': {
                'subclip': hasattr(audio_dummy, 'subclip'),
                'subclipped': hasattr(audio_dummy, 'subclipped'),
            }
        }
    }
    
    # Analyze write_videofile parameters
    try:
        write_sig = inspect.signature(video_dummy.write_videofile)
        compatibility['write_videofile_params'] = list(write_sig.parameters.keys())
    except:
        compatibility['write_videofile_params'] = ['filename']
    
    return compatibility


def attach_audio_safely(video_clip, audio_clip, compatibility_info=None):
    """Safely attach audio to video using available API.
    
    Args:
        video_clip: VideoClip to attach audio to
        audio_clip: AudioClip to attach
        compatibility_info: Result from check_moviepy_api_compatibility()
        
    Returns:
        VideoClip with audio attached
    """
    if compatibility_info is None:
        compatibility_info = check_moviepy_api_compatibility()
    
    # Get the correct method name for audio attachment
    method_name = compatibility_info['method_mappings']['set_audio']
    
    try:
        # Use the dynamically determined method
        method = getattr(video_clip, method_name)
        return method(audio_clip)
    except AttributeError:
        # Final fallback: try both known methods
        for method_name in ['with_audio', 'set_audio']:
            if hasattr(video_clip, method_name):
                method = getattr(video_clip, method_name)
                return method(audio_clip)
        
        raise RuntimeError(f"Could not find audio attachment method on video clip")

def subclip_safely(clip, start_time, end_time=None, compatibility_info=None):
    """Safely create subclip using available API (subclip vs subclipped).
    
    Args:
        clip: VideoClip or AudioClip to extract from
        start_time: Start time in seconds
        end_time: End time in seconds (None for rest of clip)
        compatibility_info: Result from check_moviepy_api_compatibility()
        
    Returns:
        New clip with specified time range
    """
    if compatibility_info is None:
        compatibility_info = check_moviepy_api_compatibility()
    
    # Get the correct method name for subclip operation
    method_name = compatibility_info['method_mappings']['subclip']
    
    try:
        # Use the dynamically determined method
        method = getattr(clip, method_name)
        if end_time is not None:
            return method(start_time, end_time)
        else:
            return method(start_time)
    except AttributeError:
        # Final fallback: try both known methods
        for method_name in ['subclipped', 'subclip']:
            if hasattr(clip, method_name):
                method = getattr(clip, method_name)
                if end_time is not None:
                    return method(start_time, end_time)
                else:
                    return method(start_time)
        
        raise RuntimeError(f"Could not find subclip method on clip type {type(clip)}")


def import_moviepy_safely():
    """Safely import MoviePy classes handling import structure changes.
    
    Returns:
        Tuple of (VideoFileClip, AudioFileClip, concatenate_videoclips, CompositeVideoClip)
    """
    try:
        # Try new import structure first (MoviePy 2.1.2+)
        from moviepy import VideoFileClip, AudioFileClip, concatenate_videoclips
        try:
            from moviepy.video.compositing.CompositeVideoClip import CompositeVideoClip
        except ImportError:
            from moviepy import CompositeVideoClip
        return VideoFileClip, AudioFileClip, concatenate_videoclips, CompositeVideoClip
    except ImportError:
        try:
            # Fallback to legacy import structure (MoviePy < 2.1.2) 
            from moviepy.editor import VideoFileClip, AudioFileClip, concatenate_videoclips, CompositeVideoClip
            return VideoFileClip, AudioFileClip, concatenate_videoclips, CompositeVideoClip
        except ImportError:
            raise RuntimeError("Could not import MoviePy with either import pattern. Please check MoviePy installation.")


def write_videofile_safely(video_clip, output_path, compatibility_info=None, **kwargs):
    """Safely write video file with parameter compatibility checking.
    
    Args:
        video_clip: VideoClip to write
        output_path: Output file path
        compatibility_info: Result from check_moviepy_api_compatibility()
        **kwargs: Parameters to pass to write_videofile
        
    Returns:
        None
    """
    if compatibility_info is None:
        compatibility_info = check_moviepy_api_compatibility()
    
    available_params = compatibility_info['write_videofile_params']
    
    # Filter kwargs to only include supported parameters
    safe_kwargs = {}
    for key, value in kwargs.items():
        if key in available_params:
            safe_kwargs[key] = value
        else:
            print(f"Warning: Parameter '{key}' not supported in this MoviePy version, skipping")
    
    try:
        video_clip.write_videofile(output_path, **safe_kwargs)
    except Exception as e:
        raise RuntimeError(f"Failed to write video file: {str(e)}")


class ClipTimeline:
    """Represents the timeline of clips matched to beats."""
    
    def __init__(self):
        self.clips: List[Dict[str, Any]] = []
        
    def add_clip(self, video_file: str, start: float, end: float, 
                 beat_position: float, score: float):
        """Add a clip to the timeline."""
        self.clips.append({
            'video_file': video_file,
            'start': start,
            'end': end,
            'beat_position': beat_position,
            'score': score,
            'duration': end - start
        })
        
    def export_json(self, file_path: str):
        """Export timeline as JSON for debugging."""
        with open(file_path, 'w') as f:
            json.dump(self.clips, f, indent=2)
            
    def get_total_duration(self) -> float:
        """Get total duration of all clips."""
        return sum(clip['duration'] for clip in self.clips)
    
    def get_clips_sorted_by_beat(self) -> List[Dict[str, Any]]:
        """Get clips sorted by their beat position."""
        return sorted(self.clips, key=lambda x: x['beat_position'])
    
    def get_summary_stats(self) -> Dict[str, Any]:
        """Get summary statistics about the timeline."""
        if not self.clips:
            return {
                'total_clips': 0,
                'total_duration': 0.0,
                'avg_score': 0.0,
                'unique_videos': 0,
                'score_range': (0.0, 0.0)
            }
        
        scores = [clip['score'] for clip in self.clips]
        unique_videos = len(set(clip['video_file'] for clip in self.clips))
        
        return {
            'total_clips': len(self.clips),
            'total_duration': self.get_total_duration(),
            'avg_score': sum(scores) / len(scores),
            'unique_videos': unique_videos,
            'score_range': (min(scores), max(scores)),
            'duration_range': (
                min(clip['duration'] for clip in self.clips),
                max(clip['duration'] for clip in self.clips)
            )
        }
    
    def validate_timeline(self, song_duration: float = None) -> Dict[str, Any]:
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
            return {'valid': False, 'issues': issues, 'warnings': warnings}
        
        # Sort clips by beat position for analysis
        sorted_clips = self.get_clips_sorted_by_beat()
        
        # Check for very short clips
        short_clips = [clip for clip in self.clips if clip['duration'] < 0.5]
        if short_clips:
            warnings.append(f"{len(short_clips)} clips are very short (<0.5s)")
        
        # Check for very long clips
        long_clips = [clip for clip in self.clips if clip['duration'] > 8.0]
        if long_clips:
            warnings.append(f"{len(long_clips)} clips are very long (>8s)")
        
        # Check score distribution
        scores = [clip['score'] for clip in self.clips]
        avg_score = sum(scores) / len(scores)
        if avg_score < 50:
            warnings.append(f"Average clip quality is low: {avg_score:.1f}")
        
        # Check video variety
        unique_videos = len(set(clip['video_file'] for clip in self.clips))
        if len(self.clips) > 5 and unique_videos == 1:
            warnings.append("All clips are from the same video - low variety")
        
        # Check timeline coverage if song duration provided
        if song_duration:
            timeline_span = sorted_clips[-1]['beat_position'] - sorted_clips[0]['beat_position']
            coverage = timeline_span / song_duration
            if coverage < 0.8:
                warnings.append(f"Timeline covers only {coverage*100:.1f}% of song")
        
        return {
            'valid': len(issues) == 0,
            'issues': issues,
            'warnings': warnings,
            'stats': self.get_summary_stats()
        }


def match_clips_to_beats(video_chunks: List[VideoChunk], beats: List[float], 
                        allowed_durations: List[float], pattern: str = 'balanced',
                        musical_start_time: float = 0.0) -> ClipTimeline:
    """Match video chunks to beat grid using variety patterns with musical intelligence.
    
    Args:
        video_chunks: List of scored video chunks
        beats: List of beat timestamps in seconds (compensated and filtered)
        allowed_durations: List of musically appropriate durations
        pattern: Variety pattern to use ('energetic', 'buildup', 'balanced', 'dramatic')
        musical_start_time: First significant beat timestamp (skip intro/buildup)
        
    Returns:
        ClipTimeline object with matched clips starting from musical content
    """
    if not video_chunks or not beats or len(beats) < 2:
        return ClipTimeline()
    
    # MUSICAL INTELLIGENCE: Filter beats to start from actual musical content
    # This fixes the 1-2 second intro misalignment issue
    effective_beats = [b for b in beats if b >= musical_start_time] if musical_start_time > 0 else beats
    
    if len(effective_beats) < 2:
        # Fallback to all beats if musical start filtering leaves too few
        effective_beats = beats
    
    # Calculate beat interval (average time between beats)
    beat_intervals = [effective_beats[i+1] - effective_beats[i] for i in range(len(effective_beats)-1)]
    avg_beat_interval = sum(beat_intervals) / len(beat_intervals)
    
    # Apply variety pattern to get beat multipliers
    total_beats = len(effective_beats) - 1  # Don't count the last beat as start of a clip
    beat_multipliers = apply_variety_pattern(pattern, total_beats)
    
    # Convert beat multipliers to target durations
    target_durations = [multiplier * avg_beat_interval for multiplier in beat_multipliers]
    
    # Estimate total clips needed
    estimated_clips = len(target_durations)
    
    # Select best clips with variety (request more than needed for flexibility)
    selected_clips = select_best_clips(video_chunks, 
                                     target_count=min(estimated_clips * 2, len(video_chunks)),
                                     variety_factor=0.3)
    
    timeline = ClipTimeline()
    current_beat_index = 0
    used_clips = set()  # Track used clips to avoid repetition
    
    for i, target_duration in enumerate(target_durations):
        if current_beat_index >= len(effective_beats):
            break
            
        # Find best matching clip for this target duration
        best_clip = None
        best_fit_score = -1
        
        for clip in selected_clips:
            if id(clip) in used_clips:
                continue
                
            # Calculate fit score based on:
            # 1. How close clip duration is to target duration
            # 2. Clip quality score
            # 3. Whether clip can be trimmed to fit exactly
            
            duration_fit = _calculate_duration_fit(clip.duration, target_duration, allowed_durations)
            if duration_fit < 0:  # Clip can't be used for this duration
                continue
                
            # Combined score: 70% quality, 30% duration fit
            fit_score = 0.7 * (clip.score / 100.0) + 0.3 * duration_fit
            
            if fit_score > best_fit_score:
                best_fit_score = fit_score
                best_clip = clip
        
        if best_clip is None:
            # No suitable clip found, skip this position
            current_beat_index += beat_multipliers[i]
            continue
            
        # Mark clip as used
        used_clips.add(id(best_clip))
        
        # Determine actual clip timing
        beat_position = effective_beats[current_beat_index]
        clip_start, clip_end, clip_duration = _fit_clip_to_duration(
            best_clip, target_duration, allowed_durations
        )
        
        # Add to timeline
        timeline.add_clip(
            video_file=best_clip.video_path,
            start=clip_start,
            end=clip_end,
            beat_position=beat_position,
            score=best_clip.score
        )
        
        # Move to next beat position
        current_beat_index += beat_multipliers[i]
    
    return timeline


def _calculate_duration_fit(clip_duration: float, target_duration: float, 
                           allowed_durations: List[float]) -> float:
    """Calculate how well a clip duration fits the target duration.
    
    Args:
        clip_duration: Duration of the video clip
        target_duration: Desired duration for this position
        allowed_durations: List of musically appropriate durations
        
    Returns:
        Fit score between 0.0 and 1.0, or -1 if clip can't be used
    """
    # Check if target duration is in allowed durations (with small tolerance)
    duration_allowed = False
    for allowed in allowed_durations:
        if abs(target_duration - allowed) < 0.1:
            duration_allowed = True
            break
    
    if not duration_allowed:
        return -1  # Target duration is not musically appropriate
    
    # Perfect match
    if abs(clip_duration - target_duration) < 0.1:
        return 1.0
    
    # Clip is longer than target - can be trimmed
    if clip_duration > target_duration:
        # Prefer clips that are close to target but slightly longer
        excess = clip_duration - target_duration
        if excess <= 2.0:  # Can trim up to 2 seconds
            return 1.0 - (excess / 4.0)  # Gentle penalty for trimming
        else:
            return 0.3  # Heavy penalty for lots of trimming
    
    # Clip is shorter than target
    else:
        shortage = target_duration - clip_duration
        if shortage <= 0.5:  # Small shortage is acceptable
            return 0.8 - (shortage / 1.0)
        else:
            return -1  # Too short, can't use


def _fit_clip_to_duration(clip: VideoChunk, target_duration: float, 
                         allowed_durations: List[float]) -> Tuple[float, float, float]:
    """Fit a clip to the target duration by trimming if necessary.
    
    Args:
        clip: Video chunk to fit
        target_duration: Desired duration
        allowed_durations: List of allowed durations
        
    Returns:
        Tuple of (start_time, end_time, actual_duration)
    """
    if clip.duration <= target_duration + 0.1:
        # Clip fits as-is
        return clip.start_time, clip.end_time, clip.duration
    
    # Clip needs trimming - trim from the end to preserve the beginning
    new_end_time = clip.start_time + target_duration
    
    # Make sure we don't exceed the original clip bounds
    new_end_time = min(new_end_time, clip.end_time)
    actual_duration = new_end_time - clip.start_time
    
    return clip.start_time, new_end_time, actual_duration


def select_best_clips(video_chunks: List[VideoChunk], target_count: int, 
                     variety_factor: float = 0.3) -> List[VideoChunk]:
    """Select best clips ensuring variety in source videos.
    
    Args:
        video_chunks: List of all available video chunks
        target_count: Number of clips to select
        variety_factor: Weight for variety vs. quality (0.0 = only quality, 1.0 = only variety)
        
    Returns:
        List of selected VideoChunk objects
    """
    if not video_chunks:
        return []
    
    if target_count <= 0:
        return []
        
    if len(video_chunks) <= target_count:
        return video_chunks.copy()
    
    # Group clips by video file for variety management
    clips_by_video = {}
    for chunk in video_chunks:
        if chunk.video_path not in clips_by_video:
            clips_by_video[chunk.video_path] = []
        clips_by_video[chunk.video_path].append(chunk)
    
    # Sort clips within each video by score (descending)
    for video_path in clips_by_video:
        clips_by_video[video_path].sort(key=lambda x: x.score, reverse=True)
    
    selected_clips = []
    
    if variety_factor >= 0.9:
        # High variety: Round-robin selection from each video
        video_paths = list(clips_by_video.keys())
        video_index = 0
        
        while len(selected_clips) < target_count:
            video_path = video_paths[video_index % len(video_paths)]
            
            # Find next non-overlapping clip from this video
            available_clips = clips_by_video[video_path]
            for clip in available_clips:
                if clip not in selected_clips and not _clips_overlap(clip, selected_clips):
                    selected_clips.append(clip)
                    break
            
            video_index += 1
            
            # Safety check: if we've tried all videos and can't find more clips
            if video_index > len(video_paths) * 10:
                break
                
    elif variety_factor <= 0.1:
        # High quality: Just take the best clips regardless of source
        all_clips_sorted = sorted(video_chunks, key=lambda x: x.score, reverse=True)
        for clip in all_clips_sorted:
            if len(selected_clips) >= target_count:
                break
            if not _clips_overlap(clip, selected_clips):
                selected_clips.append(clip)
                
    else:
        # Balanced approach: Weighted selection
        # Calculate how many clips per video (with some variety)
        num_videos = len(clips_by_video)
        base_clips_per_video = max(1, target_count // num_videos)
        remaining_clips = target_count - (base_clips_per_video * num_videos)
        
        # First pass: Get base clips from each video (highest quality)
        for video_path in clips_by_video:
            clips_from_video = 0
            for clip in clips_by_video[video_path]:
                if clips_from_video >= base_clips_per_video:
                    break
                if not _clips_overlap(clip, selected_clips):
                    selected_clips.append(clip)
                    clips_from_video += 1
        
        # Second pass: Fill remaining slots with highest quality clips
        if remaining_clips > 0:
            all_remaining_clips = []
            for video_path in clips_by_video:
                for clip in clips_by_video[video_path][base_clips_per_video:]:
                    if clip not in selected_clips:
                        all_remaining_clips.append(clip)
            
            all_remaining_clips.sort(key=lambda x: x.score, reverse=True)
            
            for clip in all_remaining_clips:
                if len(selected_clips) >= target_count:
                    break
                if not _clips_overlap(clip, selected_clips):
                    selected_clips.append(clip)
    
    return selected_clips[:target_count]


def _clips_overlap(clip: VideoChunk, existing_clips: List[VideoChunk], 
                  min_gap: float = 1.0) -> bool:
    """Check if a clip overlaps with any existing clips from the same video.
    
    Args:
        clip: Clip to check
        existing_clips: List of already selected clips
        min_gap: Minimum gap required between clips from same video (seconds)
        
    Returns:
        True if clip overlaps with any existing clip from same video
    """
    for existing in existing_clips:
        if existing.video_path == clip.video_path:
            # Check for overlap or too close proximity
            if (clip.start_time < existing.end_time + min_gap and 
                clip.end_time > existing.start_time - min_gap):
                return True
    return False


def apply_variety_pattern(pattern_name: str, beat_count: int) -> List[int]:
    """Apply variety pattern to determine clip lengths.
    
    Args:
        pattern_name: Name of variety pattern to use
        beat_count: Total number of beats to fill
        
    Returns:
        List of beat multipliers for each clip
    """
    if pattern_name not in VARIETY_PATTERNS:
        pattern_name = 'balanced'
        
    pattern = VARIETY_PATTERNS[pattern_name]
    result = []
    pattern_index = 0
    remaining_beats = beat_count
    
    while remaining_beats > 0:
        multiplier = pattern[pattern_index % len(pattern)]
        if multiplier <= remaining_beats:
            result.append(multiplier)
            remaining_beats -= multiplier
        else:
            result.append(remaining_beats)
            remaining_beats = 0
        pattern_index += 1
        
    return result


def render_video(timeline: ClipTimeline, audio_file: str, output_path: str,
                progress_callback: Optional[callable] = None) -> str:
    """Render final video with music synchronization and frame-accurate audio handling.
    
    This version includes fixes for MoviePy 2.2.1 audio-video sync issues:
    - Frame-accurate audio trimming to prevent 1-2 frame cuts
    - Precise duration validation before concatenation
    - Enhanced sync debugging and validation
    
    Args:
        timeline: ClipTimeline with all clips and timing
        audio_file: Path to music file
        output_path: Path for output video
        progress_callback: Optional callback for progress updates
        
    Returns:
        Path to rendered video file
        
    Raises:
        RuntimeError: If rendering fails
    """
    import os
    
    # Use safe import handling MoviePy version differences
    VideoFileClip, AudioFileClip, concatenate_videoclips, CompositeVideoClip = import_moviepy_safely()
    
    def report_progress(step: str, progress: float):
        """Helper to report progress if callback provided."""
        if progress_callback:
            progress_callback(step, progress)
    
    if not timeline.clips:
        raise ValueError("Timeline is empty - no clips to render")
    
    if not os.path.exists(audio_file):
        raise FileNotFoundError(f"Audio file not found: {audio_file}")
    
    # Check API compatibility at start
    compatibility_info = check_moviepy_api_compatibility()
    print(f"Debug: MoviePy API compatibility detected: {compatibility_info['version_detected']}")
    print(f"Debug: Method mappings - subclip: {compatibility_info['method_mappings']['subclip']}, set_audio: {compatibility_info['method_mappings']['set_audio']}")
    
    report_progress("Loading clips", 0.1)
    
    try:
        # Load audio track for duration calculation
        audio_clip = AudioFileClip(audio_file)
        original_audio_duration = audio_clip.duration
        print(f"Debug: Original audio duration: {original_audio_duration:.6f}s")
        
        # Get clips sorted by beat position
        sorted_clips = timeline.get_clips_sorted_by_beat()
        print(f"Debug: Timeline has {len(sorted_clips)} clips to load")
        
        # CRITICAL FIX: Pre-filter timeline to prevent index synchronization bugs
        # We need to handle potential failures BEFORE loading to maintain consistency
        
        # Load video clips in parallel with intelligent caching
        video_clips, video_cache, failed_indices = load_video_clips_parallel(
            sorted_clips, 
            progress_callback=progress_callback,
            max_workers=6  # Optimal for video I/O without overwhelming system
        )
        
        if not video_clips:
            raise RuntimeError("No video clips could be loaded successfully")
        
        print(f"Debug: Loaded {len(video_clips)} video clips successfully")
        
        # SYNCHRONIZATION FIX: Adjust timeline IMMEDIATELY after loading to maintain index consistency
        if failed_indices:
            print(f"SYNC FIX: Removing {len(failed_indices)} failed clips from timeline")
            # Create new timeline with successful clips only, maintaining order
            original_clips = timeline.clips.copy()
            successful_clips = []
            
            for i, clip in enumerate(original_clips):
                if i not in failed_indices:
                    successful_clips.append(clip)
            
            timeline.clips = successful_clips
            print(f"Debug: Timeline synchronized - {len(original_clips)} -> {len(timeline.clips)} clips")
        
        # Verify clip count consistency
        if len(video_clips) != len(timeline.clips):
            raise RuntimeError(f"Clip count mismatch: {len(video_clips)} loaded vs {len(timeline.clips)} in timeline")
        
        print(f"Debug: Final clip count - video_clips: {len(video_clips)}, timeline.clips: {len(timeline.clips)}")
        
        # FORMAT ANALYSIS & NORMALIZATION: Critical fix for visual artifacts
        format_analyzer = VideoFormatAnalyzer()
        target_format = format_analyzer.find_dominant_format(video_clips)
        
        # Detect format compatibility issues
        format_issues = format_analyzer.detect_format_compatibility_issues(video_clips)
        if format_issues:
            print("FORMAT ISSUES DETECTED:")
            for issue in format_issues:
                print(f"  - {issue['type']}: {issue['description']}")
                print(f"    Artifacts: {', '.join(issue['artifacts'])}")
        
        # Apply format normalization to prevent artifacts
        normalization_pipeline = VideoNormalizationPipeline(format_analyzer)
        normalized_video_clips = normalization_pipeline.normalize_video_clips(video_clips, target_format)
        
        # ENHANCED: Validate normalized clip durations before concatenation
        total_expected_duration = 0
        for i, clip in enumerate(normalized_video_clips):
            clip_duration = clip.duration
            expected_duration = timeline.clips[i]['duration']
            print(f"Debug: Normalized clip {i+1}: actual={clip_duration:.6f}s, expected={expected_duration:.6f}s")
            
            # Check for significant duration discrepancies
            duration_diff = abs(clip_duration - expected_duration)
            if duration_diff > 0.1:  # More than 100ms difference
                print(f"Warning: Clip {i+1} duration mismatch: {duration_diff:.6f}s difference")
            
            total_expected_duration += clip_duration
        
        print(f"Debug: Total expected video duration: {total_expected_duration:.6f}s")
        
        report_progress("Concatenating video", 0.6)
        
        # SMART CONCATENATION: Choose method based on format consistency
        if target_format.get('requires_normalization', False):
            concatenation_method = "compose"  # Better for mixed formats after normalization
            print(f"Debug: Using 'compose' method for {len(normalized_video_clips)} normalized clips")
        else:
            concatenation_method = "chain"  # Faster for consistent formats
            print(f"Debug: Using 'chain' method for {len(normalized_video_clips)} consistent clips")
        
        final_video = concatenate_videoclips(normalized_video_clips, method=concatenation_method)
        
        actual_video_duration = final_video.duration
        print(f"Debug: Concatenation successful, final video duration: {actual_video_duration:.6f}s")
        
        # Validate concatenation didn't introduce timing errors
        duration_error = abs(actual_video_duration - total_expected_duration)
        if duration_error > 0.05:  # More than 50ms error
            print(f"Warning: Concatenation timing error: {duration_error:.6f}s discrepancy")
        
        # ENHANCED AUDIO HANDLING: Frame-accurate trimming with sync compensation
        report_progress("Preparing audio", 0.75)
        
        # Calculate precise audio duration needed (accounting for MoviePy sync bugs)
        target_audio_duration = actual_video_duration
        
        # CRITICAL FIX: Use TARGET FPS for accurate audio sync calculation (not final video FPS)
        target_fps = target_format['target_fps']
        frame_duration = 1.0 / target_fps
        sync_buffer = frame_duration * 2  # 2-frame buffer to prevent cutoff
        
        print(f"Debug: Target FPS: {target_fps}, Frame duration: {frame_duration:.6f}s")
        print(f"Debug: Adding sync buffer: {sync_buffer:.6f}s to prevent audio cutoff")
        
        # ENHANCED AUDIO CALCULATION: Frame-accurate duration for mixed formats
        # Calculate total frames using TARGET fps (not varying clip fps)
        total_frames = sum(int(clip.duration * target_fps) for clip in normalized_video_clips)
        frame_accurate_video_duration = total_frames / target_fps
        
        print(f"Debug: Frame-accurate calculation: {total_frames} frames @ {target_fps}fps = {frame_accurate_video_duration:.6f}s")
        print(f"Debug: Concatenation duration: {actual_video_duration:.6f}s")
        
        # Use the more accurate calculation for audio sync
        precise_video_duration = frame_accurate_video_duration
        
        # Prepare audio with FRAME-ACCURATE timing to prevent cutoff
        if original_audio_duration > precise_video_duration:
            # Trim audio to match frame-accurate video duration, plus sync buffer
            audio_end_time = min(precise_video_duration + sync_buffer, original_audio_duration)
            print(f"Debug: FRAME-ACCURATE audio trim: {original_audio_duration:.6f}s -> {audio_end_time:.6f}s")
            print(f"Debug: Audio buffer added: {sync_buffer:.6f}s to prevent cutoff")
            trimmed_audio = subclip_safely(audio_clip, 0, audio_end_time, compatibility_info)
        else:
            # Audio is shorter than video - use full audio
            print(f"Debug: Audio ({original_audio_duration:.6f}s) shorter than video ({precise_video_duration:.6f}s)")
            trimmed_audio = audio_clip
        
        final_audio_duration = trimmed_audio.duration
        print(f"Debug: Final audio duration: {final_audio_duration:.6f}s")
        
        # ENHANCED: Final sync validation
        sync_difference = abs(final_audio_duration - actual_video_duration)
        print(f"Debug: Audio-video sync difference: {sync_difference:.6f}s")
        
        if sync_difference > 0.1:  # More than 100ms difference
            print(f"Warning: Significant audio-video sync difference: {sync_difference:.6f}s")
        
        # Attach audio using version-compatible method
        report_progress("Attaching audio", 0.8)
        print(f"Debug: Attaching audio using method: {compatibility_info['method_mappings']['set_audio']}")
        final_video = attach_audio_safely(final_video, trimmed_audio, compatibility_info)
        
        report_progress("Rendering final video", 0.85)
        
        # Get optimal codec settings with format-specific enhancements
        moviepy_params, ffmpeg_params = detect_optimal_codec_settings()
        
        # ENHANCED FFMPEG PARAMETERS: Add format consistency parameters  
        format_consistency_params = [
            '-pix_fmt', 'yuv420p',  # Consistent color format
            '-vsync', 'cfr',        # Constant frame rate conversion
            '-async', '1',          # Audio sync parameter
        ]
        
        # Add resolution/fps parameters if normalization was applied
        if target_format.get('requires_normalization', False):
            format_consistency_params.extend([
                '-r', str(target_format['target_fps']),  # Force target frame rate
                '-s', f"{target_format['target_width']}x{target_format['target_height']}"  # Force resolution
            ])
        
        # Combine all FFmpeg parameters
        enhanced_ffmpeg_params = ffmpeg_params + format_consistency_params
        print(f"Debug: Enhanced FFmpeg params: {enhanced_ffmpeg_params}")
        
        # Create output directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        print(f"Debug: Rendering final video to: {output_path}")
        
        # Prepare parameters for version-safe writing with enhanced format consistency
        write_params = {
            **moviepy_params,
            'ffmpeg_params': enhanced_ffmpeg_params,  # Use enhanced parameters
            'temp_audiofile': 'temp-audio.m4a',
            'remove_temp': True,
            'fps': target_format['target_fps'],  # Use TARGET fps instead of hardcoded 24
            'logger': None  # Suppress MoviePy logging
        }
        
        # ENHANCED: Add audio-specific parameters for better sync
        write_params.update({
            'audio_fps': 44100,  # Standard audio sample rate
            'audio_codec': 'aac',  # Compatible audio codec
            'audio_bitrate': '128k'  # Good quality audio bitrate
        })
        
        print(f"Debug: Write parameters: {list(write_params.keys())}")
        
        # Render with version-compatible parameter checking
        write_videofile_safely(final_video, output_path, compatibility_info, **write_params)
        
        print(f"Debug: Rendering completed successfully")
        
        # ENHANCED: Post-render validation
        if os.path.exists(output_path):
            # Quick validation of output file
            try:
                output_clip = VideoFileClip(output_path)
                output_duration = output_clip.duration
                output_has_audio = output_clip.audio is not None
                output_clip.close()
                
                print(f"Debug: Output validation - Duration: {output_duration:.6f}s, Has audio: {output_has_audio}")
                
                # Check for major duration discrepancies
                duration_loss = abs(output_duration - actual_video_duration)
                if duration_loss > 0.2:  # More than 200ms loss
                    print(f"Warning: Significant duration loss in output: {duration_loss:.6f}s")
                
            except Exception as validation_error:
                print(f"Warning: Could not validate output file: {validation_error}")
        
        # Clean up resources
        final_video.close()
        audio_clip.close()
        if 'trimmed_audio' in locals() and trimmed_audio != audio_clip:
            trimmed_audio.close()
        video_cache.clear()  # Clean up cached videos
        
        report_progress("Rendering complete", 1.0)
        
        if not os.path.exists(output_path):
            raise RuntimeError(f"Output file was not created: {output_path}")
        
        return output_path
        
    except Exception as e:
        # Clean up any resources
        try:
            if 'final_video' in locals():
                final_video.close()
            if 'audio_clip' in locals():
                audio_clip.close()
            if 'trimmed_audio' in locals() and 'audio_clip' in locals() and trimmed_audio != audio_clip:
                trimmed_audio.close()
            if 'video_cache' in locals():
                video_cache.clear()
        except:
            pass
        
        raise RuntimeError(f"Video rendering failed: {str(e)}")


def add_transitions(clips: List[VideoFileClip], transition_duration: float = 0.5) -> VideoFileClip:
    """Add crossfade transitions between clips using compatibility layer.
    
    Args:
        clips: List of video clips
        transition_duration: Duration of crossfade in seconds
        
    Returns:
        Composite video with transitions
    """
    # Use safe import handling MoviePy version differences
    VideoFileClip, AudioFileClip, concatenate_videoclips, CompositeVideoClip = import_moviepy_safely()
    
    # Try to import fade effects with version compatibility
    try:
        from moviepy.video.fx import fadeout, fadein
    except ImportError:
        try:
            # In MoviePy 2.x, effects might be capitalized and accessed differently
            from moviepy.video.fx import FadeOut as fadeout, FadeIn as fadein
        except ImportError:
            # If no fade effects available, skip transitions
            print("Warning: Fade effects not available, skipping transitions")
            return concatenate_videoclips(clips, method="chain")
    
    if not clips:
        raise ValueError("No clips provided for transitions")
    
    if len(clips) == 1:
        # Single clip - just add fade in/out if effects available
        clip = clips[0]
        try:
            # Add fade in at start (0.5s)
            clip = clip.fx(fadein, 0.5)
            # Add fade out at end (0.5s) 
            clip = clip.fx(fadeout, 0.5)
        except:
            # If effects fail, return original clip
            pass
        return clip
    
    # Multiple clips - add crossfades
    processed_clips = []
    
    for i, clip in enumerate(clips):
        current_clip = clip.copy()
        
        try:
            if i == 0:
                # First clip: fade in at start, fade out at end for transition
                current_clip = current_clip.fx(fadein, 0.5)  # Fade in
                if len(clips) > 1:
                    current_clip = current_clip.fx(fadeout, transition_duration)  # Fade out for next clip
            
            elif i == len(clips) - 1:
                # Last clip: fade in from previous, fade out at end
                current_clip = current_clip.fx(fadein, transition_duration)  # Fade in from previous
                current_clip = current_clip.fx(fadeout, 0.5)  # Final fade out
            
            else:
                # Middle clips: fade in from previous, fade out to next
                current_clip = current_clip.fx(fadein, transition_duration)  # Fade in from previous
                current_clip = current_clip.fx(fadeout, transition_duration)  # Fade out to next
        except:
            # If effects fail, use original clip
            pass
        
        processed_clips.append(current_clip)
    
    # Concatenate all clips with overlapping transitions
    # Note: For true crossfades, clips need to overlap in time
    # This creates fade in/out effects that provide smooth transitions
    try:
        final_video = concatenate_videoclips(processed_clips, padding=-transition_duration, method="compose")
    except:
        # If compose method fails, fallback to chain method
        final_video = concatenate_videoclips(processed_clips, method="chain")
    
    return final_video


def assemble_clips(video_files: List[str], audio_file: str, output_path: str,
                  pattern: str = 'balanced', progress_callback: Optional[callable] = None) -> str:
    """Main function to assemble clips into final video.
    
    Combines all steps:
    1. Analyze all video files
    2. Analyze audio file
    3. Match clips to beats
    4. Render final video
    
    Args:
        video_files: List of paths to video files
        audio_file: Path to music file
        output_path: Path for output video
        pattern: Variety pattern to use
        progress_callback: Optional callback for progress updates
        
    Returns:
        Path to final rendered video
        
    Raises:
        FileNotFoundError: If any input file doesn't exist
        ValueError: If no suitable clips found
        RuntimeError: If rendering fails
    """
    import os
    from .audio_analyzer import analyze_audio
    from .video_analyzer import analyze_video_file
    
    def report_progress(step: str, progress: float):
        """Helper to report progress if callback provided."""
        if progress_callback:
            progress_callback(step, progress)
    
    # Validate input files
    if not os.path.exists(audio_file):
        raise FileNotFoundError(f"Audio file not found: {audio_file}")
    
    missing_videos = [vf for vf in video_files if not os.path.exists(vf)]
    if missing_videos:
        raise FileNotFoundError(f"Video files not found: {missing_videos}")
    
    if not video_files:
        raise ValueError("No video files provided")
    
    report_progress("Starting analysis", 0.0)
    
    # Step 1: Analyze audio file
    report_progress("Analyzing audio", 0.1)
    try:
        audio_data = analyze_audio(audio_file)
        # CRITICAL FIX: Use compensated beats instead of raw beats to fix sync issues
        beats = audio_data['compensated_beats']  # Offset-corrected and filtered beats
        
        # Get musical timing information for professional synchronization
        musical_start_time = audio_data['musical_start_time']
        intro_duration = audio_data['intro_duration']
        allowed_durations = audio_data['allowed_durations']
        
        if len(beats) < 2:
            raise ValueError(f"Insufficient beats detected in audio file: {len(beats)} beats")
            
        report_progress("Audio analysis complete", 0.2)
        
    except Exception as e:
        raise RuntimeError(f"Failed to analyze audio file: {str(e)}")
    
    # Step 2: Analyze all video files
    report_progress("Analyzing videos", 0.3)
    all_video_chunks = []
    
    for i, video_file in enumerate(video_files):
        try:
            video_chunks = analyze_video_file(video_file)
            if video_chunks:
                all_video_chunks.extend(video_chunks)
                
            # Update progress for each video
            video_progress = 0.3 + (0.4 * (i + 1) / len(video_files))
            report_progress(f"Analyzed video {i+1}/{len(video_files)}", video_progress)
            
        except Exception as e:
            # Log error but continue with other videos
            print(f"Warning: Failed to analyze video {video_file}: {str(e)}")
            continue
    
    if not all_video_chunks:
        raise ValueError("No suitable video clips found in any input files")
    
    report_progress(f"Video analysis complete: {len(all_video_chunks)} clips found", 0.7)
    
    # Step 3: Match clips to beats
    report_progress("Matching clips to beats", 0.75)
    try:
        timeline = match_clips_to_beats(
            video_chunks=all_video_chunks,
            beats=beats,
            allowed_durations=allowed_durations,
            pattern=pattern,
            musical_start_time=musical_start_time  # Use musical intelligence for sync
        )
        
        if not timeline.clips:
            raise ValueError("No clips could be matched to the beat pattern")
            
        report_progress(f"Beat matching complete: {len(timeline.clips)} clips selected", 0.8)
        
    except Exception as e:
        raise RuntimeError(f"Failed to match clips to beats: {str(e)}")
    
    # Step 4: Render final video
    report_progress("Rendering video", 0.85)
    try:
        def render_progress(step_name: str, progress: float):
            # Scale render progress to final 15% of overall progress
            overall_progress = 0.85 + (0.15 * progress)
            report_progress(f"Rendering: {step_name}", overall_progress)
        
        final_video_path = render_video(
            timeline=timeline,
            audio_file=audio_file,
            output_path=output_path,
            progress_callback=render_progress
        )
        
        report_progress("Video rendering complete", 1.0)
        return final_video_path
        
    except Exception as e:
        raise RuntimeError(f"Failed to render video: {str(e)}")
    
    # Export timeline JSON for debugging (optional)
    try:
        timeline_path = output_path.replace('.mp4', '_timeline.json')
        timeline.export_json(timeline_path)
        print(f"Debug: Timeline exported to {timeline_path}")
    except Exception:
        pass  # Non-critical, ignore errors


def detect_optimal_codec_settings() -> Tuple[Dict[str, Any], List[str]]:
    """Detect and return optimal codec settings for hardware acceleration.
    
    Returns optimized codec parameters based on available hardware:
    - NVIDIA GPU: h264_nvenc with fastest presets
    - Intel GPU: h264_qsv with optimized settings
    - CPU only: libx264 with ultrafast preset (3-4x faster than medium)
    
    Returns:
        Tuple containing:
        - Dictionary of MoviePy parameters for write_videofile()
        - List of FFmpeg-specific parameters for ffmpeg_params argument
    """
    import subprocess
    import os
    
    # Default high-performance CPU settings (3-4x faster than 'medium')
    default_moviepy_params = {
        'codec': 'libx264',
        'audio_codec': 'aac',
        'threads': os.cpu_count() or 4,  # Use all CPU cores
    }
    
    default_ffmpeg_params = [
        '-preset', 'ultrafast',  # CRITICAL: Much faster than 'medium'
        '-crf', '23',            # Constant Rate Factor for quality control
    ]
    
    try:
        # Test for NVIDIA GPU acceleration (h264_nvenc)
        result = subprocess.run(['ffmpeg', '-encoders'], 
                              capture_output=True, text=True, timeout=5)
        if 'h264_nvenc' in result.stdout:
            try:
                # Test if NVENC actually works
                test_cmd = ['ffmpeg', '-f', 'lavfi', '-i', 'testsrc2=duration=1:size=320x240:rate=1', 
                           '-c:v', 'h264_nvenc', '-f', 'null', '-']
                subprocess.run(test_cmd, capture_output=True, timeout=10, check=True)
                
                moviepy_params = {
                    'codec': 'h264_nvenc',
                    'audio_codec': 'aac',
                    'threads': 1,    # NVENC doesn't need many threads
                }
                
                ffmpeg_params = [
                    '-preset', 'p1',     # Fastest NVENC preset
                    '-rc', 'vbr',        # Variable bitrate
                    '-cq', '23',         # NVENC quality parameter
                ]
                
                return moviepy_params, ffmpeg_params
                
            except (subprocess.SubprocessError, subprocess.TimeoutExpired):
                pass  # NVENC test failed, fall through to next option
                
        # Test for Intel Quick Sync (h264_qsv)
        if 'h264_qsv' in result.stdout:
            try:
                test_cmd = ['ffmpeg', '-f', 'lavfi', '-i', 'testsrc2=duration=1:size=320x240:rate=1', 
                           '-c:v', 'h264_qsv', '-f', 'null', '-']
                subprocess.run(test_cmd, capture_output=True, timeout=10, check=True)
                
                moviepy_params = {
                    'codec': 'h264_qsv',
                    'audio_codec': 'aac',
                    'threads': 2,
                }
                
                ffmpeg_params = [
                    '-preset', 'veryfast',
                ]
                
                return moviepy_params, ffmpeg_params
                
            except (subprocess.SubprocessError, subprocess.TimeoutExpired):
                pass  # QSV test failed, use CPU
                
    except (subprocess.SubprocessError, subprocess.TimeoutExpired, FileNotFoundError):
        pass  # ffmpeg not available or failed, use CPU encoding
    
    # Return optimized CPU settings if hardware acceleration unavailable
    return default_moviepy_params, default_ffmpeg_params


if __name__ == "__main__":
    # Test script for clip assembly
    print("AutoCut Clip Assembler - Test Mode")
    print("TODO: Add test with sample files")