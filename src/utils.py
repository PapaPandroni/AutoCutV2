"""
Utility Functions for AutoCut

Common helper functions used across multiple modules.
"""

import os
import logging
import subprocess
import json
from typing import List, Optional, Dict, Any
from pathlib import Path


# Supported file formats
SUPPORTED_VIDEO_FORMATS = {'.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', '.m4v'}
SUPPORTED_AUDIO_FORMATS = {'.mp3', '.wav', '.m4a', '.flac', '.aac', '.ogg'}


def setup_logging(log_level: str = "INFO") -> logging.Logger:
    """Set up logging configuration for AutoCut.
    
    Args:
        log_level: Logging level ("DEBUG", "INFO", "WARNING", "ERROR")
        
    Returns:
        Configured logger instance
    """
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('autocut.log')
        ]
    )
    return logging.getLogger('autocut')


def validate_video_file(file_path: str) -> bool:
    """Validate that a file is a supported video format.
    
    Args:
        file_path: Path to the video file
        
    Returns:
        True if file exists and is a supported video format
    """
    if not os.path.exists(file_path):
        return False
        
    file_extension = Path(file_path).suffix.lower()
    return file_extension in SUPPORTED_VIDEO_FORMATS


def validate_audio_file(file_path: str) -> bool:
    """Validate that a file is a supported audio format.
    
    Args:
        file_path: Path to the audio file
        
    Returns:
        True if file exists and is a supported audio format
    """
    if not os.path.exists(file_path):
        return False
        
    file_extension = Path(file_path).suffix.lower()
    return file_extension in SUPPORTED_AUDIO_FORMATS


def validate_input_files(video_files: List[str], audio_file: str) -> List[str]:
    """Validate all input files and return list of errors.
    
    Args:
        video_files: List of video file paths
        audio_file: Path to audio file
        
    Returns:
        List of error messages (empty if all files are valid)
    """
    errors = []
    
    if not video_files:
        errors.append("No video files provided")
    else:
        for i, video_file in enumerate(video_files):
            if not validate_video_file(video_file):
                errors.append(f"Video file {i+1} is invalid or unsupported: {video_file}")
    
    if not validate_audio_file(audio_file):
        errors.append(f"Audio file is invalid or unsupported: {audio_file}")
        
    return errors


def ensure_output_directory(output_path: str) -> str:
    """Ensure output directory exists and return absolute path.
    
    Args:
        output_path: Desired output file path
        
    Returns:
        Absolute path to output file
        
    Raises:
        OSError: If directory cannot be created
    """
    output_path = os.path.abspath(output_path)
    output_dir = os.path.dirname(output_path)
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
        
    return output_path


def format_duration(seconds: float) -> str:
    """Format duration in seconds as MM:SS string.
    
    Args:
        seconds: Duration in seconds
        
    Returns:
        Formatted duration string
    """
    minutes = int(seconds // 60)
    seconds = int(seconds % 60)
    return f"{minutes:02d}:{seconds:02d}"


def get_file_size_mb(file_path: str) -> float:
    """Get file size in megabytes.
    
    Args:
        file_path: Path to file
        
    Returns:
        File size in MB, or 0 if file doesn't exist
    """
    try:
        size_bytes = os.path.getsize(file_path)
        return size_bytes / (1024 * 1024)
    except OSError:
        return 0.0


def safe_filename(filename: str) -> str:
    """Create a safe filename by removing/replacing invalid characters.
    
    Args:
        filename: Original filename
        
    Returns:
        Safe filename suitable for filesystem
    """
    # Characters that are problematic in filenames
    invalid_chars = '<>:"/\\|?*'
    
    safe_name = filename
    for char in invalid_chars:
        safe_name = safe_name.replace(char, '_')
        
    # Remove any leading/trailing whitespace and dots
    safe_name = safe_name.strip('. ')
    
    # Ensure filename is not empty
    if not safe_name:
        safe_name = "untitled"
        
    return safe_name


class ProgressTracker:
    """Helper class for tracking and reporting progress."""
    
    def __init__(self, total_steps: int = 100):
        self.total_steps = total_steps
        self.current_step = 0
        self.callbacks: List[callable] = []
        
    def add_callback(self, callback: callable):
        """Add a progress callback function."""
        self.callbacks.append(callback)
        
    def update(self, step: int, message: str = ""):
        """Update progress and notify callbacks."""
        self.current_step = min(step, self.total_steps)
        percentage = (self.current_step / self.total_steps) * 100
        
        for callback in self.callbacks:
            try:
                callback(percentage, message)
            except Exception as e:
                # Don't let callback errors stop processing
                logging.warning(f"Progress callback error: {e}")
                
    def increment(self, message: str = ""):
        """Increment progress by one step."""
        self.update(self.current_step + 1, message)
        
    def complete(self, message: str = "Complete"):
        """Mark progress as complete."""
        self.update(self.total_steps, message)


def detect_video_codec(file_path: str) -> Dict[str, Any]:
    """Detect video codec and format information using FFprobe.
    
    Args:
        file_path: Path to the video file
        
    Returns:
        Dictionary containing codec information:
        - 'codec': Video codec name (e.g., 'h264', 'hevc')
        - 'is_hevc': Boolean indicating if codec is H.265/HEVC
        - 'resolution': (width, height) tuple
        - 'fps': Frame rate
        - 'duration': Video duration in seconds
        
    Raises:
        subprocess.CalledProcessError: If FFprobe fails
        FileNotFoundError: If FFprobe is not installed
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Video file not found: {file_path}")
    
    try:
        # Use FFprobe to get video stream information
        cmd = [
            'ffprobe', '-v', 'quiet', '-print_format', 'json',
            '-show_streams', '-select_streams', 'v:0', file_path
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        data = json.loads(result.stdout)
        
        if not data.get('streams'):
            raise ValueError(f"No video streams found in {file_path}")
        
        video_stream = data['streams'][0]
        codec_name = video_stream.get('codec_name', '').lower()
        
        # Parse frame rate
        fps_str = video_stream.get('r_frame_rate', '30/1')
        if '/' in fps_str:
            num, den = map(int, fps_str.split('/'))
            fps = num / den if den != 0 else 30.0
        else:
            fps = float(fps_str)
        
        return {
            'codec': codec_name,
            'is_hevc': codec_name in ['hevc', 'h265'],
            'resolution': (
                int(video_stream.get('width', 0)),
                int(video_stream.get('height', 0))
            ),
            'fps': fps,
            'duration': float(video_stream.get('duration', 0)),
            'pixel_format': video_stream.get('pix_fmt', 'unknown')
        }
        
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"FFprobe failed for {file_path}: {e.stderr}")
    except (json.JSONDecodeError, KeyError, ValueError) as e:
        raise RuntimeError(f"Failed to parse video information for {file_path}: {str(e)}")


def transcode_hevc_to_h264(input_path: str, output_path: str = None, 
                          progress_callback: Optional[callable] = None) -> str:
    """Transcode H.265/HEVC video to H.264 for MoviePy compatibility.
    
    Args:
        input_path: Path to input H.265 video file
        output_path: Output path (auto-generated if None)
        progress_callback: Optional callback for progress updates
        
    Returns:
        Path to transcoded H.264 video file
        
    Raises:
        subprocess.CalledProcessError: If FFmpeg transcoding fails
        FileNotFoundError: If FFmpeg is not installed
    """
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input video file not found: {input_path}")
    
    # Generate output path if not provided
    if output_path is None:
        input_stem = Path(input_path).stem
        output_path = f"{input_stem}_h264.mp4"
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    
    try:
        # FFmpeg command for H.265 to H.264 transcoding
        cmd = [
            'ffmpeg', '-i', input_path,
            '-c:v', 'libx264',          # Use H.264 codec
            '-crf', '18',               # High quality (lower is better, 18 is visually lossless)
            '-preset', 'fast',          # Balance speed vs compression
            '-c:a', 'copy',             # Copy audio without re-encoding
            '-movflags', '+faststart',  # Optimize for web playback
            '-y',                       # Overwrite output file
            output_path
        ]
        
        if progress_callback:
            progress_callback("Transcoding H.265 to H.264", 0.0)
        
        # Run FFmpeg with progress monitoring
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        
        # Monitor progress (simplified - FFmpeg progress parsing is complex)
        while process.poll() is None:
            if progress_callback:
                progress_callback("Transcoding in progress", 0.5)
        
        # Check if process completed successfully
        if process.returncode != 0:
            stderr = process.stderr.read() if process.stderr else "Unknown error"
            raise subprocess.CalledProcessError(process.returncode, cmd, stderr)
        
        if progress_callback:
            progress_callback("Transcoding complete", 1.0)
        
        return output_path
        
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"FFmpeg transcoding failed: {e.stderr}")
    except Exception as e:
        raise RuntimeError(f"Transcoding error: {str(e)}")


def preprocess_video_if_needed(file_path: str, temp_dir: str = "temp") -> str:
    """Preprocess video file if needed for MoviePy compatibility.
    
    Automatically detects problematic codecs (H.265/HEVC) and transcodes
    them to H.264 for better MoviePy compatibility.
    
    Args:
        file_path: Path to input video file
        temp_dir: Directory for temporary transcoded files
        
    Returns:
        Path to processed video file (original if no processing needed,
        transcoded file if H.265 was detected)
        
    Raises:
        RuntimeError: If codec detection or transcoding fails
    """
    try:
        # Detect video codec
        codec_info = detect_video_codec(file_path)
        
        # If not HEVC, return original file
        if not codec_info['is_hevc']:
            return file_path
        
        # HEVC detected - need to transcode
        print(f"⚠️  H.265/HEVC detected in {Path(file_path).name}")
        print("   Transcoding to H.264 for MoviePy compatibility...")
        
        # Create temp directory
        os.makedirs(temp_dir, exist_ok=True)
        
        # Generate output path in temp directory
        input_stem = Path(file_path).stem
        output_path = os.path.join(temp_dir, f"{input_stem}_h264.mp4")
        
        # Transcode file
        transcoded_path = transcode_hevc_to_h264(file_path, output_path)
        
        print(f"✅ Transcoded to: {transcoded_path}")
        return transcoded_path
        
    except Exception as e:
        print(f"⚠️  Failed to preprocess {file_path}: {str(e)}")
        print("   Continuing with original file - may cause processing issues")
        return file_path


# Configuration defaults
DEFAULT_CONFIG = {
    'min_clip_duration': 0.5,      # Minimum clip duration in seconds
    'max_clip_duration': 8.0,      # Maximum clip duration in seconds
    'scene_threshold': 30.0,       # Scene detection sensitivity
    'transition_duration': 0.5,    # Crossfade duration in seconds
    'output_quality': 'high',      # Output quality ('low', 'medium', 'high')
    'temp_dir': 'temp',           # Temporary files directory
}


def get_config_value(key: str, default=None):
    """Get configuration value with fallback to default."""
    return DEFAULT_CONFIG.get(key, default)


if __name__ == "__main__":
    # Test utility functions
    print("AutoCut Utilities - Test Mode")
    
    # Test file validation
    print(f"Video formats supported: {SUPPORTED_VIDEO_FORMATS}")
    print(f"Audio formats supported: {SUPPORTED_AUDIO_FORMATS}")
    
    # Test progress tracker
    tracker = ProgressTracker(10)
    tracker.add_callback(lambda p, m: print(f"Progress: {p:.1f}% - {m}"))
    
    for i in range(11):
        tracker.update(i, f"Step {i}")
        
    print("Utility tests completed")