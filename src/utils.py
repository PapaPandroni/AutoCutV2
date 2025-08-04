"""
Utility Functions for AutoCut

Common helper functions used across multiple modules.
"""

import os
import logging
from typing import List, Optional
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