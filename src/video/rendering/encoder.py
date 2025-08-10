"""Video encoding system with hardware acceleration and compatibility layers."""

import os
from typing import Dict, Any, List, Tuple
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


class VideoEncoder:
    """Handles video encoding with optimal codec settings and compatibility.
    
    Extracted from clip_assembler.py as part of Phase 3 refactoring.
    Provides clean interface for video encoding with hardware acceleration.
    """
    
    def __init__(self):
        self.compatibility_info = None
        
    def prepare_encoding_parameters(self, target_format: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare encoding parameters optimized for target format.
        
        Args:
            target_format: Target format specification from compositor
            
        Returns:
            Dictionary of encoding parameters
        """
        # Get optimal codec settings
        moviepy_params, ffmpeg_params = self._detect_optimal_codec_settings()
        
        # Enhanced FFmpeg parameters for format consistency
        format_consistency_params = [
            "-pix_fmt", "yuv420p",  # Consistent color format
            "-vsync", "cfr",        # Constant frame rate conversion
            "-async", "1",          # Audio sync parameter
        ]
        
        # Add resolution/fps parameters if normalization was applied
        if target_format.get("requires_normalization", False):
            format_consistency_params.extend([
                "-r", str(target_format["target_fps"]),  # Force target frame rate
                "-s", f"{target_format['target_width']}x{target_format['target_height']}",  # Force resolution
            ])
        
        # Combine all FFmpeg parameters
        enhanced_ffmpeg_params = ffmpeg_params + format_consistency_params
        logger.info(f"Enhanced FFmpeg params: {enhanced_ffmpeg_params}")
        
        # Prepare comprehensive parameters
        write_params = {
            **moviepy_params,
            "ffmpeg_params": enhanced_ffmpeg_params,
            "temp_audiofile": "temp-audio.m4a",
            "remove_temp": True,
            "fps": target_format["target_fps"],
            "logger": None,  # Suppress MoviePy logging
            # Audio-specific parameters for better sync
            "audio_fps": 44100,      # Standard audio sample rate
            "audio_codec": "aac",     # Compatible audio codec
            "audio_bitrate": "128k",  # Good quality audio bitrate
        }
        
        logger.info(f"Encoding parameters prepared: {list(write_params.keys())}")
        return write_params
        
    def encode_video(
        self, 
        final_video, 
        output_path: str, 
        encoding_params: Dict[str, Any]
    ) -> str:
        """Encode final video with compatibility layer.
        
        Args:
            final_video: Composed video ready for encoding
            output_path: Path for output video file
            encoding_params: Encoding parameters from prepare_encoding_parameters
            
        Returns:
            Path to encoded video file
            
        Raises:
            VideoProcessingError: If encoding fails
        """
        try:
            # Import compatibility layer
            try:
                from compatibility.moviepy import write_videofile_safely, check_moviepy_api_compatibility
            except ImportError:
                # Fallback if compatibility module not available
                def write_videofile_safely(video, path, compatibility_info, **kwargs):
                    video.write_videofile(path, **kwargs)
                def check_moviepy_api_compatibility():
                    return {"version_detected": "unknown", "method_mappings": {"subclip": "subclip", "set_audio": "set_audio"}}
            
            # Get compatibility info if not cached
            if not self.compatibility_info:
                self.compatibility_info = check_moviepy_api_compatibility()
                logger.info(f"MoviePy API compatibility: {self.compatibility_info['version_detected']}")
            
            # Create output directory if needed
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            logger.info(f"Encoding final video to: {output_path}")
            
            # Render with version-compatible parameter checking
            write_videofile_safely(
                final_video, 
                output_path, 
                self.compatibility_info, 
                **encoding_params
            )
            
            # Validate output
            if not os.path.exists(output_path):
                raise VideoProcessingError(f"Output file was not created: {output_path}")
            
            logger.info("Video encoding completed successfully")
            return output_path
            
        except Exception as e:
            raise VideoProcessingError(f"Video encoding failed: {str(e)}")
    
    def _detect_optimal_codec_settings(self) -> Tuple[Dict[str, Any], List[str]]:
        """Detect optimal codec settings using hardware detection.
        
        Returns:
            Tuple of (moviepy_params, ffmpeg_params)
        """
        try:
            # Import enhanced detection from hardware module
            try:
                from hardware.detection import detect_optimal_codec_settings_enhanced
            except ImportError:
                # Fallback if hardware detection not available
                def detect_optimal_codec_settings_enhanced():
                    return {"codec": "libx264", "bitrate": "5000k"}, ["-preset", "medium"], {"encoder_type": "SOFTWARE"}
            
            moviepy_params, ffmpeg_params, diagnostics = detect_optimal_codec_settings_enhanced()
            
            encoder_type = diagnostics.get("encoder_type", "UNKNOWN")
            logger.info(f"Codec settings detected: {encoder_type} encoder")
            
            return moviepy_params, ffmpeg_params
            
        except ImportError:
            logger.warning("Enhanced codec detection not available, using fallback")
            
            # Fallback codec settings
            moviepy_params = {
                "codec": "libx264",
                "bitrate": "5000k",
                "audio_codec": "aac",
            }
            
            ffmpeg_params = [
                "-preset", "medium",
                "-crf", "23",
            ]
            
            return moviepy_params, ffmpeg_params


def detect_optimal_codec_settings() -> Tuple[Dict[str, Any], List[str]]:
    """Legacy codec settings detection for backward compatibility.
    
    This function maintains backward compatibility with existing AutoCut code
    while leveraging the enhanced hardware detection system.
    
    Returns:
        Tuple containing:
        - Dictionary of MoviePy parameters for write_videofile()
        - List of FFmpeg-specific parameters for ffmpeg_params argument
    """
    encoder = VideoEncoder()
    return encoder._detect_optimal_codec_settings()