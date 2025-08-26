"""Video encoding system for AutoCut V2.

This module provides video encoding capabilities including:
- Codec detection and optimization
- Hardware acceleration support
- Encoding parameter optimization
- Compatibility layer management

Extracted from clip_assembler.py as part of system consolidation.
"""

import os
import time
from typing import Any, Dict, List, Tuple

import psutil


class VideoEncoder:
    """Handles video encoding with optimal codec settings and compatibility."""

    def __init__(self):
        self.compatibility_info = None

    def detect_optimal_codec_settings(self) -> Tuple[Dict[str, Any], List[str]]:
        """Legacy codec settings detection for backward compatibility.

        This function now delegates to the new modular VideoEncoder
        extracted as part of Phase 3 refactoring while maintaining the
        exact same interface as the original function.

        Returns:
            Tuple containing:
            - Dictionary of MoviePy parameters for write_videofile()
            - List of FFmpeg-specific parameters for ffmpeg_params argument
        """
        try:
            # Import the new modular encoder system with dual import pattern
            try:
                from video.rendering import (
                    detect_optimal_codec_settings as detect_codec_modular,
                )
            except ImportError:
                from .video.rendering import (
                    detect_optimal_codec_settings as detect_codec_modular,
                )

            # Delegate to the new modular system
            return detect_codec_modular()

        except ImportError:
            # Fallback to legacy implementation if modules not available
            return self._fallback_codec_settings()
        except Exception as e:
            raise RuntimeError(f"Codec detection failed: {e!s}")

    def detect_optimal_codec_settings_with_diagnostics(self) -> Tuple[
        Dict[str, Any], List[str], Dict[str, str],
    ]:
        """Enhanced codec settings detection with full diagnostic information.

        New interface that provides comprehensive hardware detection results
        for enhanced video processing workflows.

        Returns:
            Tuple containing:
            - Dictionary of MoviePy parameters for write_videofile()
            - List of FFmpeg-specific parameters for ffmpeg_params argument
            - Dictionary of diagnostic information and capability details
        """
        # Import enhanced detection from new hardware module with dual import pattern
        try:
            # Try absolute import first for autocut.py context
            from hardware.detection import detect_optimal_codec_settings_enhanced
        except ImportError:
            try:
                # Try relative import for package context
                from .hardware.detection import detect_optimal_codec_settings_enhanced
            except ImportError:
                # Fallback for backwards compatibility
                try:
                    from utils import detect_optimal_codec_settings_enhanced
                except ImportError:
                    from .utils import detect_optimal_codec_settings_enhanced

        return detect_optimal_codec_settings_enhanced()

    def _fallback_codec_settings(self) -> Tuple[Dict[str, Any], List[str]]:
        """Fallback codec settings when hardware detection is not available."""

        # Safe fallback settings that work across platforms
        moviepy_params = {
            "codec": "libx264",
            "bitrate": "5000k",
            "audio_codec": "aac",
            "audio_bitrate": "128k",
        }

        ffmpeg_params = [
            "-preset", "medium",
            "-crf", "23",
            "-pix_fmt", "yuv420p",
            "-movflags", "+faststart",
        ]

        return moviepy_params, ffmpeg_params

    def prepare_encoding_parameters(self, target_format: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare encoding parameters optimized for target format.

        Args:
            target_format: Target format specification from format analyzer

        Returns:
            Dictionary of encoding parameters
        """
        # Get optimal codec settings
        moviepy_params, ffmpeg_params = self.detect_optimal_codec_settings()

        # Check memory pressure for quality adjustment
        try:
            memory_percent = psutil.virtual_memory().percent

            # Adjust quality based on memory pressure
            if memory_percent > 80.0:

                # Lower quality settings for memory safety
                if "bitrate" in moviepy_params:
                    current_bitrate = moviepy_params["bitrate"]
                    if current_bitrate.endswith("k"):
                        bitrate_val = int(current_bitrate[:-1])
                        reduced_bitrate = max(2000, int(bitrate_val * 0.6))  # Minimum 2Mbps
                        moviepy_params["bitrate"] = f"{reduced_bitrate}k"

        except Exception as e:
            pass

        # Enhanced FFmpeg parameters for format consistency
        format_consistency_params = [
            "-pix_fmt", "yuv420p",  # Consistent color format
            "-vsync", "cfr",        # Constant frame rate conversion
            "-async", "1",          # Audio sync parameter
        ]

        # Add FPS parameters if normalization was applied
        if target_format.get("requires_normalization", False):
            format_consistency_params.extend([
                "-r", str(target_format["target_fps"]),  # Force target frame rate
            ])

        # Mac-specific AAC audio parameters for QuickTime compatibility
        mac_audio_compatibility_params = [
            "-profile:a", "aac_low",
            "-ar", "44100",
            "-channel_layout", "stereo",
            "-ac", "2",
            "-aac_coder", "twoloop",
            "-cutoff", "18000",
        ]

        # Stability parameters
        stability_params = [
            "-threads", "2",
            "-max_muxing_queue_size", "1024",
            "-fflags", "+genpts",
        ]

        # MP4 container optimization
        container_optimization_params = [
            "-movflags", "+faststart",
            "-f", "mp4",
        ]

        # Combine all FFmpeg parameters in proper order
        enhanced_ffmpeg_params = (
            ffmpeg_params +
            format_consistency_params +
            mac_audio_compatibility_params +
            stability_params +
            container_optimization_params
        )

        # Prepare comprehensive parameters
        return {
            **moviepy_params,
            "ffmpeg_params": enhanced_ffmpeg_params,
            "temp_audiofile": "temp-audio.m4a",
            "remove_temp": True,
            "fps": target_format["target_fps"],
            "audio_fps": 44100,
            "audio_codec": "aac",
            "audio_bitrate": "160k",
            "write_logfile": False,
        }


    def encode_video(
        self,
        final_video,
        output_path: str,
        encoding_params: Dict[str, Any],
    ) -> str:
        """Encode final video with compatibility layer.

        Args:
            final_video: Composed video ready for encoding
            output_path: Path for output video file
            encoding_params: Encoding parameters from prepare_encoding_parameters

        Returns:
            Path to encoded video file
        """
        try:
            # Import compatibility layer with dual import pattern
            try:
                from compatibility.moviepy import (
                    check_moviepy_api_compatibility,
                    write_videofile_safely,
                )
            except ImportError:
                try:
                    from .compatibility.moviepy import (
                        check_moviepy_api_compatibility,
                        write_videofile_safely,
                    )
                except ImportError:
                    # Fallback if compatibility module not available
                    def write_videofile_safely(video, path, compatibility_info, **kwargs):
                        video.write_videofile(path, **kwargs)
                    def check_moviepy_api_compatibility():
                        return {"version_detected": "unknown", "method_mappings": {"subclip": "subclip", "set_audio": "set_audio"}}

            # Get compatibility info if not cached
            if not self.compatibility_info:
                self.compatibility_info = check_moviepy_api_compatibility()

            # Create output directory if needed
            os.makedirs(os.path.dirname(output_path), exist_ok=True)


            encoding_start_time = time.time()

            # Lower process priority to prevent system lockup
            try:
                current_process = psutil.Process()
                current_process.nice(10)  # Lower priority
            except:
                pass

            # Render with version-compatible parameter checking
            write_videofile_safely(
                final_video,
                output_path,
                self.compatibility_info,
                **encoding_params,
            )

            # Validate output
            if not os.path.exists(output_path):
                raise RuntimeError(f"Output file was not created: {output_path}")

            encoding_time = time.time() - encoding_start_time
            return output_path

        except Exception as e:
            raise RuntimeError(f"Video encoding failed: {e!s}")


# Legacy functions for backward compatibility
def detect_optimal_codec_settings() -> Tuple[Dict[str, Any], List[str]]:
    """Legacy function for backward compatibility."""
    encoder = VideoEncoder()
    return encoder.detect_optimal_codec_settings()


def detect_optimal_codec_settings_with_diagnostics() -> Tuple[
    Dict[str, Any], List[str], Dict[str, str],
]:
    """Legacy function for backward compatibility."""
    encoder = VideoEncoder()
    return encoder.detect_optimal_codec_settings_with_diagnostics()
