"""
Hardware acceleration detection and capability testing for AutoCut V2.

This module provides comprehensive hardware encoder detection and validation,
extracted from the monolithic utils.py. Includes NVIDIA NVENC and Intel QSV
support with iPhone compatibility testing.
"""

import os
import subprocess
import time
import tempfile
from typing import Dict, Any, List, Tuple, Optional

# Import our custom exceptions with dual import pattern
try:
    from ..core.exceptions import HardwareAccelerationError
except ImportError:
    # Fallback for direct execution
    from core.exceptions import HardwareAccelerationError


class HardwareDetector:
    """
    Comprehensive hardware acceleration detection and testing system.

    Provides hardware encoder detection with:
    - NVIDIA NVENC capability testing with iPhone parameter validation
    - Intel QuickSync Video (QSV) support detection and testing
    - CPU fallback with optimized settings
    - Comprehensive error categorization and diagnostics
    - Performance-optimized caching to avoid repeated hardware tests
    """

    def __init__(self, cache_timeout: float = 300.0):
        """
        Initialize hardware detector with optional caching.

        Args:
            cache_timeout: How long to cache hardware detection results (seconds, default: 5 minutes)
        """
        self.cache_timeout = cache_timeout
        self._hardware_cache: Optional[Dict[str, Any]] = None
        self._cache_timestamp: Optional[float] = None

    def detect_optimal_settings(self, force_cpu: bool = False) -> Dict[str, Any]:
        """
        Enhanced hardware detection with actual capability testing and iPhone validation.

        Replaces: detect_optimal_codec_settings_enhanced() from utils.py

        Args:
            force_cpu: If True, skip hardware detection and return CPU settings

        Returns:
            Dictionary containing:
            - 'moviepy_params': Parameters for MoviePy write_videofile()
            - 'ffmpeg_params': FFmpeg-specific parameters
            - 'encoder_type': Type of encoder (NVIDIA_NVENC, INTEL_QSV, CPU)
            - 'diagnostics': Detailed capability and error information
        """
        # Check cache validity first
        current_time = time.time()
        if (
            not force_cpu
            and self._hardware_cache is not None
            and self._cache_timestamp is not None
            and current_time - self._cache_timestamp < self.cache_timeout
        ):
            return self._hardware_cache

        print("🔍 Enhanced hardware capability detection...")

        # Default high-performance CPU settings
        default_result = {
            "moviepy_params": {
                "codec": "libx264",
                "audio_codec": "aac",
                "threads": os.cpu_count() or 4,
            },
            "ffmpeg_params": ["-preset", "ultrafast", "-crf", "23"],
            "encoder_type": "CPU",
            "diagnostics": {
                "driver_status": "N/A",
                "iphone_compatible": True,
                "error_category": None,
                "diagnostic_message": "CPU encoding with optimized parameters",
            },
        }

        if force_cpu:
            print("⚡ Forced CPU encoding mode")
            return default_result

        diagnostics = {"tests_performed": [], "errors_encountered": []}

        try:
            # Step 1: Check FFmpeg availability
            if not self._check_ffmpeg_availability(diagnostics):
                return default_result

            # Step 2: List available encoders
            available_encoders = self._list_available_encoders(diagnostics)
            if not available_encoders:
                return default_result

            # Step 3: Test NVIDIA NVENC
            if "h264_nvenc" in available_encoders:
                nvenc_result = self._test_hardware_encoder(
                    "NVENC", "h264_nvenc", True, diagnostics
                )
                if nvenc_result["success"]:
                    result = {
                        "moviepy_params": {
                            "codec": "h264_nvenc",
                            "audio_codec": "aac",
                            "threads": 1,  # NVENC doesn't need many threads
                        },
                        "ffmpeg_params": [
                            "-preset",
                            "p1",  # Fastest NVENC preset
                            "-rc",
                            "vbr",  # Variable bitrate
                            "-cq",
                            "23",  # NVENC quality parameter
                        ],
                        "encoder_type": "NVIDIA_NVENC",
                        "diagnostics": {
                            "driver_status": nvenc_result.get("driver_status", "OK"),
                            "iphone_compatible": nvenc_result.get(
                                "iphone_compatible", True
                            ),
                            "error_category": None,
                            "diagnostic_message": f"NVIDIA GPU acceleration (5-10x faster): {nvenc_result.get('message', '')}",
                            **diagnostics,
                        },
                    }

                    self._hardware_cache = result
                    self._cache_timestamp = current_time
                    return result

            # Step 4: Test Intel QSV
            if "h264_qsv" in available_encoders:
                qsv_result = self._test_hardware_encoder(
                    "QSV", "h264_qsv", True, diagnostics
                )
                if qsv_result["success"]:
                    result = {
                        "moviepy_params": {
                            "codec": "h264_qsv",
                            "audio_codec": "aac",
                            "threads": 2,
                        },
                        "ffmpeg_params": [
                            "-preset",
                            "veryfast",
                        ],
                        "encoder_type": "INTEL_QSV",
                        "diagnostics": {
                            "driver_status": qsv_result.get("driver_status", "OK"),
                            "iphone_compatible": qsv_result.get(
                                "iphone_compatible", True
                            ),
                            "error_category": None,
                            "diagnostic_message": f"Intel Quick Sync acceleration (3-5x faster): {qsv_result.get('message', '')}",
                            **diagnostics,
                        },
                    }

                    self._hardware_cache = result
                    self._cache_timestamp = current_time
                    return result

            # Hardware acceleration not available
            print(
                "⚡ Using optimized CPU encoding (hardware acceleration not available)"
            )

        except Exception as e:
            diagnostics["errors_encountered"].append(
                f"Hardware detection error: {str(e)}"
            )
            print(f"⚠️  Hardware detection error: {str(e)}")

        # Return enhanced default result with diagnostics
        enhanced_default = {
            **default_result,
            "diagnostics": {**default_result["diagnostics"], **diagnostics},
        }

        self._hardware_cache = enhanced_default
        self._cache_timestamp = current_time
        return enhanced_default

    def get_cpu_settings(self, performance_mode: str = "fast") -> Dict[str, Any]:
        """
        Get optimized CPU encoding settings.

        Args:
            performance_mode: 'fast', 'balanced', or 'quality'

        Returns:
            Dictionary with CPU encoder settings
        """
        settings_map = {
            "fast": {
                "moviepy_params": {
                    "codec": "libx264",
                    "audio_codec": "aac",
                    "threads": min(os.cpu_count() or 4, 6),
                },
                "ffmpeg_params": ["-preset", "ultrafast", "-crf", "25"],
                "encoder_type": "CPU_FAST",
            },
            "balanced": {
                "moviepy_params": {
                    "codec": "libx264",
                    "audio_codec": "aac",
                    "threads": min(os.cpu_count() or 4, 4),
                },
                "ffmpeg_params": ["-preset", "fast", "-crf", "23"],
                "encoder_type": "CPU_BALANCED",
            },
            "quality": {
                "moviepy_params": {
                    "codec": "libx264",
                    "audio_codec": "aac",
                    "threads": 2,
                },
                "ffmpeg_params": ["-preset", "medium", "-crf", "21"],
                "encoder_type": "CPU_QUALITY",
            },
        }

        settings = settings_map.get(performance_mode, settings_map["fast"])
        settings["diagnostics"] = {
            "driver_status": "N/A",
            "iphone_compatible": True,
            "error_category": None,
            "diagnostic_message": f"CPU encoding - {performance_mode} mode",
        }

        return settings

    def _check_ffmpeg_availability(self, diagnostics: Dict[str, Any]) -> bool:
        """Check if FFmpeg is available and get version information."""
        try:
            result = subprocess.run(
                ["ffmpeg", "-version"], capture_output=True, text=True, timeout=5
            )
            diagnostics["ffmpeg_version"] = (
                result.stdout.split("\\n")[0] if result.stdout else "Unknown"
            )
            return True
        except (
            subprocess.SubprocessError,
            subprocess.TimeoutExpired,
            FileNotFoundError,
        ) as e:
            diagnostics["errors_encountered"].append(f"FFmpeg not available: {str(e)}")
            return False

    def _list_available_encoders(self, diagnostics: Dict[str, Any]) -> str:
        """List available FFmpeg encoders."""
        try:
            result = subprocess.run(
                ["ffmpeg", "-encoders"], capture_output=True, text=True, timeout=5
            )
            diagnostics["available_encoders"] = "Listed successfully"
            return result.stdout
        except (subprocess.SubprocessError, subprocess.TimeoutExpired) as e:
            diagnostics["errors_encountered"].append(
                f"Encoder listing failed: {str(e)}"
            )
            return ""

    def _test_hardware_encoder(
        self,
        encoder_name: str,
        encoder_codec: str,
        test_iphone_parameters: bool = True,
        diagnostics: Optional[Dict] = None,
    ) -> Dict[str, Any]:
        """
        Fast hardware encoder testing with intelligent early termination.

        Replaces: _test_hardware_encoder() from utils.py

        Args:
            encoder_name: Human-readable encoder name (e.g., 'NVENC', 'QSV')
            encoder_codec: FFmpeg codec name (e.g., 'h264_nvenc', 'h264_qsv')
            test_iphone_parameters: Whether to test iPhone-specific parameters
            diagnostics: Dictionary to store diagnostic information

        Returns:
            Dictionary with test results
        """
        if diagnostics is None:
            diagnostics = {}

        test_name = f"{encoder_name}_{encoder_codec}_test"
        diagnostics.setdefault("tests_performed", []).append(test_name)

        result = {
            "success": False,
            "driver_status": "UNKNOWN",
            "iphone_compatible": False,
            "message": "",
            "error_category": None,
        }

        temp_output = None

        try:
            print(f"   🧪 Fast-testing {encoder_name} with iPhone parameters...")

            with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as temp_file:
                temp_output = temp_file.name

            # Combined test: Basic functionality + iPhone parameters
            combined_cmd = [
                "ffmpeg",
                "-y",
                "-f",
                "lavfi",
                "-i",
                "testsrc2=duration=0.5:size=160x120:rate=5",  # Small/fast test
                "-c:v",
                encoder_codec,
                "-profile:v",
                "main",  # iPhone compatibility from start
                "-pix_fmt",
                "yuv420p",  # 8-bit requirement from start
                "-t",
                "0.5",  # 0.5s duration
                "-preset",
                ("p1" if "nvenc" in encoder_codec else "veryfast"),
                "-f",
                "mp4",
                temp_output,
            ]

            # Reduced timeout for faster failure detection
            combined_result = subprocess.run(
                combined_cmd, capture_output=True, text=True, timeout=3
            )

            if combined_result.returncode != 0:
                # Analyze error messages
                stderr_lower = combined_result.stderr.lower()

                if "driver" in stderr_lower and (
                    "version" in stderr_lower or "api" in stderr_lower
                ):
                    result["driver_status"] = "INCOMPATIBLE_VERSION"
                    result["error_category"] = "driver_version"
                    result["message"] = (
                        f"Driver incompatibility: {combined_result.stderr[:200]}"
                    )
                elif "device" in stderr_lower or "capability" in stderr_lower:
                    result["driver_status"] = "DEVICE_ERROR"
                    result["error_category"] = "device_capability"
                    result["message"] = (
                        f"Device capability issue: {combined_result.stderr[:200]}"
                    )
                elif "permission" in stderr_lower or "access" in stderr_lower:
                    result["driver_status"] = "PERMISSION_ERROR"
                    result["error_category"] = "permissions"
                    result["message"] = (
                        f"Permission issue: {combined_result.stderr[:200]}"
                    )
                else:
                    result["driver_status"] = "GENERAL_ERROR"
                    result["error_category"] = "unknown"
                    result["message"] = f"General error: {combined_result.stderr[:200]}"

                print(
                    f"   ❌ {encoder_name} fast test failed: {result['error_category']}"
                )
                return result

            # Combined test passed - validate output
            result["driver_status"] = "OK"

            # Quick format validation
            format_valid = self._validate_encoder_output_fast(
                temp_output, expected_profile="Main"
            )
            result["iphone_compatible"] = format_valid

            if format_valid:
                result["success"] = True
                result["message"] = (
                    f"{encoder_name} with iPhone parameters: OK (fast test)"
                )
                print(f"   ✅ {encoder_name} fast test + iPhone parameters: OK")
            else:
                result["message"] = (
                    f"{encoder_name} encoding works but output format incorrect"
                )
                print(f"   ⚠️  {encoder_name} encoding works but output format issue")

        except subprocess.TimeoutExpired:
            result["error_category"] = "timeout"
            result["message"] = (
                f"{encoder_name} test timed out (>3s) - likely hardware issue"
            )
            print(f"   ⏱️  {encoder_name} test timed out (3s) - fast failure detection")
        except Exception as e:
            result["error_category"] = "exception"
            result["message"] = f"{encoder_name} test exception: {str(e)[:100]}"
            print(f"   ❌ {encoder_name} test exception: {str(e)[:50]}...")

        finally:
            # Clean up test file
            if temp_output:
                try:
                    os.unlink(temp_output)
                except OSError:
                    pass

        diagnostics.setdefault("encoder_test_results", {})[encoder_name] = result
        return result

    def _validate_encoder_output_fast(
        self, video_path: str, expected_profile: str = "Main"
    ) -> bool:
        """
        Fast encoder output validation for hardware detection testing.

        Args:
            video_path: Path to test video file
            expected_profile: Expected H.264 profile

        Returns:
            True if output meets basic iPhone compatibility, False otherwise
        """
        if not os.path.exists(video_path):
            return False

        try:
            cmd = [
                "ffprobe",
                "-v",
                "quiet",
                "-select_streams",
                "v:0",
                "-show_entries",
                "stream=codec_name,profile,pix_fmt",
                "-of",
                "csv=p=0",
                video_path,
            ]

            result = subprocess.run(cmd, capture_output=True, text=True, timeout=2)
            if result.returncode != 0:
                return False

            # Parse output: codec_name,profile,pix_fmt
            output_parts = result.stdout.strip().split(",")
            if len(output_parts) != 3:
                return False

            codec_name, profile, pix_fmt = output_parts

            # Essential iPhone compatibility checks
            codec_ok = codec_name == "h264"
            profile_ok = any(
                acceptable in profile.lower() for acceptable in ["main", "baseline"]
            )
            pixfmt_ok = pix_fmt == "yuv420p"

            return codec_ok and profile_ok and pixfmt_ok

        except (subprocess.SubprocessError, subprocess.TimeoutExpired):
            return False

    def clear_cache(self) -> None:
        """Clear the hardware detection cache."""
        self._hardware_cache = None
        self._cache_timestamp = None

    def get_cache_info(self) -> Dict[str, Any]:
        """Get hardware detection cache information."""
        if self._hardware_cache and self._cache_timestamp:
            age = time.time() - self._cache_timestamp
            return {
                "cached": True,
                "cache_age_seconds": age,
                "encoder_type": self._hardware_cache.get("encoder_type", "Unknown"),
                "expires_in": max(0, self.cache_timeout - age),
            }
        else:
            return {"cached": False}

    def test_specific_encoder(
        self, encoder_name: str, encoder_codec: str
    ) -> Dict[str, Any]:
        """
        Test a specific hardware encoder without caching.

        Args:
            encoder_name: Human-readable name (e.g., 'NVENC')
            encoder_codec: FFmpeg codec (e.g., 'h264_nvenc')

        Returns:
            Dictionary with detailed test results
        """
        diagnostics = {"tests_performed": [], "errors_encountered": []}

        # Check if encoder is available first
        available_encoders = self._list_available_encoders(diagnostics)
        if encoder_codec not in available_encoders:
            return {
                "success": False,
                "error_category": "encoder_not_available",
                "message": f"{encoder_codec} not available in FFmpeg",
                "diagnostics": diagnostics,
            }

        # Test the encoder
        test_result = self._test_hardware_encoder(
            encoder_name, encoder_codec, True, diagnostics
        )

        test_result["diagnostics"] = diagnostics
        return test_result


# Global detector instance for backwards compatibility
_global_detector = HardwareDetector()


def detect_optimal_codec_settings_enhanced() -> Tuple[
    Dict[str, Any], List[str], Dict[str, str]
]:
    """
    Legacy function for backwards compatibility.

    Replaces the original detect_optimal_codec_settings_enhanced() from utils.py
    """
    result = _global_detector.detect_optimal_settings()
    return (result["moviepy_params"], result["ffmpeg_params"], result["diagnostics"])


def detect_optimal_codec_settings() -> Tuple[Dict[str, Any], List[str]]:
    """
    Legacy function for backwards compatibility.

    Replaces the original detect_optimal_codec_settings() from utils.py
    """
    cpu_settings = _global_detector.get_cpu_settings()
    return (cpu_settings["moviepy_params"], cpu_settings["ffmpeg_params"])


# Export key classes and functions
__all__ = [
    "HardwareDetector",
    # Legacy compatibility functions
    "detect_optimal_codec_settings_enhanced",
    "detect_optimal_codec_settings",
]
