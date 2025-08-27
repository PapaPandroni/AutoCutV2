"""
Video codec detection and analysis for AutoCut V2.

This module provides comprehensive video codec detection and analysis
functionality, extracted from the monolithic utils.py. Includes enhanced
compatibility checking and detailed metadata extraction.
"""

import json
import os
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Tuple

# Import our custom exceptions with dual import pattern
try:
    # Try absolute import first for autocut.py execution context
    from core.exceptions import VideoProcessingError
except ImportError:
    # Fallback to relative import for package execution context
    from ..core.exceptions import VideoProcessingError


class CodecDetector:
    """
    Comprehensive video codec detection and analysis system.

    Provides detailed codec information extraction using FFprobe with
    caching for performance and enhanced compatibility scoring for
    MoviePy integration planning.
    """

    def __init__(self, cache_timeout: float = 300.0):
        """
        Initialize codec detector with optional caching.

        Args:
            cache_timeout: How long to cache FFprobe results (seconds)
        """
        self._codec_cache: Dict[str, Dict[str, Any]] = {}
        self._cache_timeout = cache_timeout

    def detect_video_codec(
        self,
        file_path: str,
        use_cache: bool = True,
    ) -> Dict[str, Any]:
        """
        Detect video codec and format information using FFprobe with enhanced compatibility checking.

        Replaces: detect_video_codec() from utils.py

        Args:
            file_path: Path to the video file
            use_cache: Whether to use cached results if available

        Returns:
            Dictionary containing comprehensive codec information:
            - 'codec': Video codec name (e.g., 'h264', 'hevc')
            - 'is_hevc': Boolean indicating if codec is H.265/HEVC
            - 'resolution': (width, height) tuple
            - 'fps': Frame rate
            - 'duration': Video duration in seconds
            - 'container': Container format (mp4, mov, etc.)
            - 'compatibility_score': 0-100 score for MoviePy compatibility
            - 'warnings': List of potential compatibility issues
            - Additional technical details (profile, level, bitrate, etc.)

        Raises:
            VideoProcessingError: If codec detection fails
            FileNotFoundError: If file doesn't exist or FFprobe not available
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Video file not found: {file_path}")

        # Check cache first
        if use_cache:
            cache_key = f"{file_path}:{os.path.getmtime(file_path)}"
            if cache_key in self._codec_cache:
                return self._codec_cache[cache_key]

        try:
            # Use FFprobe to get comprehensive video stream information
            cmd = [
                "ffprobe",
                "-v",
                "quiet",
                "-print_format",
                "json",
                "-show_streams",
                "-show_format",
                "-select_streams",
                "v:0",
                file_path,
            ]

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True,
                timeout=15,
            )
            data = json.loads(result.stdout)

            if not data.get("streams"):
                raise VideoProcessingError(f"No video streams found in {file_path}")

            video_stream = data["streams"][0]
            format_info = data.get("format", {})
            codec_name = video_stream.get("codec_name", "").lower()

            # Enhanced codec detection with variants
            standard_codec = self._standardize_codec_name(codec_name)

            # Parse frame rate with enhanced handling
            fps = self._parse_frame_rate(video_stream.get("r_frame_rate", "30/1"))

            # Extract container format
            container = Path(file_path).suffix.lower().lstrip(".")
            format_name = format_info.get("format_name", "").lower()

            # Calculate compatibility score and warnings
            compatibility_score, warnings = self._calculate_compatibility_score(
                standard_codec,
                container,
                format_name,
                video_stream,
                format_info,
            )

            codec_info = {
                "codec": standard_codec,
                "codec_raw": codec_name,  # Original codec name from FFprobe
                "is_hevc": standard_codec == "hevc",
                "resolution": (
                    int(video_stream.get("width", 0)),
                    int(video_stream.get("height", 0)),
                ),
                "fps": fps,
                "duration": float(
                    video_stream.get("duration", format_info.get("duration", 0)),
                ),
                "pixel_format": video_stream.get("pix_fmt", "unknown"),
                "container": container,
                "format_name": format_name,
                "compatibility_score": compatibility_score,
                "warnings": warnings,
                # Additional technical details
                "bitrate": int(video_stream.get("bit_rate", 0)),
                "profile": video_stream.get("profile", "unknown"),
                "level": video_stream.get("level", "unknown"),
                "color_space": video_stream.get("color_space", "unknown"),
                "has_audio": len(
                    [
                        s
                        for s in data.get("streams", [])
                        if s.get("codec_type") == "audio"
                    ],
                )
                > 0,
                "file_size": int(format_info.get("size", 0)),
            }

            # Cache result if requested
            if use_cache:
                self._codec_cache[cache_key] = codec_info

            return codec_info

        except subprocess.CalledProcessError as e:
            raise VideoProcessingError(
                f"FFprobe failed for {file_path}",
                details={"stderr": e.stderr, "command": " ".join(cmd)},
            ) from e
        except subprocess.TimeoutExpired:
            raise VideoProcessingError(f"FFprobe timed out for {file_path}") from None
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            raise VideoProcessingError(
                f"Failed to parse video information for {file_path}: {e!s}",
            ) from e

    def _standardize_codec_name(self, codec_name: str) -> str:
        """Convert various codec name variants to standard names."""
        codec_variants = {
            "h264": ["h264", "avc", "avc1", "h.264"],
            "hevc": ["hevc", "h265", "h.265", "hvc1", "hev1"],
            "vp8": ["vp8"],
            "vp9": ["vp9"],
            "av1": ["av1"],
            "mpeg4": ["mpeg4", "mp4v", "xvid", "divx"],
            "mpeg2": ["mpeg2video", "mpeg2"],
            "mpeg1": ["mpeg1video", "mpeg1"],
        }

        # Find standard codec name
        for standard, variants in codec_variants.items():
            if codec_name in variants:
                return standard

        return codec_name  # Return original if no match found

    def _parse_frame_rate(self, fps_str: str) -> float:
        """Parse frame rate string with enhanced error handling."""
        try:
            if "/" in fps_str:
                num, den = map(int, fps_str.split("/"))
                return num / den if den != 0 else 30.0
            return float(fps_str)
        except (ValueError, ZeroDivisionError, TypeError):
            return 30.0  # Fallback to standard frame rate

    def _calculate_compatibility_score(
        self,
        codec: str,
        container: str,
        format_name: str,
        video_stream: dict,
        format_info: dict,
    ) -> Tuple[int, List[str]]:
        """
        Calculate MoviePy compatibility score and identify potential issues.

        Args:
            codec: Standardized codec name
            container: Container format (file extension)
            format_name: FFprobe format name
            video_stream: Video stream information from FFprobe
            format_info: Format information from FFprobe

        Returns:
            Tuple of (compatibility_score 0-100, list of warning messages)
        """
        score = 100
        warnings = []

        # Codec compatibility scoring
        codec_scores = {
            "h264": 100,  # Excellent compatibility
            "mpeg4": 90,  # Very good
            "vp8": 80,  # Good (web formats)
            "mpeg2": 70,  # Decent
            "hevc": 40,  # Poor (requires transcoding for iPhone compatibility)
            "vp9": 50,  # Limited support
            "av1": 30,  # Poor support
        }

        codec_score = codec_scores.get(codec, 40)  # Default for unknown codecs
        score = min(score, codec_score)

        if codec == "hevc":
            warnings.append(
                "H.265/HEVC may require transcoding for optimal compatibility",
            )
        elif codec not in codec_scores:
            warnings.append(f"Unknown codec '{codec}' may cause compatibility issues")

        # Container compatibility
        container_scores = {
            "mp4": 100,
            "mov": 95,
            "avi": 90,
            "mkv": 85,
            "webm": 80,
            "m4v": 90,
            "flv": 70,
            "wmv": 65,
            "3gp": 60,
            "vob": 50,
            "ts": 55,
            "mts": 55,
            "m2ts": 55,
        }

        container_score = container_scores.get(container, 50)
        score = min(score, container_score)

        if container_score < 70:
            warnings.append(f"Container format '{container}' may have limited support")

        # Resolution warnings
        width = int(video_stream.get("width", 0))
        height = int(video_stream.get("height", 0))

        if width > 3840 or height > 2160:
            score -= 10
            warnings.append("Very high resolution (>4K) may cause performance issues")
        elif width > 1920 or height > 1080:
            score -= 5
            warnings.append("High resolution may require more processing power")

        # Frame rate warnings
        fps_str = video_stream.get("r_frame_rate", "30/1")
        fps = self._parse_frame_rate(fps_str)

        if fps > 60:
            score -= 10
            warnings.append("High frame rate (>60fps) may cause performance issues")
        elif fps > 30:
            score -= 5
            warnings.append("High frame rate may require more processing power")

        # Pixel format compatibility
        pixel_format = video_stream.get("pix_fmt", "")
        if pixel_format in ["yuv420p10le", "yuv422p10le", "yuv444p10le"]:
            score -= 15
            warnings.append("10-bit video may have limited compatibility")
        elif pixel_format and "yuv420p" not in pixel_format:
            score -= 5
            warnings.append(f"Pixel format '{pixel_format}' may need conversion")

        # Profile/level warnings for H.264/H.265
        profile = video_stream.get("profile", "").lower()
        if codec == "hevc" and "main10" in profile:
            score -= 20
            warnings.append("H.265 Main10 profile may require transcoding")
        elif codec == "h264" and "high" in profile and "4:4:4" in profile:
            score -= 10
            warnings.append("H.264 High 4:4:4 profile may have limited support")

        # Bitrate warnings (if available)
        bitrate = int(video_stream.get("bit_rate", 0))
        if bitrate > 50_000_000:  # >50 Mbps
            score -= 10
            warnings.append("Very high bitrate may cause performance issues")

        # Duration warnings
        duration = float(video_stream.get("duration", format_info.get("duration", 0)))
        if duration > 3600:  # >1 hour
            score -= 5
            warnings.append("Long video duration may require more memory")

        return max(0, min(100, score)), warnings

    def get_iphone_compatibility_info(self, file_path: str) -> Dict[str, Any]:
        """
        Get iPhone H.265 specific compatibility information.

        Args:
            file_path: Path to video file

        Returns:
            Dictionary with iPhone-specific compatibility analysis
        """
        codec_info = self.detect_video_codec(file_path)

        iphone_info = {
            "is_iphone_compatible": True,
            "requires_transcoding": False,
            "compatibility_issues": [],
            "recommendations": [],
        }

        # Check codec
        if codec_info["codec"] == "hevc":
            iphone_info["requires_transcoding"] = True
            iphone_info["compatibility_issues"].append(
                "H.265/HEVC codec requires transcoding to H.264 for iPhone compatibility",
            )
            iphone_info["recommendations"].append(
                "Transcode to H.264 with Main or Baseline profile",
            )
        elif codec_info["codec"] != "h264":
            iphone_info["is_iphone_compatible"] = False
            iphone_info["compatibility_issues"].append(
                f"Codec {codec_info['codec']} is not iPhone compatible",
            )
            iphone_info["recommendations"].append(
                "Convert to H.264 codec for iPhone compatibility",
            )

        # Check pixel format
        if "10" in codec_info["pixel_format"]:
            iphone_info["requires_transcoding"] = True
            iphone_info["compatibility_issues"].append(
                f"10-bit pixel format ({codec_info['pixel_format']}) requires conversion to 8-bit",
            )
            iphone_info["recommendations"].append(
                "Convert to yuv420p (8-bit) pixel format",
            )

        # Check profile
        profile = codec_info.get("profile", "").lower()
        if "high 10" in profile or "main 10" in profile:
            iphone_info["requires_transcoding"] = True
            iphone_info["compatibility_issues"].append(
                f"Profile {profile} is 10-bit and requires conversion",
            )
            iphone_info["recommendations"].append(
                "Convert to Main or Baseline profile (8-bit)",
            )

        # Check container
        if codec_info["container"] not in ["mp4", "mov", "m4v"]:
            iphone_info["compatibility_issues"].append(
                f"Container {codec_info['container']} has limited iPhone compatibility",
            )
            iphone_info["recommendations"].append(
                "Use MP4 or MOV container for best iPhone compatibility",
            )

        # Update overall compatibility
        if (
            iphone_info["compatibility_issues"]
            and not iphone_info["requires_transcoding"]
        ):
            iphone_info["is_iphone_compatible"] = False

        return iphone_info

    def clear_cache(self) -> None:
        """Clear the codec detection cache."""
        self._codec_cache.clear()

    def get_cache_stats(self) -> Dict[str, int]:
        """Get cache statistics."""
        return {
            "cached_files": len(self._codec_cache),
            "cache_size_bytes": sum(
                len(str(info)) for info in self._codec_cache.values()
            ),
        }


# Global detector instance for backwards compatibility
_global_detector = CodecDetector()


def detect_video_codec(file_path: str) -> Dict[str, Any]:
    """
    Backwards compatibility function for existing code.

    This function maintains compatibility with the original detect_video_codec()
    from utils.py while using the new CodecDetector class.

    Args:
        file_path: Path to the video file

    Returns:
        Dictionary containing comprehensive codec information
    """
    return _global_detector.detect_video_codec(file_path)


# Export key functions and classes
__all__ = [
    "CodecDetector",
    "detect_video_codec",  # Backwards compatibility
]
