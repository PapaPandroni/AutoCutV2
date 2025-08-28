"""
AutoCut V2 API - Clean Public Interface

Provides a clean, well-documented API for AutoCut functionality.
This module serves as the primary interface for all AutoCut operations.
"""

import glob
import os
import time
from dataclasses import dataclass
from pathlib import Path, Path as PathType
from typing import Any, Dict, List, Optional, Union, cast

# Import core AutoCut modules (src is in path, so no relative imports needed)
from clip_assembler import assemble_clips
from hardware.detection import HardwareDetector
from utils import SUPPORTED_VIDEO_FORMATS, find_all_video_files
from video.validation import ValidationResult, VideoValidator


@dataclass
class SystemInfo:
    """System capability information"""

    has_hardware_acceleration: bool
    available_encoders: List[str]
    cpu_cores: int
    ffmpeg_available: bool
    platform: str


@dataclass
class DiagnosticReport:
    """Comprehensive system diagnostic report"""

    ffmpeg_version: str
    moviepy_version: str
    platform: str
    performance_metrics: Dict[str, Any]
    issues: List[str]
    recommendations: List[str]


@dataclass
class DemoResult:
    """Result from running demo"""

    success: bool
    output_path: Optional[str]
    processing_time: float
    error_message: Optional[str]


class AutoCutAPI:
    """
    Clean public API for AutoCut functionality

    This class provides a high-level interface to all AutoCut operations,
    including video processing, validation, system diagnostics, and demos.
    """

    def __init__(self) -> None:
        """Initialize AutoCut API with required components"""
        self.validator = VideoValidator()
        self.hardware_detector = HardwareDetector()

    def process_videos(
        self,
        video_files: List[Union[str, PathType]],
        audio_file: Union[str, PathType],
        output_path: Union[str, PathType],
        pattern: str = "balanced",
        memory_safe: bool = False,
        verbose: bool = False,
    ) -> str:
        """
        Main video processing function

        Args:
            video_files: List of video file paths
            audio_file: Path to audio file for synchronization
            output_path: Path for output video
            pattern: Editing pattern ('energetic', 'balanced', 'dramatic', 'buildup')
            memory_safe: Enable memory-safe processing (single worker)
            verbose: Enable verbose logging

        Returns:
            Path to created video file

        Raises:
            ValueError: If input files are invalid
            RuntimeError: If processing fails
        """
        if verbose:
            pass

        # Validate inputs
        for video_file in video_files:
            if not Path(video_file).exists():
                raise ValueError(f"Video file not found: {video_file}")

        if not Path(audio_file).exists():
            raise ValueError(f"Audio file not found: {audio_file}")

        # Create output directory
        output_dir = Path(output_path).parent
        if output_dir != Path():
            output_dir.mkdir(parents=True, exist_ok=True)

        # Set up progress callback for verbose mode
        progress_callback = None
        if verbose:

            def progress_callback(step: str, progress: float) -> None:
                bar_length = 30
                filled = int(bar_length * progress)
                bar = "█" * filled + "░" * (bar_length - filled)

        try:
            # Call core assembler function with dynamic or memory-optimized workers
            if memory_safe:
                # Force single worker for memory-safe mode
                max_workers = 1
            else:
                # Use automatic detection (None = dynamic profiling)
                max_workers = None

            result_path = assemble_clips(
                video_files=video_files,
                audio_file=audio_file,
                output_path=output_path,
                pattern=pattern,
                max_workers=max_workers,
                progress_callback=progress_callback,
            )

            if verbose:
                pass

            return result_path

        except Exception as e:
            raise RuntimeError(f"Video processing failed: {e!s}") from e

    def validate_video(
        self,
        video_path: Union[str, PathType],
        detailed: bool = False,
    ) -> ValidationResult:
        """
        Check video compatibility and quality

        Args:
            video_path: Path to video file
            detailed: Include detailed metadata and suggestions

        Returns:
            ValidationResult with compatibility information

        Raises:
            FileNotFoundError: If video file doesn't exist
        """
        if not Path(video_path).exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")

        # Use VideoValidator for comprehensive validation
        if detailed:
            result = self.validator.validate_iphone_compatibility(video_path)
        else:
            result = self.validator.validate_basic_format(video_path)

        return result

    def get_system_info(self) -> SystemInfo:
        """
        Get hardware acceleration capabilities and system info

        Returns:
            SystemInfo object with system capabilities
        """
        # Get hardware acceleration info using the correct method
        try:
            hw_settings = self.hardware_detector.detect_optimal_settings("fast")
            has_acceleration = hw_settings.get("encoder_type", "CPU") != "CPU"
            available_encoders = []

            # Try to get available encoders
            if has_acceleration:
                encoder_type = hw_settings.get("encoder_type", "CPU")
                available_encoders.append(encoder_type)
            available_encoders.append("CPU")  # CPU is always available

        except Exception:
            # Fallback if hardware detection fails
            has_acceleration = False
            available_encoders = ["CPU"]

        # Get CPU count
        try:
            import multiprocessing

            cpu_cores = multiprocessing.cpu_count()
        except:
            cpu_cores = 1

        # Check FFmpeg availability
        ffmpeg_available = True
        try:
            import subprocess

            result = subprocess.run(
                ["ffmpeg", "-version"],
                check=False,
                capture_output=True,
                text=True,
                timeout=5,
            )
            ffmpeg_available = result.returncode == 0
        except:
            ffmpeg_available = False

        # Get platform info
        import platform

        platform_name = f"{platform.system()} {platform.release()}"

        return SystemInfo(
            has_hardware_acceleration=has_acceleration,
            available_encoders=available_encoders,
            cpu_cores=cpu_cores,
            ffmpeg_available=ffmpeg_available,
            platform=platform_name,
        )

    def run_diagnostics(self) -> DiagnosticReport:
        """
        Run comprehensive system diagnostics

        Returns:
            DiagnosticReport with detailed system analysis
        """
        import platform
        import subprocess

        issues = []
        recommendations = []
        performance_metrics = {}

        # Get FFmpeg version
        ffmpeg_version = "Not available"
        try:
            result = subprocess.run(
                ["ffmpeg", "-version"],
                check=False,
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode == 0:
                # Extract version from first line
                first_line = result.stdout.split("\\n")[0]
                ffmpeg_version = (
                    first_line.split(" ")[2]
                    if len(first_line.split(" ")) > 2
                    else "Unknown"
                )
            else:
                issues.append("FFmpeg not available or not working")
                recommendations.append("Install FFmpeg for video processing support")
        except Exception as e:
            issues.append(f"FFmpeg check failed: {e!s}")
            recommendations.append(
                "Install FFmpeg: apt install ffmpeg (Ubuntu) or brew install ffmpeg (macOS)",
            )

        # Get MoviePy version
        moviepy_version = "Unknown"
        try:
            import moviepy

            moviepy_version = moviepy.__version__
        except ImportError:
            issues.append("MoviePy not available")
            recommendations.append("Install MoviePy: pip install moviepy")
        except AttributeError:
            moviepy_version = "Version info unavailable"

        # Hardware acceleration check and performance metrics
        system_info = self.get_system_info()
        if not system_info.has_hardware_acceleration:
            recommendations.append(
                "Consider enabling hardware acceleration for faster processing",
            )

        performance_metrics.update(
            {
                "CPU Cores": system_info.cpu_cores,
                "Hardware Acceleration": system_info.has_hardware_acceleration,
                "Available Encoders": len(system_info.available_encoders),
            },
        )

        # Test media directory check
        if not Path("test_media").exists():
            issues.append("test_media directory not found")
            recommendations.append(
                "Create test_media/ directory with sample video/audio files for testing",
            )

        return DiagnosticReport(
            ffmpeg_version=ffmpeg_version,
            moviepy_version=moviepy_version,
            platform=f"{platform.system()} {platform.release()}",
            performance_metrics=performance_metrics,
            issues=issues,
            recommendations=recommendations,
        )

    def run_demo(
        self,
        quick: bool = False,
        pattern: str = "balanced",
        test_media_dir: str = "test_media",
    ) -> DemoResult:
        """
        Run AutoCut demonstration

        Args:
            quick: Use limited files for faster demo
            pattern: Editing pattern to use
            test_media_dir: Directory containing test media

        Returns:
            DemoResult with demo outcome
        """
        start_time = time.time()

        try:
            # Find video files
            video_files = find_all_video_files(test_media_dir)

            if not video_files:
                return DemoResult(
                    success=False,
                    output_path=None,
                    processing_time=0,
                    error_message=f"No video files found in {test_media_dir}",
                )

            # Limit videos for quick demo
            if quick and len(video_files) > 3:
                video_files = video_files[:3]

            # Find audio file
            audio_extensions = ["*.mp3", "*.wav", "*.m4a", "*.flac", "*.aac", "*.ogg"]
            audio_files = []
            for ext in audio_extensions:
                audio_files.extend(glob.glob(f"{test_media_dir}/{ext}"))

            if not audio_files:
                return DemoResult(
                    success=False,
                    output_path=None,
                    processing_time=0,
                    error_message=f"No audio files found in {test_media_dir}",
                )

            # Generate output path
            timestamp = int(time.time())
            output_path = f"output/autocut_demo_{pattern}_{timestamp}.mp4"

            # Process video
            result_path = self.process_videos(
                video_files=video_files,
                audio_file=audio_files[0],
                output_path=output_path,
                pattern=pattern,
                verbose=True,
            )

            processing_time = time.time() - start_time

            return DemoResult(
                success=True,
                output_path=result_path,
                processing_time=processing_time,
                error_message=None,
            )

        except Exception as e:
            processing_time = time.time() - start_time
            return DemoResult(
                success=False,
                output_path=None,
                processing_time=processing_time,
                error_message=str(e),
            )

    def find_supported_videos(self, directory: Union[str, PathType]) -> List[str]:
        """
        Find all supported video files in directory

        Args:
            directory: Directory to search

        Returns:
            List of supported video file paths
        """
        return find_all_video_files(directory)

    def get_supported_formats(self) -> Dict[str, Union[List[str], int]]:
        """
        Get information about supported file formats

        Returns:
            Dictionary with video and audio format information
        """
        audio_formats = [".mp3", ".wav", ".m4a", ".flac", ".aac", ".ogg"]

        return {
            "video_formats": list(SUPPORTED_VIDEO_FORMATS),
            "audio_formats": audio_formats,
            "total_video_formats": len(SUPPORTED_VIDEO_FORMATS),
            "total_audio_formats": len(audio_formats),
        }
