"""
System Profiler for AutoCut - Dynamic Worker Detection

Provides comprehensive system analysis to automatically determine optimal
worker counts based on available memory, CPU cores, hardware acceleration,
and video file characteristics.
"""

import os
import platform
from dataclasses import dataclass
from typing import Any, Dict, List

try:
    import psutil
except ImportError:
    psutil = None

try:
    from .hardware.detection import HardwareDetector
    from .video.codec_detection import CodecDetector
except ImportError:
    # Fallback for direct execution
    try:
        from hardware.detection import HardwareDetector
        from video.codec_detection import CodecDetector
    except ImportError:
        # Final fallback - try importing from current directory context
        try:
            import sys

            sys.path.append(os.path.dirname(__file__))
            from hardware.detection import HardwareDetector
            from video.codec_detection import CodecDetector
        except ImportError:
            CodecDetector = None
            HardwareDetector = None


@dataclass
class SystemCapabilities:
    """Comprehensive system capabilities profile"""

    memory_total_gb: float
    memory_available_gb: float
    memory_percent_used: float
    cpu_cores: int
    cpu_threads: int
    cpu_frequency_ghz: float
    platform: str
    architecture: str
    has_hardware_acceleration: bool
    hardware_encoder_type: str
    apple_silicon: bool
    unified_memory: bool
    performance_score: float


@dataclass
class VideoMemoryProfile:
    """Memory usage profile for video processing"""

    estimated_memory_per_video_mb: float
    codec_complexity_factor: float
    resolution_factor: float
    file_size_expansion_ratio: float
    confidence_score: float  # 0-1, how confident we are in the estimate


class SystemProfiler:
    """Comprehensive system profiler for dynamic worker optimization"""

    def __init__(self):
        self.codec_detector = CodecDetector() if CodecDetector else None
        self.hardware_detector = HardwareDetector() if HardwareDetector else None
        self._cpu_benchmark_cache = None

    def get_system_capabilities(self) -> SystemCapabilities:
        """Get comprehensive system capability analysis"""

        # Memory analysis
        if psutil:
            memory = psutil.virtual_memory()
            memory_total_gb = memory.total / (1024**3)
            memory_available_gb = memory.available / (1024**3)
            memory_percent_used = memory.percent
        else:
            # Fallback estimates if psutil not available
            memory_total_gb = 8.0  # Conservative estimate
            memory_available_gb = 4.0
            memory_percent_used = 50.0

        # CPU analysis
        if psutil:
            cpu_cores = psutil.cpu_count(logical=False) or 1
            cpu_threads = psutil.cpu_count(logical=True) or 1
            try:
                cpu_freq = psutil.cpu_freq()
                cpu_frequency_ghz = (
                    cpu_freq.max / 1000.0 if cpu_freq and cpu_freq.max else 2.0
                )
            except Exception:
                cpu_frequency_ghz = 2.0  # Fallback
        else:
            import multiprocessing

            cpu_cores = multiprocessing.cpu_count()
            cpu_threads = cpu_cores
            cpu_frequency_ghz = 2.0  # Fallback estimate

        # Platform detection
        platform_name = platform.system()
        architecture = platform.machine().lower()

        # Hardware acceleration detection
        has_hw_accel = False
        hw_encoder_type = "CPU"
        if self.hardware_detector:
            try:
                hw_info = self.hardware_detector.detect_optimal_settings("fast")
                has_hw_accel = hw_info.get("encoder_type", "CPU") != "CPU"
                hw_encoder_type = hw_info.get("encoder_type", "CPU")
            except Exception:
                pass  # Hardware detection failed

        # Apple Silicon detection
        apple_silicon = platform_name == "Darwin" and (
            "arm64" in architecture or "arm" in architecture
        )
        unified_memory = apple_silicon  # Apple Silicon has unified memory

        # Performance scoring
        performance_score = self._calculate_performance_score(
            cpu_cores,
            cpu_frequency_ghz,
            memory_total_gb,
            has_hw_accel,
            apple_silicon,
        )

        return SystemCapabilities(
            memory_total_gb=memory_total_gb,
            memory_available_gb=memory_available_gb,
            memory_percent_used=memory_percent_used,
            cpu_cores=cpu_cores,
            cpu_threads=cpu_threads,
            cpu_frequency_ghz=cpu_frequency_ghz,
            platform=platform_name,
            architecture=architecture,
            has_hardware_acceleration=has_hw_accel,
            hardware_encoder_type=hw_encoder_type,
            apple_silicon=apple_silicon,
            unified_memory=unified_memory,
            performance_score=performance_score,
        )

    def _calculate_performance_score(
        self,
        cpu_cores: int,
        cpu_freq: float,
        memory_gb: float,
        has_hw_accel: bool,
        apple_silicon: bool,
    ) -> float:
        """Calculate overall system performance score (0-100)"""

        # Base score from CPU
        cpu_score = min(50, (cpu_cores * cpu_freq * 5))  # Max 50 points

        # Memory contribution
        memory_score = min(25, memory_gb * 2)  # Max 25 points

        # Hardware acceleration bonus
        hw_accel_bonus = 15 if has_hw_accel else 0

        # Apple Silicon bonus (unified memory + efficiency)
        apple_bonus = 10 if apple_silicon else 0

        total_score = cpu_score + memory_score + hw_accel_bonus + apple_bonus
        return min(100.0, total_score)

    def estimate_video_memory_usage(
        self,
        video_files: List[str],
        sample_count: int = 3,
    ) -> VideoMemoryProfile:
        """Estimate memory usage per video based on file analysis"""

        if not video_files:
            # Fallback conservative estimate
            return VideoMemoryProfile(
                estimated_memory_per_video_mb=200.0,
                codec_complexity_factor=1.5,
                resolution_factor=1.0,
                file_size_expansion_ratio=4.0,
                confidence_score=0.1,
            )

        # Analyze sample of videos (up to sample_count)
        sample_files = video_files[: min(sample_count, len(video_files))]
        memory_estimates = []
        codec_factors = []
        resolution_factors = []
        expansion_ratios = []

        def _analyze_video_with_fallback(video_file):
            """Analyze single video memory with fallback values."""
            try:
                estimate = self._analyze_single_video_memory(video_file)
                return {
                    "memory_mb": estimate["memory_mb"],
                    "codec_factor": estimate["codec_factor"],
                    "resolution_factor": estimate["resolution_factor"],
                    "expansion_ratio": estimate["expansion_ratio"]
                }
            except Exception as e:
                # Use conservative fallback for failed analysis
                return {
                    "memory_mb": 200.0,
                    "codec_factor": 1.5,
                    "resolution_factor": 1.2,
                    "expansion_ratio": 4.0
                }

        for video_file in sample_files:
            estimate = _analyze_video_with_fallback(video_file)
            memory_estimates.append(estimate["memory_mb"])
            codec_factors.append(estimate["codec_factor"])
            resolution_factors.append(estimate["resolution_factor"])
            expansion_ratios.append(estimate["expansion_ratio"])

        # Calculate averages
        avg_memory = sum(memory_estimates) / len(memory_estimates)
        avg_codec_factor = sum(codec_factors) / len(codec_factors)
        avg_resolution_factor = sum(resolution_factors) / len(resolution_factors)
        avg_expansion_ratio = sum(expansion_ratios) / len(expansion_ratios)

        # Confidence score based on sample size and analysis success
        confidence = min(1.0, len(sample_files) / max(1, sample_count))

        return VideoMemoryProfile(
            estimated_memory_per_video_mb=avg_memory,
            codec_complexity_factor=avg_codec_factor,
            resolution_factor=avg_resolution_factor,
            file_size_expansion_ratio=avg_expansion_ratio,
            confidence_score=confidence,
        )

    def _analyze_single_video_memory(self, video_file: str) -> Dict[str, float]:
        """Analyze a single video file for memory estimation"""

        # Get file size
        file_size_mb = os.path.getsize(video_file) / (1024 * 1024)

        # Codec analysis
        codec_factor = 1.0
        resolution_factor = 1.0

        if self.codec_detector:
            try:
                codec_info = self.codec_detector.detect_video_codec(video_file)

                # Codec complexity adjustment
                codec = codec_info.get("codec", "unknown").lower()
                if "hevc" in codec or "h265" in codec:
                    codec_factor = 1.3  # H.265 more CPU/memory intensive
                elif "av1" in codec:
                    codec_factor = 1.5  # AV1 very intensive
                else:
                    codec_factor = 1.0  # H.264 baseline

                # Resolution adjustment
                width = codec_info.get("width", 1920)
                height = codec_info.get("height", 1080)
                pixels = width * height

                # Resolution factor relative to 1080p
                baseline_pixels = 1920 * 1080
                resolution_factor = max(0.5, pixels / baseline_pixels)

            except Exception:
                # Fallback if codec detection fails
                codec_factor = 1.2  # Conservative estimate
                resolution_factor = 1.1

        # Memory estimation
        base_expansion_factor = 3.5  # Base file size to memory expansion
        total_expansion = base_expansion_factor * codec_factor * resolution_factor
        estimated_memory_mb = file_size_mb * total_expansion

        return {
            "memory_mb": estimated_memory_mb,
            "codec_factor": codec_factor,
            "resolution_factor": resolution_factor,
            "expansion_ratio": total_expansion,
        }

    def calculate_optimal_workers(
        self,
        capabilities: SystemCapabilities,
        video_profile: VideoMemoryProfile,
        video_count: int,
        target_memory_usage: float = 0.7,
    ) -> Dict[str, Any]:
        """Calculate optimal worker count with detailed reasoning"""

        # Memory-based calculation
        usable_memory_gb = capabilities.memory_available_gb * target_memory_usage
        usable_memory_mb = usable_memory_gb * 1024

        memory_per_worker_mb = video_profile.estimated_memory_per_video_mb
        max_workers_memory = int(
            usable_memory_mb / max(50, memory_per_worker_mb),
        )  # Min 50MB per worker

        # CPU-based calculation
        # Use 75% of CPU cores to leave room for other system processes
        max_workers_cpu = max(1, int(capabilities.cpu_cores * 0.75))

        # Hardware acceleration bonus
        hw_multiplier = 1.0
        if capabilities.has_hardware_acceleration:
            hw_multiplier = 1.4  # 40% bonus for hardware acceleration

        if capabilities.apple_silicon:
            hw_multiplier *= 1.2  # Additional 20% bonus for Apple Silicon efficiency

        # Performance-based adjustment
        if capabilities.performance_score > 80:
            performance_multiplier = 1.3  # High-performance systems can handle more
        elif capabilities.performance_score > 60:
            performance_multiplier = 1.1  # Good systems get slight bonus
        else:
            performance_multiplier = 0.9  # Lower-end systems get reduction

        # Calculate final worker count
        base_workers = min(max_workers_memory, max_workers_cpu, video_count)
        adjusted_workers = int(base_workers * hw_multiplier * performance_multiplier)

        # Apply safety bounds
        min_workers = 1
        max_workers_absolute = min(
            12,
            capabilities.cpu_cores * 2,
        )  # Never exceed 2x CPU cores or 12
        optimal_workers = max(min_workers, min(adjusted_workers, max_workers_absolute))

        return {
            "optimal_workers": optimal_workers,
            "reasoning": {
                "memory_limited_workers": max_workers_memory,
                "cpu_limited_workers": max_workers_cpu,
                "hw_acceleration_bonus": hw_multiplier,
                "performance_multiplier": performance_multiplier,
                "memory_per_worker_mb": memory_per_worker_mb,
                "target_memory_usage_percent": target_memory_usage * 100,
                "confidence_score": video_profile.confidence_score,
            },
        }

    def print_system_analysis(
        self,
        capabilities: SystemCapabilities,
        video_profile: VideoMemoryProfile,
        worker_analysis: Dict[str, Any],
    ):
        """Print detailed system analysis for user visibility"""

        if capabilities.has_hardware_acceleration:
            pass
        else:
            pass

        if capabilities.apple_silicon:
            pass

        reasoning = worker_analysis["reasoning"]

        # Memory usage prediction
        predicted_memory_mb = (
            worker_analysis["optimal_workers"] * reasoning["memory_per_worker_mb"]
        )
        predicted_memory_gb = predicted_memory_mb / 1024
        memory_usage_percent = (
            predicted_memory_gb / capabilities.memory_available_gb
        ) * 100

        if memory_usage_percent > 80 or memory_usage_percent > 60:
            pass
        else:
            pass
