"""
Domain-specific type definitions for AutoCut video processing.

This module provides comprehensive type definitions for video processing operations,
hardware capabilities, and domain-specific data structures used throughout AutoCut.
"""

from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Literal,
    Optional,
    Protocol,
    Tuple,
    TypedDict,
    Union,
)

import numpy as np
from numpy.typing import NDArray

# Core path types
PathLike = Union[str, Path]

# Video processing pattern types
PatternType = Literal["energetic", "balanced", "buildup", "dramatic"]

# Hardware acceleration types
EncoderType = Literal["NVENC", "QSV", "CPU", "VAAPI", "VideoToolbox"]


class VideoMetadata(TypedDict):
    """Video file metadata information."""

    duration: float
    fps: float
    resolution: Tuple[int, int]
    codec: str
    bitrate: int
    file_size: int
    aspect_ratio: float


class AudioMetadata(TypedDict):
    """Audio file metadata information."""

    duration: float
    sample_rate: int
    channels: int
    bitrate: int
    format: str


class HardwareCapabilities(TypedDict):
    """System hardware acceleration capabilities."""

    gpu_acceleration: bool
    supported_codecs: List[str]
    max_resolution: Tuple[int, int]
    memory_limit: int
    encoder_type: EncoderType


class VideoAnalysisResult(TypedDict):
    """Results from video quality analysis."""

    quality_score: float
    motion_score: float
    shake_score: float
    face_count: int
    brightness: float
    contrast: float
    sharpness: float
    duration: float


class SceneSegment(TypedDict):
    """Individual scene segment with timing and quality metrics."""

    start_time: float
    end_time: float
    duration: float
    quality_score: float
    motion_score: float
    has_faces: bool
    brightness: float


class BeatsInfo(TypedDict):
    """Audio beat analysis information."""

    bpm: float
    beats: List[float]
    confidence: float
    tempo_stability: float


class ClipCandidate(TypedDict):
    """Potential video clip for timeline assembly."""

    video_path: PathLike
    start_time: float
    end_time: float
    duration: float
    quality_score: float
    beat_alignment: float
    selected: bool


class TimelineConfig(TypedDict):
    """Configuration for timeline assembly."""

    pattern: PatternType
    max_clips: int
    min_clip_duration: float
    max_clip_duration: float
    beat_sync_tolerance: float


class ProcessingProgress(TypedDict):
    """Progress information for video processing operations."""

    stage: str
    progress: float
    current_file: Optional[str]
    files_completed: int
    total_files: int
    estimated_time_remaining: float


class ValidationIssue(TypedDict):
    """Individual validation issue."""

    severity: Literal["error", "warning", "info"]
    code: str
    message: str
    file_path: Optional[PathLike]
    context: Dict[str, Any]


class ValidationResults(TypedDict):
    """Complete validation results."""

    is_valid: bool
    issues: List[ValidationIssue]
    metadata: VideoMetadata
    recommendations: List[str]


class RenderingConfig(TypedDict):
    """Configuration for final video rendering."""

    output_path: PathLike
    codec: str
    bitrate: str
    fps: int
    resolution: Tuple[int, int]
    hardware_acceleration: bool


# Protocol definitions for extensibility

class VideoAnalyzer(Protocol):
    """Protocol for video analysis implementations."""

    def analyze_frame(self, frame: NDArray[np.uint8]) -> Dict[str, Any]:
        """Analyze a single video frame."""
        ...

    def analyze_sequence(self, frames: List[NDArray[np.uint8]]) -> Dict[str, Any]:
        """Analyze a sequence of frames."""
        ...


class AudioAnalyzer(Protocol):
    """Protocol for audio analysis implementations."""

    def detect_bpm(self, audio_path: PathLike) -> float:
        """Detect BPM from audio file."""
        ...

    def extract_beats(self, audio_path: PathLike) -> BeatsInfo:
        """Extract beat timestamps from audio."""
        ...


class HardwareDetector(Protocol):
    """Protocol for hardware capability detection."""

    def detect_capabilities(self) -> HardwareCapabilities:
        """Detect available hardware acceleration."""
        ...

    def test_encoder(self, encoder_type: EncoderType) -> bool:
        """Test if specific encoder is functional."""
        ...


class ProgressCallback(Protocol):
    """Protocol for progress reporting callbacks."""

    def __call__(self, stage: str, progress: float, **context: Any) -> None:
        """Report processing progress."""
        ...


# Type aliases for complex compound types
VideoFileList = List[PathLike]
ClipCandidateList = List[ClipCandidate]
AnalysisResultDict = Dict[str, VideoAnalysisResult]
HardwareSettingsDict = Dict[str, Any]
ProcessingOptionsDict = Dict[str, Any]

# Function signature types for common operations
ProcessingFunction = Callable[[VideoFileList, PathLike, PathLike], str]
ValidationFunction = Callable[[PathLike], ValidationResults] 
AnalysisFunction = Callable[[PathLike], VideoAnalysisResult]


# Export all types for use in other modules
__all__ = [
    # Basic types
    "PathLike",
    "PatternType", 
    "EncoderType",

    # Metadata types
    "VideoMetadata",
    "AudioMetadata",
    "HardwareCapabilities",

    # Analysis types
    "VideoAnalysisResult",
    "SceneSegment",
    "BeatsInfo",

    # Assembly types
    "ClipCandidate",
    "TimelineConfig",

    # Processing types
    "ProcessingProgress",
    "ValidationIssue",
    "ValidationResults", 
    "RenderingConfig",

    # Protocols
    "VideoAnalyzer",
    "AudioAnalyzer",
    "HardwareDetector",
    "ProgressCallback",

    # Type aliases
    "VideoFileList",
    "ClipCandidateList",
    "AnalysisResultDict",
    "HardwareSettingsDict",
    "ProcessingOptionsDict",

    # Function types
    "ProcessingFunction",
    "ValidationFunction",
    "AnalysisFunction",
]