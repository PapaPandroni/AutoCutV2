"""
Memory-efficient streaming video analyzer for AutoCut V2.

Uses FFmpeg-based streaming to analyze videos without loading entire clips
into memory, dramatically reducing memory usage while maintaining analysis quality.
"""

import os
import logging
import time
import numpy as np
import cv2
from typing import Dict, Any, List, Optional, Tuple, Iterator
from dataclasses import dataclass
from pathlib import Path

try:
    # Try to import DeFFcode for optimal FFmpeg integration
    from deffcode import FFdecoder
    DEFFCODE_AVAILABLE = True
except ImportError:
    DEFFCODE_AVAILABLE = False
    FFdecoder = None

try:
    # Fallback to PyAV for low-level video operations
    import av
    PYAV_AVAILABLE = True
except ImportError:
    PYAV_AVAILABLE = False
    av = None

from .hardware_accelerator import M2HardwareAccelerator
from .memory_monitor import MemoryMonitor

logger = logging.getLogger(__name__)


@dataclass
class FrameAnalysis:
    """Analysis results for a single frame."""
    frame_index: int
    timestamp: float
    quality_score: float
    sharpness: float
    brightness: float
    contrast: float
    motion_score: float = 0.0
    has_faces: bool = False
    face_count: int = 0


@dataclass
class VideoAnalysisResult:
    """Complete analysis results for a video."""
    video_path: str
    duration: float
    fps: float
    width: int
    height: int
    frame_count: int
    frames_analyzed: int
    quality_scores: List[float]
    motion_scores: List[float]
    timestamps: List[float]
    face_detections: List[bool]
    overall_quality: float
    best_segments: List[Tuple[float, float, float]]  # (start, end, score)
    analysis_time: float


class StreamingVideoAnalyzer:
    """Memory-efficient video analyzer using streaming frame processing."""
    
    def __init__(self, 
                 hardware_accelerator: Optional[M2HardwareAccelerator] = None,
                 memory_monitor: Optional[MemoryMonitor] = None,
                 analysis_fps: int = 2,  # Analyze every 2 seconds
                 max_analysis_resolution: Tuple[int, int] = (640, 360)):
        """
        Initialize streaming video analyzer.
        
        Args:
            hardware_accelerator: Hardware acceleration manager
            memory_monitor: Memory monitoring system
            analysis_fps: Frames per second to analyze (lower = less memory)
            max_analysis_resolution: Max resolution for analysis frames
        """
        self.hardware_accelerator = hardware_accelerator or M2HardwareAccelerator()
        self.memory_monitor = memory_monitor or MemoryMonitor()
        self.analysis_fps = analysis_fps
        self.max_analysis_resolution = max_analysis_resolution
        
        # Face detection setup
        self.face_cascade = self._load_face_cascade()
        
        # Motion detection state
        self.prev_frame: Optional[np.ndarray] = None
        
        # Performance tracking
        self.frames_processed = 0
        self.analysis_start_time = 0.0
        
        logger.info(f"Streaming Video Analyzer initialized:")
        logger.info(f"  Analysis FPS: {analysis_fps}")
        logger.info(f"  Max analysis resolution: {max_analysis_resolution}")
        logger.info(f"  DeFFcode available: {DEFFCODE_AVAILABLE}")
        logger.info(f"  PyAV available: {PYAV_AVAILABLE}")
        logger.info(f"  Hardware acceleration: {self.hardware_accelerator.videotoolbox_available}")
    
    def _load_face_cascade(self) -> Optional[cv2.CascadeClassifier]:
        """Load OpenCV face detection cascade."""
        try:
            cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            if os.path.exists(cascade_path):
                return cv2.CascadeClassifier(cascade_path)
            else:
                logger.warning("Face detection cascade not found - face detection disabled")
                return None
        except Exception as e:
            logger.warning(f"Could not load face detection: {e}")
            return None
    
    def analyze_video_streaming(self, video_path: str) -> VideoAnalysisResult:
        """
        Analyze video using streaming approach for minimal memory usage.
        
        Args:
            video_path: Path to video file
            
        Returns:
            VideoAnalysisResult with analysis data
        """
        logger.info(f"Starting streaming analysis of: {Path(video_path).name}")
        self.analysis_start_time = time.time()
        self.frames_processed = 0
        self.prev_frame = None
        
        # Get video metadata first
        metadata = self._get_video_metadata(video_path)
        if not metadata:
            raise ValueError(f"Could not read video metadata: {video_path}")
        
        # Calculate analysis parameters
        frame_interval = max(1, int(metadata['fps'] / self.analysis_fps))
        
        logger.info(f"Video metadata: {metadata['width']}x{metadata['height']}, "
                   f"{metadata['duration']:.1f}s, {metadata['fps']:.1f}fps")
        logger.info(f"Analysis parameters: analyze every {frame_interval} frames")
        
        # Stream and analyze frames
        frame_analyses = []
        
        try:
            for frame_analysis in self._stream_and_analyze_frames(video_path, metadata, frame_interval):
                frame_analyses.append(frame_analysis)
                
                # Memory pressure check
                if self.memory_monitor.should_pause_processing():
                    logger.warning("Pausing analysis due to memory pressure")
                    self.memory_monitor.wait_for_memory_relief(timeout=10.0)
                
                # Progress logging
                if len(frame_analyses) % 50 == 0:
                    logger.debug(f"Analyzed {len(frame_analyses)} frames...")
        
        except Exception as e:
            logger.error(f"Streaming analysis failed: {e}")
            raise
        
        # Compile results
        result = self._compile_analysis_results(video_path, metadata, frame_analyses)
        
        analysis_time = time.time() - self.analysis_start_time
        result.analysis_time = analysis_time
        
        logger.info(f"Streaming analysis complete: {len(frame_analyses)} frames in {analysis_time:.1f}s")
        logger.info(f"Overall quality: {result.overall_quality:.1f}, "
                   f"Best segments: {len(result.best_segments)}")
        
        return result
    
    def _get_video_metadata(self, video_path: str) -> Optional[Dict[str, Any]]:
        """Get video metadata without loading frames."""
        if DEFFCODE_AVAILABLE:
            try:
                # Use DeFFcode for metadata
                decoder = FFdecoder(video_path, verbose=False)
                metadata = decoder.metadata
                decoder.terminate()
                
                return {
                    'duration': metadata.get('duration', 0.0),
                    'fps': metadata.get('fps', 30.0),
                    'width': metadata.get('frame_size', [1920, 1080])[0],
                    'height': metadata.get('frame_size', [1920, 1080])[1],
                    'frame_count': metadata.get('approx_video_nframes', 0),
                }
            except Exception as e:
                logger.warning(f"DeFFcode metadata failed: {e}")
        
        if PYAV_AVAILABLE:
            try:
                # Fallback to PyAV
                container = av.open(video_path)
                stream = container.streams.video[0]
                
                metadata = {
                    'duration': float(stream.duration * stream.time_base) if stream.duration else 0.0,
                    'fps': float(stream.average_rate),
                    'width': stream.width,
                    'height': stream.height,
                    'frame_count': stream.frames,
                }
                
                container.close()
                return metadata
                
            except Exception as e:
                logger.warning(f"PyAV metadata failed: {e}")
        
        # Fallback to OpenCV (less efficient but widely compatible)
        try:
            cap = cv2.VideoCapture(video_path)
            
            metadata = {
                'duration': cap.get(cv2.CAP_PROP_FRAME_COUNT) / cap.get(cv2.CAP_PROP_FPS),
                'fps': cap.get(cv2.CAP_PROP_FPS),
                'width': int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                'height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                'frame_count': int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
            }
            
            cap.release()
            return metadata
            
        except Exception as e:
            logger.error(f"OpenCV metadata failed: {e}")
            return None
    
    def _stream_and_analyze_frames(self, video_path: str, metadata: Dict[str, Any], 
                                 frame_interval: int) -> Iterator[FrameAnalysis]:
        """Stream frames and yield analysis results."""
        
        if DEFFCODE_AVAILABLE and self.hardware_accelerator.videotoolbox_available:
            # Use DeFFcode with hardware acceleration
            yield from self._stream_with_deffcode(video_path, metadata, frame_interval)
        elif PYAV_AVAILABLE:
            # Use PyAV for efficient streaming
            yield from self._stream_with_pyav(video_path, metadata, frame_interval)
        else:
            # Fallback to OpenCV
            yield from self._stream_with_opencv(video_path, metadata, frame_interval)
    
    def _stream_with_deffcode(self, video_path: str, metadata: Dict[str, Any], 
                            frame_interval: int) -> Iterator[FrameAnalysis]:
        """Stream frames using DeFFcode with hardware acceleration."""
        
        # Configure hardware-accelerated parameters
        ffparams = self.hardware_accelerator.get_decode_params()
        ffparams.update({
            '-threads': 'auto',
            '-skip_frame': 'nokey',  # Skip non-keyframes for efficiency
        })
        
        decoder = FFdecoder(video_path, frame_format="bgr24", verbose=False, **ffparams)
        decoder.formulate()
        
        frame_index = 0
        analyzed_frames = 0
        
        try:
            for frame in decoder.generateFrame():
                if frame is None:
                    break
                
                # Analyze every nth frame
                if frame_index % frame_interval == 0:
                    timestamp = frame_index / metadata['fps']
                    
                    # Resize frame for analysis efficiency
                    analysis_frame = self._resize_for_analysis(frame)
                    
                    # Perform analysis
                    frame_analysis = self._analyze_frame(analysis_frame, frame_index, timestamp)
                    yield frame_analysis
                    
                    analyzed_frames += 1
                
                frame_index += 1
                self.frames_processed += 1
                
                # Memory pressure monitoring
                if analyzed_frames % 20 == 0:
                    if self.memory_monitor.should_pause_processing():
                        break
        
        finally:
            decoder.terminate()
    
    def _stream_with_pyav(self, video_path: str, metadata: Dict[str, Any], 
                        frame_interval: int) -> Iterator[FrameAnalysis]:
        """Stream frames using PyAV."""
        
        container = av.open(video_path)
        stream = container.streams.video[0]
        
        # Configure threading for performance
        stream.codec_context.thread_type = av.codec.context.ThreadType.FRAME
        
        frame_index = 0
        analyzed_frames = 0
        
        try:
            for frame in container.decode(stream):
                if frame_index % frame_interval == 0:
                    timestamp = frame_index / metadata['fps']
                    
                    # Convert to numpy array
                    numpy_frame = frame.to_ndarray(format='bgr24')
                    
                    # Resize for analysis
                    analysis_frame = self._resize_for_analysis(numpy_frame)
                    
                    # Perform analysis
                    frame_analysis = self._analyze_frame(analysis_frame, frame_index, timestamp)
                    yield frame_analysis
                    
                    analyzed_frames += 1
                
                frame_index += 1
                self.frames_processed += 1
                
                # Memory pressure monitoring
                if analyzed_frames % 20 == 0:
                    if self.memory_monitor.should_pause_processing():
                        break
        
        finally:
            container.close()
    
    def _stream_with_opencv(self, video_path: str, metadata: Dict[str, Any], 
                          frame_interval: int) -> Iterator[FrameAnalysis]:
        """Stream frames using OpenCV (fallback)."""
        
        cap = cv2.VideoCapture(video_path)
        
        frame_index = 0
        analyzed_frames = 0
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                if frame_index % frame_interval == 0:
                    timestamp = frame_index / metadata['fps']
                    
                    # Resize for analysis
                    analysis_frame = self._resize_for_analysis(frame)
                    
                    # Perform analysis
                    frame_analysis = self._analyze_frame(analysis_frame, frame_index, timestamp)
                    yield frame_analysis
                    
                    analyzed_frames += 1
                
                frame_index += 1
                self.frames_processed += 1
                
                # Memory pressure monitoring
                if analyzed_frames % 20 == 0:
                    if self.memory_monitor.should_pause_processing():
                        break
        
        finally:
            cap.release()
    
    def _resize_for_analysis(self, frame: np.ndarray) -> np.ndarray:
        """Resize frame to analysis resolution for memory efficiency."""
        h, w = frame.shape[:2]
        max_w, max_h = self.max_analysis_resolution
        
        # Calculate scaling to fit within max resolution
        scale_w = max_w / w
        scale_h = max_h / h
        scale = min(scale_w, scale_h, 1.0)  # Don't upscale
        
        if scale < 1.0:
            new_w = int(w * scale)
            new_h = int(h * scale)
            frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
        
        return frame
    
    def _analyze_frame(self, frame: np.ndarray, frame_index: int, timestamp: float) -> FrameAnalysis:
        """Analyze a single frame for quality and content."""
        
        # Convert to grayscale for analysis
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Quality analysis
        sharpness = self._calculate_sharpness(gray)
        brightness = self._calculate_brightness(gray)
        contrast = self._calculate_contrast(gray)
        quality_score = self._calculate_quality_score(sharpness, brightness, contrast)
        
        # Motion analysis
        motion_score = self._calculate_motion(gray)
        
        # Face detection
        has_faces, face_count = self._detect_faces(gray)
        
        return FrameAnalysis(
            frame_index=frame_index,
            timestamp=timestamp,
            quality_score=quality_score,
            sharpness=sharpness,
            brightness=brightness,
            contrast=contrast,
            motion_score=motion_score,
            has_faces=has_faces,
            face_count=face_count
        )
    
    def _calculate_sharpness(self, gray: np.ndarray) -> float:
        """Calculate frame sharpness using Laplacian variance."""
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        return float(laplacian.var())
    
    def _calculate_brightness(self, gray: np.ndarray) -> float:
        """Calculate frame brightness."""
        return float(gray.mean())
    
    def _calculate_contrast(self, gray: np.ndarray) -> float:
        """Calculate frame contrast using standard deviation."""
        return float(gray.std())
    
    def _calculate_quality_score(self, sharpness: float, brightness: float, contrast: float) -> float:
        """Calculate overall quality score from individual metrics."""
        
        # Normalize sharpness (typical range: 0-1000+)
        norm_sharpness = min(sharpness / 100.0, 1.0)
        
        # Normalize brightness (0-255 range, optimal around 128)
        brightness_score = 1.0 - abs(brightness - 128) / 128.0
        
        # Normalize contrast (typical range: 0-100+)
        norm_contrast = min(contrast / 50.0, 1.0)
        
        # Weighted combination
        quality = (0.4 * norm_sharpness + 0.3 * brightness_score + 0.3 * norm_contrast) * 100.0
        
        return max(0.0, min(100.0, quality))
    
    def _calculate_motion(self, gray: np.ndarray) -> float:
        """Calculate motion score compared to previous frame."""
        if self.prev_frame is None:
            self.prev_frame = gray.copy()
            return 0.0
        
        # Calculate optical flow magnitude
        try:
            flow = cv2.calcOpticalFlowPyrLK(self.prev_frame, gray, 
                                          np.array([]), np.array([]))
            if flow is not None and len(flow) > 0:
                motion_magnitude = np.mean(np.sqrt(flow[0]**2 + flow[1]**2))
            else:
                motion_magnitude = 0.0
        except:
            # Fallback to frame difference
            diff = cv2.absdiff(self.prev_frame, gray)
            motion_magnitude = float(diff.mean())
        
        self.prev_frame = gray.copy()
        return motion_magnitude
    
    def _detect_faces(self, gray: np.ndarray) -> Tuple[bool, int]:
        """Detect faces in the frame."""
        if self.face_cascade is None:
            return False, 0
        
        try:
            faces = self.face_cascade.detectMultiScale(
                gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
            )
            face_count = len(faces)
            return face_count > 0, face_count
        except Exception as e:
            logger.debug(f"Face detection failed: {e}")
            return False, 0
    
    def _compile_analysis_results(self, video_path: str, metadata: Dict[str, Any], 
                                frame_analyses: List[FrameAnalysis]) -> VideoAnalysisResult:
        """Compile frame analyses into final result."""
        
        if not frame_analyses:
            raise ValueError("No frames were analyzed")
        
        # Extract data arrays
        quality_scores = [fa.quality_score for fa in frame_analyses]
        motion_scores = [fa.motion_score for fa in frame_analyses]
        timestamps = [fa.timestamp for fa in frame_analyses]
        face_detections = [fa.has_faces for fa in frame_analyses]
        
        # Calculate overall metrics
        overall_quality = sum(quality_scores) / len(quality_scores)
        
        # Find best segments (high quality + good motion)
        best_segments = self._find_best_segments(frame_analyses)
        
        return VideoAnalysisResult(
            video_path=video_path,
            duration=metadata['duration'],
            fps=metadata['fps'],
            width=metadata['width'],
            height=metadata['height'],
            frame_count=metadata['frame_count'],
            frames_analyzed=len(frame_analyses),
            quality_scores=quality_scores,
            motion_scores=motion_scores,
            timestamps=timestamps,
            face_detections=face_detections,
            overall_quality=overall_quality,
            best_segments=best_segments,
            analysis_time=0.0  # Set by caller
        )
    
    def _find_best_segments(self, frame_analyses: List[FrameAnalysis], 
                          min_segment_duration: float = 2.0) -> List[Tuple[float, float, float]]:
        """Find best video segments based on quality and motion."""
        
        if len(frame_analyses) < 2:
            return []
        
        # Calculate combined scores
        combined_scores = []
        for fa in frame_analyses:
            # Combine quality and motion (motion adds dynamism)
            combined_score = fa.quality_score + (fa.motion_score * 0.1)
            # Bonus for faces in family content
            if fa.has_faces:
                combined_score *= 1.2
            combined_scores.append(combined_score)
        
        # Find segments above average quality
        avg_score = sum(combined_scores) / len(combined_scores)
        threshold = avg_score * 1.1  # 10% above average
        
        # Group consecutive good frames into segments
        segments = []
        current_segment_start = None
        current_segment_scores = []
        
        for i, (fa, score) in enumerate(zip(frame_analyses, combined_scores)):
            if score >= threshold:
                if current_segment_start is None:
                    current_segment_start = fa.timestamp
                current_segment_scores.append(score)
            else:
                if current_segment_start is not None:
                    # End current segment
                    segment_end = frame_analyses[i-1].timestamp
                    segment_duration = segment_end - current_segment_start
                    
                    if segment_duration >= min_segment_duration:
                        avg_segment_score = sum(current_segment_scores) / len(current_segment_scores)
                        segments.append((current_segment_start, segment_end, avg_segment_score))
                    
                    current_segment_start = None
                    current_segment_scores = []
        
        # Handle final segment
        if current_segment_start is not None:
            segment_end = frame_analyses[-1].timestamp
            segment_duration = segment_end - current_segment_start
            
            if segment_duration >= min_segment_duration:
                avg_segment_score = sum(current_segment_scores) / len(current_segment_scores)
                segments.append((current_segment_start, segment_end, avg_segment_score))
        
        # Sort by score and return top segments
        segments.sort(key=lambda x: x[2], reverse=True)
        return segments[:10]  # Return top 10 segments