"""Video encoding system with hardware acceleration and compatibility layers."""

import os
import time
import threading
import signal
import psutil
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


class EncodingProgressMonitor:
    """Monitor encoding progress to detect death spirals and performance issues."""
    
    def __init__(self, total_duration: float):
        self.total_duration = total_duration
        self.start_time = time.time()
        self.last_progress_time = time.time()
        self.last_progress_value = 0.0
        self.current_fps = 0.0
        self.frame_count = 0
        self.stall_count = 0
        
    def update_progress(self, progress_data):
        """Update progress from MoviePy callback."""
        current_time = time.time()
        
        try:
            # Extract progress information from MoviePy callback
            # MoviePy progress can be a string or dict
            if isinstance(progress_data, str):
                # Try to extract frame information from string
                if "frame_index" in progress_data or "chunk" in progress_data:
                    # Simple frame counting
                    self.frame_count += 1
                    elapsed = current_time - self.last_progress_time
                    if elapsed > 0:
                        self.current_fps = 1.0 / elapsed
            elif isinstance(progress_data, dict):
                # Handle dict-based progress data
                if "fps" in progress_data:
                    self.current_fps = progress_data["fps"]
                elif "frame" in progress_data:
                    self.frame_count = progress_data["frame"]
                    elapsed = current_time - self.start_time
                    if elapsed > 0:
                        self.current_fps = self.frame_count / elapsed
            
            # Detect stalls
            if current_time - self.last_progress_time > 30.0:  # No progress for 30 seconds
                self.stall_count += 1
                logger.warning(f"Encoding stall detected (#{self.stall_count}): no progress for {current_time - self.last_progress_time:.1f}s")
            
            self.last_progress_time = current_time
            
        except Exception as e:
            logger.warning(f"Failed to parse encoding progress: {e}")
    
    def is_death_spiral(self) -> bool:
        """Check if encoding is in a death spiral (< 0.1 fps)."""
        elapsed = time.time() - self.start_time
        
        # Only check after reasonable startup time
        if elapsed < 60.0:
            return False
        
        # Check for extremely low frame rate
        if self.current_fps > 0 and self.current_fps < 0.1:
            logger.warning(f"Death spiral detected: {self.current_fps:.4f} fps after {elapsed:.0f}s")
            return True
        
        # Check for encoding stalls
        if self.stall_count >= 3:
            logger.warning(f"Multiple stalls detected: {self.stall_count} stalls")
            return True
        
        return False

class VideoEncoder:
    """Handles video encoding with optimal codec settings and compatibility.
    
    Extracted from clip_assembler.py as part of Phase 3 refactoring.
    Provides clean interface for video encoding with hardware acceleration.
    """
    
    def __init__(self):
        self.compatibility_info = None
        
    def prepare_encoding_parameters(self, target_format: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare encoding parameters optimized for target format with stability enhancements.
        
        Args:
            target_format: Target format specification from compositor
            
        Returns:
            Dictionary of encoding parameters
        """
        # Get optimal codec settings
        moviepy_params, ffmpeg_params = self._detect_optimal_codec_settings()
        
        # PHASE 6C: Check memory pressure for quality adjustment
        try:
            import psutil
            memory_percent = psutil.virtual_memory().percent
            
            # Adjust quality based on memory pressure
            if memory_percent > 75.0:
                logger.warning(f"High memory ({memory_percent:.1f}%) - reducing encoding quality for stability")
                
                # Lower quality settings for memory safety
                if "bitrate" in moviepy_params:
                    # Reduce bitrate by up to 40%
                    current_bitrate = moviepy_params["bitrate"]
                    if current_bitrate.endswith("k"):
                        bitrate_val = int(current_bitrate[:-1])
                        reduced_bitrate = max(2000, int(bitrate_val * 0.6))  # Minimum 2Mbps
                        moviepy_params["bitrate"] = f"{reduced_bitrate}k"
                        logger.info(f"Reduced bitrate: {current_bitrate} → {moviepy_params['bitrate']}")
                
                # Use faster, lower quality preset
                for i, param in enumerate(ffmpeg_params):
                    if param == "-preset" and i + 1 < len(ffmpeg_params):
                        if ffmpeg_params[i + 1] in ["slow", "slower", "veryslow"]:
                            ffmpeg_params[i + 1] = "fast"
                            logger.info("Changed encoding preset to 'fast' for memory safety")
                        elif ffmpeg_params[i + 1] in ["medium"]:
                            ffmpeg_params[i + 1] = "faster"
                            logger.info("Changed encoding preset to 'faster' for memory safety")
                
                # Increase CRF (lower quality) if memory is very high
                if memory_percent > 80.0:
                    for i, param in enumerate(ffmpeg_params):
                        if param == "-crf" and i + 1 < len(ffmpeg_params):
                            current_crf = int(ffmpeg_params[i + 1])
                            emergency_crf = min(28, current_crf + 3)  # Increase CRF for lower quality
                            ffmpeg_params[i + 1] = str(emergency_crf)
                            logger.warning(f"Emergency quality reduction: CRF {current_crf} → {emergency_crf}")
                
        except Exception as e:
            logger.warning(f"Failed to check memory for quality adjustment: {e}")
        
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
        
        # PHASE 6C: Add stability parameters
        stability_params = [
            "-threads", "2",           # Limit threads to reduce memory pressure
            "-max_muxing_queue_size", "1024",  # Prevent buffer overflow
            "-fflags", "+genpts",      # Generate missing timestamps
        ]
        
        # Combine all FFmpeg parameters
        enhanced_ffmpeg_params = ffmpeg_params + format_consistency_params + stability_params
        logger.info(f"Enhanced FFmpeg params: {enhanced_ffmpeg_params}")
        
        # Prepare comprehensive parameters
        write_params = {
            **moviepy_params,
            "ffmpeg_params": enhanced_ffmpeg_params,
            "temp_audiofile": "temp-audio.m4a",
            "remove_temp": True,
            "fps": target_format["target_fps"],
            "logger": None,  # Will be replaced with progress callback in encode_video
            # Audio-specific parameters for better sync
            "audio_fps": 44100,      # Standard audio sample rate
            "audio_codec": "aac",     # Compatible audio codec
            "audio_bitrate": "128k",  # Good quality audio bitrate
            # PHASE 6C: Stability parameters
            "write_logfile": False,   # Disable log file to save I/O
            "temp_audiofile_fps": 44100,  # Ensure consistent audio fps
        }
        
        logger.info(f"Phase 6C encoding parameters prepared with stability enhancements")
        return write_params
        
    def encode_video(
        self, 
        final_video, 
        output_path: str, 
        encoding_params: Dict[str, Any]
    ) -> str:
        """Encode final video with compatibility layer and emergency crash prevention.
        
        Args:
            final_video: Composed video ready for encoding
            output_path: Path for output video file
            encoding_params: Encoding parameters from prepare_encoding_parameters
            
        Returns:
            Path to encoded video file
            
        Raises:
            VideoProcessingError: If encoding fails
        """
        import threading
        import time
        import signal
        import os
        import psutil
        
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
            
            # PHASE 6B: Emergency crash prevention during encoding
            encoding_start_time = time.time()
            encoding_result = {"success": False, "error": None}
            encoding_thread = None
            
            # Monitor encoding progress
            progress_monitor = EncodingProgressMonitor(final_video.duration)
            
            def monitored_encoding():
                """Wrapper for encoding with progress monitoring."""
                try:
                    # Lower process priority to prevent system lockup
                    current_process = psutil.Process()
                    current_process.nice(10)  # Lower priority (higher nice value)
                    logger.debug("Lowered encoding process priority to prevent system lockup")
                    
                    # Add progress callback to detect encoding death spiral
                    def progress_callback(progress_data):
                        progress_monitor.update_progress(progress_data)
                        
                        # Check for encoding death spiral (< 0.1 fps)
                        if progress_monitor.is_death_spiral():
                            logger.error(f"💀 Encoding death spiral detected! Frame rate: {progress_monitor.current_fps:.3f} fps")
                            raise VideoProcessingError("Encoding death spiral detected - aborting to prevent system crash")
                        
                        # Check memory pressure during encoding
                        memory_percent = psutil.virtual_memory().percent
                        if memory_percent > 88.0:
                            logger.error(f"🚨 Critical memory during encoding: {memory_percent:.1f}%")
                            raise VideoProcessingError(f"Critical memory pressure during encoding: {memory_percent:.1f}%")
                    
                    # Modify encoding params to include progress callback
                    safe_encoding_params = encoding_params.copy()
                    safe_encoding_params['verbose'] = True
                    safe_encoding_params['logger'] = progress_callback
                    
                    # Render with version-compatible parameter checking and monitoring
                    write_videofile_safely(
                        final_video, 
                        output_path, 
                        self.compatibility_info, 
                        **safe_encoding_params
                    )
                    
                    encoding_result["success"] = True
                    
                except Exception as e:
                    encoding_result["error"] = e
            
            # Start encoding in separate thread
            encoding_thread = threading.Thread(target=monitored_encoding, daemon=True)
            encoding_thread.start()
            
            # Monitor encoding with timeout and system protection
            MAX_ENCODING_TIME = 1800  # 30 minutes maximum
            CHECK_INTERVAL = 5.0  # Check every 5 seconds
            
            while encoding_thread.is_alive():
                elapsed = time.time() - encoding_start_time
                
                # Timeout protection
                if elapsed > MAX_ENCODING_TIME:
                    logger.error(f"🕐 Encoding timeout after {elapsed:.0f}s - aborting to prevent system crash")
                    # Terminate the encoding process
                    try:
                        current_process = psutil.Process()
                        current_process.terminate()
                    except:
                        pass
                    raise VideoProcessingError(f"Encoding timeout after {elapsed:.0f} seconds")
                
                # System protection checks
                memory_percent = psutil.virtual_memory().percent
                if memory_percent > 90.0:
                    logger.error(f"🚨 Emergency memory abort: {memory_percent:.1f}%")
                    try:
                        current_process = psutil.Process()
                        current_process.terminate()
                    except:
                        pass
                    raise VideoProcessingError(f"Emergency memory abort: {memory_percent:.1f}%")
                
                # CPU temperature protection (if available)
                try:
                    temps = psutil.sensors_temperatures()
                    if temps:
                        for name, entries in temps.items():
                            for entry in entries:
                                if entry.current and entry.current > 80.0:
                                    logger.error(f"🌡️ Emergency thermal abort: {entry.current:.1f}°C")
                                    try:
                                        current_process = psutil.Process()
                                        current_process.terminate()
                                    except:
                                        pass
                                    raise VideoProcessingError(f"Emergency thermal abort: {entry.current:.1f}°C")
                except:
                    pass  # Temperature monitoring not available on all systems
                
                # Wait before next check
                time.sleep(CHECK_INTERVAL)
            
            # Check encoding result
            if not encoding_result["success"]:
                if encoding_result["error"]:
                    raise encoding_result["error"]
                else:
                    raise VideoProcessingError("Encoding failed for unknown reason")
            
            # Validate output
            if not os.path.exists(output_path):
                raise VideoProcessingError(f"Output file was not created: {output_path}")
            
            encoding_time = time.time() - encoding_start_time
            logger.info(f"✅ Video encoding completed successfully in {encoding_time:.1f}s")
            return output_path
            
        except Exception as e:
            # Emergency cleanup
            if encoding_thread and encoding_thread.is_alive():
                logger.warning("Performing emergency cleanup of encoding thread")
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