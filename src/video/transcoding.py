"""
Video transcoding service for AutoCut V2.

This module provides comprehensive H.265 to H.264 transcoding functionality
with hardware acceleration, caching, and iPhone compatibility validation.
Extracted from the monolithic utils.py for better organization and testability.
"""

import os
import json
import hashlib
import subprocess
import time
import signal
from contextlib import contextmanager
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Callable

# Import our custom exceptions and codec detection
from ..core.exceptions import TranscodingError, VideoProcessingError, raise_transcoding_error
from .codec_detection import CodecDetector


class TranscodingService:
    """
    Comprehensive video transcoding service with hardware acceleration and caching.
    
    Provides H.265 to H.264 transcoding with:
    - Hardware acceleration (NVIDIA NVENC, Intel QSV) with CPU fallback
    - Smart transcoding cache to avoid duplicate work
    - iPhone compatibility validation and optimization
    - Comprehensive error handling and retry mechanisms
    - Progress tracking and detailed logging
    """
    
    def __init__(self, cache_timeout: float = 3600.0):
        """
        Initialize transcoding service.
        
        Args:
            cache_timeout: How long to cache transcoded files (seconds, default: 1 hour)
        """
        self.cache_timeout = cache_timeout
        self._transcoding_cache: Dict[str, Dict[str, Any]] = {}
        self.codec_detector = CodecDetector()
    
    def transcode_hevc_to_h264(self, 
                             input_path: str, 
                             output_path: Optional[str] = None,
                             progress_callback: Optional[Callable[[str, float], None]] = None,
                             max_retries: int = 2,
                             force_cpu: bool = False) -> str:
        """
        Enhanced H.265 to H.264 transcoding with hardware acceleration and iPhone compatibility.
        
        Replaces: transcode_hevc_to_h264_enhanced() from utils.py
        
        Args:
            input_path: Path to input H.265 video file
            output_path: Output path (auto-generated if None)
            progress_callback: Optional callback for progress updates (message, progress_0_to_1)
            max_retries: Maximum retry attempts if transcoding fails
            force_cpu: Force CPU encoding instead of hardware acceleration
            
        Returns:
            Path to transcoded H.264 video file (guaranteed iPhone compatible)
            
        Raises:
            TranscodingError: If all transcoding attempts fail
            FileNotFoundError: If input file doesn't exist
        """
        if not os.path.exists(input_path):
            raise FileNotFoundError(f"Input video file not found: {input_path}")
        
        # Generate output path if not provided
        if output_path is None:
            input_stem = Path(input_path).stem
            output_path = f"{input_stem}_h264.mp4"
        
        # Ensure output directory exists
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        
        start_time = time.time()
        attempt_log = []
        
        # Import hardware detection after ensuring it exists
        try:
            from ..hardware.detection import HardwareDetector
            hw_detector = HardwareDetector()
            hw_settings = hw_detector.detect_optimal_settings() if not force_cpu else None
        except ImportError:
            # Fallback if hardware detection not available yet
            hw_settings = None
        
        encoder_type = "CPU" if force_cpu or not hw_settings else hw_settings.get('encoder_type', 'CPU')
        
        print(f"🚀 Enhanced H.265→H.264 transcoding:")
        print(f"   🔧 Encoder: {encoder_type}")
        print(f"   📱 iPhone Compatible: ✅")
        print(f"   🎯 Max Retries: {max_retries}")
        
        # Attempt transcoding with fallbacks
        for attempt in range(max_retries + 1):
            try:
                # Select encoding strategy based on attempt number
                if attempt == 0 and hw_settings:
                    # First attempt: Use hardware acceleration if available
                    cmd, description = self._build_transcoding_command(
                        input_path, output_path, hw_settings, encoder_type
                    )
                elif attempt == 1 and encoder_type != 'CPU':
                    # Second attempt: Fallback to CPU if hardware was used first
                    print(f"   🔄 Retry {attempt}: Falling back to CPU encoding")
                    cmd, description = self._build_transcoding_command(
                        input_path, output_path, self._get_cpu_settings(), 'CPU'
                    )
                else:
                    # Final attempt: Conservative CPU settings
                    print(f"   🔄 Retry {attempt}: Conservative CPU encoding")
                    cmd, description = self._build_transcoding_command(
                        input_path, output_path, self._get_cpu_settings(conservative=True), 'CPU_CONSERVATIVE'
                    )
                
                attempt_start = time.time()
                if progress_callback:
                    progress_callback(f"Attempt {attempt + 1}: {description}", 0.0)
                
                print(f"   ⚡ Command: {' '.join(cmd[:10])}... ({description})")
                
                # Run FFmpeg with monitoring
                success = self._run_ffmpeg_with_monitoring(
                    cmd, attempt_start, progress_callback
                )
                
                attempt_time = time.time() - attempt_start
                
                if success:
                    # Transcoding succeeded - validate output
                    print(f"   ✅ Transcoding completed in {attempt_time:.1f}s")
                    
                    # Basic file integrity check
                    if not os.path.exists(output_path) or os.path.getsize(output_path) == 0:
                        error_msg = f"Attempt {attempt + 1}: Transcoded file missing or empty"
                        attempt_log.append(error_msg)
                        print(f"   ❌ {error_msg}")
                        if attempt < max_retries:
                            continue
                        else:
                            raise_transcoding_error(
                                f"All attempts failed file integrity: {'; '.join(attempt_log)}",
                                input_path
                            )
                    
                    # Validate iPhone compatibility
                    if self._validate_iphone_compatibility(output_path):
                        total_time = time.time() - start_time
                        
                        if progress_callback:
                            progress_callback(f"Complete & Validated ({total_time:.1f}s)", 1.0)
                        
                        self._log_transcoding_success(
                            input_path, output_path, total_time, attempt + 1, encoder_type
                        )
                        return output_path
                    else:
                        error_msg = f"Attempt {attempt + 1}: Output validation failed - not iPhone compatible"
                        attempt_log.append(error_msg)
                        print(f"   ❌ {error_msg}")
                        if attempt < max_retries:
                            continue
                        else:
                            raise_transcoding_error(
                                f"All attempts failed validation: {'; '.join(attempt_log)}",
                                input_path
                            )
                else:
                    # FFmpeg failed
                    error_msg = f"Attempt {attempt + 1} failed ({attempt_time:.1f}s): FFmpeg execution failed"
                    attempt_log.append(error_msg)
                    print(f"   ❌ {error_msg}")
                    
                    if attempt >= max_retries:
                        break
                
            except Exception as e:
                attempt_time = time.time() - attempt_start if 'attempt_start' in locals() else 0
                error_msg = f"Attempt {attempt + 1} exception ({attempt_time:.1f}s): {str(e)}"
                attempt_log.append(error_msg)
                print(f"   ❌ {error_msg}")
                
                if attempt >= max_retries:
                    break
        
        # All attempts failed
        total_time = time.time() - start_time
        comprehensive_error = f"Enhanced transcoding failed after {max_retries + 1} attempts ({total_time:.1f}s): {'; '.join(attempt_log)}"
        print(f"❌ {comprehensive_error}")
        raise_transcoding_error(comprehensive_error, input_path)
    
    def preprocess_video_if_needed(self, 
                                 file_path: str, 
                                 temp_dir: str = "temp") -> Dict[str, Any]:
        """
        Enhanced video preprocessing with smart caching and comprehensive error handling.
        
        Replaces: preprocess_video_if_needed_enhanced() from utils.py
        
        Args:
            file_path: Path to input video file
            temp_dir: Directory for temporary transcoded files
            
        Returns:
            Dictionary containing processing results and metadata
        """
        start_time = time.time()
        result = {
            'processed_path': file_path,
            'success': False,
            'transcoded': False, 
            'cached': False,
            'processing_time': 0.0,
            'error_category': None,
            'diagnostic_message': '',
            'original_codec': None
        }
        
        filename = Path(file_path).name
        
        try:
            print(f"🔍 Enhanced preprocessing: {filename}")
            
            # Check transcoding cache first
            cached_path = self._check_transcoding_cache(file_path)
            if cached_path:
                result['processed_path'] = cached_path
                result['success'] = True
                result['transcoded'] = True
                result['cached'] = True
                result['processing_time'] = time.time() - start_time
                result['diagnostic_message'] = f"Used cached transcoded file: {Path(cached_path).name}"
                print(f"   ✅ Cache hit: {Path(cached_path).name} ({result['processing_time']:.3f}s)")
                return result
            
            # Analyze video format
            print(f"   📊 Step 1: Analyzing video format...")
            try:
                codec_info = self.codec_detector.detect_video_codec(file_path)
                result['original_codec'] = {
                    'codec': codec_info['codec'],
                    'profile': codec_info.get('profile', 'unknown'),
                    'pixel_format': codec_info.get('pixel_format', 'unknown'),
                    'resolution': codec_info['resolution'],
                    'container': codec_info['container']
                }
                
                compatibility_score = codec_info.get('compatibility_score', 50)
                warnings = codec_info.get('warnings', [])
                
                print(f"   ✅ Codec analysis: {codec_info['codec']} ({compatibility_score}/100 compatibility)")
                if warnings:
                    for warning in warnings:
                        print(f"      ⚠️  {warning}")
                
            except Exception as e:
                result['error_category'] = 'CODEC_DETECTION_FAILED'
                result['diagnostic_message'] = f"Codec detection failed: {str(e)[:200]}"
                print(f"   ❌ Codec detection failed: {str(e)[:100]}...")
                result['processed_path'] = file_path
                result['processing_time'] = time.time() - start_time
                return result
            
            # Determine processing strategy
            if not codec_info['is_hevc']:
                # Not H.265 - check if additional processing needed
                if compatibility_score >= 80:
                    print(f"   ✅ {codec_info['codec']} format compatible, no processing needed")
                    result['success'] = True
                    result['diagnostic_message'] = f"Native {codec_info['codec']} compatibility (score: {compatibility_score})"
                    result['processing_time'] = time.time() - start_time
                    return result
                else:
                    print(f"   ⚠️  {codec_info['codec']} format has compatibility issues (score: {compatibility_score})")
            
            # H.265 processing
            print(f"   📱 Step 2: Testing H.265/iPhone compatibility...")
            
            # Smart compatibility testing
            if self.test_moviepy_h265_compatibility(file_path):
                print(f"   ✅ H.265 compatible with MoviePy, skipping transcoding")
                result['success'] = True
                result['diagnostic_message'] = "H.265 native MoviePy compatibility confirmed"
                result['processing_time'] = time.time() - start_time
                return result
            
            # Enhanced transcoding required
            print(f"   🔄 Step 3: H.265 transcoding required for MoviePy compatibility")
            
            # Create temp directory
            try:
                os.makedirs(temp_dir, exist_ok=True)
            except OSError as e:
                result['error_category'] = 'TEMP_DIR_CREATION_FAILED'
                result['diagnostic_message'] = f"Cannot create temp directory {temp_dir}: {str(e)}"
                print(f"   ❌ Temp directory creation failed: {str(e)}")
                result['processing_time'] = time.time() - start_time
                return result
            
            # Generate output path
            input_stem = Path(file_path).stem
            cache_key = self._get_file_cache_key(file_path)[:8]
            output_path = os.path.join(temp_dir, f"{input_stem}_h264_iphone_{cache_key}.mp4")
            
            # Enhanced transcoding
            try:
                transcoded_path = self.transcode_hevc_to_h264(
                    file_path, output_path,
                    progress_callback=lambda msg, progress: print(f"      📈 {msg}"),
                    max_retries=2
                )
                
                # Update cache
                self._update_transcoding_cache(file_path, transcoded_path)
                
                result['processed_path'] = transcoded_path
                result['transcoded'] = True
                result['success'] = True
                result['diagnostic_message'] = "Enhanced H.265→H.264 transcoding successful with iPhone compatibility validation"
                
                print(f"   ✅ Enhanced transcoding successful: {Path(transcoded_path).name}")
                
            except Exception as transcoding_error:
                error_str = str(transcoding_error)
                
                # Categorize transcoding errors
                if 'Driver does not support' in error_str or 'nvenc API version' in error_str:
                    result['error_category'] = 'HARDWARE_DRIVER_INCOMPATIBLE'
                elif 'Hardware encoder' in error_str or 'device' in error_str.lower():
                    result['error_category'] = 'HARDWARE_ENCODER_FAILED'
                elif 'All attempts failed' in error_str:
                    result['error_category'] = 'TRANSCODING_ALL_ATTEMPTS_FAILED'
                elif 'validation failed' in error_str.lower():
                    result['error_category'] = 'OUTPUT_VALIDATION_FAILED'
                else:
                    result['error_category'] = 'TRANSCODING_UNKNOWN_ERROR'
                
                result['diagnostic_message'] = f"Transcoding error: {error_str[:200]}"
                print(f"   ❌ Enhanced transcoding failed: {result['error_category']}")
                print(f"      Details: {result['diagnostic_message'][:100]}...")
                
                # Fallback to original file
                result['processed_path'] = file_path
                print(f"   ⚠️  Falling back to original file - may cause downstream processing issues")
        
        except Exception as e:
            result['error_category'] = 'PREPROCESSING_UNEXPECTED_ERROR'
            result['diagnostic_message'] = f"Unexpected preprocessing error: {str(e)[:200]}"
            result['processed_path'] = file_path
            print(f"   ❌ Unexpected preprocessing error: {str(e)[:100]}...")
            print(f"   ⚠️  Using original file as fallback")
        
        finally:
            result['processing_time'] = time.time() - start_time
            performance_indicator = "⚡" if result['cached'] else "⏱️"
            print(f"   {performance_indicator} Preprocessing completed in {result['processing_time']:.1f}s")
        
        return result
    
    def test_moviepy_h265_compatibility(self, 
                                      file_path: str, 
                                      timeout_seconds: float = 10.0) -> bool:
        """
        Test if MoviePy can load H.265 file directly without transcoding.
        
        Replaces: test_moviepy_h265_compatibility() from utils.py
        
        Args:
            file_path: Path to H.265 video file
            timeout_seconds: Maximum time to spend testing
            
        Returns:
            True if MoviePy can load the H.265 file directly, False if transcoding needed
        """
        @contextmanager
        def timeout_handler(seconds):
            def timeout_signal(signum, frame):
                raise TimeoutError("MoviePy compatibility test timed out")
            
            old_handler = signal.signal(signal.SIGALRM, timeout_signal)
            signal.alarm(int(seconds))
            
            try:
                yield
            finally:
                signal.alarm(0)
                signal.signal(signal.SIGALRM, old_handler)
        
        try:
            print(f"   🧪 Testing MoviePy H.265 compatibility (timeout: {timeout_seconds}s)...")
            start_time = time.time()
            
            # Try to import MoviePy
            try:
                from moviepy.editor import VideoFileClip
            except ImportError:
                try:
                    from moviepy import VideoFileClip
                except ImportError:
                    print(f"   ❌ MoviePy not available")
                    return False
            
            # Test H.265 loading with timeout protection
            with timeout_handler(timeout_seconds):
                video_clip = VideoFileClip(file_path)
                
                # Minimal compatibility verification
                duration = video_clip.duration
                width = video_clip.w
                height = video_clip.h
                fps = video_clip.fps
                
                # Try to get a frame to ensure decoding works
                test_frame = video_clip.get_frame(min(1.0, duration * 0.1))
                
                # Clean up
                video_clip.close()
                
                test_time = time.time() - start_time
                print(f"   ✅ Compatibility test passed ({test_time:.2f}s)")
                print(f"      📹 Video: {width}x{height} @ {fps:.1f}fps, {duration:.1f}s")
                print(f"      🎯 Frame decode: successful")
                
                return True
                
        except TimeoutError:
            print(f"   ⏱️  Compatibility test timed out after {timeout_seconds}s")
            print(f"      → H.265 loading too slow, transcoding recommended")
            return False
            
        except Exception as e:
            error_msg = str(e).lower()
            test_time = time.time() - start_time
            
            if any(keyword in error_msg for keyword in ['codec', 'decoder', 'format']):
                print(f"   ❌ Codec compatibility issue ({test_time:.2f}s): {str(e)[:100]}...")
            elif any(keyword in error_msg for keyword in ['memory', 'allocation']):
                print(f"   💾 Memory issue ({test_time:.2f}s): Large H.265 file may need transcoding")
            elif any(keyword in error_msg for keyword in ['permission', 'access']):
                print(f"   🔒 File access issue ({test_time:.2f}s): {str(e)[:100]}...")
            else:
                print(f"   ❌ Compatibility test failed ({test_time:.2f}s): {str(e)[:100]}...")
            
            return False
        
        finally:
            # Cleanup
            try:
                if 'video_clip' in locals():
                    video_clip.close()
            except:
                pass
    
    def _build_transcoding_command(self, 
                                 input_path: str, 
                                 output_path: str, 
                                 settings: Dict[str, Any], 
                                 encoder_type: str) -> Tuple[List[str], str]:
        """
        Build optimized FFmpeg transcoding command with iPhone compatibility.
        
        Args:
            input_path: Input video file path
            output_path: Output video file path
            settings: Hardware or CPU encoder settings
            encoder_type: Type of encoder being used
            
        Returns:
            Tuple of (command_list, description)
        """
        codec = settings.get('codec', 'libx264')
        
        # iPhone compatibility parameters
        iphone_params = [
            '-profile:v', 'main',         # Force Main profile for iPhone compatibility
            '-pix_fmt', 'yuv420p',        # Force 8-bit pixel format for MoviePy compatibility
            '-level', '4.1',              # Ensure broad device compatibility 
            '-movflags', '+faststart'     # Web/mobile optimization
        ]
        
        # Build command based on encoder type
        if codec == 'h264_nvenc':
            # NVIDIA GPU encoding
            cmd = [
                'ffmpeg', '-y',
                '-hwaccel', 'cuda', '-hwaccel_output_format', 'cuda',
                '-c:v', 'hevc_cuvid', '-i', input_path,
                '-c:v', 'h264_nvenc',
                '-preset', 'fast', '-crf', '25'
            ] + iphone_params + [
                '-c:a', 'aac', '-b:a', '128k',
                output_path
            ]
            description = "NVIDIA GPU + iPhone params"
            
        elif codec == 'h264_qsv':
            # Intel QuickSync encoding
            cmd = [
                'ffmpeg', '-y',
                '-hwaccel', 'qsv', '-hwaccel_output_format', 'qsv', 
                '-c:v', 'hevc_qsv', '-i', input_path,
                '-c:v', 'h264_qsv',
                '-preset', 'fast', '-crf', '25'
            ] + iphone_params + [
                '-c:a', 'aac', '-b:a', '128k',
                output_path
            ]
            description = "Intel QSV + iPhone params"
            
        else:
            # CPU encoding
            threads = min(settings.get('threads', os.cpu_count() or 4), 6)
            preset = settings.get('preset', 'ultrafast')
            crf = settings.get('crf', '25')
            
            cmd = [
                'ffmpeg', '-y', '-i', input_path,
                '-c:v', 'libx264',
                '-threads', str(threads),
                '-preset', preset, '-crf', str(crf)
            ] + iphone_params + [
                '-c:a', 'aac', '-b:a', '128k',
                output_path
            ]
            description = f"CPU {preset} ({threads}t) + iPhone params"
        
        return cmd, description
    
    def _run_ffmpeg_with_monitoring(self, 
                                  cmd: List[str], 
                                  start_time: float, 
                                  progress_callback: Optional[Callable] = None) -> bool:
        """Run FFmpeg command with progress monitoring."""
        try:
            process = subprocess.Popen(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
            )
            
            # Progress monitoring
            while process.poll() is None:
                if progress_callback:
                    elapsed = time.time() - start_time
                    progress_callback(f"Transcoding ({elapsed:.1f}s)", 0.5)
                time.sleep(1)
            
            # Get final result
            stdout, stderr = process.communicate()
            return process.returncode == 0
            
        except subprocess.SubprocessError:
            return False
    
    def _validate_iphone_compatibility(self, file_path: str) -> bool:
        """Validate that transcoded output meets iPhone compatibility requirements."""
        try:
            codec_info = self.codec_detector.detect_video_codec(file_path)
            
            # Check essential iPhone requirements
            codec_ok = codec_info.get('codec') == 'h264'
            profile_ok = 'main' in codec_info.get('profile', '').lower() or 'baseline' in codec_info.get('profile', '').lower()
            pixfmt_ok = codec_info.get('pixel_format') == 'yuv420p'
            
            return codec_ok and profile_ok and pixfmt_ok
            
        except Exception:
            return False
    
    def _get_cpu_settings(self, conservative: bool = False) -> Dict[str, Any]:
        """Get CPU encoding settings."""
        if conservative:
            return {
                'codec': 'libx264',
                'preset': 'fast',
                'crf': '23',
                'threads': 2
            }
        else:
            return {
                'codec': 'libx264',
                'preset': 'ultrafast',
                'crf': '25',
                'threads': min(os.cpu_count() or 4, 6)
            }
    
    def _log_transcoding_success(self, 
                               input_path: str, 
                               output_path: str,
                               total_time: float, 
                               attempts: int, 
                               encoder_type: str) -> None:
        """Log successful transcoding with comprehensive information."""
        try:
            input_size = os.path.getsize(input_path) / (1024 * 1024)  # MB
            output_size = os.path.getsize(output_path) / (1024 * 1024)  # MB
            
            print(f"✅ Enhanced iPhone H.265→H.264 transcoding successful:")
            print(f"   📁 Input: {Path(input_path).name}")
            print(f"   📁 Output: {Path(output_path).name}")
            print(f"   ⏱️  Time: {total_time:.1f}s (attempts: {attempts})")
            print(f"   🔧 Method: {encoder_type}")
            print(f"   📊 Size: {input_size:.1f}MB → {output_size:.1f}MB")
            print(f"   📱 iPhone Compatibility: Validated ✅")
        except Exception:
            print(f"✅ Transcoding successful: {Path(output_path).name}")
    
    def _check_transcoding_cache(self, file_path: str) -> Optional[str]:
        """Check if transcoded version exists in cache and is still valid."""
        try:
            cache_key = self._get_file_cache_key(file_path)
            
            if cache_key in self._transcoding_cache:
                cached_info = self._transcoding_cache[cache_key]
                cached_path = cached_info['transcoded_path']
                cache_time = cached_info['timestamp']
                
                # Check cache age
                if time.time() - cache_time > self.cache_timeout:
                    del self._transcoding_cache[cache_key]
                    return None
                
                # Check if cached file still exists
                if os.path.exists(cached_path):
                    print(f"   💾 Cache hit: Using cached transcoded file")
                    return cached_path
                else:
                    # Cached file was deleted
                    del self._transcoding_cache[cache_key]
                    return None
            
            return None
            
        except (OSError, KeyError):
            return None
    
    def _update_transcoding_cache(self, input_path: str, transcoded_path: str) -> None:
        """Update transcoding cache with successful transcoding result."""
        try:
            cache_key = self._get_file_cache_key(input_path)
            self._transcoding_cache[cache_key] = {
                'input_path': input_path,
                'transcoded_path': transcoded_path,
                'timestamp': time.time()
            }
        except Exception:
            # Cache update failure shouldn't break transcoding
            pass
    
    def _get_file_cache_key(self, file_path: str) -> str:
        """Generate cache key based on file path and modification time."""
        try:
            stat = os.stat(file_path)
            cache_string = f"{file_path}_{stat.st_mtime}_{stat.st_size}"
            return hashlib.md5(cache_string.encode()).hexdigest()
        except OSError:
            # Fallback to just file path if stat fails
            return hashlib.md5(file_path.encode()).hexdigest()
    
    def clear_cache(self) -> None:
        """Clear the transcoding cache."""
        self._transcoding_cache.clear()
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get transcoding cache statistics."""
        return {
            'cached_files': len(self._transcoding_cache),
            'cache_timeout': self.cache_timeout,
            'oldest_entry_age': min([
                time.time() - entry['timestamp'] 
                for entry in self._transcoding_cache.values()
            ]) if self._transcoding_cache else 0
        }


# Global transcoding service instance for backwards compatibility
_global_transcoding_service = TranscodingService()


# Backwards compatibility functions
def transcode_hevc_to_h264_enhanced(input_path: str, 
                                   output_path: str = None, 
                                   progress_callback: Optional[Callable] = None,
                                   max_retries: int = 2) -> str:
    """
    Legacy function for backwards compatibility.
    
    Replaces the original transcode_hevc_to_h264_enhanced() from utils.py
    """
    return _global_transcoding_service.transcode_hevc_to_h264(
        input_path, output_path, progress_callback, max_retries
    )


def transcode_hevc_to_h264(input_path: str, 
                          output_path: str = None, 
                          progress_callback: Optional[Callable] = None) -> str:
    """
    Legacy function for backwards compatibility.
    
    Replaces the original transcode_hevc_to_h264() from utils.py
    """
    return _global_transcoding_service.transcode_hevc_to_h264(
        input_path, output_path, progress_callback
    )


def preprocess_video_if_needed_enhanced(file_path: str, temp_dir: str = "temp") -> Dict[str, Any]:
    """
    Legacy function for backwards compatibility.
    
    Replaces the original preprocess_video_if_needed_enhanced() from utils.py
    """
    return _global_transcoding_service.preprocess_video_if_needed(file_path, temp_dir)


def preprocess_video_if_needed(file_path: str, temp_dir: str = "temp") -> str:
    """
    Legacy function for backwards compatibility.
    
    Replaces the original preprocess_video_if_needed() from utils.py
    """
    result = _global_transcoding_service.preprocess_video_if_needed(file_path, temp_dir)
    return result['processed_path']


def test_moviepy_h265_compatibility(file_path: str, timeout_seconds: float = 10.0) -> bool:
    """
    Legacy function for backwards compatibility.
    
    Replaces the original test_moviepy_h265_compatibility() from utils.py
    """
    return _global_transcoding_service.test_moviepy_h265_compatibility(file_path, timeout_seconds)


# Export key classes and functions
__all__ = [
    'TranscodingService',
    # Legacy compatibility functions
    'transcode_hevc_to_h264_enhanced',
    'transcode_hevc_to_h264',
    'preprocess_video_if_needed_enhanced',
    'preprocess_video_if_needed',
    'test_moviepy_h265_compatibility'
]