"""
Apple Silicon (M2/M1) Hardware Acceleration Support for AutoCut V2.

Provides automatic detection and configuration of VideoToolbox hardware
acceleration for optimal performance on Apple Silicon Macs.
"""

import platform
import subprocess
import logging
from typing import Dict, Any, Optional, List
import psutil

logger = logging.getLogger(__name__)


class M2HardwareAccelerator:
    """Hardware acceleration manager for Apple Silicon Macs."""
    
    def __init__(self):
        self.is_apple_silicon = self._detect_apple_silicon()
        self.videotoolbox_available = self._check_videotoolbox_support()
        self.metal_available = self._check_metal_support()
        self.optimal_params = self._determine_optimal_params()
        
        logger.info(f"Hardware Accelerator initialized:")
        logger.info(f"  Apple Silicon: {self.is_apple_silicon}")
        logger.info(f"  VideoToolbox: {self.videotoolbox_available}")
        logger.info(f"  Metal: {self.metal_available}")
    
    def _detect_apple_silicon(self) -> bool:
        """Detect if running on Apple Silicon (M1/M2)."""
        try:
            # Check macOS and ARM architecture
            if platform.system() != "Darwin":
                return False
                
            # Check for ARM64 architecture
            machine = platform.machine().lower()
            is_arm = machine in ['arm64', 'aarch64']
            
            if is_arm:
                # Additional check for Apple Silicon specifically
                try:
                    result = subprocess.run(['sysctl', '-n', 'machdep.cpu.brand_string'], 
                                          capture_output=True, text=True, timeout=5)
                    cpu_brand = result.stdout.strip().lower()
                    is_apple_cpu = 'apple' in cpu_brand
                    logger.debug(f"CPU Brand: {cpu_brand}, Apple CPU: {is_apple_cpu}")
                    return is_apple_cpu
                except Exception as e:
                    logger.warning(f"Could not detect CPU brand: {e}")
                    return is_arm  # Fallback to ARM detection
            
            return False
            
        except Exception as e:
            logger.warning(f"Apple Silicon detection failed: {e}")
            return False
    
    def _check_videotoolbox_support(self) -> bool:
        """Check if VideoToolbox hardware acceleration is available."""
        if not self.is_apple_silicon:
            return False
            
        try:
            # Test if ffmpeg supports VideoToolbox
            result = subprocess.run(['ffmpeg', '-hide_banner', '-hwaccels'], 
                                  capture_output=True, text=True, timeout=10)
            
            if result.returncode == 0:
                hwaccels = result.stdout.lower()
                has_videotoolbox = 'videotoolbox' in hwaccels
                logger.debug(f"FFmpeg hardware accelerators: {hwaccels.strip()}")
                return has_videotoolbox
            else:
                logger.warning("Could not query FFmpeg hardware accelerators")
                return False
                
        except subprocess.TimeoutExpired:
            logger.warning("FFmpeg hardware acceleration check timed out")
            return False
        except FileNotFoundError:
            logger.error("FFmpeg not found - required for hardware acceleration")
            return False
        except Exception as e:
            logger.warning(f"VideoToolbox support check failed: {e}")
            return False
    
    def _check_metal_support(self) -> bool:
        """Check if Metal Performance Shaders are available."""
        if not self.is_apple_silicon:
            return False
            
        try:
            # Check for Metal support via system_profiler
            result = subprocess.run(['system_profiler', 'SPDisplaysDataType'], 
                                  capture_output=True, text=True, timeout=15)
            
            if result.returncode == 0:
                display_info = result.stdout.lower()
                has_metal = 'metal' in display_info
                logger.debug(f"Metal support detected: {has_metal}")
                return has_metal
            else:
                # Fallback: assume Metal is available on Apple Silicon
                logger.debug("Could not check Metal support, assuming available")
                return True
                
        except Exception as e:
            logger.warning(f"Metal support check failed: {e}")
            # Fallback: assume Metal is available on Apple Silicon
            return True
    
    def _determine_optimal_params(self) -> Dict[str, Any]:
        """Determine optimal FFmpeg parameters for this hardware."""
        params = {}
        
        if self.is_apple_silicon and self.videotoolbox_available:
            # Apple Silicon with VideoToolbox
            params.update({
                # Decoding parameters
                'decode_hwaccel': 'videotoolbox',
                'decode_hwaccel_output_format': 'videotoolbox_vld',
                
                # Encoding parameters (HEVC preferred on M2)
                'encode_codec': 'hevc_videotoolbox',  # Better than h264_videotoolbox on M2
                'encode_quality': '65',  # Good balance of quality/speed (1-100 scale)
                'encode_realtime': '1',  # Real-time encoding
                'encode_threads': 'auto',
                
                # Scaling parameters
                'scale_filter': 'scale_vt',  # Hardware-accelerated scaling
                'pixel_format': 'nv12',  # Preferred format for VideoToolbox
                
                # Performance optimizations
                'preset': 'fast',
                'tune': 'fastdecode',
            })
            
            # M2-specific optimizations
            if self._is_m2_or_newer():
                params.update({
                    'encode_max_bitrate': '50M',  # Higher bitrate for M2
                    'encode_bufsize': '100M',     # Larger buffer for M2
                })
            
        else:
            # Software fallback parameters
            params.update({
                'encode_codec': 'libx264',
                'encode_preset': 'medium',
                'encode_crf': '23',  # Constant Rate Factor for quality
                'pixel_format': 'yuv420p',
            })
            
        logger.info(f"Optimal parameters determined: hardware acceleration {self.videotoolbox_available}")
        return params
    
    def _is_m2_or_newer(self) -> bool:
        """Check if running on M2 or newer chip."""
        try:
            result = subprocess.run(['sysctl', '-n', 'machdep.cpu.brand_string'], 
                                  capture_output=True, text=True, timeout=5)
            cpu_brand = result.stdout.strip().lower()
            
            # Check for M2, M3, etc. (assume anything after M1 is newer)
            is_m2_plus = any(chip in cpu_brand for chip in ['m2', 'm3', 'm4'])
            logger.debug(f"M2+ detection: {is_m2_plus} (CPU: {cpu_brand})")
            return is_m2_plus
            
        except Exception as e:
            logger.warning(f"Could not detect M2+ status: {e}")
            return False  # Conservative fallback
    
    def get_decode_params(self) -> Dict[str, str]:
        """Get optimal FFmpeg decoding parameters."""
        if not self.videotoolbox_available:
            return {}
            
        return {
            '-hwaccel': self.optimal_params['decode_hwaccel'],
            '-hwaccel_output_format': self.optimal_params['decode_hwaccel_output_format'],
        }
    
    def get_encode_params(self, target_quality: str = 'high') -> Dict[str, str]:
        """Get optimal FFmpeg encoding parameters."""
        params = {}
        
        if self.videotoolbox_available:
            params.update({
                '-c:v': self.optimal_params['encode_codec'],
                '-q:v': self._get_quality_value(target_quality),
                '-realtime': self.optimal_params['encode_realtime'],
            })
            
            # Add M2-specific parameters if available
            if 'encode_max_bitrate' in self.optimal_params:
                params['-maxrate'] = self.optimal_params['encode_max_bitrate']
                params['-bufsize'] = self.optimal_params['encode_bufsize']
                
        else:
            # Software encoding fallback
            params.update({
                '-c:v': self.optimal_params['encode_codec'],
                '-preset': self.optimal_params['encode_preset'],
                '-crf': self.optimal_params['encode_crf'],
            })
        
        params['-pix_fmt'] = self.optimal_params['pixel_format']
        return params
    
    def get_scale_params(self, width: int, height: int) -> Dict[str, str]:
        """Get optimal scaling parameters."""
        if self.videotoolbox_available and 'scale_filter' in self.optimal_params:
            return {
                '-vf': f"{self.optimal_params['scale_filter']}={width}:{height}"
            }
        else:
            return {
                '-vf': f"scale={width}:{height}"
            }
    
    def _get_quality_value(self, target_quality: str) -> str:
        """Convert quality level to appropriate codec value."""
        if self.videotoolbox_available:
            # VideoToolbox quality scale (1-100, higher = better)
            quality_map = {
                'low': '85',      # Fast encoding, lower quality
                'medium': '65',   # Balanced
                'high': '45',     # Higher quality, slower
                'max': '25',      # Maximum quality
            }
        else:
            # Software CRF scale (0-51, lower = better)
            quality_map = {
                'low': '28',      # Lower quality, faster
                'medium': '23',   # Balanced
                'high': '18',     # Higher quality
                'max': '15',      # Very high quality
            }
        
        return quality_map.get(target_quality, quality_map['medium'])
    
    def estimate_performance_gain(self) -> Dict[str, float]:
        """Estimate performance improvement with hardware acceleration."""
        if not self.videotoolbox_available:
            return {
                'decode_speedup': 1.0,
                'encode_speedup': 1.0,
                'memory_reduction': 1.0,
            }
        
        # Based on benchmarks for Apple Silicon VideoToolbox
        base_gains = {
            'decode_speedup': 2.5,    # 2.5x faster decoding
            'encode_speedup': 3.0,    # 3x faster encoding  
            'memory_reduction': 0.7,  # 30% memory reduction
        }
        
        # M2+ gets additional benefits
        if self._is_m2_or_newer():
            base_gains.update({
                'decode_speedup': 3.0,    # Even better on M2
                'encode_speedup': 3.5,    # M2 encoder improvements
                'memory_reduction': 0.6,  # 40% memory reduction
            })
        
        return base_gains
    
    def get_system_info(self) -> Dict[str, Any]:
        """Get comprehensive system information for optimization."""
        info = {
            'platform': platform.platform(),
            'architecture': platform.machine(),
            'cpu_count': psutil.cpu_count(),
            'memory_total_gb': round(psutil.virtual_memory().total / (1024**3), 1),
            'is_apple_silicon': self.is_apple_silicon,
            'videotoolbox_available': self.videotoolbox_available,
            'metal_available': self.metal_available,
        }
        
        # Add CPU information if available
        try:
            result = subprocess.run(['sysctl', '-n', 'machdep.cpu.brand_string'], 
                                  capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                info['cpu_brand'] = result.stdout.strip()
        except Exception:
            pass
        
        return info
    
    def validate_setup(self) -> List[str]:
        """Validate hardware acceleration setup and return any issues."""
        issues = []
        
        if not self.is_apple_silicon:
            issues.append("Not running on Apple Silicon - hardware acceleration unavailable")
            return issues
        
        if not self.videotoolbox_available:
            issues.append("VideoToolbox not available - check FFmpeg installation")
        
        # Check memory for 4K processing
        memory_gb = psutil.virtual_memory().total / (1024**3)
        if memory_gb < 8:
            issues.append(f"Low memory ({memory_gb:.1f}GB) - may limit 4K processing")
        elif memory_gb < 16:
            issues.append(f"Moderate memory ({memory_gb:.1f}GB) - 16GB+ recommended for 4K")
        
        # Check FFmpeg version
        try:
            result = subprocess.run(['ffmpeg', '-version'], 
                                  capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                version_info = result.stdout.lower()
                if 'videotoolbox' not in version_info:
                    issues.append("FFmpeg lacks VideoToolbox support - reinstall with hardware acceleration")
            else:
                issues.append("Could not verify FFmpeg version")
        except Exception:
            issues.append("FFmpeg not found - required for video processing")
        
        if not issues:
            issues.append("✅ Hardware acceleration setup validated successfully")
        
        return issues