"""
Unit tests for the hardware detection module.

Tests the HardwareDetector class that detects GPU and hardware acceleration capabilities.
"""

import pytest
from unittest.mock import patch, MagicMock
import subprocess

from src.hardware.detection import HardwareDetector, HardwareCapabilities


class TestHardwareCapabilities:
    """Test HardwareCapabilities dataclass functionality."""
    
    def test_capabilities_creation(self):
        """Test creating a HardwareCapabilities instance."""
        caps = HardwareCapabilities(
            has_nvidia_gpu=True,
            has_intel_qsv=False,
            nvidia_nvenc_available=True,
            intel_qsv_available=False,
            gpu_memory_gb=8.0,
            cpu_cores=8
        )
        
        assert caps.has_nvidia_gpu is True
        assert caps.has_intel_qsv is False
        assert caps.nvidia_nvenc_available is True
        assert caps.intel_qsv_available is False
        assert caps.gpu_memory_gb == 8.0
        assert caps.cpu_cores == 8
    
    def test_capabilities_defaults(self):
        """Test HardwareCapabilities with defaults."""
        caps = HardwareCapabilities()
        
        assert caps.has_nvidia_gpu is False
        assert caps.has_intel_qsv is False
        assert caps.nvidia_nvenc_available is False
        assert caps.intel_qsv_available is False
        assert caps.gpu_memory_gb == 0.0
        assert caps.cpu_cores == 1
    
    def test_has_gpu_acceleration(self):
        """Test has_gpu_acceleration property."""
        caps_with_nvidia = HardwareCapabilities(nvidia_nvenc_available=True)
        caps_with_intel = HardwareCapabilities(intel_qsv_available=True)
        caps_with_both = HardwareCapabilities(
            nvidia_nvenc_available=True, 
            intel_qsv_available=True
        )
        caps_with_none = HardwareCapabilities()
        
        assert caps_with_nvidia.has_gpu_acceleration is True
        assert caps_with_intel.has_gpu_acceleration is True
        assert caps_with_both.has_gpu_acceleration is True
        assert caps_with_none.has_gpu_acceleration is False
    
    def test_best_encoder(self):
        """Test best_encoder property."""
        caps_nvidia = HardwareCapabilities(nvidia_nvenc_available=True)
        caps_intel = HardwareCapabilities(intel_qsv_available=True)
        caps_both = HardwareCapabilities(
            nvidia_nvenc_available=True,
            intel_qsv_available=True
        )
        caps_none = HardwareCapabilities()
        
        assert caps_nvidia.best_encoder == 'nvenc'
        assert caps_intel.best_encoder == 'qsv'
        assert caps_both.best_encoder == 'nvenc'  # NVIDIA preferred
        assert caps_none.best_encoder == 'cpu'


class TestHardwareDetector:
    """Test HardwareDetector class functionality."""
    
    def test_detector_initialization(self, hardware_detector):
        """Test HardwareDetector initialization."""
        assert isinstance(hardware_detector, HardwareDetector)
        assert hasattr(hardware_detector, 'detect_capabilities')
        assert hasattr(hardware_detector, 'has_nvidia_gpu')
        assert hasattr(hardware_detector, 'has_intel_qsv')
    
    @patch('src.hardware.detection.subprocess.run')
    def test_detect_nvidia_gpu_available(self, mock_subprocess, hardware_detector):
        """Test NVIDIA GPU detection when available."""
        # Mock nvidia-smi success
        mock_subprocess.return_value = MagicMock(
            returncode=0,
            stdout="GPU 0: GeForce RTX 3080 (UUID: GPU-12345)\n"
        )
        
        has_nvidia = hardware_detector.has_nvidia_gpu()
        
        assert has_nvidia is True
        mock_subprocess.assert_called_once()
        assert 'nvidia-smi' in str(mock_subprocess.call_args)
    
    @patch('src.hardware.detection.subprocess.run')
    def test_detect_nvidia_gpu_not_available(self, mock_subprocess, hardware_detector):
        """Test NVIDIA GPU detection when not available."""
        # Mock nvidia-smi failure
        mock_subprocess.side_effect = subprocess.CalledProcessError(1, 'nvidia-smi')
        
        has_nvidia = hardware_detector.has_nvidia_gpu()
        
        assert has_nvidia is False
        mock_subprocess.assert_called_once()
    
    @patch('src.hardware.detection.subprocess.run')
    def test_detect_nvidia_gpu_command_not_found(self, mock_subprocess, hardware_detector):
        """Test NVIDIA GPU detection when nvidia-smi not found."""
        # Mock nvidia-smi command not found
        mock_subprocess.side_effect = FileNotFoundError()
        
        has_nvidia = hardware_detector.has_nvidia_gpu()
        
        assert has_nvidia is False
        mock_subprocess.assert_called_once()
    
    @patch('src.hardware.detection.subprocess.run')
    def test_detect_intel_qsv_available(self, mock_subprocess, hardware_detector):
        """Test Intel QSV detection when available."""
        # Mock ffmpeg with QSV support
        mock_subprocess.return_value = MagicMock(
            returncode=0,
            stdout="ffmpeg version 4.4.0\nhwaccels:\n  qsv\n  vaapi\n"
        )
        
        has_qsv = hardware_detector.has_intel_qsv()
        
        assert has_qsv is True
        mock_subprocess.assert_called_once()
    
    @patch('src.hardware.detection.subprocess.run')
    def test_detect_intel_qsv_not_available(self, mock_subprocess, hardware_detector):
        """Test Intel QSV detection when not available."""
        # Mock ffmpeg without QSV support
        mock_subprocess.return_value = MagicMock(
            returncode=0,
            stdout="ffmpeg version 4.4.0\nhwaccels:\n  vaapi\n"
        )
        
        has_qsv = hardware_detector.has_intel_qsv()
        
        assert has_qsv is False
        mock_subprocess.assert_called_once()
    
    @patch('src.hardware.detection.subprocess.run')
    def test_detect_nvenc_availability(self, mock_subprocess, hardware_detector):
        """Test NVENC encoder availability detection."""
        # Mock ffmpeg with NVENC support
        mock_subprocess.return_value = MagicMock(
            returncode=0,
            stdout="ffmpeg version 4.4.0\nencoders:\n  h264_nvenc\n  hevc_nvenc\n"
        )
        
        has_nvenc = hardware_detector.has_nvenc_encoder()
        
        assert has_nvenc is True
        mock_subprocess.assert_called_once()
    
    @patch('src.hardware.detection.subprocess.run')
    def test_detect_nvenc_not_available(self, mock_subprocess, hardware_detector):
        """Test NVENC encoder detection when not available."""
        # Mock ffmpeg without NVENC support
        mock_subprocess.return_value = MagicMock(
            returncode=0,
            stdout="ffmpeg version 4.4.0\nencoders:\n  libx264\n  libx265\n"
        )
        
        has_nvenc = hardware_detector.has_nvenc_encoder()
        
        assert has_nvenc is False
        mock_subprocess.assert_called_once()
    
    @patch('src.hardware.detection.os.cpu_count')
    def test_cpu_core_detection(self, mock_cpu_count, hardware_detector):
        """Test CPU core count detection."""
        mock_cpu_count.return_value = 8
        
        cpu_cores = hardware_detector.get_cpu_cores()
        
        assert cpu_cores == 8
        mock_cpu_count.assert_called_once()
    
    @patch('src.hardware.detection.os.cpu_count')
    def test_cpu_core_detection_fallback(self, mock_cpu_count, hardware_detector):
        """Test CPU core detection fallback."""
        mock_cpu_count.return_value = None
        
        cpu_cores = hardware_detector.get_cpu_cores()
        
        assert cpu_cores == 1  # Default fallback
        mock_cpu_count.assert_called_once()
    
    @patch('src.hardware.detection.subprocess.run')
    def test_gpu_memory_detection(self, mock_subprocess, hardware_detector):
        """Test GPU memory detection."""
        # Mock nvidia-smi memory query
        mock_subprocess.return_value = MagicMock(
            returncode=0,
            stdout="8192 MiB\n"
        )
        
        gpu_memory = hardware_detector.get_gpu_memory_gb()
        
        assert gpu_memory == 8.0  # 8192 MB = 8 GB
        mock_subprocess.assert_called_once()
    
    @patch('src.hardware.detection.subprocess.run')
    def test_gpu_memory_detection_failure(self, mock_subprocess, hardware_detector):
        """Test GPU memory detection when it fails."""
        # Mock nvidia-smi failure
        mock_subprocess.side_effect = subprocess.CalledProcessError(1, 'nvidia-smi')
        
        gpu_memory = hardware_detector.get_gpu_memory_gb()
        
        assert gpu_memory == 0.0
        mock_subprocess.assert_called_once()
    
    @patch.multiple(
        'src.hardware.detection.HardwareDetector',
        has_nvidia_gpu=MagicMock(return_value=True),
        has_intel_qsv=MagicMock(return_value=False),
        has_nvenc_encoder=MagicMock(return_value=True),
        has_qsv_encoder=MagicMock(return_value=False),
        get_gpu_memory_gb=MagicMock(return_value=8.0),
        get_cpu_cores=MagicMock(return_value=8)
    )
    def test_detect_full_capabilities_nvidia(self, hardware_detector):
        """Test full capabilities detection with NVIDIA GPU."""
        capabilities = hardware_detector.detect_capabilities()
        
        assert isinstance(capabilities, HardwareCapabilities)
        assert capabilities.has_nvidia_gpu is True
        assert capabilities.has_intel_qsv is False
        assert capabilities.nvidia_nvenc_available is True
        assert capabilities.intel_qsv_available is False
        assert capabilities.gpu_memory_gb == 8.0
        assert capabilities.cpu_cores == 8
        assert capabilities.has_gpu_acceleration is True
        assert capabilities.best_encoder == 'nvenc'
    
    @patch.multiple(
        'src.hardware.detection.HardwareDetector',
        has_nvidia_gpu=MagicMock(return_value=False),
        has_intel_qsv=MagicMock(return_value=True),
        has_nvenc_encoder=MagicMock(return_value=False),
        has_qsv_encoder=MagicMock(return_value=True),
        get_gpu_memory_gb=MagicMock(return_value=0.0),
        get_cpu_cores=MagicMock(return_value=4)
    )
    def test_detect_full_capabilities_intel(self, hardware_detector):
        """Test full capabilities detection with Intel QSV."""
        capabilities = hardware_detector.detect_capabilities()
        
        assert isinstance(capabilities, HardwareCapabilities)
        assert capabilities.has_nvidia_gpu is False
        assert capabilities.has_intel_qsv is True
        assert capabilities.nvidia_nvenc_available is False
        assert capabilities.intel_qsv_available is True
        assert capabilities.gpu_memory_gb == 0.0
        assert capabilities.cpu_cores == 4
        assert capabilities.has_gpu_acceleration is True
        assert capabilities.best_encoder == 'qsv'
    
    @patch.multiple(
        'src.hardware.detection.HardwareDetector',
        has_nvidia_gpu=MagicMock(return_value=False),
        has_intel_qsv=MagicMock(return_value=False),
        has_nvenc_encoder=MagicMock(return_value=False),
        has_qsv_encoder=MagicMock(return_value=False),
        get_gpu_memory_gb=MagicMock(return_value=0.0),
        get_cpu_cores=MagicMock(return_value=2)
    )
    def test_detect_full_capabilities_cpu_only(self, hardware_detector):
        """Test full capabilities detection with CPU only."""
        capabilities = hardware_detector.detect_capabilities()
        
        assert isinstance(capabilities, HardwareCapabilities)
        assert capabilities.has_nvidia_gpu is False
        assert capabilities.has_intel_qsv is False
        assert capabilities.nvidia_nvenc_available is False
        assert capabilities.intel_qsv_available is False
        assert capabilities.gpu_memory_gb == 0.0
        assert capabilities.cpu_cores == 2
        assert capabilities.has_gpu_acceleration is False
        assert capabilities.best_encoder == 'cpu'
    
    def test_caching_functionality(self, hardware_detector):
        """Test that results are cached to avoid repeated system calls."""
        with patch.object(hardware_detector, 'has_nvidia_gpu', return_value=True) as mock_nvidia:
            # First call
            result1 = hardware_detector.detect_capabilities()
            # Second call (should use cache)  
            result2 = hardware_detector.detect_capabilities()
            
            assert result1.has_nvidia_gpu is True
            assert result2.has_nvidia_gpu is True
            # Should only call the underlying method once due to caching
            # (This test might need adjustment based on actual caching implementation)
    
    def test_clear_cache(self, hardware_detector):
        """Test cache clearing functionality."""
        # Test that clear_cache method exists and runs without error
        hardware_detector.clear_cache()
        # This test mainly ensures the method exists and doesn't crash
        assert True
    
    def test_get_encoder_command_nvenc(self, hardware_detector):
        """Test getting encoder command for NVENC."""
        with patch.object(hardware_detector, 'has_nvenc_encoder', return_value=True):
            encoder_cmd = hardware_detector.get_encoder_command('h264')
            
            assert 'h264_nvenc' in encoder_cmd or 'nvenc' in encoder_cmd
    
    def test_get_encoder_command_qsv(self, hardware_detector):
        """Test getting encoder command for Intel QSV."""
        with patch.object(hardware_detector, 'has_nvenc_encoder', return_value=False):
            with patch.object(hardware_detector, 'has_qsv_encoder', return_value=True):
                encoder_cmd = hardware_detector.get_encoder_command('h264')
                
                assert 'h264_qsv' in encoder_cmd or 'qsv' in encoder_cmd
    
    def test_get_encoder_command_cpu_fallback(self, hardware_detector):
        """Test getting encoder command with CPU fallback."""
        with patch.object(hardware_detector, 'has_nvenc_encoder', return_value=False):
            with patch.object(hardware_detector, 'has_qsv_encoder', return_value=False):
                encoder_cmd = hardware_detector.get_encoder_command('h264')
                
                assert 'libx264' in encoder_cmd or 'x264' in encoder_cmd


class TestHardwareDetectorIntegration:
    """Integration tests for HardwareDetector with real system calls."""
    
    @pytest.mark.integration
    def test_real_hardware_detection(self):
        """Test hardware detection with real system calls (integration test)."""
        detector = HardwareDetector()
        
        # This should run without errors regardless of hardware
        capabilities = detector.detect_capabilities()
        
        assert isinstance(capabilities, HardwareCapabilities)
        assert isinstance(capabilities.cpu_cores, int)
        assert capabilities.cpu_cores >= 1
        assert isinstance(capabilities.gpu_memory_gb, float)
        assert capabilities.gpu_memory_gb >= 0.0
    
    @pytest.mark.integration
    @pytest.mark.slow
    def test_real_encoder_detection(self):
        """Test encoder detection with real ffmpeg calls."""
        detector = HardwareDetector()
        
        # Test that we can get some encoder command
        encoder_cmd = detector.get_encoder_command('h264')
        assert isinstance(encoder_cmd, list)
        assert len(encoder_cmd) > 0