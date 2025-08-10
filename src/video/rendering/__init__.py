"""Video rendering system for AutoCut.

This package provides a clean, modular video rendering pipeline with:
- Timeline-based composition
- Format normalization 
- Audio synchronization
- Hardware-accelerated encoding
- Transition effects

The rendering system is extracted from the original clip_assembler.py god module
as part of the Phase 3 refactoring to create maintainable, testable components.
"""

from .compositor import VideoCompositor, VideoFormatAnalyzer, VideoNormalizationPipeline
from .encoder import VideoEncoder, detect_optimal_codec_settings
from .timeline import TimelineRenderer
from .transitions import TransitionEngine, add_transitions
from .audio_sync import AudioSynchronizer, load_audio_robust
from .renderer import VideoRenderingOrchestrator, render_video

__all__ = [
    # Main rendering components
    'VideoCompositor',
    'VideoEncoder', 
    'TimelineRenderer',
    'TransitionEngine',
    'AudioSynchronizer',
    'VideoRenderingOrchestrator',
    
    # Legacy compatibility functions
    'render_video',
    'add_transitions',
    'detect_optimal_codec_settings',
    'load_audio_robust',
    
    # Format handling classes
    'VideoFormatAnalyzer',
    'VideoNormalizationPipeline'
]