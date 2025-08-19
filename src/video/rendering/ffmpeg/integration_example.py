"""
Integration example for Phase 1 Memory Optimization.

Shows how to use the new streaming video analysis system
alongside the existing MoviePy pipeline for memory efficiency.
"""

import logging
import time
from typing import List, Dict, Any
from pathlib import Path

from .streaming_analyzer import StreamingVideoAnalyzer, VideoAnalysisResult
from .hardware_accelerator import M2HardwareAccelerator
from .memory_monitor import MemoryMonitor
from .cache_manager import AnalysisCacheManager

logger = logging.getLogger(__name__)


class HybridVideoProcessor:
    """
    Hybrid video processor that uses streaming analysis for memory efficiency
    while maintaining compatibility with existing MoviePy rendering pipeline.
    """
    
    def __init__(self):
        """Initialize hybrid processor with all optimization components."""
        
        # Initialize optimization components
        self.hw_accelerator = M2HardwareAccelerator()
        self.memory_monitor = MemoryMonitor(
            warning_threshold=50.0,  # Conservative thresholds
            critical_threshold=65.0,
            check_interval=2.0
        )
        self.cache_manager = AnalysisCacheManager(
            max_cache_size_mb=500,
            max_cache_age_days=30
        )
        self.streaming_analyzer = StreamingVideoAnalyzer(
            hardware_accelerator=self.hw_accelerator,
            memory_monitor=self.memory_monitor,
            analysis_fps=2,  # Analyze every 2 seconds
            max_analysis_resolution=(640, 360)  # Memory-efficient resolution
        )
        
        # Start memory monitoring
        self.memory_monitor.start_monitoring()
        
        logger.info("Hybrid Video Processor initialized with memory optimization")
        
        # Log system capabilities
        if self.hw_accelerator.videotoolbox_available:
            logger.info("✅ Apple Silicon hardware acceleration available")
        else:
            logger.info("⚠️ Using software processing (non-Apple Silicon)")
    
    def analyze_videos_efficiently(self, video_paths: List[str]) -> Dict[str, VideoAnalysisResult]:
        """
        Analyze multiple videos using streaming approach for minimal memory usage.
        
        This replaces the memory-intensive VideoFileClip loading with streaming analysis.
        """
        logger.info(f"Starting efficient analysis of {len(video_paths)} videos")
        
        results = {}
        cache_hits = 0
        total_analysis_time = 0
        
        for i, video_path in enumerate(video_paths):
            video_name = Path(video_path).name
            logger.info(f"Processing video {i+1}/{len(video_paths)}: {video_name}")
            
            # Check memory pressure
            if self.memory_monitor.should_pause_processing():
                logger.warning("Memory pressure detected - waiting for relief")
                self.memory_monitor.wait_for_memory_relief(timeout=30.0)
            
            # Check cache first
            cached_result = self.cache_manager.get_cached_analysis(video_path)
            if cached_result:
                results[video_path] = cached_result
                cache_hits += 1
                logger.info(f"✅ Cache hit for {video_name}")
                continue
            
            # Perform streaming analysis
            try:
                start_time = time.time()
                
                analysis_result = self.streaming_analyzer.analyze_video_streaming(video_path)
                
                analysis_time = time.time() - start_time
                total_analysis_time += analysis_time
                
                # Cache the result
                self.cache_manager.cache_analysis(video_path, analysis_result)
                
                results[video_path] = analysis_result
                logger.info(f"✅ Analyzed {video_name} in {analysis_time:.1f}s")
                
            except Exception as e:
                logger.error(f"❌ Analysis failed for {video_name}: {e}")
                # Continue with other videos
                continue
        
        # Log summary statistics
        cache_hit_rate = (cache_hits / len(video_paths)) * 100 if video_paths else 0
        avg_analysis_time = total_analysis_time / (len(video_paths) - cache_hits) if (len(video_paths) - cache_hits) > 0 else 0
        
        logger.info(f"Analysis complete: {len(results)}/{len(video_paths)} videos processed")
        logger.info(f"Cache hit rate: {cache_hit_rate:.1f}% ({cache_hits}/{len(video_paths)})")
        logger.info(f"Average analysis time: {avg_analysis_time:.1f}s per video")
        
        return results
    
    def get_memory_efficient_clips(self, analysis_results: Dict[str, VideoAnalysisResult],
                                 target_duration: float = 4.0) -> List[Dict[str, Any]]:
        """
        Extract best clips from analysis results without loading videos into memory.
        
        Returns clip metadata that can be used with existing MoviePy pipeline.
        """
        logger.info("Extracting best clips from analysis results")
        
        all_clips = []
        
        for video_path, analysis in analysis_results.items():
            video_name = Path(video_path).name
            
            # Extract best segments from streaming analysis
            best_segments = analysis.best_segments
            
            for segment_start, segment_end, segment_score in best_segments:
                segment_duration = segment_end - segment_start
                
                # Skip segments that are too short or too long
                if segment_duration < 1.0 or segment_duration > 8.0:
                    continue
                
                # Create clip metadata (compatible with existing pipeline)
                clip_info = {
                    'video_path': video_path,
                    'start_time': segment_start,
                    'end_time': segment_end,
                    'duration': segment_duration,
                    'quality_score': segment_score,
                    'video_name': video_name,
                    
                    # Additional metadata from streaming analysis
                    'video_width': analysis.width,
                    'video_height': analysis.height,
                    'video_fps': analysis.fps,
                    'overall_quality': analysis.overall_quality,
                }
                
                all_clips.append(clip_info)
        
        # Sort by quality score
        all_clips.sort(key=lambda x: x['quality_score'], reverse=True)
        
        logger.info(f"Extracted {len(all_clips)} potential clips from analysis")
        return all_clips
    
    def get_optimization_stats(self) -> Dict[str, Any]:
        """Get comprehensive optimization statistics."""
        
        memory_stats = self.memory_monitor.get_memory_stats()
        cache_stats = self.cache_manager.get_cache_stats()
        hw_stats = self.hw_accelerator.get_system_info()
        
        return {
            'memory': {
                'current_percent': memory_stats['current_percent'],
                'pressure_level': memory_stats['pressure_level'],
                'recommended_batch_size': self.memory_monitor.get_adaptive_batch_size(),
                'estimated_capacity': self.memory_monitor.estimate_available_capacity(),
            },
            'cache': {
                'total_entries': cache_stats['total_entries'],
                'size_mb': cache_stats['total_size_mb'],
                'usage_percent': cache_stats['usage_percent'],
            },
            'hardware': {
                'apple_silicon': hw_stats['is_apple_silicon'],
                'videotoolbox': hw_stats['videotoolbox_available'],
                'memory_total_gb': hw_stats['memory_total_gb'],
            },
            'recommendations': self.memory_monitor.get_optimization_recommendations(),
        }
    
    def cleanup(self):
        """Clean up resources."""
        self.memory_monitor.stop_monitoring()
        logger.info("Hybrid Video Processor cleanup complete")


# Usage example for integration with existing codebase
def demonstrate_integration():
    """
    Demonstrate how to integrate memory optimization with existing workflow.
    """
    
    print("🚀 Memory Optimization Integration Example")
    print("=" * 50)
    
    # Initialize hybrid processor
    processor = HybridVideoProcessor()
    
    try:
        # Example video paths (replace with actual paths)
        video_paths = [
            "test_media/IMG_0431.mov",
            "test_media/IMG_0472.mov", 
            "test_media/IMG_0488.mov",
        ]
        
        # Filter to existing files only
        existing_videos = [path for path in video_paths if Path(path).exists()]
        
        if not existing_videos:
            print("⚠️ No test videos found - skipping analysis demonstration")
            return
        
        print(f"🎬 Analyzing {len(existing_videos)} videos with streaming approach...")
        
        # Phase 1: Memory-efficient analysis (replaces VideoFileClip loading)
        analysis_results = processor.analyze_videos_efficiently(existing_videos)
        
        # Phase 2: Extract clip metadata (no video loading)
        clip_metadata = processor.get_memory_efficient_clips(analysis_results)
        
        print(f"✅ Found {len(clip_metadata)} high-quality clips")
        
        # Phase 3: Show optimization stats
        stats = processor.get_optimization_stats()
        
        print(f"\n📊 Optimization Statistics:")
        print(f"  Memory Usage: {stats['memory']['current_percent']:.1f}%")
        print(f"  Memory Pressure: {stats['memory']['pressure_level']}")
        print(f"  Cache Entries: {stats['cache']['total_entries']}")
        print(f"  Hardware Acceleration: {'✅' if stats['hardware']['videotoolbox'] else '❌'}")
        
        print(f"\n💡 Recommendations:")
        for rec in stats['recommendations']:
            print(f"  {rec}")
        
        # At this point, you would use clip_metadata with existing MoviePy pipeline:
        # 1. Use clip metadata to create VideoFileClip objects only when needed
        # 2. Process clips in batches based on memory_monitor recommendations  
        # 3. Use hardware acceleration parameters from hw_accelerator
        # 4. Cache results to avoid reprocessing
        
        print(f"\n🎯 Integration Points:")
        print(f"  1. Replace bulk VideoFileClip loading with streaming analysis")
        print(f"  2. Use clip metadata for targeted MoviePy operations")
        print(f"  3. Apply adaptive batch sizing based on memory pressure")
        print(f"  4. Leverage cache to avoid reprocessing unchanged videos")
        
    finally:
        # Always cleanup
        processor.cleanup()


if __name__ == "__main__":
    demonstrate_integration()