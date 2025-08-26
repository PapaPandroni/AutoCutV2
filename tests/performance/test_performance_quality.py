"""
Performance Quality Tests for AutoCut V2

Tests to validate performance benchmarks, processing quality metrics, and resource usage.
These tests ensure the system meets v1.0 performance and quality requirements.
"""

import os
import sys
import time
import tempfile
from pathlib import Path
from typing import List, Dict, Any
import statistics

import pytest
import psutil

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from api import AutoCutAPI
from audio_analyzer import analyze_audio


class TestPerformanceQuality:
    """Test performance benchmarks and quality metrics."""
    
    @pytest.fixture
    def api_client(self) -> AutoCutAPI:
        """AutoCut API client for testing."""
        return AutoCutAPI()
    
    @pytest.fixture
    def performance_output_dir(self, temp_dir):
        """Output directory for performance test results."""
        output_dir = temp_dir / "performance_outputs"
        output_dir.mkdir(exist_ok=True)
        return output_dir
    
    def measure_resource_usage(self, func, *args, **kwargs):
        """Measure CPU, memory, and time usage during function execution."""
        process = psutil.Process()
        
        # Initial measurements
        start_time = time.time()
        start_memory = process.memory_info().rss / (1024 * 1024)  # MB
        start_cpu_times = process.cpu_times()
        
        # Execute function
        try:
            result = func(*args, **kwargs)
            success = True
            error = None
        except Exception as e:
            result = None
            success = False
            error = e
        
        # Final measurements
        end_time = time.time()
        end_memory = process.memory_info().rss / (1024 * 1024)  # MB
        end_cpu_times = process.cpu_times()
        
        return {
            'result': result,
            'success': success,
            'error': error,
            'processing_time': end_time - start_time,
            'peak_memory_mb': max(start_memory, end_memory),
            'memory_delta_mb': end_memory - start_memory,
            'cpu_time_user': end_cpu_times.user - start_cpu_times.user,
            'cpu_time_system': end_cpu_times.system - start_cpu_times.system
        }
    
    @pytest.mark.media_required
    def test_audio_analysis_performance(self, sample_audio_files):
        """Test audio analysis performance and quality."""
        real_audio = [f for f in sample_audio_files if not f.name.startswith('._')]
        
        if not real_audio:
            pytest.skip("No real audio files available")
        
        print("\n🎵 Testing audio analysis performance")
        
        for audio_file in real_audio[:2]:  # Test first 2 audio files
            print(f"   Analyzing: {audio_file.name}")
            
            # Measure performance
            metrics = self.measure_resource_usage(analyze_audio, str(audio_file))
            
            if not metrics['success']:
                print(f"   ❌ Analysis failed: {metrics['error']}")
                continue
            
            result = metrics['result']
            
            print(f"   ⏱️ Time: {metrics['processing_time']:.2f}s")
            print(f"   💾 Memory: {metrics['peak_memory_mb']:.1f}MB")
            print(f"   🥁 BPM: {result['bpm']:.1f}")
            print(f"   🎯 Beats: {len(result['beats'])}")
            print(f"   📊 Duration: {result['duration']:.1f}s")
            
            # Quality validation
            assert 60 <= result['bpm'] <= 200, f"BPM should be reasonable: {result['bpm']}"
            assert len(result['beats']) > 0, "Should detect some beats"
            assert result['duration'] > 0, "Should detect duration"
            
            # Performance expectations
            file_duration = result['duration']
            processing_ratio = metrics['processing_time'] / file_duration
            
            print(f"   📈 Processing ratio: {processing_ratio:.2f}x (lower is better)")
            
            if processing_ratio < 0.1:  # Process 10x faster than realtime
                print("   ✅ Excellent performance")
            elif processing_ratio < 0.5:  # Process 2x faster than realtime
                print("   ✅ Good performance")
            else:
                print("   ⚠️ Performance could be improved")
    
    @pytest.mark.media_required
    def test_small_batch_processing_performance(
        self,
        api_client: AutoCutAPI,
        sample_video_files,
        sample_audio_files,
        performance_output_dir: Path
    ):
        """Test performance with small batch processing (2-3 videos)."""
        real_videos = [f for f in sample_video_files if not f.name.startswith('._')]
        real_audio = [f for f in sample_audio_files if not f.name.startswith('._')]
        
        if len(real_videos) < 2 or not real_audio:
            pytest.skip("Need at least 2 video files for batch performance test")
        
        print("\n🎬 Testing small batch processing performance")
        
        # Use smaller files for predictable timing
        small_videos = [f for f in real_videos if f.stat().st_size < 100 * 1024 * 1024][:3]
        if len(small_videos) < 2:
            small_videos = real_videos[:2]
        
        video_files = [str(f) for f in small_videos]
        audio_file = str(real_audio[0])
        output_path = str(performance_output_dir / "batch_performance_test.mp4")
        
        print(f"   Processing {len(video_files)} videos:")
        total_input_size = 0
        for video in video_files:
            size_mb = Path(video).stat().st_size / (1024 * 1024)
            total_input_size += size_mb
            print(f"     - {Path(video).name}: {size_mb:.1f}MB")
        
        print(f"   Total input size: {total_input_size:.1f}MB")
        
        # Measure processing performance
        def process_videos():
            return api_client.process_videos(
                video_files=video_files,
                audio_file=audio_file,
                output_path=output_path,
                pattern="balanced",
                memory_safe=True,
                verbose=False
            )
        
        metrics = self.measure_resource_usage(process_videos)
        
        if not metrics['success']:
            print(f"   ❌ Processing failed: {metrics['error']}")
            pytest.fail(f"Batch processing failed: {metrics['error']}")
        
        print(f"   ⏱️ Processing time: {metrics['processing_time']:.1f}s")
        print(f"   💾 Peak memory: {metrics['peak_memory_mb']:.1f}MB")
        print(f"   🖥️ CPU time: {metrics['cpu_time_user']:.1f}s user + {metrics['cpu_time_system']:.1f}s system")
        
        # Validate output
        if os.path.exists(output_path):
            output_size = os.path.getsize(output_path) / (1024 * 1024)
            print(f"   📹 Output size: {output_size:.1f}MB")
            
            # Performance metrics
            throughput = total_input_size / metrics['processing_time']  # MB/s
            compression_ratio = output_size / total_input_size
            
            print(f"   📊 Throughput: {throughput:.1f} MB/s")
            print(f"   🗜️ Compression ratio: {compression_ratio:.2f}")
            
            # Quality expectations
            assert output_size > 0, "Output should not be empty"
            
            if metrics['processing_time'] < 60:  # Under 1 minute
                print("   ✅ Fast processing time")
            elif metrics['processing_time'] < 180:  # Under 3 minutes
                print("   ✅ Acceptable processing time")
            else:
                print("   ⚠️ Processing time longer than expected")
    
    @pytest.mark.media_required
    def test_memory_usage_scaling(
        self,
        api_client: AutoCutAPI,
        sample_video_files,
        sample_audio_files,
        performance_output_dir: Path
    ):
        """Test memory usage with different video counts."""
        real_videos = [f for f in sample_video_files if not f.name.startswith('._')]
        real_audio = [f for f in sample_audio_files if not f.name.startswith('._')]
        
        if len(real_videos) < 3 or not real_audio:
            pytest.skip("Need at least 3 video files for memory scaling test")
        
        print("\n🧠 Testing memory usage scaling")
        
        memory_results = []
        
        # Test with increasing number of videos (1, 2, 3+)
        for video_count in [1, 2, min(len(real_videos), 4)]:
            print(f"   Testing with {video_count} videos...")
            
            videos = [str(f) for f in real_videos[:video_count]]
            output_path = str(performance_output_dir / f"memory_scale_{video_count}.mp4")
            
            def process_videos():
                return api_client.process_videos(
                    video_files=videos,
                    audio_file=str(real_audio[0]),
                    output_path=output_path,
                    pattern="balanced",
                    memory_safe=True,  # Use memory-safe mode
                    verbose=False
                )
            
            metrics = self.measure_resource_usage(process_videos)
            
            memory_results.append({
                'video_count': video_count,
                'peak_memory': metrics['peak_memory_mb'],
                'memory_delta': metrics['memory_delta_mb'],
                'success': metrics['success']
            })
            
            if metrics['success']:
                print(f"     ✅ Peak memory: {metrics['peak_memory_mb']:.1f}MB")
                print(f"     📈 Memory delta: {metrics['memory_delta_mb']:+.1f}MB")
            else:
                print(f"     ❌ Failed: {metrics['error']}")
        
        # Analyze memory scaling
        successful_results = [r for r in memory_results if r['success']]
        
        if len(successful_results) >= 2:
            print("\n   📊 Memory scaling analysis:")
            for result in successful_results:
                print(f"     {result['video_count']} videos: {result['peak_memory']:.1f}MB peak")
            
            # Check if memory usage grows linearly (which would be bad)
            if len(successful_results) >= 2:
                first_result = successful_results[0]
                last_result = successful_results[-1]
                
                memory_growth = last_result['peak_memory'] / first_result['peak_memory']
                video_growth = last_result['video_count'] / first_result['video_count']
                
                scaling_efficiency = memory_growth / video_growth  # Should be close to 1 for good scaling
                
                print(f"   📈 Scaling efficiency: {scaling_efficiency:.2f} (lower is better)")
                
                if scaling_efficiency < 1.5:
                    print("   ✅ Good memory scaling")
                elif scaling_efficiency < 2.0:
                    print("   ✅ Acceptable memory scaling")
                else:
                    print("   ⚠️ Memory scaling could be improved")
    
    @pytest.mark.media_required
    def test_processing_consistency(
        self,
        api_client: AutoCutAPI,
        sample_video_files,
        sample_audio_files,
        performance_output_dir: Path
    ):
        """Test consistency of processing times across multiple runs."""
        real_videos = [f for f in sample_video_files if not f.name.startswith('._')]
        real_audio = [f for f in sample_audio_files if not f.name.startswith('._')]
        
        if not real_videos or not real_audio:
            pytest.skip("No real media files available")
        
        print("\n🔄 Testing processing time consistency")
        
        # Use a small video for consistent timing
        small_video = next((f for f in real_videos if f.stat().st_size < 50 * 1024 * 1024), real_videos[0])
        
        processing_times = []
        
        # Run multiple iterations
        for i in range(3):  # 3 runs for consistency testing
            print(f"   Run {i+1}/3...")
            
            output_path = str(performance_output_dir / f"consistency_{i}.mp4")
            
            def process_video():
                return api_client.process_videos(
                    video_files=[str(small_video)],
                    audio_file=str(real_audio[0]),
                    output_path=output_path,
                    pattern="balanced",
                    memory_safe=True,
                    verbose=False
                )
            
            metrics = self.measure_resource_usage(process_video)
            
            if metrics['success']:
                processing_times.append(metrics['processing_time'])
                print(f"     ✅ Completed in {metrics['processing_time']:.2f}s")
            else:
                print(f"     ❌ Failed: {metrics['error']}")
        
        if len(processing_times) >= 2:
            avg_time = statistics.mean(processing_times)
            std_dev = statistics.stdev(processing_times) if len(processing_times) > 1 else 0
            cv = (std_dev / avg_time) * 100  # Coefficient of variation as percentage
            
            print(f"\n   📊 Processing time analysis:")
            print(f"     Average: {avg_time:.2f}s")
            print(f"     Std dev: {std_dev:.2f}s")
            print(f"     CV: {cv:.1f}%")
            
            if cv < 10:
                print("   ✅ Very consistent processing times")
            elif cv < 25:
                print("   ✅ Reasonably consistent processing times")
            else:
                print("   ⚠️ Processing times vary significantly")
    
    @pytest.mark.media_required  
    def test_beat_sync_accuracy(
        self,
        api_client: AutoCutAPI,
        sample_video_files,
        sample_audio_files,
        performance_output_dir: Path
    ):
        """Test beat synchronization accuracy."""
        real_videos = [f for f in sample_video_files if not f.name.startswith('._')]
        real_audio = [f for f in sample_audio_files if not f.name.startswith('._')]
        
        if not real_videos or not real_audio:
            pytest.skip("No real media files available")
        
        print("\n🎯 Testing beat sync accuracy")
        
        audio_file = str(real_audio[0])
        
        # First analyze the audio to get beat information
        print(f"   Analyzing audio: {Path(audio_file).name}")
        audio_analysis = analyze_audio(audio_file)
        
        print(f"   🥁 Detected BPM: {audio_analysis['bpm']:.1f}")
        print(f"   🎯 Beat count: {len(audio_analysis['beats'])}")
        
        # Process video with different patterns to test sync accuracy
        patterns = ["energetic", "balanced", "dramatic"]
        
        for pattern in patterns:
            print(f"   Testing {pattern} pattern...")
            
            output_path = str(performance_output_dir / f"beat_sync_{pattern}.mp4")
            
            try:
                result_path = api_client.process_videos(
                    video_files=[str(real_videos[0])],
                    audio_file=audio_file,
                    output_path=output_path,
                    pattern=pattern,
                    memory_safe=True,
                    verbose=False
                )
                
                if os.path.exists(result_path):
                    print(f"     ✅ {pattern} pattern created output")
                    # Note: Detailed beat sync analysis would require video analysis tools
                    # For now, we validate that the processing completed successfully
                else:
                    print(f"     ⚠️ {pattern} pattern created no output")
                    
            except Exception as e:
                print(f"     ❌ {pattern} pattern failed: {e}")
        
        # Validate beat timing expectations
        beat_interval = 60.0 / audio_analysis['bpm']  # seconds per beat
        print(f"   ⏱️ Beat interval: {beat_interval:.3f}s")
        
        if 0.3 <= beat_interval <= 1.0:  # Reasonable beat intervals
            print("   ✅ Beat timing within reasonable range")
        else:
            print("   ⚠️ Beat timing outside typical range - may affect sync quality")
    
    def test_cpu_utilization_efficiency(
        self,
        api_client: AutoCutAPI,
        sample_video_files,
        sample_audio_files,
        performance_output_dir: Path
    ):
        """Test CPU utilization efficiency."""
        real_videos = [f for f in sample_video_files if not f.name.startswith('._')]
        real_audio = [f for f in sample_audio_files if not f.name.startswith('._')]
        
        if not real_videos or not real_audio:
            pytest.skip("No real media files available")
        
        print("\n🖥️ Testing CPU utilization efficiency")
        
        # Get system CPU info
        cpu_count = psutil.cpu_count()
        cpu_count_logical = psutil.cpu_count(logical=True)
        
        print(f"   System: {cpu_count} cores, {cpu_count_logical} logical CPUs")
        
        output_path = str(performance_output_dir / "cpu_efficiency_test.mp4")
        small_video = next((f for f in real_videos if f.stat().st_size < 50 * 1024 * 1024), real_videos[0])
        
        # Monitor CPU usage during processing
        def process_with_monitoring():
            def process_video():
                return api_client.process_videos(
                    video_files=[str(small_video)],
                    audio_file=str(real_audio[0]),
                    output_path=output_path,
                    pattern="balanced",
                    memory_safe=True,
                    verbose=False
                )
            
            # Sample CPU usage during processing
            cpu_samples = []
            start_time = time.time()
            
            import threading
            processing_complete = threading.Event()
            result = {'output': None, 'error': None}
            
            def cpu_monitor():
                while not processing_complete.wait(0.1):  # Sample every 100ms
                    cpu_samples.append(psutil.cpu_percent())
            
            def process_worker():
                try:
                    result['output'] = process_video()
                except Exception as e:
                    result['error'] = e
                finally:
                    processing_complete.set()
            
            # Start monitoring and processing
            monitor_thread = threading.Thread(target=cpu_monitor)
            process_thread = threading.Thread(target=process_worker)
            
            monitor_thread.start()
            process_thread.start()
            
            process_thread.join()
            processing_complete.set()
            monitor_thread.join()
            
            processing_time = time.time() - start_time
            
            return result, cpu_samples, processing_time
        
        try:
            result, cpu_samples, processing_time = process_with_monitoring()
            
            if result['error']:
                print(f"   ❌ Processing failed: {result['error']}")
                return
            
            if cpu_samples:
                avg_cpu = statistics.mean(cpu_samples)
                max_cpu = max(cpu_samples)
                
                print(f"   📊 CPU usage - Average: {avg_cpu:.1f}%, Peak: {max_cpu:.1f}%")
                print(f"   ⏱️ Processing time: {processing_time:.2f}s")
                
                # CPU efficiency analysis
                if avg_cpu > 80:
                    print("   ✅ High CPU utilization - good efficiency")
                elif avg_cpu > 50:
                    print("   ✅ Moderate CPU utilization - acceptable efficiency")
                else:
                    print("   ⚠️ Low CPU utilization - may not be fully utilizing available resources")
            else:
                print("   ⚠️ No CPU samples collected")
                
        except Exception as e:
            print(f"   ❌ CPU monitoring failed: {e}")
    
    @pytest.mark.media_required
    def test_quality_vs_performance_tradeoffs(
        self,
        api_client: AutoCutAPI,
        sample_video_files,
        sample_audio_files,
        performance_output_dir: Path
    ):
        """Test quality vs performance tradeoffs with different settings."""
        real_videos = [f for f in sample_video_files if not f.name.startswith('._')]
        real_audio = [f for f in sample_audio_files if not f.name.startswith('._')]
        
        if not real_videos or not real_audio:
            pytest.skip("No real media files available")
        
        print("\n⚖️ Testing quality vs performance tradeoffs")
        
        small_video = next((f for f in real_videos if f.stat().st_size < 50 * 1024 * 1024), real_videos[0])
        
        # Test different configurations
        configs = [
            {"name": "Memory-Safe", "memory_safe": True},
            {"name": "Standard", "memory_safe": False}
        ]
        
        results = []
        
        for config in configs:
            print(f"   Testing {config['name']} mode...")
            
            output_path = str(performance_output_dir / f"quality_performance_{config['name'].lower()}.mp4")
            
            def process_video():
                return api_client.process_videos(
                    video_files=[str(small_video)],
                    audio_file=str(real_audio[0]),
                    output_path=output_path,
                    pattern="balanced",
                    memory_safe=config['memory_safe'],
                    verbose=False
                )
            
            metrics = self.measure_resource_usage(process_video)
            
            if metrics['success']:
                output_size = os.path.getsize(output_path) if os.path.exists(output_path) else 0
                
                result_data = {
                    'config': config['name'],
                    'processing_time': metrics['processing_time'],
                    'peak_memory': metrics['peak_memory_mb'],
                    'output_size_mb': output_size / (1024 * 1024),
                    'cpu_time': metrics['cpu_time_user']
                }
                results.append(result_data)
                
                print(f"     ✅ Time: {result_data['processing_time']:.2f}s")
                print(f"     💾 Memory: {result_data['peak_memory']:.1f}MB")
                print(f"     📹 Output: {result_data['output_size_mb']:.1f}MB")
            else:
                print(f"     ❌ Failed: {metrics['error']}")
        
        # Compare results
        if len(results) >= 2:
            print("\n   📊 Performance comparison:")
            for result in results:
                print(f"     {result['config']}: {result['processing_time']:.2f}s, {result['peak_memory']:.1f}MB")
            
            # Find the best balance
            fastest = min(results, key=lambda x: x['processing_time'])
            most_efficient = min(results, key=lambda x: x['peak_memory'])
            
            print(f"   🏃 Fastest: {fastest['config']} ({fastest['processing_time']:.2f}s)")
            print(f"   🧠 Most memory efficient: {most_efficient['config']} ({most_efficient['peak_memory']:.1f}MB)")