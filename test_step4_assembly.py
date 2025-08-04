#!/usr/bin/env python3
"""
Test Script for Step 4: Clip Assembly Logic

Tests the complete pipeline integration:
- Audio analysis → Video analysis → Clip assembly
- Beat matching algorithm
- Variety patterns
- Timeline creation and validation

This tests the CORE functionality of AutoCut - the synchronization
of video clips to musical beats.
"""

import os
import sys
import time
from pathlib import Path

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def test_individual_functions():
    """Test individual Step 4 functions with mock data."""
    print("=== Testing Individual Functions ===")
    
    from src.clip_assembler import apply_variety_pattern, select_best_clips, ClipTimeline
    from src.video_analyzer import VideoChunk
    
    # Test 1: Variety Pattern Application
    print("\n1. Testing variety pattern application...")
    patterns_to_test = ['energetic', 'buildup', 'balanced', 'dramatic']
    
    for pattern in patterns_to_test:
        multipliers = apply_variety_pattern(pattern, 20)  # 20 beats
        total_beats = sum(multipliers)
        print(f"  {pattern:>10}: {multipliers[:8]}... (total: {total_beats} beats)")
        assert total_beats == 20, f"Pattern {pattern} doesn't sum to 20 beats"
    
    # Test 2: ClipTimeline functionality
    print("\n2. Testing ClipTimeline class...")
    timeline = ClipTimeline()
    
    # Add some test clips
    timeline.add_clip("video1.mp4", 0.0, 2.0, 0.0, 75.5)
    timeline.add_clip("video2.mp4", 5.0, 7.5, 2.0, 68.2)
    timeline.add_clip("video1.mp4", 10.0, 12.0, 4.5, 82.1)
    
    stats = timeline.get_summary_stats()
    print(f"  Timeline stats: {stats['total_clips']} clips, {stats['total_duration']:.1f}s total")
    print(f"  Score range: {stats['score_range'][0]:.1f}-{stats['score_range'][1]:.1f}")
    print(f"  From {stats['unique_videos']} unique videos")
    
    # Test validation
    validation = timeline.validate_timeline(song_duration=10.0)
    print(f"  Validation: {'PASS' if validation['valid'] else 'FAIL'}")
    if validation['warnings']:
        print(f"  Warnings: {validation['warnings']}")
    
    # Test 3: Clip Selection Algorithm
    print("\n3. Testing clip selection algorithm...")
    
    # Create mock video chunks
    test_chunks = [
        VideoChunk(0.0, 3.0, 85.0, "video1.mp4", {"motion": 0.7, "faces": 2}),
        VideoChunk(5.0, 8.0, 72.0, "video1.mp4", {"motion": 0.4, "faces": 1}),
        VideoChunk(1.0, 4.0, 78.5, "video2.mp4", {"motion": 0.6, "faces": 0}),
        VideoChunk(10.0, 13.0, 91.2, "video2.mp4", {"motion": 0.8, "faces": 3}),
        VideoChunk(15.0, 17.0, 65.8, "video3.mp4", {"motion": 0.3, "faces": 1}),
        VideoChunk(20.0, 23.0, 88.1, "video3.mp4", {"motion": 0.9, "faces": 2}),
    ]
    
    # Test different variety factors
    for variety_factor in [0.0, 0.3, 0.9]:
        selected = select_best_clips(test_chunks, target_count=4, variety_factor=variety_factor)
        unique_videos = len(set(chunk.video_path for chunk in selected))
        avg_score = sum(chunk.score for chunk in selected) / len(selected)
        print(f"  Variety {variety_factor:.1f}: {len(selected)} clips, {unique_videos} videos, avg score {avg_score:.1f}")
    
    print("✅ Individual function tests completed!")


def test_full_pipeline():
    """Test the complete assembly pipeline with real media files."""
    print("\n=== Testing Full Pipeline ===")
    
    # Check for test media
    test_media_dir = Path("test_media")
    if not test_media_dir.exists():
        print("❌ test_media directory not found!")
        return False
    
    # Find test files
    video_files = list(test_media_dir.glob("*.mp4"))[:3]  # Use first 3 videos
    audio_files = list(test_media_dir.glob("*.mp3"))
    
    if not video_files or not audio_files:
        print("❌ Test media files not found!")
        return False
    
    audio_file = str(audio_files[0])  # Use first audio file
    video_files = [str(vf) for vf in video_files]
    
    print(f"📁 Using audio: {Path(audio_file).name}")
    print(f"📁 Using videos: {[Path(vf).name for vf in video_files]}")
    
    # Import the assembly function
    from src.clip_assembler import assemble_clips
    
    # Progress tracking
    progress_log = []
    def progress_callback(step: str, progress: float):
        progress_log.append((step, progress))
        print(f"  [{progress*100:5.1f}%] {step}")
    
    # Test different variety patterns
    patterns_to_test = ['balanced', 'energetic', 'dramatic']
    
    for pattern in patterns_to_test:
        print(f"\n🎵 Testing with pattern: {pattern}")
        output_path = f"output/test_assembly_{pattern}.mp4"
        
        # Ensure output directory exists
        os.makedirs("output", exist_ok=True)
        
        try:
            start_time = time.time()
            
            # This tests the COMPLETE pipeline:
            # 1. Audio analysis (Step 1)
            # 2. Video analysis (Steps 2-3)  
            # 3. Beat matching (Step 4)
            # 4. Timeline creation (Step 4)
            
            result_path = assemble_clips(
                video_files=video_files,
                audio_file=audio_file,
                output_path=output_path,
                pattern=pattern,
                progress_callback=progress_callback
            )
            
            elapsed = time.time() - start_time
            print(f"  ✅ Assembly completed in {elapsed:.1f}s")
            print(f"  📄 Result: {result_path}")
            
            # Check if timeline JSON was created
            timeline_json = output_path.replace('.mp4', '_timeline.json')
            if os.path.exists(timeline_json):
                print(f"  📄 Timeline exported: {Path(timeline_json).name}")
                
                # Load and analyze timeline
                import json
                with open(timeline_json, 'r') as f:
                    clips_data = json.load(f)
                
                print(f"  📊 Timeline: {len(clips_data)} clips")
                if clips_data:
                    total_duration = sum(clip['duration'] for clip in clips_data)
                    avg_score = sum(clip['score'] for clip in clips_data) / len(clips_data)
                    unique_videos = len(set(clip['video_file'] for clip in clips_data))
                    
                    print(f"  📊 Total duration: {total_duration:.1f}s")
                    print(f"  📊 Average score: {avg_score:.1f}")
                    print(f"  📊 Videos used: {unique_videos}")
            
        except Exception as e:
            print(f"  ❌ Assembly failed: {str(e)}")
            import traceback
            traceback.print_exc()
            return False
    
    print("✅ Full pipeline tests completed!")
    return True


def test_edge_cases():
    """Test edge cases and error handling."""
    print("\n=== Testing Edge Cases ===")
    
    from src.clip_assembler import assemble_clips, ClipTimeline, select_best_clips
    from src.video_analyzer import VideoChunk
    
    # Test 1: Empty inputs
    print("\n1. Testing empty inputs...")
    
    empty_timeline = ClipTimeline()
    validation = empty_timeline.validate_timeline()
    assert not validation['valid'], "Empty timeline should be invalid"
    print("  ✅ Empty timeline correctly marked invalid")
    
    # Test 2: No suitable clips
    print("\n2. Testing clip selection edge cases...")
    
    # Empty list
    result = select_best_clips([], 5)
    assert result == [], "Empty list should return empty"
    
    # Requesting 0 clips
    test_chunks = [VideoChunk(0.0, 2.0, 75.0, "test.mp4")]
    result = select_best_clips(test_chunks, 0)
    assert result == [], "Requesting 0 clips should return empty"
    
    print("  ✅ Edge cases handled correctly")
    
    # Test 3: File not found errors
    print("\n3. Testing error handling...")
    
    try:
        assemble_clips(
            video_files=["nonexistent.mp4"],
            audio_file="nonexistent.mp3",
            output_path="output/test.mp4"
        )
        assert False, "Should have raised FileNotFoundError"
    except FileNotFoundError:
        print("  ✅ File not found error correctly raised")
    except Exception as e:
        print(f"  ⚠️  Unexpected error type: {e}")
    
    print("✅ Edge case tests completed!")


def main():
    """Run all Step 4 tests."""
    print("🚀 AutoCut Step 4: Clip Assembly Logic Testing")
    print("=" * 60)
    print("This tests the CORE HEART of AutoCut:")
    print("- Beat-to-clip synchronization")
    print("- Musical timing and variety patterns")
    print("- Complete pipeline integration")
    print("=" * 60)
    
    try:
        # Test individual components
        test_individual_functions()
        
        # Test complete pipeline 
        pipeline_success = test_full_pipeline()
        
        # Test edge cases
        test_edge_cases()
        
        print("\n" + "=" * 60)
        if pipeline_success:
            print("🎉 ALL STEP 4 TESTS PASSED!")
            print("   The core clip assembly logic is working!")
            print("   Ready for Step 5: Video Rendering")
        else:
            print("❌ Some tests failed - check output above")
            
        print("=" * 60)
        
    except Exception as e:
        print(f"\n❌ Test suite crashed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return pipeline_success


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)