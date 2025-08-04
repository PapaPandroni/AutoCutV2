#!/usr/bin/env python3
"""
Test Script for Step 5: Video Rendering

Tests the complete video creation pipeline:
- Timeline → Actual MP4 file rendering
- MoviePy integration and clip loading
- Music synchronization (no audio manipulation)
- Progress callbacks and error handling
- File creation and validation

This tests the FINAL piece - turning our perfect beat-synced timelines
into actual video files that users can watch!
"""

import os
import sys
import time
from pathlib import Path

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def test_render_components():
    """Test individual rendering components."""
    print("=== Testing Rendering Components ===")
    
    from src.clip_assembler import add_transitions, ClipTimeline
    
    # Test 1: Timeline validation for rendering
    print("\n1. Testing timeline preparation...")
    timeline = ClipTimeline()
    
    # Add test clips (using actual test files)
    test_video = "test_media/853957-hd_1920_1080_30fps.mp4"
    if os.path.exists(test_video):
        timeline.add_clip(test_video, 0.0, 2.0, 0.0, 75.5)
        timeline.add_clip(test_video, 5.0, 7.0, 2.5, 68.2)
        timeline.add_clip(test_video, 10.0, 12.0, 5.0, 82.1)
        
        print(f"  ✅ Timeline created with {len(timeline.clips)} clips")
        print(f"  📊 Total duration: {timeline.get_total_duration():.1f}s")
        
        # Validate timeline for rendering
        validation = timeline.validate_timeline()
        print(f"  ✅ Timeline validation: {'PASS' if validation['valid'] else 'FAIL'}")
    else:
        print(f"  ⚠️  Test video not found: {test_video}")
    
    print("✅ Component tests completed!")


def test_simple_rendering():
    """Test basic video rendering with minimal setup."""
    print("\n=== Testing Simple Rendering ===")
    
    # Check for required files
    test_media_dir = Path("test_media")
    video_files = list(test_media_dir.glob("*.mp4"))[:1]  # Use just 1 video
    audio_files = list(test_media_dir.glob("*.mp3"))
    
    if not video_files or not audio_files:
        print("❌ Test media files not found!")
        return False
    
    video_file = str(video_files[0])
    audio_file = str(audio_files[0])
    
    print(f"📁 Using video: {Path(video_file).name}")
    print(f"📁 Using audio: {Path(audio_file).name}")
    
    from src.audio_analyzer import analyze_audio
    from src.video_analyzer import analyze_video_file
    from src.clip_assembler import match_clips_to_beats, render_video
    
    try:
        print("\n🎵 Step 1: Analyzing audio...")
        audio_data = analyze_audio(audio_file)
        print(f"  ✅ BPM: {audio_data['bpm']:.1f}, Duration: {audio_data['duration']:.1f}s")
        
        print("\n🎬 Step 2: Analyzing video...")
        video_chunks = analyze_video_file(video_file)
        print(f"  ✅ Found {len(video_chunks)} video chunks")
        
        print("\n🎯 Step 3: Creating timeline...")
        timeline = match_clips_to_beats(
            video_chunks, 
            audio_data['beats'], 
            audio_data['allowed_durations'], 
            'balanced'
        )
        print(f"  ✅ Timeline created with {len(timeline.clips)} clips")
        
        print("\n🎬 Step 4: RENDERING VIDEO...")
        output_path = "output/test_step5_simple.mp4"
        os.makedirs("output", exist_ok=True)
        
        # Progress tracking
        progress_log = []
        def progress_callback(step: str, progress: float):
            progress_log.append((step, progress))
            print(f"  [{progress*100:5.1f}%] {step}")
        
        start_time = time.time()
        
        result_path = render_video(
            timeline=timeline,
            audio_file=audio_file,
            output_path=output_path,
            progress_callback=progress_callback
        )
        
        elapsed = time.time() - start_time
        print(f"  ✅ Rendering completed in {elapsed:.1f}s")
        print(f"  📄 Output: {result_path}")
        
        # Validate output file
        if os.path.exists(result_path):
            file_size = os.path.getsize(result_path) / (1024 * 1024)  # MB
            print(f"  📊 File size: {file_size:.1f} MB")
            print(f"  ✅ Video file created successfully!")
            return True
        else:
            print(f"  ❌ Output file not found: {result_path}")
            return False
            
    except Exception as e:
        print(f"  ❌ Rendering failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def test_full_pipeline():
    """Test complete pipeline with multiple videos."""
    print("\n=== Testing Full Pipeline ===")
    
    # Use the existing assemble_clips function which now includes rendering
    from src.clip_assembler import assemble_clips
    
    # Check for test media
    test_media_dir = Path("test_media")
    video_files = list(test_media_dir.glob("*.mp4"))[:2]  # Use 2 videos
    audio_files = list(test_media_dir.glob("*.mp3"))
    
    if not video_files or not audio_files:
        print("❌ Test media files not found!")
        return False
    
    video_files = [str(vf) for vf in video_files]
    audio_file = str(audio_files[0])
    
    print(f"📁 Using {len(video_files)} videos")
    print(f"📁 Using audio: {Path(audio_file).name}")
    
    # Progress tracking
    def progress_callback(step: str, progress: float):
        print(f"  [{progress*100:5.1f}%] {step}")
    
    try:
        print("\n🚀 Running complete assembly + rendering pipeline...")
        output_path = "output/test_step5_full.mp4"
        
        start_time = time.time()
        
        result_path = assemble_clips(
            video_files=video_files,
            audio_file=audio_file,
            output_path=output_path,
            pattern='energetic',  # Try energetic pattern
            progress_callback=progress_callback
        )
        
        elapsed = time.time() - start_time
        print(f"  ✅ Complete pipeline finished in {elapsed:.1f}s")
        print(f"  📄 Final video: {result_path}")
        
        # Validate final output
        if os.path.exists(result_path):
            file_size = os.path.getsize(result_path) / (1024 * 1024)  # MB
            print(f"  📊 Final file size: {file_size:.1f} MB")
            
            # Check timeline was also exported
            timeline_json = output_path.replace('.mp4', '_timeline.json')
            if os.path.exists(timeline_json):
                print(f"  📄 Timeline JSON also created: {Path(timeline_json).name}")
            
            print(f"  🎉 SUCCESS: Complete AutoCut pipeline working!")
            return True
        else:
            print(f"  ❌ Final output not created: {result_path}")
            return False
            
    except Exception as e:
        print(f"  ❌ Full pipeline failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def test_error_handling():
    """Test error handling and edge cases."""
    print("\n=== Testing Error Handling ===")
    
    from src.clip_assembler import render_video, ClipTimeline
    
    # Test 1: Empty timeline
    print("\n1. Testing empty timeline...")
    empty_timeline = ClipTimeline()
    try:
        render_video(empty_timeline, "test_media/soft-positive-summer-pop-218419.mp3", "output/test_empty.mp4")
        print("  ❌ Should have failed with empty timeline")
    except ValueError as e:
        print(f"  ✅ Correctly caught empty timeline error: {str(e)}")
    
    # Test 2: Missing audio file
    print("\n2. Testing missing audio file...")
    timeline = ClipTimeline()
    timeline.add_clip("test_media/853957-hd_1920_1080_30fps.mp4", 0.0, 2.0, 0.0, 75.0)
    try:
        render_video(timeline, "nonexistent_audio.mp3", "output/test_missing.mp4")
        print("  ❌ Should have failed with missing audio")
    except FileNotFoundError as e:
        print(f"  ✅ Correctly caught missing audio error: {str(e)}")
    
    print("✅ Error handling tests completed!")


def main():
    """Run all Step 5 tests."""
    print("🎬 AutoCut Step 5: Video Rendering Testing")
    print("=" * 60)
    print("This tests the FINAL PIECE of AutoCut:")
    print("- Timeline → Actual MP4 file creation")
    print("- MoviePy integration and music sync")
    print("- Complete pipeline: Audio → Video → Assembly → RENDERING")
    print("=" * 60)
    
    try:
        # Test individual components
        test_render_components()
        
        # Test simple rendering
        simple_success = test_simple_rendering()
        
        # Test full pipeline if simple worked
        full_success = False
        if simple_success:
            full_success = test_full_pipeline()
        
        # Test error handling
        test_error_handling()
        
        print("\n" + "=" * 60)
        if simple_success and full_success:
            print("🎉 ALL STEP 5 TESTS PASSED!")
            print("   Video rendering is working!")
            print("   🎬 AutoCut can now create actual video files!")
            print("   Ready for Step 6: GUI")
        elif simple_success:
            print("✅ Basic rendering works!")
            print("⚠️  Full pipeline had issues - check output above")
        else:
            print("❌ Video rendering not working yet")
            
        print("=" * 60)
        
        return simple_success and full_success
        
    except Exception as e:
        print(f"\n❌ Test suite crashed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)