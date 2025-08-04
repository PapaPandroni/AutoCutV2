#!/usr/bin/env python3
"""
Test script for video analysis functions
"""
import sys
import os
sys.path.append('src')

from video_analyzer import load_video, detect_scenes, score_scene, VideoChunk
import glob

def test_video_analysis():
    print("🎬 Testing Video Analysis Functions")
    print("=" * 50)
    
    # Check for quick test flag
    import sys
    max_videos = None
    if "--quick" in sys.argv:
        max_videos = 3
        print("⚡ Quick mode: Processing first 3 videos only")
    
    # Find video files in test_media
    video_extensions = ['*.mp4', '*.avi', '*.mov', '*.mkv', '*.webm']
    video_files = []
    
    for ext in video_extensions:
        video_files.extend(glob.glob(f'test_media/{ext}'))
        video_files.extend(glob.glob(f'test_media/{ext.upper()}'))
    
    if not video_files:
        print("❌ No video files found in test_media/")
        print("Please add some video files to test_media/ folder")
        print("Supported formats: MP4, AVI, MOV, MKV, WEBM")
        return
    
    print(f"Found {len(video_files)} video file(s):")
    for i, file in enumerate(video_files, 1):
        print(f"  {i}. {os.path.basename(file)}")
    
    print("\nAnalyzing each video:")
    print("-" * 30)
    
    # Limit videos if max_videos is set
    test_videos = video_files[:max_videos] if max_videos else video_files
    
    for i, video_file in enumerate(test_videos, 1):
        filename = os.path.basename(video_file)
        print(f"\n🎬 [{i}/{len(test_videos)}] {filename}")
        
        try:
            # Test load_video function
            video, metadata = load_video(video_file)
            
            print(f"   Duration: {metadata['duration']:.1f} seconds")
            print(f"   Resolution: {metadata['width']}x{metadata['height']}")
            print(f"   FPS: {metadata['fps']:.1f}")
            
            # Test scene detection
            print("   Detecting scenes...", end=" ", flush=True)
            scenes = detect_scenes(video, threshold=30.0)
            print(f"Found {len(scenes)} scenes:")
            
            # Score each scene
            scored_scenes = []
            for i, (start, end) in enumerate(scenes[:5]):  # Limit to first 5 scenes
                print(f"     Scene {i+1}: {start:.1f}s - {end:.1f}s ({end-start:.1f}s duration)")
                
                # Score the scene
                score = score_scene(video, start, end)
                print(f"       Quality score: {score:.1f}/100")
                scored_scenes.append((start, end, score))
            
            if len(scenes) > 5:
                print(f"     ... and {len(scenes)-5} more scenes")
            
            # Show best scenes
            if scored_scenes:
                best_scene = max(scored_scenes, key=lambda x: x[2])
                print(f"   🏆 Best scene: {best_scene[0]:.1f}s-{best_scene[1]:.1f}s (score: {best_scene[2]:.1f})")
            
            # Clean up video object
            video.close()
            print("   ✅ Video analysis successful!")
            
        except Exception as e:
            print(f"   ❌ Error: {e}")
    
    print(f"\n🎯 Video analysis testing complete!")

def test_without_real_files():
    """Test video analysis functions without real files (for CI/development)"""
    print("🎬 Video Analysis - Mock Test Mode")
    print("=" * 50)
    
    print("\n1. Testing function imports:")
    try:
        from video_analyzer import load_video, detect_scenes, score_scene, VideoChunk
        print("   ✅ All functions imported successfully")
    except ImportError as e:
        print(f"   ❌ Import error: {e}")
        return
    
    print("\n2. Testing VideoChunk class:")
    chunk = VideoChunk(10.0, 15.0, 85.5, "test.mp4", {"test": True})
    print(f"   Created chunk: {chunk}")
    print(f"   Duration: {chunk.duration}s")
    print(f"   Metadata: {chunk.metadata}")
    print("   ✅ VideoChunk class working")
    
    print("\n3. Testing with missing files:")
    try:
        load_video("nonexistent.mp4")
        print("   ❌ Should have raised FileNotFoundError")
    except FileNotFoundError:
        print("   ✅ Correctly handles missing files")
    except Exception as e:
        print(f"   ⚠️  Unexpected error: {e}")
    
    print("\n✅ Mock tests completed!")

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--mock":
        test_without_real_files()
    else:
        test_video_analysis()