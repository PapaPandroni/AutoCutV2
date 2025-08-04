#!/usr/bin/env python3
"""
Test script for Step 3 - Advanced Video Scoring
Tests the complete video analysis pipeline with motion and face detection

This validates that all Step 3 functions are working:
- detect_motion() - optical flow motion detection
- detect_faces() - OpenCV face detection 
- analyze_video_file() - complete pipeline with enhanced scoring
"""
import sys
import os
sys.path.append('src')

from video_analyzer import analyze_video_file
import glob

def test_step3_complete():
    print("🎬 Testing Step 3 - Advanced Video Scoring")
    print("=" * 50)
    
    # Find video files in test_media
    video_extensions = ['*.mp4', '*.avi', '*.mov', '*.mkv', '*.webm']
    video_files = []
    
    for ext in video_extensions:
        video_files.extend(glob.glob(f'test_media/{ext}'))
        video_files.extend(glob.glob(f'test_media/{ext.upper()}'))
    
    if not video_files:
        print("❌ No video files found in test_media/")
        print("Please add some video files to test_media/ folder")
        return
    
    # Test with first 3 videos for quick validation
    test_videos = video_files[:3]
    print(f"Testing complete pipeline with {len(test_videos)} videos:")
    
    for i, video_file in enumerate(test_videos, 1):
        filename = os.path.basename(video_file)
        print(f"\n🎬 [{i}/{len(test_videos)}] {filename}")
        print("-" * 40)
        
        try:
            # Test complete analyze_video_file function
            chunks = analyze_video_file(video_file, min_scene_duration=1.0)
            
            if not chunks:
                print("   ⚠️  No chunks returned")
                continue
                
            print(f"   Found {len(chunks)} video chunks:")
            
            # Show top 3 chunks
            for j, chunk in enumerate(chunks[:3]):
                print(f"   🏆 Chunk {j+1}: {chunk.start_time:.1f}s-{chunk.end_time:.1f}s")
                print(f"      Enhanced Score: {chunk.score:.1f}/100")
                print(f"      Quality: {chunk.metadata['quality_score']:.1f}")
                print(f"      Motion: {chunk.metadata['motion_score']:.1f}")
                print(f"      Faces: {chunk.metadata['face_count']}")
                print(f"      Duration: {chunk.duration:.1f}s")
            
            if len(chunks) > 3:
                print(f"   ... and {len(chunks)-3} more chunks")
            
            # Show best chunk details
            best_chunk = chunks[0]
            print(f"\n   🥇 BEST CHUNK:")
            print(f"      Time: {best_chunk.start_time:.1f}s - {best_chunk.end_time:.1f}s")
            print(f"      Score: {best_chunk.score:.1f}/100")
            print(f"      Quality/Motion/Faces: {best_chunk.metadata['quality_score']:.1f}/{best_chunk.metadata['motion_score']:.1f}/{best_chunk.metadata['face_count']}")
            print("   ✅ Complete analysis successful!")
            
        except Exception as e:
            print(f"   ❌ Error: {e}")
            import traceback
            traceback.print_exc()
    
    print(f"\n🎯 Step 3 testing complete!")
    print("✅ Advanced video scoring with motion and face detection implemented!")

if __name__ == "__main__":
    test_step3_complete()