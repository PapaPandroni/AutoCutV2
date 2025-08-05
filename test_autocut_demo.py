#!/usr/bin/env python3
"""
AutoCut Demo Test Script

Quick way to test AutoCut with all your media files.
Creates real beat-synced videos you can watch!

Usage:
  python3 test_autocut_demo.py                    # Use all videos + random music
  python3 test_autocut_demo.py --audio song.mp3   # Specify music
  python3 test_autocut_demo.py --videos 5         # Limit to 5 videos
  python3 test_autocut_demo.py --pattern dramatic # Use specific pattern
"""

import os
import sys
import glob
import argparse
import time
from pathlib import Path

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.clip_assembler import assemble_clips

def main():
    parser = argparse.ArgumentParser(description='AutoCut Demo - Create beat-synced videos')
    parser.add_argument('--audio', help='Specific audio file to use')
    parser.add_argument('--videos', type=int, help='Number of videos to use (default: all)')
    parser.add_argument('--pattern', choices=['energetic', 'balanced', 'dramatic', 'buildup'], 
                       default='balanced', help='Cutting pattern to use')
    parser.add_argument('--output', help='Output filename (default: auto-generated)')
    
    args = parser.parse_args()
    
    print("🎬 AutoCut Demo Test")
    print("=" * 50)
    
    # Find all video files
    video_files = glob.glob('test_media/*.mp4')
    if not video_files:
        print("❌ No video files found in test_media/")
        return False
    
    # Limit videos if requested
    if args.videos:
        video_files = video_files[:args.videos]
    
    print(f"📁 Found {len(video_files)} video files:")
    for i, vf in enumerate(video_files, 1):
        name = Path(vf).name
        print(f"   {i:2d}. {name}")
    
    # Find audio file
    if args.audio:
        if not os.path.exists(args.audio):
            print(f"❌ Audio file not found: {args.audio}")
            return False
        audio_file = args.audio
    else:
        # Find any supported audio file
        audio_extensions = ['*.mp3', '*.wav', '*.m4a', '*.flac', '*.aac', '*.ogg']
        audio_files = []
        for ext in audio_extensions:
            audio_files.extend(glob.glob(f'test_media/{ext}'))
        
        if not audio_files:
            print("❌ No audio files found in test_media/")
            print("   Supported formats: MP3, WAV, M4A, FLAC, AAC, OGG")
            return False
        audio_file = audio_files[0]  # Use first one
    
    print(f"🎵 Using audio: {Path(audio_file).name}")
    print(f"🎯 Pattern: {args.pattern}")
    
    # Generate output filename
    if args.output:
        output_file = args.output
    else:
        timestamp = int(time.time())
        output_file = f"output/autocut_demo_{args.pattern}_{timestamp}.mp4"
    
    print(f"📄 Output: {output_file}")
    
    # Create output directory
    os.makedirs('output', exist_ok=True)
    
    # Progress callback
    def progress_callback(step, progress):
        bar_length = 30
        filled = int(bar_length * progress)
        bar = '█' * filled + '░' * (bar_length - filled)
        print(f"\r  [{bar}] {progress*100:5.1f}% {step}", end='', flush=True)
    
    print(f"\n🚀 Creating AutoCut video...")
    print("   This may take a few minutes for rendering...")
    
    try:
        start_time = time.time()
        
        result_path = assemble_clips(
            video_files=video_files,
            audio_file=audio_file,
            output_path=output_file,
            pattern=args.pattern,
            progress_callback=progress_callback
        )
        
        elapsed = time.time() - start_time
        print(f"\n\n🎉 SUCCESS!")
        print(f"   Time: {elapsed:.1f} seconds")
        print(f"   Video: {result_path}")
        
        # Show file info
        if os.path.exists(result_path):
            file_size = os.path.getsize(result_path) / (1024 * 1024)  # MB
            print(f"   Size: {file_size:.1f} MB")
            
            # Check for timeline JSON
            timeline_json = result_path.replace('.mp4', '_timeline.json')
            if os.path.exists(timeline_json):
                print(f"   Debug: {timeline_json}")
        
        print(f"\n🎬 Open your video player and watch: {result_path}")
        print("   You should see beat-synced cuts that match the music!")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)