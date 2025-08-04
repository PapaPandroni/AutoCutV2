#!/usr/bin/env python3
"""
Test script for real audio file analysis
"""
import sys
import os
sys.path.append('src')

from audio_analyzer import analyze_audio
import glob

def test_real_audio():
    print("🎵 Testing with Real Audio Files")
    print("=" * 50)
    
    # Find audio files in test_media
    audio_extensions = ['*.mp3', '*.wav', '*.m4a', '*.flac']
    audio_files = []
    
    for ext in audio_extensions:
        audio_files.extend(glob.glob(f'test_media/{ext}'))
        audio_files.extend(glob.glob(f'test_media/{ext.upper()}'))
    
    if not audio_files:
        print("❌ No audio files found in test_media/")
        print("Please add some music files to test_media/ folder")
        print("Supported formats: MP3, WAV, M4A, FLAC")
        return
    
    print(f"Found {len(audio_files)} audio file(s):")
    for i, file in enumerate(audio_files, 1):
        print(f"  {i}. {os.path.basename(file)}")
    
    print("\nAnalyzing each file:")
    print("-" * 30)
    
    for audio_file in audio_files:
        filename = os.path.basename(audio_file)
        print(f"\n🎵 {filename}")
        
        try:
            result = analyze_audio(audio_file)
            
            print(f"   Duration: {result['duration']:.1f} seconds")
            print(f"   BPM: {result['bpm']:.1f}")
            print(f"   Beats detected: {len(result['beats'])}")
            print(f"   First few beats: {[float(b) for b in result['beats'][:5]]}")
            print(f"   Allowed clip durations: {[f'{float(d):.2f}s' for d in result['allowed_durations']]}")
            print("   ✅ Analysis successful!")
            
        except Exception as e:
            print(f"   ❌ Error: {e}")
    
    print(f"\n🎯 Real audio analysis complete!")

if __name__ == "__main__":
    test_real_audio()