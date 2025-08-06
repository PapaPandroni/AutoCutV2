#!/usr/bin/env python3
"""
Analyze multiple iPhone files to understand codec patterns.
"""

import os
import json
import subprocess
from pathlib import Path

def detect_video_codec_simple(file_path: str):
    """Simplified codec detection for analysis."""
    try:
        cmd = [
            'ffprobe', '-v', 'quiet', '-print_format', 'json',
            '-show_streams', '-show_format', '-select_streams', 'v:0', file_path
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        data = json.loads(result.stdout)
        
        if not data.get('streams'):
            return None
        
        video_stream = data['streams'][0]
        
        return {
            'filename': Path(file_path).name,
            'codec': video_stream.get('codec_name', '').lower(),
            'profile': video_stream.get('profile', 'unknown'),
            'pixel_format': video_stream.get('pix_fmt', 'unknown'),
            'resolution': f"{video_stream.get('width', 0)}x{video_stream.get('height', 0)}",
            'duration': float(video_stream.get('duration', 0)),
            'file_size_mb': os.path.getsize(file_path) / (1024 * 1024)
        }
    except Exception as e:
        return {'filename': Path(file_path).name, 'error': str(e)}

def analyze_iphone_batch():
    """Analyze all iPhone files in the source directory."""
    source_dir = "/media/peremil/Peremil/autocut_bugs/test_filmer_iphone"
    
    print("=== iPhone File Batch Analysis ===")
    print(f"Source directory: {source_dir}")
    
    if not os.path.exists(source_dir):
        print(f"ERROR: Directory not found: {source_dir}")
        return
    
    mov_files = [f for f in os.listdir(source_dir) if f.endswith('.mov')]
    mov_files.sort()
    
    print(f"Found {len(mov_files)} iPhone video files:")
    
    results = []
    for filename in mov_files:
        file_path = os.path.join(source_dir, filename)
        result = detect_video_codec_simple(file_path)
        results.append(result)
        
        if 'error' in result:
            print(f"❌ {filename}: ERROR - {result['error']}")
        else:
            print(f"📱 {filename}:")
            print(f"   Codec: {result['codec']} ({result['profile']})")
            print(f"   Pixel: {result['pixel_format']}")
            print(f"   Size: {result['resolution']}, {result['duration']:.1f}s, {result['file_size_mb']:.1f}MB")
    
    # Pattern analysis
    print(f"\n=== PATTERN ANALYSIS ===")
    
    successful_results = [r for r in results if 'error' not in r]
    if not successful_results:
        print("No successful codec analysis results")
        return
    
    codecs = set(r['codec'] for r in successful_results)
    profiles = set(r['profile'] for r in successful_results)
    pixel_formats = set(r['pixel_format'] for r in successful_results)
    
    print(f"Codecs found: {codecs}")
    print(f"Profiles found: {profiles}")
    print(f"Pixel formats found: {pixel_formats}")
    
    # Check for consistency
    if len(codecs) == 1 and len(profiles) == 1 and len(pixel_formats) == 1:
        codec = list(codecs)[0]
        profile = list(profiles)[0]
        pixel_format = list(pixel_formats)[0]
        
        print(f"\n✅ CONSISTENT PATTERN DETECTED:")
        print(f"   All files: {codec} ({profile}) with {pixel_format}")
        
        # iPhone compatibility assessment
        is_hevc = codec in ['hevc', 'h265']
        is_10bit = 'yuv420p10' in pixel_format
        is_main10 = profile == 'Main 10'
        
        print(f"\n📱 IPHONE CHARACTERISTICS:")
        print(f"   H.265/HEVC: {is_hevc}")
        print(f"   10-bit encoding: {is_10bit}")
        print(f"   Main 10 profile: {is_main10}")
        
        if is_hevc and is_10bit and is_main10:
            print(f"\n🎯 CONCLUSION: Typical iPhone 12+ HDR recording characteristics")
            print(f"   - Requires transcoding to H.264 8-bit for MoviePy")
            print(f"   - AutoCut's iPhone fix should handle this")
            print(f"   - ERROR suggests transcoding parameters not working correctly")
        
    else:
        print(f"\n⚠️  INCONSISTENT PATTERNS - mixed file types detected")

if __name__ == "__main__":
    analyze_iphone_batch()