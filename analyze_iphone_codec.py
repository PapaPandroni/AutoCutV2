#!/usr/bin/env python3
"""
Analyze iPhone video file codec using AutoCut V2's enhanced detection capabilities.
"""

import sys
import os
import json
from pathlib import Path

# Add src directory to Python path and fix imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Import directly to avoid relative import issues
import subprocess
import logging

def detect_video_codec(file_path: str):
    """Simplified version of AutoCut's codec detection for analysis."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Video file not found: {file_path}")
    
    try:
        # Use FFprobe to get comprehensive video stream information
        cmd = [
            'ffprobe', '-v', 'quiet', '-print_format', 'json',
            '-show_streams', '-show_format', '-select_streams', 'v:0', file_path
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        data = json.loads(result.stdout)
        
        if not data.get('streams'):
            raise ValueError(f"No video streams found in {file_path}")
        
        video_stream = data['streams'][0]
        format_info = data.get('format', {})
        codec_name = video_stream.get('codec_name', '').lower()
        
        # Enhanced codec detection with variants
        codec_variants = {
            'h264': ['h264', 'avc', 'avc1', 'h.264'],
            'hevc': ['hevc', 'h265', 'h.265', 'hvc1', 'hev1'],
            'vp8': ['vp8'],
            'vp9': ['vp9'], 
            'av1': ['av1'],
            'mpeg4': ['mpeg4', 'mp4v', 'xvid', 'divx'],
            'mpeg2': ['mpeg2video', 'mpeg2'],
            'mpeg1': ['mpeg1video', 'mpeg1']
        }
        
        # Determine standard codec name
        standard_codec = codec_name
        for standard, variants in codec_variants.items():
            if codec_name in variants:
                standard_codec = standard
                break
        
        # Parse frame rate with enhanced handling
        fps_str = video_stream.get('r_frame_rate', '30/1')
        if '/' in fps_str:
            try:
                num, den = map(int, fps_str.split('/'))
                fps = num / den if den != 0 else 30.0
            except (ValueError, ZeroDivisionError):
                fps = 30.0  # Fallback
        else:
            try:
                fps = float(fps_str)
            except (ValueError, TypeError):
                fps = 30.0  # Fallback
        
        # Extract container format
        container = Path(file_path).suffix.lower().lstrip('.')
        format_name = format_info.get('format_name', '').lower()
        
        # Simple compatibility scoring
        compatibility_score = 85  # Default good score
        warnings = []
        
        if standard_codec == 'hevc':
            compatibility_score = 40  # H.265 often needs transcoding
            warnings.append("H.265/HEVC codec may need transcoding for MoviePy compatibility")
        
        pixel_format = video_stream.get('pix_fmt', 'unknown')
        if 'yuv420p10' in pixel_format:
            compatibility_score -= 20
            warnings.append("10-bit encoding detected - will need transcoding to 8-bit for MoviePy")
        
        codec_info = {
            'codec': standard_codec,
            'codec_raw': codec_name,
            'is_hevc': standard_codec == 'hevc',
            'resolution': (
                int(video_stream.get('width', 0)),
                int(video_stream.get('height', 0))
            ),
            'fps': fps,
            'duration': float(video_stream.get('duration', format_info.get('duration', 0))),
            'pixel_format': pixel_format,
            'container': container,
            'format_name': format_name,
            'compatibility_score': compatibility_score,
            'warnings': warnings,
            
            # Additional technical details
            'bitrate': int(video_stream.get('bit_rate', 0)),
            'profile': video_stream.get('profile', 'unknown'),
            'level': video_stream.get('level', 'unknown'),
            'color_space': video_stream.get('color_space', 'unknown'),
            'has_audio': len([s for s in data.get('streams', []) if s.get('codec_type') == 'audio']) > 0
        }
        
        return codec_info
        
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"FFprobe failed for {file_path}: {e.stderr}")
    except (json.JSONDecodeError, KeyError, ValueError) as e:
        raise RuntimeError(f"Failed to parse video information for {file_path}: {str(e)}")

def analyze_iphone_video(file_path: str):
    """Analyze iPhone video file codec and format details."""
    print(f"=== AutoCut V2 Codec Analysis ===")
    print(f"Analyzing: {file_path}")
    print(f"File exists: {os.path.exists(file_path)}")
    
    if not os.path.exists(file_path):
        print(f"ERROR: File not found: {file_path}")
        return None
    
    try:
        # Use AutoCut's enhanced codec detection
        codec_info = detect_video_codec(file_path)
        
        print(f"\n=== CODEC ANALYSIS RESULTS ===")
        
        # Basic information
        print(f"📱 File: {Path(file_path).name}")
        print(f"📦 Container: {codec_info['container']} ({codec_info['format_name']})")
        print(f"🎬 Codec: {codec_info['codec']} (raw: {codec_info['codec_raw']})")
        print(f"🔍 H.265/HEVC: {codec_info['is_hevc']}")
        
        # Video specifications
        print(f"\n=== VIDEO SPECIFICATIONS ===")
        print(f"📐 Resolution: {codec_info['resolution'][0]}x{codec_info['resolution'][1]}")
        print(f"🎥 Frame Rate: {codec_info['fps']:.2f} fps")
        print(f"⏱️  Duration: {codec_info['duration']:.2f} seconds")
        print(f"🎨 Pixel Format: {codec_info['pixel_format']}")
        
        # Technical details
        print(f"\n=== TECHNICAL DETAILS ===")
        print(f"📊 Profile: {codec_info['profile']}")
        print(f"📈 Level: {codec_info['level']}")
        print(f"🌈 Color Space: {codec_info['color_space']}")
        print(f"📡 Bitrate: {codec_info['bitrate']} bps")
        print(f"🔊 Has Audio: {codec_info['has_audio']}")
        
        # Compatibility assessment
        print(f"\n=== AUTOCUT COMPATIBILITY ===")
        print(f"🎯 Compatibility Score: {codec_info['compatibility_score']}/100")
        
        if codec_info['warnings']:
            print(f"⚠️  Warnings:")
            for warning in codec_info['warnings']:
                print(f"   - {warning}")
        else:
            print(f"✅ No compatibility warnings")
        
        # iPhone-specific analysis
        print(f"\n=== IPHONE-SPECIFIC ANALYSIS ===")
        
        # Check for 10-bit encoding (common iPhone issue)
        is_10bit = 'yuv420p10' in codec_info['pixel_format']
        print(f"📱 10-bit encoding: {is_10bit}")
        
        if is_10bit:
            print(f"   ⚠️  This is 10-bit H.265 (iPhone 12+ HDR recording)")
            print(f"   🔧 AutoCut will force 8-bit H.264 transcoding for MoviePy compatibility")
            print(f"   ⚡ Hardware acceleration will be used for transcoding")
        
        # Check profile
        if codec_info['profile'] == 'Main 10':
            print(f"📊 Profile 'Main 10' detected - typical for iPhone H.265 HDR")
        
        # Predict processing approach
        print(f"\n=== PREDICTED AUTOCUT PROCESSING ===")
        if codec_info['is_hevc']:
            print(f"1. 🔍 H.265 detected - will test MoviePy compatibility")
            print(f"2. 🔄 Likely needs transcoding (iPhone H.265 usually incompatible)")
            print(f"3. ⚡ Hardware-accelerated transcoding will be applied")
            print(f"4. 🎯 Output: 8-bit H.264 Main profile for MoviePy")
            print(f"5. ✅ Full AutoCut pipeline processing")
        else:
            print(f"✅ Non-H.265 codec - direct processing without transcoding")
        
        return codec_info
        
    except Exception as e:
        print(f"ERROR analyzing codec: {str(e)}")
        return None

if __name__ == "__main__":
    # iPhone video file path
    iphone_file = "/media/peremil/Peremil/autocut_bugs/test_filmer_iphone/IMG_0596.mov"
    
    result = analyze_iphone_video(iphone_file)
    
    if result:
        print(f"\n=== SUMMARY ===")
        if result['is_hevc']:
            print(f"📱 iPhone H.265 file detected")
            print(f"🔧 AutoCut V2 has specific iPhone compatibility fixes")
            print(f"✅ This file should process successfully with hardware acceleration")
        else:
            print(f"📹 Non-H.265 file - should process without issues")
            
        print(f"🎯 Overall Assessment: {'COMPATIBLE' if result['compatibility_score'] > 50 else 'NEEDS_TRANSCODING'}")