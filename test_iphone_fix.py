#!/usr/bin/env python3
"""
Test iPhone transcoding issue directly with FFmpeg to verify the root cause.
"""

import os
import subprocess
import json
from pathlib import Path

def analyze_with_ffprobe(file_path: str):
    """Analyze video file with ffprobe."""
    cmd = [
        'ffprobe', '-v', 'quiet', '-print_format', 'json',
        '-show_streams', '-select_streams', 'v:0', file_path
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    data = json.loads(result.stdout)
    
    if data.get('streams'):
        stream = data['streams'][0]
        return {
            'codec': stream.get('codec_name'),
            'profile': stream.get('profile'),
            'pixel_format': stream.get('pix_fmt'),
            'width': stream.get('width'),
            'height': stream.get('height')
        }
    return None

def test_ffmpeg_transcoding():
    """Test different FFmpeg transcoding approaches to find the issue."""
    print("=== iPhone FFmpeg Transcoding Test ===")
    
    source_file = "/media/peremil/Peremil/autocut_bugs/test_filmer_iphone/IMG_0596.mov"
    temp_dir = "temp_test"
    
    if not os.path.exists(source_file):
        print(f"ERROR: Source file not found: {source_file}")
        return
    
    os.makedirs(temp_dir, exist_ok=True)
    
    try:
        print(f"\n=== SOURCE FILE ANALYSIS ===")
        source_info = analyze_with_ffprobe(source_file)
        print(f"Original: {source_info['codec']} ({source_info['profile']}) - {source_info['pixel_format']}")
        
        # Test 1: CPU transcoding with iPhone fix parameters
        print(f"\n=== TEST 1: CPU TRANSCODING WITH IPHONE FIX ===")
        output1 = os.path.join(temp_dir, "test_cpu.mp4")
        
        cmd1 = [
            'ffmpeg', '-y', '-i', source_file,
            '-c:v', 'libx264',
            '-crf', '25',
            '-preset', 'ultrafast',
            '-profile:v', 'main',      # iPhone fix: Force Main profile
            '-pix_fmt', 'yuv420p',     # iPhone fix: Force 8-bit pixel format
            '-c:a', 'copy',
            output1
        ]
        
        print(f"Command: {' '.join(cmd1[:10])}...")
        result1 = subprocess.run(cmd1, capture_output=True, text=True)
        
        if result1.returncode == 0:
            result_info1 = analyze_with_ffprobe(output1)
            print(f"CPU Result: {result_info1['codec']} ({result_info1['profile']}) - {result_info1['pixel_format']}")
            
            success1 = result_info1['profile'] == 'Main' and 'yuv420p' in result_info1['pixel_format']
            print(f"CPU iPhone Fix: {'✅ SUCCESS' if success1 else '❌ FAILED'}")
        else:
            print(f"CPU transcoding failed: {result1.stderr}")
        
        # Test 2: Try NVIDIA GPU transcoding if available
        print(f"\n=== TEST 2: NVIDIA GPU TRANSCODING ===")
        output2 = os.path.join(temp_dir, "test_nvenc.mp4")
        
        cmd2 = [
            'ffmpeg', '-y', '-i', source_file,
            '-c:v', 'h264_nvenc',
            '-preset', 'p1',
            '-rc', 'vbr',
            '-cq', '25',
            '-profile:v', 'main',      # iPhone fix: Force Main profile
            '-pix_fmt', 'yuv420p',     # iPhone fix: Force 8-bit pixel format
            '-c:a', 'copy',
            output2
        ]
        
        print(f"Command: {' '.join(cmd2[:10])}...")
        result2 = subprocess.run(cmd2, capture_output=True, text=True)
        
        if result2.returncode == 0:
            result_info2 = analyze_with_ffprobe(output2)
            print(f"NVENC Result: {result_info2['codec']} ({result_info2['profile']}) - {result_info2['pixel_format']}")
            
            success2 = result_info2['profile'] == 'Main' and 'yuv420p' in result_info2['pixel_format']
            print(f"NVENC iPhone Fix: {'✅ SUCCESS' if success2 else '❌ FAILED'}")
        else:
            print(f"NVENC transcoding failed: {result2.stderr}")
        
        # Test 3: Try without iPhone fix parameters to see default behavior
        print(f"\n=== TEST 3: WITHOUT IPHONE FIX (DEFAULT BEHAVIOR) ===")
        output3 = os.path.join(temp_dir, "test_default.mp4")
        
        cmd3 = [
            'ffmpeg', '-y', '-i', source_file,
            '-c:v', 'libx264',
            '-crf', '25',
            '-preset', 'ultrafast',
            # NO iPhone fix parameters
            '-c:a', 'copy',
            output3
        ]
        
        print(f"Command: {' '.join(cmd3[:10])}...")
        result3 = subprocess.run(cmd3, capture_output=True, text=True)
        
        if result3.returncode == 0:
            result_info3 = analyze_with_ffprobe(output3)
            print(f"Default Result: {result_info3['codec']} ({result_info3['profile']}) - {result_info3['pixel_format']}")
            
            is_high10 = result_info3['profile'] == 'High 10' and 'yuv420p10le' in result_info3['pixel_format']
            print(f"Default produces High 10: {'✅ YES (as expected)' if is_high10 else '❌ NO'}")
        else:
            print(f"Default transcoding failed: {result3.stderr}")
        
        print(f"\n=== ANALYSIS ===")
        print(f"This test will show us if:")
        print(f"1. iPhone fix parameters work when applied directly to FFmpeg")
        print(f"2. Different hardware encoders handle the parameters differently") 
        print(f"3. AutoCut's command construction has an issue")
        
    except Exception as e:
        print(f"ERROR: {str(e)}")
    
    finally:
        # Clean up
        import shutil
        if os.path.exists(temp_dir):
            try:
                shutil.rmtree(temp_dir)
                print(f"\nCleaned up temp directory: {temp_dir}")
            except:
                print(f"\nWarning: Could not clean up temp directory: {temp_dir}")

if __name__ == "__main__":
    test_ffmpeg_transcoding()