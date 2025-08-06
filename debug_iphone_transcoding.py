#!/usr/bin/env python3
"""
Debug iPhone transcoding to identify why iPhone fix parameters are not working.
"""

import sys
import os
import shutil
from pathlib import Path

# Add src directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from utils import transcode_hevc_to_h264, detect_video_codec

def debug_iphone_transcoding():
    """Debug the iPhone transcoding process step by step."""
    print("=== iPhone Transcoding Debug ===")
    
    # Use one of the iPhone files for testing
    source_file = "/media/peremil/Peremil/autocut_bugs/test_filmer_iphone/IMG_0596.mov"
    temp_dir = "temp_debug"
    
    if not os.path.exists(source_file):
        print(f"ERROR: Source file not found: {source_file}")
        return
    
    # Create temp directory
    os.makedirs(temp_dir, exist_ok=True)
    
    try:
        print(f"\n=== STEP 1: Source File Analysis ===")
        codec_info = detect_video_codec(source_file)
        print(f"Source codec: {codec_info['codec']} ({codec_info['profile']})")
        print(f"Source pixel format: {codec_info['pixel_format']}")
        print(f"Source is H.265: {codec_info['is_hevc']}")
        
        if not codec_info['is_hevc']:
            print("File is not H.265, no transcoding needed")
            return
        
        print(f"\n=== STEP 2: Transcoding Test ===")
        
        def progress_callback(message, progress):
            print(f"Progress: {message} ({progress*100:.1f}%)")
        
        # Test transcoding with detailed output
        output_path = os.path.join(temp_dir, "test_transcoded.mp4")
        
        print(f"Input: {Path(source_file).name}")
        print(f"Output: {output_path}")
        print(f"\nStarting transcoding...")
        
        transcoded_path = transcode_hevc_to_h264(
            source_file, 
            output_path,
            progress_callback
        )
        
        print(f"\n=== STEP 3: Result Analysis ===")
        if os.path.exists(transcoded_path):
            result_codec_info = detect_video_codec(transcoded_path)
            
            print(f"\n📊 TRANSCODING RESULTS:")
            print(f"Original: {codec_info['codec']} ({codec_info['profile']}) - {codec_info['pixel_format']}")
            print(f"Result:   {result_codec_info['codec']} ({result_codec_info['profile']}) - {result_codec_info['pixel_format']}")
            
            # Check if iPhone fix worked
            expected_profile = "Main"
            expected_pixel = "yuv420p"
            actual_profile = result_codec_info['profile']
            actual_pixel = result_codec_info['pixel_format']
            
            print(f"\n🎯 IPHONE FIX VERIFICATION:")
            print(f"Profile: Expected '{expected_profile}' → Got '{actual_profile}' {'✅' if actual_profile == expected_profile else '❌'}")
            print(f"Pixel:   Expected '{expected_pixel}' → Got '{actual_pixel}' {'✅' if expected_pixel in actual_pixel else '❌'}")
            
            if actual_profile != expected_profile or expected_pixel not in actual_pixel:
                print(f"\n🚨 IPHONE FIX FAILED!")
                print(f"The transcoded file still has:")
                print(f"  - Profile: {actual_profile} (should be Main)")
                print(f"  - Pixel:   {actual_pixel} (should contain yuv420p)")
                print(f"\nThis explains why MoviePy cannot parse the transcoded files.")
                
                # Try to identify the cause
                print(f"\n🔍 TROUBLESHOOTING:")
                print(f"1. FFmpeg parameters may not be applied correctly")
                print(f"2. Hardware encoder may not support profile override")
                print(f"3. Different transcoding path may be used")
                
            else:
                print(f"\n✅ IPHONE FIX WORKING!")
                print(f"Transcoded file should be compatible with MoviePy")
        else:
            print(f"ERROR: Transcoded file not created")
            
    except Exception as e:
        print(f"ERROR during transcoding: {str(e)}")
    
    finally:
        # Clean up
        if os.path.exists(temp_dir):
            try:
                shutil.rmtree(temp_dir)
                print(f"\nCleaned up temp directory: {temp_dir}")
            except:
                print(f"\nWarning: Could not clean up temp directory: {temp_dir}")

if __name__ == "__main__":
    debug_iphone_transcoding()