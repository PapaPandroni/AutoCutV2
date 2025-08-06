#!/usr/bin/env python3
"""
Test script for the surgical iPhone H.265 processing fix.

This script tests the specific fix for the MoviePy validation issue
that was preventing iPhone H.265 footage from being processed correctly.
"""

import os
import sys
import time
from pathlib import Path

# Add the src directory to Python path
sys.path.append('src')

from src.utils import transcode_hevc_to_h264_enhanced

def test_surgical_fix():
    """Test the surgical fix for iPhone H.265 transcoding validation issue."""
    
    print("🎯 SURGICAL FIX TEST: iPhone H.265 Processing")
    print("=" * 60)
    
    # Path to the iPhone footage that was failing
    test_video = "/media/peremil/Peremil/autocut_bugs/test_media/IMG_0431.mov"
    
    if not os.path.exists(test_video):
        print(f"❌ Test video not found: {test_video}")
        print("Please ensure the iPhone footage folder is mounted.")
        return False
    
    print(f"📱 Testing with: {Path(test_video).name}")
    print(f"📂 Video file exists: ✅")
    
    # Test the transcoding with our surgical fix
    try:
        print(f"\n🔧 Starting transcoding with surgical fix...")
        start_time = time.time()
        
        output_path = transcode_hevc_to_h264_enhanced(
            input_path=test_video,
            output_path="temp/surgical_fix_test.mp4"
        )
        
        end_time = time.time()
        processing_time = end_time - start_time
        
        print(f"\n✅ SUCCESS! Transcoding completed in {processing_time:.1f}s")
        print(f"📄 Output file: {output_path}")
        
        # Verify output file
        if os.path.exists(output_path):
            file_size = os.path.getsize(output_path)
            print(f"📊 Output file size: {file_size:,} bytes ({file_size/1024/1024:.1f} MB)")
            
            if file_size > 0:
                print("🎉 SURGICAL FIX SUCCESSFUL!")
                print("   - No more MoviePy validation false negatives")
                print("   - iPhone H.265 footage processes correctly")
                return True
            else:
                print("❌ Output file is empty")
                return False
        else:
            print("❌ Output file was not created")
            return False
            
    except Exception as e:
        print(f"❌ FAILED: {str(e)}")
        return False

def main():
    """Main test execution."""
    print("🧪 iPhone H.265 Surgical Fix Validation")
    print("Testing the specific MoviePy validation fix...")
    print()
    
    # Ensure temp directory exists
    os.makedirs("temp", exist_ok=True)
    
    success = test_surgical_fix()
    
    print("\n" + "=" * 60)
    if success:
        print("✅ SURGICAL FIX VALIDATED!")
        print("iPhone H.265 footage should now process correctly in AutoCut.")
    else:
        print("❌ SURGICAL FIX NEEDS REFINEMENT")
        print("Additional investigation may be required.")
    
    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)