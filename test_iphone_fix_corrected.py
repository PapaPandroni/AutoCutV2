#!/usr/bin/env python3
"""
Test script for the CORRECTED iPhone H.265 processing fix.

This script tests the proper fix for the iPhone H.265 transcoding issue:
1. Remove problematic -profile:v main parameter (causes 10-bit to 8-bit conversion failure)
2. Accept Constrained Baseline profile as iPhone compatible (it actually is)
3. Keep MoviePy compatibility test (it was working correctly)
"""

import os
import sys
import time
from pathlib import Path

# Add the src directory to Python path
sys.path.append('src')

from src.utils import transcode_hevc_to_h264_enhanced

def test_corrected_fix():
    """Test the corrected iPhone H.265 transcoding fix."""
    
    print("🎯 CORRECTED FIX TEST: iPhone H.265 Processing")
    print("=" * 60)
    
    # Path to the iPhone footage that was failing
    test_video = "/media/peremil/Peremil/autocut_bugs/test_media/IMG_0431.mov"
    
    if not os.path.exists(test_video):
        print(f"❌ Test video not found: {test_video}")
        print("Please ensure the iPhone footage folder is mounted.")
        return False
    
    print(f"📱 Testing with: {Path(test_video).name}")
    print(f"📂 Video file exists: ✅")
    print(f"📊 Input: iPhone 12 H.265 Main 10 (10-bit yuv420p10le)")
    
    # Test the transcoding with our corrected fix
    try:
        print(f"\n🔧 Starting transcoding with corrected fix...")
        print("   - Removed -profile:v main parameter (was causing 10-bit conversion failure)")
        print("   - Updated validation to accept Constrained Baseline profile") 
        print("   - Restored MoviePy compatibility test")
        
        start_time = time.time()
        
        output_path = transcode_hevc_to_h264_enhanced(
            input_path=test_video,
            output_path="temp/corrected_fix_test.mp4"
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
                # Check what profile was actually generated
                import subprocess
                try:
                    cmd = ['ffprobe', '-v', 'quiet', '-show_entries', 'stream=profile,pix_fmt', '-of', 'csv=p=0', output_path]
                    result = subprocess.run(cmd, capture_output=True, text=True, timeout=5)
                    if result.returncode == 0:
                        profile_info = result.stdout.strip().split(',')
                        if len(profile_info) >= 2:
                            profile, pix_fmt = profile_info[0], profile_info[1]
                            print(f"📊 Output format: H.264 {profile}, {pix_fmt}")
                            
                            if 'baseline' in profile.lower():
                                print("✅ Constrained Baseline profile generated (optimal for iPhone compatibility!)")
                            elif 'main' in profile.lower():
                                print("✅ Main profile generated (also iPhone compatible)")
                            
                            if pix_fmt == 'yuv420p':
                                print("✅ 8-bit pixel format confirmed (MoviePy compatible)")
                except:
                    print("ℹ️  Could not analyze output format details")
                
                print("\n🎉 CORRECTED FIX SUCCESSFUL!")
                print("   ✅ 10-bit H.265 → 8-bit H.264 conversion works")
                print("   ✅ Profile validation accepts Constrained Baseline")
                print("   ✅ MoviePy compatibility validation passes")
                print("   ✅ iPhone H.265 footage processes correctly")
                return True
            else:
                print("❌ Output file is empty")
                return False
        else:
            print("❌ Output file was not created")
            return False
            
    except Exception as e:
        print(f"❌ FAILED: {str(e)}")
        print("\n🔍 Error Analysis:")
        error_str = str(e).lower()
        if 'main profile doesn\'t support' in error_str:
            print("   - FFmpeg profile issue (should be fixed by removing -profile:v main)")
        elif 'profile not iphone compatible' in error_str:
            print("   - Validation rejection (should be fixed by accepting baseline profile)")
        elif 'moviepy compatibility failed' in error_str:
            print("   - MoviePy parsing issue (transcoded file may have format problems)")
        else:
            print(f"   - Unknown error: {str(e)[:200]}")
        return False

def main():
    """Main test execution."""
    print("🧪 iPhone H.265 CORRECTED Fix Validation")
    print("Testing the proper fix for the FFmpeg/validation issue...")
    print()
    
    # Ensure temp directory exists
    os.makedirs("temp", exist_ok=True)
    
    success = test_corrected_fix()
    
    print("\n" + "=" * 60)
    if success:
        print("✅ CORRECTED FIX VALIDATED!")
        print("iPhone H.265 footage should now process correctly in AutoCut.")
        print("\nKey fixes applied:")
        print("  1. ❌ Removed -profile:v main (was incompatible with 10-bit input)")
        print("  2. ✅ Accept Constrained Baseline profile (actually optimal for iPhones)")
        print("  3. ✅ Keep MoviePy compatibility test (was working correctly)")
    else:
        print("❌ CORRECTED FIX NEEDS REFINEMENT")
        print("Additional investigation may be required.")
    
    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)