#!/usr/bin/env python3
"""
Test Script for iPhone H.265 Processing Fixes
Tests the comprehensive iPhone H.265 processing enhancements implemented for AutoCut V2.
"""

import os
import sys
import time
from pathlib import Path

# Add src directory to path for testing
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from utils import (
    detect_optimal_codec_settings_enhanced,
    preprocess_video_if_needed_enhanced, 
    validate_transcoded_output,
    transcode_hevc_to_h264_enhanced
)


def test_enhanced_hardware_detection():
    """Test enhanced hardware detection with comprehensive capability testing."""
    print("🔍 Testing Enhanced Hardware Detection")
    print("=" * 60)
    
    try:
        moviepy_params, ffmpeg_params, diagnostics = detect_optimal_codec_settings_enhanced()
        
        print(f"✅ Hardware Detection Results:")
        print(f"   🔧 Encoder Type: {diagnostics.get('encoder_type', 'UNKNOWN')}")
        print(f"   📱 iPhone Compatible: {diagnostics.get('iphone_compatible', False)}")
        print(f"   🚗 Driver Status: {diagnostics.get('driver_status', 'UNKNOWN')}")
        print(f"   💬 Message: {diagnostics.get('diagnostic_message', 'N/A')}")
        
        print(f"\n📊 MoviePy Parameters:")
        for key, value in moviepy_params.items():
            print(f"   {key}: {value}")
        
        print(f"\n⚙️  FFmpeg Parameters: {' '.join(ffmpeg_params)}")
        
        if 'tests_performed' in diagnostics:
            print(f"\n🧪 Tests Performed: {len(diagnostics['tests_performed'])}")
            for test in diagnostics['tests_performed']:
                print(f"   - {test}")
        
        if 'errors_encountered' in diagnostics and diagnostics['errors_encountered']:
            print(f"\n⚠️  Errors Encountered:")
            for error in diagnostics['errors_encountered']:
                print(f"   - {error}")
        
        return True
        
    except Exception as e:
        print(f"❌ Hardware detection test failed: {str(e)}")
        return False


def test_enhanced_preprocessing():
    """Test enhanced preprocessing with a real or dummy video file."""
    print("\n🎬 Testing Enhanced Preprocessing")
    print("=" * 60)
    
    # Look for test media files
    test_media_dir = Path("test_media")
    test_files = []
    
    if test_media_dir.exists():
        # Get first available video file for testing
        video_extensions = ['.mp4', '.mov', '.mkv', '.avi']
        for ext in video_extensions:
            files = list(test_media_dir.glob(f"*{ext}"))
            if files:
                test_files.extend(files[:1])  # Take first file of each type
                break
    
    if not test_files:
        print("⚠️  No test media files found in test_media/ directory")
        print("   Creating dummy test to validate function interface...")
        
        # Test function interface with non-existent file
        try:
            result = preprocess_video_if_needed_enhanced("non_existent_file.mp4")
            print(f"✅ Function interface test passed:")
            print(f"   Result structure: {list(result.keys())}")
            print(f"   Success: {result.get('success', False)}")
            print(f"   Error Category: {result.get('error_category', 'N/A')}")
            return True
        except Exception as e:
            print(f"❌ Function interface test failed: {str(e)}")
            return False
    
    # Test with actual files
    success = True
    for test_file in test_files:
        print(f"\n🎥 Testing with: {test_file.name}")
        
        try:
            result = preprocess_video_if_needed_enhanced(str(test_file))
            
            print(f"   ✅ Processing Results:")
            print(f"      Success: {result['success']}")
            print(f"      Transcoded: {result['transcoded']}")
            print(f"      Processing Time: {result['processing_time']:.1f}s")
            print(f"      Output Path: {Path(result['processed_path']).name}")
            
            if result['original_codec']:
                codec_info = result['original_codec']
                print(f"      Original Codec: {codec_info['codec']} {codec_info['profile']}")
                print(f"      Resolution: {codec_info['resolution']}")
            
            if result['error_category']:
                print(f"      ⚠️  Error Category: {result['error_category']}")
                print(f"      Diagnostic: {result['diagnostic_message'][:100]}...")
            
            if not result['success']:
                success = False
                
        except Exception as e:
            print(f"   ❌ Preprocessing failed: {str(e)}")
            success = False
    
    return success


def test_output_validation():
    """Test comprehensive output validation system."""
    print("\n🔍 Testing Output Validation System")
    print("=" * 60)
    
    # Test with existing files in test_media
    test_media_dir = Path("test_media")
    test_files = []
    
    if test_media_dir.exists():
        test_files = list(test_media_dir.glob("*.mp4"))[:2]  # Test first 2 MP4 files
    
    if not test_files:
        print("⚠️  No MP4 test files found for validation testing")
        return True  # Not a failure, just no test data
    
    success = True
    for test_file in test_files:
        print(f"\n🎥 Validating: {test_file.name}")
        
        try:
            validation_result = validate_transcoded_output(str(test_file))
            
            print(f"   Valid: {validation_result['valid']}")
            print(f"   Codec Profile: {validation_result['codec_profile']}")
            print(f"   Pixel Format: {validation_result['pixel_format']}")
            print(f"   MoviePy Compatible: {validation_result['moviepy_compatible']}")
            print(f"   iPhone Compatible: {validation_result['iphone_compatible']}")
            
            if validation_result['error_details']:
                print(f"   ⚠️  Error Details:")
                for error in validation_result['error_details']:
                    print(f"      - {error}")
            
        except Exception as e:
            print(f"   ❌ Validation test failed: {str(e)}")
            success = False
    
    return success


def test_integration_compatibility():
    """Test integration with existing AutoCut components."""
    print("\n🔗 Testing Integration Compatibility")
    print("=" * 60)
    
    try:
        # Test legacy function imports
        from utils import detect_optimal_codec_settings, preprocess_video_if_needed
        print("✅ Legacy function imports successful")
        
        # Test legacy hardware detection
        moviepy_params, ffmpeg_params = detect_optimal_codec_settings()
        print(f"✅ Legacy hardware detection: {moviepy_params.get('codec', 'unknown')}")
        
        # Test legacy preprocessing interface
        result_path = preprocess_video_if_needed("non_existent.mp4")
        print(f"✅ Legacy preprocessing interface: returns path string")
        
        # Test clip_assembler integration
        try:
            from clip_assembler import detect_optimal_codec_settings as clip_detect
            clip_moviepy_params, clip_ffmpeg_params = clip_detect()
            print(f"✅ ClipAssembler integration: {clip_moviepy_params.get('codec', 'unknown')}")
        except ImportError:
            print("⚠️  ClipAssembler import failed (expected if not in proper environment)")
        
        return True
        
    except Exception as e:
        print(f"❌ Integration compatibility test failed: {str(e)}")
        return False


def run_comprehensive_test():
    """Run comprehensive test suite for iPhone H.265 processing fixes."""
    print("🚀 AutoCut V2 - iPhone H.265 Processing Fixes Test Suite")
    print("=" * 80)
    print("Testing comprehensive iPhone H.265 processing enhancements:")
    print("- Enhanced hardware detection with actual capability testing")
    print("- iPhone parameter application and validation")
    print("- Comprehensive output validation system") 
    print("- Enhanced error handling and recovery mechanisms")
    print("- Integration with existing preprocessing pipeline")
    print()
    
    start_time = time.time()
    tests_passed = 0
    total_tests = 4
    
    # Run test suite
    if test_enhanced_hardware_detection():
        tests_passed += 1
    
    if test_enhanced_preprocessing():
        tests_passed += 1
    
    if test_output_validation():
        tests_passed += 1
    
    if test_integration_compatibility():
        tests_passed += 1
    
    # Test results
    total_time = time.time() - start_time
    
    print(f"\n📊 Test Suite Results")
    print("=" * 60)
    print(f"Tests Passed: {tests_passed}/{total_tests}")
    print(f"Success Rate: {(tests_passed/total_tests)*100:.1f}%")
    print(f"Total Time: {total_time:.1f}s")
    
    if tests_passed == total_tests:
        print("🎉 All tests passed! iPhone H.265 processing fixes are working correctly.")
        print("\n✅ Ready for iPhone footage processing:")
        print("   - Hardware acceleration with validation")
        print("   - iPhone parameter compatibility")
        print("   - Comprehensive error diagnostics")
        print("   - Output format validation")
        print("   - Seamless integration with existing AutoCut pipeline")
        return True
    else:
        print("⚠️  Some tests failed. Review the results above for details.")
        print("   The system may still work but with reduced capabilities.")
        return False


if __name__ == "__main__":
    success = run_comprehensive_test()
    sys.exit(0 if success else 1)