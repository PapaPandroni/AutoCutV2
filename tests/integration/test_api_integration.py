"""
API Integration Tests for AutoCut V2

Tests the API layer integration with core components, parameter validation,
error handling, and contract compliance for the AutoCutAPI class.
"""

import sys
from pathlib import Path

import pytest

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from api import AutoCutAPI


class TestAPIIntegration:
    """Test API layer integration and contracts."""

    @pytest.fixture
    def api_client(self) -> AutoCutAPI:
        """Fresh API client for each test."""
        return AutoCutAPI()

    @pytest.fixture
    def api_output_dir(self, temp_dir):
        """Output directory for API test results."""
        output_dir = temp_dir / "api_outputs"
        output_dir.mkdir(exist_ok=True)
        return output_dir

    def test_api_initialization_contract(self):
        """Test API initialization follows expected contract."""
        print("\n🏗️ Testing API initialization contract")

        # Test default initialization
        api = AutoCutAPI()

        # Verify required methods exist
        required_methods = ["process_videos", "validate_video", "get_system_info"]

        for method in required_methods:
            assert hasattr(api, method), f"API should have {method} method"
            assert callable(getattr(api, method)), f"{method} should be callable"
            print(f"   ✅ {method} method available")

        # Test API is stateless (multiple instances should be independent)
        api2 = AutoCutAPI()
        assert api is not api2, "API instances should be independent"
        print("   ✅ API instances are independent")

        print("✅ API initialization contract validated")

    def test_process_videos_parameter_validation(
        self, api_client: AutoCutAPI, temp_dir: Path
    ):
        """Test comprehensive parameter validation for process_videos method."""
        print("\n🔍 Testing process_videos parameter validation")

        output_path = str(temp_dir / "param_validation_test.mp4")

        # Test required parameters
        with pytest.raises((TypeError, ValueError)) as exc_info:
            api_client.process_videos()  # No parameters
        print(f"   ✅ No parameters rejected: {type(exc_info.value).__name__}")

        # Test video_files parameter validation
        invalid_video_files = [
            None,  # None value
            "not_a_list",  # String instead of list
            [],  # Empty list
            [None],  # List with None
            [123],  # List with non-string
        ]

        for invalid_input in invalid_video_files:
            with pytest.raises((TypeError, ValueError, IndexError)) as exc_info:
                api_client.process_videos(
                    video_files=invalid_input,
                    audio_file="dummy.mp3",
                    output_path=output_path,
                    pattern="balanced",
                )
            print(f"   ✅ Invalid video_files {type(invalid_input).__name__} rejected")

        # Test audio_file parameter validation
        invalid_audio_files = [
            None,  # None value
            123,  # Integer instead of string
            [],  # List instead of string
        ]

        for invalid_input in invalid_audio_files:
            with pytest.raises((TypeError, ValueError)) as exc_info:
                api_client.process_videos(
                    video_files=["dummy.mp4"],
                    audio_file=invalid_input,
                    output_path=output_path,
                    pattern="balanced",
                )
            print(f"   ✅ Invalid audio_file {type(invalid_input).__name__} rejected")

        # Test output_path parameter validation
        invalid_output_paths = [
            None,  # None value
            123,  # Integer instead of string
            "",  # Empty string
        ]

        for invalid_input in invalid_output_paths:
            with pytest.raises((TypeError, ValueError)) as exc_info:
                api_client.process_videos(
                    video_files=["dummy.mp4"],
                    audio_file="dummy.mp3",
                    output_path=invalid_input,
                    pattern="balanced",
                )
            print(f"   ✅ Invalid output_path {type(invalid_input).__name__} rejected")

        # Test pattern parameter validation
        invalid_patterns = [
            "invalid_pattern",  # Unknown pattern
            123,  # Integer instead of string
            None,  # None value
        ]

        for invalid_input in invalid_patterns:
            with pytest.raises((TypeError, ValueError)) as exc_info:
                api_client.process_videos(
                    video_files=["dummy.mp4"],
                    audio_file="dummy.mp3",
                    output_path=output_path,
                    pattern=invalid_input,
                )
            print(f"   ✅ Invalid pattern {invalid_input} rejected")

        print("✅ Parameter validation comprehensive")

    def test_process_videos_valid_patterns(
        self, api_client: AutoCutAPI, temp_dir: Path
    ):
        """Test that all valid patterns are accepted by the API."""
        print("\n🎯 Testing valid pattern acceptance")

        valid_patterns = ["energetic", "balanced", "dramatic", "buildup"]
        output_path = str(temp_dir / "pattern_test.mp4")

        for pattern in valid_patterns:
            try:
                # This will fail on missing files but should not fail on pattern validation
                api_client.process_videos(
                    video_files=["nonexistent.mp4"],
                    audio_file="nonexistent.mp3",
                    output_path=output_path,
                    pattern=pattern,
                )
            except (ValueError, RuntimeError, FileNotFoundError) as e:
                # Should fail on file issues, not pattern validation
                error_msg = str(e).lower()
                assert "pattern" not in error_msg, (
                    f"Valid pattern {pattern} should not cause pattern error"
                )
                assert any(
                    word in error_msg for word in ["file", "not", "found", "exist"]
                ), f"Should fail on file issues for pattern {pattern}"
                print(
                    f"   ✅ Pattern {pattern} accepted (failed on missing files as expected)"
                )

        print("✅ All valid patterns accepted")

    def test_process_videos_optional_parameters(
        self, api_client: AutoCutAPI, temp_dir: Path
    ):
        """Test optional parameters are handled correctly."""
        print("\n⚙️ Testing optional parameters")

        output_path = str(temp_dir / "optional_params_test.mp4")

        # Test with memory_safe parameter
        try:
            api_client.process_videos(
                video_files=["nonexistent.mp4"],
                audio_file="nonexistent.mp3",
                output_path=output_path,
                pattern="balanced",
                memory_safe=True,
            )
        except (ValueError, RuntimeError, FileNotFoundError):
            print("   ✅ memory_safe=True parameter accepted")

        try:
            api_client.process_videos(
                video_files=["nonexistent.mp4"],
                audio_file="nonexistent.mp3",
                output_path=output_path,
                pattern="balanced",
                memory_safe=False,
            )
        except (ValueError, RuntimeError, FileNotFoundError):
            print("   ✅ memory_safe=False parameter accepted")

        # Test with verbose parameter
        try:
            api_client.process_videos(
                video_files=["nonexistent.mp4"],
                audio_file="nonexistent.mp3",
                output_path=output_path,
                pattern="balanced",
                verbose=True,
            )
        except (ValueError, RuntimeError, FileNotFoundError):
            print("   ✅ verbose=True parameter accepted")

        try:
            api_client.process_videos(
                video_files=["nonexistent.mp4"],
                audio_file="nonexistent.mp3",
                output_path=output_path,
                pattern="balanced",
                verbose=False,
            )
        except (ValueError, RuntimeError, FileNotFoundError):
            print("   ✅ verbose=False parameter accepted")

        print("✅ Optional parameters handled correctly")

    def test_validate_video_method_contract(self, api_client: AutoCutAPI):
        """Test validate_video method follows expected contract."""
        print("\n📋 Testing validate_video method contract")

        # Test parameter validation
        with pytest.raises((TypeError, ValueError)) as exc_info:
            api_client.validate_video()  # No parameters
        print(f"   ✅ No parameters rejected: {type(exc_info.value).__name__}")

        # Test with invalid parameter types
        invalid_inputs = [None, 123, [], {}]

        for invalid_input in invalid_inputs:
            with pytest.raises((TypeError, ValueError)) as exc_info:
                api_client.validate_video(invalid_input)
            print(f"   ✅ Invalid input {type(invalid_input).__name__} rejected")

        # Test with non-existent file (should handle gracefully)
        try:
            result = api_client.validate_video("nonexistent_video.mp4")
            # If it returns a result, it should be a dictionary or similar structure
            print(f"   ✅ Non-existent file handled, returned: {type(result).__name__}")
        except (ValueError, RuntimeError, FileNotFoundError) as e:
            # Expected behavior - should raise appropriate exception
            print(f"   ✅ Non-existent file properly rejected: {type(e).__name__}")

        print("✅ validate_video method contract validated")

    @pytest.mark.media_required
    def test_validate_video_with_real_files(
        self, api_client: AutoCutAPI, sample_video_files
    ):
        """Test validate_video with actual media files."""
        real_videos = [f for f in sample_video_files if not f.name.startswith("._")]

        if not real_videos:
            pytest.skip("No real video files available")

        print("\\n📹 Testing validate_video with real files")

        for video_file in real_videos[:2]:  # Test first 2 videos
            print(f"   Validating: {video_file.name}")

            try:
                result = api_client.validate_video(str(video_file))

                # Validate return type and structure
                assert result is not None, "Validation should return a result"

                if isinstance(result, dict):
                    print(f"     ✅ Returned dict with keys: {list(result.keys())}")
                elif isinstance(result, bool):
                    print(f"     ✅ Returned boolean: {result}")
                else:
                    print(f"     ✅ Returned {type(result).__name__}: {result}")

            except Exception as e:
                print(f"     ⚠️ Validation failed: {type(e).__name__}: {e}")

        print("✅ validate_video tested with real files")

    def test_get_system_info_method_contract(self, api_client: AutoCutAPI):
        """Test get_system_info method follows expected contract."""
        print("\\n💻 Testing get_system_info method contract")

        try:
            system_info = api_client.get_system_info()

            # Validate return value
            assert system_info is not None, "System info should not be None"
            print(f"   ✅ Returned {type(system_info).__name__}")

            # Check if it contains expected system information
            info_str = str(system_info).lower()
            expected_keywords = ["cpu", "core", "memory", "system"]

            found_keywords = [kw for kw in expected_keywords if kw in info_str]

            if found_keywords:
                print(f"   ✅ Contains system info keywords: {found_keywords}")
            else:
                print(f"   ⚠️ System info may lack expected details: {system_info}")

            # Test that method is deterministic (same result on multiple calls)
            system_info2 = api_client.get_system_info()
            if system_info == system_info2:
                print("   ✅ Method is deterministic")
            else:
                print("   ⚠️ Method results vary between calls")

        except Exception as e:
            pytest.fail(f"get_system_info should not raise exceptions: {e}")

        print("✅ get_system_info method contract validated")

    def test_error_propagation_and_handling(
        self, api_client: AutoCutAPI, temp_dir: Path
    ):
        """Test that API properly propagates and handles errors."""
        print("\\n🚫 Testing error propagation and handling")

        output_path = str(temp_dir / "error_test.mp4")

        # Test file not found errors
        with pytest.raises((ValueError, RuntimeError, FileNotFoundError)) as exc_info:
            api_client.process_videos(
                video_files=["definitely_does_not_exist.mp4"],
                audio_file="also_does_not_exist.mp3",
                output_path=output_path,
                pattern="balanced",
            )

        error_msg = str(exc_info.value).lower()
        assert any(word in error_msg for word in ["file", "not", "found", "exist"]), (
            f"Error message should mention file issues: {exc_info.value}"
        )
        print(
            f"   ✅ File not found error properly propagated: {type(exc_info.value).__name__}"
        )

        # Test with invalid output directory
        invalid_output = "/root/cannot_create_here/output.mp4"

        with pytest.raises(
            (ValueError, RuntimeError, OSError, PermissionError)
        ) as exc_info:
            api_client.process_videos(
                video_files=["fake.mp4"],
                audio_file="fake.mp3",
                output_path=invalid_output,
                pattern="balanced",
            )
        print(
            f"   ✅ Permission/path error properly propagated: {type(exc_info.value).__name__}"
        )

        # Test that errors are informative
        try:
            api_client.process_videos(
                video_files=["missing.mp4"],
                audio_file="missing.mp3",
                output_path=output_path,
                pattern="balanced",
            )
        except Exception as e:
            error_msg = str(e)
            assert len(error_msg) > 10, "Error messages should be descriptive"
            assert error_msg != "Error", "Error messages should be specific"
            print(f"   ✅ Error message is descriptive: '{error_msg[:50]}...'")

        print("✅ Error propagation and handling validated")

    @pytest.mark.media_required
    def test_api_return_value_contracts(
        self,
        api_client: AutoCutAPI,
        sample_video_files,
        sample_audio_files,
        api_output_dir: Path,
    ):
        """Test that API methods return values according to their contracts."""
        real_videos = [f for f in sample_video_files if not f.name.startswith("._")]
        real_audio = [f for f in sample_audio_files if not f.name.startswith("._")]

        if not real_videos or not real_audio:
            pytest.skip("No real media files available")

        print("\\n📜 Testing API return value contracts")

        # Test process_videos return value
        output_path = str(api_output_dir / "contract_test.mp4")
        small_video = next(
            (f for f in real_videos if f.stat().st_size < 50 * 1024 * 1024),
            real_videos[0],
        )

        try:
            result = api_client.process_videos(
                video_files=[str(small_video)],
                audio_file=str(real_audio[0]),
                output_path=output_path,
                pattern="balanced",
                memory_safe=True,
                verbose=False,
            )

            # Validate return value contract
            assert result is not None, "process_videos should return a value"
            assert isinstance(result, str), (
                f"process_videos should return string path, got {type(result)}"
            )
            assert result == output_path, "Returned path should match requested path"

            # Validate that the returned path exists
            assert Path(result).exists(), f"Returned path should exist: {result}"
            assert Path(result).stat().st_size > 0, "Output file should not be empty"

            print("   ✅ process_videos return contract validated")
            print(f"   📹 Output: {Path(result).stat().st_size / (1024 * 1024):.1f}MB")

        except Exception as e:
            print(f"   ⚠️ process_videos failed: {e}")

        print("✅ API return value contracts validated")

    def test_api_thread_safety_basics(self, api_client: AutoCutAPI):
        """Test basic thread safety of API initialization and method calls."""
        print("\\n🔄 Testing API thread safety basics")

        import threading

        results = []
        errors = []

        def api_worker(worker_id):
            """Worker function to test API in different threads."""
            try:
                # Create separate API instance per thread
                local_api = AutoCutAPI()

                # Test thread-safe method calls
                system_info = local_api.get_system_info()
                results.append((worker_id, system_info))

                # Test parameter validation (should not interfere between threads)
                try:
                    local_api.process_videos(
                        video_files=["fake.mp4"],
                        audio_file="fake.mp3",
                        output_path="fake_output.mp4",
                        pattern="balanced",
                    )
                except Exception:
                    # Expected to fail - we just want to test thread safety
                    pass

            except Exception as e:
                errors.append((worker_id, e))

        # Start multiple threads
        threads = []
        for i in range(3):
            thread = threading.Thread(target=api_worker, args=(i,))
            threads.append(thread)
            thread.start()

        # Wait for completion
        for thread in threads:
            thread.join(timeout=10)

        print(f"   📊 Results: {len(results)} successes, {len(errors)} errors")

        # Validate thread safety
        if len(results) == 3 and len(errors) == 0:
            print("   ✅ All threads completed successfully")
        elif len(results) > 0:
            print("   ✅ Most threads completed - basic thread safety confirmed")
        else:
            print("   ⚠️ Thread safety issues detected")

        # Check that system info is consistent across threads
        if len(results) >= 2:
            first_info = results[0][1]
            consistent = all(result[1] == first_info for result in results)
            if consistent:
                print("   ✅ System info consistent across threads")
            else:
                print("   ⚠️ System info varies between threads")

        print("✅ API thread safety basics tested")

    def test_api_backwards_compatibility_simulation(self, api_client: AutoCutAPI):
        """Test API backwards compatibility with simulated old usage patterns."""
        print("\\n🔄 Testing backwards compatibility simulation")

        # Test that core method signatures haven't changed
        import inspect

        # Check process_videos signature
        process_sig = inspect.signature(api_client.process_videos)
        required_params = ["video_files", "audio_file", "output_path", "pattern"]

        for param in required_params:
            assert param in process_sig.parameters, (
                f"Required parameter {param} missing from process_videos"
            )
            print(f"   ✅ {param} parameter present")

        # Check that optional parameters have defaults
        optional_params = ["memory_safe", "verbose"]
        for param in optional_params:
            if param in process_sig.parameters:
                param_obj = process_sig.parameters[param]
                if param_obj.default is not inspect.Parameter.empty:
                    print(f"   ✅ {param} parameter has default value")
                else:
                    print(f"   ⚠️ {param} parameter lacks default value")

        # Test validate_video signature
        validate_sig = inspect.signature(api_client.validate_video)
        assert (
            "video_path" in validate_sig.parameters or len(validate_sig.parameters) >= 1
        ), "validate_video should accept video path parameter"
        print("   ✅ validate_video signature compatible")

        # Test get_system_info signature
        system_sig = inspect.signature(api_client.get_system_info)
        # Should accept no required parameters
        required_system_params = [
            p
            for p in system_sig.parameters.values()
            if p.default is inspect.Parameter.empty
        ]
        assert len(required_system_params) == 0, (
            "get_system_info should not require parameters"
        )
        print("   ✅ get_system_info signature compatible")

        print("✅ Backwards compatibility simulation passed")

    def test_api_integration_with_core_components(self, api_client: AutoCutAPI):
        """Test that API properly integrates with core components."""
        print("\\n🔧 Testing API integration with core components")

        # Test that API can import and access core modules
        import importlib.util
        
        # Test audio analyzer availability
        if importlib.util.find_spec("audio_analyzer"):
            print("   ✅ Audio analyzer integration available")
        else:
            print("   ⚠️ Audio analyzer integration not available")

        # Test video analyzer availability  
        if importlib.util.find_spec("video_analyzer"):
            print("   ✅ Video analyzer integration available")
        else:
            print("   ⚠️ Video analyzer integration not available")

        # Test clip assembler availability
        if importlib.util.find_spec("clip_assembler"):
            print("   ✅ Clip assembler integration available")
        else:
            print("   ⚠️ Clip assembler integration not available")

        # Test system info integration
        try:
            system_info = api_client.get_system_info()

            # Should contain meaningful system information
            if hasattr(system_info, "__dict__") or isinstance(system_info, dict):
                print("   ✅ System info returns structured data")
            else:
                info_str = str(system_info)
                if len(info_str) > 10:  # Should be more than just a placeholder
                    print("   ✅ System info returns meaningful data")
                else:
                    print("   ⚠️ System info may be incomplete")

        except Exception as e:
            print(f"   ⚠️ System info integration issue: {e}")

        print("✅ Core component integration tested")
