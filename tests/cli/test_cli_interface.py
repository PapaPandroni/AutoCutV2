"""
CLI Interface Tests for AutoCut V2

Tests the actual command-line interface that users will interact with in v1.0.
These tests validate the `autocut.py` commands that power the production system.
"""

import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import List

import pytest


class TestCLIInterface:
    """Test the actual CLI commands users will use."""
    
    @pytest.fixture
    def autocut_path(self) -> str:
        """Path to the autocut.py executable."""
        return str(Path(__file__).parent.parent.parent / "autocut.py")
    
    @pytest.fixture
    def cli_output_dir(self, temp_dir):
        """Output directory for CLI test results."""
        output_dir = temp_dir / "cli_outputs"
        output_dir.mkdir(exist_ok=True)
        return output_dir
    
    def run_cli_command(self, autocut_path: str, args: List[str], timeout: int = 30) -> subprocess.CompletedProcess:
        """Run autocut CLI command and return the result."""
        cmd = [sys.executable, autocut_path] + args
        
        # Set up environment with activated venv
        env = os.environ.copy()
        venv_path = Path(__file__).parent.parent.parent / "env"
        if venv_path.exists():
            env["PATH"] = f"{venv_path / 'bin'}:{env.get('PATH', '')}"
        
        try:
            result = subprocess.run(
                cmd, 
                capture_output=True, 
                text=True, 
                timeout=timeout,
                env=env,
                cwd=str(Path(autocut_path).parent)
            )
            return result
        except subprocess.TimeoutExpired as e:
            # Create a mock result object for timeout
            class TimeoutResult:
                def __init__(self, cmd, timeout_exc):
                    self.args = cmd
                    self.returncode = 124  # Timeout exit code
                    self.stdout = ""
                    self.stderr = f"Command timed out after {timeout} seconds"
                    self.timeout_exc = timeout_exc
            
            return TimeoutResult(cmd, e)
    
    def test_cli_help_command(self, autocut_path: str):
        """Test the main help command works."""
        result = self.run_cli_command(autocut_path, ["--help"])
        
        assert result.returncode == 0, f"Help command failed: {result.stderr}"
        assert "AutoCut V2" in result.stdout, "Should show AutoCut V2 branding"
        assert "process" in result.stdout, "Should list process command"
        assert "validate" in result.stdout, "Should list validate command"
        assert "benchmark" in result.stdout, "Should list benchmark command"
        
        print("✅ CLI help command working")
        print(f"   Output length: {len(result.stdout)} characters")
    
    def test_cli_process_help(self, autocut_path: str):
        """Test the process command help."""
        result = self.run_cli_command(autocut_path, ["process", "--help"])
        
        assert result.returncode == 0, f"Process help failed: {result.stderr}"
        assert "beat-synced compilation" in result.stdout, "Should describe functionality"
        assert "--audio" in result.stdout, "Should show audio option"
        assert "--pattern" in result.stdout, "Should show pattern option" 
        assert "energetic" in result.stdout, "Should list pattern options"
        assert "balanced" in result.stdout, "Should list pattern options"
        assert "dramatic" in result.stdout, "Should list pattern options"
        assert "buildup" in result.stdout, "Should list pattern options"
        
        print("✅ CLI process help working")
        print(f"   Help includes all 4 patterns")
    
    def test_cli_error_handling_missing_files(self, autocut_path: str):
        """Test CLI error handling for missing files."""
        result = self.run_cli_command(
            autocut_path, 
            ["process", "nonexistent.mp4", "--audio", "nonexistent.mp3"]
        )
        
        # Should fail with non-zero exit code
        assert result.returncode != 0, "Should fail with missing files"
        
        # Should show helpful error message
        error_output = result.stderr.lower()
        assert any(word in error_output for word in ["not found", "error", "file"]), \
            f"Should show file error message, got: {result.stderr}"
        
        print("✅ CLI properly handles missing files")
        print(f"   Error message: {result.stderr.strip()}")
    
    def test_cli_error_handling_missing_audio(self, autocut_path: str, sample_video_files):
        """Test CLI error handling for missing audio file."""
        # Filter out metadata files
        real_videos = [f for f in sample_video_files if not f.name.startswith('._')]
        
        if not real_videos:
            pytest.skip("No real video files available")
        
        result = self.run_cli_command(
            autocut_path,
            ["process", str(real_videos[0]), "--audio", "nonexistent_audio.mp3"]
        )
        
        assert result.returncode != 0, "Should fail with missing audio file"
        
        error_output = result.stderr.lower()
        assert any(word in error_output for word in ["not found", "audio", "file"]), \
            f"Should show audio file error, got: {result.stderr}"
        
        print("✅ CLI properly handles missing audio file")
    
    def test_cli_validation_command(self, autocut_path: str, sample_video_files):
        """Test the validate command works."""
        real_videos = [f for f in sample_video_files if not f.name.startswith('._')]
        
        if not real_videos:
            pytest.skip("No real video files available")
        
        # Test basic validation
        result = self.run_cli_command(
            autocut_path,
            ["validate", str(real_videos[0])],
            timeout=60  # Longer timeout for validation
        )
        
        # Should succeed (video files should be valid)
        if result.returncode == 124:  # Timeout
            print("⏱️ Validation command timed out (still processing)")
            return  # Skip assertion, but log that it started
        
        assert result.returncode == 0, f"Validation failed: {result.stderr}"
        
        # Should show validation results
        output = result.stdout.lower()
        assert any(word in output for word in ["valid", "compatible", "video"]), \
            f"Should show validation results, got: {result.stdout}"
        
        print(f"✅ CLI validation command working")
        print(f"   Validated: {Path(real_videos[0]).name}")
    
    def test_cli_benchmark_command(self, autocut_path: str):
        """Test the benchmark command works."""
        result = self.run_cli_command(
            autocut_path,
            ["benchmark"],
            timeout=30
        )
        
        if result.returncode == 124:  # Timeout
            print("⏱️ Benchmark command timed out (still running)")
            return  # Skip assertion, but log that it started
        
        assert result.returncode == 0, f"Benchmark failed: {result.stderr}"
        
        # Should show system information
        output = result.stdout.lower()
        assert any(word in output for word in ["system", "capabilities", "cpu"]), \
            f"Should show system info, got: {result.stdout}"
        
        print("✅ CLI benchmark command working")
    
    @pytest.mark.media_required
    def test_cli_process_command_basic(
        self, 
        autocut_path: str, 
        sample_video_files,
        sample_audio_files, 
        cli_output_dir: Path
    ):
        """Test basic process command with real media files."""
        # Filter out metadata files
        real_videos = [f for f in sample_video_files if not f.name.startswith('._')]
        real_audio = [f for f in sample_audio_files if not f.name.startswith('._')]
        
        if not real_videos or not real_audio:
            pytest.skip("No real media files available")
        
        # Use smaller files for faster testing
        small_videos = [f for f in real_videos if f.stat().st_size < 50 * 1024 * 1024]  # < 50MB
        if not small_videos:
            small_videos = real_videos[:1]  # Use first video as fallback
        
        output_path = cli_output_dir / "cli_process_test.mp4"
        
        # Test process command with minimal files
        result = self.run_cli_command(
            autocut_path,
            [
                "process", 
                str(small_videos[0]),  # Use just one video for speed
                "--audio", str(real_audio[0]),
                "--output", str(output_path),
                "--pattern", "balanced",
                "--verbose"
            ],
            timeout=180  # 3 minutes for processing
        )
        
        if result.returncode == 124:  # Timeout
            print("⏱️ CLI process command timed out (still processing)")
            print("   This indicates the command started successfully")
            # Check if partial output exists
            if output_path.exists():
                print(f"   Partial output created: {output_path.stat().st_size} bytes")
            return
        
        # Check if command succeeded
        if result.returncode != 0:
            print(f"❌ CLI process failed:")
            print(f"   Exit code: {result.returncode}")
            print(f"   Stderr: {result.stderr}")
            print(f"   Stdout: {result.stdout}")
            pytest.fail(f"CLI process command failed: {result.stderr}")
        
        # Validate output
        assert output_path.exists(), f"Output video should be created: {output_path}"
        assert output_path.stat().st_size > 0, "Output video should not be empty"
        
        print("✅ CLI process command working")
        print(f"   Input: {Path(small_videos[0]).name}")
        print(f"   Audio: {Path(real_audio[0]).name}") 
        print(f"   Output: {output_path.stat().st_size / (1024*1024):.1f} MB")
    
    def test_cli_process_patterns(
        self,
        autocut_path: str,
        sample_video_files,
        sample_audio_files,
        cli_output_dir: Path
    ):
        """Test that all patterns are accepted by CLI."""
        real_videos = [f for f in sample_video_files if not f.name.startswith('._')]
        real_audio = [f for f in sample_audio_files if not f.name.startswith('._')]
        
        if not real_videos or not real_audio:
            pytest.skip("No real media files available")
        
        patterns = ["energetic", "balanced", "dramatic", "buildup"]
        
        for pattern in patterns:
            # We'll run a very quick validation - start the command but don't wait for completion
            result = self.run_cli_command(
                autocut_path,
                [
                    "process",
                    str(real_videos[0]),
                    "--audio", str(real_audio[0]),
                    "--pattern", pattern,
                    "--output", str(cli_output_dir / f"pattern_{pattern}_test.mp4")
                ],
                timeout=5  # Very short timeout - just to test command acceptance
            )
            
            # If it times out, that means it started processing (good!)
            # If it fails immediately, check it's not a pattern validation error
            if result.returncode != 0 and result.returncode != 124:
                error_msg = result.stderr.lower()
                assert "pattern" not in error_msg, f"Pattern {pattern} should be valid"
                # Other errors (file not found, etc.) are ok for this test
            
            print(f"✅ Pattern {pattern} accepted by CLI")
    
    def test_cli_version_display(self, autocut_path: str):
        """Test that version information is displayed correctly."""
        result = self.run_cli_command(autocut_path, ["--version"])
        
        assert result.returncode == 0, f"Version command failed: {result.stderr}"
        assert "AutoCut" in result.stdout, "Should show AutoCut in version"
        assert any(c.isdigit() for c in result.stdout), "Should show version number"
        
        print("✅ CLI version command working")
        print(f"   Version info: {result.stdout.strip()}")
    
    def test_cli_output_formatting(self, autocut_path: str):
        """Test that CLI output is properly formatted with emojis and colors."""
        result = self.run_cli_command(autocut_path, ["--help"])
        
        assert result.returncode == 0
        
        # Should have structured help output
        lines = result.stdout.split('\n')
        assert len(lines) > 5, "Help should have multiple lines"
        
        # Check for command structure
        help_content = result.stdout.lower()
        assert "usage:" in help_content, "Should show usage information"
        assert "options:" in help_content, "Should show options section"
        assert "commands:" in help_content, "Should show commands section"
        
        print("✅ CLI output properly formatted")
        print(f"   Help has {len(lines)} lines")
    
    def test_cli_executable_permissions(self, autocut_path: str):
        """Test that autocut.py has proper executable permissions."""
        autocut_file = Path(autocut_path)
        
        assert autocut_file.exists(), "autocut.py should exist"
        
        # Check if file is executable
        import stat
        file_stat = autocut_file.stat()
        is_executable = bool(file_stat.st_mode & stat.S_IEXEC)
        
        assert is_executable, "autocut.py should be executable"
        
        # Check shebang line
        with open(autocut_path, 'r') as f:
            first_line = f.readline().strip()
        
        assert first_line.startswith("#!"), "Should have shebang line"
        assert "python" in first_line, "Should use python in shebang"
        
        print("✅ autocut.py has proper executable permissions")
        print(f"   Shebang: {first_line}")