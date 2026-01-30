"""
Tests for the Wolfi CUDA Base Image Builder.

Run with: pytest tests/ -v
"""

import pytest
import sys
import os

# Add parent directory to path to import main
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from main import (
    get_image_reference,
    OS_VERSIONS,
    CUDA_VERSIONS,
    PYTHON_VERSIONS,
    FRAMEWORK_CONFIGS,
    PLATFORMS,
)


class TestImageReference:
    """Tests for image reference generation."""

    def test_base_image_reference(self):
        """Test base image tag generation."""
        ref = get_image_reference("wolfi", "12.4.1", "base", "3.11")
        assert ref == "wolfi_python_3.11_cuda_12.4.1_base"

    def test_pytorch_image_reference(self):
        """Test PyTorch image tag generation."""
        ref = get_image_reference("wolfi", "12.4.1", "pytorch", "3.11")
        assert ref == "wolfi_python_3.11_cuda_12.4.1_pytorch"

    def test_tensorflow_image_reference(self):
        """Test TensorFlow image tag generation."""
        ref = get_image_reference("wolfi", "12.6.0", "tensorflow", "3.12")
        assert ref == "wolfi_python_3.12_cuda_12.6.0_tensorflow"

    def test_different_python_versions(self):
        """Test image tags with different Python versions."""
        ref_311 = get_image_reference("wolfi", "12.4.1", "base", "3.11")
        ref_312 = get_image_reference("wolfi", "12.4.1", "base", "3.12")
        assert "3.11" in ref_311
        assert "3.12" in ref_312
        assert ref_311 != ref_312

    def test_platform_specific_reference_amd64(self):
        """Test platform-specific image tag for amd64."""
        ref = get_image_reference("wolfi", "12.4.1", "base", "3.11", "linux/amd64")
        assert ref == "wolfi_python_3.11_cuda_12.4.1_base_amd64"

    def test_platform_specific_reference_arm64(self):
        """Test platform-specific image tag for arm64."""
        ref = get_image_reference("wolfi", "12.4.1", "pytorch", "3.12", "linux/arm64")
        assert ref == "wolfi_python_3.12_cuda_12.4.1_pytorch_arm64"

    def test_no_platform_reference(self):
        """Test that None platform doesn't add suffix."""
        ref = get_image_reference("wolfi", "12.4.1", "base", "3.11", None)
        assert ref == "wolfi_python_3.11_cuda_12.4.1_base"
        assert "amd64" not in ref
        assert "arm64" not in ref


class TestConfiguration:
    """Tests for build configuration."""

    def test_os_versions_not_empty(self):
        """Ensure OS versions list is not empty."""
        assert len(OS_VERSIONS) > 0

    def test_cuda_versions_not_empty(self):
        """Ensure CUDA versions list is not empty."""
        assert len(CUDA_VERSIONS) > 0

    def test_python_versions_not_empty(self):
        """Ensure Python versions list is not empty."""
        assert len(PYTHON_VERSIONS) > 0

    def test_framework_configs_has_base(self):
        """Ensure framework configs includes base."""
        assert "base" in FRAMEWORK_CONFIGS

    def test_framework_configs_has_pytorch(self):
        """Ensure framework configs includes PyTorch."""
        assert "pytorch" in FRAMEWORK_CONFIGS

    def test_framework_configs_has_tensorflow(self):
        """Ensure framework configs includes TensorFlow."""
        assert "tensorflow" in FRAMEWORK_CONFIGS

    def test_wolfi_in_os_versions(self):
        """Ensure Wolfi is in OS versions."""
        assert "wolfi" in OS_VERSIONS

    def test_cuda_version_format(self):
        """Ensure CUDA versions follow expected format (x.y.z)."""
        for version in CUDA_VERSIONS:
            parts = version.split(".")
            assert len(parts) >= 2, f"CUDA version {version} should have at least major.minor"

    def test_python_version_format(self):
        """Ensure Python versions follow expected format (x.y)."""
        for version in PYTHON_VERSIONS:
            parts = version.split(".")
            assert len(parts) == 2, f"Python version {version} should be major.minor"

    def test_platforms_not_empty(self):
        """Ensure platforms list is not empty."""
        assert len(PLATFORMS) > 0

    def test_amd64_in_platforms(self):
        """Ensure linux/amd64 is in platforms."""
        assert "linux/amd64" in PLATFORMS

    def test_arm64_in_platforms(self):
        """Ensure linux/arm64 is in platforms."""
        assert "linux/arm64" in PLATFORMS

    def test_platform_format(self):
        """Ensure platforms follow expected format (os/arch)."""
        for platform in PLATFORMS:
            parts = platform.split("/")
            assert len(parts) == 2, f"Platform {platform} should be os/arch format"
            assert parts[0] == "linux", f"Platform {platform} should be linux-based"


class TestBuildMatrix:
    """Tests for the build matrix."""

    def test_total_image_count(self):
        """Calculate total number of images in build matrix."""
        total = (
            len(OS_VERSIONS)
            * len(CUDA_VERSIONS)
            * len(FRAMEWORK_CONFIGS)
            * len(PYTHON_VERSIONS)
        )
        # Should be reasonable number (not too many, not zero)
        assert total > 0
        assert total < 100  # Sanity check

    def test_unique_image_references(self):
        """Ensure all generated image references are unique."""
        refs = set()
        for os_ver in OS_VERSIONS:
            for cuda_ver in CUDA_VERSIONS:
                for framework in FRAMEWORK_CONFIGS.keys():
                    for py_ver in PYTHON_VERSIONS:
                        ref = get_image_reference(os_ver, cuda_ver, framework, py_ver)
                        assert ref not in refs, f"Duplicate reference: {ref}"
                        refs.add(ref)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
