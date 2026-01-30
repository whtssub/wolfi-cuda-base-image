"""
Wolfi CUDA Base Image Builder

This module builds and publishes lightweight Docker images based on Wolfi Linux
with NVIDIA CUDA support for deep learning frameworks.
"""

import sys
import asyncio
import logging
import os
from typing import List, Optional

import dagger

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Build configuration
OS_VERSIONS = ["wolfi"]
CUDA_VERSIONS = ["12.4.1", "12.6.0"]
PYTHON_VERSIONS = ["3.11", "3.12"]

# Supported platforms for multi-architecture builds
# Note: CUDA support on ARM64 may be limited depending on the packages available
PLATFORMS = ["linux/amd64", "linux/arm64"]

# Framework configurations: (tag_name, conda_packages_to_install)
FRAMEWORK_CONFIGS = {
    "base": "",
    "pytorch": "pytorch",
    "tensorflow": "tensorflow",
}


def get_image_reference(
    os_version: str,
    cuda_version: str,
    framework: str,
    python_version: str,
    platform: Optional[str] = None
) -> str:
    """Generate a standardized image reference tag."""
    base_ref = f"{os_version}_python_{python_version}_cuda_{cuda_version}_{framework}"
    if platform:
        # Convert platform to tag-friendly format (linux/amd64 -> amd64)
        arch = platform.split("/")[-1]
        return f"{base_ref}_{arch}"
    return base_ref


def build_container(
    client: dagger.Client,
    base_image: str,
    package_str: str,
    conda_packages_str: str,
    cuda_version: str,
    python_version: str,
    framework: str,
    username: str,
    repository: str,
    secret: dagger.Secret,
    platform: Optional[str] = None
) -> dagger.Container:
    """
    Build a container with the specified configuration.

    Args:
        client: Dagger client instance
        base_image: Base container image
        package_str: Space-separated list of APK packages to install
        conda_packages_str: Space-separated list of conda packages to install
        cuda_version: CUDA version for labels
        python_version: Python version for labels
        framework: Framework name for labels
        username: GitHub username for labels
        repository: Repository name for labels
        secret: Registry authentication secret
        platform: Target platform (e.g., "linux/amd64")

    Returns:
        Configured Dagger container
    """
    # Create container with optional platform specification
    if platform:
        container = client.container(platform=dagger.Platform(platform))
    else:
        container = client.container()

    return (
        container
        .from_(base_image)
        .with_user("root")
        .with_workdir("/app")
        # Install base packages from Wolfi
        .with_exec([
            "/bin/sh", "-c",
            f"apk update && apk add --no-cache {package_str}"
        ])
        # Install micromamba for CUDA packages
        .with_exec([
            "/bin/sh", "-c",
            'curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest | tar -xvj -C /usr/local bin/micromamba'
        ])
        # Install CUDA and framework packages via micromamba
        .with_exec([
            "/bin/sh", "-c",
            f'/usr/local/bin/micromamba install -y -n base -c conda-forge {conda_packages_str} && /usr/local/bin/micromamba clean --all --yes'
        ])
        # Set environment variables
        .with_env_variable("MAMBA_ROOT_PREFIX", "/root/micromamba")
        .with_env_variable("PATH", "/root/micromamba/bin:/usr/local/cuda/bin:$PATH")
        .with_env_variable("LD_LIBRARY_PATH", "/root/micromamba/lib:$LD_LIBRARY_PATH")
        # Add OCI labels for better discoverability
        .with_label("org.opencontainers.image.source", f"https://github.com/{username}/{repository}")
        .with_label("org.opencontainers.image.description", f"Wolfi-based CUDA {cuda_version} image with Python {python_version}")
        .with_label("org.opencontainers.image.licenses", "Apache-2.0")
        .with_label("org.opencontainers.image.title", f"wolfi-cuda-{framework}")
        .with_registry_auth(address="ghcr.io", username=username, secret=secret)
    )


async def build_and_publish_image(
    client: dagger.Client,
    os_version: str,
    cuda_version: str,
    framework: str,
    framework_packages: str,
    python_version: str,
    repository: str,
    username: str,
    password: str,
    platforms: Optional[List[str]] = None
) -> None:
    """
    Build and publish a CUDA container image to GitHub Container Registry.

    Args:
        client: Dagger client instance
        os_version: Base OS version (e.g., "wolfi")
        cuda_version: CUDA toolkit version (e.g., "12.4.1")
        framework: Framework name for tagging (e.g., "pytorch", "tensorflow", "base")
        framework_packages: APK packages to install for the framework
        python_version: Python version (e.g., "3.11")
        repository: GitHub repository name
        username: GitHub username for registry auth
        password: GitHub token for registry auth
        platforms: List of platforms to build for (e.g., ["linux/amd64", "linux/arm64"])
    """
    img_ref = get_image_reference(os_version, cuda_version, framework, python_version)
    logger.info(f"Building image: {img_ref}")

    # Use Chainguard's Wolfi base image
    base_image = "cgr.dev/chainguard/wolfi-base"

    # Build package list with Wolfi package names
    # Note: CUDA packages are installed via micromamba from conda-forge
    packages = [
        f"python-{python_version}",
        f"py{python_version}-pip",
        "curl",
        "bash",
    ]

    package_str = " ".join(packages)
    
    # Build conda packages list for CUDA and frameworks
    cuda_major = cuda_version.rsplit(".", 1)[0]  # e.g., "12.4"
    conda_packages = [f"cuda-toolkit={cuda_major}"]
    
    if framework_packages:
        conda_packages.append(framework_packages)
    
    conda_packages_str = " ".join(conda_packages)

    try:
        secret = client.set_secret("password", password)
        image_uri = f"ghcr.io/{username}/{repository}:{img_ref}"

        if platforms and len(platforms) > 1:
            # Multi-platform build
            logger.info(f"Building multi-platform image for: {platforms}")

            # Build containers for each platform
            platform_variants = []
            for platform in platforms:
                container = build_container(
                    client=client,
                    base_image=base_image,
                    package_str=package_str,
                    conda_packages_str=conda_packages_str,
                    cuda_version=cuda_version,
                    python_version=python_version,
                    framework=framework,
                    username=username,
                    repository=repository,
                    secret=secret,
                    platform=platform
                )
                platform_variants.append(container)

                # Also publish platform-specific tag
                platform_ref = get_image_reference(
                    os_version, cuda_version, framework, python_version, platform
                )
                platform_uri = f"ghcr.io/{username}/{repository}:{platform_ref}"
                await container.publish(platform_uri)
                logger.info(f"Published platform-specific: {platform_uri}")

            logger.info(f"Successfully published multi-arch: {image_uri}")
        else:
            # Single platform build (default to amd64)
            container = build_container(
                client=client,
                base_image=base_image,
                package_str=package_str,
                conda_packages_str=conda_packages_str,
                cuda_version=cuda_version,
                python_version=python_version,
                framework=framework,
                username=username,
                repository=repository,
                secret=secret,
                platform=platforms[0] if platforms else None
            )

            await container.publish(image_uri)
            logger.info(f"Successfully published: {image_uri}")

    except Exception as e:
        logger.error(f"Failed to build {img_ref}: {e}")
        raise


async def main() -> None:
    """Main entry point for building and publishing all image variants."""
    repository = os.environ.get("REPOSITORY", "wolfi-cuda-base-image")
    username = os.environ.get("USERNAME") or os.environ.get("username")
    password = os.environ.get("PASSWORD") or os.environ.get("password")

    # Check if multi-arch builds are enabled (set to "true" to enable)
    enable_multi_arch = os.environ.get("MULTI_ARCH", "false").lower() == "true"

    if not username or not password:
        logger.error("Environment variables 'USERNAME' and 'PASSWORD' are required.")
        sys.exit(1)

    # Determine platforms to build for
    platforms_to_build = PLATFORMS if enable_multi_arch else ["linux/amd64"]

    logger.info(f"Starting build process for repository: {repository}")
    logger.info(f"CUDA versions: {CUDA_VERSIONS}")
    logger.info(f"Python versions: {PYTHON_VERSIONS}")
    logger.info(f"Frameworks: {list(FRAMEWORK_CONFIGS.keys())}")
    logger.info(f"Platforms: {platforms_to_build}")
    logger.info(f"Multi-arch enabled: {enable_multi_arch}")

    async with dagger.Connection(dagger.Config(log_output=sys.stderr)) as client:
        tasks = [
            build_and_publish_image(
                client=client,
                os_version=os_version,
                cuda_version=cuda_version,
                framework=framework,
                framework_packages=packages,
                python_version=python_version,
                repository=repository,
                username=username,
                password=password,
                platforms=platforms_to_build
            )
            for os_version in OS_VERSIONS
            for cuda_version in CUDA_VERSIONS
            for framework, packages in FRAMEWORK_CONFIGS.items()
            for python_version in PYTHON_VERSIONS
        ]

        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Check for failures
        failures = [r for r in results if isinstance(r, Exception)]
        if failures:
            logger.error(f"{len(failures)} image(s) failed to build")
            for failure in failures:
                logger.error(f"  - {failure}")
            sys.exit(1)

    logger.info("All images built and published successfully!")


if __name__ == "__main__":
    asyncio.run(main())