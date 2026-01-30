# Wolfi CUDA Base Image



Lightweight Docker images based on [Wolfi Linux](https://wolfi.dev/) optimized for NVIDIA CUDA and deep learning frameworks. These images provide a minimal, secure base for building and deploying AI/ML applications with significantly reduced image sizes compared to official NVIDIA images.

## Features

- **Minimal footprint** - Up to 70% smaller than official NVIDIA CUDA images
- **Security-focused** - Built on Wolfi Linux with minimal attack surface
- **Multi-framework support** - Base, PyTorch, and TensorFlow variants
- **Multiple Python versions** - Support for Python 3.11 and 3.12
- **Automated builds** - CI/CD pipeline with GitHub Actions and Dagger

## Image Size Comparison

| Image Type | Size |
|------------|------|
| NVIDIA CUDA Base | 1.4 GB |
| NVIDIA CUDA 12.4.1 | 2.5 GB |
| **Wolfi CUDA 12.4.1 (Base)** | **382 MB** |
| **Wolfi CUDA 12.4.1 (TensorFlow)** | **1.2 GB** |
| **Wolfi CUDA 12.4.1 (PyTorch)** | **1.1 GB** |

## Quick Start

### Pull Pre-built Images

```bash
# Base CUDA image
docker pull ghcr.io/YOUR_USERNAME/wolfi-cuda-base-image:wolfi_python_3.11_cuda_12.4.1_base

# PyTorch image
docker pull ghcr.io/YOUR_USERNAME/wolfi-cuda-base-image:wolfi_python_3.11_cuda_12.4.1_pytorch

# TensorFlow image
docker pull ghcr.io/YOUR_USERNAME/wolfi-cuda-base-image:wolfi_python_3.11_cuda_12.4.1_tensorflow
```

### Build Locally with Docker

```bash
# Build base image
docker build -t wolfi-cuda:base .

# Build PyTorch image
docker build --build-arg FRAMEWORK=pytorch -t wolfi-cuda:pytorch .

# Build TensorFlow image
docker build --build-arg FRAMEWORK=tensorflow -t wolfi-cuda:tensorflow .

# Build with specific Python version
docker build --build-arg PYTHON_VERSION=3.12 -t wolfi-cuda:py312 .
```

### Run a Container

```bash
# Interactive shell
docker run --gpus all -it wolfi-cuda:pytorch /bin/sh

# Run a Python script
docker run --gpus all -v $(pwd):/app wolfi-cuda:pytorch python3 your_script.py
```

## Build with Dagger

This project uses [Dagger](https://dagger.io/) for reproducible builds across different environments.

### Prerequisites

- Python 3.8+
- [Dagger CLI](https://docs.dagger.io/install)
- Docker (for local builds)

### Installation

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/wolfi-cuda-base-image.git
cd wolfi-cuda-base-image

# Install Python dependencies
pip install -r requirements.txt
```

### Build and Publish

```bash
# Set environment variables
export USERNAME="your-github-username"
export PASSWORD="your-github-pat"  # Personal Access Token with packages:write scope
export REPOSITORY="wolfi-cuda-base-image"  # Optional, defaults to wolfi-cuda-base-image

# Run the build pipeline
python main.py
```

## Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `USERNAME` | Yes | GitHub username for container registry authentication |
| `PASSWORD` | Yes | GitHub Personal Access Token with `packages:write` scope |
| `REPOSITORY` | No | Repository name (default: `wolfi-cuda-base-image`) |
| `MULTI_ARCH` | No | Set to `true` to enable multi-architecture builds (amd64 + arm64) |

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        GitHub Actions                           │
│                    (CI/CD Pipeline Trigger)                     │
└─────────────────────────┬───────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│                      Dagger Pipeline                            │
│                       (main.py)                                 │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  Build Matrix:                                           │   │
│  │  • CUDA: 12.4.1, 12.6.0                                 │   │
│  │  • Python: 3.11, 3.12                                   │   │
│  │  • Framework: base, pytorch, tensorflow                  │   │
│  └─────────────────────────────────────────────────────────┘   │
└─────────────────────────┬───────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│                   Wolfi Base Image                              │
│              (cgr.dev/chainguard/wolfi-base)                    │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  + CUDA Toolkit + cuDNN + cuBLAS + Python + Framework   │   │
│  └─────────────────────────────────────────────────────────┘   │
└─────────────────────────┬───────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│              GitHub Container Registry (ghcr.io)                │
│                     Published Images                            │
└─────────────────────────────────────────────────────────────────┘
```

## Available Image Tags

Images are tagged with the following format:
```
ghcr.io/{username}/{repository}:{os}_python_{python_version}_cuda_{cuda_version}_{framework}
```

### Current Build Matrix

| OS | Python | CUDA | Framework |
|----|--------|------|-----------|
| wolfi | 3.11, 3.12 | 12.4.1, 12.6.0 | base, pytorch, tensorflow |

## Using as a Base Image

```dockerfile
FROM ghcr.io/YOUR_USERNAME/wolfi-cuda-base-image:wolfi_python_3.11_cuda_12.4.1_pytorch

# Install additional dependencies
RUN pip install transformers datasets

# Copy your application
COPY . /app
WORKDIR /app

# Run your application
CMD ["python3", "train.py"]
```

## Troubleshooting

### GPU not detected

Ensure you have the NVIDIA Container Toolkit installed:

```bash
# Ubuntu/Debian
sudo apt-get install -y nvidia-container-toolkit
sudo systemctl restart docker
```

### Permission denied on GHCR

Make sure your Personal Access Token has the `packages:write` scope and you're logged in:

```bash
echo $PASSWORD | docker login ghcr.io -u $USERNAME --password-stdin
```

### Build fails with package not found

Wolfi packages may have different names than standard Alpine packages. Check the [Wolfi package repository](https://github.com/wolfi-dev/os) for available packages.

## Contributing

Contributions are welcome! Please open an issue or submit a pull request.





## Acknowledgments

- [Chainguard](https://chainguard.dev/) for the Wolfi Linux distribution
- [Dagger](https://dagger.io/) for the portable CI/CD engine
- [NVIDIA](https://nvidia.com/) for CUDA toolkit
