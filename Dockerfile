# Wolfi CUDA Base Image
#
# This Dockerfile creates a lightweight CUDA-enabled container based on Wolfi Linux.
# It provides a minimal, secure base for AI/ML workloads.
#
# Build arguments:
#   PYTHON_VERSION - Python version to install (default: 3.11)
#   FRAMEWORK - Framework to install: base, pytorch, or tensorflow (default: base)
#
# Usage:
#   docker build -t wolfi-cuda:base .
#   docker build --build-arg FRAMEWORK=pytorch -t wolfi-cuda:pytorch .
#   docker build --build-arg FRAMEWORK=tensorflow -t wolfi-cuda:tensorflow .

ARG PYTHON_VERSION=3.11
ARG FRAMEWORK=base

# Use Chainguard's Wolfi base image for minimal attack surface
FROM cgr.dev/chainguard/wolfi-base

ARG PYTHON_VERSION
ARG FRAMEWORK

# Set environment variables for CUDA
ENV CUDA_HOME=/usr/local/cuda \
    PATH=/usr/local/cuda/bin:$PATH \
    LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Install base packages including CUDA toolkit
RUN apk update && apk add --no-cache \
    python-${PYTHON_VERSION} \
    py${PYTHON_VERSION}-pip \
    cuda-toolkit \
    cuda-cudnn \
    libcublas \
    libcufft \
    libcurand \
    libcusolver \
    libcusparse \
    && rm -rf /var/cache/apk/*

# Install framework-specific packages based on FRAMEWORK arg
RUN if [ "$FRAMEWORK" = "pytorch" ]; then \
        apk add --no-cache py3-pytorch; \
    elif [ "$FRAMEWORK" = "tensorflow" ]; then \
        apk add --no-cache py3-tensorflow; \
    fi

# Set working directory
WORKDIR /app

# Create non-root user for security
RUN adduser -D -u 1000 appuser
USER appuser

# Default command - can be overridden
CMD ["python3", "--version"]
