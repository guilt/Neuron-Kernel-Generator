#!/bin/bash

set -e

echo "Neuron Kernel Generator - Build and Generate"
echo "============================================"

# Parse arguments
DEVICE_NAME=${1:-trn1}
USE_DOCKER=${2:-true}

if [ "$USE_DOCKER" = "true" ]; then
    echo "Using Finch build for x86_64..."
    
    # Build Docker image with x86_64 platform
    echo "Building Docker image..."
    docker build --platform linux/amd64 -t neuron-kernel-generator .
    
    # Run container with x86_64 platform
    echo "Generating kernels for ${DEVICE_NAME}..."
    docker run --platform linux/amd64 --rm -v "$(pwd)/output:/workspace/output" neuron-kernel-generator ./generate_all_kernels.sh "${DEVICE_NAME}"
else
    echo "Using local build..."
    
    # Local build and generate
    ./generate_all_kernels.sh "${DEVICE_NAME}"
fi

echo ""
echo "Build and generation complete!"
echo "Check output/ directory for generated files."
