#!/bin/bash

set -e

DEVICE_NAME=${1:-trn1}
OUTPUT_DIR="output"
HLO_DIR="${OUTPUT_DIR}/generic/hlo"
NEFF_DIR="${OUTPUT_DIR}/${DEVICE_NAME}/neff"

echo "Generating all kernels for device: ${DEVICE_NAME}"

# Create output directories
mkdir -p "${HLO_DIR}" "${NEFF_DIR}"

# Build if needed
if [ ! -f "build/kernel-generator" ]; then
    echo "Building kernel generator..."
    mkdir -p build
    cd build
    cmake -DUSE_MOCK_XLA=OFF ..
    cmake --build .
    cd ..
fi

# Define kernels and their shapes
KERNELS="add:1024,1024 mul:1024,1024 sub:1024,1024 matmul:512,512 transpose:256,512 reshape:1024,1024 relu:4096 gelu:4096 silu:4096 softmax:32000"

echo "Generating kernels..."

for kernel_spec in $KERNELS; do
    kernel=$(echo $kernel_spec | cut -d: -f1)
    shape=$(echo $kernel_spec | cut -d: -f2 | tr ',' ' ')
    
    echo "  Generating ${kernel} with shape [${shape}]..."
    
    # Generate kernel (output goes directly to output/ directories)
    ./build/kernel-generator ${kernel} ${shape} ${kernel}
done

echo ""
echo "Generated files:"
echo "HLO files: $(ls -1 ${HLO_DIR}/*.hlo 2>/dev/null | wc -l | tr -d ' ')"
echo "NEFF files: $(ls -1 ${NEFF_DIR}/*.neff 2>/dev/null | wc -l | tr -d ' ')"
echo ""
echo "Output structure:"
find ${OUTPUT_DIR} -type f | sort
