# Neuron Kernel Generator

A modular C++ tool for generating HLO and NEFF files from GGML/llama.cpp kernels for AWS Neuron (Inferentia/Trainium).

## Features

- **Modular Architecture**: Clean base class with extensible kernel implementations
- **GGML Support**: Essential kernels for llama.cpp backend development
- **Mock Testing**: Build and test without full Neuron SDK installation
- **Docker Ready**: Designed for Neuron SDK Docker environment
- **Automated Generation**: Scripts to generate all kernels with organized output

## Available Kernels

### Arithmetic Operations
- `add` - Element-wise addition
- `mul` - Element-wise multiplication  
- `sub` - Element-wise subtraction

### Matrix Operations
- `matmul` - Matrix multiplication
- `transpose` - Matrix transpose
- `reshape` - Tensor reshape

### Activation Functions
- `relu` - Rectified Linear Unit
- `gelu` - Gaussian Error Linear Unit
- `silu` - Sigmoid Linear Unit (Swish)
- `softmax` - Softmax normalization

## Quick Start

### Generate All Kernels (Recommended)
```bash
# Local generation
./generate_all_kernels.sh trn1

# Docker generation
./build_and_generate.sh trn1 true

# Docker Compose
docker-compose up
```

### Manual Build and Generate
```bash
mkdir build && cd build
cmake ..
make

# Generate individual kernels
./kernel-generator add 1024 1024 my_add_kernel
./kernel-generator matmul 512 512 llama_matmul
```

## Output Structure

```
output/
├── generic/hlo/          # HLO files (device-independent)
│   ├── add.hlo
│   ├── matmul.hlo
│   └── ...
└── trn1/neff/           # NEFF files (device-specific)
    ├── add.neff
    ├── matmul.neff
    └── ...
```

## Docker Usage (Production)

### Build and Run
```bash
# Build image
docker build -t neuron-kernel-generator .

# Generate for trn1
docker run --rm -v $(pwd)/output:/workspace/output neuron-kernel-generator

# Generate for inf2
docker run --rm -v $(pwd)/output:/workspace/output neuron-kernel-generator ./generate_all_kernels.sh inf2
```

### Docker Compose
```bash
# Generate for trn1
docker-compose up neuron-kernel-generator

# Generate for inf2
docker-compose up neuron-kernel-generator-inf2
```

## Architecture

```
kernel-generator.cpp          # Main entry point
base_kernel.h/cpp            # Abstract base class
mock_xla.h                   # Mock XLA for testing
kernels/
├── arithmetic_kernels.h/cpp # Add, mul, sub
├── matrix_kernels.h/cpp     # MatMul, transpose, reshape
└── activation_kernels.h/cpp # ReLU, GELU, SiLU, softmax
scripts/
├── generate_all_kernels.sh  # Generate all kernels
├── build_and_generate.sh    # Build and generate wrapper
└── docker-compose.yml       # Docker orchestration
```

## Adding New Kernels

1. **Create kernel class**:
```cpp
class MyKernel : public BaseKernel {
public:
    MyKernel(const std::vector<int64_t>& shape) : BaseKernel(shape, "my_kernel") {}
protected:
    std::unique_ptr<xla::HloModule> build_hlo() override;
};
```

2. **Register in factory**:
```cpp
{"my_kernel", [](const std::vector<int64_t>& shape) { 
    return std::make_unique<MyKernel>(shape); 
}}
```

3. **Add to generation script**:
```bash
# In generate_all_kernels.sh
KERNELS="... my_kernel:1024,1024"
```

## Output Files

- `output/generic/hlo/<kernel>.hlo` - HLO intermediate representation
- `output/<device>/neff/<kernel>.neff` - Neuron Executable File Format

## Requirements

- **Development**: CMake 3.16+, C++17
- **Production**: AWS Neuron SDK with neuronx-cc compiler
- **Testing**: Mock XLA headers (included)
- **Docker**: Docker Engine for containerized builds

## GGML Backend Integration

This tool generates the NEFF files needed for a GGML Neuron backend. Each kernel corresponds to GGML operations used in llama.cpp.

Generated files can be directly integrated into your GGML backend by loading the appropriate NEFF files for your target device.

## License

[MIT License](LICENSE.md).


## Thank You and Feedback

Reach out to us for any feedback or contributions.

Now Enjoy!

* Author: Karthik Kumar Viswanathan
* Web   : https://karthikkumar.org
* Email : me@karthikkumar.org
