cmake_minimum_required(VERSION 3.16)
project(neuron-kernel-generator)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Option to use mock XLA for testing
option(USE_MOCK_XLA "Use mock XLA headers for testing" OFF)

# Add executable
add_executable(kernel-generator
    src/kernel-generator.cpp
    src/base_kernel.cpp
    src/kernels/arithmetic_kernels.cpp
    src/kernels/matrix_kernels.cpp
    src/kernels/activation_kernels.cpp
)

# Include directories
target_include_directories(kernel-generator PRIVATE src)

if(USE_MOCK_XLA)
    target_compile_definitions(kernel-generator PRIVATE USE_MOCK_XLA)
    message(STATUS "Using mock XLA headers for testing")
else()
    # Find and link with Neuron XLA libraries
    find_path(XLA_INCLUDE_DIR 
        NAMES tensorflow/compiler/xla/client/xla_builder.h
        PATHS 
            /opt/aws/neuron/lib/python*/site-packages/torch_neuronx/xla_impl/include
            /usr/local/lib/python*/site-packages/torch_neuronx/xla_impl/include
            /opt/conda/lib/python*/site-packages/torch_neuronx/xla_impl/include
	    /usr/local/include
            /opt/homebrew/include
        NO_DEFAULT_PATH
    )
    
    if(XLA_INCLUDE_DIR)
        target_include_directories(kernel-generator PRIVATE ${XLA_INCLUDE_DIR})
        message(STATUS "Found XLA headers at: ${XLA_INCLUDE_DIR}")
    else()
        message(WARNING "XLA headers not found, falling back to mock")
        target_compile_definitions(kernel-generator PRIVATE USE_MOCK_XLA)
    endif()
endif()

# Compiler flags
target_compile_options(kernel-generator PRIVATE -Wall -Wextra -O2)
