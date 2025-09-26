#include "kernel_registry.h"

#include "kernels/activation_kernels.h"
#include "kernels/arithmetic_kernels.h"
#include "kernels/attention_kernels.h"
#include "kernels/comparison_kernels.h"
#include "kernels/matrix_kernels.h"
#include "kernels/normalization_kernels.h"
#include "kernels/quantization_kernels.h"
#include "kernels/reduction_kernels.h"
#include "kernels/tensor_kernels.h"

#include <cstdlib>
#include <fstream>
#include <iostream>

std::map<std::string, KernelBuilder> KernelRegistry::kernels_;
bool KernelRegistry::initialized_ = false;

void KernelRegistry::initialize()
{
    if (initialized_)
        return;

    ArithmeticKernels::register_kernels();
    MatrixKernels::register_kernels();
    ActivationKernels::register_kernels();
    AttentionKernels::register_kernels();
    TensorKernels::register_kernels();
    ReductionKernels::register_kernels();
    QuantizationKernels::register_kernels();
    ComparisonKernels::register_kernels();
    NormalizationKernels::register_kernels();

    initialized_ = true;
}

void KernelRegistry::register_kernel(const std::string& name,
                                     KernelBuilder builder)
{
    kernels_[name] = builder;
}

bool KernelRegistry::generate_kernel(const std::string& name,
                                     const std::vector<int64_t>& shape,
                                     const std::string& output_name)
{
    initialize();

    auto it = kernels_.find(name);
    if (it == kernels_.end())
    {
        return false;
    }

    auto module = it->second(shape, output_name);
    if (!module)
    {
        std::cerr << "Failed to build kernel: " << name << std::endl;
        return false;
    }

    write_hlo_and_compile(std::move(module), output_name);
    return true;
}

void KernelRegistry::print_available_kernels()
{
    initialize();

    std::cout << "Available kernels:" << std::endl;
    for (const auto& [name, _] : kernels_)
    {
        std::cout << "  " << name << std::endl;
    }
}

void KernelRegistry::write_hlo_and_compile(
    std::unique_ptr<xla::HloModule> module, const std::string& output_name)
{
    // Write HLO file
    std::string hlo_filename = output_name + ".hlo";
    std::ofstream hlo_file(hlo_filename);
    hlo_file << module->ToString();
    hlo_file.close();

    // Compile to NEFF
    std::string compile_cmd = "neuronx-cc compile " + hlo_filename +
        " --target trn1 -o " + output_name + ".neff";
    int result_code = system(compile_cmd.c_str());

    if (result_code == 0)
    {
        std::cout << "Generated: " << hlo_filename << " and "
                  << output_name << ".neff" << std::endl;
    }
    else
    {
        std::cout << "HLO generated: " << hlo_filename
                  << " (NEFF compilation failed)" << std::endl;
    }
}
