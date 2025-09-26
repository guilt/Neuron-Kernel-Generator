#include "kernels/activation_kernels.h"
#include "kernels/arithmetic_kernels.h"
#include "kernels/matrix_kernels.h"

#include <functional>
#include <iostream>
#include <map>
#include <memory>
#include <string>
#include <vector>

using KernelFactory =
    std::function<std::unique_ptr<BaseKernel>(const std::vector<int64_t>&)>;

std::map<std::string, KernelFactory> kernel_factories = {
    // Arithmetic
    {      "add",
     [](const std::vector<int64_t>& shape)
     {
     return std::make_unique<AddKernel>(shape);
     }             },
    {      "mul",
     [](const std::vector<int64_t>& shape)
     {
     return std::make_unique<MulKernel>(shape);
     }             },
    {      "sub",
     [](const std::vector<int64_t>& shape)
     {
     return std::make_unique<SubKernel>(shape);
     }             },

    // Matrix operations
    {   "matmul",
     [](const std::vector<int64_t>& shape)
     {
     return std::make_unique<MatMulKernel>(shape);
     }             },
    {"transpose",
     [](const std::vector<int64_t>& shape)
     {
     return std::make_unique<TransposeKernel>(shape);
     }             },
    {  "reshape",
     [](const std::vector<int64_t>& shape)
     {
     return std::make_unique<ReshapeKernel>(shape, shape);
     }             },

    // Activations
    {     "relu",
     [](const std::vector<int64_t>& shape)
     {
     return std::make_unique<ReLUKernel>(shape);
     }             },
    {     "gelu",
     [](const std::vector<int64_t>& shape)
     {
     return std::make_unique<GELUKernel>(shape);
     }             },
    {     "silu",
     [](const std::vector<int64_t>& shape)
     {
     return std::make_unique<SiLUKernel>(shape);
     }             },
    {  "softmax", [](const std::vector<int64_t>& shape)
 {
 return std::make_unique<SoftMaxKernel>(shape);
 }}
};

int main(int argc, char* argv[])
{
    if (argc < 2)
    {
        std::cout << "Usage: " << argv[0]
                  << " <kernel_type> [shape...] [output_name]" << std::endl;
        std::cout << "Available kernels: ";
        for (const auto& [name, _] : kernel_factories)
        {
            std::cout << name << " ";
        }
        std::cout << std::endl;
        return 1;
    }

    std::string kernel_name = argv[1];
    std::vector<int64_t> shape;
    std::string output_name = kernel_name + "_kernel";

    // Parse arguments
    for (int i = 2; i < argc; i++)
    {
        char* endptr;
        long val = strtol(argv[i], &endptr, 10);
        if (*endptr == '\0')
        {
            shape.push_back(val);
        }
        else
        {
            output_name = argv[i];
            break;
        }
    }

    if (shape.empty())
    {
        shape = {1024, 1024};
    }

    // Create and compile kernel
    auto factory_it = kernel_factories.find(kernel_name);
    if (factory_it == kernel_factories.end())
    {
        std::cerr << "Unknown kernel: " << kernel_name << std::endl;
        return 1;
    }

    auto kernel = factory_it->second(shape);
    kernel->compile(output_name);

    return 0;
}
