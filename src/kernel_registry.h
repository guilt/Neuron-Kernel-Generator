#pragma once
#include "tensorflow/compiler/xla/service/hlo_module.h"

#include <functional>
#include <map>
#include <string>
#include <vector>

using KernelBuilder = std::function<std::unique_ptr<xla::HloModule>(
    const std::vector<int64_t>&, const std::string&)>;

class KernelRegistry
{
  private:
    static std::map<std::string, KernelBuilder> kernels_;
    static bool initialized_;

  public:
    static void initialize();
    static void register_kernel(const std::string& name,
                                KernelBuilder builder);
    static bool generate_kernel(const std::string& name,
                                const std::vector<int64_t>& shape,
                                const std::string& output_name);
    static void print_available_kernels();
    static void write_hlo_and_compile(std::unique_ptr<xla::HloModule> module,
                                      const std::string& output_name);
};
