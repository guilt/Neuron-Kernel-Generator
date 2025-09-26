#pragma once
#ifdef USE_MOCK_XLA
#include "mock_xla.h"
#else
#include "tensorflow/compiler/xla/service/hlo_module.h"
#endif
#include <memory>
#include <string>
#include <vector>

class BaseKernel
{
  protected:
    std::vector<int64_t> shape_;
    std::string name_;

    virtual std::unique_ptr<xla::HloModule> build_hlo() = 0;
    void write_hlo_and_compile_neff(const std::string& output_name);

  public:
    BaseKernel(const std::vector<int64_t>& shape, const std::string& name) :
        shape_(shape), name_(name)
    {
    }

    virtual ~BaseKernel() = default;
    virtual void compile(const std::string& output_name);
};
