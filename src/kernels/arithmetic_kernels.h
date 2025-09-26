#pragma once
#include "../base_kernel.h"

class AddKernel : public BaseKernel
{
  public:
    AddKernel(const std::vector<int64_t>& shape) : BaseKernel(shape, "add")
    {
    }

  protected:
    std::unique_ptr<xla::HloModule> build_hlo() override;
};

class MulKernel : public BaseKernel
{
  public:
    MulKernel(const std::vector<int64_t>& shape) : BaseKernel(shape, "mul")
    {
    }

  protected:
    std::unique_ptr<xla::HloModule> build_hlo() override;
};

class SubKernel : public BaseKernel
{
  public:
    SubKernel(const std::vector<int64_t>& shape) : BaseKernel(shape, "sub")
    {
    }

  protected:
    std::unique_ptr<xla::HloModule> build_hlo() override;
};
