#pragma once
#include "../base_kernel.h"

class ReLUKernel : public BaseKernel
{
  public:
    ReLUKernel(const std::vector<int64_t>& shape) :
        BaseKernel(shape, "relu")
    {
    }

  protected:
    std::unique_ptr<xla::HloModule> build_hlo() override;
};

class GELUKernel : public BaseKernel
{
  public:
    GELUKernel(const std::vector<int64_t>& shape) :
        BaseKernel(shape, "gelu")
    {
    }

  protected:
    std::unique_ptr<xla::HloModule> build_hlo() override;
};

class SiLUKernel : public BaseKernel
{
  public:
    SiLUKernel(const std::vector<int64_t>& shape) :
        BaseKernel(shape, "silu")
    {
    }

  protected:
    std::unique_ptr<xla::HloModule> build_hlo() override;
};

class SoftMaxKernel : public BaseKernel
{
  private:
    int64_t axis_;

  public:
    SoftMaxKernel(const std::vector<int64_t>& shape, int64_t axis = -1) :
        BaseKernel(shape, "softmax"), axis_(axis)
    {
    }

  protected:
    std::unique_ptr<xla::HloModule> build_hlo() override;
};
