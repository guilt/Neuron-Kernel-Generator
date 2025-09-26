#pragma once
#include "../base_kernel.h"

class MatMulKernel : public BaseKernel
{
  public:
    MatMulKernel(const std::vector<int64_t>& shape) :
        BaseKernel(shape, "matmul")
    {
    }

  protected:
    std::unique_ptr<xla::HloModule> build_hlo() override;
};

class TransposeKernel : public BaseKernel
{
  private:
    std::vector<int64_t> permutation_;

  public:
    TransposeKernel(const std::vector<int64_t>& shape,
                    const std::vector<int64_t>& perm = {1, 0}) :
        BaseKernel(shape, "transpose"), permutation_(perm)
    {
    }

  protected:
    std::unique_ptr<xla::HloModule> build_hlo() override;
};

class ReshapeKernel : public BaseKernel
{
  private:
    std::vector<int64_t> new_shape_;

  public:
    ReshapeKernel(const std::vector<int64_t>& shape,
                  const std::vector<int64_t>& new_shape) :
        BaseKernel(shape, "reshape"), new_shape_(new_shape)
    {
    }

  protected:
    std::unique_ptr<xla::HloModule> build_hlo() override;
};
