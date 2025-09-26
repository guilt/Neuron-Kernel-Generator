#include "matrix_kernels.h"
#ifdef USE_MOCK_XLA
#include "../mock_xla.h"
#else
#include "tensorflow/compiler/xla/service/hlo_computation.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/shape_util.h"
#endif

std::unique_ptr<xla::HloModule> MatMulKernel::build_hlo()
{
    auto module =
        std::make_unique<xla::HloModule>(name_, xla::HloModuleConfig{});
    auto builder = xla::HloComputation::Builder(name_ + "_computation");

    if (shape_.size() != 2)
    {
        return nullptr; // MatMul requires 2D tensors
    }

    xla::Shape shape_a =
        xla::ShapeUtil::MakeShape(xla::F32, {shape_[0], shape_[1]});
    xla::Shape shape_b =
        xla::ShapeUtil::MakeShape(xla::F32, {shape_[1], shape_[0]});
    xla::Shape result_shape =
        xla::ShapeUtil::MakeShape(xla::F32, {shape_[0], shape_[0]});

    auto param_a = builder.AddInstruction(
        xla::HloInstruction::CreateParameter(0, shape_a, "a"));
    auto param_b = builder.AddInstruction(
        xla::HloInstruction::CreateParameter(1, shape_b, "b"));

    xla::DotDimensionNumbers dot_dims;
    dot_dims.add_lhs_contracting_dimensions(1);
    dot_dims.add_rhs_contracting_dimensions(0);

    auto result = builder.AddInstruction(xla::HloInstruction::CreateDot(
        result_shape, param_a, param_b, dot_dims, nullptr));

    auto computation = builder.Build(result);
    module->AddEntryComputation(std::move(computation));
    return module;
}

std::unique_ptr<xla::HloModule> TransposeKernel::build_hlo()
{
    auto module =
        std::make_unique<xla::HloModule>(name_, xla::HloModuleConfig{});
    auto builder = xla::HloComputation::Builder(name_ + "_computation");

    xla::Shape input_shape = xla::ShapeUtil::MakeShape(xla::F32, shape_);
    auto param = builder.AddInstruction(
        xla::HloInstruction::CreateParameter(0, input_shape, "input"));

    auto result = builder.AddInstruction(xla::HloInstruction::CreateTranspose(
        xla::ShapeUtil::MakeShape(
            xla::F32, {shape_[permutation_[0]], shape_[permutation_[1]]}),
        param, permutation_));

    auto computation = builder.Build(result);
    module->AddEntryComputation(std::move(computation));
    return module;
}

std::unique_ptr<xla::HloModule> ReshapeKernel::build_hlo()
{
    auto module =
        std::make_unique<xla::HloModule>(name_, xla::HloModuleConfig{});
    auto builder = xla::HloComputation::Builder(name_ + "_computation");

    xla::Shape input_shape = xla::ShapeUtil::MakeShape(xla::F32, shape_);
    xla::Shape output_shape =
        xla::ShapeUtil::MakeShape(xla::F32, new_shape_);

    auto param = builder.AddInstruction(
        xla::HloInstruction::CreateParameter(0, input_shape, "input"));
    auto result = builder.AddInstruction(
        xla::HloInstruction::CreateReshape(output_shape, param));

    auto computation = builder.Build(result);
    module->AddEntryComputation(std::move(computation));
    return module;
}
