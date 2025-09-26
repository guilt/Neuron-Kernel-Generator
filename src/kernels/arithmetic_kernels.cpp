#include "arithmetic_kernels.h"
#ifdef USE_MOCK_XLA
#include "../mock_xla.h"
#else
#include "tensorflow/compiler/xla/service/hlo_computation.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/shape_util.h"
#endif

std::unique_ptr<xla::HloModule> AddKernel::build_hlo()
{
    auto module =
        std::make_unique<xla::HloModule>(name_, xla::HloModuleConfig{});
    auto builder = xla::HloComputation::Builder(name_ + "_computation");

    xla::Shape tensor_shape = xla::ShapeUtil::MakeShape(xla::F32, shape_);

    auto param_a = builder.AddInstruction(
        xla::HloInstruction::CreateParameter(0, tensor_shape, "a"));
    auto param_b = builder.AddInstruction(
        xla::HloInstruction::CreateParameter(1, tensor_shape, "b"));
    auto result = builder.AddInstruction(xla::HloInstruction::CreateBinary(
        tensor_shape, xla::HloOpcode::kAdd, param_a, param_b));

    auto computation = builder.Build(result);
    module->AddEntryComputation(std::move(computation));
    return module;
}

std::unique_ptr<xla::HloModule> MulKernel::build_hlo()
{
    auto module =
        std::make_unique<xla::HloModule>(name_, xla::HloModuleConfig{});
    auto builder = xla::HloComputation::Builder(name_ + "_computation");

    xla::Shape tensor_shape = xla::ShapeUtil::MakeShape(xla::F32, shape_);

    auto param_a = builder.AddInstruction(
        xla::HloInstruction::CreateParameter(0, tensor_shape, "a"));
    auto param_b = builder.AddInstruction(
        xla::HloInstruction::CreateParameter(1, tensor_shape, "b"));
    auto result = builder.AddInstruction(xla::HloInstruction::CreateBinary(
        tensor_shape, xla::HloOpcode::kMultiply, param_a, param_b));

    auto computation = builder.Build(result);
    module->AddEntryComputation(std::move(computation));
    return module;
}

std::unique_ptr<xla::HloModule> SubKernel::build_hlo()
{
    auto module =
        std::make_unique<xla::HloModule>(name_, xla::HloModuleConfig{});
    auto builder = xla::HloComputation::Builder(name_ + "_computation");

    xla::Shape tensor_shape = xla::ShapeUtil::MakeShape(xla::F32, shape_);

    auto param_a = builder.AddInstruction(
        xla::HloInstruction::CreateParameter(0, tensor_shape, "a"));
    auto param_b = builder.AddInstruction(
        xla::HloInstruction::CreateParameter(1, tensor_shape, "b"));
    auto result = builder.AddInstruction(xla::HloInstruction::CreateBinary(
        tensor_shape, xla::HloOpcode::kSubtract, param_a, param_b));

    auto computation = builder.Build(result);
    module->AddEntryComputation(std::move(computation));
    return module;
}
