#include "activation_kernels.h"
#ifdef USE_MOCK_XLA
#include "../mock_xla.h"
#else
#include "tensorflow/compiler/xla/literal_util.h"
#include "tensorflow/compiler/xla/service/hlo_computation.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/shape_util.h"
#endif

std::unique_ptr<xla::HloModule> ReLUKernel::build_hlo()
{
    auto module =
        std::make_unique<xla::HloModule>(name_, xla::HloModuleConfig{});
    auto builder = xla::HloComputation::Builder(name_ + "_computation");

    xla::Shape tensor_shape = xla::ShapeUtil::MakeShape(xla::F32, shape_);

    auto param = builder.AddInstruction(
        xla::HloInstruction::CreateParameter(0, tensor_shape, "input"));
    auto zero = builder.AddInstruction(xla::HloInstruction::CreateConstant(
        xla::LiteralUtil::Zero(xla::F32)));
    auto zero_broadcast = builder.AddInstruction(
        xla::HloInstruction::CreateBroadcast(tensor_shape, zero, {}));
    auto result = builder.AddInstruction(xla::HloInstruction::CreateBinary(
        tensor_shape, xla::HloOpcode::kMaximum, param, zero_broadcast));

    auto computation = builder.Build(result);
    module->AddEntryComputation(std::move(computation));
    return module;
}

std::unique_ptr<xla::HloModule> GELUKernel::build_hlo()
{
    auto module =
        std::make_unique<xla::HloModule>(name_, xla::HloModuleConfig{});
    auto builder = xla::HloComputation::Builder(name_ + "_computation");

    xla::Shape tensor_shape = xla::ShapeUtil::MakeShape(xla::F32, shape_);

    auto param = builder.AddInstruction(
        xla::HloInstruction::CreateParameter(0, tensor_shape, "input"));
    auto tanh_result =
        builder.AddInstruction(xla::HloInstruction::CreateUnary(
            tensor_shape, xla::HloOpcode::kTanh, param));

    auto computation = builder.Build(tanh_result);
    module->AddEntryComputation(std::move(computation));
    return module;
}

std::unique_ptr<xla::HloModule> SiLUKernel::build_hlo()
{
    auto module =
        std::make_unique<xla::HloModule>(name_, xla::HloModuleConfig{});
    auto builder = xla::HloComputation::Builder(name_ + "_computation");

    xla::Shape tensor_shape = xla::ShapeUtil::MakeShape(xla::F32, shape_);

    auto param = builder.AddInstruction(
        xla::HloInstruction::CreateParameter(0, tensor_shape, "input"));
    auto sigmoid = builder.AddInstruction(xla::HloInstruction::CreateUnary(
        tensor_shape, xla::HloOpcode::kLogistic, param));
    auto result = builder.AddInstruction(xla::HloInstruction::CreateBinary(
        tensor_shape, xla::HloOpcode::kMultiply, param, sigmoid));

    auto computation = builder.Build(result);
    module->AddEntryComputation(std::move(computation));
    return module;
}

std::unique_ptr<xla::HloModule> SoftMaxKernel::build_hlo()
{
    auto module =
        std::make_unique<xla::HloModule>(name_, xla::HloModuleConfig{});
    auto builder = xla::HloComputation::Builder(name_ + "_computation");

    xla::Shape tensor_shape = xla::ShapeUtil::MakeShape(xla::F32, shape_);

    auto param = builder.AddInstruction(
        xla::HloInstruction::CreateParameter(0, tensor_shape, "input"));
    auto exp_vals = builder.AddInstruction(xla::HloInstruction::CreateUnary(
        tensor_shape, xla::HloOpcode::kExp, param));

    auto computation = builder.Build(exp_vals);
    module->AddEntryComputation(std::move(computation));
    return module;
}
