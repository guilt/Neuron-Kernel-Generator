#pragma once
#include <memory>
#include <sstream>
#include <string>
#include <vector>

// Mock XLA classes for testing without full TensorFlow installation
namespace xla
{

enum class PrimitiveType
{
    F32
};
enum class HloOpcode
{
    kAdd,
    kMultiply,
    kSubtract,
    kMaximum,
    kTanh,
    kLogistic,
    kExp,
    kDivide
};

constexpr PrimitiveType F32 = PrimitiveType::F32;

class Shape
{
  public:
    Shape(PrimitiveType type, const std::vector<int64_t>& dims) :
        type_(type), dims_(dims)
    {
    }
    PrimitiveType element_type() const
    {
        return type_;
    }
    const std::vector<int64_t>& dimensions() const
    {
        return dims_;
    }

    std::string ToString() const
    {
        std::stringstream ss;
        ss << "f32[";
        for (size_t i = 0; i < dims_.size(); ++i)
        {
            if (i > 0)
                ss << ",";
            ss << dims_[i];
        }
        ss << "]";
        return ss.str();
    }

  private:
    PrimitiveType type_;
    std::vector<int64_t> dims_;
};

class ShapeUtil
{
  public:
    static Shape MakeShape(PrimitiveType type,
                           const std::vector<int64_t>& dims)
    {
        return Shape(type, dims);
    }
};

class Literal
{
  public:
    static Literal Zero(PrimitiveType)
    {
        return Literal();
    }
    static Literal MinValue(PrimitiveType)
    {
        return Literal();
    }
    static Literal CreateR0(float)
    {
        return Literal();
    }
};

class LiteralUtil
{
  public:
    static Literal Zero(PrimitiveType type)
    {
        return Literal::Zero(type);
    }
    static Literal MinValue(PrimitiveType type)
    {
        return Literal::MinValue(type);
    }
    template <typename T> static Literal CreateR0(T val)
    {
        return Literal::CreateR0(val);
    }
};

// Global instruction counter for unique IDs
static int instruction_id = 0;

class HloInstruction
{
  private:
    int id_;
    std::string op_name_;
    Shape shape_;
    std::vector<HloInstruction*> operands_;

  public:
    HloInstruction(const std::string& op, const Shape& shape) :
        id_(++instruction_id), op_name_(op), shape_(shape)
    {
    }

    static std::unique_ptr<HloInstruction> CreateParameter(
        int64_t param_num, const Shape& shape, const std::string& name)
    {
        auto inst = std::make_unique<HloInstruction>("parameter", shape);
        inst->op_name_ = "parameter(" + std::to_string(param_num) + ")";
        return inst;
    }

    static std::unique_ptr<HloInstruction> CreateConstant(const Literal&)
    {
        return std::make_unique<HloInstruction>("constant", Shape(F32, {}));
    }

    static std::unique_ptr<HloInstruction> CreateBinary(const Shape& shape,
                                                        HloOpcode op,
                                                        HloInstruction* lhs,
                                                        HloInstruction* rhs)
    {
        std::string op_str;
        switch (op)
        {
            case HloOpcode::kAdd:
                op_str = "add";
                break;
            case HloOpcode::kMultiply:
                op_str = "multiply";
                break;
            case HloOpcode::kSubtract:
                op_str = "subtract";
                break;
            case HloOpcode::kMaximum:
                op_str = "maximum";
                break;
            case HloOpcode::kDivide:
                op_str = "divide";
                break;
            default:
                op_str = "binary_op";
                break;
        }
        auto inst = std::make_unique<HloInstruction>(op_str, shape);
        inst->operands_ = {lhs, rhs};
        return inst;
    }

    static std::unique_ptr<HloInstruction> CreateUnary(
        const Shape& shape, HloOpcode op, HloInstruction* operand)
    {
        std::string op_str;
        switch (op)
        {
            case HloOpcode::kTanh:
                op_str = "tanh";
                break;
            case HloOpcode::kLogistic:
                op_str = "logistic";
                break;
            case HloOpcode::kExp:
                op_str = "exponential";
                break;
            default:
                op_str = "unary_op";
                break;
        }
        auto inst = std::make_unique<HloInstruction>(op_str, shape);
        inst->operands_ = {operand};
        return inst;
    }

    static std::unique_ptr<HloInstruction> CreateBroadcast(
        const Shape& shape, HloInstruction* operand,
        const std::vector<int64_t>&)
    {
        auto inst = std::make_unique<HloInstruction>("broadcast", shape);
        inst->operands_ = {operand};
        return inst;
    }

    static std::unique_ptr<HloInstruction> CreateTranspose(
        const Shape& shape, HloInstruction* operand,
        const std::vector<int64_t>&)
    {
        auto inst = std::make_unique<HloInstruction>("transpose", shape);
        inst->operands_ = {operand};
        return inst;
    }

    static std::unique_ptr<HloInstruction> CreateReshape(
        const Shape& shape, HloInstruction* operand)
    {
        auto inst = std::make_unique<HloInstruction>("reshape", shape);
        inst->operands_ = {operand};
        return inst;
    }

    static std::unique_ptr<HloInstruction> CreateDot(
        const Shape& shape, HloInstruction* lhs, HloInstruction* rhs,
        const struct DotDimensionNumbers&, void*)
    {
        auto inst = std::make_unique<HloInstruction>("dot", shape);
        inst->operands_ = {lhs, rhs};
        return inst;
    }

    static std::unique_ptr<HloInstruction> CreateReduce(
        const Shape& shape, HloInstruction* operand, HloInstruction* init,
        const std::vector<int64_t>&, HloOpcode)
    {
        auto inst = std::make_unique<HloInstruction>("reduce", shape);
        inst->operands_ = {operand, init};
        return inst;
    }

    std::string ToString() const
    {
        std::stringstream ss;
        ss << "  %" << id_ << " = " << shape_.ToString() << " " << op_name_;
        if (!operands_.empty())
        {
            ss << "(";
            for (size_t i = 0; i < operands_.size(); ++i)
            {
                if (i > 0)
                    ss << ", ";
                ss << "%" << operands_[i]->id_;
            }
            ss << ")";
        }
        return ss.str();
    }

    int id() const
    {
        return id_;
    }
};

struct DotDimensionNumbers
{
    void add_lhs_contracting_dimensions(int64_t)
    {
    }
    void add_rhs_contracting_dimensions(int64_t)
    {
    }
};

class HloComputation
{
  private:
    std::vector<std::unique_ptr<HloInstruction>> instructions_;
    HloInstruction* root_;
    std::string name_;

  public:
    HloComputation(const std::string& name) : root_(nullptr), name_(name)
    {
    }

    class Builder
    {
        std::string name_;
        std::vector<std::unique_ptr<HloInstruction>> instructions_;

      public:
        Builder(const std::string& name) : name_(name)
        {
        }

        HloInstruction* AddInstruction(std::unique_ptr<HloInstruction> inst)
        {
            auto* ptr = inst.get();
            instructions_.push_back(std::move(inst));
            return ptr;
        }

        std::unique_ptr<HloComputation> Build(HloInstruction* root)
        {
            auto comp = std::make_unique<HloComputation>(name_);
            comp->instructions_ = std::move(instructions_);
            comp->root_ = root;
            return comp;
        }
    };

    std::string ToString() const
    {
        std::stringstream ss;
        ss << "ENTRY " << name_ << " {\n";
        for (const auto& inst : instructions_)
        {
            ss << inst->ToString() << "\n";
        }
        ss << "  ROOT %" << (root_ ? root_->id() : 0) << "\n";
        ss << "}";
        return ss.str();
    }
};

class HloModuleConfig
{
};

class HloModule
{
    std::string name_;
    std::unique_ptr<HloComputation> entry_computation_;

  public:
    HloModule(const std::string& name, const HloModuleConfig&) :
        name_(name)
    {
    }

    void AddEntryComputation(std::unique_ptr<HloComputation> comp)
    {
        entry_computation_ = std::move(comp);
    }

    std::string ToString() const
    {
        std::stringstream ss;
        ss << "HloModule " << name_ << "\n\n";
        if (entry_computation_)
        {
            ss << entry_computation_->ToString();
        }
        ss << "\n";
        return ss.str();
    }
};

} // namespace xla
