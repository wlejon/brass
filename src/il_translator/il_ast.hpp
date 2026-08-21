#pragma once

#include <string>
#include <vector>
#include <cstdint>

namespace brass::il {

enum class BronzeType {
    Void,
    Bool,
    I32,
    F64,
    Str,
    Dynamic,
    Unknown
};

enum class BronzeOp {
    ConstF64,
    ConstI32,
    ConstBool,
    ConstUndefined,
    ConstNull,
    ConstBigInt,
    Add,
    Sub,
    Neg,
    Mul,
    Div,
    Mod,
    Pow,
    BitAnd,
    BitOr,
    BitXor,
    Shl,
    Shr,
    UShr,
    BitNot,
    ToInt32,
    ToNumeric,
    NumericStep,
    CmpLt,
    CmpGt,
    CmpLe,
    CmpGe,
    CmpEq,
    CmpNe,
    StrictEq,
    LooseEq,
    RelLt,
    RelGt,
    RelLe,
    RelGe,
    NumTruthy,
    TypeOf,
    ToStr,
    Box,
    Unbox,
    Call,
    CallDynamic,
    NameResolve,
    EnvCreate,
    EnvGet,
    EnvSet,
    EnvGetTdz,
    EnvInitTdz,
    CreateFunc,
    CreateArray,
    PropSet,
    ElemGet,
    ElemSet,
    Print,
    PrintErr,
    Ret,
    Jump,
    Branch,
    Unknown
};

struct BronzeBlockTarget {
    uint32_t block_id = 0;
    std::vector<uint32_t> args;
};

struct BronzeInstruction {
    uint32_t result_id = UINT32_MAX;
    BronzeType result_type = BronzeType::Void;
    BronzeOp op = BronzeOp::Unknown;
    std::vector<uint32_t> operands;

    double imm_f64 = 0.0;
    int64_t imm_i64 = 0;
    bool imm_bool = false;
    BronzeType box_type = BronzeType::Unknown;
    bool raw_unbox = false;
    std::string callee_name;
    std::string string_literal;

    uint32_t depth = 0;
    uint32_t index = 0;
    uint32_t param_count = 0;

    BronzeBlockTarget target;
    BronzeBlockTarget else_target;
};

struct BronzeBlock {
    uint32_t id = 0;
    std::vector<std::pair<uint32_t, BronzeType>> params;
    uint32_t handler_id = UINT32_MAX;
    std::vector<BronzeInstruction> instructions;
};

struct BronzeFunction {
    std::string name;
    std::vector<std::pair<uint32_t, BronzeType>> params;
    BronzeType return_type = BronzeType::Void;
    bool is_exported = false;
    std::vector<BronzeBlock> blocks;
};

struct BronzeModuleAST {
    std::string name;
    std::vector<BronzeFunction> functions;
};

const char* bronze_type_name(BronzeType t);
const char* bronze_op_name(BronzeOp op);

} // namespace brass::il
