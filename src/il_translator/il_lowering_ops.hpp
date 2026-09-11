#pragma once

#include "il_ast.hpp"
#include <brass/mir/builder.hpp>
#include <brass/mir/function.hpp>
#include <unordered_map>

namespace brass::il {

class IlLowering;

bool is_ops_il_op(BronzeOp op);

bool lower_ops_instruction(
    IlLowering* lowering,
    const BronzeInstruction& inst_ast,
    Builder& b,
    Function* fn,
    std::unordered_map<uint32_t, Value*>& val_map,
    Value*& res_val
);

} // namespace brass::il
