#pragma once

#include <brass/mir/sccp.hpp>
#include <brass/mir/opcodes.hpp>

namespace brass {

LatticeValue evaluate_unary(Opcode op, Type res_type, const LatticeValue& val);
LatticeValue evaluate_binary(Opcode op, Type res_type, const LatticeValue& lhs, const LatticeValue& rhs);
LatticeValue evaluate_conversion(Opcode op, Type res_type, const LatticeValue& val);
LatticeValue evaluate_select(const LatticeValue& cond, const LatticeValue& tv, const LatticeValue& fv);

} // namespace brass
