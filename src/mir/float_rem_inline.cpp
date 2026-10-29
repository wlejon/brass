#include <brass/mir/float_rem_inline.hpp>
#include <brass/mir/block.hpp>
#include <brass/mir/builder.hpp>
#include <brass/mir/runtime_symbols.hpp>
#include <brass/mir/uses.hpp>
#include <cmath>
#include <cstdint>
#include <string>
#include <vector>

namespace brass {

namespace {

// `v` as an i64 and whether that is exactly `v`: a conversion from an
// integer is exact by construction and a constant is decided here, so only
// a double of unknown origin pays the round trip. The round trip compares
// BITS, an integer compare the branch fuses with, where an f64 `eq` is
// materialized from two flags: a NaN fails it, an out-of-range double
// (converted to INT64_MIN, back to -2^63) fails it, and so does -0, whose
// bits are not +0's — a -0 operand takes the call. On a target whose
// conversion saturates (arm64's fcvtzs), 2^63 converts to INT64_MAX, which
// converts back to 2^63 and would pass; INT64_MAX is no double's exact value,
// so excluding it refuses exactly that case.
struct IntOperand {
    Value* i64 = nullptr;
    Value* ok = nullptr;  // null: always exact
    bool is_const = false;
    int64_t value = 0;
};

// Whether `v` is a constant the integer path can never take: not an integer
// a 64-bit integer holds (a fraction, a NaN, an infinity, 2^63 or past it).
bool const_refused(const Value* v) {
    const Instruction* def = v->is_instruction() ? v->defining_instruction() : nullptr;
    if (!def || def->opcode() != Opcode::fconst_f64) return false;
    const double c = def->imm_f64();
    // The i64 range is [-2^63, 2^63); the compares are false for a NaN.
    const bool in_range = c >= -9223372036854775808.0 && c < 9223372036854775808.0;
    return !in_range || std::floor(c) != c;
}

// Fills `out` for an operand const_refused passed.
void operand_int(Builder& b, Value* v, IntOperand& out) {
    if (const Instruction* def = v->is_instruction() ? v->defining_instruction() : nullptr) {
        if (def->opcode() == Opcode::sitofp_f64_i32 && def->operand_count() == 1) {
            out.i64 = b.build_sext_i64(def->operand(0));
            return;
        }
        if (def->opcode() == Opcode::sitofp_f64_i64 && def->operand_count() == 1) {
            out.i64 = def->operand(0);
            return;
        }
        if (def->opcode() == Opcode::fconst_f64) {
            out.is_const = true;
            out.value = static_cast<int64_t>(def->imm_f64());
            out.i64 = b.build_iconst_i64(out.value);
            return;
        }
    }
    out.i64 = b.build_fptosi_i64(v);
    Value* same_bits =
        b.build_eq(b.build_bitcast_i64_f64(b.build_sitofp_f64_i64(out.i64)), b.build_bitcast_i64_f64(v));
    out.ok = b.build_and(same_bits, b.build_ne(out.i64, b.build_iconst_i64(INT64_MAX)));
}

bool is_frem_call(const Instruction* inst) {
    return inst && inst->opcode() == Opcode::call && inst->operand_count() == 2 && inst->type() == Type::f64() &&
           inst->result() && inst->operand(0)->type() == Type::f64() && inst->operand(1)->type() == Type::f64() &&
           callee_has_role(*inst, SymbolRole::FloatRem);
}

}  // namespace

size_t inline_float_rem(Function& fn) {
    std::vector<Instruction*> sites;
    for (BasicBlock* bb : fn.blocks()) {
        if (!bb) continue;
        for (Instruction* inst : *bb) {
            if (!is_frem_call(inst)) continue;
            // A constant operand the integer path can never take keeps the
            // plain call: a divisor of 0 or -1 (-0 compares equal to 0) as
            // well as any operand const_refused names.
            if (const_refused(inst->operand(0)) || const_refused(inst->operand(1))) continue;
            const Instruction* ddef =
                inst->operand(1)->is_instruction() ? inst->operand(1)->defining_instruction() : nullptr;
            if (ddef && ddef->opcode() == Opcode::fconst_f64 && (ddef->imm_f64() == 0.0 || ddef->imm_f64() == -1.0)) {
                continue;
            }
            sites.push_back(inst);
        }
    }
    if (sites.empty()) return 0;

    Builder b(fn);
    for (Instruction* call : sites) {
        BasicBlock* bb = call->parent();
        const std::string prefix = "frem_" + std::to_string(fn.next_block_id());
        BasicBlock* fast = b.append_block(prefix + "_int");
        BasicBlock* slow = b.append_block(prefix + "_cold");
        BasicBlock* cont = b.append_block(prefix + "_cont");
        while (Instruction* moved = call->next()) {
            bb->remove_instruction(moved);
            cont->append_instruction(moved);
        }
        Value* result = b.add_block_param(cont, Type::f64());
        replace_all_uses(fn, call->result(), result);
        bb->remove_instruction(call);
        slow->append_instruction(call);

        Value* a = call->operand(0);
        Value* d = call->operand(1);
        b.position_at_end(bb);
        IntOperand ai, di;
        operand_int(b, a, ai);
        operand_int(b, d, di);
        // srem traps on a zero divisor and on INT64_MIN / -1; fmod answers
        // NaN for the first, and -1 is cheap enough to leave to the call.
        // (d + 1) >u 1 is d not in {0, -1}.
        Value* ok = nullptr;
        auto conj = [&](Value* c) { ok = ok ? b.build_and(ok, c) : c; };
        if (ai.ok) conj(ai.ok);
        if (di.ok) conj(di.ok);
        if (!di.is_const) conj(b.build_ugt(b.build_add(di.i64, b.build_iconst_i64(1)), b.build_iconst_i64(1)));
        if (ok) {
            b.build_br_if(ok, fast, slow);
        } else {
            b.build_br(fast);
        }

        // The integer remainder is fmod's exact value, so its conversion is
        // exact, and it already has the dividend's sign unless it is zero;
        // OR-ing the dividend's sign bit in makes a zero -0 for a negative
        // dividend, as fmod does, and changes nothing else.
        b.position_at_end(fast);
        Value* rem = b.build_smod(ai.i64, di.i64);
        Value* rem_bits = b.build_bitcast_i64_f64(b.build_sitofp_f64_i64(rem));
        Value* sign = b.build_and(b.build_bitcast_i64_f64(a), b.build_iconst_i64(INT64_MIN));
        b.build_br(cont, {b.build_bitcast_f64_i64(b.build_or(rem_bits, sign))});

        b.position_at_end(slow);
        b.build_br(cont, {call->result()});
    }
    fn.rebuild_cfg_predecessors();
    return sites.size();
}

}  // namespace brass
