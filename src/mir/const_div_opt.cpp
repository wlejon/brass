#include "const_div_opt.hpp"

#include <brass/codegen/div_magic.hpp>
#include <brass/mir/builder.hpp>
#include <bit>

namespace brass::detail {

namespace {

Value* build_unsigned_magic_i32_mir(Builder& b, Value* lhs, uint32_t d, bool is_rem) {
    const codegen::UnsignedDivMagic32 mg = codegen::unsigned_div_magic32(d);
    Value* x = lhs;
    if (mg.pre_shift > 0u) {
        x = b.build_lshr(x, b.build_iconst_i32(static_cast<int32_t>(mg.pre_shift)));
    }
    Value* x64 = b.build_zext_i64(x);
    Value* m64 = b.build_iconst_i64(static_cast<int64_t>(static_cast<uint64_t>(mg.multiplier)));
    Value* prod64 = b.build_mul(x64, m64);
    Value* q = nullptr;
    if (!mg.needs_add) {
        Value* shifted = b.build_lshr(prod64, b.build_iconst_i64(static_cast<int64_t>(32u + mg.post_shift)));
        q = b.build_trunc_i32(shifted);
    } else {
        Value* hi = b.build_trunc_i32(b.build_lshr(prod64, b.build_iconst_i64(32)));
        Value* diff = b.build_sub(x, hi);
        Value* half = b.build_lshr(diff, b.build_iconst_i32(1));
        Value* sum = b.build_add(half, hi);
        q = (mg.post_shift > 1u)
            ? b.build_lshr(sum, b.build_iconst_i32(static_cast<int32_t>(mg.post_shift - 1u)))
            : sum;
    }
    if (!is_rem) return q;
    Value* prod = b.build_mul(q, b.build_iconst_i32(static_cast<int32_t>(d)));
    return b.build_sub(lhs, prod);
}

Value* build_signed_magic_i32_mir(Builder& b, Value* lhs, int32_t d, bool is_rem) {
    const codegen::SignedDivMagic32 mg = codegen::signed_div_magic32(d);
    Value* lhs64 = b.build_sext_i64(lhs);
    Value* m64 = b.build_iconst_i64(static_cast<int64_t>(mg.multiplier));
    Value* prod64 = b.build_mul(lhs64, m64);
    Value* q = b.build_trunc_i32(b.build_lshr(prod64, b.build_iconst_i64(32)));
    if (mg.multiplier < 0) {
        q = b.build_add(q, lhs);
    }
    if (mg.shift > 0) {
        q = b.build_ashr(q, b.build_iconst_i32(mg.shift));
    }
    Value* sign = b.build_lshr(lhs, b.build_iconst_i32(31));
    q = b.build_add(q, sign);
    if (!is_rem) return q;
    Value* prod = b.build_mul(q, b.build_iconst_i32(d));
    return b.build_sub(lhs, prod);
}

} // namespace

Value* try_lower_const_div_i32_mir(
    Module& mod,
    Function& fn,
    Instruction* cur,
    Value* lhs,
    Opcode op,
    int64_t div_imm,
    const std::function<bool()>& prove_lhs_non_neg) {

    if (!cur || !lhs || cur->type() != Type::i32() || lhs->type() != Type::i32()) {
        return nullptr;
    }

    const bool is_rem = (op == Opcode::umod || op == Opcode::smod);
    if (op == Opcode::udiv || op == Opcode::umod) {
        const uint32_t d = static_cast<uint32_t>(div_imm);
        if (!codegen::takes_unsigned_div_magic32(d)) return nullptr;
        Builder b(mod);
        b.set_function(&fn);
        b.position_before(cur);
        return build_unsigned_magic_i32_mir(b, lhs, d, is_rem);
    }

    if (op == Opcode::sdiv || op == Opcode::smod) {
        const int32_t sd = static_cast<int32_t>(div_imm);
        if (!codegen::takes_signed_div_magic32(sd)) return nullptr;
        Builder b(mod);
        b.set_function(&fn);
        b.position_before(cur);
        if (prove_lhs_non_neg && prove_lhs_non_neg()) {
            return build_unsigned_magic_i32_mir(b, lhs, static_cast<uint32_t>(sd), is_rem);
        }
        return build_signed_magic_i32_mir(b, lhs, sd, is_rem);
    }

    return nullptr;
}

} // namespace brass::detail
