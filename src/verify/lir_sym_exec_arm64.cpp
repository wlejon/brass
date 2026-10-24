#include "lir_sym_exec_arm64.hpp"
#include <brass/target/aarch64/aarch64_emit.hpp>
#include <algorithm>
#include <cstring>

namespace brass::verify {

using namespace brass::codegen;

void Arm64InitEnv::init_defaults(BVContext& ctx) {
    for (uint8_t i = 0; i < 31; ++i) {
        auto g = static_cast<aarch64::GPR>(i);
        gpr_init[i] = ctx.make_var(std::string(aarch64::to_string(g, aarch64::OperandSize::Xword)), 64, false);
    }
    gpr_init[31] = ctx.make_var("sp", 64, false);
    for (uint8_t i = 0; i < 32; ++i) {
        std::string vname = "v" + std::to_string(i);
        vec_lo_init[i] = ctx.make_var(vname + "_lo", 64, true);
        vec_hi_init[i] = ctx.make_var(vname + "_hi", 64, true);
    }
    nzcv_init.n = ctx.make_var("init_n", 1, false);
    nzcv_init.z = ctx.make_var("init_z", 1, false);
    nzcv_init.c = ctx.make_var("init_c", 1, false);
    nzcv_init.v = ctx.make_var("init_v", 1, false);
}

BVNodeId Arm64InitEnv::get_vreg_init(BVContext& ctx, uint32_t id) {
    auto [it, ins] = vreg_init.try_emplace(id, 0);
    if (ins) it->second = ctx.make_var("vreg" + std::to_string(id), 64, false);
    return it->second;
}

BVNodeId Arm64InitEnv::get_spill_init(BVContext& ctx, int32_t slot) {
    auto [it, ins] = spill_init.try_emplace(slot, 0);
    if (ins) it->second = ctx.make_var("spill_" + std::to_string(slot), 64, false);
    return it->second;
}

BVNodeId Arm64InitEnv::get_local_init(BVContext& ctx, int32_t off) {
    auto [it, ins] = local_init.try_emplace(off, 0);
    if (ins) it->second = ctx.make_var("local_" + std::to_string(off), 64, false);
    return it->second;
}

BVNodeId Arm64InitEnv::read_init_mem(BVContext& ctx, BVNodeId addr, uint32_t sz) {
    uint32_t bits = sz * 8;
    for (const auto& e : mem_reads) if (e.addr == addr && e.size == sz) return e.val;
    BVNodeId v = ctx.make_var("arm64mem_" + std::to_string(mem_reads.size()), bits, false);
    for (const auto& e : mem_reads) {
        if (e.size == sz) v = ctx.make_ite(ctx.make_binary(BVOp::Eq, addr, e.addr, 1), e.val, v);
    }
    mem_reads.push_back({addr, sz, v});
    return v;
}

BVNodeId eval_arm64_cond(BVContext& ctx, const Arm64NzcvState& f, aarch64::Condition cond) {
    using aarch64::Condition;
    if (f.from_subs) {
        switch (cond) {
            case Condition::EQ: return ctx.make_binary(BVOp::Eq,  f.subs_a, f.subs_b, 1);
            case Condition::NE: return ctx.make_binary(BVOp::Ne,  f.subs_a, f.subs_b, 1);
            case Condition::CS: return ctx.make_binary(BVOp::Uge, f.subs_a, f.subs_b, 1);
            case Condition::CC: return ctx.make_binary(BVOp::Ult, f.subs_a, f.subs_b, 1);
            case Condition::HI: return ctx.make_binary(BVOp::Ugt, f.subs_a, f.subs_b, 1);
            case Condition::LS: return ctx.make_binary(BVOp::Ule, f.subs_a, f.subs_b, 1);
            case Condition::GE: return ctx.make_binary(BVOp::Sge, f.subs_a, f.subs_b, 1);
            case Condition::LT: return ctx.make_binary(BVOp::Slt, f.subs_a, f.subs_b, 1);
            case Condition::GT: return ctx.make_binary(BVOp::Sgt, f.subs_a, f.subs_b, 1);
            case Condition::LE: return ctx.make_binary(BVOp::Sle, f.subs_a, f.subs_b, 1);
            default: break;
        }
    }
    switch (cond) {
        case Condition::EQ: return f.z;
        case Condition::NE: return ctx.make_unary(BVOp::Not, f.z, 1);
        case Condition::CS: return f.c;
        case Condition::CC: return ctx.make_unary(BVOp::Not, f.c, 1);
        case Condition::MI: return f.n;
        case Condition::PL: return ctx.make_unary(BVOp::Not, f.n, 1);
        case Condition::VS: return f.v;
        case Condition::VC: return ctx.make_unary(BVOp::Not, f.v, 1);
        case Condition::HI: return ctx.make_binary(BVOp::And, f.c, ctx.make_unary(BVOp::Not, f.z, 1), 1);
        case Condition::LS: return ctx.make_binary(BVOp::Or, ctx.make_unary(BVOp::Not, f.c, 1), f.z, 1);
        case Condition::GE: return ctx.make_unary(BVOp::Not, ctx.make_binary(BVOp::Xor, f.n, f.v, 1), 1);
        case Condition::LT: return ctx.make_binary(BVOp::Xor, f.n, f.v, 1);
        case Condition::GT:
            return ctx.make_binary(BVOp::And, ctx.make_unary(BVOp::Not, f.z, 1),
                                   ctx.make_unary(BVOp::Not, ctx.make_binary(BVOp::Xor, f.n, f.v, 1), 1), 1);
        case Condition::LE: return ctx.make_binary(BVOp::Or, f.z, ctx.make_binary(BVOp::Xor, f.n, f.v, 1), 1);
        case Condition::AL: case Condition::NV: return ctx.make_const(1, 1);
        default: return ctx.make_const(0, 1);
    }
}

void set_arm64_adds_flags(BVContext& ctx, Arm64NzcvState& f, BVNodeId a, BVNodeId b, BVNodeId res, uint32_t w) {
    f.n = ctx.make_extract(res, w - 1, w - 1);
    f.z = ctx.make_binary(BVOp::Eq, res, ctx.make_const(0, w), 1);
    f.c = ctx.make_binary(BVOp::UAddOvf, a, b, 1);
    f.v = ctx.make_binary(BVOp::SAddOvf, a, b, 1);
    f.from_subs = false;
}

void set_arm64_subs_flags(BVContext& ctx, Arm64NzcvState& f, BVNodeId a, BVNodeId b, BVNodeId res, uint32_t w) {
    f.n = ctx.make_extract(res, w - 1, w - 1);
    f.z = ctx.make_binary(BVOp::Eq, a, b, 1);
    // ARMv8-A SUBS / CMP sets Carry (C) to NOT borrow: C = 1 iff a >=_u b.
    f.c = ctx.make_binary(BVOp::Uge, a, b, 1);
    f.v = ctx.make_binary(BVOp::SSubOvf, a, b, 1);
    f.from_subs = true;
    f.subs_a = a;
    f.subs_b = b;
}

void set_arm64_ands_flags(BVContext& ctx, Arm64NzcvState& f, BVNodeId res, uint32_t w) {
    f.n = ctx.make_extract(res, w - 1, w - 1);
    f.z = ctx.make_binary(BVOp::Eq, res, ctx.make_const(0, w), 1);
    f.c = ctx.make_const(0, 1);
    f.v = ctx.make_const(0, 1);
    f.from_subs = false;
}

void set_arm64_fcmp_flags(BVContext& ctx, Arm64NzcvState& f, BVNodeId a, BVNodeId b) {
    uint32_t w = ctx.width(a);
    BVNodeId sa = ctx.make_extract(a, w - 1, w - 1);
    BVNodeId sb = ctx.make_extract(b, w - 1, w - 1);
    BVNodeId abs_a = ctx.make_zext(ctx.make_extract(a, w - 2, 0), 1);
    BVNodeId abs_b = ctx.make_zext(ctx.make_extract(b, w - 2, 0), 1);
    BVNodeId abs_lo = (a <= b) ? abs_a : abs_b;
    BVNodeId abs_hi = (a <= b) ? abs_b : abs_a;
    BVNodeId inf_node = ctx.make_const((w == 32) ? 0x7F800000ULL : 0x7FF0000000000000ULL, w);

    BVNodeId unord = ctx.make_binary(BVOp::Or,
        ctx.make_binary(BVOp::Ult, inf_node, abs_lo, 1),
        ctx.make_binary(BVOp::Ult, inf_node, abs_hi, 1), 1);
    BVNodeId not_unord = ctx.make_unary(BVOp::Not, unord, 1);

    BVNodeId abs_eq = ctx.make_binary(BVOp::Eq, abs_lo, abs_hi, 1);
    BVNodeId abs_lo_lt_hi = ctx.make_binary(BVOp::Ult, abs_lo, abs_hi, 1);
    BVNodeId abs_hi_lt_lo = ctx.make_unary(BVOp::Not, ctx.make_binary(BVOp::Or, abs_lo_lt_hi, abs_eq, 1), 1);
    BVNodeId abs_a_lt_b = (a <= b) ? abs_lo_lt_hi : abs_hi_lt_lo;
    BVNodeId abs_b_lt_a = (a <= b) ? abs_hi_lt_lo : abs_lo_lt_hi;

    BVNodeId diff_sign = ctx.make_binary(BVOp::Xor, sa, sb, 1);
    BVNodeId same_sign = ctx.make_unary(BVOp::Not, diff_sign, 1);
    BVNodeId both_zero = ctx.make_binary(BVOp::And, abs_eq,
        ctx.make_binary(BVOp::Eq, abs_lo, ctx.make_const(0, w), 1), 1);
    BVNodeId eq_raw = ctx.make_binary(BVOp::Or,
        ctx.make_binary(BVOp::And, abs_eq, same_sign, 1), both_zero, 1);

    BVNodeId same_sign_lt = ctx.make_ite(sa, abs_b_lt_a, abs_a_lt_b);
    BVNodeId diff_sign_lt = ctx.make_binary(BVOp::And, sa, ctx.make_unary(BVOp::Not, both_zero, 1), 1);
    BVNodeId lt_raw = ctx.make_ite(diff_sign, diff_sign_lt, same_sign_lt);

    // ARMv8-A FCMP: Unordered -> NZCV = 0011; Equal -> 0110; Less -> 1000; Greater -> 0010.
    f.n = ctx.make_binary(BVOp::And, not_unord, lt_raw, 1);
    f.z = ctx.make_binary(BVOp::And, not_unord, eq_raw, 1);
    f.c = ctx.make_unary(BVOp::Not, f.n, 1);
    f.v = unord;
    f.from_subs = false;
}

Arm64ExecState execute_arm64_block(BVContext& ctx, const LirBlock& block, Arm64InitEnv& env) {
    Arm64ExecState st;
    for (int i = 0; i < 32; ++i) {
        st.gpr[i] = env.gpr_init[i];
        st.vec_lo[i] = env.vec_lo_init[i];
        st.vec_hi[i] = env.vec_hi_init[i];
    }
    st.nzcv = env.nzcv_init;
    st.cond_branch_taken = ctx.make_const(0, 1);

    auto read_reg64 = [&](const LirOperand& op) -> BVNodeId {
        if (op.is_preg()) {
            if (op.preg_val.is_gpr()) {
                if (op.preg_val.code == static_cast<uint8_t>(aarch64::GPR::XZR)) return ctx.make_const(0, 64);
                if (op.preg_val.code < 32) return st.gpr[op.preg_val.code];
            }
            if (op.preg_val.is_xmm() && op.preg_val.code < 32) return st.vec_lo[op.preg_val.code];
        }
        if (op.is_vreg()) {
            auto [it, ins] = st.vregs.try_emplace(op.vreg_val.id, 0);
            if (ins) it->second = env.get_vreg_init(ctx, op.vreg_val.id);
            return it->second;
        }
        return ctx.make_const(0, 64);
    };

    auto eval_mem_addr = [&](const LirMem& m) -> BVNodeId {
        BVNodeId addr = ctx.make_const(0, 64);
        bool has_base_or_idx = false;
        if (m.base_preg.is_valid()) {
            addr = read_reg64(LirOperand::preg(m.base_preg, 8));
            has_base_or_idx = true;
        } else if (m.base_vreg.is_valid()) {
            addr = read_reg64(LirOperand::vreg(m.base_vreg, 8));
            has_base_or_idx = true;
        }
        if (m.index_preg.is_valid() || m.index_vreg.is_valid()) {
            BVNodeId idx = m.index_preg.is_valid()
                ? read_reg64(LirOperand::preg(m.index_preg, 8))
                : read_reg64(LirOperand::vreg(m.index_vreg, 8));
            uint8_t sc = x64::scale_value(m.scale);
            if (sc > 1) idx = ctx.make_binary(BVOp::Mul, idx, ctx.make_const(sc, 64), 64);
            addr = has_base_or_idx ? ctx.make_binary(BVOp::Add, addr, idx, 64) : idx;
            has_base_or_idx = true;
        }
        if (!has_base_or_idx) addr = st.gpr[static_cast<uint8_t>(aarch64::GPR::FP)];
        if (m.disp != 0) {
            addr = ctx.make_binary(BVOp::Add, addr,
                ctx.make_const(static_cast<uint64_t>(static_cast<int64_t>(m.disp)), 64), 64);
        }
        return addr;
    };

    auto read_mem_sized = [&](BVNodeId addr, uint32_t sz) -> BVNodeId {
        uint32_t bits = sz * 8;
        BVNodeId val = env.read_init_mem(ctx, addr, sz);
        for (const auto& s : st.mem_stores) {
            if (s.size == sz) {
                BVNodeId hit = ctx.make_binary(BVOp::Eq, addr, s.addr, 1);
                BVNodeId sv = ctx.width(s.val) == bits ? s.val : ctx.make_extract(s.val, bits - 1, 0);
                val = ctx.make_ite(hit, sv, val);
            } else if (s.size > sz && s.addr == addr) {
                val = ctx.make_extract(s.val, bits - 1, 0);
            }
        }
        return val;
    };

    auto read_op = [&](const LirOperand& op, uint32_t width_bits) -> BVNodeId {
        BVNodeId raw64 = 0;
        if (op.is_preg() || op.is_vreg()) {
            raw64 = read_reg64(op);
        } else if (op.is_imm_int()) {
            return ctx.make_const(static_cast<uint64_t>(op.imm_int), width_bits);
        } else if (op.is_imm_float()) {
            uint64_t u = 0;
            std::memcpy(&u, &op.imm_float, 8);
            return ctx.make_const(u, width_bits);
        } else if (op.is_spill_slot()) {
            auto it = st.spill_slots.find(op.spill_slot);
            raw64 = (it != st.spill_slots.end()) ? it->second : env.get_spill_init(ctx, op.spill_slot);
        } else if (op.is_local_slot()) {
            auto it = st.local_slots.find(op.local_offset);
            raw64 = (it != st.local_slots.end()) ? it->second : env.get_local_init(ctx, op.local_offset);
        } else if (op.is_mem()) {
            return read_mem_sized(eval_mem_addr(op.mem_val), width_bits / 8);
        } else {
            return ctx.make_const(0, width_bits);
        }
        return (width_bits == 64) ? raw64 : ctx.make_extract(raw64, width_bits - 1, 0);
    };

    // Write to destination operand with exact ARMv8-A semantics:
    // - Any 32-bit W-register write (or narrow write) to a GPR unconditionally zero-extends to 64 bits!
    // - Writes to XZR (code 32) are discarded.
    // - Scalar S/D writes to V0..V31 zero the upper 64 bits of Vn.
    auto write_op = [&](const LirOperand& dst, BVNodeId val) {
        uint32_t w = ctx.width(val);
        BVNodeId v64 = (w == 64) ? val : ctx.make_zext(val, 64 - w);
        if (dst.is_preg()) {
            if (dst.preg_val.is_gpr()) {
                uint8_t c = dst.preg_val.code;
                if (c < 32 && c != static_cast<uint8_t>(aarch64::GPR::XZR)) st.gpr[c] = v64;
            } else if (dst.preg_val.is_xmm() && dst.preg_val.code < 32) {
                uint8_t c = dst.preg_val.code;
                st.vec_lo[c] = v64;
                st.vec_hi[c] = ctx.make_const(0, 64);
            }
        } else if (dst.is_vreg()) {
            st.vregs[dst.vreg_val.id] = v64;
        } else if (dst.is_spill_slot()) {
            st.spill_slots[dst.spill_slot] = v64;
        } else if (dst.is_local_slot()) {
            st.local_slots[dst.local_offset] = v64;
        } else if (dst.is_mem()) {
            BVNodeId addr = eval_mem_addr(dst.mem_val);
            uint32_t sz = w / 8;
            if (dst.size == 1 || dst.size == 2 || dst.size == 4 || dst.size == 8) {
                sz = dst.size;
                if (w > sz * 8) val = ctx.make_extract(val, sz * 8 - 1, 0);
                else if (w < sz * 8) val = ctx.make_zext(val, sz * 8 - w);
            }
            st.mem_stores.push_back({addr, sz, val});
        }
    };

    auto read_vec128 = [&](const LirOperand& op) -> std::pair<BVNodeId, BVNodeId> {
        st.uses_vec128 = true;
        if (op.is_preg() && op.preg_val.is_xmm() && op.preg_val.code < 32) {
            return {st.vec_lo[op.preg_val.code], st.vec_hi[op.preg_val.code]};
        }
        if (op.is_mem()) {
            BVNodeId addr_lo = eval_mem_addr(op.mem_val);
            BVNodeId addr_hi = ctx.make_binary(BVOp::Add, addr_lo, ctx.make_const(8, 64), 64);
            return {read_mem_sized(addr_lo, 8), read_mem_sized(addr_hi, 8)};
        }
        return {read_op(op, 64), ctx.make_const(0, 64)};
    };

    auto write_vec128 = [&](const LirOperand& dst, BVNodeId lo, BVNodeId hi) {
        st.uses_vec128 = true;
        if (dst.is_preg() && dst.preg_val.is_xmm() && dst.preg_val.code < 32) {
            st.vec_lo[dst.preg_val.code] = lo;
            st.vec_hi[dst.preg_val.code] = hi;
        } else if (dst.is_mem()) {
            BVNodeId addr_lo = eval_mem_addr(dst.mem_val);
            BVNodeId addr_hi = ctx.make_binary(BVOp::Add, addr_lo, ctx.make_const(8, 64), 64);
            st.mem_stores.push_back({addr_lo, 8, lo});
            st.mem_stores.push_back({addr_hi, 8, hi});
        } else {
            write_op(dst, lo);
        }
    };

    for (const auto& inst_ptr : block.instructions) {
        if (!inst_ptr) continue;
        const LirInst& inst = *inst_ptr;
        LirOpcode op = inst.opcode;

        auto get_bin_lhs = [&](uint32_t w) -> BVNodeId {
            if (inst.uses.size() >= 2) return read_op(inst.uses[0], w);
            if (!inst.defs.empty()) return read_op(inst.defs[0], w);
            return ctx.make_const(0, w);
        };
        auto get_bin_rhs = [&](uint32_t w) -> BVNodeId {
            return !inst.uses.empty() ? read_op(inst.uses.back(), w) : ctx.make_const(0, w);
        };

        switch (op) {
            case LirOpcode::Nop: case LirOpcode::KeepAlive:
            case LirOpcode::Cdq: case LirOpcode::Cqo: case LirOpcode::Ret:
                break;

            case LirOpcode::Mov: case LirOpcode::Movabs: {
                if (inst.defs.empty() || inst.uses.empty()) break;
                const auto &dst = inst.defs[0], &src = inst.uses[0];
                uint8_t sz = (dst.size == 1 || src.size == 1) ? 1
                           : (dst.size == 2 || src.size == 2) ? 2
                           : (dst.size == 4 || src.size == 4) ? 4 : 8;
                if ((op == LirOpcode::Mov && dst.is_preg() &&
                     (src.is_preg() || src.is_vreg() || src.is_imm_int() || src.is_imm_float())) ||
                    dst.is_spill_slot() || dst.is_local_slot() || dst.is_vreg()) {
                    sz = 8;
                }
                write_op(dst, read_op(src, static_cast<uint32_t>(sz) * 8));
                break;
            }

            case LirOpcode::Mov32: {
                if (inst.defs.empty() || inst.uses.empty()) break;
                const auto &dst = inst.defs[0], &src = inst.uses[0];
                uint8_t sz = 4;
                if (!src.is_preg() && !src.is_vreg() && !src.is_imm_int()) {
                    if (dst.size == 1 || src.size == 1) sz = 1;
                    else if (dst.size == 2 || src.size == 2) sz = 2;
                } else if (!dst.is_preg() && !dst.is_vreg() && (dst.size == 1 || dst.size == 2)) {
                    sz = dst.size;
                }
                write_op(dst, read_op(src, static_cast<uint32_t>(sz) * 8));
                break;
            }

            case LirOpcode::Movsxd:
                if (!inst.defs.empty() && !inst.uses.empty()) write_op(inst.defs[0], ctx.make_sext(read_op(inst.uses[0], 32), 32));
                break;
            case LirOpcode::Movsx8:
                if (!inst.defs.empty() && !inst.uses.empty()) write_op(inst.defs[0], ctx.make_sext(read_op(inst.uses[0], 8), 56));
                break;
            case LirOpcode::Movsx16:
                if (!inst.defs.empty() && !inst.uses.empty()) write_op(inst.defs[0], ctx.make_sext(read_op(inst.uses[0], 16), 48));
                break;
            case LirOpcode::Movzx8:
                if (!inst.defs.empty() && !inst.uses.empty()) write_op(inst.defs[0], ctx.make_zext(read_op(inst.uses[0], 8), 56));
                break;
            case LirOpcode::Movzx16:
                if (!inst.defs.empty() && !inst.uses.empty()) write_op(inst.defs[0], ctx.make_zext(read_op(inst.uses[0], 16), 48));
                break;

            // Non-flag-setting AArch64 ADD / SUB / MUL / DIV / logic / shifts leave NZCV untouched!
            case LirOpcode::Add: case LirOpcode::Add32:
            case LirOpcode::Sub: case LirOpcode::Sub32:
            case LirOpcode::Imul: case LirOpcode::Imul32: {
                uint32_t w = (op == LirOpcode::Add32 || op == LirOpcode::Sub32 || op == LirOpcode::Imul32) ? 32 : 64;
                BVOp bop = (op == LirOpcode::Add || op == LirOpcode::Add32) ? BVOp::Add
                         : (op == LirOpcode::Sub || op == LirOpcode::Sub32) ? BVOp::Sub : BVOp::Mul;
                write_op(inst.defs[0], ctx.make_binary(bop, get_bin_lhs(w), get_bin_rhs(w), w));
                break;
            }

            // Flag-setting ADDS / SUBS update NZCV deterministically.
            case LirOpcode::Adds: case LirOpcode::Adds32: {
                uint32_t w = (op == LirOpcode::Adds32) ? 32 : 64;
                BVNodeId a = get_bin_lhs(w), b = get_bin_rhs(w), res = ctx.make_binary(BVOp::Add, a, b, w);
                write_op(inst.defs[0], res);
                set_arm64_adds_flags(ctx, st.nzcv, a, b, res, w);
                break;
            }

            case LirOpcode::Subs: case LirOpcode::Subs32: {
                uint32_t w = (op == LirOpcode::Subs32) ? 32 : 64;
                BVNodeId a = get_bin_lhs(w), b = get_bin_rhs(w), res = ctx.make_binary(BVOp::Sub, a, b, w);
                write_op(inst.defs[0], res);
                set_arm64_subs_flags(ctx, st.nzcv, a, b, res, w);
                break;
            }

            case LirOpcode::Smulh: case LirOpcode::Umulh: {
                BVNodeId a = get_bin_lhs(64), b = get_bin_rhs(64);
                BVNodeId a_lo = ctx.make_zext(ctx.make_extract(a, 31, 0), 32);
                BVNodeId a_hi = (op == LirOpcode::Smulh) ? ctx.make_sext(ctx.make_extract(a, 63, 32), 32)
                                                         : ctx.make_zext(ctx.make_extract(a, 63, 32), 32);
                BVNodeId b_lo = ctx.make_zext(ctx.make_extract(b, 31, 0), 32);
                BVNodeId b_hi = (op == LirOpcode::Smulh) ? ctx.make_sext(ctx.make_extract(b, 63, 32), 32)
                                                         : ctx.make_zext(ctx.make_extract(b, 63, 32), 32);
                BVNodeId p0 = ctx.make_binary(BVOp::Mul, a_lo, b_lo, 64);
                BVNodeId p1 = ctx.make_binary(BVOp::Mul, a_lo, b_hi, 64);
                BVNodeId p2 = ctx.make_binary(BVOp::Mul, a_hi, b_lo, 64);
                BVNodeId p3 = ctx.make_binary(BVOp::Mul, a_hi, b_hi, 64);
                BVNodeId mid = ctx.make_binary(BVOp::Add,
                    ctx.make_binary(BVOp::LShr, p0, ctx.make_const(32, 64), 64),
                    ctx.make_binary(BVOp::Add, ctx.make_zext(ctx.make_extract(p1, 31, 0), 32),
                                               ctx.make_zext(ctx.make_extract(p2, 31, 0), 32), 64), 64);
                BVOp sh_op = (op == LirOpcode::Smulh) ? BVOp::AShr : BVOp::LShr;
                BVNodeId hi = ctx.make_binary(BVOp::Add, p3,
                    ctx.make_binary(BVOp::Add, ctx.make_binary(sh_op, p1, ctx.make_const(32, 64), 64),
                        ctx.make_binary(BVOp::Add, ctx.make_binary(sh_op, p2, ctx.make_const(32, 64), 64),
                                                   ctx.make_binary(BVOp::LShr, mid, ctx.make_const(32, 64), 64), 64), 64), 64);
                write_op(inst.defs[0], hi);
                break;
            }

            case LirOpcode::Idiv: case LirOpcode::Idiv32:
            case LirOpcode::Div:  case LirOpcode::Div32: {
                uint32_t w = (op == LirOpcode::Idiv32 || op == LirOpcode::Div32) ? 32 : 64;
                BVOp dop = (op == LirOpcode::Idiv || op == LirOpcode::Idiv32) ? BVOp::SDiv : BVOp::UDiv;
                write_op(inst.defs[0], ctx.make_binary(dop, get_bin_lhs(w), get_bin_rhs(w), w));
                break;
            }

            case LirOpcode::And: case LirOpcode::And32:
            case LirOpcode::Or:  case LirOpcode::Or32:
            case LirOpcode::Xor: case LirOpcode::Xor32: {
                uint32_t w = (op == LirOpcode::And32 || op == LirOpcode::Or32 || op == LirOpcode::Xor32) ? 32 : 64;
                BVOp bop = (op == LirOpcode::And || op == LirOpcode::And32) ? BVOp::And
                         : (op == LirOpcode::Or  || op == LirOpcode::Or32)  ? BVOp::Or : BVOp::Xor;
                write_op(inst.defs[0], ctx.make_binary(bop, get_bin_lhs(w), get_bin_rhs(w), w));
                break;
            }

            case LirOpcode::Not: case LirOpcode::Not32: {
                uint32_t w = (op == LirOpcode::Not32) ? 32 : 64;
                BVNodeId a = inst.uses.empty() ? read_op(inst.defs[0], w) : read_op(inst.uses[0], w);
                write_op(inst.defs[0], ctx.make_unary(BVOp::Not, a, w));
                break;
            }

            case LirOpcode::Neg: case LirOpcode::Neg32: {
                uint32_t w = (op == LirOpcode::Neg32) ? 32 : 64;
                BVNodeId a = inst.uses.empty() ? read_op(inst.defs[0], w) : read_op(inst.uses[0], w);
                BVOp uop = (inst.defs[0].is_preg() && inst.defs[0].preg_val.is_xmm()) ? BVOp::FNeg : BVOp::Neg;
                write_op(inst.defs[0], ctx.make_unary(uop, a, w));
                break;
            }

            case LirOpcode::Shl: case LirOpcode::Shl32:
            case LirOpcode::Shr: case LirOpcode::Shr32:
            case LirOpcode::Sar: case LirOpcode::Sar32: {
                uint32_t w = (op == LirOpcode::Shl32 || op == LirOpcode::Shr32 || op == LirOpcode::Sar32) ? 32 : 64;
                BVOp sop = (op == LirOpcode::Shl || op == LirOpcode::Shl32) ? BVOp::Shl
                         : (op == LirOpcode::Shr || op == LirOpcode::Shr32) ? BVOp::LShr : BVOp::AShr;
                write_op(inst.defs[0], ctx.make_binary(sop, get_bin_lhs(w), get_bin_rhs(w), w));
                break;
            }

            case LirOpcode::Popcnt: case LirOpcode::Popcnt32:
            case LirOpcode::Lzcnt:  case LirOpcode::Lzcnt32:
            case LirOpcode::Tzcnt:  case LirOpcode::Tzcnt32:
            case LirOpcode::Bsf:    case LirOpcode::Bsf32: {
                uint32_t w = (op == LirOpcode::Popcnt32 || op == LirOpcode::Lzcnt32 ||
                              op == LirOpcode::Tzcnt32  || op == LirOpcode::Bsf32) ? 32 : 64;
                BVOp uop = (op == LirOpcode::Popcnt || op == LirOpcode::Popcnt32) ? BVOp::Popcnt
                         : (op == LirOpcode::Lzcnt  || op == LirOpcode::Lzcnt32)  ? BVOp::Clz : BVOp::Ctz;
                write_op(inst.defs[0], ctx.make_unary(uop, read_op(inst.uses[0], w), w));
                break;
            }

            case LirOpcode::Bsr: case LirOpcode::Bsr32: {
                uint32_t w = (op == LirOpcode::Bsr32) ? 32 : 64;
                BVNodeId clz = ctx.make_unary(BVOp::Clz, read_op(inst.uses[0], w), w);
                write_op(inst.defs[0], ctx.make_binary(BVOp::Sub, ctx.make_const(w - 1, w), clz, w));
                break;
            }

            case LirOpcode::Cmp: case LirOpcode::Cmp32: {
                uint32_t w = (op == LirOpcode::Cmp32) ? 32 : 64;
                BVNodeId a = read_op(inst.uses[0], w), b = read_op(inst.uses[1], w);
                set_arm64_subs_flags(ctx, st.nzcv, a, b, ctx.make_binary(BVOp::Sub, a, b, w), w);
                break;
            }

            case LirOpcode::Test: case LirOpcode::Test32: {
                uint32_t w = (op == LirOpcode::Test32) ? 32 : 64;
                BVNodeId a = read_op(inst.uses[0], w), b = read_op(inst.uses[1], w);
                set_arm64_ands_flags(ctx, st.nzcv, ctx.make_binary(BVOp::And, a, b, w), w);
                break;
            }

            case LirOpcode::Setcc: {
                // On AArch64, Setcc lowers to `cset Xd, cond` (`csinc Xd, XZR, XZR, invert(cond)`),
                // writing 0 or 1 to all 64 bits of Xd!
                BVNodeId bit = eval_arm64_cond(ctx, st.nzcv, aarch64::to_aarch64_cond(inst.condition));
                write_op(inst.defs[0], ctx.make_zext(bit, 63));
                break;
            }

            case LirOpcode::Cmovcc: {
                BVNodeId cond_bit = eval_arm64_cond(ctx, st.nzcv, aarch64::to_aarch64_cond(inst.condition));
                uint32_t w = (inst.defs[0].size == 4) ? 32 : 64;
                BVNodeId f_val = (inst.uses.size() >= 2) ? read_op(inst.uses[0], w) : read_op(inst.defs[0], w);
                write_op(inst.defs[0], ctx.make_ite(cond_bit, read_op(inst.uses.back(), w), f_val));
                break;
            }

            case LirOpcode::Lea: {
                if (inst.defs.empty() || inst.uses.empty()) break;
                const auto& u = inst.uses[0];
                BVNodeId fp = st.gpr[static_cast<uint8_t>(aarch64::GPR::FP)];
                if (u.is_mem()) {
                    write_op(inst.defs[0], eval_mem_addr(u.mem_val));
                } else if (u.is_spill_slot()) {
                    write_op(inst.defs[0], ctx.make_binary(BVOp::Add, fp,
                        ctx.make_const(static_cast<uint64_t>(static_cast<int64_t>(u.spill_slot) * 8), 64), 64));
                } else if (u.is_local_slot()) {
                    write_op(inst.defs[0], ctx.make_binary(BVOp::Add, fp,
                        ctx.make_const(static_cast<uint64_t>(static_cast<int64_t>(u.local_offset)), 64), 64));
                }
                break;
            }

            case LirOpcode::Push: {
                if (inst.uses.empty()) break;
                st.gpr[31] = ctx.make_binary(BVOp::Sub, st.gpr[31], ctx.make_const(16, 64), 64);
                st.mem_stores.push_back({st.gpr[31], 8, read_op(inst.uses[0], 64)});
                break;
            }

            case LirOpcode::Pop: {
                if (inst.defs.empty()) break;
                BVNodeId sp_old = st.gpr[31], val = read_mem_sized(sp_old, 8);
                st.gpr[31] = ctx.make_binary(BVOp::Add, sp_old, ctx.make_const(16, 64), 64);
                write_op(inst.defs[0], val);
                break;
            }

            case LirOpcode::ParallelCopy: {
                size_t n = std::min(inst.defs.size(), inst.uses.size());
                struct PendingVal { LirOperand dst; bool is_128; BVNodeId lo; BVNodeId hi; };
                std::vector<PendingVal> pending;
                pending.reserve(n);
                for (size_t i = 0; i < n; ++i) {
                    const auto &dst = inst.defs[i], &src = inst.uses[i];
                    if (dst.size == 16 || src.size == 16) {
                        auto [lo, hi] = read_vec128(src);
                        pending.push_back({dst, true, lo, hi});
                    } else {
                        uint32_t w = (dst.size == 4 && src.size == 4) ? 32 : 64;
                        if (dst.is_spill_slot() && src.is_preg() && src.preg_val.is_gpr()) w = 64;
                        pending.push_back({dst, false, read_op(src, w), 0});
                    }
                }
                for (const auto& pv : pending) {
                    if (pv.is_128) write_vec128(pv.dst, pv.lo, pv.hi);
                    else write_op(pv.dst, pv.lo);
                }
                break;
            }

            case LirOpcode::Movsd: case LirOpcode::Movss:
            case LirOpcode::Movq_gx: case LirOpcode::Movq_xg:
            case LirOpcode::Movd_gx: case LirOpcode::Movd_xg: {
                if (inst.defs.empty() || inst.uses.empty()) break;
                uint32_t w = (op == LirOpcode::Movss) ? 32
                           : (op == LirOpcode::Movsd && (inst.defs[0].size == 4 || inst.uses[0].size == 4)) ? 32
                           : ((op == LirOpcode::Movd_gx || op == LirOpcode::Movd_xg) &&
                              inst.defs[0].size != 8 && inst.uses[0].size != 8) ? 32 : 64;
                write_op(inst.defs[0], read_op(inst.uses[0], w));
                break;
            }

            case LirOpcode::Addsd: case LirOpcode::Addss:
            case LirOpcode::Subsd: case LirOpcode::Subss:
            case LirOpcode::Mulsd: case LirOpcode::Mulss:
            case LirOpcode::Divsd: case LirOpcode::Divss:
            case LirOpcode::Minsd: case LirOpcode::Minss:
            case LirOpcode::Maxsd: case LirOpcode::Maxss: {
                uint32_t w = (op == LirOpcode::Addss || op == LirOpcode::Subss || op == LirOpcode::Mulss ||
                              op == LirOpcode::Divss || op == LirOpcode::Minss || op == LirOpcode::Maxss) ? 32 : 64;
                BVOp fop = (op == LirOpcode::Addsd || op == LirOpcode::Addss) ? BVOp::FAdd
                         : (op == LirOpcode::Subsd || op == LirOpcode::Subss) ? BVOp::FSub
                         : (op == LirOpcode::Mulsd || op == LirOpcode::Mulss) ? BVOp::FMul
                         : (op == LirOpcode::Divsd || op == LirOpcode::Divss) ? BVOp::FDiv
                         : (op == LirOpcode::Minsd || op == LirOpcode::Minss) ? BVOp::FMin : BVOp::FMax;
                write_op(inst.defs[0], ctx.make_binary(fop, get_bin_lhs(w), get_bin_rhs(w), w));
                break;
            }

            case LirOpcode::Sqrtsd:  case LirOpcode::Sqrtss:
            case LirOpcode::Floor64: case LirOpcode::Floor32:
            case LirOpcode::Ceil64:  case LirOpcode::Ceil32:
            case LirOpcode::Round64: case LirOpcode::Round32:
            case LirOpcode::Fabs64:  case LirOpcode::Fabs32:
            case LirOpcode::Fneg:    case LirOpcode::Fneg32: {
                uint32_t w = (op == LirOpcode::Sqrtss  || op == LirOpcode::Floor32 || op == LirOpcode::Ceil32 ||
                              op == LirOpcode::Round32 || op == LirOpcode::Fabs32  || op == LirOpcode::Fneg32) ? 32 : 64;
                BVOp uop = (op == LirOpcode::Sqrtsd  || op == LirOpcode::Sqrtss)  ? BVOp::FSqrt
                         : (op == LirOpcode::Floor64 || op == LirOpcode::Floor32) ? BVOp::FFloor
                         : (op == LirOpcode::Ceil64  || op == LirOpcode::Ceil32)  ? BVOp::FCeil
                         : (op == LirOpcode::Round64 || op == LirOpcode::Round32) ? BVOp::FRound
                         : (op == LirOpcode::Fabs64  || op == LirOpcode::Fabs32)  ? BVOp::FAbs : BVOp::FNeg;
                write_op(inst.defs[0], ctx.make_unary(uop, read_op(inst.uses.back(), w), w));
                break;
            }

            case LirOpcode::Cvtsi2sd: case LirOpcode::Cvtsi2sd32:
            case LirOpcode::Cvtsi2ss: case LirOpcode::Cvtsi2ss32: {
                uint32_t src_w = (op == LirOpcode::Cvtsi2sd32 || op == LirOpcode::Cvtsi2ss32) ? 32 : 64;
                uint32_t dst_w = (op == LirOpcode::Cvtsi2ss   || op == LirOpcode::Cvtsi2ss32) ? 32 : 64;
                write_op(inst.defs[0], ctx.make_unary(BVOp::SiToFp, read_op(inst.uses[0], src_w), dst_w));
                break;
            }

            case LirOpcode::Cvttsd2si: case LirOpcode::Cvttsd2si32:
            case LirOpcode::Cvttss2si: case LirOpcode::Cvttss2si32: {
                uint32_t src_w = (op == LirOpcode::Cvttss2si   || op == LirOpcode::Cvttss2si32) ? 32 : 64;
                uint32_t dst_w = (op == LirOpcode::Cvttsd2si32 || op == LirOpcode::Cvttss2si32) ? 32 : 64;
                write_op(inst.defs[0], ctx.make_unary(BVOp::FpToSi, read_op(inst.uses[0], src_w), dst_w));
                break;
            }

            case LirOpcode::Cvtsd2ss:
                write_op(inst.defs[0], ctx.make_unary(BVOp::FpTrunc, read_op(inst.uses[0], 64), 32));
                break;
            case LirOpcode::Cvtss2sd:
                write_op(inst.defs[0], ctx.make_unary(BVOp::FpExt, read_op(inst.uses[0], 32), 64));
                break;

            case LirOpcode::Ucomisd: case LirOpcode::Ucomiss: {
                uint32_t w = (op == LirOpcode::Ucomiss) ? 32 : 64;
                set_arm64_fcmp_flags(ctx, st.nzcv, read_op(inst.uses[0], w), read_op(inst.uses[1], w));
                break;
            }

            case LirOpcode::Movaps:  case LirOpcode::Movups:
            case LirOpcode::Vmovaps: case LirOpcode::Vmovups: {
                if (inst.defs.empty() || inst.uses.empty()) break;
                auto [lo, hi] = read_vec128(inst.uses[0]);
                write_vec128(inst.defs[0], lo, hi);
                break;
            }

            case LirOpcode::Xorpd:  case LirOpcode::Xorps:  case LirOpcode::Pxor:
            case LirOpcode::Vxorps: case LirOpcode::Vxorpd: case LirOpcode::Vpxor:
            case LirOpcode::Pand:   case LirOpcode::Vandps: case LirOpcode::Vandpd: case LirOpcode::Vpand:
            case LirOpcode::Por:    case LirOpcode::Vorps:  case LirOpcode::Vorpd:  case LirOpcode::Vpor: {
                if (inst.defs.empty()) break;
                const auto& lhs_op = (inst.uses.size() >= 2) ? inst.uses[0] : inst.defs[0];
                auto [a_lo, a_hi] = read_vec128(lhs_op);
                auto [b_lo, b_hi] = read_vec128(inst.uses.back());
                BVOp bop = (op == LirOpcode::Pand || op == LirOpcode::Vandps ||
                            op == LirOpcode::Vandpd || op == LirOpcode::Vpand) ? BVOp::And
                         : (op == LirOpcode::Por || op == LirOpcode::Vorps ||
                            op == LirOpcode::Vorpd || op == LirOpcode::Vpor) ? BVOp::Or : BVOp::Xor;
                write_vec128(inst.defs[0], ctx.make_binary(bop, a_lo, b_lo, 64), ctx.make_binary(bop, a_hi, b_hi, 64));
                break;
            }

            case LirOpcode::Pnot: {
                if (inst.defs.empty() || inst.uses.empty()) break;
                auto [lo, hi] = read_vec128(inst.uses[0]);
                write_vec128(inst.defs[0], ctx.make_unary(BVOp::Not, lo, 64), ctx.make_unary(BVOp::Not, hi, 64));
                break;
            }

            case LirOpcode::Paddq:  case LirOpcode::Vpaddq:
            case LirOpcode::Psubq:  case LirOpcode::Vpsubq:
            case LirOpcode::Addpd:  case LirOpcode::Vaddpd:
            case LirOpcode::Subpd:  case LirOpcode::Vsubpd:
            case LirOpcode::Mulpd:  case LirOpcode::Vmulpd:
            case LirOpcode::Divpd:  case LirOpcode::Vdivpd: {
                if (inst.defs.empty()) break;
                const auto& lhs_op = (inst.uses.size() >= 2) ? inst.uses[0] : inst.defs[0];
                auto [a_lo, a_hi] = read_vec128(lhs_op);
                auto [b_lo, b_hi] = read_vec128(inst.uses.back());
                BVOp vop = (op == LirOpcode::Paddq || op == LirOpcode::Vpaddq) ? BVOp::Add
                         : (op == LirOpcode::Psubq || op == LirOpcode::Vpsubq) ? BVOp::Sub
                         : (op == LirOpcode::Addpd || op == LirOpcode::Vaddpd) ? BVOp::FAdd
                         : (op == LirOpcode::Subpd || op == LirOpcode::Vsubpd) ? BVOp::FSub
                         : (op == LirOpcode::Mulpd || op == LirOpcode::Vmulpd) ? BVOp::FMul : BVOp::FDiv;
                write_vec128(inst.defs[0], ctx.make_binary(vop, a_lo, b_lo, 64), ctx.make_binary(vop, a_hi, b_hi, 64));
                break;
            }

            case LirOpcode::Jmp:
                if (!inst.uses.empty() && inst.uses[0].is_label()) {
                    st.has_uncond_branch = true;
                    st.uncond_target_label = inst.uses[0].label_id;
                }
                break;

            case LirOpcode::Jcc:
                if (!inst.uses.empty() && inst.uses[0].is_label()) {
                    st.has_cond_branch = true;
                    st.cond_branch_taken = eval_arm64_cond(ctx, st.nzcv, aarch64::to_aarch64_cond(inst.condition));
                    st.cond_target_label = inst.uses[0].label_id;
                }
                break;

            default:
                st.supported = false;
                st.error_msg = "Unsupported AArch64 LIR opcode in equivalence checker: " + std::string(to_string(op));
                return st;
        }
    }
    return st;
}

} // namespace brass::verify
