#include <brass/verify/smt_equiv.hpp>
#include "bv_solver.hpp"
#include <brass/codegen/lir.hpp>
#include <brass/codegen/lir_flags.hpp>
#include <brass/codegen/peephole.hpp>
#include <algorithm>
#include <sstream>
#include <unordered_map>

namespace brass::verify {

namespace {

using namespace brass::codegen;

struct LirFlagsState {
    BVNodeId zf = 0;
    BVNodeId sf = 0;
    BVNodeId of = 0;
    BVNodeId cf = 0;
    BVNodeId pf = 0;
};

struct LirSymMemStore {
    BVNodeId addr = 0;
    uint32_t size = 8;
    BVNodeId val = 0;
};

struct LirInitEnv {
    BVNodeId gpr_init[16]{};
    BVNodeId xmm_lo_init[16]{};
    BVNodeId xmm_hi_init[16]{};
    LirFlagsState flags_init{};
    std::unordered_map<uint32_t, BVNodeId> vreg_init;
    std::unordered_map<int32_t, BVNodeId> spill_init;
    std::unordered_map<int32_t, BVNodeId> local_init;

    struct MemReadEntry {
        BVNodeId addr;
        uint32_t size;
        BVNodeId val;
    };
    std::vector<MemReadEntry> mem_reads;
    uint32_t clobber_seq = 0;

    void init_defaults(BVContext& ctx) {
        for (uint8_t i = 0; i < 16; ++i) {
            auto g = static_cast<x64::GPR>(i);
            gpr_init[i] = ctx.make_var(std::string(x64::to_string(g, x64::OperandSize::Qword)), 64, false);
        }
        for (uint8_t i = 0; i < 16; ++i) {
            auto x = static_cast<x64::XMM>(i);
            std::string xname(x64::to_string(x));
            xmm_lo_init[i] = ctx.make_var(xname + "_lo", 64, true);
            xmm_hi_init[i] = ctx.make_var(xname + "_hi", 64, true);
        }
        flags_init.zf = ctx.make_var("init_zf", 1, false);
        flags_init.sf = ctx.make_var("init_sf", 1, false);
        flags_init.of = ctx.make_var("init_of", 1, false);
        flags_init.cf = ctx.make_var("init_cf", 1, false);
        flags_init.pf = ctx.make_var("init_pf", 1, false);
    }

    BVNodeId get_vreg_init(BVContext& ctx, uint32_t id) {
        auto it = vreg_init.find(id);
        if (it != vreg_init.end()) return it->second;
        BVNodeId v = ctx.make_var("vreg" + std::to_string(id), 64, false);
        vreg_init[id] = v;
        return v;
    }

    BVNodeId get_spill_init(BVContext& ctx, int32_t slot) {
        auto it = spill_init.find(slot);
        if (it != spill_init.end()) return it->second;
        BVNodeId v = ctx.make_var("spill_" + std::to_string(slot), 64, false);
        spill_init[slot] = v;
        return v;
    }

    BVNodeId get_local_init(BVContext& ctx, int32_t off) {
        auto it = local_init.find(off);
        if (it != local_init.end()) return it->second;
        BVNodeId v = ctx.make_var("local_" + std::to_string(off), 64, false);
        local_init[off] = v;
        return v;
    }

    BVNodeId read_init_mem(BVContext& ctx, BVNodeId addr, uint32_t sz) {
        uint32_t bits = sz * 8;
        for (const auto& e : mem_reads) {
            if (e.addr == addr && e.size == sz) return e.val;
        }
        BVNodeId v = ctx.make_var("lirmem_" + std::to_string(mem_reads.size()), bits, false);
        for (const auto& e : mem_reads) {
            if (e.size == sz) {
                BVNodeId eq = ctx.make_binary(BVOp::Eq, addr, e.addr, 1);
                v = ctx.make_ite(eq, e.val, v);
            }
        }
        mem_reads.push_back({addr, sz, v});
        return v;
    }
};

struct LirExecState {
    bool supported = true;
    std::string error_msg;

    BVNodeId gpr[16]{};
    BVNodeId xmm_lo[16]{};
    BVNodeId xmm_hi[16]{};
    LirFlagsState flags{};
    std::unordered_map<uint32_t, BVNodeId> vregs;
    std::unordered_map<int32_t, BVNodeId> spill_slots;
    std::unordered_map<int32_t, BVNodeId> local_slots;
    std::vector<LirSymMemStore> mem_stores;

    bool flags_defined = true;
    bool has_branch = false;
    BVNodeId branch_taken = 0;
    uint32_t taken_label = UINT32_MAX;
    uint32_t fallthrough_label = UINT32_MAX;
};

BVNodeId eval_condition(BVContext& ctx, const LirFlagsState& f, x64::Condition cond) {
    using x64::Condition;
    switch (cond) {
        case Condition::O:   return f.of;
        case Condition::NO:  return ctx.make_unary(BVOp::Not, f.of, 1);
        case Condition::B:   return f.cf;
        case Condition::AE:  return ctx.make_unary(BVOp::Not, f.cf, 1);
        case Condition::E:   return f.zf;
        case Condition::NE:  return ctx.make_unary(BVOp::Not, f.zf, 1);
        case Condition::BE:  return ctx.make_binary(BVOp::Or, f.cf, f.zf, 1);
        case Condition::A:   return ctx.make_unary(BVOp::Not, ctx.make_binary(BVOp::Or, f.cf, f.zf, 1), 1);
        case Condition::S:   return f.sf;
        case Condition::NS:  return ctx.make_unary(BVOp::Not, f.sf, 1);
        case Condition::P:   return f.pf;
        case Condition::NP:  return ctx.make_unary(BVOp::Not, f.pf, 1);
        case Condition::L:   return ctx.make_binary(BVOp::Xor, f.sf, f.of, 1);
        case Condition::GE:  return ctx.make_unary(BVOp::Not, ctx.make_binary(BVOp::Xor, f.sf, f.of, 1), 1);
        case Condition::LE:  return ctx.make_binary(BVOp::Or, f.zf, ctx.make_binary(BVOp::Xor, f.sf, f.of, 1), 1);
        case Condition::G:   return ctx.make_unary(BVOp::Not,
                                 ctx.make_binary(BVOp::Or, f.zf, ctx.make_binary(BVOp::Xor, f.sf, f.of, 1), 1), 1);
        default:             return ctx.make_const(0, 1);
    }
}

void set_add_flags(BVContext& ctx, LirFlagsState& f, BVNodeId a, BVNodeId b, BVNodeId res, uint32_t w) {
    f.zf = ctx.make_binary(BVOp::Eq, res, ctx.make_const(0, w), 1);
    f.sf = ctx.make_extract(res, w - 1, w - 1);
    f.of = ctx.make_binary(BVOp::SAddOvf, a, b, 1);
    f.cf = ctx.make_binary(BVOp::UAddOvf, a, b, 1);
    f.pf = ctx.make_const(0, 1);
}

void set_sub_flags(BVContext& ctx, LirFlagsState& f, BVNodeId a, BVNodeId b, BVNodeId res, uint32_t w) {
    f.zf = ctx.make_binary(BVOp::Eq, res, ctx.make_const(0, w), 1);
    f.sf = ctx.make_extract(res, w - 1, w - 1);
    f.of = ctx.make_binary(BVOp::SSubOvf, a, b, 1);
    f.cf = ctx.make_binary(BVOp::USubOvf, a, b, 1);
    f.pf = ctx.make_const(0, 1);
}

void set_logic_flags(BVContext& ctx, LirFlagsState& f, BVNodeId res, uint32_t w) {
    f.zf = ctx.make_binary(BVOp::Eq, res, ctx.make_const(0, w), 1);
    f.sf = ctx.make_extract(res, w - 1, w - 1);
    f.of = ctx.make_const(0, 1);
    f.cf = ctx.make_const(0, 1);
    f.pf = ctx.make_const(0, 1);
}

LirExecState execute_lir_block(BVContext& ctx, const LirBlock& block, LirInitEnv& env) {
    LirExecState st;
    for (int i = 0; i < 16; ++i) {
        st.gpr[i] = env.gpr_init[i];
        st.xmm_lo[i] = env.xmm_lo_init[i];
        st.xmm_hi[i] = env.xmm_hi_init[i];
    }
    st.flags = env.flags_init;
    st.branch_taken = ctx.make_const(0, 1);

    auto read_reg64 = [&](const LirOperand& op) -> BVNodeId {
        if (op.is_preg()) {
            if (op.preg_val.is_gpr() && op.preg_val.code < 16) return st.gpr[op.preg_val.code];
            if (op.preg_val.is_xmm() && op.preg_val.code < 16) return st.xmm_lo[op.preg_val.code];
        }
        if (op.is_vreg()) {
            auto it = st.vregs.find(op.vreg_val.id);
            if (it != st.vregs.end()) return it->second;
            BVNodeId v = env.get_vreg_init(ctx, op.vreg_val.id);
            st.vregs[op.vreg_val.id] = v;
            return v;
        }
        return ctx.make_const(0, 64);
    };

    auto eval_mem_addr = [&](const LirMem& m) -> BVNodeId {
        BVNodeId addr = ctx.make_const(0, 64);
        if (m.base_preg.is_valid() && m.base_preg.code < 16) {
            addr = st.gpr[m.base_preg.code];
        } else if (m.base_vreg.is_valid()) {
            addr = read_reg64(LirOperand::vreg(m.base_vreg));
        }
        if (m.index_preg.is_valid() && m.index_preg.code < 16) {
            BVNodeId idx = st.gpr[m.index_preg.code];
            uint8_t sc = x64::scale_value(m.scale);
            if (sc > 1) idx = ctx.make_binary(BVOp::Mul, idx, ctx.make_const(sc, 64), 64);
            addr = ctx.make_binary(BVOp::Add, addr, idx, 64);
        } else if (m.index_vreg.is_valid()) {
            BVNodeId idx = read_reg64(LirOperand::vreg(m.index_vreg));
            uint8_t sc = x64::scale_value(m.scale);
            if (sc > 1) idx = ctx.make_binary(BVOp::Mul, idx, ctx.make_const(sc, 64), 64);
            addr = ctx.make_binary(BVOp::Add, addr, idx, 64);
        }
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
            uint64_t u = static_cast<uint64_t>(op.imm_int);
            return ctx.make_const(u, width_bits);
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
            BVNodeId addr = eval_mem_addr(op.mem_val);
            uint32_t sz = width_bits / 8;
            return read_mem_sized(addr, sz);
        } else {
            return ctx.make_const(0, width_bits);
        }
        if (width_bits == 64) return raw64;
        return ctx.make_extract(raw64, width_bits - 1, 0);
    };

    // Write to destination operand with exact x64 semantics:
    // - 32-bit write to GPR zero-extends to 64 bits!
    // - 8-bit or 16-bit write to GPR (e.g. Setcc) preserves upper bits!
    auto write_op = [&](const LirOperand& dst, BVNodeId val, bool zero_ext_32_on_gpr = true) {
        uint32_t w = ctx.width(val);
        if (dst.is_preg()) {
            if (dst.preg_val.is_gpr() && dst.preg_val.code < 16) {
                uint8_t c = dst.preg_val.code;
                if (w == 64) {
                    st.gpr[c] = val;
                } else if (w == 32 && zero_ext_32_on_gpr) {
                    st.gpr[c] = ctx.make_zext(val, 32);
                } else {
                    BVNodeId hi = ctx.make_extract(st.gpr[c], 63, w);
                    st.gpr[c] = ctx.make_concat(hi, val);
                }
            } else if (dst.preg_val.is_xmm() && dst.preg_val.code < 16) {
                uint8_t c = dst.preg_val.code;
                st.xmm_lo[c] = (w == 64) ? val : ctx.make_zext(val, 64 - w);
            }
        } else if (dst.is_vreg()) {
            BVNodeId v64 = (w == 64) ? val : ctx.make_zext(val, 64 - w);
            st.vregs[dst.vreg_val.id] = v64;
        } else if (dst.is_spill_slot()) {
            BVNodeId v64 = (w == 64) ? val : ctx.make_zext(val, 64 - w);
            st.spill_slots[dst.spill_slot] = v64;
        } else if (dst.is_local_slot()) {
            BVNodeId v64 = (w == 64) ? val : ctx.make_zext(val, 64 - w);
            st.local_slots[dst.local_offset] = v64;
        } else if (dst.is_mem()) {
            BVNodeId addr = eval_mem_addr(dst.mem_val);
            uint32_t sz = w / 8;
            if ( dst.size == 1 || dst.size == 2 || dst.size == 4 || dst.size == 8 ) {
                sz = dst.size;
                if (w > sz * 8) val = ctx.make_extract(val, sz * 8 - 1, 0);
                else if (w < sz * 8) val = ctx.make_zext(val, sz * 8 - w);
            }
            st.mem_stores.push_back({addr, sz, val});
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
            if (!inst.uses.empty()) return read_op(inst.uses.back(), w);
            return ctx.make_const(0, w);
        };

        switch (op) {
            case LirOpcode::Nop:
            case LirOpcode::KeepAlive:
                break;

            case LirOpcode::Mov:
            case LirOpcode::Movabs: {
                if (inst.defs.empty() || inst.uses.empty()) break;
                uint32_t w = (inst.defs[0].is_mem() && inst.defs[0].size < 8)
                    ? static_cast<uint32_t>(inst.defs[0].size) * 8 : 64;
                BVNodeId v = read_op(inst.uses[0], w);
                write_op(inst.defs[0], v);
                break;
            }
            case LirOpcode::Mov32: {
                if (inst.defs.empty() || inst.uses.empty()) break;
                BVNodeId v = read_op(inst.uses[0], 32);
                write_op(inst.defs[0], v, true);
                break;
            }
            case LirOpcode::Movsxd: {
                if (inst.defs.empty() || inst.uses.empty()) break;
                BVNodeId v32 = read_op(inst.uses[0], 32);
                write_op(inst.defs[0], ctx.make_sext(v32, 32));
                break;
            }
            case LirOpcode::Movsx8: {
                if (inst.defs.empty() || inst.uses.empty()) break;
                BVNodeId v8 = read_op(inst.uses[0], 8);
                if (inst.defs[0].size == 4) {
                    write_op(inst.defs[0], ctx.make_sext(v8, 24), true);
                } else {
                    write_op(inst.defs[0], ctx.make_sext(v8, 56));
                }
                break;
            }
            case LirOpcode::Movsx16: {
                if (inst.defs.empty() || inst.uses.empty()) break;
                BVNodeId v16 = read_op(inst.uses[0], 16);
                if (inst.defs[0].size == 4) {
                    write_op(inst.defs[0], ctx.make_sext(v16, 16), true);
                } else {
                    write_op(inst.defs[0], ctx.make_sext(v16, 48));
                }
                break;
            }
            case LirOpcode::Movzx8: {
                if (inst.defs.empty() || inst.uses.empty()) break;
                BVNodeId v8 = read_op(inst.uses[0], 8);
                write_op(inst.defs[0], ctx.make_zext(v8, 56));
                break;
            }
            case LirOpcode::Movzx16: {
                if (inst.defs.empty() || inst.uses.empty()) break;
                BVNodeId v16 = read_op(inst.uses[0], 16);
                write_op(inst.defs[0], ctx.make_zext(v16, 48));
                break;
            }

            case LirOpcode::Add:
            case LirOpcode::Add32:
            case LirOpcode::Adds:
            case LirOpcode::Adds32: {
                uint32_t w = (op == LirOpcode::Add32 || op == LirOpcode::Adds32) ? 32 : 64;
                BVNodeId a = get_bin_lhs(w);
                BVNodeId b = get_bin_rhs(w);
                BVNodeId res = ctx.make_binary(BVOp::Add, a, b, w);
                write_op(inst.defs[0], res, true);
                set_add_flags(ctx, st.flags, a, b, res, w);
                st.flags_defined = true;
                break;
            }

            case LirOpcode::Sub:
            case LirOpcode::Sub32:
            case LirOpcode::Subs:
            case LirOpcode::Subs32: {
                uint32_t w = (op == LirOpcode::Sub32 || op == LirOpcode::Subs32) ? 32 : 64;
                BVNodeId a = get_bin_lhs(w);
                BVNodeId b = get_bin_rhs(w);
                BVNodeId res = ctx.make_binary(BVOp::Sub, a, b, w);
                write_op(inst.defs[0], res, true);
                set_sub_flags(ctx, st.flags, a, b, res, w);
                st.flags_defined = true;
                break;
            }

            case LirOpcode::Imul:
            case LirOpcode::Imul32: {
                uint32_t w = (op == LirOpcode::Imul32) ? 32 : 64;
                BVNodeId a = get_bin_lhs(w);
                BVNodeId b = get_bin_rhs(w);
                BVNodeId res = ctx.make_binary(BVOp::Mul, a, b, w);
                write_op(inst.defs[0], res, true);
                BVNodeId ovf = ctx.make_binary(BVOp::SMulOvf, a, b, 1);
                st.flags.of = ovf;
                st.flags.cf = ovf;
                st.flags.zf = ctx.make_var("clob_zf_" + std::to_string(env.clobber_seq), 1, false);
                st.flags.sf = ctx.make_var("clob_sf_" + std::to_string(env.clobber_seq++), 1, false);
                break;
            }

            case LirOpcode::And:
            case LirOpcode::And32:
            case LirOpcode::Or:
            case LirOpcode::Or32:
            case LirOpcode::Xor:
            case LirOpcode::Xor32: {
                uint32_t w = (op == LirOpcode::And32 || op == LirOpcode::Or32 || op == LirOpcode::Xor32) ? 32 : 64;
                BVNodeId a = get_bin_lhs(w);
                BVNodeId b = get_bin_rhs(w);
                BVOp bop = (op == LirOpcode::And || op == LirOpcode::And32) ? BVOp::And
                         : (op == LirOpcode::Or  || op == LirOpcode::Or32)  ? BVOp::Or : BVOp::Xor;
                BVNodeId res = ctx.make_binary(bop, a, b, w);
                write_op(inst.defs[0], res, true);
                set_logic_flags(ctx, st.flags, res, w);
                st.flags_defined = true;
                break;
            }

            case LirOpcode::Not:
            case LirOpcode::Not32: {
                uint32_t w = (op == LirOpcode::Not32) ? 32 : 64;
                BVNodeId a = inst.uses.empty() ? read_op(inst.defs[0], w) : read_op(inst.uses[0], w);
                BVNodeId res = ctx.make_unary(BVOp::Not, a, w);
                write_op(inst.defs[0], res, true);
                break;
            }

            case LirOpcode::Neg:
            case LirOpcode::Neg32: {
                uint32_t w = (op == LirOpcode::Neg32) ? 32 : 64;
                BVNodeId a = inst.uses.empty() ? read_op(inst.defs[0], w) : read_op(inst.uses[0], w);
                BVNodeId zero = ctx.make_const(0, w);
                BVNodeId res = ctx.make_unary(BVOp::Neg, a, w);
                write_op(inst.defs[0], res, true);
                set_sub_flags(ctx, st.flags, zero, a, res, w);
                st.flags_defined = true;
                break;
            }

            case LirOpcode::Shl:
            case LirOpcode::Shl32:
            case LirOpcode::Shr:
            case LirOpcode::Shr32:
            case LirOpcode::Sar:
            case LirOpcode::Sar32: {
                uint32_t w = (op == LirOpcode::Shl32 || op == LirOpcode::Shr32 || op == LirOpcode::Sar32) ? 32 : 64;
                BVNodeId a = get_bin_lhs(w);
                BVNodeId b = get_bin_rhs(w);
                BVOp sop = (op == LirOpcode::Shl || op == LirOpcode::Shl32) ? BVOp::Shl
                         : (op == LirOpcode::Shr || op == LirOpcode::Shr32) ? BVOp::LShr : BVOp::AShr;
                BVNodeId res = ctx.make_binary(sop, a, b, w);
                write_op(inst.defs[0], res, true);
                // x64 shifts leave flags unchanged when shift count == 0
                BVNodeId mask = ctx.make_const(w - 1, w);
                BVNodeId cnt = ctx.make_binary(BVOp::And, b, mask, w);
                BVNodeId nz = ctx.make_binary(BVOp::Ne, cnt, ctx.make_const(0, w), 1);
                BVNodeId new_zf = ctx.make_binary(BVOp::Eq, res, ctx.make_const(0, w), 1);
                BVNodeId new_sf = ctx.make_extract(res, w - 1, w - 1);
                st.flags.zf = ctx.make_ite(nz, new_zf, st.flags.zf);
                st.flags.sf = ctx.make_ite(nz, new_sf, st.flags.sf);
                st.flags.of = ctx.make_var("clob_of_" + std::to_string(env.clobber_seq), 1, false);
                st.flags.cf = ctx.make_var("clob_cf_" + std::to_string(env.clobber_seq++), 1, false);
                break;
            }

            case LirOpcode::Popcnt:
            case LirOpcode::Popcnt32:
            case LirOpcode::Lzcnt:
            case LirOpcode::Lzcnt32:
            case LirOpcode::Tzcnt:
            case LirOpcode::Tzcnt32: {
                uint32_t w = (op == LirOpcode::Popcnt32 || op == LirOpcode::Lzcnt32 || op == LirOpcode::Tzcnt32) ? 32 : 64;
                BVNodeId a = read_op(inst.uses[0], w);
                BVOp uop = (op == LirOpcode::Popcnt || op == LirOpcode::Popcnt32) ? BVOp::Popcnt
                         : (op == LirOpcode::Lzcnt  || op == LirOpcode::Lzcnt32)  ? BVOp::Clz : BVOp::Ctz;
                BVNodeId res = ctx.make_unary(uop, a, w);
                write_op(inst.defs[0], res, true);
                break;
            }

            case LirOpcode::Cdq: {
                BVNodeId eax = ctx.make_extract(st.gpr[static_cast<uint8_t>(x64::GPR::RAX)], 31, 0);
                BVNodeId sign = ctx.make_sext(ctx.make_extract(eax, 31, 31), 31);
                st.gpr[static_cast<uint8_t>(x64::GPR::RDX)] = ctx.make_zext(sign, 32);
                break;
            }
            case LirOpcode::Cqo: {
                BVNodeId rax = st.gpr[static_cast<uint8_t>(x64::GPR::RAX)];
                BVNodeId sign = ctx.make_sext(ctx.make_extract(rax, 63, 63), 63);
                st.gpr[static_cast<uint8_t>(x64::GPR::RDX)] = sign;
                break;
            }

            case LirOpcode::Cmp:
            case LirOpcode::Cmp32: {
                uint32_t w = (op == LirOpcode::Cmp32) ? 32 : 64;
                BVNodeId a = read_op(inst.uses[0], w);
                BVNodeId b = read_op(inst.uses[1], w);
                BVNodeId diff = ctx.make_binary(BVOp::Sub, a, b, w);
                set_sub_flags(ctx, st.flags, a, b, diff, w);
                st.flags_defined = true;
                break;
            }

            case LirOpcode::Test:
            case LirOpcode::Test32: {
                uint32_t w = (op == LirOpcode::Test32) ? 32 : 64;
                BVNodeId a = read_op(inst.uses[0], w);
                BVNodeId b = read_op(inst.uses[1], w);
                BVNodeId res = ctx.make_binary(BVOp::And, a, b, w);
                set_logic_flags(ctx, st.flags, res, w);
                st.flags_defined = true;
                break;
            }

            case LirOpcode::Setcc: {
                BVNodeId bit = eval_condition(ctx, st.flags, inst.condition);
                BVNodeId byte_val = ctx.make_zext(bit, 7);
                // x64 setcc writes only the low 8 bits of the destination GPR!
                write_op(inst.defs[0], byte_val, false);
                break;
            }

            case LirOpcode::Cmovcc: {
                BVNodeId cond_bit = eval_condition(ctx, st.flags, inst.condition);
                uint32_t w = (inst.defs[0].size == 4) ? 32 : 64;
                BVNodeId f_val = read_op(inst.defs[0], w);
                BVNodeId t_val = read_op(inst.uses.back(), w);
                BVNodeId res = ctx.make_ite(cond_bit, t_val, f_val);
                write_op(inst.defs[0], res, true);
                break;
            }

            case LirOpcode::Lea: {
                if (inst.defs.empty() || inst.uses.empty()) break;
                BVNodeId addr = 0;
                if (inst.uses[0].is_mem()) {
                    addr = eval_mem_addr(inst.uses[0].mem_val);
                } else if (inst.uses[0].is_spill_slot()) {
                    addr = ctx.make_binary(BVOp::Add, st.gpr[static_cast<uint8_t>(x64::GPR::RBP)],
                        ctx.make_const(static_cast<uint64_t>(int64_t{-8} * (inst.uses[0].spill_slot + 1)), 64), 64);
                } else if (inst.uses[0].is_local_slot()) {
                    addr = ctx.make_binary(BVOp::Add, st.gpr[static_cast<uint8_t>(x64::GPR::RBP)],
                        ctx.make_const(static_cast<uint64_t>(static_cast<int64_t>(inst.uses[0].local_offset)), 64), 64);
                } else {
                    break;
                }
                write_op(inst.defs[0], (inst.defs[0].size == 4) ? ctx.make_extract(addr, 31, 0) : addr, true);
                break;
            }

            case LirOpcode::Push: {
                if (inst.uses.empty()) break;
                BVNodeId v = read_op(inst.uses[0], 64);
                BVNodeId rsp = ctx.make_binary(BVOp::Sub, st.gpr[static_cast<uint8_t>(x64::GPR::RSP)], ctx.make_const(8, 64), 64);
                st.gpr[static_cast<uint8_t>(x64::GPR::RSP)] = rsp;
                st.mem_stores.push_back({rsp, 8, v});
                break;
            }
            case LirOpcode::Pop: {
                if (inst.defs.empty()) break;
                BVNodeId rsp = st.gpr[static_cast<uint8_t>(x64::GPR::RSP)];
                write_op(inst.defs[0], read_mem_sized(rsp, 8), true);
                st.gpr[static_cast<uint8_t>(x64::GPR::RSP)] = ctx.make_binary(BVOp::Add, rsp, ctx.make_const(8, 64), 64);
                break;
            }

            case LirOpcode::ParallelCopy: {
                size_t pairs = std::min(inst.defs.size(), inst.uses.size());
                std::vector<std::pair<LirOperand, BVNodeId>> pending;
                pending.reserve(pairs);
                for (size_t k = 0; k < pairs; ++k) {
                    const auto &dst = inst.defs[k], &src = inst.uses[k];
                    uint32_t w = (dst.size == 4 || src.size == 4) ? 32 : 64;
                    if (dst.is_spill_slot() && src.is_preg() && src.preg_val.is_gpr()) w = 64;
                    pending.push_back({dst, read_op(src, w)});
                }
                for (const auto& [dst, v] : pending) write_op(dst, v, true);
                break;
            }

            case LirOpcode::Movsd:
            case LirOpcode::Movss:
            case LirOpcode::Movq_gx:
            case LirOpcode::Movq_xg:
            case LirOpcode::Movd_gx:
            case LirOpcode::Movd_xg:
            case LirOpcode::Movaps:
            case LirOpcode::Movups:
            case LirOpcode::Vmovaps:
            case LirOpcode::Vmovups: {
                if (inst.defs.empty() || inst.uses.empty()) break;
                uint32_t w = (op == LirOpcode::Movss || op == LirOpcode::Movd_gx || op == LirOpcode::Movd_xg) ? 32 : 64;
                BVNodeId v = read_op(inst.uses[0], w);
                write_op(inst.defs[0], v, true);
                break;
            }

            case LirOpcode::Addsd:
            case LirOpcode::Addss:
            case LirOpcode::Subsd:
            case LirOpcode::Subss:
            case LirOpcode::Mulsd:
            case LirOpcode::Mulss:
            case LirOpcode::Divsd:
            case LirOpcode::Divss:
            case LirOpcode::Minsd:
            case LirOpcode::Minss:
            case LirOpcode::Maxsd:
            case LirOpcode::Maxss: {
                uint32_t w = (op == LirOpcode::Addss || op == LirOpcode::Subss ||
                              op == LirOpcode::Mulss || op == LirOpcode::Divss ||
                              op == LirOpcode::Minss || op == LirOpcode::Maxss) ? 32 : 64;
                BVNodeId a = get_bin_lhs(w);
                BVNodeId b = get_bin_rhs(w);
                BVOp fop = (op == LirOpcode::Addsd || op == LirOpcode::Addss) ? BVOp::FAdd
                         : (op == LirOpcode::Subsd || op == LirOpcode::Subss) ? BVOp::FSub
                         : (op == LirOpcode::Mulsd || op == LirOpcode::Mulss) ? BVOp::FMul
                         : (op == LirOpcode::Divsd || op == LirOpcode::Divss) ? BVOp::FDiv
                         : (op == LirOpcode::Minsd || op == LirOpcode::Minss) ? BVOp::FMin : BVOp::FMax;
                write_op(inst.defs[0], ctx.make_binary(fop, a, b, w), true);
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
                write_op(inst.defs[0], ctx.make_unary(uop, read_op(inst.uses.back(), w), w), true);
                break;
            }

            case LirOpcode::Cvtsi2sd: case LirOpcode::Cvtsi2sd32:
            case LirOpcode::Cvtsi2ss: case LirOpcode::Cvtsi2ss32: {
                uint32_t src_w = (op == LirOpcode::Cvtsi2sd32 || op == LirOpcode::Cvtsi2ss32) ? 32 : 64;
                uint32_t dst_w = (op == LirOpcode::Cvtsi2ss   || op == LirOpcode::Cvtsi2ss32) ? 32 : 64;
                write_op(inst.defs[0], ctx.make_unary(BVOp::SiToFp, read_op(inst.uses[0], src_w), dst_w), true);
                break;
            }

            case LirOpcode::Cvttsd2si: case LirOpcode::Cvttsd2si32:
            case LirOpcode::Cvttss2si: case LirOpcode::Cvttss2si32: {
                uint32_t src_w = (op == LirOpcode::Cvttss2si   || op == LirOpcode::Cvttss2si32) ? 32 : 64;
                uint32_t dst_w = (op == LirOpcode::Cvttsd2si32 || op == LirOpcode::Cvttss2si32) ? 32 : 64;
                write_op(inst.defs[0], ctx.make_unary(BVOp::FpToSi, read_op(inst.uses[0], src_w), dst_w), true);
                break;
            }

            case LirOpcode::Cvtsd2ss:
                write_op(inst.defs[0], ctx.make_unary(BVOp::FpTrunc, read_op(inst.uses[0], 64), 32), true);
                break;
            case LirOpcode::Cvtss2sd:
                write_op(inst.defs[0], ctx.make_unary(BVOp::FpExt, read_op(inst.uses[0], 32), 64), true);
                break;

            case LirOpcode::Ucomisd:
            case LirOpcode::Ucomiss: {
                uint32_t w = (op == LirOpcode::Ucomiss) ? 32 : 64;
                BVNodeId a = read_op(inst.uses[0], w), b = read_op(inst.uses[1], w);
                BVNodeId unord = ctx.make_binary(BVOp::Or,
                    ctx.make_unary(BVOp::Not, ctx.make_binary(BVOp::FEq, a, a, 1), 1),
                    ctx.make_unary(BVOp::Not, ctx.make_binary(BVOp::FEq, b, b, 1), 1), 1);
                BVNodeId eq = ctx.make_binary(BVOp::FEq, a, b, 1);
                BVNodeId lt = ctx.make_binary(BVOp::FLt, a, b, 1);
                st.flags.zf = ctx.make_binary(BVOp::Or, unord, eq, 1);
                st.flags.cf = ctx.make_binary(BVOp::Or, unord, lt, 1);
                st.flags.pf = unord;
                st.flags.sf = ctx.make_const(0, 1);
                st.flags.of = ctx.make_const(0, 1);
                st.flags_defined = true;
                break;
            }

            case LirOpcode::Xorpd:
            case LirOpcode::Xorps:
            case LirOpcode::Pxor: {
                if (inst.defs.empty()) break;
                BVNodeId a = get_bin_lhs(64);
                BVNodeId b = get_bin_rhs(64);
                write_op(inst.defs[0], ctx.make_binary(BVOp::Xor, a, b, 64), true);
                if (inst.defs[0].is_preg() && inst.defs[0].preg_val.is_xmm()) {
                    uint8_t c = inst.defs[0].preg_val.code;
                    if (inst.uses.size() >= 2 && inst.uses[0].is_preg() && inst.uses[1].is_preg() &&
                        inst.uses[0].preg_val == inst.uses[1].preg_val) {
                        st.xmm_hi[c] = ctx.make_const(0, 64);
                    }
                }
                break;
            }

            case LirOpcode::Jmp: {
                if (!inst.uses.empty() && inst.uses[0].is_label()) {
                    if (!st.has_branch) {
                        st.has_branch = true;
                        st.branch_taken = ctx.make_const(1, 1);
                        st.taken_label = inst.uses[0].label_id;
                    } else {
                        st.fallthrough_label = inst.uses[0].label_id;
                    }
                }
                break;
            }

            case LirOpcode::Jcc: {
                if (!inst.uses.empty() && inst.uses[0].is_label()) {
                    st.has_branch = true;
                    st.branch_taken = eval_condition(ctx, st.flags, inst.condition);
                    st.taken_label = inst.uses[0].label_id;
                }
                break;
            }

            case LirOpcode::Ret:
                break;

            default:
                st.supported = false;
                st.error_msg = "Unsupported LIR opcode in equivalence checker: " +
                               std::string(to_string(op));
                return st;
        }
    }

    return st;
}

} // namespace

LirEquivResult verify_x64_lir_block_equivalence(
    const codegen::LirBlock& before,
    const codegen::LirBlock& after,
    uint16_t live_out_gpr_mask,
    uint16_t live_out_xmm_mask,
    bool check_flags,
    uint32_t fallthrough_block_id) {

    LirEquivResult res;
    BVContext ctx;
    LirInitEnv env;
    env.init_defaults(ctx);

    LirExecState st_b = execute_lir_block(ctx, before, env);
    if (!st_b.supported) {
        res.status = EquivStatus::Unsupported;
        res.diagnostic = st_b.error_msg;
        return res;
    }
    LirExecState st_a = execute_lir_block(ctx, after, env);
    if (!st_a.supported) {
        res.status = EquivStatus::Unsupported;
        res.diagnostic = st_a.error_msg;
        return res;
    }

    struct NamedCheck {
        std::string name;
        BVNodeId b_val;
        BVNodeId a_val;
        BVNodeId diff;
    };
    std::vector<NamedCheck> checks;

    for (uint8_t i = 0; i < 16; ++i) {
        if (live_out_gpr_mask & (1u << i)) {
            auto g = static_cast<x64::GPR>(i);
            BVNodeId d = ctx.make_binary(BVOp::Ne, st_b.gpr[i], st_a.gpr[i], 1);
            checks.push_back({std::string(x64::to_string(g, x64::OperandSize::Qword)), st_b.gpr[i], st_a.gpr[i], d});
        }
    }
    for (uint8_t i = 0; i < 16; ++i) {
        if (live_out_xmm_mask & (1u << i)) {
            auto x = static_cast<x64::XMM>(i);
            std::string xn(x64::to_string(x));
            BVNodeId d_lo = ctx.make_binary(BVOp::Ne, st_b.xmm_lo[i], st_a.xmm_lo[i], 1);
            checks.push_back({xn + ".lo", st_b.xmm_lo[i], st_a.xmm_lo[i], d_lo});
        }
    }

    // Check any vregs defined in either block
    for (const auto& [vid, b_v] : st_b.vregs) {
        auto it = st_a.vregs.find(vid);
        if (it != st_a.vregs.end()) {
            BVNodeId d = ctx.make_binary(BVOp::Ne, b_v, it->second, 1);
            checks.push_back({"vreg" + std::to_string(vid), b_v, it->second, d});
        }
    }

    // Check spill slots written in either block
    std::vector<int32_t> all_spills;
    for (const auto& [slot, _] : st_b.spill_slots) all_spills.push_back(slot);
    for (const auto& [slot, _] : st_a.spill_slots) {
        if (std::find(all_spills.begin(), all_spills.end(), slot) == all_spills.end()) {
            all_spills.push_back(slot);
        }
    }
    for (int32_t slot : all_spills) {
        BVNodeId bv = st_b.spill_slots.count(slot) ? st_b.spill_slots[slot] : env.get_spill_init(ctx, slot);
        BVNodeId av = st_a.spill_slots.count(slot) ? st_a.spill_slots[slot] : env.get_spill_init(ctx, slot);
        BVNodeId d = ctx.make_binary(BVOp::Ne, bv, av, 1);
        checks.push_back({"spill[" + std::to_string(slot) + "]", bv, av, d});
    }

    // Check memory stores
    auto eval_final_mem = [&](const std::vector<LirSymMemStore>& stores, BVNodeId addr, uint32_t sz) {
        BVNodeId v = env.read_init_mem(ctx, addr, sz);
        for (const auto& s : stores) {
            if (s.size == sz) v = ctx.make_ite(ctx.make_binary(BVOp::Eq, addr, s.addr, 1), s.val, v);
        }
        return v;
    };
    for (const auto* side : {&st_b.mem_stores, &st_a.mem_stores}) {
        for (const auto& ms : *side) {
            BVNodeId bv = eval_final_mem(st_b.mem_stores, ms.addr, ms.size);
            BVNodeId av = eval_final_mem(st_a.mem_stores, ms.addr, ms.size);
            checks.push_back({"mem_store", bv, av, ctx.make_binary(BVOp::Ne, bv, av, 1)});
        }
    }

    if (check_flags) {
        checks.push_back({"ZF", st_b.flags.zf, st_a.flags.zf, ctx.make_binary(BVOp::Xor, st_b.flags.zf, st_a.flags.zf, 1)});
        checks.push_back({"SF", st_b.flags.sf, st_a.flags.sf, ctx.make_binary(BVOp::Xor, st_b.flags.sf, st_a.flags.sf, 1)});
        checks.push_back({"OF", st_b.flags.of, st_a.flags.of, ctx.make_binary(BVOp::Xor, st_b.flags.of, st_a.flags.of, 1)});
        checks.push_back({"CF", st_b.flags.cf, st_a.flags.cf, ctx.make_binary(BVOp::Xor, st_b.flags.cf, st_a.flags.cf, 1)});
    }

    if (st_b.has_branch || st_a.has_branch || fallthrough_block_id != UINT32_MAX) {
        auto target_of = [&](const LirExecState& st) -> BVNodeId {
            uint32_t ft = (st.fallthrough_label != UINT32_MAX) ? st.fallthrough_label : fallthrough_block_id;
            BVNodeId ft_node = ctx.make_const(ft, 32);
            return st.has_branch ? ctx.make_ite(st.branch_taken, ctx.make_const(st.taken_label, 32), ft_node) : ft_node;
        };
        BVNodeId tb = target_of(st_b), ta = target_of(st_a);
        checks.push_back({"branch_target", tb, ta, ctx.make_binary(BVOp::Ne, tb, ta, 1)});
    }

    BVNodeId total_diff = ctx.make_const(0, 1);
    for (const auto& c : checks) {
        total_diff = ctx.make_binary(BVOp::Or, total_diff, c.diff, 1);
    }

    SolveOutcome solved = solve_nonzero(ctx, total_diff, 10000);
    if (solved.status == SolveOutcome::Unsat) {
        res.status = EquivStatus::ProvenEquivalent;
        return res;
    }
    if (solved.status == SolveOutcome::Unknown) {
        res.status = EquivStatus::Unknown;
        res.diagnostic = "Solver budget exhausted in LIR equivalence check";
        return res;
    }

    res.status = EquivStatus::Counterexample;
    std::ostringstream diag;
    diag << "x64 LIR equivalence counterexample in block '" << before.name << "': ";
    for (const auto& c : checks) {
        if (ctx.eval_concrete(c.diff, solved.model) != 0) {
            uint64_t bv = ctx.eval_concrete(c.b_val, solved.model);
            uint64_t av = ctx.eval_concrete(c.a_val, solved.model);
            diag << c.name << " mismatch (before=0x" << std::hex << bv
                 << ", after=0x" << av << std::dec << ")";
            break;
        }
    }
    res.diagnostic = diag.str();
    return res;
}

static bool are_x64_flags_live_out_of_block(const LirFunction& fn, size_t block_idx) noexcept {
    if (block_idx >= fn.blocks.size()) return false;
    for (const LirBlock* succ : fn.blocks[block_idx]->successors) {
        if (!succ) continue;
        for (const auto& inst : succ->instructions) {
            if (!inst) continue;
            if (lir_reads_flags(*inst, Arch::x64)) return true;
            if (lir_kills_flags(*inst, Arch::x64)) break;
        }
    }
    return false;
}

LirEquivResult verify_x64_lir_function_equivalence(
    const codegen::LirFunction& before, const codegen::LirFunction& after) {

    if (before.blocks.size() != after.blocks.size()) {
        return {EquivStatus::Counterexample, "Block count mismatch between before and after LirFunction"};
    }
    for (size_t i = 0; i < before.blocks.size(); ++i) {
        uint32_t fallthrough_id = (i + 1 < before.blocks.size()) ? before.blocks[i + 1]->id : UINT32_MAX;
        bool check_flags = are_x64_flags_live_out_of_block(before, i) || are_x64_flags_live_out_of_block(after, i);
        LirEquivResult blk_res = verify_x64_lir_block_equivalence(
            *before.blocks[i], *after.blocks[i], 0xFFFF, 0xFFFF, check_flags, fallthrough_id);
        if (blk_res.status == EquivStatus::Counterexample || blk_res.status == EquivStatus::Unknown) return blk_res;
    }
    return {EquivStatus::ProvenEquivalent, {}};
}

LirEquivResult verify_x64_peephole(const codegen::LirFunction& fn) {
    LirFunction copy;
    copy.name = fn.name;
    copy.return_type = fn.return_type;
    copy.calling_conv = fn.calling_conv;
    copy.vreg_table = fn.vreg_table;
    copy.frame = fn.frame;
    copy.resume_entries = fn.resume_entries;
    copy.reserved_gprs = fn.reserved_gprs;
    copy.forced_saved_gprs = fn.forced_saved_gprs;

    std::unordered_map<uint32_t, LirBlock*> id_map;
    for (const auto& blk : fn.blocks) {
        auto cloned = std::make_unique<LirBlock>(blk->id, blk->name);
        cloned->loop_depth = blk->loop_depth;
        cloned->is_landing_pad = blk->is_landing_pad;
        for (const auto& inst : blk->instructions) {
            if (inst) cloned->append_inst(std::make_unique<LirInst>(*inst));
        }
        id_map[cloned->id] = cloned.get();
        copy.blocks.push_back(std::move(cloned));
    }
    for (size_t i = 0; i < fn.blocks.size(); ++i) {
        for (const LirBlock* succ : fn.blocks[i]->successors) {
            if (succ && id_map.count(succ->id)) link_blocks(*copy.blocks[i], *id_map[succ->id]);
        }
    }
    run_lir_peephole_optimizations(copy);
    return verify_x64_lir_function_equivalence(fn, copy);
}

} // namespace brass::verify
