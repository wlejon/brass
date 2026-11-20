#include <brass/verify/smt_equiv.hpp>
#include <brass/target/x64/x64_isel.hpp>
#include <brass/target/target.hpp>
#include <brass/target/calling_conv.hpp>
#include <brass/mir/function.hpp>
#include "mir_sym_exec.hpp"

#include <algorithm>
#include <array>
#include <cstring>
#include <sstream>
#include <unordered_map>
#include <vector>

namespace brass::verify {

using codegen::LirBlock;
using codegen::LirFunction;
using codegen::LirInst;
using codegen::LirOpcode;
using codegen::LirOperand;
using x64::Condition;
using x64::GPR;

namespace {

constexpr uint64_t kLirLocalSlotBase = 0x7FFF000000000000ULL;
constexpr uint64_t kLirCanonicalRbp  = 0x7FFD000000000000ULL;

uint32_t mir_type_width(Type t) {
    switch (t.kind()) {
        case TypeKind::I8:     return 8;
        case TypeKind::I16:    return 16;
        case TypeKind::I32:
        case TypeKind::F32:    return 32;
        case TypeKind::I64:
        case TypeKind::F64:
        case TypeKind::Ptr:
        case TypeKind::GCRef:
        case TypeKind::Tagged: return 64;
        default:               return 0;
    }
}

BVNodeId fit_width(BVContext& ctx, BVNodeId n, uint32_t w, bool sext = false) {
    uint32_t cur = ctx.width(n);
    if (cur == w) return n;
    if (cur > w) return ctx.make_extract(n, w - 1, 0);
    return sext ? ctx.make_sext(n, w - cur) : ctx.make_zext(n, w - cur);
}

BVNodeId parity_low8(BVContext& ctx, BVNodeId v) {
    BVNodeId p = ctx.make_extract(v, 0, 0);
    for (uint32_t bit = 1; bit < 8; ++bit) {
        p = ctx.make_binary(BVOp::Xor, p, ctx.make_extract(v, bit, bit), 1);
    }
    return ctx.make_unary(BVOp::Not, p, 1);
}

bool is_const_value(const BVContext& ctx, BVNodeId n, uint64_t& val_out) {
    if (n < ctx.nodes().size() && ctx.nodes()[n].op == BVOp::Const) {
        val_out = ctx.nodes()[n].imm;
        return true;
    }
    return false;
}

bool is_local_stack_addr(const BVContext& ctx, BVNodeId addr) {
    uint64_t c = 0;
    if (is_const_value(ctx, addr, c)) {
        return (c >= kLirLocalSlotBase && c < (kLirLocalSlotBase + 0x1000000ULL)) ||
               (c >= kLirCanonicalRbp  && c < (kLirCanonicalRbp  + 0x1000000ULL));
    }
    if (addr < ctx.nodes().size() &&
        (ctx.nodes()[addr].op == BVOp::Add || ctx.nodes()[addr].op == BVOp::Sub)) {
        return is_local_stack_addr(ctx, ctx.nodes()[addr].a);
    }
    return false;
}

bool same_reg_operand(const LirOperand& a, const LirOperand& b) {
    if (a.is_vreg() && b.is_vreg()) return a.vreg_val == b.vreg_val;
    if (a.is_preg() && b.is_preg()) return a.preg_val == b.preg_val;
    return false;
}

struct LirExecState {
    std::array<BVNodeId, 16> gprs{};
    std::array<BVNodeId, 16> xmms{};
    std::unordered_map<uint32_t, BVNodeId> vregs;
    BVNodeId zf = 0, cf = 0, sf = 0, of = 0, pf = 0;
    BVNodeId cond = 0;
    BVNodeId cqo_rax = UINT32_MAX;
    BVNodeId cqo_rdx = UINT32_MAX;
    BVNodeId cdq_eax = UINT32_MAX;
    BVNodeId cdq_edx = UINT32_MAX;

    static void merge_into(BVContext& ctx, LirExecState& dst, BVNodeId ec, const LirExecState& src) {
        dst.cond = ctx.make_binary(BVOp::Or, dst.cond, ec, 1);
        for (size_t i = 0; i < 16; ++i) {
            if (dst.gprs[i] != src.gprs[i]) dst.gprs[i] = ctx.make_ite(ec, src.gprs[i], dst.gprs[i]);
            if (dst.xmms[i] != src.xmms[i]) dst.xmms[i] = ctx.make_ite(ec, src.xmms[i], dst.xmms[i]);
        }
        for (const auto& [vid, nv] : src.vregs) {
            auto it = dst.vregs.find(vid);
            if (it == dst.vregs.end()) {
                dst.vregs[vid] = nv;
            } else if (it->second != nv) {
                uint32_t w = std::max(ctx.width(it->second), ctx.width(nv));
                it->second = ctx.make_ite(ec, fit_width(ctx, nv, w), fit_width(ctx, it->second, w));
            }
        }
        auto mb = [&](BVNodeId& d, BVNodeId s) { if (d != s) d = ctx.make_ite(ec, s, d); };
        mb(dst.zf, src.zf); mb(dst.cf, src.cf); mb(dst.sf, src.sf); mb(dst.of, src.of); mb(dst.pf, src.pf);
        if (dst.cqo_rax != src.cqo_rax || dst.cqo_rdx != src.cqo_rdx) {
            dst.cqo_rax = UINT32_MAX;
            dst.cqo_rdx = UINT32_MAX;
        }
        if (dst.cdq_eax != src.cdq_eax || dst.cdq_edx != src.cdq_edx) {
            dst.cdq_eax = UINT32_MAX;
            dst.cdq_edx = UINT32_MAX;
        }
    }
};

struct LirExecResult {
    bool supported = true;
    bool hit_truncation = false;
    std::string error_msg;
    BVNodeId completed = 0;
    BVNodeId trapped = 0;
    BVNodeId returned = 0;
    BVNodeId ret_val = 0;
    std::vector<SymStore> stores;
};

BVNodeId eval_cond(BVContext& ctx, const LirExecState& st, Condition cc) {
    switch (cc) {
        case Condition::O:   return st.of;
        case Condition::NO:  return ctx.make_unary(BVOp::Not, st.of, 1);
        case Condition::B:   return st.cf;
        case Condition::AE:  return ctx.make_unary(BVOp::Not, st.cf, 1);
        case Condition::E:   return st.zf;
        case Condition::NE:  return ctx.make_unary(BVOp::Not, st.zf, 1);
        case Condition::BE:  return ctx.make_binary(BVOp::Or, st.cf, st.zf, 1);
        case Condition::A:   return ctx.make_unary(BVOp::Not, ctx.make_binary(BVOp::Or, st.cf, st.zf, 1), 1);
        case Condition::S:   return st.sf;
        case Condition::NS:  return ctx.make_unary(BVOp::Not, st.sf, 1);
        case Condition::P:   return st.pf;
        case Condition::NP:  return ctx.make_unary(BVOp::Not, st.pf, 1);
        case Condition::L:   return ctx.make_binary(BVOp::Xor, st.sf, st.of, 1);
        case Condition::GE:  return ctx.make_unary(BVOp::Not, ctx.make_binary(BVOp::Xor, st.sf, st.of, 1), 1);
        case Condition::LE:  return ctx.make_binary(BVOp::Or, st.zf, ctx.make_binary(BVOp::Xor, st.sf, st.of, 1), 1);
        case Condition::G:   return ctx.make_unary(BVOp::Not, ctx.make_binary(BVOp::Or, st.zf, ctx.make_binary(BVOp::Xor, st.sf, st.of, 1), 1), 1);
        default:             return ctx.make_const(0, 1);
    }
}

void compute_lir_rpo(const LirFunction& fn, const std::unordered_map<uint32_t, size_t>& id_to_idx,
                     size_t bi, std::vector<uint8_t>& visited, std::vector<size_t>& postorder) {
    if (bi >= fn.blocks.size() || visited[bi]) return;
    visited[bi] = 1;
    const LirBlock* bb = fn.blocks[bi].get();
    bool has_uncond_end = false;
    for (const auto& inst : bb->instructions) {
        if (inst->opcode == LirOpcode::Jmp) {
            if (!inst->uses.empty() && inst->uses[0].is_label()) {
                auto it = id_to_idx.find(inst->uses[0].label_id);
                if (it != id_to_idx.end()) compute_lir_rpo(fn, id_to_idx, it->second, visited, postorder);
            }
            has_uncond_end = true;
        } else if (inst->opcode == LirOpcode::Jcc) {
            if (!inst->uses.empty() && inst->uses[0].is_label()) {
                auto it = id_to_idx.find(inst->uses[0].label_id);
                if (it != id_to_idx.end()) compute_lir_rpo(fn, id_to_idx, it->second, visited, postorder);
            }
        } else if (inst->opcode == LirOpcode::Ret || inst->opcode == LirOpcode::Trap) {
            has_uncond_end = true;
        }
    }
    if (!has_uncond_end && bi + 1 < fn.blocks.size()) compute_lir_rpo(fn, id_to_idx, bi + 1, visited, postorder);
    postorder.push_back(bi);
}

void set_sub_flags(BVContext& ctx, LirExecState& st, BVNodeId a, BVNodeId b, BVNodeId res, uint32_t w) {
    st.zf = ctx.make_binary(BVOp::Eq, res, ctx.make_const(0, w), 1);
    st.sf = ctx.make_extract(res, w - 1, w - 1);
    st.cf = ctx.make_binary(BVOp::Ult, a, b, 1);
    BVNodeId ma = ctx.make_extract(a, w - 1, w - 1);
    BVNodeId mb = ctx.make_extract(b, w - 1, w - 1);
    BVNodeId mr = ctx.make_extract(res, w - 1, w - 1);
    st.of = ctx.make_binary(BVOp::And, ctx.make_binary(BVOp::Xor, ma, mb, 1),
                            ctx.make_binary(BVOp::Xor, ma, mr, 1), 1);
    st.pf = parity_low8(ctx, res);
}

void set_add_flags(BVContext& ctx, LirExecState& st, BVNodeId a, BVNodeId b, BVNodeId res, uint32_t w) {
    st.zf = ctx.make_binary(BVOp::Eq, res, ctx.make_const(0, w), 1);
    st.sf = ctx.make_extract(res, w - 1, w - 1);
    st.cf = ctx.make_binary(BVOp::Ult, res, a, 1);
    BVNodeId ma = ctx.make_extract(a, w - 1, w - 1);
    BVNodeId mb = ctx.make_extract(b, w - 1, w - 1);
    BVNodeId mr = ctx.make_extract(res, w - 1, w - 1);
    st.of = ctx.make_binary(BVOp::And,
                            ctx.make_unary(BVOp::Not, ctx.make_binary(BVOp::Xor, ma, mb, 1), 1),
                            ctx.make_binary(BVOp::Xor, ma, mr, 1), 1);
    st.pf = parity_low8(ctx, res);
}

void set_logic_flags(BVContext& ctx, LirExecState& st, BVNodeId res, uint32_t w) {
    st.zf = ctx.make_binary(BVOp::Eq, res, ctx.make_const(0, w), 1);
    st.sf = ctx.make_extract(res, w - 1, w - 1);
    st.cf = ctx.make_const(0, 1);
    st.of = ctx.make_const(0, 1);
    st.pf = parity_low8(ctx, res);
}

LirExecResult symbolically_execute_lir(
    BVContext& ctx, const Function& mir_fn, const LirFunction& lir_fn,
    const std::vector<BVNodeId>& param_vars, InitMemEnv& init_mem, const EquivOptions& opts) {

    LirExecResult out;
    out.completed = ctx.make_const(0, 1);
    out.trapped = ctx.make_const(0, 1);
    out.returned = ctx.make_const(0, 1);
    uint32_t ret_w = mir_type_width(mir_fn.return_type());
    out.ret_val = ret_w > 0 ? ctx.make_const(0, ret_w) : 0;

    if (lir_fn.blocks.empty()) {
        out.supported = false;
        out.error_msg = "LirFunction has no blocks";
        return out;
    }

    std::unordered_map<uint32_t, size_t> id_to_idx;
    for (size_t i = 0; i < lir_fn.blocks.size(); ++i) id_to_idx[lir_fn.blocks[i]->id] = i;
    std::vector<uint8_t> visited(lir_fn.blocks.size(), 0);
    std::vector<size_t> postorder;
    compute_lir_rpo(lir_fn, id_to_idx, 0, visited, postorder);
    std::reverse(postorder.begin(), postorder.end());
    std::vector<size_t> rpo_pos(lir_fn.blocks.size(), SIZE_MAX);
    for (size_t i = 0; i < postorder.size(); ++i) rpo_pos[postorder[i]] = i;

    LirExecState seed;
    seed.cond = ctx.make_const(1, 1);
    for (size_t i = 0; i < 16; ++i) {
        seed.gprs[i] = ctx.make_var("init_gpr_" + std::to_string(i), 64, false);
        seed.xmms[i] = ctx.make_var("init_xmm_" + std::to_string(i), 128, false);
    }
    seed.gprs[static_cast<size_t>(GPR::RSP)] = ctx.make_const(kLirCanonicalRbp, 64);
    seed.gprs[static_cast<size_t>(GPR::RBP)] = ctx.make_const(kLirCanonicalRbp, 64);
    seed.zf = seed.cf = seed.sf = seed.of = seed.pf = ctx.make_const(0, 1);

    const auto& cc = lir_fn.calling_conv;
    size_t gpr_idx = 0, xmm_idx = 0;
    for (size_t i = 0; i < mir_fn.param_count(); ++i) {
        Type pt = mir_fn.param_type(i);
        if (pt.is_vector()) {
            out.supported = false;
            out.error_msg = "Vector parameters not yet supported in x64 ISel verifier";
            return out;
        }
        BVNodeId pv = fit_width(ctx, param_vars[i], pt.is_float() ? 128 : 64, false);
        auto place_stack_arg = [&](int32_t disp) {
            BVNodeId saddr = ctx.make_const(kLirCanonicalRbp + static_cast<uint64_t>(disp), 64);
            out.stores.push_back({ctx.make_const(1, 1), saddr, 8, fit_width(ctx, param_vars[i], 64, false), true});
        };
        if (cc.kind() == CallingConvKind::Win64) {
            if (i < 4) {
                if (pt.is_float()) seed.xmms[i] = pv;
                else seed.gprs[static_cast<size_t>(cc.arg_gpr(i))] = pv;
            } else {
                place_stack_arg(static_cast<int32_t>(48 + (i - 4) * 8));
            }
        } else if (pt.is_float()) {
            if (xmm_idx < cc.num_arg_xmms()) {
                seed.xmms[static_cast<size_t>(cc.arg_xmm(xmm_idx++))] = pv;
            } else {
                size_t si = (xmm_idx - cc.num_arg_xmms()) + (gpr_idx > cc.num_arg_gprs() ? (gpr_idx - cc.num_arg_gprs()) : 0);
                place_stack_arg(static_cast<int32_t>(16 + si * 8));
                xmm_idx++;
            }
        } else {
            if (gpr_idx < cc.num_arg_gprs()) {
                seed.gprs[static_cast<size_t>(cc.arg_gpr(gpr_idx++))] = pv;
            } else {
                size_t si = (gpr_idx - cc.num_arg_gprs()) + (xmm_idx > cc.num_arg_xmms() ? (xmm_idx - cc.num_arg_xmms()) : 0);
                place_stack_arg(static_cast<int32_t>(16 + si * 8));
                gpr_idx++;
            }
        }
    }

    const uint32_t max_level = opts.loop_unroll_bound;
    std::vector<std::vector<std::vector<LirExecState>>> incoming(
        max_level + 1, std::vector<std::vector<LirExecState>>(postorder.size()));
    incoming[0][0].push_back(std::move(seed));

    auto route_label = [&](size_t src_rpo, uint32_t level, BVNodeId edge_cond, uint32_t label_id,
                           const LirExecState& st) {
        if (ctx.nodes()[edge_cond].op == BVOp::Const && ctx.nodes()[edge_cond].imm == 0) return;
        auto it = id_to_idx.find(label_id);
        if (it == id_to_idx.end()) return;
        size_t dst_rpo = rpo_pos[it->second];
        if (dst_rpo == SIZE_MAX) return;
        uint32_t dst_level = (dst_rpo <= src_rpo) ? (level + 1) : level;
        if (dst_level > max_level) { out.hit_truncation = true; return; }
        LirExecState nxt = st;
        nxt.cond = edge_cond;
        incoming[dst_level][dst_rpo].push_back(std::move(nxt));
    };

    for (uint32_t level = 0; level <= max_level; ++level) {
        for (size_t ri = 0; ri < postorder.size(); ++ri) {
            auto& edges = incoming[level][ri];
            if (edges.empty()) continue;

            LirExecState st = edges[0];
            for (size_t ei = 1; ei < edges.size(); ++ei) {
                LirExecState::merge_into(ctx, st, edges[ei].cond, edges[ei]);
            }

            const LirBlock* bb = lir_fn.blocks[postorder[ri]].get();
            BVNodeId cur_cond = st.cond;

            auto is_gpr_op = [](const LirOperand& op) {
                return (op.is_vreg() && op.vreg_val.is_gpr()) || (op.is_preg() && op.preg_val.is_gpr());
            };
            auto eval_mem_addr = [&](const LirOperand& op) -> BVNodeId {
                const auto& m = op.mem_val;
                BVNodeId addr = ctx.make_const(0, 64);
                bool has = false;
                if (m.base_vreg.is_valid()) {
                    addr = st.vregs.count(m.base_vreg.id) ? fit_width(ctx, st.vregs[m.base_vreg.id], 64) : ctx.make_const(0, 64);
                    has = true;
                } else if (m.base_preg.is_gpr()) {
                    addr = st.gprs[static_cast<size_t>(m.base_preg.as_gpr())];
                    has = true;
                }
                if (m.has_index()) {
                    BVNodeId idx_v = m.index_vreg.is_valid()
                        ? (st.vregs.count(m.index_vreg.id) ? fit_width(ctx, st.vregs[m.index_vreg.id], 64) : ctx.make_const(0, 64))
                        : st.gprs[static_cast<size_t>(m.index_preg.as_gpr())];
                    uint64_t sc = static_cast<uint64_t>(x64::scale_value(m.scale));
                    if (sc != 1) idx_v = ctx.make_binary(BVOp::Mul, idx_v, ctx.make_const(sc, 64), 64);
                    addr = has ? ctx.make_binary(BVOp::Add, addr, idx_v, 64) : idx_v;
                    has = true;
                }
                if (m.disp != 0 || !has) {
                    BVNodeId dv = ctx.make_const(static_cast<uint64_t>(static_cast<int64_t>(m.disp)), 64);
                    addr = has ? ctx.make_binary(BVOp::Add, addr, dv, 64) : dv;
                }
                return addr;
            };
            auto read_op = [&](const LirOperand& op, uint32_t w) -> BVNodeId {
                if (op.is_vreg()) {
                    auto it = st.vregs.find(op.vreg_val.id);
                    return fit_width(ctx, (it != st.vregs.end()) ? it->second : ctx.make_const(0, w), w, false);
                }
                if (op.is_preg()) {
                    if (op.preg_val.is_gpr()) return fit_width(ctx, st.gprs[op.preg_val.code & 15u], w, false);
                    if (op.preg_val.is_xmm()) return fit_width(ctx, st.xmms[op.preg_val.code & 15u], w, false);
                }
                if (op.is_imm_int()) return ctx.make_const(static_cast<uint64_t>(op.imm_int), w);
                if (op.is_imm_float()) {
                    uint64_t b = 0;
                    if (w == 32) { float f = static_cast<float>(op.imm_float); std::memcpy(&b, &f, 4); }
                    else         { double d = op.imm_float; std::memcpy(&b, &d, 8); }
                    return ctx.make_const(b, w);
                }
                if (op.is_mem()) {
                    uint32_t nb = op.size ? op.size : (w / 8);
                    return fit_width(ctx, read_memory_bv(ctx, init_mem, out.stores, eval_mem_addr(op), nb), w, false);
                }
                if (op.is_local_slot()) return ctx.make_const(kLirLocalSlotBase + static_cast<uint32_t>(op.local_offset), w);
                return ctx.make_const(0, w);
            };
            auto write_op = [&](const LirOperand& op, BVNodeId val, bool zero_upper_gpr = true) {
                uint32_t w = ctx.width(val);
                if (op.is_vreg()) {
                    if (op.vreg_val.is_gpr()) {
                        if (w == 32 && zero_upper_gpr) {
                            st.vregs[op.vreg_val.id] = fit_width(ctx, val, 64, false);
                        } else if ((w == 8 || w == 16) && !zero_upper_gpr) {
                            BVNodeId prev = st.vregs.count(op.vreg_val.id) ? fit_width(ctx, st.vregs[op.vreg_val.id], 64) : ctx.make_const(0, 64);
                            st.vregs[op.vreg_val.id] = ctx.make_concat(ctx.make_extract(prev, 63, w), val);
                        } else {
                            st.vregs[op.vreg_val.id] = fit_width(ctx, val, op.vreg_val.size * 8u, false);
                        }
                    } else {
                        st.vregs[op.vreg_val.id] = fit_width(ctx, val, 128, false);
                    }
                    return;
                }
                if (op.is_preg()) {
                    if (op.preg_val.is_gpr()) {
                        size_t idx = op.preg_val.code & 15u;
                        if (w == 64 || (w == 32 && zero_upper_gpr)) st.gprs[idx] = fit_width(ctx, val, 64, false);
                        else st.gprs[idx] = ctx.make_concat(ctx.make_extract(st.gprs[idx], 63, w), val);
                    } else if (op.preg_val.is_xmm()) {
                        st.xmms[op.preg_val.code & 15u] = fit_width(ctx, val, 128, false);
                    }
                    return;
                }
                if (op.is_mem()) {
                    uint32_t nb = op.size ? op.size : std::max(1u, w / 8);
                    BVNodeId addr = eval_mem_addr(op);
                    out.stores.push_back({cur_cond, addr, nb, fit_width(ctx, val, nb * 8, false),
                                          is_local_stack_addr(ctx, addr)});
                }
            };

            bool terminated = false, has_branch = false;
            BVNodeId branch_taken = ctx.make_const(0, 1);
            uint32_t taken_label = UINT32_MAX, fallthrough_label = UINT32_MAX;

            for (const auto& ip : bb->instructions) {
                const LirInst& inst = *ip;
                const LirOpcode op = inst.opcode;
                auto bin_lhs = [&](uint32_t w) { return read_op(inst.uses.size() >= 2 ? inst.uses[0] : inst.defs[0], w); };
                auto bin_rhs = [&](uint32_t w) { return read_op(inst.uses.back(), w); };
                auto emit_trap = [&](BVNodeId trap_c) {
                    BVNodeId trap_now = ctx.make_binary(BVOp::And, cur_cond, trap_c, 1);
                    out.trapped = ctx.make_binary(BVOp::Or, out.trapped, trap_now, 1);
                    out.completed = ctx.make_binary(BVOp::Or, out.completed, trap_now, 1);
                    cur_cond = ctx.make_binary(BVOp::And, cur_cond, ctx.make_unary(BVOp::Not, trap_c, 1), 1);
                };

                switch (op) {
                    case LirOpcode::Nop:
                    case LirOpcode::KeepAlive:
                        break;

                    case LirOpcode::ParallelCopy: {
                        std::vector<std::pair<BVNodeId, bool>> nv;
                        size_t n = std::min(inst.defs.size(), inst.uses.size());
                        for (size_t k = 0; k < n; ++k) {
                            uint32_t w = inst.defs[k].size ? static_cast<uint32_t>(inst.defs[k].size) * 8 : 64;
                            nv.emplace_back(read_op(inst.uses[k], w), is_gpr_op(inst.defs[k]));
                        }
                        for (size_t k = 0; k < n; ++k) write_op(inst.defs[k], nv[k].first, nv[k].second);
                        break;
                    }
                    case LirOpcode::Mov: case LirOpcode::Movabs: case LirOpcode::Mov32: {
                        uint32_t w = (op == LirOpcode::Mov32) ? 32 : (inst.defs[0].size ? inst.defs[0].size * 8u : 64u);
                        write_op(inst.defs[0], read_op(inst.uses[0], w), true);
                        break;
                    }
                    case LirOpcode::Movsxd:
                        write_op(inst.defs[0], ctx.make_sext(read_op(inst.uses[0], 32), 32), true);
                        break;
                    case LirOpcode::Movsx8: case LirOpcode::Movsx16:
                    case LirOpcode::Movzx8: case LirOpcode::Movzx16: {
                        uint32_t sw = (op == LirOpcode::Movsx8 || op == LirOpcode::Movzx8) ? 8 : 16;
                        uint32_t dw = (inst.defs[0].size == 4) ? 32 : 64;
                        BVNodeId sv = read_op(inst.uses[0], sw);
                        write_op(inst.defs[0], (op == LirOpcode::Movsx8 || op == LirOpcode::Movsx16)
                                 ? ctx.make_sext(sv, dw - sw) : ctx.make_zext(sv, dw - sw), true);
                        break;
                    }
                    case LirOpcode::Lea: {
                        const auto& u = inst.uses[0];
                        BVNodeId a = u.is_local_slot()
                            ? ctx.make_const(kLirLocalSlotBase + static_cast<uint32_t>(u.local_offset), 64)
                            : eval_mem_addr(u);
                        write_op(inst.defs[0], fit_width(ctx, a, inst.defs[0].size == 4 ? 32 : 64), true);
                        break;
                    }
                    case LirOpcode::Add: case LirOpcode::Add32:
                    case LirOpcode::Adds: case LirOpcode::Adds32: {
                        uint32_t w = (op == LirOpcode::Add32 || op == LirOpcode::Adds32) ? 32 : 64;
                        BVNodeId a = bin_lhs(w), b = bin_rhs(w), r = ctx.make_binary(BVOp::Add, a, b, w);
                        write_op(inst.defs[0], r, true);
                        set_add_flags(ctx, st, a, b, r, w);
                        break;
                    }
                    case LirOpcode::Sub: case LirOpcode::Sub32:
                    case LirOpcode::Subs: case LirOpcode::Subs32: {
                        uint32_t w = (op == LirOpcode::Sub32 || op == LirOpcode::Subs32) ? 32 : 64;
                        BVNodeId a = bin_lhs(w), b = bin_rhs(w), r = ctx.make_binary(BVOp::Sub, a, b, w);
                        write_op(inst.defs[0], r, true);
                        set_sub_flags(ctx, st, a, b, r, w);
                        break;
                    }
                    case LirOpcode::Imul: case LirOpcode::Imul32: {
                        uint32_t w = (op == LirOpcode::Imul32) ? 32 : 64;
                        BVNodeId a = bin_lhs(w), b = bin_rhs(w), r = ctx.make_binary(BVOp::Mul, a, b, w);
                        write_op(inst.defs[0], r, true);
                        st.of = st.cf = ctx.make_binary(BVOp::SMulOvf, a, b, 1);
                        break;
                    }
                    case LirOpcode::Smulh:
                    case LirOpcode::Umulh: {
                        BVNodeId a = read_op(inst.uses[0], 64), b = read_op(inst.uses[1], 64);
                        BVNodeId a_lo = ctx.make_zext(ctx.make_extract(a, 31, 0), 32);
                        BVNodeId a_hi = ctx.make_zext(ctx.make_extract(a, 63, 32), 32);
                        BVNodeId b_lo = ctx.make_zext(ctx.make_extract(b, 31, 0), 32);
                        BVNodeId b_hi = ctx.make_zext(ctx.make_extract(b, 63, 32), 32);
                        BVNodeId p00 = ctx.make_binary(BVOp::Mul, a_lo, b_lo, 64);
                        BVNodeId p01 = ctx.make_binary(BVOp::Mul, a_lo, b_hi, 64);
                        BVNodeId p10 = ctx.make_binary(BVOp::Mul, a_hi, b_lo, 64);
                        BVNodeId p11 = ctx.make_binary(BVOp::Mul, a_hi, b_hi, 64);
                        BVNodeId mask32 = ctx.make_const(0xFFFFFFFFULL, 64);
                        BVNodeId c32 = ctx.make_const(32ULL, 64);
                        BVNodeId mid = ctx.make_binary(BVOp::Add,
                            ctx.make_binary(BVOp::LShr, p00, c32, 64),
                            ctx.make_binary(BVOp::Add,
                                ctx.make_binary(BVOp::And, p01, mask32, 64),
                                ctx.make_binary(BVOp::And, p10, mask32, 64), 64), 64);
                        BVNodeId hi_u = ctx.make_binary(BVOp::Add, p11,
                            ctx.make_binary(BVOp::Add,
                                ctx.make_binary(BVOp::LShr, p01, c32, 64),
                                ctx.make_binary(BVOp::Add,
                                    ctx.make_binary(BVOp::LShr, p10, c32, 64),
                                    ctx.make_binary(BVOp::LShr, mid, c32, 64), 64), 64), 64);
                        st.gprs[static_cast<size_t>(GPR::RAX)] = ctx.make_binary(BVOp::Mul, a, b, 64);
                        if (op == LirOpcode::Umulh) {
                            st.gprs[static_cast<size_t>(GPR::RDX)] = hi_u;
                        } else {
                            BVNodeId z64 = ctx.make_const(0ULL, 64);
                            BVNodeId adj_a = ctx.make_ite(ctx.make_extract(a, 63, 63), b, z64);
                            BVNodeId adj_b = ctx.make_ite(ctx.make_extract(b, 63, 63), a, z64);
                            BVNodeId hi_s = ctx.make_binary(BVOp::Sub, ctx.make_binary(BVOp::Sub, hi_u, adj_a, 64), adj_b, 64);
                            st.gprs[static_cast<size_t>(GPR::RDX)] = hi_s;
                        }
                        break;
                    }
                    case LirOpcode::Cdq: {
                        BVNodeId eax = fit_width(ctx, st.gprs[static_cast<size_t>(GPR::RAX)], 32);
                        BVNodeId msb = ctx.make_extract(eax, 31, 31);
                        BVNodeId edx = ctx.make_zext(
                            ctx.make_ite(msb, ctx.make_const(0xFFFFFFFFu, 32), ctx.make_const(0, 32)), 32);
                        st.gprs[static_cast<size_t>(GPR::RDX)] = edx;
                        st.cdq_eax = eax; st.cdq_edx = edx;
                        break;
                    }
                    case LirOpcode::Cqo: {
                        BVNodeId rax = st.gprs[static_cast<size_t>(GPR::RAX)];
                        BVNodeId rdx = ctx.make_ite(ctx.make_extract(rax, 63, 63),
                                                    ctx.make_const(~0ULL, 64), ctx.make_const(0ULL, 64));
                        st.gprs[static_cast<size_t>(GPR::RDX)] = rdx;
                        st.cqo_rax = rax; st.cqo_rdx = rdx;
                        break;
                    }
                    case LirOpcode::Idiv32: case LirOpcode::Div32: {
                        BVNodeId r32 = fit_width(ctx, st.gprs[static_cast<size_t>(GPR::RAX)], 32);
                        BVNodeId d32_full = st.gprs[static_cast<size_t>(GPR::RDX)];
                        BVNodeId d32 = fit_width(ctx, d32_full, 32);
                        BVNodeId dv = read_op(inst.uses.back(), 32);
                        bool is_s = (op == LirOpcode::Idiv32);
                        uint64_t rdx_c = 1;
                        if (is_s && r32 == st.cdq_eax && d32_full == st.cdq_edx) {
                            BVNodeId ovf = ctx.make_binary(BVOp::And,
                                ctx.make_binary(BVOp::Eq, r32, ctx.make_const(0x80000000ULL, 32), 1),
                                ctx.make_binary(BVOp::Eq, dv, ctx.make_const(0xFFFFFFFFULL, 32), 1), 1);
                            emit_trap(ctx.make_binary(BVOp::Or, ctx.make_binary(BVOp::Eq, dv, ctx.make_const(0, 32), 1), ovf, 1));
                            st.gprs[static_cast<size_t>(GPR::RAX)] = ctx.make_zext(ctx.make_binary(BVOp::SDiv, r32, dv, 32), 32);
                            st.gprs[static_cast<size_t>(GPR::RDX)] = ctx.make_zext(ctx.make_binary(BVOp::SRem, r32, dv, 32), 32);
                            break;
                        }
                        if (!is_s && is_const_value(ctx, d32_full, rdx_c) && (rdx_c & 0xFFFFFFFFULL) == 0) {
                            emit_trap(ctx.make_binary(BVOp::Eq, dv, ctx.make_const(0, 32), 1));
                            st.gprs[static_cast<size_t>(GPR::RAX)] = ctx.make_zext(ctx.make_binary(BVOp::UDiv, r32, dv, 32), 32);
                            st.gprs[static_cast<size_t>(GPR::RDX)] = ctx.make_zext(ctx.make_binary(BVOp::URem, r32, dv, 32), 32);
                            break;
                        }
                        BVNodeId num64 = ctx.make_concat(d32, r32);
                        BVNodeId dv64 = is_s ? ctx.make_sext(dv, 32) : ctx.make_zext(dv, 32);
                        BVNodeId q64 = ctx.make_binary(is_s ? BVOp::SDiv : BVOp::UDiv, num64, dv64, 64);
                        BVNodeId r64 = ctx.make_binary(is_s ? BVOp::SRem : BVOp::URem, num64, dv64, 64);
                        BVNodeId q32 = ctx.make_extract(q64, 31, 0);
                        BVNodeId ovf = is_s ? ctx.make_binary(BVOp::Ne, ctx.make_sext(q32, 32), q64, 1)
                                            : ctx.make_binary(BVOp::Ne, ctx.make_extract(q64, 63, 32), ctx.make_const(0, 32), 1);
                        emit_trap(ctx.make_binary(BVOp::Or, ctx.make_binary(BVOp::Eq, dv, ctx.make_const(0, 32), 1), ovf, 1));
                        st.gprs[static_cast<size_t>(GPR::RAX)] = ctx.make_zext(q32, 32);
                        st.gprs[static_cast<size_t>(GPR::RDX)] = ctx.make_zext(ctx.make_extract(r64, 31, 0), 32);
                        break;
                    }
                    case LirOpcode::Idiv: {
                        BVNodeId rax = st.gprs[static_cast<size_t>(GPR::RAX)];
                        BVNodeId rdx = st.gprs[static_cast<size_t>(GPR::RDX)];
                        BVNodeId dv = read_op(inst.uses.back(), 64);
                        if (rax != st.cqo_rax || rdx != st.cqo_rdx) {
                            return {false, false, "64-bit idiv without preceding cqo is unsupported", 0, 0, 0, 0, {}};
                        }
                        BVNodeId ovf = ctx.make_binary(BVOp::And,
                            ctx.make_binary(BVOp::Eq, rax, ctx.make_const(0x8000000000000000ULL, 64), 1),
                            ctx.make_binary(BVOp::Eq, dv, ctx.make_const(~0ULL, 64), 1), 1);
                        emit_trap(ctx.make_binary(BVOp::Or, ctx.make_binary(BVOp::Eq, dv, ctx.make_const(0, 64), 1), ovf, 1));
                        st.gprs[static_cast<size_t>(GPR::RAX)] = ctx.make_binary(BVOp::SDiv, rax, dv, 64);
                        st.gprs[static_cast<size_t>(GPR::RDX)] = ctx.make_binary(BVOp::SRem, rax, dv, 64);
                        break;
                    }
                    case LirOpcode::Div: {
                        BVNodeId rax = st.gprs[static_cast<size_t>(GPR::RAX)];
                        BVNodeId rdx = st.gprs[static_cast<size_t>(GPR::RDX)];
                        BVNodeId dv = read_op(inst.uses.back(), 64);
                        uint64_t rdx_c = 1;
                        if (!is_const_value(ctx, rdx, rdx_c) || rdx_c != 0) {
                            return {false, false, "64-bit div with non-zero RDX is unsupported", 0, 0, 0, 0, {}};
                        }
                        emit_trap(ctx.make_binary(BVOp::Eq, dv, ctx.make_const(0, 64), 1));
                        st.gprs[static_cast<size_t>(GPR::RAX)] = ctx.make_binary(BVOp::UDiv, rax, dv, 64);
                        st.gprs[static_cast<size_t>(GPR::RDX)] = ctx.make_binary(BVOp::URem, rax, dv, 64);
                        break;
                    }
                    case LirOpcode::And: case LirOpcode::And32:
                    case LirOpcode::Or:  case LirOpcode::Or32:
                    case LirOpcode::Xor: case LirOpcode::Xor32: {
                        uint32_t w = (op == LirOpcode::And32 || op == LirOpcode::Or32 || op == LirOpcode::Xor32) ? 32 : 64;
                        if ((op == LirOpcode::Xor || op == LirOpcode::Xor32) &&
                            inst.uses.size() >= 2 && same_reg_operand(inst.uses[0], inst.uses[1])) {
                            BVNodeId z = ctx.make_const(0, w);
                            write_op(inst.defs[0], z, true);
                            set_logic_flags(ctx, st, z, w);
                            break;
                        }
                        BVOp bop = (op == LirOpcode::And || op == LirOpcode::And32) ? BVOp::And
                                 : (op == LirOpcode::Or  || op == LirOpcode::Or32)  ? BVOp::Or : BVOp::Xor;
                        BVNodeId r = ctx.make_binary(bop, bin_lhs(w), bin_rhs(w), w);
                        write_op(inst.defs[0], r, true);
                        set_logic_flags(ctx, st, r, w);
                        break;
                    }
                    case LirOpcode::Not: case LirOpcode::Not32: {
                        uint32_t w = (op == LirOpcode::Not32) ? 32 : 64;
                        write_op(inst.defs[0], ctx.make_unary(BVOp::Not, read_op(inst.uses[0], w), w), true);
                        break;
                    }
                    case LirOpcode::Neg: case LirOpcode::Neg32: {
                        uint32_t w = (op == LirOpcode::Neg32) ? 32 : 64;
                        BVNodeId a = read_op(inst.uses[0], w), r = ctx.make_unary(BVOp::Neg, a, w);
                        write_op(inst.defs[0], r, true);
                        set_sub_flags(ctx, st, ctx.make_const(0, w), a, r, w);
                        break;
                    }
                    case LirOpcode::Shl: case LirOpcode::Shl32:
                    case LirOpcode::Shr: case LirOpcode::Shr32:
                    case LirOpcode::Sar: case LirOpcode::Sar32: {
                        uint32_t w = (op == LirOpcode::Shl32 || op == LirOpcode::Shr32 || op == LirOpcode::Sar32) ? 32 : 64;
                        BVOp sop = (op == LirOpcode::Shl || op == LirOpcode::Shl32) ? BVOp::Shl
                                 : (op == LirOpcode::Shr || op == LirOpcode::Shr32) ? BVOp::LShr : BVOp::AShr;
                        write_op(inst.defs[0], ctx.make_binary(sop, bin_lhs(w), bin_rhs(w), w), true);
                        break;
                    }
                    case LirOpcode::Popcnt: case LirOpcode::Popcnt32:
                    case LirOpcode::Lzcnt:  case LirOpcode::Lzcnt32:
                    case LirOpcode::Tzcnt:  case LirOpcode::Tzcnt32: {
                        uint32_t w = (op == LirOpcode::Popcnt32 || op == LirOpcode::Lzcnt32 || op == LirOpcode::Tzcnt32) ? 32 : 64;
                        BVOp uop = (op == LirOpcode::Popcnt || op == LirOpcode::Popcnt32) ? BVOp::Popcnt
                                 : (op == LirOpcode::Lzcnt  || op == LirOpcode::Lzcnt32)  ? BVOp::Clz : BVOp::Ctz;
                        write_op(inst.defs[0], ctx.make_unary(uop, read_op(inst.uses[0], w), w), true);
                        break;
                    }
                    case LirOpcode::Cmp: case LirOpcode::Cmp32: {
                        uint32_t w = (op == LirOpcode::Cmp32) ? 32 : 64;
                        BVNodeId a = read_op(inst.uses[0], w), b = read_op(inst.uses[1], w);
                        set_sub_flags(ctx, st, a, b, ctx.make_binary(BVOp::Sub, a, b, w), w);
                        break;
                    }
                    case LirOpcode::Test: case LirOpcode::Test32: {
                        uint32_t w = (op == LirOpcode::Test32) ? 32 : 64;
                        BVNodeId a = read_op(inst.uses[0], w), b = read_op(inst.uses[1], w);
                        set_logic_flags(ctx, st, ctx.make_binary(BVOp::And, a, b, w), w);
                        break;
                    }
                    case LirOpcode::Setcc:
                        write_op(inst.defs[0], ctx.make_zext(eval_cond(ctx, st, inst.condition), 7), false);
                        break;
                    case LirOpcode::Cmovcc: {
                        BVNodeId bit = eval_cond(ctx, st, inst.condition);
                        uint32_t w = (inst.defs[0].size == 4) ? 32 : 64;
                        BVNodeId f_v = read_op(inst.uses.size() >= 2 ? inst.uses[0] : inst.defs[0], w);
                        write_op(inst.defs[0], ctx.make_ite(bit, read_op(inst.uses.back(), w), f_v), true);
                        break;
                    }
                    case LirOpcode::Movsd: case LirOpcode::Movss:
                    case LirOpcode::Movq_gx: case LirOpcode::Movq_xg:
                    case LirOpcode::Movd_gx: case LirOpcode::Movd_xg: {
                        uint32_t w = (op == LirOpcode::Movss || op == LirOpcode::Movd_gx || op == LirOpcode::Movd_xg) ? 32 : 64;
                        write_op(inst.defs[0], read_op(inst.uses[0], w), true);
                        break;
                    }
                    case LirOpcode::Addsd: case LirOpcode::Addss:
                    case LirOpcode::Subsd: case LirOpcode::Subss:
                    case LirOpcode::Mulsd: case LirOpcode::Mulss:
                    case LirOpcode::Divsd: case LirOpcode::Divss:
                    case LirOpcode::Minsd: case LirOpcode::Minss:
                    case LirOpcode::Maxsd: case LirOpcode::Maxss: {
                        uint32_t w = (op == LirOpcode::Addss || op == LirOpcode::Subss ||
                                      op == LirOpcode::Mulss || op == LirOpcode::Divss ||
                                      op == LirOpcode::Minss || op == LirOpcode::Maxss) ? 32 : 64;
                        BVOp fop = (op == LirOpcode::Addsd || op == LirOpcode::Addss) ? BVOp::FAdd
                                 : (op == LirOpcode::Subsd || op == LirOpcode::Subss) ? BVOp::FSub
                                 : (op == LirOpcode::Mulsd || op == LirOpcode::Mulss) ? BVOp::FMul
                                 : (op == LirOpcode::Divsd || op == LirOpcode::Divss) ? BVOp::FDiv
                                 : (op == LirOpcode::Minsd || op == LirOpcode::Minss) ? BVOp::FMin : BVOp::FMax;
                        write_op(inst.defs[0], ctx.make_binary(fop, bin_lhs(w), bin_rhs(w), w), true);
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
                        write_op(inst.defs[0], ctx.make_unary(uop, read_op(inst.uses[0], w), w), true);
                        break;
                    }
                    case LirOpcode::Cvtsi2sd: case LirOpcode::Cvtsi2sd32:
                    case LirOpcode::Cvtsi2ss: case LirOpcode::Cvtsi2ss32: {
                        uint32_t sw = (op == LirOpcode::Cvtsi2sd32 || op == LirOpcode::Cvtsi2ss32) ? 32 : 64;
                        uint32_t dw = (op == LirOpcode::Cvtsi2ss   || op == LirOpcode::Cvtsi2ss32) ? 32 : 64;
                        write_op(inst.defs[0], ctx.make_unary(BVOp::SiToFp, read_op(inst.uses.back(), sw), dw), true);
                        break;
                    }
                    case LirOpcode::Cvttsd2si: case LirOpcode::Cvttsd2si32:
                    case LirOpcode::Cvttss2si: case LirOpcode::Cvttss2si32: {
                        uint32_t sw = (op == LirOpcode::Cvttss2si   || op == LirOpcode::Cvttss2si32) ? 32 : 64;
                        uint32_t dw = (op == LirOpcode::Cvttsd2si32 || op == LirOpcode::Cvttss2si32) ? 32 : 64;
                        write_op(inst.defs[0], ctx.make_unary(BVOp::FpToSi, read_op(inst.uses[0], sw), dw), true);
                        break;
                    }
                    case LirOpcode::Cvtsd2ss:
                        write_op(inst.defs[0], ctx.make_unary(BVOp::FpTrunc, read_op(inst.uses[0], 64), 32), true);
                        break;
                    case LirOpcode::Cvtss2sd:
                        write_op(inst.defs[0], ctx.make_unary(BVOp::FpExt, read_op(inst.uses[0], 32), 64), true);
                        break;
                    case LirOpcode::Ucomisd: case LirOpcode::Ucomiss: {
                        uint32_t w = (op == LirOpcode::Ucomiss) ? 32 : 64;
                        BVNodeId a = read_op(inst.uses[0], w), b = read_op(inst.uses[1], w);
                        BVNodeId unord = ctx.make_binary(BVOp::Or,
                            ctx.make_unary(BVOp::Not, ctx.make_binary(BVOp::FEq, a, a, 1), 1),
                            ctx.make_unary(BVOp::Not, ctx.make_binary(BVOp::FEq, b, b, 1), 1), 1);
                        st.zf = ctx.make_binary(BVOp::Or, unord, ctx.make_binary(BVOp::FEq, a, b, 1), 1);
                        st.cf = ctx.make_binary(BVOp::Or, unord, ctx.make_binary(BVOp::FLt, a, b, 1), 1);
                        st.pf = unord;
                        st.sf = st.of = ctx.make_const(0, 1);
                        break;
                    }
                    case LirOpcode::Xorpd: case LirOpcode::Xorps: case LirOpcode::Pxor: {
                        uint32_t w = (op == LirOpcode::Xorps) ? 64 : 128;
                        if (inst.uses.size() >= 2 && same_reg_operand(inst.uses[0], inst.uses[1])) {
                            write_op(inst.defs[0], ctx.make_const(0, 128), true);
                        } else {
                            write_op(inst.defs[0], ctx.make_binary(BVOp::Xor, bin_lhs(w), bin_rhs(w), w), true);
                        }
                        break;
                    }
                    case LirOpcode::Jmp:
                        if (inst.uses.empty() || !inst.uses[0].is_label()) {
                            return {false, false, "Indirect jmp unsupported in x64 ISel verifier", 0, 0, 0, 0, {}};
                        }
                        if (!has_branch) {
                            route_label(ri, level, cur_cond, inst.uses[0].label_id, st);
                            terminated = true;
                        } else {
                            fallthrough_label = inst.uses[0].label_id;
                        }
                        break;
                    case LirOpcode::Jcc:
                        if (inst.uses.empty() || !inst.uses[0].is_label()) {
                            return {false, false, "Jcc missing target label", 0, 0, 0, 0, {}};
                        }
                        has_branch = true;
                        branch_taken = eval_cond(ctx, st, inst.condition);
                        taken_label = inst.uses[0].label_id;
                        break;
                    case LirOpcode::Ret: {
                        bool first_ret = (ctx.nodes()[out.returned].op == BVOp::Const &&
                                          ctx.nodes()[out.returned].imm == 0);
                        out.returned = ctx.make_binary(BVOp::Or, out.returned, cur_cond, 1);
                        out.completed = ctx.make_binary(BVOp::Or, out.completed, cur_cond, 1);
                        if (!mir_fn.return_type().is_void()) {
                            BVNodeId rv = mir_fn.return_type().is_float()
                                ? fit_width(ctx, st.xmms[0], ret_w, false)
                                : fit_width(ctx, st.gprs[static_cast<size_t>(GPR::RAX)], ret_w, false);
                            out.ret_val = first_ret ? rv : ctx.make_ite(cur_cond, rv, out.ret_val);
                        }
                        terminated = true;
                        break;
                    }
                    case LirOpcode::Trap:
                        out.trapped = ctx.make_binary(BVOp::Or, out.trapped, cur_cond, 1);
                        out.completed = ctx.make_binary(BVOp::Or, out.completed, cur_cond, 1);
                        cur_cond = ctx.make_const(0, 1);
                        terminated = true;
                        break;
                    default:
                        return {false, false,
                                "Unsupported LIR opcode in x64 ISel verifier: " + std::string(codegen::to_string(op)),
                                0, 0, 0, 0, {}};
                }
            }

            if (has_branch) {
                route_label(ri, level, ctx.make_binary(BVOp::And, cur_cond, branch_taken, 1), taken_label, st);
                BVNodeId not_taken = ctx.make_binary(BVOp::And, cur_cond, ctx.make_unary(BVOp::Not, branch_taken, 1), 1);
                if (fallthrough_label != UINT32_MAX) {
                    route_label(ri, level, not_taken, fallthrough_label, st);
                } else if (postorder[ri] + 1 < lir_fn.blocks.size()) {
                    route_label(ri, level, not_taken, lir_fn.blocks[postorder[ri] + 1]->id, st);
                }
            } else if (!terminated && postorder[ri] + 1 < lir_fn.blocks.size()) {
                route_label(ri, level, cur_cond, lir_fn.blocks[postorder[ri] + 1]->id, st);
            }
        }
    }
    return out;
}

} // namespace

EquivResult verify_x64_isel_equivalence(
    const Function& mir_fn, const LirFunction& lir_fn, const EquivOptions& opts) {

    EquivResult res;
    BVContext ctx;
    std::vector<BVNodeId> param_vars;
    for (size_t i = 0; i < mir_fn.param_count(); ++i) {
        Type pt = mir_fn.param_type(i);
        uint32_t w = mir_type_width(pt);
        if (w == 0) {
            res.status = EquivStatus::Unsupported;
            res.diagnostic = "Unsupported parameter type in x64 ISel verifier";
            return res;
        }
        param_vars.push_back(ctx.make_var("arg" + std::to_string(i), w, pt.is_float()));
    }
    if (!mir_fn.return_type().is_void() && mir_type_width(mir_fn.return_type()) == 0) {
        res.status = EquivStatus::Unsupported;
        res.diagnostic = "Unsupported return type in x64 ISel verifier";
        return res;
    }

    InitMemEnv init_mem;
    SymExecOutput out_mir = symbolically_execute_mir_region(ctx, mir_fn, param_vars, init_mem, opts, {});
    if (!out_mir.supported) {
        res.status = EquivStatus::Unsupported;
        res.diagnostic = out_mir.error_msg;
        return res;
    }
    LirExecResult out_lir = symbolically_execute_lir(ctx, mir_fn, lir_fn, param_vars, init_mem, opts);
    if (!out_lir.supported) {
        res.status = EquivStatus::Unsupported;
        res.diagnostic = out_lir.error_msg;
        return res;
    }

    BVNodeId trap_diff = ctx.make_binary(BVOp::Xor, out_mir.trapped, out_lir.trapped, 1);
    BVNodeId no_trap = ctx.make_binary(
        BVOp::And, ctx.make_unary(BVOp::Not, out_mir.trapped, 1),
        ctx.make_unary(BVOp::Not, out_lir.trapped, 1), 1);
    BVNodeId both_ret = ctx.make_binary(BVOp::And, out_mir.returned, out_lir.returned, 1);
    BVNodeId ret_diff = ctx.make_const(0, 1);
    if (!mir_fn.return_type().is_void()) {
        ret_diff = ctx.make_binary(BVOp::And, ctx.make_binary(BVOp::And, no_trap, both_ret, 1),
                                   ctx.make_binary(BVOp::Ne, out_mir.ret_val, out_lir.ret_val, 1), 1);
    }

    BVNodeId mem_diff = ctx.make_const(0, 1);
    if (opts.check_memory) {
        std::vector<BVNodeId> check_addrs;
        auto add_addrs = [&](const std::vector<SymStore>& stores) {
            for (const auto& s : stores) {
                if (s.is_local_alloca) continue;
                for (uint32_t b = 0; b < s.num_bytes; ++b) {
                    BVNodeId ba = (b == 0) ? s.addr : ctx.make_binary(BVOp::Add, s.addr, ctx.make_const(b, 64), 64);
                    if (std::find(check_addrs.begin(), check_addrs.end(), ba) == check_addrs.end()) {
                        check_addrs.push_back(ba);
                    }
                }
            }
        };
        add_addrs(out_mir.stores);
        add_addrs(out_lir.stores);
        for (BVNodeId ba : check_addrs) {
            BVNodeId mb = read_byte_at(ctx, init_mem, out_mir.stores, ba);
            BVNodeId lb = read_byte_at(ctx, init_mem, out_lir.stores, ba);
            mem_diff = ctx.make_binary(BVOp::Or, mem_diff,
                ctx.make_binary(BVOp::And, no_trap, ctx.make_binary(BVOp::Ne, mb, lb, 1), 1), 1);
        }
    }

    BVNodeId mismatch = ctx.make_binary(BVOp::Or, trap_diff, ctx.make_binary(BVOp::Or, ret_diff, mem_diff, 1), 1);
    SolveOutcome solved = solve_nonzero(ctx, mismatch, opts.max_sat_conflicts);
    if (solved.status == SolveOutcome::Unsat) {
        if (out_mir.hit_truncation || out_lir.hit_truncation) {
            res.status = EquivStatus::Unknown;
            res.diagnostic = "Bounded loop unrolling truncated x64 ISel check";
            return res;
        }
        res.status = EquivStatus::ProvenEquivalent;
        return res;
    }
    if (solved.status == SolveOutcome::Unknown) {
        res.status = EquivStatus::Unknown;
        res.diagnostic = "Solver returned Unknown for x64 ISel check";
        return res;
    }

    res.status = EquivStatus::Counterexample;
    std::ostringstream diag;
    diag << "MIR != x64 LIR counterexample in @" << mir_fn.name()
         << " (trap_diff=" << ctx.eval_concrete(trap_diff, solved.model)
         << ", ret_diff=" << ctx.eval_concrete(ret_diff, solved.model)
         << ", mem_diff=" << ctx.eval_concrete(mem_diff, solved.model)
         << ", mir_ret=0x" << std::hex << ctx.eval_concrete(out_mir.ret_val, solved.model)
         << ", lir_ret=0x" << ctx.eval_concrete(out_lir.ret_val, solved.model) << ")";
    res.diagnostic = diag.str();
    return res;
}

EquivResult verify_x64_isel(const Function& mir_fn, bool avx, const EquivOptions& opts) {
    Target target = Target::x64_linux();
    CallingConvention cc = CallingConvention::sysv64();
    x64::X64ISel isel(target, cc);
    isel.set_avx(avx);
    std::unique_ptr<LirFunction> lir_fn = isel.lower(mir_fn);
    if (!lir_fn) {
        return {EquivStatus::Unsupported, {}, "X64ISel::lower returned null"};
    }
    return verify_x64_isel_equivalence(mir_fn, *lir_fn, opts);
}

} // namespace brass::verify
