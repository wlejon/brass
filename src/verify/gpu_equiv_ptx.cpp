#include "gpu_equiv_internal.hpp"
#include <brass/target/ptx/ptx_cleanup.hpp>
#include <brass/target/ptx/ptx_isel.hpp>
#include <cstring>
#include <unordered_map>

namespace brass::verify {

namespace {

uint64_t ptx_reg_key(ptx::Reg r) noexcept {
    return (static_cast<uint64_t>(r.cls) << 32) | static_cast<uint64_t>(r.index);
}

uint32_t ptx_reg_width(ptx::Reg r) noexcept {
    switch (r.cls) {
    case ptx::RegClass::Pred: return 1;
    case ptx::RegClass::B32:
    case ptx::RegClass::F32:  return 32;
    case ptx::RegClass::B64:
    case ptx::RegClass::F64:  return 64;
    }
    return 32;
}

uint32_t ptx_type_width(ptx::Type t) noexcept {
    uint32_t w = ptx::bit_width(t);
    return w ? w : 32;
}

std::unordered_map<uint64_t, BVNodeId> specialize_regs_for_guard(
    const BVContext& ctx,
    const std::unordered_map<uint64_t, BVNodeId>& regs,
    BVNodeId g,
    bool g_is_true) {
    std::unordered_map<uint64_t, BVNodeId> out = regs;
    for (auto& [k, v] : out) {
        const BVNode& n = bv_node(ctx, v);
        if (n.op != BVOp::Ite) continue;
        if (n.a == g) {
            v = g_is_true ? n.b : n.c;
        } else if (bv_node(ctx, n.a).op == BVOp::Not && bv_node(ctx, n.a).a == g) {
            v = g_is_true ? n.c : n.b;
        }
    }
    return out;
}

} // namespace

GpuExecTrace execute_gpu_ptx(
    BVContext& ctx,
    const ptx::Function& fn,
    GpuSymEnv& env,
    const EquivOptions& opts) {
    GpuExecTrace trace;
    trace.completed = bv_true(ctx);
    trace.trapped = bv_false(ctx);
    trace.returned = bv_false(ctx);
    trace.domain_assumption = bv_true(ctx);

    if (fn.blocks.empty()) return trace;

    if (env.param_vars.empty()) {
        std::vector<Type> inferred;
        for (const auto& p : fn.params) {
            if (p.type == ptx::Type::f32) inferred.push_back(Type::f32());
            else if (p.type == ptx::Type::f64) inferred.push_back(Type::f64());
            else if (ptx::bit_width(p.type) == 64) inferred.push_back(Type::ptr());
            else inferred.push_back(Type::i32());
        }
        if (env.coord_bounds == 0) {
            init_gpu_sym_env(ctx, env, inferred);
        } else {
            for (size_t i = 0; i < inferred.size(); ++i) {
                uint32_t w = gpu_type_bv_width(inferred[i]);
                env.param_vars.push_back(ctx.make_var("gpu_param_" + std::to_string(i), w ? w : 64));
            }
            env.param_types = inferred;
        }
    }

    std::unordered_map<std::string, size_t> param_map;
    for (size_t i = 0; i < fn.params.size(); ++i) {
        param_map[fn.params[i].name] = i;
    }

    std::unordered_map<std::string, uint32_t> shared_map;
    for (size_t i = 0; i < fn.shared.size(); ++i) {
        const auto& sd = fn.shared[i];
        uint32_t arr_idx = static_cast<uint32_t>(i);
        uint32_t ebytes = ptx::bit_width(sd.type) / 8;
        if (ebytes == 0) ebytes = 4;
        uint64_t sym_base = kGpuSharedBase0 + static_cast<uint64_t>(arr_idx) * kGpuSharedStride;
        trace.shared_allocs.push_back({arr_idx, sd.count, ebytes, sym_base});
        shared_map[sd.name] = arr_idx;
    }

    std::unordered_map<std::string, size_t> label_to_idx;
    for (size_t i = 0; i < fn.blocks.size(); ++i) {
        label_to_idx[fn.blocks[i]->label] = i;
    }

    struct ActiveState {
        size_t block_idx = 0;
        BVNodeId cond = 0;
        std::unordered_map<uint64_t, BVNodeId> regs;
    };

    std::vector<ActiveState> worklist;
    worklist.push_back({0, bv_true(ctx), {}});

    std::unordered_map<size_t, uint32_t> visits;
    uint32_t max_visits = opts.loop_unroll_bound + 2;
    uint32_t total_steps = 0;
    uint32_t barrier_seq = 0;

    std::unordered_map<BVNodeId, BVNodeId> mul_log2e_arg;
    std::unordered_map<BVNodeId, BVNodeId> lg2_arg;

    auto route_edge = [&](size_t dst_idx, BVNodeId edge_cond,
                          const std::unordered_map<uint64_t, BVNodeId>& next_regs) {
        if (bv_node(ctx, edge_cond).op == BVOp::Const && bv_node(ctx, edge_cond).imm == 0) return;
        if (dst_idx >= fn.blocks.size()) return;
        for (auto& st : worklist) {
            if (st.block_idx == dst_idx) {
                BVNodeId merged = bv_bin(ctx, BVOp::Or, 1, st.cond, edge_cond);
                for (auto& [rk, rv] : next_regs) {
                    auto it = st.regs.find(rk);
                    if (it == st.regs.end()) {
                        st.regs[rk] = rv;
                    } else if (it->second != rv && ctx.width(it->second) == ctx.width(rv)) {
                        it->second = bv_ite(ctx, ctx.width(rv), edge_cond, rv, it->second);
                    }
                }
                st.cond = merged;
                return;
            }
        }
        worklist.push_back({dst_idx, edge_cond, next_regs});
    };

    while (!worklist.empty()) {
        if (++total_steps > opts.max_paths * 2) {
            trace.supported = false;
            trace.error_msg = "PTX symbolic execution exceeded block step budget";
            return trace;
        }
        ActiveState cur = std::move(worklist.front());
        worklist.erase(worklist.begin());
        if (bv_node(ctx, cur.cond).op == BVOp::Const && bv_node(ctx, cur.cond).imm == 0) continue;

        if (++visits[cur.block_idx] > max_visits) {
            BVNodeId not_c = bv_un(ctx, BVOp::Not, 1, cur.cond);
            trace.completed = bv_bin(ctx, BVOp::And, 1, trace.completed, not_c);
            continue;
        }

        const ptx::Block& bb = *fn.blocks[cur.block_idx];
        auto get_reg = [&](ptx::Reg r) -> BVNodeId {
            uint64_t k = ptx_reg_key(r);
            auto it = cur.regs.find(k);
            if (it != cur.regs.end()) return it->second;
            return bv_const(ctx, ptx_reg_width(r), 0);
        };

        auto eval_special = [&](ptx::SpecialReg sr) -> BVNodeId {
            switch (sr) {
            case ptx::SpecialReg::tid_x: return env.tid[0];
            case ptx::SpecialReg::tid_y: return env.tid[1];
            case ptx::SpecialReg::tid_z: return env.tid[2];
            case ptx::SpecialReg::ntid_x: return env.ntid[0];
            case ptx::SpecialReg::ntid_y: return env.ntid[1];
            case ptx::SpecialReg::ntid_z: return env.ntid[2];
            case ptx::SpecialReg::ctaid_x: return env.ctaid[0];
            case ptx::SpecialReg::ctaid_y: return env.ctaid[1];
            case ptx::SpecialReg::ctaid_z: return env.ctaid[2];
            case ptx::SpecialReg::nctaid_x: return env.nctaid[0];
            case ptx::SpecialReg::nctaid_y: return env.nctaid[1];
            case ptx::SpecialReg::nctaid_z: return env.nctaid[2];
            case ptx::SpecialReg::laneid:  return env.laneid;
            default:
                return bv_const(ctx, ptx::reg_class_for(sr) == ptx::RegClass::B64 ? 64 : 32, 0);
            }
        };

        auto eval_op = [&](const ptx::Operand& op, uint32_t target_w, bool sign_ext = false) -> BVNodeId {
            switch (op.kind) {
            case ptx::OperandKind::Reg:
                return gpu_coerce_width(ctx, get_reg(op.reg_val), target_w, sign_ext);
            case ptx::OperandKind::ImmInt:
                return bv_const(ctx, target_w, static_cast<uint64_t>(op.imm_int));
            case ptx::OperandKind::ImmFloat: {
                if (op.imm_is_f32 || target_w == 32) {
                    float f = static_cast<float>(op.imm_float);
                    uint32_t bits = 0;
                    std::memcpy(&bits, &f, 4);
                    return bv_const(ctx, 32, bits);
                }
                double d = op.imm_float;
                uint64_t bits = 0;
                std::memcpy(&bits, &d, 8);
                return bv_const(ctx, 64, bits);
            }
            case ptx::OperandKind::Special:
                return gpu_coerce_width(ctx, eval_special(op.special_reg), target_w, false);
            case ptx::OperandKind::Symbol:
                if (shared_map.count(op.name)) {
                    uint32_t aidx = shared_map[op.name];
                    return bv_const(ctx, 64, trace.shared_allocs[aidx].sym_base_addr);
                }
                return bv_const(ctx, target_w, 0);
            case ptx::OperandKind::Addr: {
                BVNodeId base = op.addr_base.valid()
                    ? gpu_coerce_width(ctx, get_reg(op.addr_base), 64, false)
                    : (shared_map.count(op.addr_symbol)
                        ? bv_const(ctx, 64, trace.shared_allocs[shared_map[op.addr_symbol]].sym_base_addr)
                        : bv_const(ctx, 64, 0));
                if (op.disp != 0) {
                    base = gpu_add_addr64(ctx, base, bv_const(ctx, 64, static_cast<uint64_t>(static_cast<int64_t>(op.disp))));
                }
                return base;
            }
            default:
                return bv_const(ctx, target_w, 0);
            }
        };

        bool terminated = false;
        for (const auto& inst : bb.insts) {
            if (bv_node(ctx, cur.cond).op == BVOp::Const && bv_node(ctx, cur.cond).imm == 0) {
                terminated = true;
                break;
            }
            BVNodeId guard_v = bv_true(ctx);
            if (inst.has_guard) {
                guard_v = get_reg(inst.guard_reg);
                if (inst.guard_negated) guard_v = bv_un(ctx, BVOp::Not, 1, guard_v);
            }
            BVNodeId eff_cond = bv_bin(ctx, BVOp::And, 1, cur.cond, guard_v);

            if (inst.op == ptx::Opcode::bra) {
                const std::string& target_lbl = inst.srcs[0].name;
                auto lit = label_to_idx.find(target_lbl);
                size_t dst_idx = (lit != label_to_idx.end()) ? lit->second : fn.blocks.size();
                if (inst.has_guard) {
                    auto taken_regs = specialize_regs_for_guard(ctx, cur.regs, guard_v, true);
                    route_edge(dst_idx, eff_cond, taken_regs);
                    cur.regs = specialize_regs_for_guard(ctx, cur.regs, guard_v, false);
                    cur.cond = bv_bin(ctx, BVOp::And, 1, cur.cond, bv_un(ctx, BVOp::Not, 1, guard_v));
                } else {
                    route_edge(dst_idx, cur.cond, cur.regs);
                    cur.cond = bv_false(ctx);
                    terminated = true;
                    break;
                }
                continue;
            }
            if (inst.op == ptx::Opcode::ret || inst.op == ptx::Opcode::exit) {
                trace.returned = bv_bin(ctx, BVOp::Or, 1, trace.returned, eff_cond);
                if (!inst.has_guard) {
                    cur.cond = bv_false(ctx);
                    terminated = true;
                    break;
                }
                cur.cond = bv_bin(ctx, BVOp::And, 1, cur.cond, bv_un(ctx, BVOp::Not, 1, guard_v));
                continue;
            }
            if (inst.op == ptx::Opcode::trap) {
                if (inst.has_guard) {
                    // Guarded trap emitted by PtxISel::lower_int_div_rem on divisor == 0
                    BVNodeId not_eff = bv_un(ctx, BVOp::Not, 1, eff_cond);
                    trace.domain_assumption = bv_bin(ctx, BVOp::And, 1, trace.domain_assumption, not_eff);
                    cur.cond = bv_bin(ctx, BVOp::And, 1, cur.cond, bv_un(ctx, BVOp::Not, 1, guard_v));
                } else {
                    trace.trapped = bv_bin(ctx, BVOp::Or, 1, trace.trapped, eff_cond);
                    cur.cond = bv_false(ctx);
                    terminated = true;
                    break;
                }
                continue;
            }

            uint32_t iw = ptx_type_width(inst.type);
            uint32_t dw = (!inst.dsts.empty() && inst.dsts[0].is_reg())
                ? ptx_reg_width(inst.dsts[0].reg_val) : iw;
            BVNodeId res = 0;

            switch (inst.op) {
            case ptx::Opcode::mov:
                res = eval_op(inst.srcs[0], dw, false);
                break;
            case ptx::Opcode::ld: {
                if (inst.state_space == ptx::StateSpace::param) {
                    std::string pname = inst.srcs[0].is_param() ? inst.srcs[0].name : inst.srcs[0].addr_symbol;
                    size_t pidx = param_map.count(pname) ? param_map[pname] : 0;
                    res = (pidx < env.param_vars.size())
                        ? gpu_coerce_width(ctx, env.param_vars[pidx], dw)
                        : bv_const(ctx, dw, 0);
                } else if (inst.state_space == ptx::StateSpace::shared) {
                    BVNodeId addr = eval_op(inst.srcs[0], 64);
                    uint32_t nb = iw / 8;
                    if (nb == 0) nb = 4;
                    uint32_t arr_idx = 0;
                    BVNodeId boff = 0, eidx = 0;
                    resolve_shared_address(ctx, addr, trace.shared_allocs, arr_idx, boff, eidx);
                    record_shared_access(ctx, trace, eff_cond, false, arr_idx, eidx, nb);
                    BVNodeId caddr = make_shared_canonical_addr(ctx, arr_idx, eidx, nb);
                    BVNodeId raw = env.read_mem(ctx, true, trace.shared_stores, caddr, nb);
                    res = gpu_coerce_width(ctx, raw, dw, ptx::is_signed(inst.type));
                } else {
                    BVNodeId addr = eval_op(inst.srcs[0], 64);
                    uint32_t nb = iw / 8;
                    if (nb == 0) nb = 4;
                    record_global_access(ctx, trace, env, eff_cond, false, addr, nb);
                    BVNodeId raw = env.read_mem(ctx, false, trace.global_stores, addr, nb);
                    res = gpu_coerce_width(ctx, raw, dw, ptx::is_signed(inst.type));
                }
                break;
            }
            case ptx::Opcode::st: {
                BVNodeId addr = eval_op(inst.srcs[0], 64);
                uint32_t nb = iw / 8;
                if (nb == 0) nb = 4;
                BVNodeId val = gpu_coerce_width(ctx, eval_op(inst.srcs[1], nb * 8), nb * 8);
                if (inst.state_space == ptx::StateSpace::shared) {
                    uint32_t arr_idx = 0;
                    BVNodeId boff = 0, eidx = 0;
                    resolve_shared_address(ctx, addr, trace.shared_allocs, arr_idx, boff, eidx);
                    record_shared_access(ctx, trace, eff_cond, true, arr_idx, eidx, nb);
                    BVNodeId caddr = make_shared_canonical_addr(ctx, arr_idx, eidx, nb);
                    trace.shared_stores.push_back({eff_cond, caddr, nb, val});
                } else {
                    record_global_access(ctx, trace, env, eff_cond, true, addr, nb);
                    trace.global_stores.push_back({eff_cond, addr, nb, val});
                }
                break;
            }
            case ptx::Opcode::add:
            case ptx::Opcode::sub: {
                BVNodeId a = eval_op(inst.srcs[0], dw);
                BVNodeId b = eval_op(inst.srcs[1], dw);
                if (ptx::is_float(inst.type)) {
                    res = env.eval_uf(ctx, inst.op == ptx::Opcode::add ? "fp_add" : "fp_sub", {a, b}, dw, true);
                } else if (inst.op == ptx::Opcode::add && dw == 64) {
                    res = gpu_add_addr64(ctx, a, b);
                } else {
                    res = bv_bin(ctx, inst.op == ptx::Opcode::add ? BVOp::Add : BVOp::Sub, dw, a, b);
                }
                break;
            }
            case ptx::Opcode::mul: {
                if (ptx::is_float(inst.type)) {
                    BVNodeId a = eval_op(inst.srcs[0], dw);
                    BVNodeId b = eval_op(inst.srcs[1], dw);
                    if (dw == 32 && bv_node(ctx, b).op == BVOp::Const) {
                        uint32_t cbits = static_cast<uint32_t>(bv_node(ctx, b).imm);
                        if (cbits == 0x3FB8AA3Bu) { // 1.44269504f (log2(e))
                            res = env.eval_uf(ctx, "fp_mul", {a, b}, 32, true);
                            mul_log2e_arg[res] = a;
                            break;
                        }
                        if (cbits == 0x3F317218u && lg2_arg.count(a)) { // 0.69314718f (ln(2))
                            res = env.eval_uf(ctx, "gpu_log", {lg2_arg[a]}, 32, true);
                            break;
                        }
                    }
                    res = env.eval_uf(ctx, "fp_mul", {a, b}, dw, true);
                } else if (inst.mul_mode == ptx::MulMode::wide) {
                    bool is_s = ptx::is_signed(inst.type);
                    BVNodeId a64 = bv_ext(ctx, is_s, 64, eval_op(inst.srcs[0], 32));
                    BVNodeId b64 = bv_ext(ctx, is_s, 64, eval_op(inst.srcs[1], 32));
                    res = gpu_mul_or_shl(ctx, 64, a64, b64);
                } else if (inst.mul_mode == ptx::MulMode::hi) {
                    bool is_s = ptx::is_signed(inst.type);
                    BVNodeId a64 = bv_ext(ctx, is_s, 64, eval_op(inst.srcs[0], 32));
                    BVNodeId b64 = bv_ext(ctx, is_s, 64, eval_op(inst.srcs[1], 32));
                    res = bv_extract(ctx, 63, 32, bv_bin(ctx, BVOp::Mul, 64, a64, b64));
                } else {
                    BVNodeId a = eval_op(inst.srcs[0], dw);
                    BVNodeId b = eval_op(inst.srcs[1], dw);
                    res = gpu_mul_or_shl(ctx, dw, a, b);
                }
                break;
            }
            case ptx::Opcode::mad:
            case ptx::Opcode::fma: {
                BVNodeId a = eval_op(inst.srcs[0], dw);
                BVNodeId b = eval_op(inst.srcs[1], dw);
                BVNodeId c = eval_op(inst.srcs[2], dw);
                if (ptx::is_float(inst.type) || inst.op == ptx::Opcode::fma) {
                    res = env.eval_uf(ctx, "fp_fma", {a, b, c}, dw, true);
                } else {
                    res = bv_bin(ctx, BVOp::Add, dw, gpu_mul_or_shl(ctx, dw, a, b), c);
                }
                break;
            }
            case ptx::Opcode::div:
            case ptx::Opcode::rem: {
                BVNodeId a = eval_op(inst.srcs[0], dw);
                BVNodeId b = eval_op(inst.srcs[1], dw);
                if (ptx::is_float(inst.type)) {
                    res = env.eval_uf(ctx, inst.op == ptx::Opcode::div ? "fp_div" : "fp_rem", {a, b}, dw, true);
                } else {
                    BVNodeId b_nz = bv_bin(ctx, BVOp::Ne, 1, b, bv_const(ctx, dw, 0));
                    trace.domain_assumption = bv_bin(ctx, BVOp::And, 1, trace.domain_assumption,
                        bv_bin(ctx, BVOp::Or, 1, bv_un(ctx, BVOp::Not, 1, eff_cond), b_nz));
                    bool is_u = ptx::is_unsigned(inst.type) || ptx::is_bits(inst.type);
                    BVOp bop = (inst.op == ptx::Opcode::div)
                        ? (is_u ? BVOp::UDiv : BVOp::SDiv)
                        : (is_u ? BVOp::URem : BVOp::SRem);
                    res = bv_bin(ctx, bop, dw, a, b);
                }
                break;
            }
            case ptx::Opcode::neg: {
                BVNodeId a = eval_op(inst.srcs[0], dw);
                if (ptx::is_float(inst.type)) {
                    res = bv_bin(ctx, BVOp::Xor, dw, a, bv_const(ctx, dw, 1ULL << (dw - 1)));
                } else {
                    res = bv_un(ctx, BVOp::Neg, dw, a);
                }
                break;
            }
            case ptx::Opcode::abs: {
                BVNodeId a = eval_op(inst.srcs[0], dw);
                res = bv_bin(ctx, BVOp::And, dw, a, bv_const(ctx, dw, ~(1ULL << (dw - 1))));
                break;
            }
            case ptx::Opcode::min:
            case ptx::Opcode::max: {
                BVNodeId a = eval_op(inst.srcs[0], dw);
                BVNodeId b = eval_op(inst.srcs[1], dw);
                if (ptx::is_float(inst.type)) {
                    res = env.eval_uf(ctx, inst.op == ptx::Opcode::min ? "gpu_fmin" : "gpu_fmax", {a, b}, dw, true);
                } else {
                    BVOp cop = ptx::is_signed(inst.type)
                        ? (inst.op == ptx::Opcode::min ? BVOp::Slt : BVOp::Sgt)
                        : (inst.op == ptx::Opcode::min ? BVOp::Ult : BVOp::Ugt);
                    res = bv_ite(ctx, dw, bv_bin(ctx, cop, 1, a, b), a, b);
                }
                break;
            }
            case ptx::Opcode::and_:
            case ptx::Opcode::or_:
            case ptx::Opcode::xor_: {
                BVNodeId a = eval_op(inst.srcs[0], dw);
                BVNodeId b = eval_op(inst.srcs[1], dw);
                BVOp bop = (inst.op == ptx::Opcode::and_) ? BVOp::And
                         : (inst.op == ptx::Opcode::or_)  ? BVOp::Or : BVOp::Xor;
                res = bv_bin(ctx, bop, dw, a, b);
                break;
            }
            case ptx::Opcode::not_:
                res = bv_un(ctx, BVOp::Not, dw, eval_op(inst.srcs[0], dw));
                break;
            case ptx::Opcode::shl:
            case ptx::Opcode::shr: {
                BVNodeId a = eval_op(inst.srcs[0], dw);
                BVNodeId b = eval_op(inst.srcs[1], dw, false);
                BVOp bop = (inst.op == ptx::Opcode::shl) ? BVOp::Shl
                         : (ptx::is_signed(inst.type) ? BVOp::AShr : BVOp::LShr);
                res = bv_bin(ctx, bop, dw, a, b);
                break;
            }
            case ptx::Opcode::setp: {
                uint32_t cw = ptx_type_width(inst.type);
                BVNodeId a = eval_op(inst.srcs[0], cw);
                BVNodeId b = eval_op(inst.srcs[1], cw, ptx::is_signed(inst.type));
                if (ptx::is_float(inst.type)) {
                    const char* cname = (inst.cmp_op == ptx::CmpOp::eq || inst.cmp_op == ptx::CmpOp::equ) ? "fp_cmp_eq"
                                      : (inst.cmp_op == ptx::CmpOp::ne || inst.cmp_op == ptx::CmpOp::neu) ? "fp_cmp_ne"
                                      : (inst.cmp_op == ptx::CmpOp::lt || inst.cmp_op == ptx::CmpOp::ltu) ? "fp_cmp_slt"
                                      : (inst.cmp_op == ptx::CmpOp::le || inst.cmp_op == ptx::CmpOp::leu) ? "fp_cmp_sle"
                                      : (inst.cmp_op == ptx::CmpOp::gt || inst.cmp_op == ptx::CmpOp::gtu) ? "fp_cmp_sgt"
                                      : "fp_cmp_sge";
                    res = env.eval_uf(ctx, cname, {a, b}, 1, false);
                } else {
                    bool is_u = ptx::is_unsigned(inst.type) || ptx::is_bits(inst.type);
                    BVOp cop = BVOp::Eq;
                    switch (inst.cmp_op) {
                    case ptx::CmpOp::eq: cop = BVOp::Eq; break;
                    case ptx::CmpOp::ne: cop = BVOp::Ne; break;
                    case ptx::CmpOp::lt: cop = is_u ? BVOp::Ult : BVOp::Slt; break;
                    case ptx::CmpOp::le: cop = is_u ? BVOp::Ule : BVOp::Sle; break;
                    case ptx::CmpOp::gt: cop = is_u ? BVOp::Ugt : BVOp::Sgt; break;
                    case ptx::CmpOp::ge: cop = is_u ? BVOp::Uge : BVOp::Sge; break;
                    case ptx::CmpOp::lo: cop = BVOp::Ult; break;
                    case ptx::CmpOp::ls: cop = BVOp::Ule; break;
                    case ptx::CmpOp::hi: cop = BVOp::Ugt; break;
                    case ptx::CmpOp::hs: cop = BVOp::Uge; break;
                    default: break;
                    }
                    res = bv_bin(ctx, cop, 1, a, b);
                }
                break;
            }
            case ptx::Opcode::selp: {
                BVNodeId tv = eval_op(inst.srcs[0], dw);
                BVNodeId fv = eval_op(inst.srcs[1], dw);
                BVNodeId c1 = eval_op(inst.srcs[2], 1);
                if (bv_node(ctx, fv).op == BVOp::SDiv || bv_node(ctx, fv).op == BVOp::SRem) {
                    const BVNode& cn = bv_node(ctx, c1);
                    if (cn.op == BVOp::Eq && bv_node(ctx, cn.b).op == BVOp::Const &&
                        bv_node(ctx, cn.b).imm == width_mask(dw)) {
                        res = fv;
                        break;
                    }
                }
                res = bv_ite(ctx, dw, c1, tv, fv);
                break;
            }
            case ptx::Opcode::cvt: {
                if (inst.type == ptx::Type::f32 && inst.src_type == ptx::Type::f16) {
                    res = env.eval_uf(ctx, "gpu_f16_to_f32", {eval_op(inst.srcs[0], 32)}, 32, true);
                } else if (inst.type == ptx::Type::f16 && inst.src_type == ptx::Type::f32) {
                    res = env.eval_uf(ctx, "gpu_f32_to_f16", {eval_op(inst.srcs[0], 32)}, 32, false);
                } else if (ptx::is_float(inst.type) && ptx::is_float(inst.src_type)) {
                    if (inst.type == inst.src_type) {
                        const char* rname = (inst.rounding == ptx::Rounding::rmi) ? "gpu_floor"
                                          : (inst.rounding == ptx::Rounding::rpi) ? "gpu_ceil"
                                          : (inst.rounding == ptx::Rounding::rni) ? "gpu_round" : "gpu_round";
                        res = env.eval_uf(ctx, rname, {eval_op(inst.srcs[0], dw)}, dw, true);
                    } else {
                        res = env.eval_uf(ctx, dw == 64 ? "cvt_fpext" : "cvt_fptrunc", {eval_op(inst.srcs[0], ptx_type_width(inst.src_type))}, dw, true);
                    }
                } else if (ptx::is_float(inst.type) && !ptx::is_float(inst.src_type)) {
                    const char* cname = ptx::is_signed(inst.src_type) ? "cvt_sitofp" : "cvt_uitofp";
                    res = env.eval_uf(ctx, cname, {eval_op(inst.srcs[0], ptx_type_width(inst.src_type))}, dw, true);
                } else if (!ptx::is_float(inst.type) && ptx::is_float(inst.src_type)) {
                    const char* cname = ptx::is_signed(inst.type) ? "cvt_fptosi" : "cvt_fptoui";
                    res = env.eval_uf(ctx, cname, {eval_op(inst.srcs[0], ptx_type_width(inst.src_type))}, dw, false);
                } else {
                    uint32_t sw = ptx_type_width(inst.src_type);
                    BVNodeId sv = eval_op(inst.srcs[0], sw);
                    res = gpu_coerce_width(ctx, sv, dw, ptx::is_signed(inst.src_type));
                }
                break;
            }
            case ptx::Opcode::sqrt:
                res = env.eval_uf(ctx, "gpu_sqrt", {eval_op(inst.srcs[0], dw)}, dw, true);
                break;
            case ptx::Opcode::rsqrt:
                res = env.eval_uf(ctx, "gpu_rsqrt", {eval_op(inst.srcs[0], dw)}, dw, true);
                break;
            case ptx::Opcode::sin:
                res = env.eval_uf(ctx, "gpu_sin", {eval_op(inst.srcs[0], dw)}, dw, true);
                break;
            case ptx::Opcode::cos:
                res = env.eval_uf(ctx, "gpu_cos", {eval_op(inst.srcs[0], dw)}, dw, true);
                break;
            case ptx::Opcode::ex2: {
                BVNodeId a = eval_op(inst.srcs[0], dw);
                if (mul_log2e_arg.count(a)) {
                    res = env.eval_uf(ctx, "gpu_exp", {mul_log2e_arg[a]}, dw, true);
                } else {
                    res = env.eval_uf(ctx, "gpu_ex2", {a}, dw, true);
                }
                break;
            }
            case ptx::Opcode::lg2: {
                BVNodeId a = eval_op(inst.srcs[0], dw);
                res = env.eval_uf(ctx, "gpu_lg2", {a}, dw, true);
                lg2_arg[res] = a;
                break;
            }
            case ptx::Opcode::rcp:
                res = env.eval_uf(ctx, "fp_div", {bv_const(ctx, 32, 0x3F800000u), eval_op(inst.srcs[0], 32)}, 32, true);
                break;
            case ptx::Opcode::activemask:
                res = bv_const(ctx, 32, 0xFFFFFFFFu);
                break;
            case ptx::Opcode::bar: {
                BVNodeId bid = inst.srcs.empty() ? bv_const(ctx, 32, 0) : eval_op(inst.srcs[0], 32);
                trace.barriers.push_back({eff_cond, bid, false, barrier_seq++});
                break;
            }
            case ptx::Opcode::shfl: {
                BVNodeId v32 = eval_op(inst.srcs[0], 32);
                BVNodeId d32 = eval_op(inst.srcs[1], 32);
                uint32_t mode = (inst.shfl_mode == ptx::ShflMode::up) ? 1
                              : (inst.shfl_mode == ptx::ShflMode::bfly) ? 2
                              : (inst.shfl_mode == ptx::ShflMode::idx) ? 3 : 0;
                trace.barriers.push_back({eff_cond, bv_const(ctx, 32, mode), true, barrier_seq++});
                res = env.eval_shuffle_mode(ctx, mode, v32, d32);
                break;
            }
            case ptx::Opcode::atom: {
                BVNodeId addr = eval_op(inst.srcs[0], 64);
                BVNodeId arg = eval_op(inst.srcs[1], dw);
                uint32_t nb = dw / 8;
                bool is_sh = (inst.state_space == ptx::StateSpace::shared);
                BVNodeId caddr = addr;
                if (is_sh) {
                    uint32_t arr_idx = 0;
                    BVNodeId boff = 0, eidx = 0;
                    resolve_shared_address(ctx, addr, trace.shared_allocs, arr_idx, boff, eidx);
                    record_shared_access(ctx, trace, eff_cond, false, arr_idx, eidx, nb);
                    record_shared_access(ctx, trace, eff_cond, true, arr_idx, eidx, nb);
                    caddr = make_shared_canonical_addr(ctx, arr_idx, eidx, nb);
                } else {
                    record_global_access(ctx, trace, env, eff_cond, false, caddr, nb);
                    record_global_access(ctx, trace, env, eff_cond, true, caddr, nb);
                }
                auto& st_vec = is_sh ? trace.shared_stores : trace.global_stores;
                BVNodeId old_v = env.read_mem(ctx, is_sh, st_vec, caddr, nb);
                BVNodeId new_v = arg;
                if (inst.atom_op == ptx::AtomOp::add) {
                    new_v = ptx::is_float(inst.type)
                        ? env.eval_uf(ctx, "fp_add", {old_v, arg}, dw, true)
                        : bv_bin(ctx, BVOp::Add, dw, old_v, arg);
                } else if (inst.atom_op == ptx::AtomOp::min) {
                    new_v = bv_ite(ctx, dw, bv_bin(ctx, BVOp::Slt, 1, old_v, arg), old_v, arg);
                } else if (inst.atom_op == ptx::AtomOp::max) {
                    new_v = bv_ite(ctx, dw, bv_bin(ctx, BVOp::Sgt, 1, old_v, arg), old_v, arg);
                }
                st_vec.push_back({eff_cond, caddr, nb, new_v});
                res = old_v;
                break;
            }
            default:
                break;
            }

            if (!inst.dsts.empty() && inst.dsts[0].is_reg() && res != 0) {
                ptx::Reg dreg = inst.dsts[0].reg_val;
                uint64_t k = ptx_reg_key(dreg);
                res = gpu_coerce_width(ctx, res, ptx_reg_width(dreg));
                if (inst.has_guard) {
                    BVNodeId old_v = get_reg(dreg);
                    cur.regs[k] = bv_ite(ctx, ptx_reg_width(dreg), guard_v, res, old_v);
                } else {
                    cur.regs[k] = res;
                }
            }
        }

        if (!terminated && (bv_node(ctx, cur.cond).op != BVOp::Const || bv_node(ctx, cur.cond).imm != 0)) {
            if (cur.block_idx + 1 < fn.blocks.size()) {
                route_edge(cur.block_idx + 1, cur.cond, cur.regs);
            } else {
                trace.returned = bv_bin(ctx, BVOp::Or, 1, trace.returned, cur.cond);
            }
        }
    }
    return trace;
}

EquivResult verify_ptx_equivalence(
    const Function& mir_fn,
    const ptx::Function& ptx_mod,
    std::string* err_out,
    const EquivOptions& opts) {
    BVContext ctx;
    GpuSymEnv env;
    init_gpu_sym_env(ctx, env, mir_fn.param_types());

    GpuExecTrace mir_trace = execute_gpu_mir(ctx, mir_fn, env, opts);
    GpuExecTrace ptx_trace = execute_gpu_ptx(ctx, ptx_mod, env, opts);
    return compare_gpu_traces(ctx, env, mir_trace, ptx_trace, "MIR", "PTX", err_out, opts);
}

EquivResult verify_ptx_equivalence(
    const Function& mir_fn,
    const target::PtxOptions& ptx_opts,
    std::string* err_out,
    const EquivOptions& opts) {
    try {
        ptx::PtxISel isel;
        ptx::Function ptx_fn = isel.lower(mir_fn);
        if (ptx_opts.cleanup) ptx::cleanup(ptx_fn);
        return verify_ptx_equivalence(mir_fn, ptx_fn, err_out, opts);
    } catch (const std::exception& ex) {
        if (err_out) *err_out = ex.what();
        return EquivResult{EquivStatus::Unsupported, {}, ex.what()};
    }
}

} // namespace brass::verify
