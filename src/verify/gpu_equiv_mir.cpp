#include "gpu_equiv_internal.hpp"
#include <brass/mir/block.hpp>
#include <brass/mir/instruction.hpp>
#include <brass/mir/opcodes.hpp>
#include <cstring>
#include <unordered_map>

namespace brass::verify {

GpuExecTrace execute_gpu_mir(
    BVContext& ctx,
    const Function& fn,
    GpuSymEnv& env,
    const EquivOptions& opts) {
    GpuExecTrace trace;
    trace.completed = bv_true(ctx);
    trace.trapped = bv_false(ctx);
    trace.returned = bv_false(ctx);
    trace.domain_assumption = bv_true(ctx);

    if (fn.blocks().empty()) return trace;

    struct ActiveState {
        const BasicBlock* bb = nullptr;
        BVNodeId cond = 0;
        std::unordered_map<uint32_t, BVNodeId> vals;
    };
    std::vector<ActiveState> worklist;
    ActiveState init;
    init.bb = fn.entry_block();
    init.cond = bv_true(ctx);
    for (size_t i = 0; i < init.bb->param_count() && i < env.param_vars.size(); ++i) {
        if (init.bb->param(i)) {
            init.vals[init.bb->param(i)->id()] = env.param_vars[i];
        }
    }
    worklist.push_back(std::move(init));

    std::unordered_map<const BasicBlock*, uint32_t> visits;
    uint32_t max_visits = opts.loop_unroll_bound + 2;
    uint32_t total_steps = 0;
    uint32_t barrier_seq = 0;

    auto route_edge = [&](const BranchTarget& target, BVNodeId edge_cond,
                          const std::unordered_map<uint32_t, BVNodeId>& cur_vals) {
        if (bv_node(ctx, edge_cond).op == BVOp::Const && bv_node(ctx, edge_cond).imm == 0) return;
        if (!target.block) return;
        std::unordered_map<uint32_t, BVNodeId> next_vals = cur_vals;
        const auto& params = target.block->params();
        for (size_t i = 0; i < params.size() && i < target.args.size(); ++i) {
            if (params[i] && target.args[i]) {
                auto it = cur_vals.find(target.args[i]->id());
                if (it != cur_vals.end()) next_vals[params[i]->id()] = it->second;
            }
        }
        for (auto& st : worklist) {
            if (st.bb == target.block) {
                BVNodeId merged_cond = bv_bin(ctx, BVOp::Or, 1, st.cond, edge_cond);
                for (auto& [vid, vnode] : next_vals) {
                    auto it = st.vals.find(vid);
                    if (it == st.vals.end()) {
                        st.vals[vid] = vnode;
                    } else if (it->second != vnode && ctx.width(it->second) == ctx.width(vnode)) {
                        it->second = bv_ite(ctx, ctx.width(vnode), edge_cond, vnode, it->second);
                    }
                }
                st.cond = merged_cond;
                return;
            }
        }
        worklist.push_back({target.block, edge_cond, std::move(next_vals)});
    };

    while (!worklist.empty()) {
        if (++total_steps > opts.max_paths) {
            trace.supported = false;
            trace.error_msg = "MIR GPU symbolic execution exceeded block step budget";
            return trace;
        }
        ActiveState cur = std::move(worklist.front());
        worklist.erase(worklist.begin());
        if (bv_node(ctx, cur.cond).op == BVOp::Const && bv_node(ctx, cur.cond).imm == 0) continue;

        if (++visits[cur.bb] > max_visits) {
            BVNodeId not_c = bv_un(ctx, BVOp::Not, 1, cur.cond);
            trace.completed = bv_bin(ctx, BVOp::And, 1, trace.completed, not_c);
            continue;
        }

        auto get_v = [&](const Value* v) -> BVNodeId {
            if (!v) return 0;
            auto it = cur.vals.find(v->id());
            if (it != cur.vals.end()) return it->second;
            if (v->is_instruction() && v->defining_instruction()) {
                const Instruction* d = v->defining_instruction();
                if (d->opcode() == Opcode::iconst_i32) {
                    return bv_const(ctx, 32, static_cast<uint32_t>(d->imm_i32()));
                }
                if (d->opcode() == Opcode::iconst_i64) {
                    return bv_const(ctx, 64, static_cast<uint64_t>(d->imm_i64()));
                }
                if (d->opcode() == Opcode::fconst_f64) {
                    if (d->type().kind() == TypeKind::F32) {
                        float f = static_cast<float>(d->imm_f64());
                        uint32_t bits = 0;
                        std::memcpy(&bits, &f, 4);
                        return bv_const(ctx, 32, bits);
                    }
                    double dv = d->imm_f64();
                    uint64_t bits = 0;
                    std::memcpy(&bits, &dv, 8);
                    return bv_const(ctx, 64, bits);
                }
            }
            uint32_t w = gpu_type_bv_width(v->type());
            return bv_const(ctx, w ? w : 32, 0);
        };

        auto as_cond1 = [&](const Value* v) -> BVNodeId {
            BVNodeId b = get_v(v);
            if (ctx.width(b) == 1) return b;
            return bv_bin(ctx, BVOp::Ne, 1, b, bv_const(ctx, ctx.width(b), 0));
        };

        for (const Instruction* inst : *cur.bb) {
            Opcode op = inst->opcode();
            uint32_t rw = gpu_type_bv_width(inst->type());
            if (rw == 0) rw = 32;
            BVNodeId res = 0;

            switch (op) {
            case Opcode::iconst_i32:
                res = bv_const(ctx, 32, static_cast<uint32_t>(inst->imm_i32()));
                break;
            case Opcode::iconst_i64:
                res = bv_const(ctx, 64, static_cast<uint64_t>(inst->imm_i64()));
                break;
            case Opcode::fconst_f64:
                res = get_v(inst->result());
                break;
            case Opcode::add:
            case Opcode::sub:
            case Opcode::mul: {
                BVNodeId a = gpu_coerce_width(ctx, get_v(inst->operand(0)), rw);
                BVNodeId b = gpu_coerce_width(ctx, get_v(inst->operand(1)), rw);
                if (inst->type().is_float()) {
                    const char* fn_name = (op == Opcode::add) ? "fp_add" : (op == Opcode::sub) ? "fp_sub" : "fp_mul";
                    res = env.eval_uf(ctx, fn_name, {a, b}, rw, true);
                } else if (op == Opcode::add && rw == 64) {
                    res = gpu_add_addr64(ctx, a, b);
                } else if (op == Opcode::mul) {
                    res = gpu_mul_or_shl(ctx, rw, a, b);
                } else {
                    res = bv_bin(ctx, op == Opcode::add ? BVOp::Add : BVOp::Sub, rw, a, b);
                }
                break;
            }
            case Opcode::sdiv:
            case Opcode::udiv:
            case Opcode::smod:
            case Opcode::umod: {
                BVNodeId a = gpu_coerce_width(ctx, get_v(inst->operand(0)), rw);
                BVNodeId b = gpu_coerce_width(ctx, get_v(inst->operand(1)), rw);
                if (inst->type().is_float()) {
                    res = env.eval_uf(ctx, (op == Opcode::sdiv || op == Opcode::udiv) ? "fp_div" : "fp_rem", {a, b}, rw, true);
                } else {
                    BVNodeId b_nz = bv_bin(ctx, BVOp::Ne, 1, b, bv_const(ctx, rw, 0));
                    BVNodeId not_reach = bv_un(ctx, BVOp::Not, 1, cur.cond);
                    trace.domain_assumption = bv_bin(
                        ctx, BVOp::And, 1, trace.domain_assumption,
                        bv_bin(ctx, BVOp::Or, 1, not_reach, b_nz));
                    BVOp bv_op = (op == Opcode::sdiv) ? BVOp::SDiv
                               : (op == Opcode::udiv) ? BVOp::UDiv
                               : (op == Opcode::smod) ? BVOp::SRem : BVOp::URem;
                    res = bv_bin(ctx, bv_op, rw, a, b);
                }
                break;
            }
            case Opcode::fma_f32:
            case Opcode::fma_f64: {
                BVNodeId a = gpu_coerce_width(ctx, get_v(inst->operand(0)), rw);
                BVNodeId b = gpu_coerce_width(ctx, get_v(inst->operand(1)), rw);
                BVNodeId c = gpu_coerce_width(ctx, get_v(inst->operand(2)), rw);
                res = env.eval_uf(ctx, "fp_fma", {a, b, c}, rw, true);
                break;
            }
            case Opcode::neg: {
                BVNodeId a = gpu_coerce_width(ctx, get_v(inst->operand(0)), rw);
                if (inst->type().is_float()) {
                    res = bv_bin(ctx, BVOp::Xor, rw, a, bv_const(ctx, rw, 1ULL << (rw - 1)));
                } else {
                    res = bv_un(ctx, BVOp::Neg, rw, a);
                }
                break;
            }
            case Opcode::fabs_f32:
            case Opcode::fabs_f64: {
                BVNodeId a = gpu_coerce_width(ctx, get_v(inst->operand(0)), rw);
                res = bv_bin(ctx, BVOp::And, rw, a, bv_const(ctx, rw, ~(1ULL << (rw - 1))));
                break;
            }
            case Opcode::sqrt_f32:
            case Opcode::sqrt_f64:
                res = env.eval_uf(ctx, "gpu_sqrt", {gpu_coerce_width(ctx, get_v(inst->operand(0)), rw)}, rw, true);
                break;
            case Opcode::floor_f32:
            case Opcode::floor_f64:
                res = env.eval_uf(ctx, "gpu_floor", {gpu_coerce_width(ctx, get_v(inst->operand(0)), rw)}, rw, true);
                break;
            case Opcode::ceil_f32:
            case Opcode::ceil_f64:
                res = env.eval_uf(ctx, "gpu_ceil", {gpu_coerce_width(ctx, get_v(inst->operand(0)), rw)}, rw, true);
                break;
            case Opcode::round_f32:
            case Opcode::round_f64:
                res = env.eval_uf(ctx, "gpu_round", {gpu_coerce_width(ctx, get_v(inst->operand(0)), rw)}, rw, true);
                break;
            case Opcode::fmin_f32:
            case Opcode::fmin_f64:
                res = env.eval_uf(ctx, "gpu_fmin", {
                    gpu_coerce_width(ctx, get_v(inst->operand(0)), rw),
                    gpu_coerce_width(ctx, get_v(inst->operand(1)), rw)}, rw, true);
                break;
            case Opcode::fmax_f32:
            case Opcode::fmax_f64:
                res = env.eval_uf(ctx, "gpu_fmax", {
                    gpu_coerce_width(ctx, get_v(inst->operand(0)), rw),
                    gpu_coerce_width(ctx, get_v(inst->operand(1)), rw)}, rw, true);
                break;
            case Opcode::and_:
            case Opcode::or_:
            case Opcode::xor_: {
                BVNodeId a = gpu_coerce_width(ctx, get_v(inst->operand(0)), rw);
                BVNodeId b = gpu_coerce_width(ctx, get_v(inst->operand(1)), rw);
                BVOp bop = (op == Opcode::and_) ? BVOp::And : (op == Opcode::or_) ? BVOp::Or : BVOp::Xor;
                res = bv_bin(ctx, bop, rw, a, b);
                break;
            }
            case Opcode::not_:
                res = bv_un(ctx, BVOp::Not, rw, gpu_coerce_width(ctx, get_v(inst->operand(0)), rw));
                break;
            case Opcode::shl:
            case Opcode::lshr:
            case Opcode::ashr: {
                BVNodeId a = gpu_coerce_width(ctx, get_v(inst->operand(0)), rw);
                BVNodeId b = gpu_coerce_width(ctx, get_v(inst->operand(1)), rw);
                BVOp bop = (op == Opcode::shl) ? BVOp::Shl : (op == Opcode::lshr) ? BVOp::LShr : BVOp::AShr;
                res = bv_bin(ctx, bop, rw, a, b);
                break;
            }
            case Opcode::eq:
            case Opcode::ne:
            case Opcode::slt:
            case Opcode::sle:
            case Opcode::sgt:
            case Opcode::sge:
            case Opcode::ult:
            case Opcode::ule:
            case Opcode::ugt:
            case Opcode::uge: {
                uint32_t ow = gpu_type_bv_width(inst->operand(0)->type());
                if (ow == 0) ow = 32;
                BVNodeId a = gpu_coerce_width(ctx, get_v(inst->operand(0)), ow);
                BVNodeId b = gpu_coerce_width(ctx, get_v(inst->operand(1)), ow);
                BVNodeId cmp1 = 0;
                if (inst->operand(0)->type().is_float()) {
                    cmp1 = env.eval_uf(ctx, std::string("fp_cmp_") + std::string(opcode_name(op)), {a, b}, 1, false);
                } else {
                    BVOp cop = (op == Opcode::eq)  ? BVOp::Eq  : (op == Opcode::ne)  ? BVOp::Ne
                             : (op == Opcode::slt) ? BVOp::Slt : (op == Opcode::sle) ? BVOp::Sle
                             : (op == Opcode::sgt) ? BVOp::Sgt : (op == Opcode::sge) ? BVOp::Sge
                             : (op == Opcode::ult) ? BVOp::Ult : (op == Opcode::ule) ? BVOp::Ule
                             : (op == Opcode::ugt) ? BVOp::Ugt : BVOp::Uge;
                    cmp1 = bv_bin(ctx, cop, 1, a, b);
                }
                res = bv_ext(ctx, false, 32, cmp1);
                break;
            }
            case Opcode::select: {
                BVNodeId c1 = as_cond1(inst->operand(0));
                BVNodeId tv = gpu_coerce_width(ctx, get_v(inst->operand(1)), rw);
                BVNodeId fv = gpu_coerce_width(ctx, get_v(inst->operand(2)), rw);
                res = bv_ite(ctx, rw, c1, tv, fv);
                break;
            }
            case Opcode::sext_i64:
                res = gpu_coerce_width(ctx, get_v(inst->operand(0)), 64, true);
                break;
            case Opcode::zext_i64:
                res = gpu_coerce_width(ctx, get_v(inst->operand(0)), 64, false);
                break;
            case Opcode::trunc_i32:
                res = gpu_coerce_width(ctx, get_v(inst->operand(0)), 32, false);
                break;
            case Opcode::trunc_i8:
                res = gpu_coerce_width(ctx, get_v(inst->operand(0)), 8, false);
                break;
            case Opcode::bitcast_i32_f32:
            case Opcode::bitcast_f32_i32:
                res = gpu_coerce_width(ctx, get_v(inst->operand(0)), 32, false);
                break;
            case Opcode::bitcast_i64_f64:
            case Opcode::bitcast_f64_i64:
                res = gpu_coerce_width(ctx, get_v(inst->operand(0)), 64, false);
                break;
            case Opcode::sitofp_f32_i32:
            case Opcode::sitofp_f64_i32:
            case Opcode::sitofp_f32_i64:
            case Opcode::sitofp_f64_i64:
                res = env.eval_uf(ctx, "cvt_sitofp", {get_v(inst->operand(0))}, rw, true);
                break;
            case Opcode::fptosi_i32:
            case Opcode::fptosi_i32_f32:
            case Opcode::fptosi_i64:
            case Opcode::fptosi_i64_f32:
                res = env.eval_uf(ctx, "cvt_fptosi", {get_v(inst->operand(0))}, rw, false);
                break;
            case Opcode::fptrunc_f32_f64:
                res = env.eval_uf(ctx, "cvt_fptrunc", {get_v(inst->operand(0))}, 32, true);
                break;
            case Opcode::fpext_f64_f32:
                res = env.eval_uf(ctx, "cvt_fpext", {get_v(inst->operand(0))}, 64, true);
                break;
            case Opcode::load:
            case Opcode::load_indexed: {
                BVNodeId base = gpu_coerce_width(ctx, get_v(inst->operand(0)), 64, false);
                BVNodeId addr = base;
                if (op == Opcode::load_indexed) {
                    BVNodeId idx = gpu_coerce_width(ctx, get_v(inst->operand(1)), 64, true);
                    if (inst->scale() > 1) {
                        idx = gpu_mul_or_shl(ctx, 64, idx, bv_const(ctx, 64, inst->scale()));
                    }
                    addr = gpu_add_addr64(ctx, base, idx);
                }
                if (inst->offset() != 0) {
                    addr = gpu_add_addr64(ctx, addr, bv_const(ctx, 64, static_cast<uint64_t>(inst->offset())));
                }
                uint32_t nbytes = rw / 8;
                record_global_access(ctx, trace, env, cur.cond, false, addr, nbytes);
                res = env.read_mem(ctx, false, trace.global_stores, addr, nbytes);
                break;
            }
            case Opcode::store:
            case Opcode::store_indexed: {
                BVNodeId base = gpu_coerce_width(ctx, get_v(inst->operand(0)), 64, false);
                BVNodeId addr = base;
                const Value* val_op = (op == Opcode::store_indexed) ? inst->operand(2) : inst->operand(1);
                if (op == Opcode::store_indexed) {
                    BVNodeId idx = gpu_coerce_width(ctx, get_v(inst->operand(1)), 64, true);
                    if (inst->scale() > 1) {
                        idx = gpu_mul_or_shl(ctx, 64, idx, bv_const(ctx, 64, inst->scale()));
                    }
                    addr = gpu_add_addr64(ctx, base, idx);
                }
                if (inst->offset() != 0) {
                    addr = gpu_add_addr64(ctx, addr, bv_const(ctx, 64, static_cast<uint64_t>(inst->offset())));
                }
                uint32_t vw = gpu_type_bv_width(val_op->type());
                if (vw == 0) vw = 32;
                BVNodeId val = gpu_coerce_width(ctx, get_v(val_op), vw, false);
                uint32_t nbytes = vw / 8;
                record_global_access(ctx, trace, env, cur.cond, true, addr, nbytes);
                trace.global_stores.push_back({cur.cond, addr, nbytes, val});
                break;
            }
            case Opcode::call: {
                std::string_view sym = inst->symbol();
                if (sym == "ptx_tid_x") res = env.tid[0];
                else if (sym == "ptx_tid_y") res = env.tid[1];
                else if (sym == "ptx_tid_z") res = env.tid[2];
                else if (sym == "ptx_ntid_x") res = env.ntid[0];
                else if (sym == "ptx_ntid_y") res = env.ntid[1];
                else if (sym == "ptx_ntid_z") res = env.ntid[2];
                else if (sym == "ptx_ctaid_x") res = env.ctaid[0];
                else if (sym == "ptx_ctaid_y") res = env.ctaid[1];
                else if (sym == "ptx_ctaid_z") res = env.ctaid[2];
                else if (sym == "ptx_nctaid_x") res = env.nctaid[0];
                else if (sym == "ptx_nctaid_y") res = env.nctaid[1];
                else if (sym == "ptx_nctaid_z") res = env.nctaid[2];
                else if (sym == "ptx_laneid" || sym == "ptx_lane_id") res = env.laneid;
                else if (sym == "ptx_global_tid_x" || sym == "ptx_global_id_x") res = env.global_tid[0];
                else if (sym == "ptx_mul_hi_u32") {
                    BVNodeId a64 = bv_ext(ctx, false, 64, gpu_coerce_width(ctx, get_v(inst->operand(0)), 32));
                    BVNodeId b64 = bv_ext(ctx, false, 64, gpu_coerce_width(ctx, get_v(inst->operand(1)), 32));
                    res = bv_extract(ctx, 63, 32, bv_bin(ctx, BVOp::Mul, 64, a64, b64));
                } else if (sym == "ptx_mul_wide_u32" || sym == "ptx_mul_wide_s32") {
                    bool is_s = (sym == "ptx_mul_wide_s32");
                    BVNodeId a64 = bv_ext(ctx, is_s, 64, gpu_coerce_width(ctx, get_v(inst->operand(0)), 32));
                    BVNodeId b64 = bv_ext(ctx, is_s, 64, gpu_coerce_width(ctx, get_v(inst->operand(1)), 32));
                    res = gpu_mul_or_shl(ctx, 64, a64, b64);
                } else if (sym == "ptx_mad_lo_u32") {
                    BVNodeId a = gpu_coerce_width(ctx, get_v(inst->operand(0)), 32);
                    BVNodeId b = gpu_coerce_width(ctx, get_v(inst->operand(1)), 32);
                    BVNodeId c = gpu_coerce_width(ctx, get_v(inst->operand(2)), 32);
                    res = bv_bin(ctx, BVOp::Add, 32, gpu_mul_or_shl(ctx, 32, a, b), c);
                } else if (sym.starts_with("ptx_shared_alloc_")) {
                    uint32_t arr_idx = static_cast<uint32_t>(trace.shared_allocs.size());
                    uint32_t ebytes = (sym.ends_with("f64") || sym.ends_with("i64")) ? 8 : 4;
                    BVNodeId cnt_node = get_v(inst->operand(0));
                    uint32_t cnt = (bv_node(ctx, cnt_node).op == BVOp::Const)
                        ? static_cast<uint32_t>(bv_node(ctx, cnt_node).imm) : 256u;
                    uint64_t sym_base = kGpuSharedBase0 + static_cast<uint64_t>(arr_idx) * kGpuSharedStride;
                    trace.shared_allocs.push_back({arr_idx, cnt, ebytes, sym_base});
                    res = bv_const(ctx, 64, sym_base);
                } else if (sym.starts_with("ptx_shared_load_")) {
                    bool indexed = sym.ends_with("_indexed");
                    uint32_t ebytes = (sym.find("64") != std::string_view::npos) ? 8 : 4;
                    uint32_t shift = shift_for_elem_bytes(ebytes);
                    BVNodeId ptr = gpu_coerce_width(ctx, get_v(inst->operand(0)), 64);
                    uint32_t arr_idx = 0;
                    BVNodeId base_byte_off = 0, base_elem_idx = 0;
                    resolve_shared_address(ctx, ptr, trace.shared_allocs, arr_idx, base_byte_off, base_elem_idx);
                    BVNodeId elem_idx = base_elem_idx;
                    if (inst->operand_count() > 1 && inst->operand(1)) {
                        BVNodeId ex = get_v(inst->operand(1));
                        BVNodeId ex_idx = indexed
                            ? gpu_coerce_width(ctx, ex, 32, false)
                            : extract_shared_elem_index_32(ctx, ex, shift);
                        elem_idx = (bv_node(ctx, base_elem_idx).op == BVOp::Const && bv_node(ctx, base_elem_idx).imm == 0)
                            ? ex_idx
                            : bv_bin(ctx, BVOp::Add, 32, base_elem_idx, ex_idx);
                    }
                    record_shared_access(ctx, trace, cur.cond, false, arr_idx, elem_idx, ebytes);
                    BVNodeId caddr = make_shared_canonical_addr(ctx, arr_idx, elem_idx, ebytes);
                    res = env.read_mem(ctx, true, trace.shared_stores, caddr, ebytes);
                } else if (sym.starts_with("ptx_shared_store_")) {
                    bool indexed = sym.ends_with("_indexed");
                    uint32_t ebytes = (sym.find("64") != std::string_view::npos) ? 8 : 4;
                    uint32_t shift = shift_for_elem_bytes(ebytes);
                    BVNodeId ptr = gpu_coerce_width(ctx, get_v(inst->operand(0)), 64);
                    const Value* val_op = indexed ? inst->operand(2) : inst->operand(1);
                    const Value* ex_op = indexed ? inst->operand(1) : (inst->operand_count() > 2 ? inst->operand(2) : nullptr);
                    uint32_t arr_idx = 0;
                    BVNodeId base_byte_off = 0, base_elem_idx = 0;
                    resolve_shared_address(ctx, ptr, trace.shared_allocs, arr_idx, base_byte_off, base_elem_idx);
                    BVNodeId elem_idx = base_elem_idx;
                    if (ex_op) {
                        BVNodeId ex = get_v(ex_op);
                        BVNodeId ex_idx = indexed
                            ? gpu_coerce_width(ctx, ex, 32, false)
                            : extract_shared_elem_index_32(ctx, ex, shift);
                        elem_idx = (bv_node(ctx, base_elem_idx).op == BVOp::Const && bv_node(ctx, base_elem_idx).imm == 0)
                            ? ex_idx
                            : bv_bin(ctx, BVOp::Add, 32, base_elem_idx, ex_idx);
                    }
                    record_shared_access(ctx, trace, cur.cond, true, arr_idx, elem_idx, ebytes);
                    BVNodeId caddr = make_shared_canonical_addr(ctx, arr_idx, elem_idx, ebytes);
                    BVNodeId sval = gpu_coerce_width(ctx, get_v(val_op), ebytes * 8);
                    trace.shared_stores.push_back({cur.cond, caddr, ebytes, sval});
                } else if (sym == "bar.sync" || sym == "ptx_sync" || sym == "ptx_bar_sync" || sym == "ptx_bar_sync_count") {
                    BVNodeId bid = (inst->operand_count() > 0 && inst->operand(0))
                        ? gpu_coerce_width(ctx, get_v(inst->operand(0)), 32)
                        : bv_const(ctx, 32, 0);
                    trace.barriers.push_back({cur.cond, bid, false, barrier_seq++});
                } else if (sym.starts_with("ptx_shfl_") || sym == "shfl_down_sync_f32") {
                    bool has_mask = (sym == "ptx_shfl_down_sync_f32" || sym == "shfl_down_sync_f32");
                    BVNodeId v32 = get_v(inst->operand(has_mask ? 1 : 0));
                    BVNodeId d32 = get_v(inst->operand(has_mask ? 2 : 1));
                    uint32_t mode = (sym.find("_up_") != std::string_view::npos) ? 1
                                  : (sym.find("_bfly_") != std::string_view::npos || sym.find("_xor_") != std::string_view::npos) ? 2
                                  : (sym.find("_idx_") != std::string_view::npos) ? 3 : 0;
                    trace.barriers.push_back({cur.cond, bv_const(ctx, 32, mode), true, barrier_seq++});
                    res = env.eval_shuffle_mode(ctx, mode, v32, d32);
                } else if (sym.starts_with("ptx_load_u") || sym.starts_with("ptx_load_s")) {
                    bool is_s = sym.starts_with("ptx_load_s");
                    uint32_t nb = sym.ends_with("16") ? 2 : 1;
                    BVNodeId addr = gpu_coerce_width(ctx, get_v(inst->operand(0)), 64);
                    if (inst->operand_count() > 1 && inst->operand(1)) {
                        addr = gpu_add_addr64(ctx, addr, gpu_coerce_width(ctx, get_v(inst->operand(1)), 64, true));
                    }
                    record_global_access(ctx, trace, env, cur.cond, false, addr, nb);
                    BVNodeId raw = env.read_mem(ctx, false, trace.global_stores, addr, nb);
                    res = bv_ext(ctx, is_s, 32, raw);
                } else if (sym == "ptx_store_u8" || sym == "ptx_store_u16") {
                    uint32_t nb = sym.ends_with("16") ? 2 : 1;
                    BVNodeId addr = gpu_coerce_width(ctx, get_v(inst->operand(0)), 64);
                    if (inst->operand_count() > 2 && inst->operand(2)) {
                        addr = gpu_add_addr64(ctx, addr, gpu_coerce_width(ctx, get_v(inst->operand(2)), 64, true));
                    }
                    BVNodeId val = gpu_coerce_width(ctx, get_v(inst->operand(1)), nb * 8);
                    record_global_access(ctx, trace, env, cur.cond, true, addr, nb);
                    trace.global_stores.push_back({cur.cond, addr, nb, val});
                } else if (sym.starts_with("ptx_atom_")) {
                    bool is_sh = sym.find("_shared_") != std::string_view::npos;
                    BVNodeId ptr = gpu_coerce_width(ctx, get_v(inst->operand(0)), 64);
                    BVNodeId arg = gpu_coerce_width(ctx, get_v(inst->operand(1)), rw);
                    uint32_t nb = rw / 8;
                    BVNodeId caddr = ptr;
                    if (is_sh) {
                        uint32_t arr_idx = 0;
                        BVNodeId boff = 0, eidx = 0;
                        resolve_shared_address(ctx, ptr, trace.shared_allocs, arr_idx, boff, eidx);
                        record_shared_access(ctx, trace, cur.cond, false, arr_idx, eidx, nb);
                        record_shared_access(ctx, trace, cur.cond, true, arr_idx, eidx, nb);
                        caddr = make_shared_canonical_addr(ctx, arr_idx, eidx, nb);
                    } else {
                        record_global_access(ctx, trace, env, cur.cond, false, caddr, nb);
                        record_global_access(ctx, trace, env, cur.cond, true, caddr, nb);
                    }
                    auto& st_vec = is_sh ? trace.shared_stores : trace.global_stores;
                    BVNodeId old_v = env.read_mem(ctx, is_sh, st_vec, caddr, nb);
                    BVNodeId new_v = arg;
                    if (sym.find("_add_") != std::string_view::npos) {
                        new_v = sym.ends_with("f32")
                            ? env.eval_uf(ctx, "fp_add", {old_v, arg}, rw, true)
                            : bv_bin(ctx, BVOp::Add, rw, old_v, arg);
                    } else if (sym.find("_min_") != std::string_view::npos) {
                        new_v = bv_ite(ctx, rw, bv_bin(ctx, BVOp::Slt, 1, old_v, arg), old_v, arg);
                    } else if (sym.find("_max_") != std::string_view::npos) {
                        new_v = bv_ite(ctx, rw, bv_bin(ctx, BVOp::Sgt, 1, old_v, arg), old_v, arg);
                    }
                    st_vec.push_back({cur.cond, caddr, nb, new_v});
                    res = old_v;
                } else if (sym == "ptx_rcp" || sym == "ptx_rcp_approx") {
                    res = env.eval_uf(ctx, "fp_div", {bv_const(ctx, 32, 0x3F800000u), gpu_coerce_width(ctx, get_v(inst->operand(0)), 32)}, 32, true);
                } else if (sym == "ptx_div_approx") {
                    res = env.eval_uf(ctx, "fp_div", {gpu_coerce_width(ctx, get_v(inst->operand(0)), 32), gpu_coerce_width(ctx, get_v(inst->operand(1)), 32)}, 32, true);
                } else if (sym == "rsqrtf" || sym == "rsqrt" || sym == "ptx_rsqrt") {
                    res = env.eval_uf(ctx, "gpu_rsqrt", {gpu_coerce_width(ctx, get_v(inst->operand(0)), rw)}, rw, true);
                } else if (sym == "sqrtf" || sym == "sqrt" || sym == "ptx_sqrt" || sym == "ptx_sqrt_rn") {
                    res = env.eval_uf(ctx, "gpu_sqrt", {gpu_coerce_width(ctx, get_v(inst->operand(0)), rw)}, rw, true);
                } else if (sym == "sinf" || sym == "sin" || sym == "ptx_sin") {
                    res = env.eval_uf(ctx, "gpu_sin", {gpu_coerce_width(ctx, get_v(inst->operand(0)), rw)}, rw, true);
                } else if (sym == "cosf" || sym == "cos" || sym == "ptx_cos") {
                    res = env.eval_uf(ctx, "gpu_cos", {gpu_coerce_width(ctx, get_v(inst->operand(0)), rw)}, rw, true);
                } else if (sym == "ex2f" || sym == "ex2" || sym == "ptx_ex2") {
                    res = env.eval_uf(ctx, "gpu_ex2", {gpu_coerce_width(ctx, get_v(inst->operand(0)), rw)}, rw, true);
                } else if (sym == "lg2f" || sym == "ptx_lg2") {
                    res = env.eval_uf(ctx, "gpu_lg2", {gpu_coerce_width(ctx, get_v(inst->operand(0)), rw)}, rw, true);
                } else if (sym == "expf" || sym == "exp" || sym == "ptx_exp") {
                    res = env.eval_uf(ctx, "gpu_exp", {gpu_coerce_width(ctx, get_v(inst->operand(0)), rw)}, rw, true);
                } else if (sym == "logf" || sym == "log" || sym == "ptx_log") {
                    res = env.eval_uf(ctx, "gpu_log", {gpu_coerce_width(ctx, get_v(inst->operand(0)), rw)}, rw, true);
                } else if (sym == "fabsf" || sym == "fabs" || sym == "ptx_fabs") {
                    BVNodeId a = gpu_coerce_width(ctx, get_v(inst->operand(0)), rw);
                    res = bv_bin(ctx, BVOp::And, rw, a, bv_const(ctx, rw, ~(1ULL << (rw - 1))));
                } else if (sym == "fminf" || sym == "fmin" || sym == "ptx_fmin") {
                    res = env.eval_uf(ctx, "gpu_fmin", {gpu_coerce_width(ctx, get_v(inst->operand(0)), rw), gpu_coerce_width(ctx, get_v(inst->operand(1)), rw)}, rw, true);
                } else if (sym == "fmaxf" || sym == "fmax" || sym == "ptx_fmax") {
                    res = env.eval_uf(ctx, "gpu_fmax", {gpu_coerce_width(ctx, get_v(inst->operand(0)), rw), gpu_coerce_width(ctx, get_v(inst->operand(1)), rw)}, rw, true);
                } else if (sym == "i32_to_f32" || sym == "ptx_i32_to_f32" || sym == "ptx_i64_to_f32") {
                    res = env.eval_uf(ctx, "cvt_sitofp", {get_v(inst->operand(0))}, 32, true);
                } else if (sym == "ptx_u32_to_f32" || sym == "ptx_u64_to_f32") {
                    res = env.eval_uf(ctx, "cvt_uitofp", {get_v(inst->operand(0))}, 32, true);
                } else if (sym == "ptx_f32_to_i32") {
                    res = env.eval_uf(ctx, "cvt_fptosi", {get_v(inst->operand(0))}, 32, false);
                } else if (sym == "ptx_f32_to_u32") {
                    res = env.eval_uf(ctx, "cvt_fptoui", {get_v(inst->operand(0))}, 32, false);
                } else if (sym == "ptx_f32_to_f64") {
                    res = env.eval_uf(ctx, "cvt_fpext", {get_v(inst->operand(0))}, 64, true);
                } else if (sym == "ptx_f64_to_f32") {
                    res = env.eval_uf(ctx, "cvt_fptrunc", {get_v(inst->operand(0))}, 32, true);
                } else if (sym == "ptx_f16_to_f32") {
                    res = env.eval_uf(ctx, "gpu_f16_to_f32", {gpu_coerce_width(ctx, get_v(inst->operand(0)), 32)}, 32, true);
                } else if (sym == "ptx_f32_to_f16") {
                    res = env.eval_uf(ctx, "gpu_f32_to_f16", {gpu_coerce_width(ctx, get_v(inst->operand(0)), 32)}, 32, false);
                } else if (sym == "clampf" || sym == "ptx_clamp") {
                    res = env.eval_uf(ctx, "gpu_clamp", {
                        gpu_coerce_width(ctx, get_v(inst->operand(0)), rw),
                        gpu_coerce_width(ctx, get_v(inst->operand(1)), rw),
                        gpu_coerce_width(ctx, get_v(inst->operand(2)), rw)}, rw, true);
                } else if (sym == "mixf" || sym == "ptx_mix") {
                    res = env.eval_uf(ctx, "gpu_mix", {
                        gpu_coerce_width(ctx, get_v(inst->operand(0)), rw),
                        gpu_coerce_width(ctx, get_v(inst->operand(1)), rw),
                        gpu_coerce_width(ctx, get_v(inst->operand(2)), rw)}, rw, true);
                } else if (sym == "smoothstepf" || sym == "ptx_smoothstep") {
                    res = env.eval_uf(ctx, "gpu_smoothstep", {
                        gpu_coerce_width(ctx, get_v(inst->operand(0)), rw),
                        gpu_coerce_width(ctx, get_v(inst->operand(1)), rw),
                        gpu_coerce_width(ctx, get_v(inst->operand(2)), rw)}, rw, true);
                } else if (sym == "stepf" || sym == "ptx_step") {
                    res = env.eval_uf(ctx, "gpu_step", {
                        gpu_coerce_width(ctx, get_v(inst->operand(0)), rw),
                        gpu_coerce_width(ctx, get_v(inst->operand(1)), rw)}, rw, true);
                } else if (sym == "fractf" || sym == "ptx_fract") {
                    res = env.eval_uf(ctx, "gpu_fract", {gpu_coerce_width(ctx, get_v(inst->operand(0)), rw)}, rw, true);
                } else if (sym == "signf" || sym == "ptx_sign") {
                    res = env.eval_uf(ctx, "gpu_sign", {gpu_coerce_width(ctx, get_v(inst->operand(0)), rw)}, rw, true);
                } else if (sym == "tanhf" || sym == "ptx_tanh") {
                    res = env.eval_uf(ctx, "gpu_tanh", {gpu_coerce_width(ctx, get_v(inst->operand(0)), rw)}, rw, true);
                } else if (sym == "spirv_vertex_index") {
                    res = env.eval_uf(ctx, "spirv_vertex_index", {}, 32, false);
                } else if (sym == "spirv_instance_index") {
                    res = env.eval_uf(ctx, "spirv_instance_index", {}, 32, false);
                } else if (sym == "spirv_frag_coord_x" || sym == "spirv_frag_coord_y" ||
                           sym == "spirv_frag_coord_z" || sym == "spirv_frag_coord_w") {
                    uint32_t c = sym.ends_with("_x") ? 0u : sym.ends_with("_y") ? 1u : sym.ends_with("_z") ? 2u : 3u;
                    res = env.eval_uf(ctx, "spirv_frag_coord", {bv_const(ctx, 32, c)}, 32, true);
                } else if (sym == "spirv_stage_input_f32") {
                    res = env.eval_uf(ctx, "spirv_stage_in", {
                        gpu_coerce_width(ctx, get_v(inst->operand(0)), 32),
                        gpu_coerce_width(ctx, get_v(inst->operand(1)), 32)}, 32, true);
                } else if (sym == "spirv_set_vertex_position" && inst->operand_count() == 4) {
                    for (uint32_t c = 0; c < 4; ++c) {
                        BVNodeId cv = gpu_coerce_width(ctx, get_v(inst->operand(c)), 32);
                        trace.global_stores.push_back({cur.cond, bv_const(ctx, 64, 0x6000000000000000ULL + c * 4), 4, cv});
                    }
                } else if (sym == "spirv_stage_output_f32" && inst->operand_count() >= 3) {
                    BVNodeId loc_n = get_v(inst->operand(0));
                    BVNodeId comp_n = get_v(inst->operand(1));
                    uint64_t loc = (bv_node(ctx, loc_n).op == BVOp::Const) ? bv_node(ctx, loc_n).imm : 0;
                    uint64_t comp = (bv_node(ctx, comp_n).op == BVOp::Const) ? bv_node(ctx, comp_n).imm : 0;
                    BVNodeId cv = gpu_coerce_width(ctx, get_v(inst->operand(2)), 32);
                    trace.global_stores.push_back({cur.cond, bv_const(ctx, 64, 0x6000000000001000ULL + loc * 16 + comp * 4), 4, cv});
                } else if (sym == "spirv_stage_output_vec4" && inst->operand_count() == 5) {
                    BVNodeId loc_n = get_v(inst->operand(0));
                    uint64_t loc = (bv_node(ctx, loc_n).op == BVOp::Const) ? bv_node(ctx, loc_n).imm : 0;
                    for (uint32_t c = 0; c < 4; ++c) {
                        BVNodeId cv = gpu_coerce_width(ctx, get_v(inst->operand(1 + c)), 32);
                        trace.global_stores.push_back({cur.cond, bv_const(ctx, 64, 0x6000000000001000ULL + loc * 16 + c * 4), 4, cv});
                    }
                } else {
                    std::vector<BVNodeId> args;
                    for (size_t i = 0; i < inst->operand_count(); ++i) args.push_back(get_v(inst->operand(i)));
                    if (!inst->type().is_void()) res = env.eval_uf(ctx, sym, args, rw, inst->type().is_float());
                }
                break;
            }
            case Opcode::br:
                route_edge(inst->branch_target(), cur.cond, cur.vals);
                break;
            case Opcode::br_if: {
                BVNodeId c1 = as_cond1(inst->operand(0));
                BVNodeId t_cond = bv_bin(ctx, BVOp::And, 1, cur.cond, c1);
                BVNodeId f_cond = bv_bin(ctx, BVOp::And, 1, cur.cond, bv_un(ctx, BVOp::Not, 1, c1));
                route_edge(inst->true_target(), t_cond, cur.vals);
                route_edge(inst->false_target(), f_cond, cur.vals);
                break;
            }
            case Opcode::ret:
                trace.returned = bv_bin(ctx, BVOp::Or, 1, trace.returned, cur.cond);
                break;
            case Opcode::unreachable:
                trace.trapped = bv_bin(ctx, BVOp::Or, 1, trace.trapped, cur.cond);
                break;
            default:
                break;
            }
            if (inst->result() && res != 0) {
                cur.vals[inst->result()->id()] = res;
            }
        }
    }
    return trace;
}

} // namespace brass::verify
