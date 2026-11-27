#include <brass/verify/gpu_sandbox_instrument.hpp>

#include <brass/core/diagnostics.hpp>
#include <brass/mir/block.hpp>
#include <brass/mir/builder.hpp>
#include <brass/mir/cross_module.hpp>
#include <brass/mir/dominators.hpp>
#include <brass/mir/gpu_portability.hpp>
#include <brass/mir/loop_analysis.hpp>
#include <brass/mir/module.hpp>
#include <brass/mir/opcodes.hpp>
#include <brass/mir/uses.hpp>
#include <brass/mir/verifier.hpp>
#include <brass/target/gpu_intrinsics.hpp>

#include <algorithm>
#include <sstream>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace brass::verify {

namespace {

std::string join_messages(const std::vector<std::string>& msgs) {
    std::ostringstream oss;
    for (size_t i = 0; i < msgs.size(); ++i) {
        if (i > 0) oss << "; ";
        oss << msgs[i];
    }
    return oss.str();
}

std::string with_loc(const DebugContext* dbg, DebugLoc loc, std::string msg) {
    if (!dbg) return msg;
    const std::string f = format_debug_loc(*dbg, loc);
    if (f.empty()) return msg;
    return f + ": " + msg;
}

bool get_const_i64(const Value* v, int64_t& out) noexcept {
    if (!v || !v->is_instruction()) return false;
    const Instruction* inst = v->defining_instruction();
    if (!inst) return false;
    if (inst->opcode() == Opcode::iconst_i32) {
        out = inst->imm_i32();
        return true;
    }
    if (inst->opcode() == Opcode::iconst_i64) {
        out = inst->imm_i64();
        return true;
    }
    return false;
}

enum class PtrRootKind : uint8_t { Param, Alloca, Unknown };

struct PtrChain {
    PtrRootKind kind = PtrRootKind::Unknown;
    uint32_t param_index = 0;
    std::vector<Value*> addend_offsets; // byte offsets already in i64 or convertible to i64
};

PtrChain trace_ptr_chain(const Function& fn, Value* ptr, int depth = 0) {
    PtrChain chain;
    if (!ptr || depth > 32) return chain;

    const BasicBlock* entry = fn.entry_block();
    if (ptr->is_block_param()) {
        if (ptr->defining_block() == entry) {
            chain.kind = PtrRootKind::Param;
            chain.param_index = ptr->param_index();
        }
        return chain;
    }
    if (!ptr->is_instruction()) return chain;

    const Instruction* inst = ptr->defining_instruction();
    if (!inst) return chain;
    if (inst->opcode() == Opcode::alloca_) {
        chain.kind = PtrRootKind::Alloca;
        return chain;
    }
    if (inst->opcode() == Opcode::add && inst->type() == Type::ptr() && inst->operand_count() == 2) {
        Value* lhs = inst->operand(0);
        Value* rhs = inst->operand(1);
        if (lhs && lhs->type() == Type::ptr()) {
            PtrChain base = trace_ptr_chain(fn, lhs, depth + 1);
            if (base.kind != PtrRootKind::Unknown) {
                if (rhs) base.addend_offsets.push_back(rhs);
                return base;
            }
        }
        if (rhs && rhs->type() == Type::ptr()) {
            PtrChain base = trace_ptr_chain(fn, rhs, depth + 1);
            if (base.kind != PtrRootKind::Unknown) {
                if (lhs) base.addend_offsets.push_back(lhs);
                return base;
            }
        }
    }
    return chain;
}

Value* coerce_to_i64(Builder& b, Value* v, bool unsigned_ext = false) {
    if (!v) return b.build_iconst_i64(0);
    if (v->type() == Type::i64()) return v;
    if (v->type() == Type::i32()) {
        return unsigned_ext ? b.build_zext_i64(v) : b.build_sext_i64(v);
    }
    if (v->type() == Type::i8() || v->type() == Type::i16()) {
        return b.build_zext_i64(v);
    }
    return b.build_iconst_i64(0);
}

const GpuBufferSpec* find_buffer_spec(const GpuProverOptions& opts, uint32_t param_index) noexcept {
    for (const auto& b : opts.buffers) {
        if (b.param_index == param_index) return &b;
    }
    return nullptr;
}

Value* build_buffer_byte_capacity(Builder& b, const Function& fn, const GpuBufferSpec& spec) {
    if (spec.static_byte_size > 0) {
        return b.build_iconst_i64(static_cast<int64_t>(spec.static_byte_size));
    }
    const BasicBlock* entry = fn.entry_block();
    if (!entry) return nullptr;

    auto param_i64 = [&](int32_t idx) -> Value* {
        if (idx < 0 || static_cast<size_t>(idx) >= entry->param_count()) return nullptr;
        return coerce_to_i64(b, const_cast<Value*>(entry->param(static_cast<size_t>(idx))), true);
    };

    Value* elems = param_i64(spec.length_param_index);
    if (!elems) return nullptr;
    if (spec.length_param_index2 >= 0) {
        Value* d2 = param_i64(spec.length_param_index2);
        if (!d2) return nullptr;
        elems = b.build_mul(elems, d2);
    }
    if (spec.length_param_index3 >= 0) {
        Value* d3 = param_i64(spec.length_param_index3);
        if (!d3) return nullptr;
        elems = b.build_mul(elems, d3);
    }
    const uint64_t scale_bytes = static_cast<uint64_t>(std::max<uint32_t>(1u, spec.elem_scale)) *
                                 static_cast<uint64_t>(std::max<uint32_t>(1u, spec.elem_bytes));
    if (scale_bytes == 1) return elems;
    return b.build_mul(elems, b.build_iconst_i64(static_cast<int64_t>(scale_bytes)));
}

Value* build_zero_value(Builder& b, Type t) {
    switch (t.kind()) {
        case TypeKind::I8:
        case TypeKind::I16:
        case TypeKind::I32:
            return b.build_iconst_i32(0);
        case TypeKind::I64:
            return b.build_iconst_i64(0);
        case TypeKind::F32:
            return b.build_fconst_f32(0.0f);
        case TypeKind::F64:
            return b.build_fconst_f64(0.0);
        case TypeKind::F32x4:
        case TypeKind::F64x2:
        case TypeKind::I32x4:
        case TypeKind::I64x2:
        case TypeKind::F32x8:
        case TypeKind::F64x4:
        case TypeKind::I32x8:
        case TypeKind::I64x4:
            return b.build_vzero(t);
        default:
            return nullptr;
    }
}

void replace_all_value_uses(Function& fn, const Value* old_val, Value* new_val) {
    if (!old_val || !new_val || old_val == new_val) return;
    for (BasicBlock* bb : fn.blocks()) {
        if (!bb) continue;
        for (Instruction* inst : *bb) {
            if (inst) replace_uses_in(*inst, old_val, new_val);
        }
    }
}

bool is_global_mem_intrinsic(std::string_view raw_sym, bool& is_load, uint32_t& access_bytes) noexcept {
    const std::string_view sym = target::canonical_gpu_intrinsic(raw_sym);
    if (sym == "ptx_load_u8" || sym == "ptx_load_s8") {
        is_load = true;
        access_bytes = 1;
        return true;
    }
    if (sym == "ptx_load_u16" || sym == "ptx_load_s16") {
        is_load = true;
        access_bytes = 2;
        return true;
    }
    if (sym == "ptx_store_u8") {
        is_load = false;
        access_bytes = 1;
        return true;
    }
    if (sym == "ptx_store_u16") {
        is_load = false;
        access_bytes = 2;
        return true;
    }
    if (sym.rfind("ptx_atom_", 0) == 0 && sym.rfind("ptx_atom_shared_", 0) != 0) {
        is_load = true;
        const bool is_i64 = sym.size() >= 4 &&
            (sym.compare(sym.size() - 4, 4, "_i64") == 0 || sym.compare(sym.size() - 4, 4, "_u64") == 0);
        access_bytes = is_i64 ? 8 : 4;
        return true;
    }
    return false;
}

bool is_barrier_call(const Instruction* inst) noexcept {
    if (!inst || inst->opcode() != Opcode::call) return false;
    const std::string_view sym = target::canonical_gpu_intrinsic(inst->symbol());
    return sym == "ptx_sync" || sym == "ptx_bar_sync" || sym == "ptx_subgroup_barrier" ||
           sym == "ptx_bar_sync_count";
}

bool try_detect_static_affine_bound(
    const LoopInfo& loop,
    uint32_t max_iters) noexcept {
    BasicBlock* header = loop.header();
    if (!header || !header->terminator()) return false;
    const Instruction* term = header->terminator();
    if (term->opcode() != Opcode::br_if || term->operand_count() < 1) return false;

    BasicBlock* t_bb = term->true_target().block;
    BasicBlock* f_bb = term->false_target().block;
    const bool t_in = loop.contains(t_bb);
    const bool f_in = loop.contains(f_bb);
    if (t_in == f_in) return false;

    const Instruction* cond = term->operand(0) && term->operand(0)->is_instruction()
        ? term->operand(0)->defining_instruction() : nullptr;
    if (!cond || cond->operand_count() != 2) return false;

    const Value* iv = cond->operand(0);
    int64_t bound = 0;
    if (!iv || !iv->is_block_param() || iv->defining_block() != header || !get_const_i64(cond->operand(1), bound)) {
        return false;
    }
    const size_t p_idx = iv->param_index();

    bool has_init = false;
    int64_t init_val = 0;
    bool has_step = false;
    int64_t step_val = 0;

    for (BasicBlock* pred : header->predecessors()) {
        if (!pred || !pred->terminator()) continue;
        const bool pred_in_loop = loop.contains(pred);
        for_each_edge(*pred->terminator(), [&](const BranchTarget& bt) {
            if (bt.block != header || p_idx >= bt.args.size()) return;
            const Value* arg = bt.args[p_idx];
            if (!pred_in_loop) {
                int64_t cv = 0;
                if (!get_const_i64(arg, cv) || (has_init && cv != init_val)) {
                    init_val = INT64_MIN;
                    has_init = false;
                    return;
                }
                init_val = cv;
                has_init = true;
            } else {
                const Instruction* step_inst = arg && arg->is_instruction() ? arg->defining_instruction() : nullptr;
                int64_t delta = 0;
                if (step_inst && step_inst->operand_count() == 2 && step_inst->operand(0) == iv &&
                    get_const_i64(step_inst->operand(1), delta) &&
                    (step_inst->opcode() == Opcode::add || step_inst->opcode() == Opcode::sub)) {
                    if (step_inst->opcode() == Opcode::sub) delta = -delta;
                    if (!has_step) {
                        step_val = delta;
                        has_step = true;
                    } else if (step_val != delta) {
                        step_val = 0;
                    }
                } else {
                    has_step = false;
                }
            }
        });
        if (!pred_in_loop && !has_init) return false;
    }

    if (!has_init || !has_step || step_val == 0) return false;

    const Opcode cmp = cond->opcode();
    // Stay condition is `cond` when t_in is true; otherwise `!cond`.
    if (t_in && step_val > 0 && (cmp == Opcode::slt || cmp == Opcode::ult || cmp == Opcode::sle || cmp == Opcode::ule)) {
        if (init_val > bound) return true; // 0 iterations
        const uint64_t span = static_cast<uint64_t>(bound - init_val) +
                              ((cmp == Opcode::sle || cmp == Opcode::ule) ? 1ULL : 0ULL);
        const uint64_t trips = (span + static_cast<uint64_t>(step_val) - 1ULL) / static_cast<uint64_t>(step_val);
        return trips <= max_iters;
    }
    if (!t_in && step_val > 0 && (cmp == Opcode::sge || cmp == Opcode::uge || cmp == Opcode::sgt || cmp == Opcode::ugt)) {
        if (init_val > bound) return true;
        const uint64_t span = static_cast<uint64_t>(bound - init_val) +
                              ((cmp == Opcode::sgt || cmp == Opcode::ugt) ? 1ULL : 0ULL);
        const uint64_t trips = (span + static_cast<uint64_t>(step_val) - 1ULL) / static_cast<uint64_t>(step_val);
        return trips <= max_iters;
    }
    return false;
}

void append_arg_to_edges_targeting(Instruction* term, const BasicBlock* target, Value* arg) {
    if (!term || !arg) return;
    for_each_edge(*term, [&](BranchTarget& bt) {
        if (bt.block == target) bt.args.push_back(arg);
    });
}

} // namespace

GpuClampResult instrument_unproven_gpu_accesses(
    Function& fn,
    const GpuProverOptions& prover_opts,
    const GpuSandboxResult* smt_result,
    const GpuClampOptions& opts) {
    GpuClampResult res;
    if (!opts.force_clamp_all && smt_result &&
        smt_result->proof_tier == SandboxProofTier::FullSmt &&
        smt_result->global_bounds_proven &&
        smt_result->memory_safe) {
        res.diagnostic = "SMT already proved all global bounds";
        return res;
    }
    if (!fn.parent() || !fn.entry_block()) {
        res.ok = false;
        res.violations.push_back("invalid function/module for GPU access clamping");
        res.diagnostic = res.violations.back();
        return res;
    }

    struct AccessItem {
        Instruction* inst = nullptr;
        Value* base_ptr = nullptr;
        Value* index = nullptr;
        uint8_t scale = 1;
        int32_t offset = 0;
        uint32_t access_bytes = 4;
        bool is_load = true;
    };

    std::vector<Instruction*> access_insts;
    for (BasicBlock* bb : fn.blocks()) {
        if (!bb) continue;
        for (Instruction* inst : *bb) {
            if (!inst) continue;
            const Opcode op = inst->opcode();
            if (op == Opcode::load || op == Opcode::vload || op == Opcode::load_indexed ||
                op == Opcode::store || op == Opcode::vstore || op == Opcode::store_indexed) {
                access_insts.push_back(inst);
            } else if (op == Opcode::call && !inst->symbol().empty()) {
                bool is_ld = true;
                uint32_t sz = 0;
                if (is_global_mem_intrinsic(inst->symbol(), is_ld, sz)) {
                    access_insts.push_back(inst);
                }
            }
        }
    }

    Module& mod = *fn.parent();
    Builder b(mod);
    b.set_function(&fn);

    for (Instruction* inst : access_insts) {
        BasicBlock* cur_bb = inst ? inst->parent() : nullptr;
        if (!cur_bb || cur_bb->parent() != &fn) continue;

        AccessItem item;
        item.inst = inst;
        const Opcode op = inst->opcode();
        if (op == Opcode::load || op == Opcode::vload) {
            item.base_ptr = inst->operand(0);
            item.offset = inst->offset();
            item.access_bytes = static_cast<uint32_t>(std::max<size_t>(1, inst->type().size_in_bytes()));
            item.is_load = true;
        } else if (op == Opcode::load_indexed) {
            item.base_ptr = inst->operand(0);
            item.index = inst->operand(1);
            item.scale = inst->scale();
            item.offset = inst->offset();
            item.access_bytes = static_cast<uint32_t>(std::max<size_t>(1, inst->type().size_in_bytes()));
            item.is_load = true;
        } else if (op == Opcode::store || op == Opcode::vstore) {
            Value* v = inst->operand(1);
            item.base_ptr = inst->operand(0);
            item.offset = inst->offset();
            item.access_bytes = v ? static_cast<uint32_t>(std::max<size_t>(1, v->type().size_in_bytes())) : 4;
            item.is_load = false;
        } else if (op == Opcode::store_indexed) {
            Value* v = inst->operand(2);
            item.base_ptr = inst->operand(0);
            item.index = inst->operand(1);
            item.scale = inst->scale();
            item.offset = inst->offset();
            item.access_bytes = v ? static_cast<uint32_t>(std::max<size_t>(1, v->type().size_in_bytes())) : 4;
            item.is_load = false;
        } else if (op == Opcode::call && !inst->symbol().empty()) {
            item.base_ptr = inst->operand(0);
            is_global_mem_intrinsic(inst->symbol(), item.is_load, item.access_bytes);
        }

        const PtrChain chain = trace_ptr_chain(fn, item.base_ptr);
        if (chain.kind == PtrRootKind::Alloca) continue;
        const GpuBufferSpec* bspec = (chain.kind == PtrRootKind::Param)
            ? find_buffer_spec(prover_opts, chain.param_index)
            : nullptr;

        if (chain.kind == PtrRootKind::Unknown || !bspec) {
            ++res.untracked_accesses;
            if (opts.reject_untracked_pointers) {
                std::string msg = (chain.kind == PtrRootKind::Unknown)
                    ? "global memory access via untracked/derived pointer cannot be sandboxed"
                    : ("global memory access to param " + std::to_string(chain.param_index) + " missing GpuBufferSpec");
                res.violations.push_back(with_loc(opts.debug_ctx, inst->loc(), std::move(msg)));
                res.ok = false;
            }
            continue;
        }

        b.position_before(inst);
        b.set_current_loc(inst->loc());

        Value* byte_off = b.build_iconst_i64(0);
        for (Value* off_v : chain.addend_offsets) {
            byte_off = b.build_add(byte_off, coerce_to_i64(b, off_v, false));
        }
        if (item.index) {
            Value* idx64 = coerce_to_i64(b, item.index, false);
            if (item.scale > 1) {
                idx64 = b.build_mul(idx64, b.build_iconst_i64(static_cast<int64_t>(item.scale)));
            }
            byte_off = b.build_add(byte_off, idx64);
        }
        if (item.offset != 0) {
            byte_off = b.build_add(byte_off, b.build_iconst_i64(static_cast<int64_t>(item.offset)));
        }

        Value* cap_bytes = build_buffer_byte_capacity(b, fn, *bspec);
        if (!cap_bytes) {
            res.ok = false;
            res.violations.push_back(with_loc(opts.debug_ctx, inst->loc(), "invalid GpuBufferSpec dimension parameter"));
            continue;
        }

        Value* acc_bytes = b.build_iconst_i64(static_cast<int64_t>(item.access_bytes));
        Value* can_fit = b.build_uge(cap_bytes, acc_bytes);
        Value* max_off = b.build_sub(cap_bytes, acc_bytes);
        Value* in_bounds = b.build_and(can_fit, b.build_ule(byte_off, max_off));

        // Split `cur_bb` around `inst`: tail instructions move to `cont_bb`.
        BasicBlock* cont_bb = b.append_block(std::string(cur_bb->name()) + ".sb_cont");
        cont_bb->set_cold(cur_bb->is_cold());

        std::vector<Instruction*> tail_insts;
        for (Instruction* t = inst->next(); t != nullptr; t = t->next()) {
            tail_insts.push_back(t);
        }
        for (Instruction* t : tail_insts) {
            cur_bb->remove_instruction(t);
            cont_bb->append_instruction(t);
        }

        Value* old_res = inst->produces_value() ? inst->result() : nullptr;
        Value* merged_param = nullptr;
        if (old_res) {
            merged_param = b.add_block_param(cont_bb, old_res->type());
            replace_all_value_uses(fn, old_res, merged_param);
        }

        BasicBlock* safe_bb = b.append_block(std::string(cur_bb->name()) + ".sb_ok");
        safe_bb->set_cold(cur_bb->is_cold());
        cur_bb->remove_instruction(inst);
        safe_bb->append_instruction(inst);

        if (item.is_load) {
            ++res.clamped_global_loads;
            BasicBlock* oob_bb = b.append_block(std::string(cur_bb->name()) + ".sb_oob");
            oob_bb->set_cold(true);

            b.position_at_end(cur_bb);
            b.build_br_if(in_bounds, safe_bb, {}, oob_bb, {});

            b.position_at_end(safe_bb);
            if (old_res) {
                b.build_br(cont_bb, {old_res});
            } else {
                b.build_br(cont_bb);
            }

            b.position_at_end(oob_bb);
            if (old_res) {
                Value* zero_v = opts.zero_oob_loads ? build_zero_value(b, old_res->type()) : nullptr;
                if (!zero_v) zero_v = build_zero_value(b, old_res->type());
                if (!zero_v) {
                    res.ok = false;
                    res.violations.push_back(with_loc(opts.debug_ctx, inst->loc(), "unsupported OOB zero type"));
                    continue;
                }
                b.build_br(cont_bb, {zero_v});
            } else {
                b.build_br(cont_bb);
            }
        } else {
            ++res.clamped_global_stores;
            b.position_at_end(cur_bb);
            b.build_br_if(in_bounds, safe_bb, {}, cont_bb, {});

            b.position_at_end(safe_bb);
            b.build_br(cont_bb);
        }
    }

    b.clear_current_loc();
    fn.rebuild_cfg_predecessors();
    if (!verify_function(fn)) {
        res.ok = false;
        res.violations.push_back("instrument_unproven_gpu_accesses produced invalid MIR");
    }
    res.diagnostic = res.violations.empty() ? "gpu global access clamping completed" : join_messages(res.violations);
    return res;
}

GpuLoopFuelResult instrument_gpu_loop_fuel(Function& fn, const GpuLoopFuelOptions& opts) {
    GpuLoopFuelResult res;
    if (!fn.parent() || !fn.entry_block()) {
        res.ok = false;
        res.violations.push_back("invalid function/module for GPU loop fuel");
        res.diagnostic = res.violations.back();
        return res;
    }

    fn.rebuild_cfg_predecessors();
    DominatorTree dom(fn);
    LoopAnalysis la(fn, dom);
    std::vector<LoopInfo*> loops = la.post_order_loops();
    if (loops.empty()) {
        res.diagnostic = "no loops present";
        return res;
    }

    GpuDivergenceAnalysis div(fn);
    Builder b(*fn.parent());
    b.set_function(&fn);

    for (LoopInfo* loop : loops) {
        if (!loop || !loop->header()) continue;
        if (try_detect_static_affine_bound(*loop, opts.max_loop_iterations)) {
            ++res.loops_statically_bounded;
            continue;
        }

        BasicBlock* header = loop->header();
        Instruction* header_term = header->terminator();
        if (!header_term) {
            res.ok = false;
            res.violations.push_back("loop header missing terminator");
            continue;
        }

        bool has_barrier = false;
        DebugLoc barrier_loc;
        for (BasicBlock* lbb : loop->blocks()) {
            if (!lbb) continue;
            for (const Instruction* inst : *lbb) {
                if (is_barrier_call(inst)) {
                    has_barrier = true;
                    barrier_loc = inst->loc();
                    break;
                }
            }
            if (has_barrier) break;
        }

        if (has_barrier && div.is_under_divergent_control(header) && opts.reject_unbounded_divergent_barrier_loops) {
            res.ok = false;
            res.violations.push_back(with_loc(opts.debug_ctx, barrier_loc,
                                              "unbounded loop with barrier under divergent control rejected"));
            continue;
        }

        std::vector<BasicBlock*> ext_exits;
        for (BasicBlock* lbb : loop->blocks()) {
            if (!lbb) continue;
            for (BasicBlock* s : lbb->successors()) {
                if (s && !loop->contains(s) && std::find(ext_exits.begin(), ext_exits.end(), s) == ext_exits.end()) {
                    ext_exits.push_back(s);
                }
            }
        }

        const bool header_controls_exit =
            (header_term->opcode() == Opcode::br_if) &&
            (loop->contains(header_term->true_target().block) != loop->contains(header_term->false_target().block));

        if (!header_controls_exit) {
            if (ext_exits.size() > 1 || (ext_exits.size() == 1 && ext_exits[0]->param_count() > 0) ||
                (ext_exits.empty() && !fn.return_type().is_void())) {
                res.ok = false;
                res.violations.push_back(with_loc(opts.debug_ctx, header_term->loc(),
                                                  "unbounded loop shape unsupported for structured fuel injection"));
                continue;
            }
        }

        Value* fuel_param = b.add_block_param(header, Type::i32());

        // Pass initial fuel on every outside edge entering `header`.
        for (BasicBlock* pred : header->predecessors()) {
            if (!pred || !pred->terminator() || loop->contains(pred)) continue;
            Instruction* pterm = pred->terminator();
            b.position_before(pterm);
            Value* init_fuel = b.build_iconst_i32(static_cast<int32_t>(opts.max_loop_iterations));
            append_arg_to_edges_targeting(pterm, header, init_fuel);
        }

        // Collect all backedge instructions targeting `header`.
        std::vector<Instruction*> latch_terms;
        for (BasicBlock* pred : header->predecessors()) {
            if (pred && pred->terminator() && loop->contains(pred)) {
                latch_terms.push_back(pred->terminator());
            }
        }

        if (header_controls_exit) {
            b.position_before(header_term);
            b.set_current_loc(header_term->loc());
            Value* zero = b.build_iconst_i32(0);
            Value* one = b.build_iconst_i32(1);
            Value* next_fuel = b.build_sub(fuel_param, one);
            for (Instruction* lt : latch_terms) {
                append_arg_to_edges_targeting(lt, header, next_fuel);
            }
            const bool true_in_loop = loop->contains(header_term->true_target().block);
            const bool is_self_loop =
                (header_term->true_target().block == header) ||
                (header_term->false_target().block == header);
            Value* fuel_to_check = is_self_loop ? next_fuel : fuel_param;
            Value* orig_cond = header_term->operand(0);
            Value* new_cond = true_in_loop
                ? b.build_and(orig_cond, b.build_ugt(fuel_to_check, zero))
                : b.build_or(orig_cond, b.build_eq(fuel_to_check, zero));
            header_term->set_operand(0, new_cond);
        } else {
            BasicBlock* exit_bb = ext_exits.empty() ? nullptr : ext_exits.front();
            if (!exit_bb) {
                exit_bb = b.append_block(std::string(header->name()) + ".fuel_exit");
                b.position_at_end(exit_bb);
                b.build_ret_void();
            }

            BasicBlock* body_cont = b.append_block(std::string(header->name()) + ".fuel_body");
            std::vector<Instruction*> moved;
            for (Instruction* inst : *header) {
                if (inst) moved.push_back(inst);
            }
            for (Instruction* inst : moved) {
                header->remove_instruction(inst);
                body_cont->append_instruction(inst);
            }

            b.position_at_end(header);
            Value* zero = b.build_iconst_i32(0);
            Value* has_fuel = b.build_ugt(fuel_param, zero);
            b.build_br_if(has_fuel, body_cont, {}, exit_bb, {});

            b.position_before(header_term);
            Value* next_fuel = b.build_sub(fuel_param, b.build_iconst_i32(1));
            for (Instruction* lt : latch_terms) {
                append_arg_to_edges_targeting(lt, header, next_fuel);
            }
        }

        ++res.loops_fuel_instrumented;
    }

    b.clear_current_loc();
    fn.rebuild_cfg_predecessors();
    if (res.ok && !verify_function(fn)) {
        res.ok = false;
        res.violations.push_back("instrument_gpu_loop_fuel produced invalid MIR");
    }
    res.diagnostic = res.violations.empty() ? "gpu loop fuel check completed" : join_messages(res.violations);
    return res;
}

GpuHardenResult harden_and_verify_gpu_sandbox(Function& mir_fn, const GpuHardenOptions& opts) {
    GpuHardenResult res;
    res.initial_sandbox = verify_gpu_kernel_sandbox(mir_fn, opts.sandbox_opts);
    if (!res.initial_sandbox.portability_ok ||
        (res.initial_sandbox.spirv_checked() && (!res.initial_sandbox.spirv_compiled || !res.initial_sandbox.spirv_parsed))) {
        res.tier = HardeningTier::Rejected;
        res.violations = res.initial_sandbox.violations;
        res.diagnostic = res.initial_sandbox.diagnostic;
        return res;
    }

    if (opts.enable_loop_fuel) {
        res.fuel = instrument_gpu_loop_fuel(mir_fn, opts.fuel_opts);
        if (!res.fuel.ok) {
            res.tier = HardeningTier::Rejected;
            res.violations.insert(res.violations.end(), res.fuel.violations.begin(), res.fuel.violations.end());
            res.diagnostic = join_messages(res.violations);
            return res;
        }
    }

    const bool smt_proven_all =
        res.initial_sandbox.passed() &&
        res.initial_sandbox.proof_tier == SandboxProofTier::FullSmt &&
        res.initial_sandbox.global_bounds_proven &&
        res.initial_sandbox.memory_safe &&
        res.fuel.loops_fuel_instrumented == 0;

    if (smt_proven_all) {
        res.final_sandbox = res.initial_sandbox;
        res.tier = HardeningTier::PureSmtZeroOverhead;
        res.diagnostic = "sandbox verified via pure SMT (zero runtime overhead)";
        return res;
    }

    if (opts.enable_auto_clamp) {
        res.clamp = instrument_unproven_gpu_accesses(mir_fn, opts.sandbox_opts.prover_opts, &res.initial_sandbox, opts.clamp_opts);
        if (!res.clamp.ok) {
            res.tier = HardeningTier::Rejected;
            res.violations.insert(res.violations.end(), res.clamp.violations.begin(), res.clamp.violations.end());
            res.diagnostic = join_messages(res.violations);
            return res;
        }
    } else if (!res.initial_sandbox.global_bounds_proven || !res.initial_sandbox.memory_safe) {
        res.tier = HardeningTier::Rejected;
        res.violations = res.initial_sandbox.violations;
        if (res.violations.empty()) res.violations.push_back("unproven memory accesses and auto-clamp disabled");
        res.diagnostic = join_messages(res.violations);
        return res;
    }

    GpuSandboxOptions post_opts = opts.sandbox_opts;
    post_opts.run_smt_prover = false;
    post_opts.require_full_smt = false;
    post_opts.verify_ptx_equiv = false;
    post_opts.reference_mir = nullptr;
    res.final_sandbox = verify_gpu_kernel_sandbox(mir_fn, post_opts);

    if (!res.final_sandbox.passed()) {
        res.tier = HardeningTier::Rejected;
        res.violations = res.final_sandbox.violations;
        res.diagnostic = res.final_sandbox.diagnostic;
        return res;
    }

    res.tier = HardeningTier::HybridSmtClamped;
    res.diagnostic = "sandbox hardened via hybrid static/SMT + runtime clamping/fuel";
    return res;
}

} // namespace brass::verify
