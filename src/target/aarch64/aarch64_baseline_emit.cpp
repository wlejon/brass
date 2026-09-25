// AArch64 baseline tier: the pre-scan, the frame, parameters, control flow
// and the compiled function. The tier mirrors the x64 baseline
// (codegen/baseline_emit.cpp); see aarch64_baseline_emit_internal.hpp for
// the frame.
#include "aarch64_baseline_emit_internal.hpp"
#include <brass/target/aarch64/aarch64_isel.hpp>
#include <brass/codegen/baseline_jit.hpp>
#include <brass/gc/runtime_gc.hpp>
#include <brass/runtime/tiering.hpp>
#include <brass/runtime/code_installer.hpp>
#include <algorithm>
#include <cstring>
#include <stdexcept>

extern "C" void brass_tier1_record_invocation_fb(void* feedback);

namespace brass::aarch64 {

using namespace brass::codegen;

namespace {

// Rejects, before any code is emitted, a function this tier does not
// compile. The tiering layer catches UnsupportedOperation and leaves the
// function in the interpreter.
void check_supported(const Function& fn, const Target& target) {
    auto check_type = [&](Type t, std::string_view what) {
        if (t.is_v256()) {
            throw_unsupported(kA64BaselineStage,
                std::string("256-bit vector ") + std::string(what) + " in " + std::string(fn.name()));
        }
    };
    check_type(fn.return_type(), "return");
    for (Type t : fn.param_types()) check_type(t, "parameter");
    if (const BasicBlock* entry = fn.entry_block()) {
        std::vector<Type> types;
        for (const auto* p : entry->params()) types.push_back(p->type());
        const auto locs = a64_assign_args(target, types, nullptr);
        for (size_t i = 0; i < types.size(); ++i) {
            if (types[i].is_vector() && !locs[i].in_reg) {
                throw_unsupported(kA64BaselineStage, "vector parameter passed on the stack in " + std::string(fn.name()));
            }
        }
    }
    for (const auto* bb : fn.blocks()) {
        if (!bb) continue;
        for (const auto* p : bb->params()) check_type(p->type(), "block parameter");
        for (const auto* inst : *bb) {
            if (!inst) continue;
            const Opcode op = inst->opcode();
            if (!BaselineJitCompiler::supports_opcode(op)) {
                throw_unsupported(kA64BaselineStage, opcode_name(op));
            }
            // Exceptions are compiled by the x64 tier only: here a function
            // that throws or catches stays in the interpreter.
            if (op == Opcode::throw_ || op == Opcode::invoke || op == Opcode::landing_pad ||
                op == Opcode::resume) {
                throw_unsupported(kA64BaselineStage, opcode_name(op));
            }
            if (inst->produces_value()) check_type(inst->type(), "value");
            for (size_t i = 0; i < inst->operand_count(); ++i) {
                if (inst->operand(i)) check_type(inst->operand(i)->type(), "operand");
            }
            check_aarch64_baseline_vector_inst(*inst, target);
            if (op == Opcode::guard) {
                const Module* mod = fn.parent();
                const bool has_stub = mod && !inst->symbol().empty() && mod->get_function(inst->symbol());
                if (!has_stub && !fn.get_resume_target(inst->resume_id())) {
                    throw_unsupported(kA64BaselineStage, "guard with no exit stub or resume target");
                }
            }
        }
    }
}

// Branches to `taken` when the value is non-zero, reading it at its width.
void branch_if_nonzero(AArch64BaselineEmitter& em, const Value* cond, Label taken) {
    em.load_gpr(GPR::X0, cond);
    if (bl_is_int32(cond->type())) em.enc.cbnz32(GPR::X0, taken);
    else em.enc.cbnz(GPR::X0, taken);
}

void emit_guard(AArch64BaselineEmitter& em, const Instruction& inst) {
    Label cont = em.buffer.create_label();
    branch_if_nonzero(em, inst.operand(0), cont);

    // Guard failed: the same exits the interpreter takes, in its order.
    const Module* mod = em.fn.parent();
    std::vector<const Value*> state(inst.state_map().begin(), inst.state_map().end());
    if (mod && !inst.symbol().empty() && mod->get_function(inst.symbol())) {
        // The exit stub finishes the function: its result is ours.
        const Value* result = nullptr;
        em.emit_call(inst.symbol(), nullptr, state, result, inst.site_id());
        em.emit_return();
    } else {
        BranchTarget resume(em.fn.get_resume_target(inst.resume_id()), inst.state_map());
        em.copy_block_args(resume);
        em.enc.b(em.block_labels.at(resume.block->id()));
    }
    em.buffer.bind(cont);
}

void emit_switch(AArch64BaselineEmitter& em, const Instruction& inst) {
    auto& enc = em.enc;
    // A narrow value owns only the low bytes of its slot; zero-extend exactly
    // its width and compare the cases masked to it.
    const size_t cond_bytes = inst.operand(0)->type().size_in_bytes();
    const Value* cond = inst.operand(0);
    if (cond_bytes == 1) enc.ldrb(GPR::X0, em.slot_addr(cond, 1));
    else if (cond_bytes == 2) enc.ldrh(GPR::X0, em.slot_addr(cond, 2));
    else if (cond_bytes == 4) enc.ldr32(GPR::X0, em.slot_addr(cond, 4));
    else enc.ldr(GPR::X0, em.slot_addr(cond, 8));
    const uint64_t mask = cond_bytes < 8 ? (uint64_t{1} << (cond_bytes * 8)) - 1 : ~uint64_t{0};
    std::vector<Label> case_labels;
    case_labels.reserve(inst.switch_cases().size());
    for (const auto& sc : inst.switch_cases()) {
        Label body = em.buffer.create_label();
        case_labels.push_back(body);
        const uint64_t v = static_cast<uint64_t>(sc.value) & mask;
        if (v <= 4095) {
            enc.cmp(GPR::X0, static_cast<uint32_t>(v));
        } else {
            enc.mov(GPR::X1, v);
            enc.cmp(GPR::X0, GPR::X1);
        }
        enc.b(Condition::EQ, body);
    }
    em.copy_block_args(inst.default_target());
    enc.b(em.block_labels.at(inst.default_target().block->id()));
    for (size_t i = 0; i < inst.switch_cases().size(); ++i) {
        em.buffer.bind(case_labels[i]);
        em.copy_block_args(inst.switch_cases()[i].target);
        enc.b(em.block_labels.at(inst.switch_cases()[i].target.block->id()));
    }
}

// Calls, safepoints, speculation and terminators.
void emit_control_op(AArch64BaselineEmitter& em, const Instruction& inst) {
    auto& enc = em.enc;
    const Opcode op = inst.opcode();
    switch (op) {
        case Opcode::safepoint:
            em.call_abs(reinterpret_cast<const void*>(&brass_gc_safepoint));
            em.record_safepoint(inst.site_id());
            return;
        case Opcode::call:
        case Opcode::patchable_call:
        case Opcode::call_indirect: {
            const bool indirect = (op == Opcode::call_indirect);
            std::vector<const Value*> args;
            for (size_t i = indirect ? 1 : 0; i < inst.operand_count(); ++i) args.push_back(inst.operand(i));
            // A patchable call's symbol names the patch site; the callee is
            // the extra symbol. Baseline code registers no patch sites, so
            // it always calls the default callee.
            std::string_view callee = inst.symbol();
            if (op == Opcode::patchable_call && !inst.extra_symbol().empty()) callee = inst.extra_symbol();
            em.emit_call(callee, indirect ? inst.operand(0) : nullptr, args,
                         inst.produces_value() ? inst.result() : nullptr, inst.site_id());
            return;
        }
        case Opcode::br:
            em.copy_block_args(inst.branch_target());
            enc.b(em.block_labels.at(inst.branch_target().block->id()));
            return;
        case Opcode::br_if: {
            Label true_edge = em.buffer.create_label();
            branch_if_nonzero(em, inst.operand(0), true_edge);
            em.copy_block_args(inst.false_target());
            enc.b(em.block_labels.at(inst.false_target().block->id()));
            em.buffer.bind(true_edge);
            em.copy_block_args(inst.true_target());
            enc.b(em.block_labels.at(inst.true_target().block->id()));
            return;
        }
        case Opcode::switch_:
            emit_switch(em, inst);
            return;
        case Opcode::ret: {
            if (inst.operand_count() > 0 && inst.operand(0) != nullptr) {
                const Value* rval = inst.operand(0);
                if (bl_is_v128(rval->type())) em.load_v(FPR::V0, rval);
                else if (rval->type().is_float()) em.load_fp(FPR::V0, rval);
                else em.load_gpr(GPR::X0, rval);
            }
            em.emit_return();
            return;
        }
        case Opcode::unreachable:
            // A trap by definition, not a gap.
            enc.brk(0);
            return;
        case Opcode::guard:
            emit_guard(em, inst);
            return;
        case Opcode::resume_point:
        case Opcode::keep_alive:
            // A metadata marker, and a use of a value whose slot is already
            // a root for the whole frame: no-ops in forward execution.
            return;
        default:
            // The pre-scan admitted it, so an emitter is missing: a bug.
            throw_unsupported(kA64BaselineStage, std::string("no emitter for ") + std::string(opcode_name(op)));
    }
}

struct Pass {
    CodeBuffer buffer;
    FunctionStackMap stack_map;
    uint32_t stp_end = 0;
    uint32_t mov_end = 0;
    uint32_t tls_save_end = 0;
    uint32_t prologue_end = 0;
    uint32_t alloc_bytes = 0;
    bool uses_lazy_stubs = false;
    std::vector<std::string> lazy_call_symbols;
    std::vector<std::string> lazy_addr_symbols;
    // Where each source position begins (BaselineCompiledFunction::line_table).
    std::vector<DebugLineEntry> lines;
};

} // namespace

codegen::BaselineCompiledFunction compile_baseline_aarch64(
    const Function& fn,
    Target target,
    BaselineSymbolResolver resolver,
    runtime::TieringRegistry* tiering,
    std::shared_ptr<codegen::LazySymbolTable> lazy,
    BaselineSymbolResolver function_address
) {
    check_supported(fn, target);

    // 1. Frame: slots, the parallel-copy area and the outgoing arguments.
    const bool preserves_tls = fn.parent() && fn.parent()->pinned_tls_register();
    // The pinned-TLS register is pushed right below the frame record, at
    // fp - 16, so the slots start after it.
    const BaselineFrameLayout layout = layout_baseline_frame(fn, preserves_tls ? 16 : 0, kA64BaselineStage);
    const int32_t out_bytes = static_cast<int32_t>(a64_baseline_outgoing_bytes(fn, target));
    const int32_t copy_bytes = static_cast<int32_t>(a64_baseline_copy_bytes(fn));
    const int32_t slot_bytes = (layout.size + 15) & ~15;
    const int32_t frame_bytes = slot_bytes + copy_bytes + out_bytes;

    runtime::TieringRegistry& registry = tiering ? *tiering : runtime::TieringRegistry::instance();
    runtime::TieringFeedback* feedback = fn.name().empty() ? nullptr : &registry.get_feedback(fn.name());

    // Short branches whose label ends up out of reach are re-emitted long on
    // the next pass (CodeBuffer::relax_requests), as the optimizing tier
    // does (aarch64_emit.cpp).
    auto emit_pass = [&](const std::vector<uint32_t>& long_sites) {
        Pass pass;
        CodeBuffer& buffer = pass.buffer;
        buffer.add_long_branch_sites(long_sites);
        AArch64Encoder enc(buffer);
        pass.stack_map.function_name = std::string(fn.name());

        Label fn_entry_label = buffer.create_label();
        buffer.bind(fn_entry_label);

        // 2. Prologue (described to the unwinder: append_aarch64_baseline_eh_frame)
        enc.stp(GPR::FP, GPR::LR, pre_idx(GPR::SP, -16));
        pass.stp_end = static_cast<uint32_t>(buffer.size());
        enc.mov(GPR::FP, GPR::SP);
        pass.mov_end = static_cast<uint32_t>(buffer.size());
        int32_t alloc_bytes = frame_bytes;
        if (preserves_tls) {
            enc.str(kPinnedTlsGpr, pre_idx(GPR::SP, -16));
            pass.tls_save_end = static_cast<uint32_t>(buffer.size());
            alloc_bytes -= 16;
        }
        if (alloc_bytes > 0) enc.sub(GPR::SP, GPR::SP, static_cast<uint32_t>(alloc_bytes));
        pass.alloc_bytes = static_cast<uint32_t>(alloc_bytes);
        pass.prologue_end = static_cast<uint32_t>(buffer.size());

        std::unordered_map<uint32_t, Label> block_labels;
        for (const auto* bb : fn.blocks()) {
            if (bb) block_labels[bb->id()] = buffer.create_label();
        }

        AArch64BaselineEmitter em{
            buffer, enc, target, fn, layout, block_labels, pass.stack_map, resolver,
            frame_bytes, out_bytes, fn_entry_label, preserves_tls, lazy
        };
        em.function_address = function_address;

        // Zero the root slots so a GC before their first store sees null.
        for (int32_t off : layout.gcref_slots) enc.str(GPR::XZR, em.off_addr(off, 8));
        for (int32_t off : layout.tagged_slots) enc.str(GPR::XZR, em.off_addr(off, 8));

        // 3. Incoming parameters into the entry block's parameter slots.
        if (const BasicBlock* entry_bb = fn.entry_block()) {
            std::vector<Type> types;
            for (const auto* p : entry_bb->params()) types.push_back(p->type());
            const auto locs = a64_assign_args(target, types, nullptr);
            for (size_t i = 0; i < entry_bb->param_count(); ++i) {
                const Value* param = entry_bb->param(i);
                const A64ArgLoc& loc = locs[i];
                const Type t = param->type();
                if (loc.in_reg) {
                    if (!loc.fp) em.store_gpr(param, static_cast<GPR>(loc.reg));
                    else if (bl_is_v128(t)) em.store_v(param, static_cast<FPR>(loc.reg));
                    else em.store_fp(param, static_cast<FPR>(loc.reg));
                    continue;
                }
                // The caller's outgoing area, above the frame record. The
                // pre-scan admitted no vector here.
                const int64_t at = 16 + static_cast<int64_t>(loc.stack_offset);
                switch (loc.size) {
                    case 1: enc.ldrb(GPR::X9, em.based(GPR::FP, at, 1)); enc.str32(GPR::X9, em.slot_addr(param, 4)); break;
                    case 2: enc.ldrh(GPR::X9, em.based(GPR::FP, at, 2)); enc.str32(GPR::X9, em.slot_addr(param, 4)); break;
                    case 4: enc.ldr32(GPR::X9, em.based(GPR::FP, at, 4)); enc.str32(GPR::X9, em.slot_addr(param, 4)); break;
                    default:
                        if (bl_is_int32(t) || bl_is_f32(t)) {
                            enc.ldr32(GPR::X9, em.based(GPR::FP, at, 4));
                            enc.str32(GPR::X9, em.slot_addr(param, 4));
                        } else {
                            enc.ldr(GPR::X9, em.based(GPR::FP, at, 8));
                            enc.str(GPR::X9, em.slot_addr(param, 8));
                        }
                        break;
                }
            }
        }

        // Record-invocation hook: the function's TieringFeedback, from the
        // registry of the program this code belongs to, is resolved now and
        // baked in (the registry retires, never frees, feedback objects, and
        // lives as long as the program that owns this code).
        if (feedback) {
            enc.mov(GPR::X0, reinterpret_cast<uint64_t>(feedback));
            em.call_abs(reinterpret_cast<const void*>(&brass_tier1_record_invocation_fb));
        }

        // 4. Code for each block.
        for (const auto* bb : fn.blocks()) {
            if (!bb) continue;
            buffer.bind(block_labels[bb->id()]);
            for (const auto* inst_ptr : *bb) {
                if (!inst_ptr) continue;
                const Instruction& inst = *inst_ptr;
                if (inst.loc().is_valid() && (pass.lines.empty() || pass.lines.back().loc != inst.loc())) {
                    pass.lines.push_back({static_cast<uint32_t>(buffer.size()), inst.loc()});
                }
                if (emit_baseline_aarch64_vec_op(em, inst)) continue;
                if (emit_baseline_aarch64_op(em, inst)) continue;
                if (emit_baseline_aarch64_fp_op(em, inst)) continue;
                emit_control_op(em, inst);
            }
        }
        pass.uses_lazy_stubs = em.uses_lazy_stubs;
        pass.lazy_call_symbols = std::move(em.lazy_call_symbols);
        pass.lazy_addr_symbols = std::move(em.lazy_addr_symbols);
        return pass;
    };

    std::vector<uint32_t> long_sites;
    Pass pass = emit_pass(long_sites);
    while (!pass.buffer.relax_requests().empty()) {
        const size_t before = long_sites.size();
        for (uint32_t site : pass.buffer.relax_requests()) {
            if (std::find(long_sites.begin(), long_sites.end(), site) == long_sites.end()) long_sites.push_back(site);
        }
        if (long_sites.size() == before) {
            throw std::runtime_error("compile_baseline_aarch64: branch relaxation made no progress in " +
                                     std::string(fn.name()));
        }
        pass = emit_pass(long_sites);
    }

    // Executable memory: the code, then (when this process runs it) the
    // unwind data that lets a C++ exception from a helper unwind through
    // the frame: .xdata + RUNTIME_FUNCTION on Windows, .eh_frame elsewhere.
    const size_t code_bytes = pass.buffer.size();
    std::vector<uint8_t> image(pass.buffer.data(), pass.buffer.data() + code_bytes);
    const Target host = Target::host();
    const bool runs_here = host.is_aarch64() && host.is_windows() == target.is_windows();
    A64BaselinePrologue prologue;
    prologue.stp_end = pass.stp_end;
    prologue.mov_end = pass.mov_end;
    prologue.tls_save_end = pass.tls_save_end;
    prologue.alloc_bytes = pass.alloc_bytes;
    size_t unwind_off = 0;
    if (runs_here) {
        unwind_off = target.is_windows()
            ? append_aarch64_baseline_win_unwind(image, static_cast<uint32_t>(code_bytes), prologue)
            : append_aarch64_baseline_eh_frame(image, static_cast<uint32_t>(code_bytes), prologue);
    }
    auto mem_block = std::make_shared<JitMemoryBlock>(image.size());
    if (!mem_block->is_valid()) {
        throw std::runtime_error("compile_baseline_aarch64: Failed to allocate executable memory for " + std::string(fn.name()));
    }
    std::memcpy(mem_block->data(), image.data(), image.size());
    if (!mem_block->make_executable_read_only()) {
        throw std::runtime_error("compile_baseline_aarch64: could not make the code of " + std::string(fn.name()) + " executable");
    }
    if (runs_here && !mem_block->register_unwind_info(unwind_off, 1)) {
        throw std::runtime_error("compile_baseline_aarch64: the unwinder refused the unwind data of " + std::string(fn.name()));
    }

    void* entry_ptr = mem_block->data();
    pass.stack_map.function_address = reinterpret_cast<uintptr_t>(entry_ptr);
    pass.stack_map.code_size = static_cast<uint32_t>(code_bytes);

    BaselineCompiledFunction compiled(
        fn.name(), fn.return_type(), fn.param_types(), mem_block, entry_ptr, code_bytes, std::move(pass.stack_map));
    if (pass.uses_lazy_stubs) compiled.set_link_keepalive(lazy);
    compiled.set_lazy_call_symbols(std::move(pass.lazy_call_symbols));
    compiled.set_lazy_addr_symbols(std::move(pass.lazy_addr_symbols));
    compiled.set_line_table(std::move(pass.lines));
    return compiled;
}

void check_aarch64_baseline_supported(const Function& fn, const Target& target) { check_supported(fn, target); }

} // namespace brass::aarch64
