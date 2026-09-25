#include "baseline_emit_internal.hpp"
#include <brass/codegen/baseline_jit.hpp>
#include <brass/codegen/unsupported_operation.hpp>
#include <brass/target/aarch64/aarch64_baseline_emit.hpp>
#include <brass/target/x64/x64_encoder.hpp>
#include <brass/target/calling_conv.hpp>
#include <brass/gc/runtime_gc.hpp>
#include <brass/runtime/tiering.hpp>
#include <brass/runtime/type_feedback.hpp>
#include <brass/runtime/code_installer.hpp>
#include <brass/runtime/multi_tier_pipeline.hpp>
#include <brass/mir/module.hpp>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <algorithm>

extern "C" void brass_tier1_record_invocation_fb(void* feedback);

namespace brass::codegen {

using namespace brass::x64;

bool BaselineJitCompiler::supports_opcode(Opcode op) noexcept {
    // Vector opcodes are compiled for 128-bit types; the pre-scan rejects
    // 256-bit vectors and the few type / operand combinations it does not.
    switch (op) {
        // Exceptions: the interpreter tier propagates MIR exceptions as C++
        // exceptions and native tiers as a frame-chain unwind; a baseline
        // frame between the two has neither bridge, so a function that
        // throws or catches stays in the interpreter.
        case Opcode::throw_:
        case Opcode::invoke:
        case Opcode::landing_pad:
        case Opcode::resume:
        // Coroutines: the interpreter's coroutine frames hold a MIR Function*
        // and resume by interpreting it, native tiers hold a code pointer;
        // a frame created in one tier cannot be resumed in the other.
        case Opcode::coro_create:
        case Opcode::coro_suspend:
        case Opcode::coro_resume:
        case Opcode::coro_destroy:
            return false;
        default:
            return true;
    }
}

namespace {

// Rejects, before any code is emitted, a function the x64 baseline tier does
// not compile. The tiering layer catches UnsupportedOperation and leaves the
// function in the interpreter.
void check_x64_baseline_supported(const Function& fn, const Target& target, const CallingConvention& cc) {
    // 128-bit vectors are compiled; 256-bit ones are not.
    auto check_type = [&](Type t, std::string_view what) {
        if (t.is_v256()) {
            throw_unsupported(kX64BaselineStage,
                std::string("256-bit vector ") + std::string(what) + " in " + std::string(fn.name()));
        }
    };
    check_type(fn.return_type(), "return");
    for (Type t : fn.param_types()) check_type(t, "parameter");
    if (const BasicBlock* entry = fn.entry_block()) {
        std::vector<Type> types;
        for (const auto* p : entry->params()) types.push_back(p->type());
        if (!bl_vectors_in_registers(target, cc, types)) {
            throw_unsupported(kX64BaselineStage, "vector parameter passed on the stack in " + std::string(fn.name()));
        }
    }
    for (const auto* bb : fn.blocks()) {
        if (!bb) continue;
        for (const auto* p : bb->params()) check_type(p->type(), "block parameter");
        for (const auto* inst : *bb) {
            if (!inst) continue;
            Opcode op = inst->opcode();
            if (!BaselineJitCompiler::supports_opcode(op)) {
                throw_unsupported(kX64BaselineStage, opcode_name(op));
            }
            if (inst->produces_value()) check_type(inst->type(), "value");
            for (size_t i = 0; i < inst->operand_count(); ++i) {
                if (inst->operand(i)) check_type(inst->operand(i)->type(), "operand");
            }
            check_x64_baseline_vector_inst(fn, *inst, target, cc);
            if (op == Opcode::guard) {
                const Function* stub = fn.guard_exit_stub(*inst);
                std::string why;
                if (stub && !fn.guard_exit_stub_matches(*inst, *stub, why)) {
                    throw_unsupported(kX64BaselineStage, why);
                }
                if (!stub && !fn.get_resume_target(inst->resume_id())) {
                    throw_unsupported(kX64BaselineStage, "guard with no exit stub or resume target");
                }
            }
        }
    }
}

void emit_guard(X64BaselineEmitter& em, const Instruction& inst) {
    Label cont = em.buffer.create_label();
    em.test_cond(inst.operand(0));
    em.enc.jne(cont);

    // Guard failed: the same exits the interpreter takes, in its order.
    std::vector<const Value*> state(inst.state_map().begin(), inst.state_map().end());
    if (em.fn.guard_exit_stub(inst)) {
        // The exit stub finishes the function: its result is ours.
        em.emit_call(inst.symbol(), nullptr, state, nullptr, inst.site_id());
        em.emit_return();
    } else {
        BranchTarget resume(em.fn.get_resume_target(inst.resume_id()), inst.state_map());
        em.copy_block_args(resume);
        em.enc.jmp(em.block_label(resume.block));
    }
    em.buffer.bind(cont);
}

void emit_switch(X64BaselineEmitter& em, const Instruction& inst) {
    auto& enc = em.enc;
    // A narrow value owns only the low bytes of its slot; zero-extend exactly
    // its width and compare at that width.
    const Type cond_ty = inst.operand(0)->type();
    const size_t cond_bytes = cond_ty.size_in_bytes();
    const MemAddress cond = em.slot_addr(inst.operand(0));
    if (cond_bytes == 1) enc.movzx8(GPR::RAX, cond);
    else if (cond_bytes == 2) enc.movzx16(GPR::RAX, cond);
    else if (cond_bytes == 4) enc.mov32(GPR::RAX, cond);
    else enc.mov(GPR::RAX, cond);
    std::vector<Label> case_labels;
    case_labels.reserve(inst.switch_cases().size());
    for (const auto& sc : inst.switch_cases()) {
        Label case_body = em.buffer.create_label();
        case_labels.push_back(case_body);
        if (cond_bytes < 8) {
            const uint64_t mask = (uint64_t{1} << (cond_bytes * 8)) - 1;
            enc.cmp32(GPR::RAX, static_cast<int32_t>(static_cast<uint32_t>(
                static_cast<uint64_t>(sc.value) & mask)));
        } else if (sc.value >= INT32_MIN && sc.value <= INT32_MAX) {
            enc.cmp(GPR::RAX, static_cast<int32_t>(sc.value));
        } else {
            enc.movabs(GPR::RCX, static_cast<uint64_t>(sc.value));
            enc.cmp(GPR::RAX, GPR::RCX);
        }
        enc.je(case_body);
    }
    em.copy_block_args(inst.default_target());
    enc.jmp(em.block_label(inst.default_target().block));
    for (size_t i = 0; i < inst.switch_cases().size(); ++i) {
        em.buffer.bind(case_labels[i]);
        em.copy_block_args(inst.switch_cases()[i].target);
        enc.jmp(em.block_label(inst.switch_cases()[i].target.block));
    }
}

// Calls, safepoints, speculation and terminators.
void emit_control_op(X64BaselineEmitter& em, const Instruction& inst) {
    auto& enc = em.enc;
    Opcode op = inst.opcode();
    switch (op) {
        case Opcode::safepoint: {
            em.call_abs(reinterpret_cast<const void*>(&brass_gc_safepoint));
            em.record_safepoint(inst.site_id());
            return;
        }
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
            enc.jmp(em.block_label(inst.branch_target().block));
            return;
        case Opcode::br_if: {
            em.test_cond(inst.operand(0));
            Label true_edge = em.buffer.create_label();
            enc.jne(true_edge);
            em.copy_block_args(inst.false_target());
            enc.jmp(em.block_label(inst.false_target().block));
            em.buffer.bind(true_edge);
            em.copy_block_args(inst.true_target());
            enc.jmp(em.block_label(inst.true_target().block));
            return;
        }
        case Opcode::switch_:
            emit_switch(em, inst);
            return;
        case Opcode::ret: {
            if (inst.operand_count() > 0 && inst.operand(0) != nullptr) {
                const Value* rval = inst.operand(0);
                if (bl_is_v128(rval->type())) {
                    enc.movups(XMM::XMM0, em.slot_addr(rval));
                } else if (rval->type().is_float()) {
                    if (bl_is_f32(rval->type())) enc.movss(XMM::XMM0, em.slot_addr(rval));
                    else enc.movsd(XMM::XMM0, em.slot_addr(rval));
                } else {
                    em.load_gpr(GPR::RAX, rval);
                }
            }
            em.emit_return();
            return;
        }
        case Opcode::unreachable:
            // A trap by definition, not a gap.
            enc.ud2();
            return;
        case Opcode::guard:
            emit_guard(em, inst);
            return;
        case Opcode::resume_point:
        case Opcode::osr_entry:
            // Metadata markers: no-ops in forward execution.
            return;
        default:
            // The pre-scan admitted it, so an emitter is missing: a bug.
            throw_unsupported(kX64BaselineStage, std::string("no emitter for ") + std::string(opcode_name(op)));
    }
}

} // namespace

bool BaselineJitCompiler::passes_prescan(const Function& fn, Target target) const {
    if (target.is_aarch64()) return true;
    try {
        check_x64_baseline_supported(fn, target, CallingConvention::for_target(target));
    } catch (const UnsupportedOperation&) {
        return false;
    }
    return true;
}

BaselineCompiledFunction BaselineJitCompiler::compile(const Function& fn, Target target) {
    if (target.is_aarch64()) {
        return aarch64::compile_baseline_aarch64(fn, target, [this, &fn](std::string_view name) {
            return resolve_symbol_in(fn, name);
        }, &dispatch_table().tiering(), lazy_, [this, &fn](std::string_view name) {
            return function_address_in(fn, name);
        });
    }

    CallingConvention cc = CallingConvention::for_target(target);
    check_x64_baseline_supported(fn, target, cc);

    CodeBuffer buffer;
    X64Encoder enc(buffer);

    // 1. Stack frame slots [rbp - offset], shared between values whose live
    //    ranges do not overlap (baseline_frame.cpp).
    bool preserves_r13 = (fn.parent() && fn.parent()->pinned_tls_register());
    BaselineFrameLayout layout = layout_baseline_frame(fn, preserves_r13 ? 8 : 0, kX64BaselineStage);
    const auto& slot_map = layout.slot_map;
    const auto& alloca_offsets = layout.alloca_offsets;
    const auto& gcref_slots = layout.gcref_slots;

    int32_t frame_size = (layout.size + 15) & ~15;
    if (target.is_windows()) frame_size += 32; // shadow space for outgoing calls
    frame_size = (frame_size + 15) & ~15;
    if (frame_size == 0) frame_size = 16;

    auto slot_off_addr = [&](int32_t off) { return MemAddress::base_disp(GPR::RBP, -off); };

    Label fn_entry_label = buffer.create_label();
    buffer.bind(fn_entry_label);

    // 2. Prologue (described to the unwinder: baseline_unwind.cpp)
    X64BaselinePrologue prologue;
    prologue.frame_size = frame_size;
    enc.push(GPR::RBP);
    prologue.push_end = static_cast<uint32_t>(buffer.size());
    enc.mov(GPR::RBP, GPR::RSP);
    prologue.mov_end = static_cast<uint32_t>(buffer.size());
    enc.sub(GPR::RSP, frame_size);
    prologue.alloc_end = static_cast<uint32_t>(buffer.size());
    if (preserves_r13) {
        enc.mov(MemAddress::base_disp(GPR::RBP, -8), GPR::R13);
        prologue.r13_save_end = static_cast<uint32_t>(buffer.size());
    }

    // Zero the gcref slots so a GC before their first store sees null.
    if (!gcref_slots.empty()) {
        enc.xor_(GPR::RAX, GPR::RAX);
        for (int32_t off : gcref_slots) enc.mov(slot_off_addr(off), GPR::RAX);
    }

    // 3. Incoming parameters into the entry block's parameter slots
    if (BasicBlock* entry_bb = fn.entry_block()) {
        size_t gpr_idx = 0, xmm_idx = 0, stack_idx = 0;
        static constexpr GPR kWinGprs[4] = {GPR::RCX, GPR::RDX, GPR::R8, GPR::R9};
        static constexpr XMM kWinXmms[4] = {XMM::XMM0, XMM::XMM1, XMM::XMM2, XMM::XMM3};
        for (size_t i = 0; i < entry_bb->param_count(); ++i) {
            const Value* param = entry_bb->param(i);
            MemAddress dst = slot_off_addr(slot_map.at(param));
            const bool is_vec = bl_is_v128(param->type());
            const bool is_flt = bl_in_xmm(param->type());
            auto from_stack = [&](int32_t caller_off) {
                // The pre-scan admitted no vector on the stack.
                if (is_vec) throw_unsupported(kX64BaselineStage, "vector parameter on the stack");
                enc.mov(GPR::RAX, MemAddress::base_disp(GPR::RBP, caller_off));
                enc.mov(dst, GPR::RAX);
            };
            auto from_xmm = [&](XMM x) {
                if (is_vec) enc.movups(dst, x);
                else enc.movsd(dst, x);
            };
            if (target.is_windows()) {
                if (i < 4) {
                    if (is_flt) from_xmm(kWinXmms[i]);
                    else enc.mov(dst, kWinGprs[i]);
                } else {
                    from_stack(static_cast<int32_t>(16 + 32 + (i - 4) * 8));
                }
            } else if (is_flt) {
                if (xmm_idx < cc.arg_xmms().size()) from_xmm(cc.arg_xmms()[xmm_idx++]);
                else from_stack(static_cast<int32_t>(16 + stack_idx++ * 8));
            } else {
                if (gpr_idx < cc.arg_gprs().size()) enc.mov(dst, cc.arg_gprs()[gpr_idx++]);
                else from_stack(static_cast<int32_t>(16 + stack_idx++ * 8));
            }
        }
    }

    // Record-invocation hook: the function's TieringFeedback, from the
    // registry of the program this compiler publishes into, is resolved now
    // and baked in (the registry retires, never frees, feedback objects, and
    // lives as long as the program that owns this code). The hook reaches
    // the program's pipeline through it.
    if (!fn.name().empty()) {
        runtime::TieringFeedback* feedback = &dispatch_table().tiering().get_feedback(fn.name());
        enc.movabs(target.is_windows() ? GPR::RCX : GPR::RDI, reinterpret_cast<uint64_t>(feedback));
        enc.movabs(GPR::R11, reinterpret_cast<uint64_t>(reinterpret_cast<void*>(&brass_tier1_record_invocation_fb)));
        enc.call(GPR::R11);
    }

    // 4. Block labels
    std::unordered_map<uint32_t, Label> block_labels;
    for (const auto* bb : fn.blocks()) {
        if (bb) block_labels[bb->id()] = buffer.create_label();
    }

    FunctionStackMap fn_stack_map;
    fn_stack_map.function_name = std::string(fn.name());

    X64BaselineEmitter emitter{
        buffer, enc, target, fn, slot_map, alloca_offsets, block_labels, fn_stack_map,
        [this, &fn](std::string_view name) { return resolve_symbol_in(fn, name); },
        frame_size, cc, fn_entry_label, gcref_slots, preserves_r13, lazy_.get()
    };
    emitter.function_address = [this, &fn](std::string_view name) { return function_address_in(fn, name); };

    // 5. Code for each block
    for (const auto* bb : fn.blocks()) {
        if (!bb) continue;
        buffer.bind(block_labels[bb->id()]);
        for (const auto* inst_ptr : *bb) {
            if (!inst_ptr) continue;
            const Instruction& inst = *inst_ptr;
            if (emit_baseline_x64_vec_op(emitter, inst)) continue;
            if (emit_baseline_x64_op(emitter, inst)) continue;
            if (emit_baseline_x64_fp_op(emitter, inst)) continue;
            emit_control_op(emitter, inst);
        }
    }

    // Executable memory: the code, then (when this process runs it) the
    // unwind data that lets a C++ exception from a helper unwind through
    // the frame.
    size_t code_bytes = buffer.size();
    std::vector<uint8_t> image(buffer.data(), buffer.data() + code_bytes);
    const Target host = Target::host();
    const bool runs_here = host.is_x64() && host.is_windows() == target.is_windows();
    size_t unwind_off = 0;
    if (runs_here) {
        unwind_off = append_x64_baseline_unwind(image, prologue, static_cast<uint32_t>(code_bytes),
                                                target.is_windows());
    }
    auto mem_block = std::make_shared<JitMemoryBlock>(image.size());
    if (!mem_block->is_valid()) {
        throw std::runtime_error("BaselineJitCompiler: Failed to allocate executable memory for " + std::string(fn.name()));
    }
    std::memcpy(mem_block->data(), image.data(), image.size());
    if (!mem_block->make_executable_read_only()) {
        throw std::runtime_error("BaselineJitCompiler: could not make the code of " + std::string(fn.name()) + " executable");
    }
    if (runs_here && !mem_block->register_unwind_info(unwind_off, 1)) {
        throw std::runtime_error("BaselineJitCompiler: the unwinder refused the unwind data of " + std::string(fn.name()));
    }

    void* entry_ptr = mem_block->data();
    fn_stack_map.function_address = reinterpret_cast<uintptr_t>(entry_ptr);
    fn_stack_map.code_size = static_cast<uint32_t>(code_bytes);

    BaselineCompiledFunction compiled(
        fn.name(), fn.return_type(), fn.param_types(), mem_block, entry_ptr, code_bytes, std::move(fn_stack_map));
    if (emitter.uses_lazy_stubs) compiled.set_link_keepalive(lazy_);
    compiled.set_lazy_call_symbols(std::move(emitter.lazy_call_symbols));
    compiled.set_lazy_addr_symbols(std::move(emitter.lazy_addr_symbols));
    return compiled;
}

} // namespace brass::codegen
