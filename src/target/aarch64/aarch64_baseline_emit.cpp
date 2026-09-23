#include "aarch64_baseline_emit_internal.hpp"
#include <brass/target/aarch64/aarch64_encoder.hpp>
#include <brass/target/aarch64/code_buffer.hpp>
#include <brass/target/calling_conv.hpp>
#include <brass/gc/runtime_gc.hpp>
#include <brass/mir/gc_refs.hpp>
#include <brass/runtime/tiering.hpp>
#include <brass/runtime/type_feedback.hpp>
#include <brass/runtime/code_installer.hpp>
#include <brass/runtime/multi_tier_pipeline.hpp>
#include <brass/core/string_pool.hpp>
#include <cstring>
#include <cmath>
#include <stdexcept>
#include <algorithm>
#include <iostream>
#include <unordered_map>
#include <vector>

extern "C" void brass_tier1_record_invocation_fb(void* feedback);

namespace brass::aarch64 {

using namespace brass::codegen;

MemAddress AArch64BaselineEmitter::ensure_accessible_mem(const MemAddress& mem, GPR scratch, int size_bytes) const {
    if (mem.mode != AddrMode::Offset) return mem;
    int64_t off = mem.offset;
    if (off >= -256 && off <= 255) {
        return mem;
    }
    if (size_bytes > 0 && off >= 0 && (off % size_bytes == 0) && (off / size_bytes <= 4095)) {
        return mem;
    }
    if (off >= 0 && off <= 4095) {
        enc.add(scratch, mem.base, static_cast<uint32_t>(off));
    } else if (off < 0 && -off <= 4095) {
        enc.sub(scratch, mem.base, static_cast<uint32_t>(-off));
    } else {
        enc.mov(scratch, static_cast<uint64_t>(off));
        enc.add(scratch, mem.base, scratch);
    }
    return ptr(scratch, 0);
}

MemAddress AArch64BaselineEmitter::slot_addr(const Value* val) const {
    auto it = slot_map.find(val);
    int32_t off = 16 + ((it != slot_map.end()) ? it->second : 0);
    return ensure_accessible_mem(ptr(GPR::FP, off));
}

MemAddress AArch64BaselineEmitter::slot_off_addr(int32_t off) const {
    return ensure_accessible_mem(ptr(GPR::FP, 16 + off));
}

void* AArch64BaselineEmitter::resolve_sym(std::string_view name) const {
    if (resolver) {
        void* ptr_val = resolver(name);
        if (ptr_val) return ptr_val;
    }
    auto* handle = runtime::FunctionDispatchTable::instance().find(name);
    if (handle && handle->native_entry()) {
        return handle->native_entry();
    }
    return nullptr;
}

void* AArch64BaselineEmitter::resolve_or_stub(std::string_view name) {
    if (void* addr = resolve_sym(name)) return addr;
    if (!lazy) return nullptr;
    uses_lazy_stubs = true;
    return lazy->stub_for(name);
}

void AArch64BaselineEmitter::copy_block_args(const BranchTarget& target_branch) {
    if (!target_branch.block || target_branch.args.empty()) return;
    const auto& params = target_branch.block->params();
    size_t count = std::min(target_branch.args.size(), params.size());
    for (size_t j = 0; j < count; ++j) {
        const Value* a = target_branch.args[j];
        enc.ldr(GPR::X0, slot_addr(a));
        enc.str(GPR::X0, pre_idx(GPR::SP, -16));
    }
    for (size_t j = 0; j < count; ++j) {
        size_t idx = count - 1 - j;
        enc.ldr(GPR::X0, post_idx(GPR::SP, 16));
        enc.str(GPR::X0, slot_addr(params[idx]));
    }
}

codegen::BaselineCompiledFunction compile_baseline_aarch64(
    const Function& fn,
    Target target,
    BaselineSymbolResolver resolver,
    runtime::TieringRegistry* tiering,
    std::shared_ptr<codegen::LazySymbolTable> lazy
) {
    (void)target;
    CodeBuffer buffer;
    AArch64Encoder enc(buffer);

    // 1. Assign deterministic stack frame slots relative to FP: [fp, 16 + offset]
    std::unordered_map<const Value*, int32_t> slot_map;
    std::unordered_map<const Instruction*, int32_t> alloca_offsets;
    std::vector<int32_t> gcref_slots;
    int32_t current_offset = 0;

    auto alloc_slot = [&](const Value* val) -> int32_t {
        if (!val) return 0;
        auto it = slot_map.find(val);
        if (it != slot_map.end()) return it->second;

        int32_t size = 8;
        if (val->type().is_vector()) {
            size = val->type().is_v256() ? 32 : 16;
        }
        int32_t slot_off = current_offset;
        current_offset += size;
        slot_map[val] = slot_off;
        // A derived gcref points inside an object and is never live across a
        // GC point (verifier_gc.cpp), so it is not a root.
        if (val->type().is_gcref() && !is_derived_gcref(val)) {
            gcref_slots.push_back(slot_off);
        }
        return slot_off;
    };

    for (const auto* bb : fn.blocks()) {
        if (!bb) continue;
        for (const auto* param : bb->params()) {
            alloc_slot(param);
        }
        for (const auto* inst : *bb) {
            if (!inst) continue;
            if (inst->opcode() == Opcode::alloca_) {
                int32_t buf_size = inst->imm_i32();
                int32_t align = inst->offset() > 0 ? inst->offset() : 16;
                if (align < 16) align = 16;
                current_offset = (current_offset + align - 1) & ~(align - 1);
                alloca_offsets[inst] = current_offset;
                current_offset += buf_size;
            }
            if (inst->produces_value()) {
                alloc_slot(inst->result());
            }
        }
    }

    // Align frame size to 16 bytes: 16 bytes header (FP + LR) + total local slots
    int32_t total_slots = (current_offset + 15) & ~15;
    int32_t frame_size = 16 + total_slots;
    frame_size = (frame_size + 15) & ~15;
    if (frame_size < 16) frame_size = 16;

    Label fn_entry_label = buffer.create_label();
    buffer.bind(fn_entry_label);

    // 2. Prologue: stp fp, lr, [sp, #-frame_size]!; mov fp, sp
    if (frame_size <= 504) {
        enc.stp(GPR::FP, GPR::LR, pre_idx(GPR::SP, -frame_size));
    } else {
        if (frame_size <= 4095) {
            enc.sub(GPR::SP, GPR::SP, static_cast<uint32_t>(frame_size));
        } else {
            enc.mov(GPR::X16, static_cast<uint64_t>(frame_size));
            enc.sub(GPR::SP, GPR::SP, GPR::X16);
        }
        enc.stp(GPR::FP, GPR::LR, ptr(GPR::SP, 0));
    }
    enc.mov(GPR::FP, GPR::SP);

    // 4. Create block labels
    std::unordered_map<uint32_t, Label> block_labels;
    for (const auto* bb : fn.blocks()) {
        if (bb) block_labels[bb->id()] = buffer.create_label();
    }

    FunctionStackMap fn_stack_map;
    fn_stack_map.function_name = std::string(fn.name());

    AArch64BaselineEmitter emitter{
        buffer,
        enc,
        target,
        fn,
        slot_map,
        alloca_offsets,
        block_labels,
        fn_stack_map,
        resolver,
        frame_size,
        std::move(lazy)
    };

    // Zero-initialize gcref slots for moving GC safety
    for (int32_t off : gcref_slots) {
        enc.str(GPR::XZR, emitter.slot_off_addr(off));
    }

    // 3. Store incoming parameters into entry block parameter slots
    BasicBlock* entry_bb = fn.entry_block();
    if (entry_bb) {
        size_t gpr_idx = 0;
        size_t fpr_idx = 0;
        size_t stack_bytes = 0;
        bool is_apple = target.is_macos();

        for (size_t i = 0; i < entry_bb->param_count(); ++i) {
            const Value* param = entry_bb->param(i);
            int32_t p_off = slot_map[param];
            bool is_flt = param->type().is_float();
            uint8_t sz = static_cast<uint8_t>(param->type().size_in_bytes());
            if (sz == 0) sz = 8;

            if (is_flt) {
                if (fpr_idx < 8) {
                    FPR src_fpr = static_cast<FPR>(fpr_idx++);
                    if (param->type().kind() == TypeKind::F32) {
                        enc.str_s(src_fpr, emitter.slot_off_addr(p_off));
                    } else {
                        enc.str(src_fpr, emitter.slot_off_addr(p_off));
                    }
                } else {
                    size_t align = is_apple ? sz : 8;
                    stack_bytes = (stack_bytes + align - 1) & ~(align - 1);
                    int32_t caller_off = static_cast<int32_t>(frame_size + stack_bytes);
                    if (param->type().kind() == TypeKind::F32) {
                        enc.ldr_s(FPR::V0, emitter.ensure_accessible_mem(ptr(GPR::FP, caller_off), GPR::X16, 4));
                        enc.str_s(FPR::V0, emitter.slot_off_addr(p_off));
                    } else {
                        enc.ldr(FPR::V0, emitter.ensure_accessible_mem(ptr(GPR::FP, caller_off), GPR::X16, 8));
                        enc.str(FPR::V0, emitter.slot_off_addr(p_off));
                    }
                    stack_bytes += sz;
                }
            } else {
                if (gpr_idx < 8) {
                    GPR src_gpr = static_cast<GPR>(gpr_idx++);
                    if (sz == 4) {
                        enc.str32(src_gpr, emitter.slot_off_addr(p_off));
                    } else {
                        enc.str(src_gpr, emitter.slot_off_addr(p_off));
                    }
                } else {
                    size_t align = is_apple ? sz : 8;
                    stack_bytes = (stack_bytes + align - 1) & ~(align - 1);
                    int32_t caller_off = static_cast<int32_t>(frame_size + stack_bytes);
                    if (sz == 4) {
                        enc.ldr32(GPR::X0, emitter.ensure_accessible_mem(ptr(GPR::FP, caller_off), GPR::X16, 4));
                        enc.str32(GPR::X0, emitter.slot_off_addr(p_off));
                    } else {
                        enc.ldr(GPR::X0, emitter.ensure_accessible_mem(ptr(GPR::FP, caller_off), GPR::X16, 8));
                        enc.str(GPR::X0, emitter.slot_off_addr(p_off));
                    }
                    stack_bytes += sz;
                }
            }
        }
    }

    // Record-invocation hook: the function's TieringFeedback, from the
    // registry of the program this code belongs to, is resolved now and
    // baked in (the registry retires, never frees, feedback objects, and
    // lives as long as the program that owns this code). The hook reaches
    // the program's pipeline through it.
    if (!fn.name().empty()) {
        runtime::TieringRegistry& registry = tiering ? *tiering : runtime::TieringRegistry::instance();
        runtime::TieringFeedback* feedback = &registry.get_feedback(fn.name());
        enc.mov(GPR::X0, reinterpret_cast<uint64_t>(feedback));
        void* hook_ptr = reinterpret_cast<void*>(&brass_tier1_record_invocation_fb);
        enc.mov(GPR::X16, reinterpret_cast<uint64_t>(hook_ptr));
        enc.blr(GPR::X16);
    }

    // 5. Code emission for each basic block
    for (const auto* bb : fn.blocks()) {
        if (!bb) continue;
        buffer.bind(block_labels[bb->id()]);

        for (const auto* inst_ptr : *bb) {
            if (!inst_ptr) continue;
            const Instruction& inst = *inst_ptr;

            if (emit_baseline_aarch64_op(emitter, inst)) {
                continue;
            }

            Opcode op = inst.opcode();
            switch (op) {
                // Calls & Safepoints
                case Opcode::safepoint: {
                    void* sf = reinterpret_cast<void*>(&brass_gc_safepoint);
                    enc.mov(GPR::X16, reinterpret_cast<uint64_t>(sf));
                    enc.blr(GPR::X16);

                    StackMapRecord rec;
                    rec.instruction_offset = static_cast<uint32_t>(buffer.size());
                    rec.frame_size = static_cast<uint32_t>(frame_size);
                    rec.safepoint_id = inst.site_id();
                    for (int32_t off : gcref_slots) {
                        rec.add_root(StackMapRootLocation::frame_slot(16 + off));
                    }
                    fn_stack_map.add_record(std::move(rec));
                    break;
                }
                case Opcode::call:
                case Opcode::call_indirect:
                case Opcode::patchable_call: {
                    bool is_indirect = (op == Opcode::call_indirect);
                    size_t num_args = is_indirect ? (inst.operand_count() > 0 ? inst.operand_count() - 1 : 0) : inst.operand_count();
                    size_t arg_start = is_indirect ? 1 : 0;

                    size_t gpr_idx = 0;
                    size_t fpr_idx = 0;
                    size_t stack_args = 0;

                    for (size_t i = 0; i < num_args; ++i) {
                        const Value* arg = inst.operand(arg_start + i);
                        bool is_flt = arg->type().is_float();
                        if (is_flt) {
                            if (fpr_idx < 8) fpr_idx++;
                            else stack_args++;
                        } else {
                            if (gpr_idx < 8) gpr_idx++;
                            else stack_args++;
                        }
                    }

                    int32_t call_stack_alloc = static_cast<int32_t>(stack_args * 8);
                    call_stack_alloc = (call_stack_alloc + 15) & ~15;

                    if (call_stack_alloc > 0) {
                        enc.sub(GPR::SP, GPR::SP, static_cast<uint32_t>(call_stack_alloc));
                    }

                    gpr_idx = 0;
                    fpr_idx = 0;
                    size_t cur_stack_arg = 0;

                    for (size_t i = 0; i < num_args; ++i) {
                        const Value* arg = inst.operand(arg_start + i);
                        bool is_flt = arg->type().is_float();

                        if (is_flt) {
                            if (fpr_idx < 8) {
                                FPR dst_fpr = static_cast<FPR>(fpr_idx++);
                                if (arg->type().kind() == TypeKind::F32) {
                                    enc.ldr_s(dst_fpr, emitter.slot_addr(arg));
                                } else {
                                    enc.ldr(dst_fpr, emitter.slot_addr(arg));
                                }
                            } else {
                                enc.ldr(FPR::V0, emitter.slot_addr(arg));
                                enc.str(FPR::V0, ptr(GPR::SP, static_cast<int64_t>(cur_stack_arg++ * 8)));
                            }
                        } else {
                            if (gpr_idx < 8) {
                                GPR dst_gpr = static_cast<GPR>(gpr_idx++);
                                enc.ldr(dst_gpr, emitter.slot_addr(arg));
                            } else {
                                enc.ldr(GPR::X0, emitter.slot_addr(arg));
                                enc.str(GPR::X0, ptr(GPR::SP, static_cast<int64_t>(cur_stack_arg++ * 8)));
                            }
                        }
                    }

                    if (is_indirect) {
                        enc.ldr(GPR::X16, emitter.slot_addr(inst.operand(0)));
                        enc.blr(GPR::X16);
                    } else {
                        if (inst.symbol() == fn.name()) {
                            enc.bl(fn_entry_label);
                        } else {
                            void* sym_addr = emitter.resolve_sym(inst.symbol());
                            if (sym_addr) {
                                enc.mov(GPR::X16, reinterpret_cast<uint64_t>(sym_addr));
                                enc.blr(GPR::X16);
                            } else {
                                auto* handle = runtime::FunctionDispatchTable::instance().get_or_create(inst.symbol());
                                enc.mov(GPR::X16, reinterpret_cast<uint64_t>(handle->native_entry_ptr()));
                                enc.ldr(GPR::X16, ptr(GPR::X16, 0));
                                enc.blr(GPR::X16);
                            }
                        }
                    }

                    if (call_stack_alloc > 0) {
                        enc.add(GPR::SP, GPR::SP, static_cast<uint32_t>(call_stack_alloc));
                    }

                    StackMapRecord rec;
                    rec.instruction_offset = static_cast<uint32_t>(buffer.size());
                    rec.frame_size = static_cast<uint32_t>(frame_size);
                    rec.safepoint_id = inst.site_id();
                    for (int32_t off : gcref_slots) {
                        rec.add_root(StackMapRootLocation::frame_slot(16 + off));
                    }
                    fn_stack_map.add_record(std::move(rec));

                    if (inst.produces_value()) {
                        if (inst.type().is_float()) {
                            if (inst.type().kind() == TypeKind::F32) {
                                enc.str_s(FPR::V0, emitter.slot_addr(inst.result()));
                            } else {
                                enc.str(FPR::V0, emitter.slot_addr(inst.result()));
                            }
                        } else if (inst.type().is_i32()) {
                            enc.str32(GPR::X0, emitter.slot_addr(inst.result()));
                        } else {
                            enc.str(GPR::X0, emitter.slot_addr(inst.result()));
                        }
                    }
                    break;
                }

                // Terminators
                case Opcode::br: {
                    emitter.copy_block_args(inst.branch_target());
                    enc.b(block_labels[inst.branch_target().block->id()]);
                    break;
                }
                case Opcode::br_if: {
                    enc.ldr32(GPR::X0, emitter.slot_addr(inst.operand(0)));
                    enc.cmp32(GPR::X0, 0);
                    Label true_edge = buffer.create_label();
                    enc.b(Condition::NE, true_edge);

                    emitter.copy_block_args(inst.false_target());
                    enc.b(block_labels[inst.false_target().block->id()]);

                    buffer.bind(true_edge);
                    emitter.copy_block_args(inst.true_target());
                    enc.b(block_labels[inst.true_target().block->id()]);
                    break;
                }
                case Opcode::switch_: {
                    // A narrow value owns only the low bytes of its 8-byte
                    // slot (str32 on store), so the rest is stale stack.
                    // Zero-extend exactly its width and mask each case to
                    // match, or the dispatch reads garbage.
                    const size_t cond_bytes = inst.operand(0)->type().size_in_bytes();
                    const MemAddress cond_slot = emitter.slot_addr(inst.operand(0));
                    if (cond_bytes == 1) enc.ldrb(GPR::X0, cond_slot);
                    else if (cond_bytes == 2) enc.ldrh(GPR::X0, cond_slot);
                    else if (cond_bytes == 4) enc.ldr32(GPR::X0, cond_slot);
                    else enc.ldr(GPR::X0, cond_slot);
                    const uint64_t case_mask = cond_bytes < 8
                        ? (uint64_t{1} << (cond_bytes * 8)) - 1
                        : ~uint64_t{0};
                    std::vector<Label> case_labels;
                    case_labels.reserve(inst.switch_cases().size());
                    for (const auto& sc : inst.switch_cases()) {
                        Label case_body = buffer.create_label();
                        case_labels.push_back(case_body);
                        enc.mov(GPR::X1, static_cast<uint64_t>(sc.value) & case_mask);
                        enc.cmp(GPR::X0, GPR::X1);
                        enc.b(Condition::EQ, case_body);
                    }

                    emitter.copy_block_args(inst.default_target());
                    enc.b(block_labels[inst.default_target().block->id()]);

                    for (size_t i = 0; i < inst.switch_cases().size(); ++i) {
                        buffer.bind(case_labels[i]);
                        emitter.copy_block_args(inst.switch_cases()[i].target);
                        enc.b(block_labels[inst.switch_cases()[i].target.block->id()]);
                    }
                    break;
                }
                case Opcode::ret: {
                    if (inst.operand_count() > 0 && inst.operand(0) != nullptr) {
                        const Value* rval = inst.operand(0);
                        if (rval->type().is_float()) {
                            if (rval->type().kind() == TypeKind::F32) {
                                enc.ldr_s(FPR::V0, emitter.slot_addr(rval));
                            } else {
                                enc.ldr(FPR::V0, emitter.slot_addr(rval));
                            }
                        } else if (rval->type().is_i32()) {
                            enc.ldr32(GPR::X0, emitter.slot_addr(rval));
                        } else {
                            enc.ldr(GPR::X0, emitter.slot_addr(rval));
                        }
                    }
                    if (frame_size <= 504) {
                        enc.ldp(GPR::FP, GPR::LR, post_idx(GPR::SP, frame_size));
                    } else {
                        enc.ldp(GPR::FP, GPR::LR, ptr(GPR::SP, 0));
                        if (frame_size <= 4095) {
                            enc.add(GPR::SP, GPR::SP, static_cast<uint32_t>(frame_size));
                        } else {
                            enc.mov(GPR::X16, static_cast<uint64_t>(frame_size));
                            enc.add(GPR::SP, GPR::SP, GPR::X16);
                        }
                    }
                    enc.ret();
                    break;
                }
                case Opcode::unreachable: {
                    enc.brk(0);
                    break;
                }
                case Opcode::guard: {
                    enc.ldr32(GPR::X0, emitter.slot_addr(inst.operand(0)));
                    enc.cmp32(GPR::X0, 0);
                    Label cont_lbl = buffer.create_label();
                    enc.b(Condition::NE, cont_lbl);
                    enc.brk(0);
                    buffer.bind(cont_lbl);
                    break;
                }
                case Opcode::resume_point:
                case Opcode::osr_entry: {
                    break;
                }

                default: {
                    enc.brk(0);
                    break;
                }
            }
        }
    }

    // The baseline tier emits in one pass; a short branch that did not reach
    // its label would be left unpatched, so it is an error, not code.
    if (!buffer.relax_requests().empty()) {
        throw std::runtime_error("compile_baseline_aarch64: a conditional branch in " + std::string(fn.name()) +
                                 " is out of range and the baseline tier does not relax branches");
    }

    // Allocate executable memory and copy code
    size_t code_bytes = buffer.size();
    auto mem_block = std::make_shared<JitMemoryBlock>(code_bytes);
    if (!mem_block->is_valid()) {
        throw std::runtime_error("compile_baseline_aarch64: Failed to allocate executable memory for " + std::string(fn.name()));
    }
    std::memcpy(mem_block->data(), buffer.data(), code_bytes);
    if (!mem_block->make_executable_read_only()) {
        throw std::runtime_error("compile_baseline_aarch64: could not make the code of " + std::string(fn.name()) + " executable");
    }

    void* entry_ptr = mem_block->data();
    uintptr_t fn_address = reinterpret_cast<uintptr_t>(entry_ptr);
    fn_stack_map.function_address = fn_address;
    fn_stack_map.code_size = static_cast<uint32_t>(code_bytes);

    BaselineCompiledFunction compiled(
        fn.name(),
        fn.return_type(),
        fn.param_types(),
        mem_block,
        entry_ptr,
        code_bytes,
        std::move(fn_stack_map)
    );
    if (emitter.uses_lazy_stubs) compiled.set_link_keepalive(emitter.lazy);
    return compiled;
}

} // namespace brass::aarch64
