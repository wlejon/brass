#include "baseline_emit_internal.hpp"
#include <brass/codegen/baseline_jit.hpp>
#include <brass/target/aarch64/aarch64_baseline_emit.hpp>
#include <brass/target/x64/x64_encoder.hpp>
#include <brass/target/calling_conv.hpp>
#include <brass/gc/runtime_gc.hpp>
#include <brass/runtime/tiering.hpp>
#include <brass/runtime/type_feedback.hpp>
#include <brass/runtime/code_installer.hpp>
#include <brass/runtime/multi_tier_pipeline.hpp>
#include <brass/core/string_pool.hpp>
#include <brass/il_translator/il_translator.hpp>
#include <brass/pgo/instrument.hpp>
#include <cstdint>
#include <cstring>
#include <cmath>
#include <stdexcept>
#include <algorithm>
#include <iostream>

extern "C" void brass_tier1_record_invocation(const char* fn_name);

namespace brass::codegen {

using namespace brass::x64;

void* X64BaselineEmitter::resolve_sym(std::string_view name) const {
    if (resolver) {
        void* addr = resolver(name);
        if (addr) return addr;
    }
    return nullptr;
}

void X64BaselineEmitter::copy_block_args(const BranchTarget& target_branch) {
    if (!target_branch.block || target_branch.args.empty()) return;
    const auto& params = target_branch.block->params();
    size_t count = std::min(target_branch.args.size(), params.size());
    for (size_t j = 0; j < count; ++j) {
        const Value* a = target_branch.args[j];
        enc.mov(GPR::RAX, slot_addr(a));
        enc.push(GPR::RAX);
    }
    for (size_t j = 0; j < count; ++j) {
        size_t idx = count - 1 - j;
        enc.pop(GPR::RAX);
        enc.mov(slot_addr(params[idx]), GPR::RAX);
    }
}

BaselineCompiledFunction BaselineJitCompiler::compile(const Function& fn, Target target) {
    if (target.is_aarch64()) {
        return aarch64::compile_baseline_aarch64(fn, target, [this](std::string_view name) {
            return resolve_symbol(name);
        });
    }

    CodeBuffer buffer;
    X64Encoder enc(buffer);
    CallingConvention cc = CallingConvention::for_target(target);

    // 1. Assign deterministic stack frame slots [rbp - offset]
    std::unordered_map<const Value*, int32_t> slot_map;
    std::unordered_map<const Instruction*, int32_t> alloca_offsets;
    std::vector<int32_t> gcref_slots;
    bool preserves_r13 = (fn.parent() && fn.parent()->pinned_tls_register());
    int32_t current_offset = preserves_r13 ? 8 : 0;

    auto alloc_slot = [&](const Value* val) -> int32_t {
        if (!val) return 0;
        auto it = slot_map.find(val);
        if (it != slot_map.end()) return it->second;

        int32_t size = 8;
        if (val->type().is_vector()) {
            size = val->type().is_v256() ? 32 : 16;
        }
        current_offset += size;
        slot_map[val] = current_offset;
        if (val->type().is_gcref()) {
            gcref_slots.push_back(current_offset);
        }
        return current_offset;
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
                current_offset += buf_size;
                alloca_offsets[inst] = current_offset;
            }
            if (inst->produces_value()) {
                alloc_slot(inst->result());
            }
        }
    }

    // Align frame size to 16 bytes
    int32_t total_slots = (current_offset + 15) & ~15;
    int32_t frame_size = total_slots;
    if (target.is_windows()) {
        frame_size += 32; // 32-byte shadow space for outgoing calls
    }
    frame_size = (frame_size + 15) & ~15;
    if (frame_size == 0) frame_size = 16;

    auto slot_addr = [&](const Value* val) -> MemAddress {
        int32_t off = slot_map[val];
        return MemAddress::base_disp(GPR::RBP, -off);
    };

    auto slot_off_addr = [&](int32_t off) -> MemAddress {
        return MemAddress::base_disp(GPR::RBP, -off);
    };

    Label fn_entry_label = buffer.create_label();
    buffer.bind(fn_entry_label);

    // 2. Prologue: push rbp; mov rbp, rsp; sub rsp, frame_size
    enc.push(GPR::RBP);
    enc.mov(GPR::RBP, GPR::RSP);
    enc.sub(GPR::RSP, frame_size);

    if (preserves_r13) {
        enc.mov(MemAddress::base_disp(GPR::RBP, -8), GPR::R13);
    }

    // Zero-initialize gcref slots for moving GC safety
    if (!gcref_slots.empty()) {
        enc.xor_(GPR::RAX, GPR::RAX);
        for (int32_t off : gcref_slots) {
            enc.mov(slot_off_addr(off), GPR::RAX);
        }
    }

    // 3. Store incoming parameters from ABI registers / stack into entry block parameter slots
    BasicBlock* entry_bb = fn.entry_block();
    if (entry_bb) {
        size_t gpr_idx = 0;
        size_t xmm_idx = 0;
        size_t stack_idx = 0;

        for (size_t i = 0; i < entry_bb->param_count(); ++i) {
            const Value* param = entry_bb->param(i);
            int32_t p_off = slot_map[param];
            bool is_flt = param->type().is_float();

            if (target.is_windows()) {
                if (i == 0) {
                    if (is_flt) enc.movsd(slot_off_addr(p_off), XMM::XMM0);
                    else enc.mov(slot_off_addr(p_off), GPR::RCX);
                } else if (i == 1) {
                    if (is_flt) enc.movsd(slot_off_addr(p_off), XMM::XMM1);
                    else enc.mov(slot_off_addr(p_off), GPR::RDX);
                } else if (i == 2) {
                    if (is_flt) enc.movsd(slot_off_addr(p_off), XMM::XMM2);
                    else enc.mov(slot_off_addr(p_off), GPR::R8);
                } else if (i == 3) {
                    if (is_flt) enc.movsd(slot_off_addr(p_off), XMM::XMM3);
                    else enc.mov(slot_off_addr(p_off), GPR::R9);
                } else {
                    int32_t caller_stack_off = static_cast<int32_t>(16 + 32 + (i - 4) * 8);
                    if (is_flt) {
                        enc.movsd(XMM::XMM0, MemAddress::base_disp(GPR::RBP, caller_stack_off));
                        enc.movsd(slot_off_addr(p_off), XMM::XMM0);
                    } else {
                        enc.mov(GPR::RAX, MemAddress::base_disp(GPR::RBP, caller_stack_off));
                        enc.mov(slot_off_addr(p_off), GPR::RAX);
                    }
                }
            } else {
                // SysV64
                if (is_flt) {
                    if (xmm_idx < cc.arg_xmms().size()) {
                        XMM src_xmm = cc.arg_xmms()[xmm_idx++];
                        if (param->type().kind() == TypeKind::F32) {
                            enc.movss(slot_off_addr(p_off), src_xmm);
                        } else {
                            enc.movsd(slot_off_addr(p_off), src_xmm);
                        }
                    } else {
                        int32_t caller_stack_off = static_cast<int32_t>(16 + stack_idx++ * 8);
                        enc.movsd(XMM::XMM0, MemAddress::base_disp(GPR::RBP, caller_stack_off));
                        enc.movsd(slot_off_addr(p_off), XMM::XMM0);
                    }
                } else {
                    if (gpr_idx < cc.arg_gprs().size()) {
                        GPR src_gpr = cc.arg_gprs()[gpr_idx++];
                        enc.mov(slot_off_addr(p_off), src_gpr);
                    } else {
                        int32_t caller_stack_off = static_cast<int32_t>(16 + stack_idx++ * 8);
                        enc.mov(GPR::RAX, MemAddress::base_disp(GPR::RBP, caller_stack_off));
                        enc.mov(slot_off_addr(p_off), GPR::RAX);
                    }
                }
            }
        }
    }

    // Record invocation hook
    if (!fn.name().empty()) {
        static std::unordered_map<std::string, std::string> name_cache;
        std::string fn_name_copy(fn.name());
        auto it_name = name_cache.try_emplace(fn_name_copy, fn_name_copy);
        const char* name_cstr = it_name.first->second.c_str();

        if (target.is_windows()) {
            enc.movabs(GPR::RCX, reinterpret_cast<uint64_t>(name_cstr));
        } else {
            enc.movabs(GPR::RDI, reinterpret_cast<uint64_t>(name_cstr));
        }
        void* hook_ptr = reinterpret_cast<void*>(&brass_tier1_record_invocation);
        enc.movabs(GPR::R11, reinterpret_cast<uint64_t>(hook_ptr));
        enc.call(GPR::R11);
    }

    // 4. Create block labels
    std::unordered_map<uint32_t, Label> block_labels;
    for (const auto* bb : fn.blocks()) {
        if (bb) block_labels[bb->id()] = buffer.create_label();
    }

    FunctionStackMap fn_stack_map;
    fn_stack_map.function_name = std::string(fn.name());

    X64BaselineEmitter emitter{
        buffer,
        enc,
        target,
        fn,
        slot_map,
        alloca_offsets,
        block_labels,
        fn_stack_map,
        [this](std::string_view name) { return resolve_symbol(name); },
        frame_size
    };

    // 5. Code emission for each basic block
    for (const auto* bb : fn.blocks()) {
        if (!bb) continue;
        buffer.bind(block_labels[bb->id()]);

        for (const auto* inst_ptr : *bb) {
            if (!inst_ptr) continue;
            const Instruction& inst = *inst_ptr;

            if (emit_baseline_x64_op(emitter, inst)) {
                continue;
            }

            Opcode op = inst.opcode();
            switch (op) {
                // Calls & Safepoints
                case Opcode::safepoint: {
                    void* sf = reinterpret_cast<void*>(&brass_gc_safepoint);
                    enc.movabs(GPR::R11, reinterpret_cast<uint64_t>(sf));
                    enc.call(GPR::R11);

                    StackMapRecord rec;
                    rec.instruction_offset = static_cast<uint32_t>(buffer.size());
                    rec.frame_size = static_cast<uint32_t>(frame_size);
                    rec.safepoint_id = inst.site_id();
                    for (int32_t off : gcref_slots) {
                        rec.add_root(StackMapRootLocation::frame_slot(-off));
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

                    // Compute stack args needed
                    size_t gpr_idx = 0;
                    size_t xmm_idx = 0;
                    size_t stack_args = 0;

                    for (size_t i = 0; i < num_args; ++i) {
                        const Value* arg = inst.operand(arg_start + i);
                        bool is_flt = arg->type().is_float();
                        if (target.is_windows()) {
                            if (i >= 4) stack_args++;
                        } else {
                            if (is_flt) {
                                if (xmm_idx < cc.arg_xmms().size()) xmm_idx++;
                                else stack_args++;
                            } else {
                                if (gpr_idx < cc.arg_gprs().size()) gpr_idx++;
                                else stack_args++;
                            }
                        }
                    }

                    int32_t call_stack_alloc = 0;
                    if (target.is_windows()) {
                        call_stack_alloc = static_cast<int32_t>(32 + stack_args * 8);
                    } else {
                        call_stack_alloc = static_cast<int32_t>(stack_args * 8);
                    }
                    call_stack_alloc = (call_stack_alloc + 15) & ~15;

                    if (call_stack_alloc > 0) {
                        enc.sub(GPR::RSP, call_stack_alloc);
                    }

                    // Populate arguments
                    gpr_idx = 0;
                    xmm_idx = 0;
                    size_t cur_stack_arg = 0;

                    for (size_t i = 0; i < num_args; ++i) {
                        const Value* arg = inst.operand(arg_start + i);
                        bool is_flt = arg->type().is_float();

                        if (target.is_windows()) {
                            if (i == 0) {
                                if (is_flt) enc.movsd(XMM::XMM0, slot_addr(arg));
                                else enc.mov(GPR::RCX, slot_addr(arg));
                            } else if (i == 1) {
                                if (is_flt) enc.movsd(XMM::XMM1, slot_addr(arg));
                                else enc.mov(GPR::RDX, slot_addr(arg));
                            } else if (i == 2) {
                                if (is_flt) enc.movsd(XMM::XMM2, slot_addr(arg));
                                else enc.mov(GPR::R8, slot_addr(arg));
                            } else if (i == 3) {
                                if (is_flt) enc.movsd(XMM::XMM3, slot_addr(arg));
                                else enc.mov(GPR::R9, slot_addr(arg));
                            } else {
                                enc.mov(GPR::RAX, slot_addr(arg));
                                enc.mov(MemAddress::base_disp(GPR::RSP, static_cast<int32_t>(32 + (i - 4) * 8)), GPR::RAX);
                            }
                        } else {
                            if (is_flt) {
                                if (xmm_idx < cc.arg_xmms().size()) {
                                    XMM dst_xmm = cc.arg_xmms()[xmm_idx++];
                                    enc.movsd(dst_xmm, slot_addr(arg));
                                } else {
                                    enc.mov(GPR::RAX, slot_addr(arg));
                                    enc.mov(MemAddress::base_disp(GPR::RSP, static_cast<int32_t>(cur_stack_arg++ * 8)), GPR::RAX);
                                }
                            } else {
                                if (gpr_idx < cc.arg_gprs().size()) {
                                    GPR dst_gpr = cc.arg_gprs()[gpr_idx++];
                                    enc.mov(dst_gpr, slot_addr(arg));
                                } else {
                                    enc.mov(GPR::RAX, slot_addr(arg));
                                    enc.mov(MemAddress::base_disp(GPR::RSP, static_cast<int32_t>(cur_stack_arg++ * 8)), GPR::RAX);
                                }
                            }
                        }
                    }

                    // Execute Call
                    if (is_indirect) {
                        enc.mov(GPR::R11, slot_addr(inst.operand(0)));
                        enc.call(GPR::R11);
                    } else {
                        if (inst.symbol() == fn.name()) {
                            enc.call(fn_entry_label);
                        } else {
                            void* sym_addr = resolve_symbol(inst.symbol());
                            if (sym_addr) {
                                enc.movabs(GPR::R11, reinterpret_cast<uint64_t>(sym_addr));
                                enc.call(GPR::R11);
                            } else {
                                auto* handle = runtime::FunctionDispatchTable::instance().get_or_create(inst.symbol());
                                enc.movabs(GPR::R11, reinterpret_cast<uint64_t>(handle->native_entry_ptr()));
                                enc.mov(GPR::R11, MemAddress::base_disp(GPR::R11, 0));
                                enc.call(GPR::R11);
                            }
                        }
                    }

                    if (call_stack_alloc > 0) {
                        enc.add(GPR::RSP, call_stack_alloc);
                    }

                    // Record Stack Map
                    StackMapRecord rec;
                    rec.instruction_offset = static_cast<uint32_t>(buffer.size());
                    rec.frame_size = static_cast<uint32_t>(frame_size);
                    rec.safepoint_id = inst.site_id();
                    for (int32_t off : gcref_slots) {
                        rec.add_root(StackMapRootLocation::frame_slot(-off));
                    }
                    fn_stack_map.add_record(std::move(rec));

                    // Store return value
                    if (inst.produces_value()) {
                        if (inst.type().is_float()) {
                            if (inst.type().kind() == TypeKind::F32) {
                                enc.movss(slot_addr(inst.result()), XMM::XMM0);
                            } else {
                                enc.movsd(slot_addr(inst.result()), XMM::XMM0);
                            }
                        } else if (inst.type().is_i32()) {
                            enc.mov32(slot_addr(inst.result()), GPR::RAX);
                        } else {
                            enc.mov(slot_addr(inst.result()), GPR::RAX);
                        }
                    }
                    break;
                }

                // Terminators
                case Opcode::br: {
                    emitter.copy_block_args(inst.branch_target());
                    enc.jmp(block_labels[inst.branch_target().block->id()]);
                    break;
                }
                case Opcode::br_if: {
                    enc.mov32(GPR::RAX, slot_addr(inst.operand(0)));
                    enc.test32(GPR::RAX, GPR::RAX);
                    Label true_edge = buffer.create_label();
                    enc.jne(true_edge);

                    // False edge
                    emitter.copy_block_args(inst.false_target());
                    enc.jmp(block_labels[inst.false_target().block->id()]);

                    // True edge
                    buffer.bind(true_edge);
                    emitter.copy_block_args(inst.true_target());
                    enc.jmp(block_labels[inst.true_target().block->id()]);
                    break;
                }
                case Opcode::switch_: {
                    // A narrow value owns only the low bytes of its 8-byte
                    // slot (loads and stores of it are mov32/mov8), so the
                    // rest is stale stack. Zero-extend exactly its width and
                    // compare at that width, or the dispatch reads garbage.
                    const Type cond_ty = inst.operand(0)->type();
                    const size_t cond_bytes = cond_ty.size_in_bytes();
                    if (cond_bytes == 1) enc.movzx8(GPR::RAX, slot_addr(inst.operand(0)));
                    else if (cond_bytes == 2) enc.movzx16(GPR::RAX, slot_addr(inst.operand(0)));
                    else if (cond_bytes == 4) enc.mov32(GPR::RAX, slot_addr(inst.operand(0)));
                    else enc.mov(GPR::RAX, slot_addr(inst.operand(0)));
                    std::vector<Label> case_labels;
                    case_labels.reserve(inst.switch_cases().size());
                    for (const auto& sc : inst.switch_cases()) {
                        Label case_body = buffer.create_label();
                        case_labels.push_back(case_body);
                        if (cond_bytes < 8) {
                            // Mask the case to the condition's width so a
                            // negative case matches its zero-extended bits.
                            const uint64_t mask = (uint64_t{1} << (cond_bytes * 8)) - 1;
                            enc.cmp32(GPR::RAX, static_cast<int32_t>(static_cast<uint32_t>(
                                static_cast<uint64_t>(sc.value) & mask)));
                        } else if (sc.value >= INT32_MIN && sc.value <= INT32_MAX) {
                            enc.cmp(GPR::RAX, static_cast<int32_t>(sc.value));
                        } else {
                            // cmp's imm32 is sign-extended; a wider case
                            // needs a register.
                            enc.movabs(GPR::RCX, static_cast<uint64_t>(sc.value));
                            enc.cmp(GPR::RAX, GPR::RCX);
                        }
                        enc.je(case_body);
                    }
                    // Default
                    emitter.copy_block_args(inst.default_target());
                    enc.jmp(block_labels[inst.default_target().block->id()]);

                    // Cases
                    for (size_t i = 0; i < inst.switch_cases().size(); ++i) {
                        buffer.bind(case_labels[i]);
                        emitter.copy_block_args(inst.switch_cases()[i].target);
                        enc.jmp(block_labels[inst.switch_cases()[i].target.block->id()]);
                    }
                    break;
                }
                case Opcode::ret: {
                    if (inst.operand_count() > 0 && inst.operand(0) != nullptr) {
                        const Value* rval = inst.operand(0);
                        if (rval->type().is_float()) {
                            if (rval->type().kind() == TypeKind::F32) {
                                enc.movss(XMM::XMM0, slot_addr(rval));
                            } else {
                                enc.movsd(XMM::XMM0, slot_addr(rval));
                            }
                        } else if (rval->type().is_i32()) {
                            enc.mov32(GPR::RAX, slot_addr(rval));
                        } else {
                            enc.mov(GPR::RAX, slot_addr(rval));
                        }
                    }
                    if (preserves_r13) {
                        enc.mov(GPR::R13, MemAddress::base_disp(GPR::RBP, -8));
                    }
                    enc.mov(GPR::RSP, GPR::RBP);
                    enc.pop(GPR::RBP);
                    enc.ret();
                    break;
                }
                case Opcode::unreachable: {
                    enc.ud2();
                    break;
                }
                case Opcode::guard: {
                    enc.mov32(GPR::RAX, slot_addr(inst.operand(0)));
                    enc.test32(GPR::RAX, GPR::RAX);
                    Label cont_lbl = buffer.create_label();
                    enc.jne(cont_lbl);
                    enc.ud2();
                    buffer.bind(cont_lbl);
                    break;
                }
                case Opcode::resume_point:
                case Opcode::osr_entry: {
                    // No-op in baseline linear JIT
                    break;
                }

                default: {
                    // Unsupported opcode in baseline fallback
                    enc.ud2();
                    break;
                }
            }
        }
    }

    // Allocate executable memory and copy code
    size_t code_bytes = buffer.size();
    auto mem_block = std::make_shared<JitMemoryBlock>(code_bytes);
    if (!mem_block->is_valid()) {
        throw std::runtime_error("BaselineJitCompiler: Failed to allocate executable memory for " + std::string(fn.name()));
    }
    std::memcpy(mem_block->data(), buffer.data(), code_bytes);
    if (!mem_block->make_executable_read_only()) {
        throw std::runtime_error("BaselineJitCompiler: could not make the code of " + std::string(fn.name()) + " executable");
    }

    void* entry_ptr = mem_block->data();
    uintptr_t fn_address = reinterpret_cast<uintptr_t>(entry_ptr);
    fn_stack_map.function_address = fn_address;
    fn_stack_map.code_size = static_cast<uint32_t>(code_bytes);

    return BaselineCompiledFunction(
        fn.name(),
        fn.return_type(),
        fn.param_types(),
        mem_block,
        entry_ptr,
        code_bytes,
        std::move(fn_stack_map)
    );
}

} // namespace brass::codegen
