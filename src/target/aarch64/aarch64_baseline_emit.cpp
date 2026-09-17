#include <brass/target/aarch64/aarch64_baseline_emit.hpp>
#include <brass/target/aarch64/aarch64_encoder.hpp>
#include <brass/target/aarch64/code_buffer.hpp>
#include <brass/target/calling_conv.hpp>
#include <brass/gc/runtime_gc.hpp>
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

extern "C" void brass_tier1_record_invocation(const char* fn_name);

namespace brass::aarch64 {

using namespace brass::codegen;

codegen::BaselineCompiledFunction compile_baseline_aarch64(
    const Function& fn,
    Target target,
    BaselineSymbolResolver resolver
) {
    (void)target;
    CodeBuffer buffer;
    AArch64Encoder enc(buffer);

    // 1. Assign deterministic stack frame slots relative to FP: [fp, 16 + offset]
    std::unordered_map<const Value*, int32_t> slot_map;
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
        if (val->type().is_gcref()) {
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

    auto ensure_accessible_mem = [&](const MemAddress& mem, GPR scratch = GPR::X16) -> MemAddress {
        if (mem.mode != AddrMode::Offset) return mem;
        int64_t off = mem.offset;
        if (off >= -256 && off <= 16380 && (off >= 0 ? (off % 8 == 0) : true)) {
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
    };

    auto slot_addr = [&](const Value* val) -> MemAddress {
        int32_t off = 16 + slot_map[val];
        return ensure_accessible_mem(ptr(GPR::FP, off));
    };

    auto slot_off_addr = [&](int32_t off) -> MemAddress {
        return ensure_accessible_mem(ptr(GPR::FP, 16 + off));
    };

    auto resolve_sym = [&](std::string_view name) -> void* {
        if (resolver) {
            void* ptr_val = resolver(name);
            if (ptr_val) return ptr_val;
        }
        auto* handle = runtime::FunctionDispatchTable::instance().find(name);
        if (handle && handle->native_entry()) {
            return handle->native_entry();
        }
        return nullptr;
    };

    Label fn_entry_label = buffer.create_label();
    buffer.bind(fn_entry_label);

    // 2. Prologue: stp fp, lr, [sp, #-frame_size]!; mov fp, sp
    if (frame_size <= 512) {
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

    // Zero-initialize gcref slots for moving GC safety
    for (int32_t off : gcref_slots) {
        enc.str(GPR::XZR, slot_off_addr(off));
    }

    // 3. Store incoming parameters into entry block parameter slots
    BasicBlock* entry_bb = fn.entry_block();
    if (entry_bb) {
        size_t gpr_idx = 0;
        size_t fpr_idx = 0;
        size_t stack_idx = 0;

        for (size_t i = 0; i < entry_bb->param_count(); ++i) {
            const Value* param = entry_bb->param(i);
            int32_t p_off = slot_map[param];
            bool is_flt = param->type().is_float();

            if (is_flt) {
                if (fpr_idx < 8) {
                    FPR src_fpr = static_cast<FPR>(fpr_idx++);
                    if (param->type().kind() == TypeKind::F32) {
                        enc.str_s(src_fpr, slot_off_addr(p_off));
                    } else {
                        enc.str(src_fpr, slot_off_addr(p_off));
                    }
                } else {
                    int32_t caller_stack_off = frame_size + static_cast<int32_t>(stack_idx++ * 8);
                    if (param->type().kind() == TypeKind::F32) {
                        enc.ldr_s(FPR::V0, ensure_accessible_mem(ptr(GPR::FP, caller_stack_off)));
                        enc.str_s(FPR::V0, slot_off_addr(p_off));
                    } else {
                        enc.ldr(FPR::V0, ensure_accessible_mem(ptr(GPR::FP, caller_stack_off)));
                        enc.str(FPR::V0, slot_off_addr(p_off));
                    }
                }
            } else {
                if (gpr_idx < 8) {
                    GPR src_gpr = static_cast<GPR>(gpr_idx++);
                    if (param->type().is_i32()) {
                        enc.str32(src_gpr, slot_off_addr(p_off));
                    } else {
                        enc.str(src_gpr, slot_off_addr(p_off));
                    }
                } else {
                    int32_t caller_stack_off = frame_size + static_cast<int32_t>(stack_idx++ * 8);
                    if (param->type().is_i32()) {
                        enc.ldr32(GPR::X0, ensure_accessible_mem(ptr(GPR::FP, caller_stack_off)));
                        enc.str32(GPR::X0, slot_off_addr(p_off));
                    } else {
                        enc.ldr(GPR::X0, ensure_accessible_mem(ptr(GPR::FP, caller_stack_off)));
                        enc.str(GPR::X0, slot_off_addr(p_off));
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

        enc.mov(GPR::X0, reinterpret_cast<uint64_t>(name_cstr));
        void* hook_ptr = reinterpret_cast<void*>(&brass_tier1_record_invocation);
        enc.mov(GPR::X16, reinterpret_cast<uint64_t>(hook_ptr));
        enc.blr(GPR::X16);
    }

    // 4. Create block labels
    std::unordered_map<uint32_t, Label> block_labels;
    for (const auto* bb : fn.blocks()) {
        if (bb) block_labels[bb->id()] = buffer.create_label();
    }

    FunctionStackMap fn_stack_map;
    fn_stack_map.function_name = std::string(fn.name());

    auto copy_block_args = [&](const BranchTarget& target_branch) {
        if (!target_branch.block || target_branch.args.empty()) return;
        const auto& params = target_branch.block->params();
        size_t count = std::min(target_branch.args.size(), params.size());
        if (count == 0) return;

        uint32_t temp_alloc = static_cast<uint32_t>((count * 8 + 15) & ~size_t(15));
        enc.sub(GPR::SP, GPR::SP, temp_alloc);

        for (size_t j = 0; j < count; ++j) {
            const Value* a = target_branch.args[j];
            enc.ldr(GPR::X0, slot_addr(a));
            enc.str(GPR::X0, ptr(GPR::SP, static_cast<int64_t>(j * 8)));
        }
        for (size_t j = 0; j < count; ++j) {
            enc.ldr(GPR::X0, ptr(GPR::SP, static_cast<int64_t>(j * 8)));
            enc.str(GPR::X0, slot_addr(params[j]));
        }

        enc.add(GPR::SP, GPR::SP, temp_alloc);
    };

    // 5. Code emission for each basic block
    for (const auto* bb : fn.blocks()) {
        if (!bb) continue;
        buffer.bind(block_labels[bb->id()]);

        for (const auto* inst_ptr : *bb) {
            if (!inst_ptr) continue;
            const Instruction& inst = *inst_ptr;
            Opcode op = inst.opcode();

            switch (op) {
                // Constants
                case Opcode::iconst_i32:
                case Opcode::patchable_const_i32: {
                    enc.mov32(GPR::X0, static_cast<uint32_t>(inst.imm_i32()));
                    enc.str32(GPR::X0, slot_addr(inst.result()));
                    break;
                }
                case Opcode::iconst_i64:
                case Opcode::patchable_const_i64: {
                    enc.mov(GPR::X0, static_cast<uint64_t>(inst.imm_i64()));
                    enc.str(GPR::X0, slot_addr(inst.result()));
                    break;
                }
                case Opcode::fconst_f64: {
                    double fv = inst.imm_f64();
                    uint64_t bits;
                    std::memcpy(&bits, &fv, 8);
                    enc.mov(GPR::X0, bits);
                    enc.str(GPR::X0, slot_addr(inst.result()));
                    break;
                }
                case Opcode::func_addr: {
                    void* addr = resolve_sym(inst.symbol());
                    enc.mov(GPR::X0, reinterpret_cast<uint64_t>(addr));
                    enc.str(GPR::X0, slot_addr(inst.result()));
                    break;
                }

                // Conversions
                case Opcode::sext_i64: {
                    enc.ldrsw(GPR::X0, slot_addr(inst.operand(0)));
                    enc.str(GPR::X0, slot_addr(inst.result()));
                    break;
                }
                case Opcode::zext_i64: {
                    enc.ldr32(GPR::X0, slot_addr(inst.operand(0)));
                    enc.str(GPR::X0, slot_addr(inst.result()));
                    break;
                }
                case Opcode::trunc_i32: {
                    enc.ldr32(GPR::X0, slot_addr(inst.operand(0)));
                    enc.str32(GPR::X0, slot_addr(inst.result()));
                    break;
                }
                case Opcode::fptosi_i32: {
                    enc.ldr(FPR::V0, slot_addr(inst.operand(0)));
                    enc.fcvtzs_d32(GPR::X0, FPR::V0);
                    enc.str32(GPR::X0, slot_addr(inst.result()));
                    break;
                }
                case Opcode::fptosi_i64: {
                    enc.ldr(FPR::V0, slot_addr(inst.operand(0)));
                    enc.fcvtzs_d(GPR::X0, FPR::V0);
                    enc.str(GPR::X0, slot_addr(inst.result()));
                    break;
                }
                case Opcode::sitofp_f64_i32: {
                    enc.ldr32(GPR::X0, slot_addr(inst.operand(0)));
                    enc.scvtf_d32(FPR::V0, GPR::X0);
                    enc.str(FPR::V0, slot_addr(inst.result()));
                    break;
                }
                case Opcode::sitofp_f64_i64: {
                    enc.ldr(GPR::X0, slot_addr(inst.operand(0)));
                    enc.scvtf_d(FPR::V0, GPR::X0);
                    enc.str(FPR::V0, slot_addr(inst.result()));
                    break;
                }
                case Opcode::bitcast_i64_f64:
                case Opcode::bitcast_f64_i64: {
                    enc.ldr(GPR::X0, slot_addr(inst.operand(0)));
                    enc.str(GPR::X0, slot_addr(inst.result()));
                    break;
                }

                // Arithmetic & Logic
                case Opcode::add: {
                    if (inst.type().is_float()) {
                        enc.ldr(FPR::V0, slot_addr(inst.operand(0)));
                        enc.ldr(FPR::V1, slot_addr(inst.operand(1)));
                        enc.fadd(FPR::V0, FPR::V0, FPR::V1);
                        enc.str(FPR::V0, slot_addr(inst.result()));
                    } else if (inst.type().is_i32()) {
                        enc.ldr32(GPR::X0, slot_addr(inst.operand(0)));
                        enc.ldr32(GPR::X1, slot_addr(inst.operand(1)));
                        enc.add32(GPR::X0, GPR::X0, GPR::X1);
                        enc.str32(GPR::X0, slot_addr(inst.result()));
                    } else {
                        enc.ldr(GPR::X0, slot_addr(inst.operand(0)));
                        enc.ldr(GPR::X1, slot_addr(inst.operand(1)));
                        enc.add(GPR::X0, GPR::X0, GPR::X1);
                        enc.str(GPR::X0, slot_addr(inst.result()));
                    }
                    break;
                }
                case Opcode::sub: {
                    if (inst.type().is_float()) {
                        enc.ldr(FPR::V0, slot_addr(inst.operand(0)));
                        enc.ldr(FPR::V1, slot_addr(inst.operand(1)));
                        enc.fsub(FPR::V0, FPR::V0, FPR::V1);
                        enc.str(FPR::V0, slot_addr(inst.result()));
                    } else if (inst.type().is_i32()) {
                        enc.ldr32(GPR::X0, slot_addr(inst.operand(0)));
                        enc.ldr32(GPR::X1, slot_addr(inst.operand(1)));
                        enc.sub32(GPR::X0, GPR::X0, GPR::X1);
                        enc.str32(GPR::X0, slot_addr(inst.result()));
                    } else {
                        enc.ldr(GPR::X0, slot_addr(inst.operand(0)));
                        enc.ldr(GPR::X1, slot_addr(inst.operand(1)));
                        enc.sub(GPR::X0, GPR::X0, GPR::X1);
                        enc.str(GPR::X0, slot_addr(inst.result()));
                    }
                    break;
                }
                case Opcode::mul: {
                    if (inst.type().is_float()) {
                        enc.ldr(FPR::V0, slot_addr(inst.operand(0)));
                        enc.ldr(FPR::V1, slot_addr(inst.operand(1)));
                        enc.fmul(FPR::V0, FPR::V0, FPR::V1);
                        enc.str(FPR::V0, slot_addr(inst.result()));
                    } else if (inst.type().is_i32()) {
                        enc.ldr32(GPR::X0, slot_addr(inst.operand(0)));
                        enc.ldr32(GPR::X1, slot_addr(inst.operand(1)));
                        enc.mul32(GPR::X0, GPR::X0, GPR::X1);
                        enc.str32(GPR::X0, slot_addr(inst.result()));
                    } else {
                        enc.ldr(GPR::X0, slot_addr(inst.operand(0)));
                        enc.ldr(GPR::X1, slot_addr(inst.operand(1)));
                        enc.mul(GPR::X0, GPR::X0, GPR::X1);
                        enc.str(GPR::X0, slot_addr(inst.result()));
                    }
                    break;
                }
                case Opcode::sdiv: {
                    if (inst.type().is_float()) {
                        enc.ldr(FPR::V0, slot_addr(inst.operand(0)));
                        enc.ldr(FPR::V1, slot_addr(inst.operand(1)));
                        enc.fdiv(FPR::V0, FPR::V0, FPR::V1);
                        enc.str(FPR::V0, slot_addr(inst.result()));
                    } else if (inst.type().is_i32()) {
                        enc.ldr32(GPR::X0, slot_addr(inst.operand(0)));
                        enc.ldr32(GPR::X1, slot_addr(inst.operand(1)));
                        enc.sdiv32(GPR::X0, GPR::X0, GPR::X1);
                        enc.str32(GPR::X0, slot_addr(inst.result()));
                    } else {
                        enc.ldr(GPR::X0, slot_addr(inst.operand(0)));
                        enc.ldr(GPR::X1, slot_addr(inst.operand(1)));
                        enc.sdiv(GPR::X0, GPR::X0, GPR::X1);
                        enc.str(GPR::X0, slot_addr(inst.result()));
                    }
                    break;
                }
                case Opcode::smod: {
                    if (inst.type().is_float()) {
                        enc.ldr(FPR::V0, slot_addr(inst.operand(0)));
                        enc.ldr(FPR::V1, slot_addr(inst.operand(1)));
                        void* fmod_ptr = reinterpret_cast<void*>(static_cast<double(*)(double, double)>(&std::fmod));
                        enc.mov(GPR::X16, reinterpret_cast<uint64_t>(fmod_ptr));
                        enc.blr(GPR::X16);
                        enc.str(FPR::V0, slot_addr(inst.result()));
                    } else if (inst.type().is_i32()) {
                        enc.ldr32(GPR::X0, slot_addr(inst.operand(0)));
                        enc.ldr32(GPR::X1, slot_addr(inst.operand(1)));
                        enc.sdiv32(GPR::X2, GPR::X0, GPR::X1);
                        enc.msub32(GPR::X0, GPR::X2, GPR::X1, GPR::X0);
                        enc.str32(GPR::X0, slot_addr(inst.result()));
                    } else {
                        enc.ldr(GPR::X0, slot_addr(inst.operand(0)));
                        enc.ldr(GPR::X1, slot_addr(inst.operand(1)));
                        enc.sdiv(GPR::X2, GPR::X0, GPR::X1);
                        enc.msub(GPR::X0, GPR::X2, GPR::X1, GPR::X0);
                        enc.str(GPR::X0, slot_addr(inst.result()));
                    }
                    break;
                }
                case Opcode::udiv: {
                    if (inst.type().is_i32()) {
                        enc.ldr32(GPR::X0, slot_addr(inst.operand(0)));
                        enc.ldr32(GPR::X1, slot_addr(inst.operand(1)));
                        enc.udiv32(GPR::X0, GPR::X0, GPR::X1);
                        enc.str32(GPR::X0, slot_addr(inst.result()));
                    } else {
                        enc.ldr(GPR::X0, slot_addr(inst.operand(0)));
                        enc.ldr(GPR::X1, slot_addr(inst.operand(1)));
                        enc.udiv(GPR::X0, GPR::X0, GPR::X1);
                        enc.str(GPR::X0, slot_addr(inst.result()));
                    }
                    break;
                }
                case Opcode::umod: {
                    if (inst.type().is_i32()) {
                        enc.ldr32(GPR::X0, slot_addr(inst.operand(0)));
                        enc.ldr32(GPR::X1, slot_addr(inst.operand(1)));
                        enc.udiv32(GPR::X2, GPR::X0, GPR::X1);
                        enc.msub32(GPR::X0, GPR::X2, GPR::X1, GPR::X0);
                        enc.str32(GPR::X0, slot_addr(inst.result()));
                    } else {
                        enc.ldr(GPR::X0, slot_addr(inst.operand(0)));
                        enc.ldr(GPR::X1, slot_addr(inst.operand(1)));
                        enc.udiv(GPR::X2, GPR::X0, GPR::X1);
                        enc.msub(GPR::X0, GPR::X2, GPR::X1, GPR::X0);
                        enc.str(GPR::X0, slot_addr(inst.result()));
                    }
                    break;
                }
                case Opcode::neg: {
                    if (inst.type().is_float()) {
                        enc.ldr(FPR::V0, slot_addr(inst.operand(0)));
                        enc.fneg(FPR::V0, FPR::V0);
                        enc.str(FPR::V0, slot_addr(inst.result()));
                    } else if (inst.type().is_i32()) {
                        enc.ldr32(GPR::X0, slot_addr(inst.operand(0)));
                        enc.neg32(GPR::X0, GPR::X0);
                        enc.str32(GPR::X0, slot_addr(inst.result()));
                    } else {
                        enc.ldr(GPR::X0, slot_addr(inst.operand(0)));
                        enc.neg(GPR::X0, GPR::X0);
                        enc.str(GPR::X0, slot_addr(inst.result()));
                    }
                    break;
                }
                case Opcode::and_: {
                    if (inst.type().is_i32()) {
                        enc.ldr32(GPR::X0, slot_addr(inst.operand(0)));
                        enc.ldr32(GPR::X1, slot_addr(inst.operand(1)));
                        enc.and32(GPR::X0, GPR::X0, GPR::X1);
                        enc.str32(GPR::X0, slot_addr(inst.result()));
                    } else {
                        enc.ldr(GPR::X0, slot_addr(inst.operand(0)));
                        enc.ldr(GPR::X1, slot_addr(inst.operand(1)));
                        enc.and_(GPR::X0, GPR::X0, GPR::X1);
                        enc.str(GPR::X0, slot_addr(inst.result()));
                    }
                    break;
                }
                case Opcode::or_: {
                    if (inst.type().is_i32()) {
                        enc.ldr32(GPR::X0, slot_addr(inst.operand(0)));
                        enc.ldr32(GPR::X1, slot_addr(inst.operand(1)));
                        enc.orr32(GPR::X0, GPR::X0, GPR::X1);
                        enc.str32(GPR::X0, slot_addr(inst.result()));
                    } else {
                        enc.ldr(GPR::X0, slot_addr(inst.operand(0)));
                        enc.ldr(GPR::X1, slot_addr(inst.operand(1)));
                        enc.orr(GPR::X0, GPR::X0, GPR::X1);
                        enc.str(GPR::X0, slot_addr(inst.result()));
                    }
                    break;
                }
                case Opcode::xor_: {
                    if (inst.type().is_float()) {
                        enc.ldr(FPR::V0, slot_addr(inst.operand(0)));
                        enc.ldr(FPR::V1, slot_addr(inst.operand(1)));
                        enc.vec_eor(FPR::V0, FPR::V0, FPR::V1);
                        enc.str(FPR::V0, slot_addr(inst.result()));
                    } else if (inst.type().is_i32()) {
                        enc.ldr32(GPR::X0, slot_addr(inst.operand(0)));
                        enc.ldr32(GPR::X1, slot_addr(inst.operand(1)));
                        enc.eor32(GPR::X0, GPR::X0, GPR::X1);
                        enc.str32(GPR::X0, slot_addr(inst.result()));
                    } else {
                        enc.ldr(GPR::X0, slot_addr(inst.operand(0)));
                        enc.ldr(GPR::X1, slot_addr(inst.operand(1)));
                        enc.eor(GPR::X0, GPR::X0, GPR::X1);
                        enc.str(GPR::X0, slot_addr(inst.result()));
                    }
                    break;
                }
                case Opcode::not_: {
                    if (inst.type().is_i32()) {
                        enc.ldr32(GPR::X0, slot_addr(inst.operand(0)));
                        enc.mvn32(GPR::X0, GPR::X0);
                        enc.str32(GPR::X0, slot_addr(inst.result()));
                    } else {
                        enc.ldr(GPR::X0, slot_addr(inst.operand(0)));
                        enc.mvn(GPR::X0, GPR::X0);
                        enc.str(GPR::X0, slot_addr(inst.result()));
                    }
                    break;
                }
                case Opcode::shl: {
                    if (inst.type().is_i32()) {
                        enc.ldr32(GPR::X0, slot_addr(inst.operand(0)));
                        enc.ldr32(GPR::X1, slot_addr(inst.operand(1)));
                        enc.lsl32(GPR::X0, GPR::X0, GPR::X1);
                        enc.str32(GPR::X0, slot_addr(inst.result()));
                    } else {
                        enc.ldr(GPR::X0, slot_addr(inst.operand(0)));
                        enc.ldr(GPR::X1, slot_addr(inst.operand(1)));
                        enc.lsl(GPR::X0, GPR::X0, GPR::X1);
                        enc.str(GPR::X0, slot_addr(inst.result()));
                    }
                    break;
                }
                case Opcode::lshr: {
                    if (inst.type().is_i32()) {
                        enc.ldr32(GPR::X0, slot_addr(inst.operand(0)));
                        enc.ldr32(GPR::X1, slot_addr(inst.operand(1)));
                        enc.lsr32(GPR::X0, GPR::X0, GPR::X1);
                        enc.str32(GPR::X0, slot_addr(inst.result()));
                    } else {
                        enc.ldr(GPR::X0, slot_addr(inst.operand(0)));
                        enc.ldr(GPR::X1, slot_addr(inst.operand(1)));
                        enc.lsr(GPR::X0, GPR::X0, GPR::X1);
                        enc.str(GPR::X0, slot_addr(inst.result()));
                    }
                    break;
                }
                case Opcode::ashr: {
                    if (inst.type().is_i32()) {
                        enc.ldr32(GPR::X0, slot_addr(inst.operand(0)));
                        enc.ldr32(GPR::X1, slot_addr(inst.operand(1)));
                        enc.asr32(GPR::X0, GPR::X0, GPR::X1);
                        enc.str32(GPR::X0, slot_addr(inst.result()));
                    } else {
                        enc.ldr(GPR::X0, slot_addr(inst.operand(0)));
                        enc.ldr(GPR::X1, slot_addr(inst.operand(1)));
                        enc.asr(GPR::X0, GPR::X0, GPR::X1);
                        enc.str(GPR::X0, slot_addr(inst.result()));
                    }
                    break;
                }
                case Opcode::clz: {
                    if (inst.type().is_i32()) {
                        enc.ldr32(GPR::X0, slot_addr(inst.operand(0)));
                        enc.clz32(GPR::X0, GPR::X0);
                        enc.str32(GPR::X0, slot_addr(inst.result()));
                    } else {
                        enc.ldr(GPR::X0, slot_addr(inst.operand(0)));
                        enc.clz(GPR::X0, GPR::X0);
                        enc.str(GPR::X0, slot_addr(inst.result()));
                    }
                    break;
                }
                case Opcode::ctz: {
                    if (inst.type().is_i32()) {
                        enc.ldr32(GPR::X0, slot_addr(inst.operand(0)));
                        enc.rbit32(GPR::X0, GPR::X0);
                        enc.clz32(GPR::X0, GPR::X0);
                        enc.str32(GPR::X0, slot_addr(inst.result()));
                    } else {
                        enc.ldr(GPR::X0, slot_addr(inst.operand(0)));
                        enc.rbit(GPR::X0, GPR::X0);
                        enc.clz(GPR::X0, GPR::X0);
                        enc.str(GPR::X0, slot_addr(inst.result()));
                    }
                    break;
                }
                case Opcode::popcnt: {
                    if (inst.type().is_i32()) {
                        enc.ldr32(GPR::X0, slot_addr(inst.operand(0)));
                        enc.fmov_from_gpr32(FPR::V0, GPR::X0);
                        enc.cnt_8b(FPR::V0, FPR::V0);
                        enc.uaddlv_h(FPR::V0, FPR::V0);
                        enc.fmov_to_gpr32(GPR::X0, FPR::V0);
                        enc.str32(GPR::X0, slot_addr(inst.result()));
                    } else {
                        enc.ldr(GPR::X0, slot_addr(inst.operand(0)));
                        enc.fmov_from_gpr(FPR::V0, GPR::X0);
                        enc.cnt_8b(FPR::V0, FPR::V0);
                        enc.uaddlv_h(FPR::V0, FPR::V0);
                        enc.fmov_to_gpr(GPR::X0, FPR::V0);
                        enc.str(GPR::X0, slot_addr(inst.result()));
                    }
                    break;
                }

                // Comparisons
                case Opcode::eq:
                case Opcode::ne:
                case Opcode::slt:
                case Opcode::ult:
                case Opcode::sle:
                case Opcode::ule:
                case Opcode::sgt:
                case Opcode::ugt:
                case Opcode::sge:
                case Opcode::uge: {
                    bool is_flt = inst.operand(0)->type().is_float();
                    if (is_flt) {
                        enc.ldr(FPR::V0, slot_addr(inst.operand(0)));
                        enc.ldr(FPR::V1, slot_addr(inst.operand(1)));
                        enc.fcmp(FPR::V0, FPR::V1);
                        Condition cond = Condition::EQ;
                        switch (op) {
                            case Opcode::eq:  cond = Condition::EQ; break;
                            case Opcode::ne:  cond = Condition::NE; break;
                            case Opcode::slt:
                            case Opcode::ult: cond = Condition::MI; break;
                            case Opcode::sle:
                            case Opcode::ule: cond = Condition::LS; break;
                            case Opcode::sgt:
                            case Opcode::ugt: cond = Condition::GT; break;
                            case Opcode::sge:
                            case Opcode::uge: cond = Condition::GE; break;
                            default: break;
                        }
                        enc.cset(GPR::X0, cond);
                        enc.str32(GPR::X0, slot_addr(inst.result()));
                    } else {
                        bool is_32 = inst.operand(0)->type().is_i32();
                        if (is_32) {
                            enc.ldr32(GPR::X0, slot_addr(inst.operand(0)));
                            enc.ldr32(GPR::X1, slot_addr(inst.operand(1)));
                            enc.cmp32(GPR::X0, GPR::X1);
                        } else {
                            enc.ldr(GPR::X0, slot_addr(inst.operand(0)));
                            enc.ldr(GPR::X1, slot_addr(inst.operand(1)));
                            enc.cmp(GPR::X0, GPR::X1);
                        }
                        Condition cond = Condition::EQ;
                        switch (op) {
                            case Opcode::eq:  cond = Condition::EQ; break;
                            case Opcode::ne:  cond = Condition::NE; break;
                            case Opcode::slt: cond = Condition::LT; break;
                            case Opcode::ult: cond = Condition::CC; break;
                            case Opcode::sle: cond = Condition::LE; break;
                            case Opcode::ule: cond = Condition::LS; break;
                            case Opcode::sgt: cond = Condition::GT; break;
                            case Opcode::ugt: cond = Condition::HI; break;
                            case Opcode::sge: cond = Condition::GE; break;
                            case Opcode::uge: cond = Condition::CS; break;
                            default: break;
                        }
                        enc.cset(GPR::X0, cond);
                        enc.str32(GPR::X0, slot_addr(inst.result()));
                    }
                    break;
                }

                // Selection
                case Opcode::select: {
                    enc.ldr32(GPR::X0, slot_addr(inst.operand(0)));
                    enc.cmp32(GPR::X0, 0);
                    if (inst.type().is_float()) {
                        Label done_lbl = buffer.create_label();
                        enc.ldr(FPR::V0, slot_addr(inst.operand(1)));
                        enc.b(Condition::NE, done_lbl);
                        enc.ldr(FPR::V0, slot_addr(inst.operand(2)));
                        buffer.bind(done_lbl);
                        enc.str(FPR::V0, slot_addr(inst.result()));
                    } else if (inst.type().is_i32()) {
                        enc.ldr32(GPR::X1, slot_addr(inst.operand(1)));
                        enc.ldr32(GPR::X2, slot_addr(inst.operand(2)));
                        enc.csel32(GPR::X0, GPR::X1, GPR::X2, Condition::NE);
                        enc.str32(GPR::X0, slot_addr(inst.result()));
                    } else {
                        enc.ldr(GPR::X1, slot_addr(inst.operand(1)));
                        enc.ldr(GPR::X2, slot_addr(inst.operand(2)));
                        enc.csel(GPR::X0, GPR::X1, GPR::X2, Condition::NE);
                        enc.str(GPR::X0, slot_addr(inst.result()));
                    }
                    break;
                }

                // Memory
                case Opcode::load: {
                    enc.ldr(GPR::X0, slot_addr(inst.operand(0)));
                    MemAddress mem = ensure_accessible_mem(ptr(GPR::X0, inst.offset()), GPR::X16);
                    if (inst.type().is_float()) {
                        if (inst.type().kind() == TypeKind::F32) {
                            enc.ldr_s(FPR::V0, mem);
                            enc.str_s(FPR::V0, slot_addr(inst.result()));
                        } else {
                            enc.ldr(FPR::V0, mem);
                            enc.str(FPR::V0, slot_addr(inst.result()));
                        }
                    } else if (inst.type().is_i32()) {
                        enc.ldr32(GPR::X1, mem);
                        enc.str32(GPR::X1, slot_addr(inst.result()));
                    } else {
                        enc.ldr(GPR::X1, mem);
                        enc.str(GPR::X1, slot_addr(inst.result()));
                    }
                    break;
                }
                case Opcode::store: {
                    enc.ldr(GPR::X0, slot_addr(inst.operand(0)));
                    MemAddress mem = ensure_accessible_mem(ptr(GPR::X0, inst.offset()), GPR::X16);
                    if (inst.operand(1)->type().is_float()) {
                        if (inst.operand(1)->type().kind() == TypeKind::F32) {
                            enc.ldr_s(FPR::V0, slot_addr(inst.operand(1)));
                            enc.str_s(FPR::V0, mem);
                        } else {
                            enc.ldr(FPR::V0, slot_addr(inst.operand(1)));
                            enc.str(FPR::V0, mem);
                        }
                    } else if (inst.operand(1)->type().is_i32()) {
                        enc.ldr32(GPR::X1, slot_addr(inst.operand(1)));
                        enc.str32(GPR::X1, mem);
                    } else {
                        enc.ldr(GPR::X1, slot_addr(inst.operand(1)));
                        enc.str(GPR::X1, mem);
                    }
                    break;
                }
                case Opcode::load_indexed: {
                    enc.ldr(GPR::X0, slot_addr(inst.operand(0)));
                    enc.ldr(GPR::X1, slot_addr(inst.operand(1)));
                    uint8_t shift = 0;
                    if (inst.scale() == 8) shift = 3;
                    else if (inst.scale() == 4) shift = 2;
                    else if (inst.scale() == 2) shift = 1;

                    if (inst.offset() != 0) {
                        enc.mov(GPR::X16, static_cast<uint64_t>(inst.offset()));
                        enc.add(GPR::X0, GPR::X0, GPR::X16);
                    }
                    MemAddress mem = MemAddress::base_index(GPR::X0, GPR::X1, ExtendType::UXTX, shift);
                    if (inst.type().is_float()) {
                        enc.ldr(FPR::V0, mem);
                        enc.str(FPR::V0, slot_addr(inst.result()));
                    } else if (inst.type().is_i32()) {
                        enc.ldr32(GPR::X2, mem);
                        enc.str32(GPR::X2, slot_addr(inst.result()));
                    } else {
                        enc.ldr(GPR::X2, mem);
                        enc.str(GPR::X2, slot_addr(inst.result()));
                    }
                    break;
                }
                case Opcode::store_indexed: {
                    enc.ldr(GPR::X0, slot_addr(inst.operand(0)));
                    enc.ldr(GPR::X1, slot_addr(inst.operand(1)));
                    uint8_t shift = 0;
                    if (inst.scale() == 8) shift = 3;
                    else if (inst.scale() == 4) shift = 2;
                    else if (inst.scale() == 2) shift = 1;

                    if (inst.offset() != 0) {
                        enc.mov(GPR::X16, static_cast<uint64_t>(inst.offset()));
                        enc.add(GPR::X0, GPR::X0, GPR::X16);
                    }
                    MemAddress mem = MemAddress::base_index(GPR::X0, GPR::X1, ExtendType::UXTX, shift);
                    if (inst.operand(2)->type().is_float()) {
                        enc.ldr(FPR::V0, slot_addr(inst.operand(2)));
                        enc.str(FPR::V0, mem);
                    } else if (inst.operand(2)->type().is_i32()) {
                        enc.ldr32(GPR::X2, slot_addr(inst.operand(2)));
                        enc.str32(GPR::X2, mem);
                    } else {
                        enc.ldr(GPR::X2, slot_addr(inst.operand(2)));
                        enc.str(GPR::X2, mem);
                    }
                    break;
                }
                case Opcode::write_barrier: {
                    enc.ldr(GPR::X0, slot_addr(inst.operand(0)));
                    enc.ldr(GPR::X1, slot_addr(inst.operand(1)));
                    void* wb = reinterpret_cast<void*>(&brass_gc_write_barrier);
                    enc.mov(GPR::X16, reinterpret_cast<uint64_t>(wb));
                    enc.blr(GPR::X16);
                    break;
                }

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
                                    enc.ldr_s(dst_fpr, slot_addr(arg));
                                } else {
                                    enc.ldr(dst_fpr, slot_addr(arg));
                                }
                            } else {
                                enc.ldr(FPR::V0, slot_addr(arg));
                                enc.str(FPR::V0, ptr(GPR::SP, static_cast<int64_t>(cur_stack_arg++ * 8)));
                            }
                        } else {
                            if (gpr_idx < 8) {
                                GPR dst_gpr = static_cast<GPR>(gpr_idx++);
                                enc.ldr(dst_gpr, slot_addr(arg));
                            } else {
                                enc.ldr(GPR::X0, slot_addr(arg));
                                enc.str(GPR::X0, ptr(GPR::SP, static_cast<int64_t>(cur_stack_arg++ * 8)));
                            }
                        }
                    }

                    if (is_indirect) {
                        enc.ldr(GPR::X16, slot_addr(inst.operand(0)));
                        enc.blr(GPR::X16);
                    } else {
                        if (inst.symbol() == fn.name()) {
                            enc.bl(fn_entry_label);
                        } else {
                            void* sym_addr = resolve_sym(inst.symbol());
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
                                enc.str_s(FPR::V0, slot_addr(inst.result()));
                            } else {
                                enc.str(FPR::V0, slot_addr(inst.result()));
                            }
                        } else if (inst.type().is_i32()) {
                            enc.str32(GPR::X0, slot_addr(inst.result()));
                        } else {
                            enc.str(GPR::X0, slot_addr(inst.result()));
                        }
                    }
                    break;
                }

                // Terminators
                case Opcode::br: {
                    copy_block_args(inst.branch_target());
                    enc.b(block_labels[inst.branch_target().block->id()]);
                    break;
                }
                case Opcode::br_if: {
                    enc.ldr32(GPR::X0, slot_addr(inst.operand(0)));
                    enc.cmp32(GPR::X0, 0);
                    Label true_edge = buffer.create_label();
                    enc.b(Condition::NE, true_edge);

                    copy_block_args(inst.false_target());
                    enc.b(block_labels[inst.false_target().block->id()]);

                    buffer.bind(true_edge);
                    copy_block_args(inst.true_target());
                    enc.b(block_labels[inst.true_target().block->id()]);
                    break;
                }
                case Opcode::switch_: {
                    enc.ldr(GPR::X0, slot_addr(inst.operand(0)));
                    std::vector<Label> case_labels;
                    case_labels.reserve(inst.switch_cases().size());
                    for (const auto& sc : inst.switch_cases()) {
                        Label case_body = buffer.create_label();
                        case_labels.push_back(case_body);
                        enc.mov(GPR::X1, static_cast<uint64_t>(sc.value));
                        enc.cmp(GPR::X0, GPR::X1);
                        enc.b(Condition::EQ, case_body);
                    }

                    copy_block_args(inst.default_target());
                    enc.b(block_labels[inst.default_target().block->id()]);

                    for (size_t i = 0; i < inst.switch_cases().size(); ++i) {
                        buffer.bind(case_labels[i]);
                        copy_block_args(inst.switch_cases()[i].target);
                        enc.b(block_labels[inst.switch_cases()[i].target.block->id()]);
                    }
                    break;
                }
                case Opcode::ret: {
                    if (inst.operand_count() > 0 && inst.operand(0) != nullptr) {
                        const Value* rval = inst.operand(0);
                        if (rval->type().is_float()) {
                            if (rval->type().kind() == TypeKind::F32) {
                                enc.ldr_s(FPR::V0, slot_addr(rval));
                            } else {
                                enc.ldr(FPR::V0, slot_addr(rval));
                            }
                        } else if (rval->type().is_i32()) {
                            enc.ldr32(GPR::X0, slot_addr(rval));
                        } else {
                            enc.ldr(GPR::X0, slot_addr(rval));
                        }
                    }
                    if (frame_size <= 512) {
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
                    enc.ldr32(GPR::X0, slot_addr(inst.operand(0)));
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

    // Allocate executable memory and copy code
    size_t code_bytes = buffer.size();
    auto mem_block = std::make_shared<JitMemoryBlock>(code_bytes);
    if (!mem_block->is_valid()) {
        throw std::runtime_error("compile_baseline_aarch64: Failed to allocate executable memory for " + std::string(fn.name()));
    }
    std::memcpy(mem_block->data(), buffer.data(), code_bytes);
    mem_block->make_executable_read_only();

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

} // namespace brass::aarch64
