// x64 baseline tier: calls, helper calls, stack maps and the small slot
// helpers the opcode emitters share.
#include "baseline_emit_internal.hpp"
#include <brass/runtime/code_installer.hpp>
#include <algorithm>

namespace brass::codegen {

using namespace brass::x64;

void* X64BaselineEmitter::resolve_sym(std::string_view name) const {
    if (resolver) {
        if (void* addr = resolver(name)) return addr;
    }
    return nullptr;
}

void* X64BaselineEmitter::resolve_or_stub(std::string_view name) {
    if (void* addr = resolve_sym(name)) return addr;
    if (!lazy) {
        throw_unsupported(kX64BaselineStage, "unresolved symbol " + std::string(name) + " with no lazy-link table");
    }
    uses_lazy_stubs = true;
    return lazy->stub_for(name);
}

void X64BaselineEmitter::copy_block_args(const BranchTarget& target_branch) {
    if (!target_branch.block || target_branch.args.empty()) return;
    const auto& params = target_branch.block->params();
    size_t count = std::min(target_branch.args.size(), params.size());
    // Through the machine stack, so a parallel copy whose sources are also
    // destinations reads every source before any destination is written.
    // A 128-bit vector goes as two words, high first.
    for (size_t j = 0; j < count; ++j) {
        const Value* arg = target_branch.args[j];
        if (bl_is_v128(params[j]->type())) {
            enc.mov(GPR::RAX, slot_addr_at(arg, 8));
            enc.push(GPR::RAX);
        }
        enc.mov(GPR::RAX, slot_addr(arg));
        enc.push(GPR::RAX);
    }
    for (size_t j = 0; j < count; ++j) {
        size_t idx = count - 1 - j;
        enc.pop(GPR::RAX);
        enc.mov(slot_addr(params[idx]), GPR::RAX);
        if (bl_is_v128(params[idx]->type())) {
            enc.pop(GPR::RAX);
            enc.mov(slot_addr_at(params[idx], 8), GPR::RAX);
        }
    }
}

void X64BaselineEmitter::call_abs(const void* fn_ptr) {
    enc.movabs(GPR::R11, reinterpret_cast<uint64_t>(fn_ptr));
    enc.call(GPR::R11);
}

void X64BaselineEmitter::record_safepoint(uint32_t site_id) {
    StackMapRecord rec;
    rec.instruction_offset = static_cast<uint32_t>(buffer.size());
    rec.frame_size = static_cast<uint32_t>(frame_size);
    rec.safepoint_id = site_id;
    for (int32_t off : gcref_slots) {
        rec.add_root(StackMapRootLocation::frame_slot(-off));
    }
    fn_stack_map.add_record(std::move(rec));
}

void X64BaselineEmitter::load_gpr(GPR dst, const Value* v) {
    if (bl_is_int32(v->type()) || bl_is_f32(v->type())) enc.mov32(dst, slot_addr(v));
    else enc.mov(dst, slot_addr(v));
}

void X64BaselineEmitter::store_gpr(const Value* v, GPR src) {
    if (bl_is_int32(v->type()) || bl_is_f32(v->type())) enc.mov32(slot_addr(v), src);
    else enc.mov(slot_addr(v), src);
}

void X64BaselineEmitter::test_cond(const Value* cond) {
    load_gpr(GPR::RAX, cond);
    enc.test(GPR::RAX, GPR::RAX);
}

void X64BaselineEmitter::emit_return() {
    if (preserves_r13) {
        enc.mov(GPR::R13, MemAddress::base_disp(GPR::RBP, -8));
    }
    enc.mov(GPR::RSP, GPR::RBP);
    enc.pop(GPR::RBP);
    enc.ret();
}

void X64BaselineEmitter::emit_call(std::string_view symbol, const Value* indirect,
                                   const std::vector<const Value*>& args, const Value* result,
                                   uint32_t site_id) {
    const bool win = target.is_windows();
    const size_t num_args = args.size();

    // Stack arguments needed.
    size_t gpr_idx = 0, xmm_idx = 0, stack_args = 0;
    for (size_t i = 0; i < num_args; ++i) {
        bool is_flt = bl_in_xmm(args[i]->type());
        if (win) {
            if (i >= 4) stack_args++;
        } else if (is_flt) {
            if (xmm_idx < cc.arg_xmms().size()) xmm_idx++; else stack_args++;
        } else {
            if (gpr_idx < cc.arg_gprs().size()) gpr_idx++; else stack_args++;
        }
    }
    int32_t call_stack_alloc = static_cast<int32_t>((win ? 32 : 0) + stack_args * 8);
    call_stack_alloc = (call_stack_alloc + 15) & ~15;
    if (call_stack_alloc > 0) enc.sub(GPR::RSP, call_stack_alloc);

    // The callee pointer is read before any argument register is written.
    if (indirect) enc.mov(GPR::R11, slot_addr(indirect));

    static constexpr GPR kWinGprs[4] = {GPR::RCX, GPR::RDX, GPR::R8, GPR::R9};
    static constexpr XMM kWinXmms[4] = {XMM::XMM0, XMM::XMM1, XMM::XMM2, XMM::XMM3};
    gpr_idx = 0;
    xmm_idx = 0;
    size_t cur_stack_arg = 0;
    auto push_stack = [&](const Value* arg, int32_t disp) {
        // The pre-scan admitted no vector argument on the stack.
        if (bl_is_v128(arg->type())) throw_unsupported(kX64BaselineStage, "vector argument on the stack");
        enc.mov(GPR::RAX, slot_addr(arg));
        enc.mov(MemAddress::base_disp(GPR::RSP, disp), GPR::RAX);
    };
    // A 128-bit vector travels whole in its XMM register.
    auto load_xmm = [&](XMM x, const Value* arg) {
        if (bl_is_v128(arg->type())) enc.movups(x, slot_addr(arg));
        else enc.movsd(x, slot_addr(arg));
    };
    for (size_t i = 0; i < num_args; ++i) {
        const Value* arg = args[i];
        bool is_flt = bl_in_xmm(arg->type());
        if (win) {
            if (i < 4) {
                if (is_flt) load_xmm(kWinXmms[i], arg);
                else load_gpr(kWinGprs[i], arg);
            } else {
                push_stack(arg, static_cast<int32_t>(32 + (i - 4) * 8));
            }
        } else if (is_flt) {
            if (xmm_idx < cc.arg_xmms().size()) load_xmm(cc.arg_xmms()[xmm_idx++], arg);
            else push_stack(arg, static_cast<int32_t>(cur_stack_arg++ * 8));
        } else {
            if (gpr_idx < cc.arg_gprs().size()) load_gpr(cc.arg_gprs()[gpr_idx++], arg);
            else push_stack(arg, static_cast<int32_t>(cur_stack_arg++ * 8));
        }
    }

    if (indirect) {
        enc.call(GPR::R11);
    } else if (symbol == fn.name()) {
        enc.call(fn_entry_label);
    } else {
        call_abs(resolve_or_stub(symbol));
    }

    // Keyed by the return address, so recorded before the stack is popped.
    record_safepoint(site_id);
    if (call_stack_alloc > 0) enc.add(GPR::RSP, call_stack_alloc);

    if (result) {
        Type rt = result->type();
        if (bl_is_v128(rt)) {
            enc.movups(slot_addr(result), XMM::XMM0);
        } else if (rt.is_float()) {
            if (bl_is_f32(rt)) enc.movss(slot_addr(result), XMM::XMM0);
            else enc.movsd(slot_addr(result), XMM::XMM0);
        } else {
            store_gpr(result, GPR::RAX);
        }
    }
}

} // namespace brass::codegen
