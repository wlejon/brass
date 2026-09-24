// AArch64 baseline tier: slot addressing, the argument convention, calls,
// block-argument copies, stack maps and the frame's size needs.
#include "aarch64_baseline_emit_internal.hpp"
#include <brass/runtime/code_installer.hpp>
#include <algorithm>

namespace brass::aarch64 {

using namespace brass::codegen;

namespace {

bool fits_access(int64_t disp, int size) {
    if (disp >= -256 && disp <= 255) return true;
    return disp >= 0 && disp % size == 0 && disp / size <= 4095;
}

// The value's type as the convention places it: floats and 128-bit vectors
// in V registers.
bool in_fp_reg(Type t) { return t.is_float() || t.is_v128(); }

uint32_t stack_size(Type t) {
    uint32_t sz = static_cast<uint32_t>(t.size_in_bytes());
    return sz == 0 ? 8 : sz;
}

} // namespace

std::vector<A64ArgLoc> a64_assign_args(const Target& target, const std::vector<Type>& types,
                                       uint32_t* stack_bytes) {
    // aarch64_isel.cpp's entry and call lowering: X0-X7 and V0-V7 fill
    // independently; on the stack a value takes an 8-byte slot (16 for a
    // vector, 16-aligned), or on Apple its natural size and alignment.
    const bool apple = target.is_macos();
    std::vector<A64ArgLoc> locs;
    locs.reserve(types.size());
    unsigned gpr = 0, fpr = 0;
    uint32_t stack = 0;
    for (Type t : types) {
        A64ArgLoc loc;
        loc.fp = in_fp_reg(t);
        unsigned& next = loc.fp ? fpr : gpr;
        if (next < 8) {
            loc.in_reg = true;
            loc.reg = next++;
        } else {
            const uint32_t sz = stack_size(t);
            const uint32_t align = apple ? std::min<uint32_t>(sz, 16) : (sz >= 16 ? 16 : 8);
            stack = (stack + align - 1) & ~(align - 1);
            loc.stack_offset = stack;
            loc.size = sz;
            stack += apple ? sz : (sz >= 16 ? 16 : 8);
        }
        locs.push_back(loc);
    }
    if (stack_bytes) *stack_bytes = stack;
    return locs;
}

MemAddress AArch64BaselineEmitter::based(GPR base, int64_t disp, int size) const {
    if (fits_access(disp, size)) return ptr(base, disp);
    if (disp >= 0) enc.add(GPR::X17, base, static_cast<uint32_t>(disp));
    else if (-disp <= 0xFFFFFF) enc.sub(GPR::X17, base, static_cast<uint32_t>(-disp));
    else {
        enc.mov(GPR::X17, static_cast<uint64_t>(disp));
        enc.add(GPR::X17, base, GPR::X17);
    }
    return ptr(GPR::X17, 0);
}

MemAddress AArch64BaselineEmitter::off_addr(int32_t off, int size) const {
    // Below fp; the same byte is sp + (frame_bytes - off), where a scaled
    // offset reaches much further than ldur's 256 bytes.
    const int64_t from_fp = -static_cast<int64_t>(off);
    if (from_fp >= -256) return ptr(GPR::FP, from_fp);
    return based(GPR::SP, static_cast<int64_t>(frame_bytes) - off, size);
}

MemAddress AArch64BaselineEmitter::slot_addr(const Value* val, int size) const {
    auto it = layout.slot_map.find(val);
    if (it == layout.slot_map.end()) {
        throw_unsupported(kA64BaselineStage, "operand with no frame slot in " + std::string(fn.name()));
    }
    return off_addr(it->second, size);
}

MemAddress AArch64BaselineEmitter::slot_addr_at(const Value* val, int32_t byte, int size) const {
    auto it = layout.slot_map.find(val);
    if (it == layout.slot_map.end()) {
        throw_unsupported(kA64BaselineStage, "operand with no frame slot in " + std::string(fn.name()));
    }
    return off_addr(it->second - byte, size);
}

void* AArch64BaselineEmitter::resolve_sym(std::string_view name) const {
    if (resolver) {
        if (void* addr = resolver(name)) return addr;
    }
    return nullptr;
}

void* AArch64BaselineEmitter::resolve_or_stub(std::string_view name) {
    if (void* addr = resolve_sym(name)) return addr;
    if (!lazy) {
        throw_unsupported(kA64BaselineStage, "unresolved symbol " + std::string(name) + " with no lazy-link table");
    }
    uses_lazy_stubs = true;
    return lazy->stub_for(name);
}

void AArch64BaselineEmitter::load_gpr(GPR dst, const Value* v) {
    if (bl_is_int32(v->type()) || bl_is_f32(v->type())) enc.ldr32(dst, slot_addr(v, 4));
    else enc.ldr(dst, slot_addr(v, 8));
}

void AArch64BaselineEmitter::store_gpr(const Value* v, GPR src) {
    if (bl_is_int32(v->type()) || bl_is_f32(v->type())) enc.str32(src, slot_addr(v, 4));
    else enc.str(src, slot_addr(v, 8));
}

void AArch64BaselineEmitter::load_fp(FPR dst, const Value* v) {
    if (bl_is_f32(v->type())) enc.ldr_s(dst, slot_addr(v, 4));
    else enc.ldr(dst, slot_addr(v, 8));
}

void AArch64BaselineEmitter::store_fp(const Value* v, FPR src) {
    if (bl_is_f32(v->type())) enc.str_s(src, slot_addr(v, 4));
    else enc.str(src, slot_addr(v, 8));
}

void AArch64BaselineEmitter::load_v(FPR dst, const Value* v) { enc.ldr_q(dst, slot_addr(v, 16)); }
void AArch64BaselineEmitter::store_v(const Value* v, FPR src) { enc.str_q(src, slot_addr(v, 16)); }

void AArch64BaselineEmitter::copy_block_args(const BranchTarget& target_branch) {
    if (!target_branch.block || target_branch.args.empty()) return;
    const auto& params = target_branch.block->params();
    const size_t count = std::min(target_branch.args.size(), params.size());
    // Through the parallel-copy area, so a copy whose sources are also
    // destinations reads every source before any destination is written.
    std::vector<int32_t> at(count);
    int32_t pos = copy_area;
    for (size_t j = 0; j < count; ++j) {
        const bool vec = bl_is_v128(params[j]->type());
        if (vec) pos = (pos + 15) & ~15;
        at[j] = pos;
        if (vec) {
            load_v(FPR::V0, target_branch.args[j]);
            enc.str_q(FPR::V0, based(GPR::SP, pos, 16));
        } else {
            enc.ldr(GPR::X0, slot_addr(target_branch.args[j], 8));
            enc.str(GPR::X0, based(GPR::SP, pos, 8));
        }
        pos += vec ? 16 : 8;
    }
    for (size_t j = 0; j < count; ++j) {
        if (bl_is_v128(params[j]->type())) {
            enc.ldr_q(FPR::V0, based(GPR::SP, at[j], 16));
            store_v(params[j], FPR::V0);
        } else {
            enc.ldr(GPR::X0, based(GPR::SP, at[j], 8));
            enc.str(GPR::X0, slot_addr(params[j], 8));
        }
    }
}

void AArch64BaselineEmitter::call_abs(const void* fn_ptr) {
    enc.mov(GPR::X16, reinterpret_cast<uint64_t>(fn_ptr));
    enc.blr(GPR::X16);
}

void AArch64BaselineEmitter::record_safepoint(uint32_t site_id) {
    StackMapRecord rec;
    rec.instruction_offset = static_cast<uint32_t>(buffer.size());
    rec.frame_size = static_cast<uint32_t>(frame_bytes);
    rec.safepoint_id = site_id;
    for (int32_t off : layout.gcref_slots) {
        rec.add_root(StackMapRootLocation::frame_slot(-off));
    }
    fn_stack_map.add_record(std::move(rec));
}

void AArch64BaselineEmitter::emit_return() {
    if (preserves_tls) enc.ldr(GPR::X28, ptr(GPR::FP, -16));
    enc.mov(GPR::SP, GPR::FP);
    enc.ldp(GPR::FP, GPR::LR, post_idx(GPR::SP, 16));
    enc.ret();
}

void AArch64BaselineEmitter::emit_call(std::string_view symbol, const Value* indirect,
                                       const std::vector<const Value*>& args, const Value* result,
                                       uint32_t site_id) {
    std::vector<Type> types;
    types.reserve(args.size());
    for (const Value* a : args) types.push_back(a->type());
    const std::vector<A64ArgLoc> locs = a64_assign_args(target, types, nullptr);

    // The callee pointer first: argument loads use only X0-X7 / V0-V7 and
    // X9 / V16 / X17 as scratch, never X16.
    if (indirect) enc.ldr(GPR::X16, slot_addr(indirect, 8));

    for (size_t i = 0; i < args.size(); ++i) {
        const Value* arg = args[i];
        const A64ArgLoc& loc = locs[i];
        const Type t = arg->type();
        if (loc.in_reg) {
            if (!loc.fp) load_gpr(static_cast<GPR>(loc.reg), arg);
            else if (bl_is_v128(t)) load_v(static_cast<FPR>(loc.reg), arg);
            else load_fp(static_cast<FPR>(loc.reg), arg);
            continue;
        }
        const int64_t at = loc.stack_offset;
        if (bl_is_v128(t)) {
            load_v(FPR::V16, arg);
            enc.str_q(FPR::V16, based(GPR::SP, at, 16));
            continue;
        }
        // The value's bytes at its stack size: an integer or float's slot
        // holds it in its low bytes.
        switch (loc.size) {
            case 1: enc.ldr32(GPR::X9, slot_addr(arg, 4)); enc.strb(GPR::X9, based(GPR::SP, at, 1)); break;
            case 2: enc.ldr32(GPR::X9, slot_addr(arg, 4)); enc.strh(GPR::X9, based(GPR::SP, at, 2)); break;
            case 4: enc.ldr32(GPR::X9, slot_addr(arg, 4)); enc.str32(GPR::X9, based(GPR::SP, at, 4)); break;
            default:
                if (bl_is_int32(t) || bl_is_f32(t)) enc.ldr32(GPR::X9, slot_addr(arg, 4));
                else enc.ldr(GPR::X9, slot_addr(arg, 8));
                enc.str(GPR::X9, based(GPR::SP, at, 8));
                break;
        }
    }

    if (indirect) {
        enc.blr(GPR::X16);
    } else if (symbol == fn.name()) {
        enc.bl(fn_entry_label);
    } else {
        void* target_addr = resolve_sym(symbol);
        if (!target_addr) {
            target_addr = resolve_or_stub(symbol);
            if (std::find(lazy_call_symbols.begin(), lazy_call_symbols.end(), symbol) == lazy_call_symbols.end()) {
                lazy_call_symbols.emplace_back(symbol);
            }
        }
        call_abs(target_addr);
    }
    // Keyed by the return address.
    record_safepoint(site_id);

    if (result) {
        const Type rt = result->type();
        if (bl_is_v128(rt)) store_v(result, FPR::V0);
        else if (rt.is_float()) store_fp(result, FPR::V0);
        else store_gpr(result, GPR::X0);
    }
}

namespace {

template <typename F>
void for_each_call(const Function& fn, F&& visit) {
    for (const auto* bb : fn.blocks()) {
        if (!bb) continue;
        for (const auto* inst : *bb) {
            if (!inst) continue;
            std::vector<Type> types;
            switch (inst->opcode()) {
                case Opcode::call:
                case Opcode::patchable_call:
                case Opcode::call_indirect: {
                    const size_t first = inst->opcode() == Opcode::call_indirect ? 1 : 0;
                    for (size_t i = first; i < inst->operand_count(); ++i) types.push_back(inst->operand(i)->type());
                    visit(types);
                    break;
                }
                case Opcode::guard:
                    // A guard's exit stub is called with the state map.
                    for (const Value* v : inst->state_map()) types.push_back(v->type());
                    visit(types);
                    break;
                default:
                    break;
            }
        }
    }
}

} // namespace

uint32_t a64_baseline_outgoing_bytes(const Function& fn, const Target& target) {
    uint32_t most = 0;
    for_each_call(fn, [&](const std::vector<Type>& types) {
        uint32_t bytes = 0;
        (void)a64_assign_args(target, types, &bytes);
        most = std::max(most, bytes);
    });
    return (most + 15) & ~15u;
}

uint32_t a64_baseline_copy_bytes(const Function& fn) {
    uint32_t most = 0;
    auto visit = [&](const BranchTarget& t) {
        if (!t.block) return;
        uint32_t bytes = 0;
        const auto& params = t.block->params();
        const size_t count = std::min(t.args.size(), params.size());
        for (size_t j = 0; j < count; ++j) {
            if (bl_is_v128(params[j]->type())) bytes = ((bytes + 15) & ~15u) + 16;
            else bytes += 8;
        }
        most = std::max(most, bytes);
    };
    for (const auto* bb : fn.blocks()) {
        if (!bb) continue;
        for (const auto* inst : *bb) {
            if (!inst) continue;
            switch (inst->opcode()) {
                case Opcode::br: visit(inst->branch_target()); break;
                case Opcode::br_if: visit(inst->true_target()); visit(inst->false_target()); break;
                case Opcode::switch_:
                    visit(inst->default_target());
                    for (const auto& sc : inst->switch_cases()) visit(sc.target);
                    break;
                case Opcode::guard:
                    if (BasicBlock* resume = fn.get_resume_target(inst->resume_id())) {
                        visit(BranchTarget(resume, inst->state_map()));
                    }
                    break;
                default: break;
            }
        }
    }
    return (most + 15) & ~15u;
}

} // namespace brass::aarch64
