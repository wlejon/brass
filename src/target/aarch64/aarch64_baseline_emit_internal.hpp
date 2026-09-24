#pragma once

// AArch64 baseline tier: the emitter the opcode files share. The tier is the
// x64 baseline (codegen/baseline_emit*.cpp) on AArch64 - the same slot
// conventions and frame layout (codegen/baseline_frame.hpp), the same
// pre-scan, the same exits - so the two agree with the interpreter and with
// each other on everything they compile.
//
// Frame, a standard AAPCS64 frame record:
//
//   fp + 16 ...          incoming stack arguments
//   fp + 8               lr
//   fp                   caller's fp
//   [fp - 16             the caller's pinned-TLS register, when saved]
//   fp - off             the value / alloca slot at layout offset `off`
//   sp + out_bytes ...   the parallel-copy area (copy_block_args)
//   sp                   outgoing stack arguments
//
// sp is set once by the prologue and never moves, so slots are reachable
// from both fp (a short negative offset) and sp (a scaled positive one).
// Every value lives in its slot between instructions; the emitters work in
// X0-X7 / V0-V7 and use X16 for call targets, X17 for addressing.

#include <brass/target/aarch64/aarch64_baseline_emit.hpp>
#include <brass/target/aarch64/aarch64_encoder.hpp>
#include <brass/target/aarch64/code_buffer.hpp>
#include <brass/target/calling_conv.hpp>
#include <brass/codegen/unsupported_operation.hpp>
#include "../../codegen/baseline_frame.hpp"
#include <memory>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace brass::aarch64 {

// The stage name every AArch64 baseline rejection carries.
inline constexpr std::string_view kA64BaselineStage = "aarch64 baseline";

// Where an argument travels under the brass AArch64 convention (tier 2's
// entry and call lowering, aarch64_isel.cpp): integers in X0-X7, floats and
// 128-bit vectors in V0-V7, the rest on the stack at `stack_offset` - in
// 8-byte (16 for a vector) slots, or on Apple packed at natural size and
// alignment.
struct A64ArgLoc {
    bool in_reg = false;
    bool fp = false;          // a V register (float or vector)
    unsigned reg = 0;         // X / V register number when in_reg
    uint32_t stack_offset = 0;
    uint32_t size = 8;        // bytes the value occupies on the stack
};
std::vector<A64ArgLoc> a64_assign_args(const Target& target, const std::vector<Type>& types,
                                       uint32_t* stack_bytes);

struct AArch64BaselineEmitter {
    CodeBuffer& buffer;
    AArch64Encoder& enc;
    Target target;
    const Function& fn;
    const codegen::BaselineFrameLayout& layout;
    const std::unordered_map<uint32_t, Label>& block_labels;
    FunctionStackMap& fn_stack_map;
    BaselineSymbolResolver resolver;
    // Bytes `sub sp` reserves below fp: slots, parallel-copy area, outgoing
    // arguments. fp == sp + frame_bytes.
    int32_t frame_bytes = 0;
    int32_t copy_area = 0;    // offset of the parallel-copy area from sp
    Label fn_entry_label;
    bool preserves_tls = false;
    std::shared_ptr<codegen::LazySymbolTable> lazy;
    bool uses_lazy_stubs = false;
    std::vector<std::string> lazy_call_symbols;

    // The slot of a value / at a layout offset, for an access of `size`
    // bytes: [fp, #-off] when that encodes, [sp, #frame_bytes-off] when
    // that does, else X17 = its address.
    MemAddress slot_addr(const Value* val, int size) const;
    MemAddress slot_addr_at(const Value* val, int32_t byte, int size) const;
    MemAddress off_addr(int32_t off, int size) const;
    // [base, #disp] for an access of `size` bytes, through X17 when the
    // displacement does not encode.
    MemAddress based(GPR base, int64_t disp, int size) const;

    void* resolve_sym(std::string_view name) const;
    void* resolve_or_stub(std::string_view name);

    // A value's slot to / from a register at the value's width (32-bit for
    // narrow integers and f32, whole for the rest).
    void load_gpr(GPR dst, const Value* v);
    void store_gpr(const Value* v, GPR src);
    void load_fp(FPR dst, const Value* v);
    void store_fp(const Value* v, FPR src);
    void load_v(FPR dst, const Value* v);
    void store_v(const Value* v, FPR src);

    void copy_block_args(const BranchTarget& target_branch);
    // mov x16, fn; blr x16.
    void call_abs(const void* fn_ptr);
    void record_safepoint(uint32_t site_id);
    void emit_return();
    void emit_call(std::string_view symbol, const Value* indirect,
                   const std::vector<const Value*>& args, const Value* result, uint32_t site_id);
};

// The frame's outgoing-argument and parallel-copy needs, for the prologue.
uint32_t a64_baseline_outgoing_bytes(const Function& fn, const Target& target);
uint32_t a64_baseline_copy_bytes(const Function& fn);

// Each returns true if it handled the opcode.
bool emit_baseline_aarch64_op(AArch64BaselineEmitter& emitter, const Instruction& inst);
bool emit_baseline_aarch64_fp_op(AArch64BaselineEmitter& emitter, const Instruction& inst);
bool emit_baseline_aarch64_vec_op(AArch64BaselineEmitter& emitter, const Instruction& inst);

// Rejects a vector use the tier does not compile (as the x64 tier's
// check_x64_baseline_vector_inst).
void check_aarch64_baseline_vector_inst(const Instruction& inst, const Target& target);

// The fixed prologue, as code offsets just past each step:
//
//   stp x29, x30, [sp, #-16]!   stp_end
//   mov x29, sp                 mov_end
//   [str x28, [sp, #-16]!]      tls_save_end (0 when the register is not saved)
//   [sub sp, sp, #alloc_bytes]
struct A64BaselinePrologue {
    uint32_t stp_end = 0;
    uint32_t mov_end = 0;
    uint32_t tls_save_end = 0;
    uint32_t alloc_bytes = 0;
};

// Unwind data for the prologue, appended to `image` after the code; each
// returns the offset to pass to JitMemoryBlock::register_unwind_info. DWARF
// .eh_frame (CIE, one FDE, terminator), or Windows ARM64 .xdata followed by
// its RUNTIME_FUNCTION.
size_t append_aarch64_baseline_eh_frame(std::vector<uint8_t>& image, uint32_t code_size,
                                        const A64BaselinePrologue& prologue);
size_t append_aarch64_baseline_win_unwind(std::vector<uint8_t>& image, uint32_t code_size,
                                          const A64BaselinePrologue& prologue);

} // namespace brass::aarch64
