#pragma once

#include <brass/codegen/baseline_jit.hpp>
#include "baseline_frame.hpp"
#include <brass/codegen/unsupported_operation.hpp>
#include <brass/runtime/exception.hpp>
#include <brass/target/x64/x64_encoder.hpp>
#include <brass/target/calling_conv.hpp>
#include <unordered_map>
#include <string>
#include <string_view>
#include <vector>

namespace brass::codegen {

using namespace brass::x64;

// The stage name every x64 baseline rejection carries (UnsupportedOperation).
inline constexpr std::string_view kX64BaselineStage = "x64 baseline";

// Slot conventions and the frame layout: baseline_frame.hpp.

// Passed in an XMM register: floats and 128-bit vectors.
inline bool bl_in_xmm(Type t) { return t.is_float() || t.is_v128(); }

// True when every 128-bit vector among `types`, laid out as the arguments
// of a brass x64 call, travels in an XMM register: Win64 gives argument i
// XMMi for i < 4, SysV the next free argument XMM. That is where tier 2's
// entry and the invoke thunks put a vector; a vector on the stack has no
// layout all tiers agree on, so this tier rejects it.
bool bl_vectors_in_registers(const Target& target, const CallingConvention& cc,
                             const std::vector<Type>& types);

// Rejects (UnsupportedOperation) an instruction whose vector use this tier
// does not compile: every vector opcode is checked, and a vector-typed
// operand or result of any other opcode is allowed only for select, the
// memory accesses, calls and ret.
void check_x64_baseline_vector_inst(const Function& fn, const Instruction& inst, const Target& target,
                                    const CallingConvention& cc);

struct X64BaselineEmitter {
    CodeBuffer& buffer;
    X64Encoder& enc;
    Target target;
    const Function& fn;
    const std::unordered_map<const Value*, int32_t>& slot_map;
    const std::unordered_map<const Instruction*, int32_t>& alloca_offsets;
    const std::unordered_map<uint32_t, Label>& block_labels;
    FunctionStackMap& fn_stack_map;
    BaselineSymbolResolver resolver;
    int32_t frame_size = 0;
    CallingConvention cc;
    Label fn_entry_label;
    const std::vector<int32_t>& gcref_slots;
    const std::vector<int32_t>& tagged_slots;
    bool preserves_r13 = false;
    // Stubs for symbols unresolved at compile time; set when one is used, so
    // the compiled function keeps the table alive.
    LazySymbolTable* lazy = nullptr;
    bool uses_lazy_stubs = false;
    // The direct-call targets that got a stub, each once.
    std::vector<std::string> lazy_call_symbols;
    // The func_addr targets that got a stub, each once: their address may
    // be taken and never called, so they are resolved on first call.
    std::vector<std::string> lazy_addr_symbols;
    // func_addr of a module function: its canonical address
    // (BaselineJitCompiler::function_address_in), or null.
    BaselineSymbolResolver function_address;

    // The invoke call sites: [begin, end) holds the call, whose return
    // address is `end`, and a throw out of it lands at `pad`.
    struct EhScope {
        uint32_t begin = 0;
        uint32_t end = 0;
        Label pad;
    };
    std::vector<EhScope> eh_scopes;
    // An invoke whose unwind edge passes arguments lands on a trampoline of
    // its own that copies them (emitted after the blocks), not on the pad.
    std::vector<std::pair<Label, const Instruction*>> unwind_trampolines;

    MemAddress slot_addr(const Value* val) const {
        auto it = slot_map.find(val);
        if (it == slot_map.end()) {
            throw_unsupported(kX64BaselineStage, "operand with no frame slot in " + std::string(fn.name()));
        }
        return MemAddress::base_disp(GPR::RBP, -it->second);
    }

    // Byte `byte` of a (vector) value's slot.
    MemAddress slot_addr_at(const Value* val, int32_t byte) const {
        MemAddress base = slot_addr(val);
        return MemAddress::base_disp(GPR::RBP, base.disp + byte);
    }

    MemAddress slot_off_addr(int32_t off) const {
        return MemAddress::base_disp(GPR::RBP, -off);
    }

    Label block_label(const BasicBlock* bb) const { return block_labels.at(bb->id()); }

    void* resolve_sym(std::string_view name) const;
    // The symbol's address if it resolves now, else its lazy-link stub.
    void* resolve_or_stub(std::string_view name);
    void copy_block_args(const BranchTarget& target_branch);

    // mov r11, fn; call r11 — for runtime helpers taking register args.
    void call_abs(const void* fn_ptr);
    // Records a stack map for the return address just emitted.
    void record_safepoint(uint32_t site_id);
    // Loads a value's slot into a GPR at the value's width (zero-extended
    // for 32-bit values) and stores one back.
    void load_gpr(GPR dst, const Value* v);
    void store_gpr(const Value* v, GPR src);
    // Loads a condition (any integer width) and sets ZF = (cond == 0).
    void test_cond(const Value* cond);
    // Emits the epilogue and `ret`, the value already in RAX / XMM0.
    void emit_return();
    // A call with the platform C convention to `symbol` (resolved, or through
    // its lazy-link stub) or to the pointer in `indirect`; the result, if
    // any, goes to `result`'s slot. `call_start` / `call_end`, when given,
    // receive the offsets of the call instruction and of its return address.
    void emit_call(std::string_view symbol, const Value* indirect,
                   const std::vector<const Value*>& args, const Value* result,
                   uint32_t site_id, uint32_t* call_start = nullptr, uint32_t* call_end = nullptr);

    // The exception opcodes: invoke, landing_pad, throw and resume.
    void emit_invoke(const Instruction& inst);
    void emit_landing_pad(const Instruction& inst);
    void emit_raise(const Instruction& inst);
    // The unwind trampolines the invokes asked for; after the last block.
    void emit_unwind_trampolines();
};

// The function's exception table for the brass frame walker and the Win64
// scope table: every invoke's call and pad as code offsets, and the frame
// the walker restores (RSP = RBP - frame_size, R13 at [RBP-8] when saved).
runtime::FunctionExceptionTable baseline_exception_table(const X64BaselineEmitter& em,
                                                         std::string_view name);

// Code offsets just past each step of the fixed prologue
// `push rbp ; mov rbp, rsp ; sub rsp, frame_size ; [mov [rbp-8], r13]`
// (r13_save_end is 0 when R13 is not saved).
struct X64BaselinePrologue {
    uint32_t push_end = 0;
    uint32_t mov_end = 0;
    uint32_t alloc_end = 0;
    uint32_t r13_save_end = 0;
    int32_t frame_size = 0;
};

// Appends unwind data for the function in image[0, code_size) - Win64
// UNWIND_INFO + RUNTIME_FUNCTION, or a DWARF .eh_frame elsewhere - and
// returns the offset to pass to JitMemoryBlock::register_unwind_info. With
// scopes in `eh`, the unwind data names brass's personality routine (through
// a jump thunk in the image) and carries the scope table, so the OS unwinder
// lands a brass value at the frame's pads.
size_t append_x64_baseline_unwind(std::vector<uint8_t>& image, const X64BaselinePrologue& prologue,
                                  uint32_t code_size, bool windows,
                                  const runtime::FunctionExceptionTable& eh);

// Each returns true if it handled the opcode.
bool emit_baseline_x64_op(X64BaselineEmitter& emitter, const Instruction& inst);
// Integer operations on i8 / i16 operands (baseline_emit_narrow.cpp); false
// when `inst` is not one.
bool emit_baseline_x64_narrow(X64BaselineEmitter& emitter, const Instruction& inst);
bool emit_baseline_x64_fp_op(X64BaselineEmitter& emitter, const Instruction& inst);
// Vector opcodes, and select / loads / stores / ret of a vector value.
bool emit_baseline_x64_vec_op(X64BaselineEmitter& emitter, const Instruction& inst);

} // namespace brass::codegen
