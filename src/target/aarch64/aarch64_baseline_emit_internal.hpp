#pragma once

#include <brass/target/aarch64/aarch64_baseline_emit.hpp>
#include <brass/target/aarch64/aarch64_encoder.hpp>
#include <brass/target/aarch64/code_buffer.hpp>
#include <brass/target/calling_conv.hpp>
#include <memory>
#include <unordered_map>
#include <string_view>
#include <vector>

namespace brass::aarch64 {

struct AArch64BaselineEmitter {
    CodeBuffer& buffer;
    AArch64Encoder& enc;
    Target target;
    const Function& fn;
    const std::unordered_map<const Value*, int32_t>& slot_map;
    const std::unordered_map<const Instruction*, int32_t>& alloca_offsets;
    const std::unordered_map<uint32_t, Label>& block_labels;
    FunctionStackMap& fn_stack_map;
    BaselineSymbolResolver resolver;
    int32_t frame_size = 0;
    // Lazy-link stubs for func_addr of a symbol that does not resolve yet
    // (null: none, and such a func_addr yields null).
    std::shared_ptr<codegen::LazySymbolTable> lazy;
    bool uses_lazy_stubs = false;

    MemAddress ensure_accessible_mem(const MemAddress& mem, GPR scratch = GPR::X16, int size_bytes = 8) const;
    MemAddress slot_addr(const Value* val) const;
    MemAddress slot_off_addr(int32_t off) const;

    void* resolve_sym(std::string_view name) const;
    // resolve_sym, else the symbol's lazy-link stub.
    void* resolve_or_stub(std::string_view name);
    void copy_block_args(const BranchTarget& target_branch);
};

// Returns true if the opcode was handled.
bool emit_baseline_aarch64_op(AArch64BaselineEmitter& emitter, const Instruction& inst);

} // namespace brass::aarch64
