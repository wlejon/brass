#pragma once

#include <brass/target/x64/code_buffer.hpp>
#include <brass/target/x64/x64_encoder.hpp>
#include <brass/target/x64/x64_frame.hpp>
#include <brass/codegen/lir.hpp>
#include <brass/gc/stack_map.hpp>
#include <vector>
#include <unordered_map>
#include <cstddef>
#include <cstdint>

namespace brass::codegen {

struct SafepointRecord {
    size_t code_offset = 0;
    uint32_t safepoint_id = 0;
    std::vector<int32_t> live_gcref_spill_offsets;
    std::vector<x64::GPR> live_gcref_registers;
};

struct CompilationResult {
    x64::CodeBuffer code_buffer;
    std::vector<SafepointRecord> safepoints;
    FunctionStackMap stack_map;
    size_t entry_offset = 0;
    std::unordered_map<uint32_t, size_t> block_offsets;
};

class EmitContext {
public:
    EmitContext(const LirFunction& fn, const Target& target);

    CompilationResult compile();

private:
    const LirFunction& fn_;
    Target target_;
    x64::CodeBuffer buffer_;
    x64::X64Encoder enc_;
    std::vector<SafepointRecord> safepoints_;
    std::vector<StackMapRecord> stack_map_records_;
    std::unordered_map<uint32_t, x64::Label> block_labels_;

    x64::MemAddress to_mem_address(const LirOperand& op) const;
    x64::GPR to_gpr(const LirOperand& op) const;
    x64::XMM to_xmm(const LirOperand& op) const;

    void emit_instruction(const LirInst& inst, bool is_entry_block, bool is_first_inst);
};

CompilationResult compile_lir_to_x64(const LirFunction& fn, const Target& target);

} // namespace brass::codegen
