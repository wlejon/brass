#pragma once

#include <brass/target/aarch64/code_buffer.hpp>
#include <brass/target/aarch64/aarch64_encoder.hpp>
#include <brass/target/aarch64/aarch64_frame.hpp>
#include <brass/codegen/lir.hpp>
#include <brass/codegen/emit_context.hpp>
#include <brass/gc/stack_map.hpp>
#include <vector>
#include <unordered_map>
#include <cstddef>
#include <cstdint>

#include <brass/runtime/resume_table.hpp>
#include <brass/runtime/patcher.hpp>
#include <brass/runtime/deopt.hpp>
#include <brass/runtime/exception.hpp>
#include <brass/debug/debug_section.hpp>

namespace brass::aarch64 {

struct AArch64CompilationResult {
    aarch64::CodeBuffer code_buffer;
    std::vector<codegen::SafepointRecord> safepoints;
    FunctionStackMap stack_map;
    size_t entry_offset = 0;
    size_t osr_entry_offset = 0;
    std::unordered_map<uint32_t, size_t> block_offsets;
    runtime::FunctionResumeTable resume_table;
    std::vector<runtime::PatchSite> patch_sites;
    FunctionDebugTable debug_table;
    runtime::FunctionExceptionTable exception_table;
};

class AArch64EmitContext {
public:
    AArch64EmitContext(const codegen::LirFunction& fn, const Target& target);

    AArch64CompilationResult compile();

private:
    const codegen::LirFunction& fn_;
    Target target_;
    aarch64::CodeBuffer buffer_;
    aarch64::AArch64Encoder enc_;
    std::vector<codegen::SafepointRecord> safepoints_;
    std::vector<StackMapRecord> stack_map_records_;
    std::unordered_map<uint32_t, aarch64::Label> block_labels_;
    std::vector<runtime::PatchSite> patch_sites_;

    struct PendingExceptionScope {
        size_t call_start = 0;
        size_t call_end = 0;
        uint32_t unwind_block_id = UINT32_MAX;
    };
    std::vector<PendingExceptionScope> pending_exception_scopes_;

    MemAddress to_mem_address(const codegen::LirOperand& op);
    MemAddress ensure_accessible_mem(const MemAddress& mem, GPR scratch = GPR::X16);
    GPR to_gpr(const codegen::LirOperand& op) const;
    FPR to_fpr(const codegen::LirOperand& op) const;

    void emit_instruction(const codegen::LirInst& inst, bool is_entry_block, bool is_first_inst);
    void emit_mov_instruction(const codegen::LirInst& inst);
    void emit_alu_instruction(const codegen::LirInst& inst);
    void emit_fp_instruction(const codegen::LirInst& inst);
    void emit_vec_instruction(const codegen::LirInst& inst);
    void emit_parallel_copy(const codegen::LirInst& inst);
    void emit_control_instruction(const codegen::LirInst& inst);
};

AArch64CompilationResult compile_lir_to_aarch64(const codegen::LirFunction& fn, const Target& target);

} // namespace brass::aarch64
