#pragma once

#include <brass/target/x64/code_buffer.hpp>
#include <brass/target/x64/x64_encoder.hpp>
#include <brass/target/x64/x64_frame.hpp>
#include <brass/codegen/branch_relay.hpp>
#include <brass/codegen/lir.hpp>
#include <brass/codegen/unsupported_operation.hpp>
#include <brass/gc/stack_map.hpp>
#include <optional>
#include <vector>
#include <unordered_map>
#include <cstddef>
#include <cstdint>

#include <brass/runtime/resume_table.hpp>
#include <brass/runtime/patcher.hpp>
#include <brass/runtime/deopt.hpp>
#include <brass/runtime/exception.hpp>
#include <brass/debug/debug_section.hpp>

namespace brass::codegen {

// Code (function-relative [begin, end)) that runs with RSP `bytes` below
// the frame the prologue built: a guard exit's deopt record. The unwind
// info describes it with a chained entry, so a stack walk or exception
// dispatch passing through a call made there finds the caller's frame.
struct StackAdjustRegion {
    uint32_t begin = 0;
    uint32_t end = 0;
    uint32_t bytes = 0;
};

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
    runtime::FunctionResumeTable resume_table;
    std::vector<runtime::PatchSite> patch_sites;
    FunctionDebugTable debug_table;
    runtime::FunctionExceptionTable exception_table;
    std::vector<StackAdjustRegion> stack_adjust_regions;
};

class EmitContext {
public:
    EmitContext(const LirFunction& fn, const Target& target);

    CompilationResult compile();

private:
    const LirFunction& fn_;
    Target target_;
    codegen::FrameInfo frame_;
    x64::CodeBuffer buffer_;
    x64::X64Encoder enc_;
    std::vector<SafepointRecord> safepoints_;
    std::vector<StackAdjustRegion> stack_adjust_regions_;
    std::vector<StackMapRecord> stack_map_records_;
    std::unordered_map<uint32_t, x64::Label> block_labels_;
    std::unordered_map<uint8_t, int32_t> callee_gpr_to_slot_;
    // FrameInfo::constant_slots: slot -> its pool entry's label
    // (emit_constant_pool.cpp).
    std::unordered_map<int32_t, x64::Label> pool_labels_;
    // The constant slots whose value the host keeps at the pinned register
    // (LirFunction::pinned_constants): slot -> offset; read there, never pooled.
    std::unordered_map<int32_t, int32_t> pinned_slots_;
    std::vector<runtime::PatchSite> patch_sites_;

    struct PendingExceptionScope {
        size_t call_start = 0;
        size_t call_end = 0;
        uint32_t unwind_block_id = UINT32_MAX;
    };
    std::vector<PendingExceptionScope> pending_exception_scopes_;

    // The function writes 256-bit YMM registers, leaving their upper halves
    // dirty; then every call and return that does not pass a 256-bit value
    // clears them first (vzeroupper), as the x64 ABIs expect, or the
    // legacy-SSE code that runs next pays a merge on each instruction.
    bool dirties_upper_ = false;
    void clear_upper_state();
    CompilationResult compile_once();

    // Branch relaxation. A block-to-block branch whose target is already
    // placed (backward) takes the 2-byte form when it reaches; a forward one
    // takes it where `short_fwd_` (by the branch's ordinal in emission order)
    // says the first pass, all rel32, measured it in reach, and compile()
    // emits again with those short. Shrinking only brings a branch and its
    // target closer, but alignment padding can grow, so a short branch that
    // does not reach is reported (`overflow_ords_`) and emitted long once more.
    //
    // A long branch may instead relay through a nearby branch to the same
    // target (codegen/branch_relay.hpp): `relay_to_` names, by ordinal, the
    // branch it jumps onto in 2-byte form, and that branch binds a label at
    // its first byte. A forward relay out of reach is reported like a short
    // branch and emitted long once more; a backward one is measured exactly.
    // Both live in emit_branches.cpp.
    void emit_jump(x64::Condition cond, uint32_t target_id);
    CompilationResult relax_branches(CompilationResult first);
    struct ForwardBranch {
        uint32_t ordinal = 0;
        size_t end = 0;        // offset just past the branch
        x64::Label label;
    };
    std::vector<uint8_t> short_fwd_;
    std::vector<ForwardBranch> forward_branches_;
    std::unordered_map<size_t, uint32_t> short_patch_ord_;  // rel8 patch offset -> ordinal
    std::vector<uint32_t> overflow_ords_;
    uint32_t branch_ordinal_ = 0;
    std::vector<BranchSite> sites_;        // this emission's branches, by ordinal
    std::vector<uint8_t> block_cold_;      // by block id: LirBlock::cold
    bool emitting_cold_ = false;           // the block being emitted is cold
    // Where a branch to a block goes while blocks are emitted (emit()).
    const class BranchThreading* threading_ = nullptr;
    std::vector<uint32_t> relay_to_;       // the plan emitted: ordinal -> hub ordinal, or UINT32_MAX
    std::vector<x64::Label> hub_labels_;   // by ordinal, valid where a branch relays onto it

    x64::MemAddress to_mem_address(const LirOperand& op) const;
    x64::GPR to_gpr(const LirOperand& op) const;
    x64::XMM to_xmm(const LirOperand& op) const;

    void emit_instruction(const LirInst& inst, bool is_entry_block, bool is_first_inst);
    void emit_mov_instruction(const LirInst& inst);
    void emit_alu_instruction(const LirInst& inst);
    void emit_sse_instruction(const LirInst& inst);
    std::optional<x64::XMM> three_operand_src1(const LirInst& inst) const;
    void emit_vec_instruction(const LirInst& inst);
    void emit_vex_lanes_instruction(const LirInst& inst);
    void emit_parallel_copy(const LirInst& inst);
    void emit_control_instruction(const LirInst& inst);
    // The GC points (calls, safepoints): the callee-saved registers holding
    // live roots go to their designated slots before the call and come back
    // after it; the record at return_offset lists every live root (and
    // `sp`, for a safepoint, where each one is), plus every word of every
    // `alloca.tagged` buffer.
    void spill_callee_saved_roots(const LirInst& inst);
    void reload_callee_saved_roots(const LirInst& inst);
    void record_gc_point(const LirInst& inst, size_t return_offset, SafepointRecord* sp);
    // Zeroes the `alloca.tagged` buffers (after the prologue).
    void zero_tagged_locals();
    void begin_constant_pool();
    void emit_constant_pool();
    // The offset from the pinned register of `value` when this function may
    // read it there (LirFunction::pinned_constants), else none.
    std::optional<int32_t> pinned_constant(uint64_t value) const;
    // `mov dst, imm` as a read of the pinned copy when the immediate has no
    // shorter encoding than a movabs and the host keeps it; false otherwise.
    bool mov_pinned_constant(x64::GPR dst, int64_t imm);
    // Guard exit with no handler or resumer: calls the exit stub as
    // stub(state values...) from the deopt record at [rsp + slots_disp]. The
    // stack arguments use the outgoing area the guard exit allocated below
    // the record; RSP does not move.
    void emit_exit_stub_call(const LirInst& inst, int32_t slots_disp);
    void pad_invoke_return();
};

CompilationResult compile_lir_to_x64(const LirFunction& fn, const Target& target);

} // namespace brass::codegen
