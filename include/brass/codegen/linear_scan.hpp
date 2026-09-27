#pragma once

#include <brass/codegen/lir.hpp>
#include <brass/codegen/live_range.hpp>
#include <brass/target/calling_conv.hpp>
#include <vector>
#include <unordered_map>
#include <cstdint>

namespace brass::codegen {

struct InstConstraints {
    uint32_t clobbered_gprs = 0;
    uint32_t clobbered_xmms = 0;
    uint32_t pinned_gprs = 0;
    uint32_t pinned_xmms = 0;
    constexpr bool empty() const noexcept {
        return (clobbered_gprs | clobbered_xmms | pinned_gprs | pinned_xmms) == 0;
    }
    InstConstraints& operator|=(const InstConstraints& o) noexcept {
        clobbered_gprs |= o.clobbered_gprs;
        clobbered_xmms |= o.clobbered_xmms;
        pinned_gprs |= o.pinned_gprs;
        pinned_xmms |= o.pinned_xmms;
        return *this;
    }
    constexpr InstConstraints operator|(const InstConstraints& o) const noexcept {
        return InstConstraints{
            clobbered_gprs | o.clobbered_gprs,
            clobbered_xmms | o.clobbered_xmms,
            pinned_gprs | o.pinned_gprs,
            pinned_xmms | o.pinned_xmms
        };
    }
};

// The allocator splits live intervals (Wimmer & Mössenböck, "Optimized
// Interval Splitting in a Linear Scan Register Allocator"): a value lives in
// a register where one is free for it and in its spill slot elsewhere, so a
// hot loop keeps its own values in registers while a value it does not use
// waits in memory, instead of one location for the value's whole life.
//
// An interval is cut into pieces, each a window [from, to] of instruction
// positions over the liveness interval, with one location each. Instructions
// sit at even positions; a piece ends and the next begins at an odd one, the
// point between two instructions where the move from one location to the
// other goes. A piece that ends at a block boundary is joined to the next by
// moves on the control-flow edges instead (linear_scan_resolve.cpp).
struct AllocPiece {
    const LiveInterval* base = nullptr;
    uint32_t from = 0;          // window, inclusive
    uint32_t to = UINT32_MAX;   // window, inclusive
    uint32_t start = 0;         // first position live inside the window
    uint32_t end = 0;           // last position live inside the window
    PReg reg;                   // or spilled
    bool spilled = false;
    uint32_t seq = 0;           // creation order, a deterministic tie-break
    VReg vreg() const noexcept { return base->vreg; }
};

class LinearScanAllocator {
public:
    LinearScanAllocator(
        LirFunction& fn,
        LivenessAnalysis& liveness,
        const CallingConvention& cc
    );

    void allocate();

    uint32_t used_callee_saved_gprs() const noexcept { return used_callee_gprs_; }
    uint32_t used_callee_saved_xmms() const noexcept { return used_callee_xmms_; }
    size_t num_spill_slots() const noexcept { return next_spill_slot_; }

    // Where `v` lives at instruction position `id` (the piece covering it):
    // a register operand, a spill slot operand, or none when v is not live.
    LirOperand location_at(VReg v, uint32_t id, uint8_t size) const;

private:
    LirFunction& fn_;
    LivenessAnalysis& liveness_;
    CallingConvention cc_;

    std::vector<PReg> available_gprs_;
    std::vector<PReg> available_xmms_;

    uint32_t used_callee_gprs_ = 0;
    uint32_t used_callee_xmms_ = 0;
    size_t next_spill_slot_ = 0;

    std::unordered_map<uint32_t, std::vector<VReg>> coalesce_hints_;
    std::unordered_map<uint32_t, std::vector<PReg>> fixed_preg_hints_;
    std::unordered_multimap<uint64_t, PReg> vreg_at_preg_hints_;

    // Per register class, the instructions that can block a register for a
    // value live across them, in id order: `clob` are clobbers, registers
    // held between a write and a read (compute_held_pregs) and the
    // caller-saved registers at every call and safepoint; `pin` are
    // physical-register operands and fixed constraints, which a value whose
    // own fixed position it is may take; `call` is every register at a call,
    // for a vector no callee-saved register holds in full. constraint_or_
    // [class][wide] is a sparse table over clob | pin (| call when wide):
    // level k, index i holds the OR of [i, i + 2^k), so the first
    // instruction blocking a set of registers in a range is a descent, not a
    // walk.
    struct ConstraintWord {
        uint32_t clob = 0;
        uint32_t pin = 0;
        uint32_t call = 0;
    };
    std::vector<uint32_t> constrained_ids_[2];
    std::vector<ConstraintWord> constraint_words_[2];
    std::vector<std::vector<uint32_t>> constraint_or_[2][2];
    // Distinct vregs that appear as the index register of a memory operand.
    std::vector<VReg> mem_index_vregs_;
    // Guard exit block id -> the one Jcc position entering it (the place its
    // GuardExit reads its state from).
    std::unordered_map<uint32_t, uint32_t> guard_exit_at_;

    // Pieces, and per vreg id the indices of its pieces in window order.
    std::vector<AllocPiece> pieces_;
    std::vector<std::vector<uint32_t>> vreg_pieces_;
    std::vector<int32_t> vreg_slot_;
    // Values that keep one location for their whole life: GC references
    // (stack maps name one place per value), and values live into a block
    // entered other than by an ordinary branch (a landing pad, a resume
    // point, an edge no move can be placed on).
    std::vector<uint8_t> unsplittable_;
    std::vector<uint8_t> is_mem_index_;
    // Per vreg, its one definition's position (UINT32_MAX: several or none).
    std::vector<uint32_t> def_pos_;
    std::vector<uint32_t> active_;   // piece indices holding a register
    uint32_t next_seq_ = 0;

    struct BlockSpan {
        uint32_t start = 0;
        uint32_t end = 0;
        uint32_t depth = 0;      // loop depth of the block
        uint32_t entry_depth = 0; // loop depth of entering it in layout order
        // The run of blocks around it at its depth or deeper, first start
        // and last end: its innermost loop, the layout keeping loops
        // contiguous (LirFunction::sort_blocks_rpo).
        uint32_t loop_from = 0;
        uint32_t loop_to = 0;
    };
    std::vector<BlockSpan> spans_;   // non-empty blocks, by start position
    // Range-minimum over spans_[i].entry_depth: level k, index i holds the
    // index in [i, i + 2^k) of the least depth, the last one among equals.
    std::vector<std::vector<uint32_t>> depth_rmq_;

    void init_register_pools();
    void build_coalesce_hints();
    void build_constraint_index();
    void build_block_spans();
    // The position by which a holder of a register needs its value again,
    // seen from `pos`: its next use, or, for one read earlier in the loop
    // around `pos` and live at its end, that end (the next iteration).
    uint32_t needed_again(const AllocPiece& holder, uint32_t pos) const;
    void compute_held_pregs(const LirBlock& block, std::vector<InstConstraints>& out) const;
    // True for a vector value that no callee-saved register holds in full:
    // AAPCS64 preserves only the low 64 bits of V8..V15, Win64 only the low
    // 128 bits of XMM6..XMM15. Such a value is spilled across a call.
    bool fpr_wider_than_callee_saved(const VReg& vreg) const;
    bool is_callee_saved(PReg reg) const;
    void mark_callee_saved(PReg reg);
    int32_t allocate_spill_slot(bool is_gcref, uint8_t size);
    int32_t allocate_spill_slot(bool is_gcref) { return allocate_spill_slot(is_gcref, 8); }
    int32_t slot_of(VReg v);

    // linear_scan_pieces.cpp: the interval-splitting scan.
    void run_scan();
    uint32_t new_piece(const LiveInterval* base, uint32_t from, uint32_t to);
    bool piece_covers(const AllocPiece& p, uint32_t id) const;
    uint32_t next_use(const AllocPiece& p, uint32_t pos, bool reg_only) const;
    uint32_t last_use_at_or_before(const AllocPiece& p, uint32_t pos) const;
    uint32_t first_intersection(const AllocPiece& a, const AllocPiece& b) const;
    // The last position a piece may reach while holding `reg`: one before
    // the first instruction in it that clobbers or pins the register
    // (UINT32_MAX when none does).
    void reg_limits(const AllocPiece& p, uint32_t out[32]) const;
    size_t first_constrained(int cls, bool wide, size_t i, size_t end, uint32_t want) const;
    void evict_from(uint32_t idx, uint32_t pos, std::vector<uint32_t>& unhandled);
    uint32_t own_fixed_regs(const AllocPiece& p, uint32_t pos) const;
    uint32_t always_blocked(const AllocPiece& p) const;
    uint32_t split_position(uint32_t lo, uint32_t hi) const;
    uint32_t split_piece(uint32_t idx, uint32_t q);
    void spill_and_requeue(uint32_t idx, uint32_t after, std::vector<uint32_t>& unhandled);
    void push_unhandled(std::vector<uint32_t>& unhandled, uint32_t idx) const;
    void assign(uint32_t idx, PReg reg);
    bool try_allocate_free(uint32_t idx, std::vector<uint32_t>& unhandled);
    void allocate_blocked(uint32_t idx, std::vector<uint32_t>& unhandled);
    std::vector<PReg> hint_regs(const AllocPiece& p) const;

    // linear_scan_resolve.cpp: moves where a value changes location, between
    // two pieces inside a block (before the instruction at the next
    // position) or on a control-flow edge (at the end of its source, at the
    // start of its target, or in a block of its own when the edge is
    // critical). Each list is one parallel move.
    struct PendingMove {
        LirOperand dst;
        LirOperand src;
        bool is_xmm = false;
        uint8_t size = 8;
    };
    struct EdgeSplit {
        LirBlock* pred = nullptr;
        LirBlock* succ = nullptr;
        std::vector<PendingMove> moves;
    };
    enum class EdgePlace : uint8_t { EndOfPred, StartOfSucc, Split, None };
    std::unordered_map<uint32_t, std::vector<PendingMove>> moves_before_;   // inst id
    std::unordered_map<uint32_t, std::vector<PendingMove>> moves_after_;    // inst id
    std::unordered_map<uint32_t, std::vector<PendingMove>> moves_at_start_; // block id
    std::unordered_map<uint32_t, std::vector<PendingMove>> moves_at_end_;   // block id
    std::vector<EdgeSplit> edge_splits_;
    std::vector<uint32_t> landing_pads_;
    std::vector<uint32_t> resume_targets_;
    bool emitted_parallel_copy_ = false;
    bool is_guard_exit(const LirBlock* b) const;
    bool is_entered_implicitly(const LirBlock* b) const;
    EdgePlace classify_edge(const LirBlock& pred, const LirBlock& succ, const LirBlock* next_in_order,
                            size_t succ_pred_count) const;
    void mark_unsplittable_live_ins();
    void collect_resolution_moves();
    void emit_moves(std::vector<PendingMove>& moves, std::vector<std::unique_ptr<LirInst>>& out);
    void split_critical_edges();

    void rewrite_instructions();
    // Fill live_gcrefs of every call and safepoint with the gcref vregs live
    // across it (live after the site and not defined by it).
    void record_live_gcrefs();
};

void run_linear_scan_regalloc(
    LirFunction& fn,
    LivenessAnalysis& liveness,
    const CallingConvention& cc
);

} // namespace brass::codegen
