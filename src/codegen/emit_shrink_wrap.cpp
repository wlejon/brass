// The x64 emitter's half of shrink-wrapping (codegen/shrink_wrap.hpp): which
// blocks of the allocated code run with the frame, where it is built and
// torn down, and the frame states the unwind info describes.
#include <brass/codegen/emit_context.hpp>
#include <brass/codegen/branch_threading.hpp>
#include <brass/codegen/shrink_wrap.hpp>
#include <algorithm>

namespace brass::codegen {

using namespace brass::x64;

namespace {

constexpr uint32_t kStubBase = 0x40000000u;
constexpr uint32_t kUndef = UINT32_MAX;

bool reads_flags(LirOpcode op) {
    return op == LirOpcode::Jcc || op == LirOpcode::Setcc || op == LirOpcode::Cmovcc;
}

bool writes_flags(LirOpcode op) {
    switch (op) {
        case LirOpcode::Cmp: case LirOpcode::Cmp32: case LirOpcode::Test: case LirOpcode::Test32:
        case LirOpcode::Ucomisd: case LirOpcode::Ucomiss:
        case LirOpcode::Add: case LirOpcode::Add32: case LirOpcode::Sub: case LirOpcode::Sub32:
        case LirOpcode::And: case LirOpcode::And32: case LirOpcode::Or: case LirOpcode::Or32:
        case LirOpcode::Xor: case LirOpcode::Xor32:
            return true;
        default:
            return false;
    }
}

// Whether a definition writes all of a general register (no merge with what
// it held). Only plain writes count; anything else is taken to read it too.
bool writes_whole(LirOpcode op) {
    switch (op) {
        case LirOpcode::Mov: case LirOpcode::Mov32: case LirOpcode::Movabs:
        case LirOpcode::Movzx8: case LirOpcode::Movzx16: case LirOpcode::Movsx8: case LirOpcode::Movsx16:
        case LirOpcode::Movsxd: case LirOpcode::Lea: case LirOpcode::ParallelCopy: case LirOpcode::Movq_xg:
        case LirOpcode::Pop:
            return true;
        default:
            return false;
    }
}

uint32_t gpr_bit(const PReg& p) {
    return p.is_gpr() && p.code < 16 ? 1u << p.code : 0u;
}

// The registers of `mask` an instruction reads, and those it writes whole.
void reg_effects(const LirInst& inst, uint32_t mask, uint32_t& reads, uint32_t& kills) {
    reads = kills = 0;
    // Two writes of a whole register that name it as a source but do not
    // read it: `xor r, r` (zeroing) and `imul dst, src, imm`.
    if (inst.defs.size() == 1 && inst.defs[0].is_preg() && inst.defs[0].size >= 4) {
        const PReg d = inst.defs[0].preg_val;
        const LirOpcode op = inst.opcode;
        bool whole = false;
        if (op == LirOpcode::Xor || op == LirOpcode::Xor32 || op == LirOpcode::Sub || op == LirOpcode::Sub32) {
            whole = !inst.uses.empty();
            for (const auto& u : inst.uses) whole = whole && u.is_preg() && u.preg_val == d;
        } else if ((op == LirOpcode::Imul || op == LirOpcode::Imul32) && inst.uses.size() == 2 &&
                   inst.uses[0].is_preg() && inst.uses[0].preg_val != d && inst.uses[1].is_imm_int()) {
            whole = true;
            reads = gpr_bit(inst.uses[0].preg_val) & mask;
        }
        if (whole) {
            kills = gpr_bit(d) & mask;
            return;
        }
    }
    auto mem_reads = [&](const LirOperand& op) {
        if (op.is_mem()) reads |= gpr_bit(op.mem_val.base_preg) | gpr_bit(op.mem_val.index_preg);
    };
    for (const auto& op : inst.uses) {
        if (op.is_preg()) reads |= gpr_bit(op.preg_val);
        mem_reads(op);
    }
    for (const auto& op : inst.defs) {
        mem_reads(op);
        if (!op.is_preg()) continue;
        if (writes_whole(inst.opcode) && op.size >= 4) kills |= gpr_bit(op.preg_val);
        else reads |= gpr_bit(op.preg_val);
    }
    reads &= mask;
    kills &= mask;
}

} // namespace

void EmitContext::plan_shrink_wrap(const BranchThreading& threading) {
    wrapped_ = false;
    cur_framed_ = true;
    framed_.clear();
    save_point_.clear();
    exits_frame_.clear();
    pos_of_id_.clear();
    stub_targets_.clear();
    stub_of_.clear();
    frame_rows_.clear();
    if (frame_.is_leaf || !shrink_wrap_shape_ok(fn_, fn_.calling_conv)) return;

    const size_t n = fn_.blocks.size();
    for (size_t i = 0; i < n; ++i) pos_of_id_[fn_.blocks[i]->id] = i;
    // Nodes: the blocks the emitter emits, in order.
    std::vector<uint32_t> node_of(n, kUndef), pos_of;
    for (size_t i = 0; i < n; ++i) {
        if (threading.skipped(i)) continue;
        node_of[i] = static_cast<uint32_t>(pos_of.size());
        pos_of.push_back(static_cast<uint32_t>(i));
    }
    if (pos_of.empty() || pos_of[0] != 0) return;
    const size_t nodes = pos_of.size();
    auto node_for_id = [&](uint32_t id) -> uint32_t {
        auto it = pos_of_id_.find(id);
        return it == pos_of_id_.end() ? kUndef : node_of[it->second];
    };

    const uint32_t mask = frame_gpr_mask(fn_, fn_.calling_conv) | frame_.saved_callee_gprs;
    std::vector<int32_t> free_slots;
    for (const auto& [slot, value] : frame_.constant_slots) free_slots.push_back(slot);
    std::sort(free_slots.begin(), free_slots.end());

    FrameGraph g;
    g.succs.resize(nodes);
    g.cold.assign(nodes, 0);
    g.needs_frame.assign(nodes, 0);
    g.returns.assign(nodes, 0);
    std::vector<uint32_t> gen(nodes, 0), kill(nodes, 0);
    std::vector<uint8_t> flags_in(nodes, 0);  // reads flags it did not set
    for (uint32_t k = 0; k < nodes; ++k) {
        const size_t i = pos_of[k];
        const LirBlock& b = *fn_.blocks[i];
        const ClosingBranches closing = closing_branches(fn_, i);
        const size_t limit = b.instructions.size() - closing.count;
        bool flags_set = false;
        auto add_succ = [&](uint32_t id) {
            const uint32_t t = node_for_id(id);
            if (t == kUndef) return false;
            g.succs[k].push_back(t);
            return true;
        };
        for (size_t x = 0; x < b.instructions.size(); ++x) {
            const LirInst& inst = *b.instructions[x];
            if (x < limit && inst.is_branch()) {
                if (inst.uses.empty() || !inst.uses[0].is_label() || !add_succ(inst.uses[0].label_id)) return;
            }
            g.needs_frame[k] |= inst_needs_frame(inst, mask, free_slots);
            g.returns[k] |= inst.opcode == LirOpcode::Ret;
            if (reads_flags(inst.opcode) && !flags_set) flags_in[k] = 1;
            flags_set = flags_set || writes_flags(inst.opcode);
            uint32_t reads = 0, kills = 0;
            reg_effects(inst, mask, reads, kills);
            gen[k] |= reads & ~kill[k];
            kill[k] |= kills;
        }
        if (closing.count > 0) {
            if (closing.conditional && !add_succ(threading.target(closing.taken))) return;
            if (!add_succ(threading.target(closing.other))) return;
        } else {
            const auto& ins = b.instructions;
            const LirOpcode last = ins.empty() ? LirOpcode::Nop : ins.back()->opcode;
            if (last != LirOpcode::Ret && last != LirOpcode::Trap && last != LirOpcode::GuardExit) {
                if (k + 1 >= nodes) return;
                g.succs[k].push_back(k + 1);
            }
        }
        g.cold[k] = b.cold;
    }

    const FrameRegions regions = choose_frame_regions(g);
    if (!regions.wrapped) return;

    // No callee-saved register (nor rbp) may carry a value into a save
    // point: code without a frame names none, so such a value would cross
    // the teardown of another region and come back as the caller's.
    std::vector<uint32_t> live_in(nodes, 0);
    for (bool changed = true; changed;) {
        changed = false;
        for (size_t k = nodes; k-- > 0;) {
            uint32_t out = 0;
            for (uint32_t s : g.succs[k]) out |= live_in[s];
            const uint32_t in = gen[k] | (out & ~kill[k]);
            if (in != live_in[k]) {
                live_in[k] = in;
                changed = true;
            }
        }
    }
    for (uint32_t s : regions.saves) {
        // The prologue's `sub rsp` writes the flags.
        if (live_in[s] != 0 || flags_in[s]) return;
    }

    framed_.assign(n, 0);
    save_point_.assign(n, 0);
    exits_frame_.assign(n, 0);
    for (uint32_t k = 0; k < nodes; ++k) {
        // A block the entry does not reach never runs; it keeps the frame
        // it would need.
        framed_[pos_of[k]] = regions.region[k] != FrameRegions::kNone || g.needs_frame[k];
    }
    for (uint32_t s : regions.saves) save_point_[pos_of[s]] = 1;
    wrapped_ = true;
    for (uint32_t k = 0; k < nodes; ++k) {
        if (!framed_[pos_of[k]]) continue;
        for (uint32_t s : g.succs[k]) {
            if (!framed_at(fn_.blocks[pos_of[s]]->id)) exits_frame_[pos_of[k]] = 1;
        }
    }
    cur_framed_ = false;
}

bool EmitContext::framed_at(uint32_t block_id) const {
    // A save point is entered without the frame: its prologue builds it.
    auto it = pos_of_id_.find(block_id);
    return it == pos_of_id_.end() || (framed_[it->second] != 0 && save_point_[it->second] == 0);
}

uint32_t EmitContext::edge_target(uint32_t block_id) {
    if (!wrapped_ || !cur_framed_ || framed_at(block_id)) return block_id;
    auto it = stub_of_.find(block_id);
    if (it != stub_of_.end()) return it->second;
    const uint32_t stub = kStubBase + static_cast<uint32_t>(stub_targets_.size());
    stub_targets_.push_back(block_id);
    stub_of_[block_id] = stub;
    block_labels_[stub] = buffer_.create_label();
    return stub;
}

void EmitContext::note_frame_state(FrameState state, uint32_t lowered) {
    note_frame_state_at(buffer_.size(), state, lowered);
}

void EmitContext::note_frame_state_at(size_t offset, FrameState state, uint32_t lowered) {
    if (!wrapped_) return;
    auto same = [&](const FrameRow& r) { return r.state == state && r.lowered == lowered; };
    if (!frame_rows_.empty() && frame_rows_.back().offset == offset) {
        frame_rows_.pop_back();
    }
    const FrameRow before = frame_rows_.empty() ? FrameRow{} : frame_rows_.back();
    if (same(before)) return;
    frame_rows_.push_back(FrameRow{static_cast<uint32_t>(offset), state, lowered});
}

void EmitContext::enter_block_frame_state(size_t block_pos) {
    if (!wrapped_) return;
    if (save_point_[block_pos]) {
        note_frame_state(FrameState::Frameless);
        const size_t at = buffer_.size();
        X64FrameLayout::emit_prologue(enc_, frame_, fn_.calling_conv);
        // push rbp is one byte, mov rbp, rsp three.
        note_frame_state_at(at + 1, FrameState::Pushed);
        note_frame_state_at(at + 4, FrameState::Framed);
        cur_framed_ = true;
        return;
    }
    cur_framed_ = framed_[block_pos] != 0;
    note_frame_state(cur_framed_ ? FrameState::Framed : FrameState::Frameless);
}

void EmitContext::emit_frame_exit() {
    X64FrameLayout::emit_teardown(enc_, frame_, true);
    note_frame_state(FrameState::Frameless);
    cur_framed_ = false;
}

void EmitContext::emit_frame_exit_stubs() {
    for (size_t k = 0; k < stub_targets_.size(); ++k) {
        buffer_.bind(block_labels_[kStubBase + static_cast<uint32_t>(k)]);
        cur_framed_ = true;
        note_frame_state(FrameState::Framed);
        emit_frame_exit();
        emit_jump(Condition::None, stub_targets_[k]);
    }
}

void EmitContext::emit_return_epilogue() {
    if (wrapped_ && !cur_framed_) {
        enc_.ret();
        return;
    }
    X64FrameLayout::emit_epilogue(enc_, frame_, fn_.calling_conv);
    if (wrapped_) {
        // The `ret` runs with rbp popped; what follows it still has the frame.
        note_frame_state_at(buffer_.size() - 1, FrameState::Frameless);
        note_frame_state(FrameState::Framed);
    }
}

bool EmitContext::lower_rsp_for(const LirInst& inst) {
    if (!wrapped_ || cur_framed_ || !reads_rsp(inst)) return false;
    const auto k = static_cast<int32_t>(8 + frame_.total_frame_size);
    enc_.lea(GPR::RSP, ptr(GPR::RSP, -k));
    note_frame_state(FrameState::Lowered, static_cast<uint32_t>(k));
    return true;
}

void EmitContext::raise_rsp() {
    const auto k = static_cast<int32_t>(8 + frame_.total_frame_size);
    enc_.lea(GPR::RSP, ptr(GPR::RSP, k));
    note_frame_state(FrameState::Frameless);
}

} // namespace brass::codegen
