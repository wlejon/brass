// Block-to-block branches of the x64 emitter: their 2-byte forms where they
// reach (relaxation), and relays through a nearby branch to the same target
// where they do not (codegen/branch_relay.hpp).
#include <brass/codegen/emit_context.hpp>

namespace brass::codegen {

using namespace brass::x64;

namespace {
// How far a relay may reach on the emission it is planned from: a rel8 reaches
// 127 bytes ahead and 128 back, and the margin covers what alignment padding
// may add between the two branches when the code around them shrinks.
constexpr int64_t kRelayReach = 104;
}  // namespace

CompilationResult EmitContext::relax_branches(CompilationResult first) {
    // Branch relaxation: the forward branches the all-rel32 pass measured in
    // reach of a 2-byte form, emitted short; any that alignment padding put
    // out of reach after all goes long and the function is emitted again.
    std::vector<uint8_t> fits(branch_ordinal_, 0);
    bool any = false;
    for (const ForwardBranch& fb : forward_branches_) {
        const int64_t disp = static_cast<int64_t>(first.code_buffer.label_offset(fb.label)) -
                             static_cast<int64_t>(fb.end);
        if (disp <= 127) fits[fb.ordinal] = 1, any = true;
    }
    // The rest that are long relay where a branch to their target is near.
    std::vector<BranchSite> planned = sites_;
    for (uint32_t i = 0; i < planned.size(); ++i) {
        if (fits[i]) planned[i].is_long = false;
    }
    std::vector<uint32_t> relay = plan_branch_relays(planned, kRelayReach);
    for (uint32_t r : relay) any = any || r != UINT32_MAX;
    for (int attempt = 0; any && attempt < 4; ++attempt) {
        EmitContext again(fn_, target_);
        again.dirties_upper_ = dirties_upper_;
        again.short_fwd_ = fits;
        again.relay_to_ = relay;
        again.hub_labels_.assign(relay.size(), Label{});
        for (uint32_t h : relay) {
            if (h != UINT32_MAX && !again.hub_labels_[h].is_valid()) again.hub_labels_[h] = again.buffer_.create_label();
        }
        again.buffer_.collect_rel8_overflows(true);
        CompilationResult shorter = again.compile_once();
        if (again.overflow_ords_.empty()) return shorter;
        for (uint32_t ord : again.overflow_ords_) {
            if (relay[ord] != UINT32_MAX) relay[ord] = UINT32_MAX;
            else fits[ord] = 0;
        }
    }
    return first;
}

void EmitContext::emit_jump(Condition cond, uint32_t target_id) {
    const uint32_t ordinal = branch_ordinal_++;
    const Label label = block_labels_[target_id];
    const size_t start = buffer_.size();
    if (ordinal < hub_labels_.size() && hub_labels_[ordinal].is_valid()) buffer_.bind(hub_labels_[ordinal]);
    auto emit_short = [&](Label to) {
        if (cond == Condition::None) enc_.jmp_short(to);
        else enc_.j_short(cond, to);
    };
    auto record = [&](bool is_long) {
        sites_.push_back(BranchSite{cond, target_id, start, is_long});
    };
    if (buffer_.is_bound(label)) {
        // Backward: the distance is known exactly.
        const int64_t disp = static_cast<int64_t>(buffer_.label_offset(label)) - static_cast<int64_t>(start + 2);
        if (disp >= -128) {
            emit_short(label);
            record(false);
            return;
        }
    } else if (ordinal < short_fwd_.size() && short_fwd_[ordinal]) {
        emit_short(label);
        short_patch_ord_[buffer_.size() - 1] = ordinal;
        record(false);
        return;
    }
    if (ordinal < relay_to_.size() && relay_to_[ordinal] != UINT32_MAX) {
        const Label hub = hub_labels_[relay_to_[ordinal]];
        if (!buffer_.is_bound(hub)) {
            emit_short(hub);
            short_patch_ord_[buffer_.size() - 1] = ordinal;
            record(false);
            return;
        }
        const int64_t disp = static_cast<int64_t>(buffer_.label_offset(hub)) - static_cast<int64_t>(start + 2);
        if (disp >= -128) {
            emit_short(hub);
            record(false);
            return;
        }
    }
    if (cond == Condition::None) enc_.jmp_near(label);
    else enc_.j_near(cond, label);
    record(true);
    if (!buffer_.is_bound(label)) forward_branches_.push_back(ForwardBranch{ordinal, buffer_.size(), label});
}

} // namespace brass::codegen
