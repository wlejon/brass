#include <brass/mir/speculation_sites.hpp>

#include <brass/mir/block_liveness.hpp>
#include <brass/mir/builder.hpp>
#include <brass/mir/cfg_simplify.hpp>
#include <brass/mir/function.hpp>
#include <brass/mir/module.hpp>

#include <algorithm>
#include <cstdio>

namespace brass {

SpecFeedback::SpecFeedback(ArmPolicy policy) : policy_(std::move(policy)) {}

uint32_t SpecFeedback::add_site(uint8_t kind, std::string_view fn, std::string_view tag) {
    std::lock_guard<std::mutex> lock(mutex_);
    misses_.push_back(0);
    ran_.push_back(0);
    failed_.push_back(0);
    kinds_.push_back(kind);
    fns_.emplace_back(fn);
    tags_.emplace_back(tag);
    return static_cast<uint32_t>(misses_.size() - 1);
}

std::vector<SpecFeedback::SiteRecord> SpecFeedback::snapshot() const {
    std::lock_guard<std::mutex> lock(mutex_);
    std::vector<SiteRecord> out;
    out.reserve(misses_.size());
    for (size_t i = 0; i < misses_.size(); ++i) {
        const uint32_t misses = *static_cast<const volatile uint32_t*>(&misses_[i]);
        out.push_back({kinds_[i], misses, ran_locked(static_cast<uint32_t>(i)), fns_[i], tags_[i]});
    }
    return out;
}

SpecFeedback::SiteRecord SpecFeedback::site(uint32_t site) const {
    std::lock_guard<std::mutex> lock(mutex_);
    if (site >= misses_.size()) return {};
    const uint32_t misses = *static_cast<const volatile uint32_t*>(&misses_[site]);
    return {kinds_[site], misses, ran_locked(site), fns_[site], tags_[site]};
}

bool SpecFeedback::ran_locked(uint32_t site) const {
    if (!records_ran(kinds_[site])) return true;
    return *static_cast<const volatile uint32_t*>(&ran_[site]) != 0;
}

uintptr_t SpecFeedback::ran_flag_address(uint32_t site) const {
    std::lock_guard<std::mutex> lock(mutex_);
    if (site >= ran_.size() || !records_ran(kinds_[site])) return 0;
    return reinterpret_cast<uintptr_t>(&ran_[site]);
}

uint32_t* SpecFeedback::ran_flag(uint32_t site) {
    std::lock_guard<std::mutex> lock(mutex_);
    return site < ran_.size() ? &ran_[site] : nullptr;
}

std::optional<uint8_t> SpecFeedback::kind_of(uint32_t site) const {
    std::lock_guard<std::mutex> lock(mutex_);
    if (site >= kinds_.size()) return std::nullopt;
    return kinds_[site];
}

uint32_t* SpecFeedback::counter(uint32_t site) {
    std::lock_guard<std::mutex> lock(mutex_);
    return site < misses_.size() ? &misses_[site] : nullptr;
}

bool SpecFeedback::should_speculate(uint32_t site) const {
    uint8_t kind = 0;
    uint32_t misses = 0;
    bool ran = true;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        if (site >= misses_.size() || failed_[site]) return false;
        kind = kinds_[site];
        // Written by lowered code on any thread without a lock: a stale count
        // only makes the guess older, and a wrong guess costs a deopt.
        misses = *static_cast<const volatile uint32_t*>(&misses_[site]);
        ran = ran_locked(site);
    }
    return policy_ && policy_(kind, misses, ran);
}

void SpecFeedback::mark_failed(uint32_t site) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (site < failed_.size()) failed_[site] = 1;
}

size_t SpecFeedback::size() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return misses_.size();
}

void SpecSiteEmitter::begin_function(bool can_deopt, std::string_view name) {
    fn_name_ = name;
    can_deopt_ = can_deopt && feedback_ != nullptr;
    pending_.clear();
}

void SpecSiteEmitter::begin_replay(uint32_t first, uint32_t count) {
    replaying_ = true;
    replay_ok_ = true;
    replay_first_ = first;
    replay_count_ = count;
    taken_ = 0;
}

bool SpecSiteEmitter::end_replay() {
    replaying_ = false;
    return replay_ok_ && taken_ == replay_count_;
}

void SpecSiteEmitter::begin_recording() {
    taken_ = 0;
    first_taken_ = 0;
    consecutive_ = true;
}

uint32_t SpecSiteEmitter::take_site(uint8_t kind, std::string_view tag) {
    if (replaying_) {
        // A site past the first build's, or of another kind: the body is
        // not the one built first, which end_replay reports.
        const uint32_t i = taken_++;
        if (i >= replay_count_) {
            replay_ok_ = false;
            return feedback_->add_site(kind, fn_name_, tag);
        }
        const uint32_t site = replay_first_ + i;
        if (feedback_->kind_of(site) != kind) replay_ok_ = false;
        return site;
    }
    const uint32_t site = feedback_->add_site(kind, fn_name_, tag);
    if (taken_ == 0) first_taken_ = site;
    else if (site != first_taken_ + taken_) consecutive_ = false;
    ++taken_;
    return site;
}

void SpecSiteEmitter::emit_branch(Builder& b, Value* hit, BasicBlock* fast, BasicBlock* slow, uint8_t kind,
                                  std::string_view tag, Value** counter_out) {
    uint32_t* count = nullptr;
    if (feedback_) {
        const uint32_t site = take_site(kind, tag);
        count = feedback_->counter(site);
        if (can_deopt_) {
            // Reached: set right in front of the guard, where
            // arm_speculation_sites finds it and drops it from tier 2's copy.
            if (feedback_->records_ran(kind)) {
                Value* flag = b.build_iconst_i64(
                    static_cast<int64_t>(reinterpret_cast<uintptr_t>(feedback_->ran_flag(site))));
                b.build_store(Type::i32(), flag, 0, b.build_iconst_i32(1));
            }
            // The label names no function: the guard has no exit stub, only
            // its resume target.
            Instruction* guard = b.build_guard(b.build_iconst_i32(1), "brass.spec");
            guard->set_resume_id(site);
            b.current_block()->parent()->add_resume_point(site, slow);
            pending_.push_back({guard, slow});
        }
    }
    b.build_br_if(hit, fast, slow);
    b.position_at_end(slow);
    if (counter_out) {
        *counter_out = b.build_iconst_i64(static_cast<int64_t>(reinterpret_cast<uintptr_t>(count)));
        return;
    }
    if (count) {
        Value* addr = b.build_iconst_i64(static_cast<int64_t>(reinterpret_cast<uintptr_t>(count)));
        Value* old = b.build_load(Type::i32(), addr, 0);
        b.build_store(Type::i32(), addr, 0, b.build_add(old, b.build_iconst_i32(1)));
    }
}

void SpecSiteEmitter::finish_function(Function& fn) {
    if (pending_.empty()) return;
    std::vector<const BasicBlock*> targets;
    targets.reserve(pending_.size());
    for (const Pending& p : pending_) targets.push_back(p.slow);
    const auto live_ins = block_live_ins(fn, targets);
    for (const Pending& p : pending_) {
        auto it = live_ins.find(p.slow);
        if (it == live_ins.end()) continue;
        std::vector<Value*> state;
        state.reserve(it->second.size());
        for (const Value* v : it->second) state.push_back(const_cast<Value*>(v));
        std::sort(state.begin(), state.end(), [](const Value* a, const Value* c) { return a->id() < c->id(); });
        p.guard->state_map() = std::move(state);
    }
    pending_.clear();
}

namespace {

// The ran flag's store in front of `guard` (SpecSiteEmitter::emit_branch):
// tier 2 has no use for it, armed or not, and it is a store on every run.
void drop_ran_store(BasicBlock& bb, Instruction* guard, const SpecFeedback& feedback, uint32_t site) {
    const uintptr_t flag = feedback.ran_flag_address(site);
    if (flag == 0) return;
    Instruction* store = guard->prev();
    if (!store || store->opcode() != Opcode::store || store->operand_count() < 2) return;
    const Instruction* addr = store->operand(0) ? store->operand(0)->defining_instruction() : nullptr;
    if (!addr || addr->opcode() != Opcode::iconst_i64 || static_cast<uintptr_t>(addr->imm_i64()) != flag) return;
    bb.remove_instruction(store);
}

} // namespace

void arm_speculation_sites(Module& mod, const SpecFeedback& feedback, const ArmSpeculationOptions& options) {
    const size_t sites = feedback.size();
    for (Function* fn : mod.functions()) {
        if (!fn || fn->block_count() == 0) continue;
        std::vector<uint32_t> seen;
        uint32_t armed = 0;
        for (BasicBlock* bb : fn->blocks()) {
            if (!bb) continue;
            Instruction* guard = nullptr;
            for (Instruction* inst : *bb) {
                if (inst && inst->opcode() == Opcode::guard) guard = inst;
            }
            if (!guard || guard->resume_id() >= sites) continue;
            Instruction* br = guard->next();
            Value* placeholder = guard->operand(0);
            const Instruction* def = placeholder ? placeholder->defining_instruction() : nullptr;
            if (!br || br->opcode() != Opcode::br_if || !def || def->opcode() != Opcode::iconst_i32 ||
                def->imm_i32() != 1) {
                continue;
            }
            const uint32_t site = guard->resume_id();
            seen.push_back(site);
            drop_ran_store(*bb, guard, feedback, site);
            if (!options.arm_none && feedback.should_speculate(site)) {
                // The guard takes the hit test; the branch always goes fast.
                guard->set_operand(0, br->operand(0));
                br->set_operand(0, placeholder);
                ++armed;
                if (options.trace >= 2) {
                    const SpecFeedback::SiteRecord r = feedback.site(site);
                    std::fprintf(stderr, "spec-site %u kind=%u misses=%u fn=%s tag=%s\n", site,
                                 static_cast<unsigned>(r.kind), r.misses, r.fn.c_str(), r.tag.c_str());
                }
            } else {
                bb->remove_instruction(guard);
            }
        }
        for (uint32_t site : seen) fn->remove_resume_point(site);
        if (armed != 0) {
            // A pinned branch leaves its slow side unreachable. Dropped here,
            // before the tier-2 passes, so inlining, GVN and GVN-PRE never
            // walk it.
            CfgSimplifyOptions prune;
            prune.enable_block_merge = false;
            prune.enable_param_elimination = false;
            prune.enable_trampoline_elimination = false;
            prune.max_iterations = 2;
            cfg_simplify_function(*fn, prune);
        }
        if (options.after_function) options.after_function(*fn);
        if (options.trace && !seen.empty()) {
            std::fprintf(stderr, "spec: %.*s armed %u of %zu\n", static_cast<int>(fn->name().size()),
                         fn->name().data(), armed, seen.size());
        }
    }
}

} // namespace brass
