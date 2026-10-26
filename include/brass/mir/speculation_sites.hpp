#pragma once

// Speculation sites: how a front end's lower tiers give tier 2 something to
// speculate on, and somewhere to deoptimize to.
//
// A fast path that is right only for some operands (two numbers, an inline
// cache hit, a builtin as the callee) is lowered as `br_if %hit, fast, slow`,
// where `slow` is the generic code. In a function that can deoptimize, the
// site also carries a GUARD in front of that branch: its condition is the
// constant 1 in the lower tiers' function, its state map every value live
// into `slow`, and `slow` is its resume target. Tier 0 and Tier 1 never fail
// it, so they run the branch; the guard is there so that tier-2 code has a
// guard of the same resume id to finish the call at, in Tier 0.
//
// The slow path counts its runs in the program's SpecFeedback. When a
// function tiers up, arm_speculation_sites (run by the front end's tier-2
// front pass, MultiTierPipeline::set_tier2_front_pass) reads the counts: a
// site the front end's policy calls monomorphic gets its guard armed with
// `%hit` and its branch pinned to `fast`, so the tier-2 code is the fast
// path alone and a miss deoptimizes to Tier 0; any other site drops its
// guard and keeps the branch. The copy drops every resume point either way
// (the lower tier holds the resume blocks), which leaves the slow blocks of
// armed sites unreachable.
//
// Kinds are the front end's: a small number naming what a site tests, which
// its policy reads with the site's count to decide whether to arm it.

#include <cstdint>
#include <deque>
#include <functional>
#include <mutex>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace brass {

class BasicBlock;
class Builder;
class Function;
class Instruction;
class Module;
class Value;

// One program's speculation sites: per site, its kind and how many times its
// slow path ran. The counters live as long as the object, at stable
// addresses the lowered code increments directly.
class SpecFeedback {
public:
    // Whether tier 2 should arm a site of `kind` whose slow path ran
    // `misses` times.
    using ArmPolicy = std::function<bool(uint8_t kind, uint32_t misses)>;
    explicit SpecFeedback(ArmPolicy policy);

    // `fn` is the site's function and `tag` what the site tests (a property
    // key, a helper): with the kind and the site's occurrence they name it,
    // for a front end that records sites (a profile).
    uint32_t add_site(uint8_t kind, std::string_view fn = {}, std::string_view tag = {});
    uint32_t* counter(uint32_t site);
    // Site `site`'s kind; nullopt for an id no site has.
    std::optional<uint8_t> kind_of(uint32_t site) const;
    // Whether tier 2 should arm site `site`'s guard: the policy's answer,
    // unless the site's guard has failed before.
    bool should_speculate(uint32_t site) const;
    // Site `site`'s guard failed often enough that tier 2 dropped the code
    // it was armed in: it is never armed again (its counts before did not
    // predict it, and arming it again would repeat the deopts). Tell it of
    // each invalidation (MultiTierPipeline::set_invalidation_observer).
    void mark_failed(uint32_t site);
    size_t size() const;

    struct SiteRecord {
        uint8_t kind = 0;
        uint32_t misses = 0;
        std::string fn;
        std::string tag;
    };
    // Every site, in the order added, with its count now.
    std::vector<SiteRecord> snapshot() const;
    SiteRecord site(uint32_t site) const;

private:
    ArmPolicy policy_;
    mutable std::mutex mutex_;
    std::deque<uint32_t> misses_;
    std::deque<uint8_t> failed_;
    std::deque<uint8_t> kinds_;
    std::deque<std::string> fns_;
    std::deque<std::string> tags_;
};

// Emits one lowering's sites, a function at a time.
class SpecSiteEmitter {
public:
    void set_feedback(SpecFeedback* feedback) { feedback_ = feedback; }
    SpecFeedback* feedback() const noexcept { return feedback_; }
    [[nodiscard]] bool has_feedback() const noexcept { return feedback_ != nullptr; }

    // Starts a function `name`; `can_deopt` false (or no feedback) lowers
    // every site as its branch, with its count.
    void begin_function(bool can_deopt, std::string_view name = {});
    // A body built again must be the body its code was compiled from: its
    // sites are the ones its first build added, ids and counters alike, not
    // new ones. Between begin_replay and end_replay the emitter hands out
    // sites first..first+count in order instead of adding sites; end_replay
    // says whether exactly those were taken. Outside a replay,
    // sites_taken() counts the sites added since begin_recording, and
    // first_site_taken() the first of them.
    void begin_replay(uint32_t first, uint32_t count);
    bool end_replay();
    void begin_recording();
    uint32_t sites_taken() const noexcept { return taken_; }
    uint32_t first_site_taken() const noexcept { return first_taken_; }
    // Whether the sites added since begin_recording have consecutive ids.
    bool sites_consecutive() const noexcept { return consecutive_; }

    // Ends the current block with the site's branch to `fast` / `slow` (and
    // its guard, where the function can deopt), then leaves the builder at
    // the end of `slow`, after its miss count. `slow` must be new and empty.
    // With `counter_out`, `slow` does not count: *counter_out is the
    // counter's address (0 without feedback), for a helper the slow path
    // calls anyway to increment.
    void emit_branch(Builder& b, Value* hit, BasicBlock* fast, BasicBlock* slow, uint8_t kind,
                     std::string_view tag = {}, Value** counter_out = nullptr);
    // Gives each guard of the function its state: the values live into its
    // resume block. Run once the function's CFG is final.
    void finish_function(Function& fn);

private:
    struct Pending {
        Instruction* guard;
        BasicBlock* slow;
    };
    // The next site's id: added to the feedback, or the replay's next.
    uint32_t take_site(uint8_t kind, std::string_view tag);

    SpecFeedback* feedback_ = nullptr;
    bool can_deopt_ = false;
    bool replaying_ = false;
    bool replay_ok_ = true;
    bool consecutive_ = true;
    uint32_t replay_first_ = 0;
    uint32_t replay_count_ = 0;
    uint32_t taken_ = 0;
    uint32_t first_taken_ = 0;
    std::string fn_name_;
    std::vector<Pending> pending_;
};

struct ArmSpeculationOptions {
    // Arm nothing: every guard is dropped (the guards-vs-branches A/B).
    bool arm_none = false;
    // 1: a line per function copy with guards (armed of total); 2: also a
    // line per armed site.
    int trace = 0;
    // Run on each function of the copy once its sites are decided.
    std::function<void(Function&)> after_function;
};

// The tier-2 front pass of a program lowered with `feedback`: arms each site
// guard of `mod`'s functions that the feedback says to, drops the others,
// and prunes the slow paths the armed sites leave unreachable.
void arm_speculation_sites(Module& mod, const SpecFeedback& feedback, const ArmSpeculationOptions& options = {});

} // namespace brass
