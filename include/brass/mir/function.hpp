#pragma once

#include <brass/mir/types.hpp>
#include <brass/mir/block.hpp>
#include <brass/core/arena.hpp>
#include <atomic>
#include <memory>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>
#include <cstdint>
#include <utility>

namespace brass {

class Module;

class Function {
public:
    Function() noexcept = default;
    Function(std::string_view name, Type return_type, std::vector<Type> param_types)
        : name_(name), return_type_(return_type), param_types_(std::move(param_types)) {}

    std::string_view name() const noexcept { return name_; }
    void set_name(std::string_view name) noexcept { name_ = name; }

    Type return_type() const noexcept { return return_type_; }
    void set_return_type(Type t) noexcept { return_type_ = t; }

    const std::vector<Type>& param_types() const noexcept { return param_types_; }
    void set_param_types(std::vector<Type> types) { param_types_ = std::move(types); }
    size_t param_count() const noexcept { return param_types_.size(); }
    Type param_type(size_t idx) const noexcept {
        return (idx < param_types_.size()) ? param_types_[idx] : Type::void_type();
    }

    bool is_param_noalias(size_t idx) const noexcept {
        return idx < param_noalias_.size() && param_noalias_[idx];
    }
    void set_param_noalias(size_t idx, bool noalias = true) {
        if (idx >= param_noalias_.size()) {
            param_noalias_.resize(idx + 1, false);
        }
        param_noalias_[idx] = noalias;
    }

    Module* parent() const noexcept { return parent_; }
    void set_parent(Module* m) noexcept { parent_ = m; }

    const std::vector<BasicBlock*>& blocks() const noexcept { return blocks_; }
    std::vector<BasicBlock*>& blocks() noexcept { return blocks_; }
    size_t block_count() const noexcept { return blocks_.size(); }

    // A lazy function is a definition whose body its module builds on first
    // use (Module::set_body_provider, Module::materialize): until then it
    // has no blocks, and nothing may read its blocks, which another thread
    // may be building. has_body() is the question "a definition, not a
    // declaration?"; body_ready() is "may its blocks be read now?".
    bool is_lazy() const noexcept { return body_state_.load(std::memory_order_acquire) != kBodyReady; }
    bool body_ready() const noexcept { return !is_lazy(); }
    bool has_body() const noexcept { return is_lazy() || !blocks_.empty(); }
    // Called by the module's front end before the program runs, on a
    // function with no blocks yet.
    void mark_lazy() noexcept { body_state_.store(kBodyLazy, std::memory_order_release); }
    // Called by Module::materialize once the body is built: publishes it.
    void mark_body_ready() noexcept { body_state_.store(kBodyReady, std::memory_order_release); }

    // A body the module's provider builds can be built again: Module::
    // materialize gives each one its own arena (body_arena, which a Builder
    // positioned in the function allocates from), so the module can drop it
    // once its code exists (Module::release_bodies) and build it anew the
    // next time something reads it. A front end makes the same claim for a
    // body it built up front with own_body_arena before building it, and
    // takes on the provider's promise for it: built again, it is the same
    // body.
    Arena* body_arena() const noexcept { return body_arena_.get(); }
    // Gives the function a fresh arena for the body about to be built (sized
    // after the last one, if it had one), and notes the ids it starts from.
    void own_body_arena();
    // Whether the body was built at least once: a lazy function that was
    // never built has never run, while one released has.
    bool built_before() const noexcept { return built_before_; }
    void note_built() noexcept { built_before_ = true; }
    // A body something keeps pointers into beyond any read of it (an OSR
    // entry's plan names its blocks and values): never released.
    void pin_body() const noexcept { body_pinned_.store(true, std::memory_order_release); }
    bool body_pinned() const noexcept { return body_pinned_.load(std::memory_order_acquire); }
    // Whether release_body may drop the body now.
    bool body_releasable() const noexcept { return body_arena_ && !body_pinned() && body_ready(); }
    // Drops the body (blocks, resume points, the arena) and makes the
    // function lazy again, with the ids it was first built from. Returns the
    // arena's bytes. Module::release_bodies' part, under its body lock.
    size_t release_body();

    // Set once a handle is bound to this function (the only Functions code
    // is compiled from, so also every deopt target): runtime::forget_module
    // scans the tables' handles only for a module with such a function, so
    // dropping a private copy (a tier-2 or OSR compile's) costs nothing there.
    void mark_dispatch_referenced() const noexcept { dispatch_referenced_.store(true, std::memory_order_release); }
    bool dispatch_referenced() const noexcept { return dispatch_referenced_.load(std::memory_order_acquire); }

    BasicBlock* entry_block() const noexcept {
        return blocks_.empty() ? nullptr : blocks_.front();
    }

    void append_block(BasicBlock* bb);
    void prepend_block(BasicBlock* bb);
    void remove_block(BasicBlock* bb);
    void sort_blocks_rpo();
    BasicBlock* get_block_by_name(std::string_view name) const noexcept;
    BasicBlock* get_block_by_id(uint32_t id) const noexcept;

    void add_resume_point(uint32_t resume_id, BasicBlock* target);
    BasicBlock* get_resume_target(uint32_t resume_id) const noexcept;
    // Drops `resume_id`'s entry (no-op when it has none). A tier-2 copy whose
    // guards finish in Tier 0 needs no resume blocks of its own, and a block
    // that stays a resume target stays reachable.
    void remove_resume_point(uint32_t resume_id);
    const std::vector<std::pair<uint32_t, BasicBlock*>>& resume_points() const noexcept {
        return resume_points_;
    }

    // The first guard of this function with `resume_id`, or null.
    const Instruction* find_guard(uint32_t resume_id) const noexcept;
    // find_guard for every resume id at once (one walk of the body): what a
    // caller looking up many ids uses, since find_guard walks the body each
    // time.
    std::unordered_map<uint32_t, const Instruction*> guards_by_resume_id() const;
    // The resume id a new guard of this function gets: one past the largest
    // id of its guards, 0 when it has none. A resume id names one Tier-0
    // guard; tier-2 deopt finds the guard to resume at by it.
    uint32_t next_guard_resume_id() const noexcept;
    // Allocates a new guard's resume id: next_guard_resume_id() the first
    // time (one scan), then a counter, so building N guards costs O(N)
    // rather than a scan of the whole body per guard. Ids stay unique; a
    // guard removed since leaves a gap instead of being reused. A guard given
    // an explicit id afterwards is reported with note_guard_resume_id.
    uint32_t take_guard_resume_id() noexcept;
    void note_guard_resume_id(uint32_t id) noexcept {
        if (guard_id_known_ && id >= next_guard_id_) next_guard_id_ = id + 1;
    }
    // A guard's exit stub: the function of this function's module named by
    // the guard's exit label, or null when the label names none (it is then
    // only a label). Called as stub(state values...); its result is this
    // function's result. See "Guard exits" in docs/mir_reference.md.
    const Function* guard_exit_stub(const Instruction& guard) const noexcept;
    // Whether `stub` has the exit-stub signature for `guard` of this
    // function: one parameter per state value, of its type, and this
    // function's return type. When not, `why` says what differs.
    bool guard_exit_stub_matches(const Instruction& guard, const Function& stub, std::string& why) const;

    bool allow_fp_reassociation() const noexcept { return allow_fp_reassociation_; }
    void set_allow_fp_reassociation(bool allow) noexcept { allow_fp_reassociation_ = allow; }

    bool is_deterministic_fp() const noexcept { return deterministic_fp_; }
    void set_deterministic_fp(bool det = true) noexcept { deterministic_fp_ = det; }

    bool requires_barrier_certs() const noexcept { return requires_barrier_certs_; }
    void set_requires_barrier_certs(bool req = true) noexcept { requires_barrier_certs_ = req; }

    const std::vector<GuardElimCert>& guard_elim_certs() const noexcept { return guard_elim_certs_; }
    std::vector<GuardElimCert>& guard_elim_certs() noexcept { return guard_elim_certs_; }
    void add_guard_elim_cert(GuardElimCert cert) { guard_elim_certs_.push_back(cert); }
    void clear_guard_elim_certs() noexcept { guard_elim_certs_.clear(); }

    // A lowered coroutine body's frame shape, recorded by coroutine lowering
    // (compute_coro_frame_layout): the slot count and which slots hold
    // references. Recorded rather than rederived because the passes that run
    // on the body afterwards may retype or fold its frame accesses.
    bool has_coro_frame_layout() const noexcept { return has_coro_frame_layout_; }
    uint32_t coro_frame_slot_count() const noexcept { return coro_frame_slot_count_; }
    const std::vector<uint64_t>& coro_frame_ref_bits() const noexcept { return coro_frame_ref_bits_; }
    void set_coro_frame_layout(uint32_t slot_count, std::vector<uint64_t> ref_bits) {
        has_coro_frame_layout_ = true;
        coro_frame_slot_count_ = slot_count;
        coro_frame_ref_bits_ = std::move(ref_bits);
    }

    uint32_t next_value_id() noexcept { return next_value_id_++; }
    uint32_t next_block_id() noexcept { return next_block_id_++; }
    uint32_t current_next_value_id() const noexcept { return next_value_id_; }
    uint32_t current_next_block_id() const noexcept { return next_block_id_; }
    void set_next_value_id(uint32_t id) noexcept { next_value_id_ = id; }
    void set_next_block_id(uint32_t id) noexcept { next_block_id_ = id; }

    void rebuild_cfg_predecessors();

private:
    static constexpr uint8_t kBodyReady = 0;
    static constexpr uint8_t kBodyLazy = 1;
    std::atomic<uint8_t> body_state_{kBodyReady};
    mutable std::atomic<bool> dispatch_referenced_{false};
    mutable std::atomic<bool> body_pinned_{false};
    bool built_before_ = false;
    std::unique_ptr<Arena> body_arena_;
    // The last body's arena size, and the ids the body is built from.
    size_t body_bytes_ = 0;
    uint32_t body_first_value_id_ = 0;
    uint32_t body_first_block_id_ = 0;

    std::string_view name_;
    Type return_type_ = Type::void_type();
    std::vector<Type> param_types_;
    std::vector<bool> param_noalias_;

    std::vector<BasicBlock*> blocks_;
    Module* parent_ = nullptr;

    std::vector<std::pair<uint32_t, BasicBlock*>> resume_points_;
    // resume id -> position in resume_points_, built once the table is large
    // (resume_point_position); empty otherwise.
    mutable std::unordered_map<uint32_t, uint32_t> resume_index_;
    size_t resume_point_position(uint32_t resume_id) const noexcept;

    uint32_t next_value_id_ = 0;
    uint32_t next_block_id_ = 0;
    uint32_t next_guard_id_ = 0;
    bool guard_id_known_ = false;
    bool allow_fp_reassociation_ = false;
    bool deterministic_fp_ = false;
    bool requires_barrier_certs_ = false;

    bool has_coro_frame_layout_ = false;
    uint32_t coro_frame_slot_count_ = 0;
    std::vector<uint64_t> coro_frame_ref_bits_;
    std::vector<GuardElimCert> guard_elim_certs_;
};

} // namespace brass
