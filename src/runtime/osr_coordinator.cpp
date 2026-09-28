// On-stack replacement (osr_coordinator.hpp): a hot loop's OSR entry
// function, compiled on the shared CompilePool and entered from the
// FastInterpreter's backedge once it is ready.

#include <brass/runtime/osr_coordinator.hpp>
#include <brass/codegen/jit_exec.hpp>
#include <brass/mir/loop_opt.hpp>
#include <brass/mir/osr_entry.hpp>
#include <brass/runtime/code_installer.hpp>
#include <brass/runtime/compile_pool.hpp>
#include <brass/runtime/multi_tier_pipeline.hpp>
#include <brass/runtime/tier_timeline.hpp>
#include <brass/target/target.hpp>
#include <brass/vm/fast_interpreter.hpp>
#include "../vm/fast_interpreter_impl.hpp"
#include "tier2_link.hpp"
#include <stdexcept>
#include <string>
#include <vector>

namespace brass::runtime {

struct OsrCoordinator::Entry {
    // Invalid: the code's speculation failed deopt_threshold times; it is
    // never entered again, but frames may still be running it, so it stays
    // alive (retired_) until the program is released.
    enum class State : uint8_t { Compiling, Ready, Failed, Invalid };
    std::atomic<State> state{State::Compiling};
    OsrEntryPlan plan;
    std::string name;
    std::shared_ptr<codegen::JitExecutionEngine> jit;
    void* entry = nullptr;
    const BasicBlock* header = nullptr;
    std::atomic<uint64_t> deopts{0};
    mutable std::atomic<bool> entered{false};
    // The code's leave flag (OsrEntryPlan::leave_check), read by the code at
    // each backedge to its header: set when the code is invalidated, so a
    // frame still in it goes back to the interpreter there instead of
    // calling a failing callee copy on every iteration.
    mutable std::atomic<uint64_t> leave{0};
};

namespace {

// The OSR'd frame is the native code's while it runs: the interpreter's
// record of it leaves the thread's frame chain, so a stack walk sees the
// function once, as the native frame.
class HiddenFrame {
public:
    explicit HiddenFrame(FastFrame& frame) noexcept : top_(FastInterpreter::thread_frame_top()), saved_(top_) {
        top_ = frame.thread_prev;
    }
    ~HiddenFrame() { top_ = saved_; }
    HiddenFrame(const HiddenFrame&) = delete;
    HiddenFrame& operator=(const HiddenFrame&) = delete;

private:
    FastFrame*& top_;
    FastFrame* saved_;
};

constexpr uint8_t kOsrPriority = 255;

static_assert(sizeof(std::atomic<uint64_t>) == sizeof(uint64_t) && std::atomic<uint64_t>::is_always_lock_free,
              "OSR code reads the leave flag as a plain u64");

// A value the OSR code wrote back as the interpreter's register holds it: a
// narrower value's store left the slot's upper bytes as they were on entry
// (the interpreter keeps a narrow value zero-extended, as trunc8 and trunc16
// leave it).
uint64_t register_bits(Type t, uint64_t bits) {
    switch (t.kind()) {
        case TypeKind::I8:
            return bits & 0xFFull;
        case TypeKind::I16:
            return bits & 0xFFFFull;
        case TypeKind::I32:
        case TypeKind::F32:
            return bits & 0xFFFFFFFFull;
        default:
            return bits;
    }
}

} // namespace

OsrCoordinator& OsrCoordinator::instance() {
    static OsrCoordinator s_instance;
    return s_instance;
}

OsrCoordinator::OsrCoordinator() = default;

OsrCoordinator::OsrCoordinator(TieringRegistry& registry) : registry_(&registry) {
    if (registry.is_default()) {
        throw std::logic_error("OsrCoordinator: the default program's coordinator is OsrCoordinator::instance()");
    }
}

OsrCoordinator::~OsrCoordinator() {
    release_program();
}

TieringRegistry& OsrCoordinator::registry() const noexcept {
    return registry_ ? *registry_ : TieringRegistry::instance();
}

void OsrCoordinator::set_threshold(uint64_t threshold) noexcept {
    threshold_ = threshold;
    registry().default_config().backedge_osr_threshold = threshold;
}

void OsrCoordinator::release_program() {
    {
        std::lock_guard<std::mutex> lock(mutex_);
        released_ = true;
    }
    CompilePool& pool = CompilePool::shared();
    pool.cancel(this);
    pool.wait_owner(this);
    std::lock_guard<std::mutex> lock(mutex_);
    entries_.clear();
    retired_.clear();
    reoptimizations_.clear();
}

void OsrCoordinator::note_deopt(const std::shared_ptr<Entry>& e, uint32_t resume_id) {
    const uint64_t threshold = registry().default_config().deopt_threshold;
    if (e->deopts.fetch_add(1, std::memory_order_relaxed) + 1 < threshold) return;
    Entry::State ready = Entry::State::Ready;
    if (!e->state.compare_exchange_strong(ready, Entry::State::Invalid, std::memory_order_acq_rel)) return;
    e->leave.store(1, std::memory_order_release);
    const std::string_view fn_name = e->plan.function ? e->plan.function->name() : std::string_view(e->name);
    record_tier_instant(TierEventKind::Invalidate, fn_name);
    registry().dispatch_table().pipeline().notify_invalidation(fn_name, resume_id);
    // The continuation runs the loop in a Tier-0 frame nested under this
    // code; entering the code again at its next backedge would fail the
    // same guard and nest another frame, without bound. With a front pass
    // the loop may get a new entry, compiled against the feedback the
    // failures produced, a bounded number of times, as tier-up does.
    const bool front_pass = static_cast<bool>(registry().dispatch_table().pipeline().tier2_front_pass());
    std::lock_guard<std::mutex> lock(mutex_);
    if (released_) return;
    auto it = entries_.find(e->header);
    if (it == entries_.end() || it->second != e) return;
    if (front_pass && ++reoptimizations_[e->header] <= kMaxReoptimizations) {
        retired_.push_back(std::move(it->second));
        entries_.erase(it);
    }
}

void OsrCoordinator::stop_compiles() {
    CompilePool& pool = CompilePool::shared();
    pool.cancel(this);
    pool.wait_owner(this);
    // Nothing of this coordinator's is queued or running now: an entry still
    // compiling was dropped.
    std::lock_guard<std::mutex> lock(mutex_);
    for (auto it = entries_.begin(); it != entries_.end();) {
        if (it->second->state.load(std::memory_order_acquire) == Entry::State::Compiling) {
            it = entries_.erase(it);
        } else {
            ++it;
        }
    }
}

bool OsrCoordinator::runs_program() const noexcept {
    return registry().dispatch_table().pipeline().is_initialized();
}

bool OsrCoordinator::try_osr_migration(FastInterpreter&, const Function& fn, BasicBlock* loop_header,
                                       FastFrame& frame, RuntimeValue& out_result) {
    if (!enabled_ || !loop_header || !runs_program()) return false;
    std::shared_ptr<Entry> e;
    bool start = false;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        if (released_) return false;
        auto it = entries_.find(loop_header);
        if (it == entries_.end()) {
            // One entry of a function compiles at a time. The backedge count
            // is the function's, so a frame past the threshold asks at every
            // loop header it crosses, and an entry is a copy of the whole
            // rest of the function: each would cost a full compile of the
            // same code. The frame enters the first one at its header, and a
            // header it never reaches again asks once that one is done.
            for (const auto& [header, other] : entries_) {
                if (header->parent() == loop_header->parent() &&
                    other->state.load(std::memory_order_acquire) == Entry::State::Compiling) {
                    return false;
                }
            }
            auto slot = std::make_shared<Entry>();
            slot->header = loop_header;
            it = entries_.emplace(loop_header, std::move(slot)).first;
            start = true;
        }
        e = it->second;
    }
    if (start) {
        request_entry(fn, *loop_header, e);
        return false;
    }
    if (e->state.load(std::memory_order_acquire) != Entry::State::Ready) return false;
    return enter(fn, frame, *e, out_result);
}

void OsrCoordinator::request_entry(const Function& fn, const BasicBlock& header, const std::shared_ptr<Entry>& e) {
    TierEventScope timed(TierEventKind::OsrRequest, fn.name());
    auto fail = [&] {
        timed.set_ok(false);
        e->state.store(Entry::State::Failed, std::memory_order_release);
    };
    FunctionDispatchTable& table = registry().dispatch_table();
    // The function's handle is where a failed guard of the code resumes;
    // one bound to another Function is not this code's.
    FunctionHandle* handle = table.get_or_create(fn.name(), &fn);
    if (!fn.parent() || !handle || handle->mir_function() != &fn) return fail();

    // Only the call targets are read here, on the interpreter's thread: its
    // type feedback grows as it runs. Planning the entry and copying it
    // with its callees read the program's MIR, which running it does not
    // change, and take milliseconds (a loop nest's liveness, the region's
    // SSA repair), so they run on the worker, ahead of the compile, while
    // the interpreter carries on with the loop.
    auto job = [this, &fn, hdr = &header, e, targets = detail::speculated_call_targets(table, fn.name())] {
        std::unique_ptr<Module> mod = copy_entry(fn, *hdr, e, targets);
        if (!mod) {
            e->state.store(Entry::State::Failed, std::memory_order_release);
            return;
        }
        compile_entry(fn, *mod, e);
    };
    if (!CompilePool::shared().submit(this, kOsrPriority, std::move(job))) fail();
}

std::unique_ptr<Module> OsrCoordinator::copy_entry(const Function& fn, const BasicBlock& header,
                                                   const std::shared_ptr<Entry>& e,
                                                   const std::vector<std::string>& targets) {
    TierEventScope timed(TierEventKind::OsrRequest, fn.name());
    timed.set_ok(false);
    const Module* src = fn.parent();
    FunctionDispatchTable& table = registry().dispatch_table();
    std::string why;
    auto plan = plan_osr_entry(fn, header, &why);
    if (!plan) return nullptr;
    e->plan = std::move(*plan);
    e->plan.leave_check = region_has_call(e->plan);
    e->name = std::string(fn.name()) + ".osr" + std::to_string(header.id());

    // The copy: the entry function, the bodies of what it calls and of its
    // speculated call targets, every other program function declared.
    std::unique_ptr<Module> mod;
    try {
        mod = std::make_unique<Module>(src->name());
        mod->set_allow_fp_reassociation(src->allow_fp_reassociation());
        mod->set_pinned_tls_register(src->pinned_tls_register());
        mod->copy_declarations_from(*src);
        if (!build_osr_entry_function(e->plan, *mod, e->name, &why)) return nullptr;
        clone_callee_closure(*src, *mod, e->plan.region, targets, &fn);
        // Every program function the copy carries a body of has its handle
        // now: a guard of that body failing in the OSR code resumes through
        // it (compile_entry registers the resumer), and a callee the
        // interpreter had not yet called has none, which left such a
        // failure with nowhere to go.
        for (const Function* f : mod->functions()) {
            if (!f || f->block_count() == 0 || f->name() == e->name) continue;
            if (const Function* def = src->get_function(f->name())) table.get_or_create(f->name(), def);
        }
        detail::bind_declared_handles(table, *mod, *src);
    } catch (const std::exception&) {
        return nullptr;
    }
    timed.set_ok(true);
    return mod;
}

void OsrCoordinator::compile_entry(const Function& fn, Module& copy, const std::shared_ptr<Entry>& shared) {
    Entry& e = *shared;
    Module* const mod = &copy;
    TierEventScope timed(TierEventKind::OsrCompile, fn.name());
    auto fail = [&] {
        timed.set_ok(false);
        e.state.store(Entry::State::Failed, std::memory_order_release);
    };
    FunctionDispatchTable& table = registry().dispatch_table();
    MultiTierPipeline& pipeline = table.pipeline();
    FunctionHandle* handle = table.find(fn.name());
    if (!handle || handle->mir_function() != &fn) return fail();

    // The program's own functions this copy defines: their func_addrs yield
    // the program's addresses, their guards resume in their Functions.
    std::unordered_map<std::string, const Function*> siblings;
    for (const Function* f : mod->functions()) {
        if (!f || f->block_count() == 0 || f->name() == e.name) continue;
        const FunctionHandle* h = table.find(f->name());
        const Function* def = fn.parent()->get_function(f->name());
        if (h && def && h->mir_function() == def) siblings.emplace(std::string(f->name()), def);
    }

    std::shared_ptr<codegen::JitExecutionEngine> jit;
    try {
        std::string errors;
        if (!detail::run_tier2_optimization_pipeline(*mod, table, errors)) return fail();
        const Function* entry_fn = mod->get_function(e.name);
        std::string why;
        if (!entry_fn || !deopt_targets_valid(*entry_fn, &fn, why)) return fail();
        // Every other body's guards need a Tier-0 Function to resume in (its
        // resumer, registered below): a guard failing where there is none
        // would be fatal, so such a copy stays uncompiled.
        for (const Function* f : mod->functions()) {
            if (!f || f->block_count() == 0 || f == entry_fn) continue;
            auto sib = siblings.find(std::string(f->name()));
            if (!deopt_targets_valid(*f, sib != siblings.end() ? sib->second : nullptr, why)) return fail();
        }
        jit = detail::make_tier2_engine(table, Target::host());
        jit->set_code_tier(debug::JitTier::Osr);
        auto known = [&](std::string_view name) { return name == fn.name() || siblings.count(std::string(name)) != 0; };
        detail::canonicalize_function_addresses(*mod, known, pipeline, *jit);
        detail::link_declared_functions(*mod, table, *jit);
        detail::link_coroutine_bodies(*mod, table, *jit);
        if (!jit->compile_and_load(*mod)) return fail();
    } catch (const std::exception&) {
        return fail();
    }

    void* entry = jit->get_symbol_address(e.name);
    if (!entry) return fail();
    pipeline.add_stack_maps(jit->stack_maps());
    for (const auto& lf : jit->loaded_functions()) {
        const Function* f = mod->get_function(lf.name);
        if (!f || f->block_count() == 0) continue;
        const std::string_view shown = lf.name == e.name ? fn.name() : std::string_view(lf.name);
        pipeline.notify_code_installed({shown, TierLevel::Tier2_Optimized, lf.code, lf.size, &lf.lines});
    }
    // Weak: the entry owns the code the resumers are registered for. A
    // guard of a carried callee body failing counts against the entry as
    // its own guards do: the loop keeps calling that copy.
    std::weak_ptr<Entry> weak = shared;
    auto count_against_entry = [this, weak](const DeoptFrame& frame) {
        if (std::shared_ptr<Entry> live = weak.lock()) note_deopt(live, frame.resume_id);
    };
    for (const Function* f : mod->functions()) {
        if (!f || f->block_count() == 0) continue;
        void* addr = jit->get_symbol_address(f->name());
        if (!addr) continue;
        if (f->name() == e.name) {
            table.register_code_address(addr, fn.name());
            continue;
        }
        table.register_code_address(addr, f->name());
        auto sib = siblings.find(std::string(f->name()));
        FunctionHandle* h = sib != siblings.end() ? table.find(f->name()) : nullptr;
        std::string why;
        if (h && deopt_targets_valid(*f, sib->second, why)) {
            detail::register_tier2_resumer(table, *h, addr, sib->second, count_against_entry);
        }
    }
    detail::register_tier2_resumer(table, *handle, entry, &fn, count_against_entry);
    e.jit = std::move(jit);
    e.entry = entry;
    e.state.store(Entry::State::Ready, std::memory_order_release);
}

bool OsrCoordinator::enter(const Function& fn, FastFrame& frame, const Entry& e, RuntimeValue& out_result) {
    const BytecodeFunction* bfn = frame.bfn;
    if (!bfn || FastInterpreter::thread_frame_top() != &frame) return false;
    // The frame's live values, where the entry reads them: a register each,
    // raw (a narrower value is read from its low bytes).
    const size_t n = e.plan.live_ins.size();
    std::vector<uint64_t> buffer(e.plan.buffer_slots(), 0);
    std::vector<uint32_t> regs(n, 0);
    for (size_t i = 0; i < n; ++i) {
        const auto& li = e.plan.live_ins[i];
        if (li.rematerialize) continue;
        const BcReg r = bfn->reg_of_ssa(li.value->id());
        if (r == kNoReg || r >= frame.num_registers) return false;
        regs[i] = r;
        buffer[i] = frame.registers[r];
    }
    if (e.plan.leave_check) buffer[e.plan.leave_flag_slot()] = reinterpret_cast<uintptr_t>(&e.leave);
    const std::vector<Type> params{Type::ptr()};
    const std::vector<RuntimeValue> args{RuntimeValue::from_ptr(buffer.data())};
    // Counted on entry: a throw can end the OSR call.
    total_osr_migrations_++;
    if (!e.entered.exchange(true, std::memory_order_relaxed)) record_tier_instant(TierEventKind::OsrEnter, fn.name());
    RuntimeValue result;
    {
        HiddenFrame hidden(frame);
        result = invoke_native_address(e.entry, fn.return_type(), &params, args);
    }
    if (!e.plan.leave_check || buffer[e.plan.left_slot()] == 0) {
        out_result = result;
        return true;
    }
    // The code was invalidated and left at a backedge to the header with the
    // loop's values in the buffer: the frame takes them back and goes on
    // with the backedge, in the interpreter.
    for (size_t i = 0; i < n; ++i) {
        const auto& li = e.plan.live_ins[i];
        if (li.rematerialize) continue;
        frame.registers[regs[i]] = register_bits(li.value->type(), buffer[i]);
    }
    total_osr_leaves_++;
    return false;
}

} // namespace brass::runtime
