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
#include <brass/target/target.hpp>
#include <brass/vm/fast_interpreter.hpp>
#include "../vm/fast_interpreter_impl.hpp"
#include "tier2_link.hpp"
#include <stdexcept>
#include <string>
#include <vector>

namespace brass::runtime {

struct OsrCoordinator::Entry {
    enum class State : uint8_t { Compiling, Ready, Failed };
    std::atomic<State> state{State::Compiling};
    OsrEntryPlan plan;
    std::string name;
    std::shared_ptr<codegen::JitExecutionEngine> jit;
    void* entry = nullptr;
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
        std::shared_ptr<Entry>& slot = entries_[loop_header];
        if (!slot) {
            slot = std::make_shared<Entry>();
            start = true;
        }
        e = slot;
    }
    if (start) {
        request_entry(fn, *loop_header, e);
        return false;
    }
    if (e->state.load(std::memory_order_acquire) != Entry::State::Ready) return false;
    return enter(fn, frame, *e, out_result);
}

void OsrCoordinator::request_entry(const Function& fn, const BasicBlock& header, const std::shared_ptr<Entry>& e) {
    auto fail = [&] { e->state.store(Entry::State::Failed, std::memory_order_release); };
    const Module* src = fn.parent();
    FunctionDispatchTable& table = registry().dispatch_table();
    // The function's handle is where a failed guard of the code resumes;
    // one bound to another Function is not this code's.
    FunctionHandle* handle = table.get_or_create(fn.name(), &fn);
    if (!src || !handle || handle->mir_function() != &fn) return fail();

    std::string why;
    auto plan = plan_osr_entry(fn, header, &why);
    if (!plan) return fail();
    e->plan = std::move(*plan);
    e->name = std::string(fn.name()) + ".osr" + std::to_string(header.id());

    // The copy: the entry function, the bodies of what it calls and of its
    // speculated call targets, every other program function declared.
    std::unique_ptr<Module> mod;
    try {
        mod = std::make_unique<Module>(src->name());
        mod->set_allow_fp_reassociation(src->allow_fp_reassociation());
        mod->set_pinned_tls_register(src->pinned_tls_register());
        mod->copy_declarations_from(*src);
        if (!build_osr_entry_function(e->plan, *mod, e->name, &why)) return fail();
        clone_callee_closure(*src, *mod, e->plan.region, detail::speculated_call_targets(table, fn.name()), &fn);
        detail::bind_declared_handles(table, *mod, *src);
    } catch (const std::exception&) {
        return fail();
    }

    auto job = [this, &fn, e, m = std::shared_ptr<Module>(std::move(mod))] { compile_entry(fn, *m, *e); };
    if (!CompilePool::shared().submit(this, kOsrPriority, std::move(job))) fail();
}

void OsrCoordinator::compile_entry(const Function& fn, Module& copy, Entry& e) {
    Module* const mod = &copy;
    auto fail = [&] { e.state.store(Entry::State::Failed, std::memory_order_release); };
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
        jit = detail::make_tier2_engine(table, Target::host());
        auto known = [&](std::string_view name) { return name == fn.name() || siblings.count(std::string(name)) != 0; };
        detail::canonicalize_function_addresses(*mod, known, pipeline, *jit);
        detail::link_declared_functions(*mod, table, *jit);
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
        if (h && deopt_targets_valid(*f, sib->second, why)) detail::register_tier2_resumer(table, *h, addr, sib->second);
    }
    detail::register_tier2_resumer(table, *handle, entry, &fn);
    e.jit = std::move(jit);
    e.entry = entry;
    e.state.store(Entry::State::Ready, std::memory_order_release);
}

bool OsrCoordinator::enter(const Function& fn, FastFrame& frame, const Entry& e, RuntimeValue& out_result) {
    const BytecodeFunction* bfn = frame.bfn;
    if (!bfn || FastInterpreter::thread_frame_top() != &frame) return false;
    // The frame's live values, where the entry reads them: a register each,
    // raw (a narrower value is read from its low bytes).
    std::vector<uint64_t> buffer(e.plan.live_ins.size(), 0);
    for (size_t i = 0; i < e.plan.live_ins.size(); ++i) {
        const auto& li = e.plan.live_ins[i];
        if (li.rematerialize) continue;
        auto it = bfn->ssa_to_reg.find(li.value->id());
        if (it == bfn->ssa_to_reg.end() || it->second >= frame.num_registers) return false;
        buffer[i] = frame.registers[it->second];
    }
    const std::vector<Type> params{Type::ptr()};
    const std::vector<RuntimeValue> args{RuntimeValue::from_ptr(buffer.data())};
    // Counted on entry: a throw can end the OSR call.
    total_osr_migrations_++;
    HiddenFrame hidden(frame);
    out_result = invoke_native_address(e.entry, fn.return_type(), &params, args);
    return true;
}

} // namespace brass::runtime
