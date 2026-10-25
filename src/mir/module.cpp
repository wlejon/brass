#include <brass/mir/module.hpp>
#include <brass/core/env.hpp>
#include <brass/runtime/code_installer.hpp>
#include <algorithm>
#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <mutex>
#include <stdexcept>
#include <string>

namespace brass {

struct Module::LazyBodies {
    std::mutex mutex;
    BodyProvider provider;
    std::atomic<size_t> built{0};
    std::atomic<size_t> released{0};
    // BodyReads held now, and whether release_bodies is running (BodyRead's
    // handshake with it: each side writes its own flag, then reads the
    // other's).
    std::atomic<uint32_t> readers{0};
    std::atomic<bool> releasing{false};
};

namespace {

// The live modules with a body provider, for release_idle_bodies.
std::mutex& provider_modules_mutex() {
    static std::mutex m;
    return m;
}
std::vector<Module*>& provider_modules() {
    static std::vector<Module*> mods;
    return mods;
}

void forget_provider_module(Module* mod) {
    std::lock_guard<std::mutex> lock(provider_modules_mutex());
    std::erase(provider_modules(), mod);
}

} // namespace

Module::Module() noexcept = default;

Module::Module(Module&& other) noexcept
    : arena_(std::move(other.arena_)),
      string_pool_(std::move(other.string_pool_)),
      name_(other.name_),
      functions_(std::move(other.functions_)),
      function_map_(std::move(other.function_map_)),
      external_symbols_(std::move(other.external_symbols_)),
      symbol_roles_(std::move(other.symbol_roles_)),
      string_symbols_(std::move(other.string_symbols_)),
      allow_fp_reassociation_(other.allow_fp_reassociation_),
      deterministic_fp_(other.deterministic_fp_),
      pinned_tls_register_(other.pinned_tls_register_),
      has_loop_optimizations_(other.has_loop_optimizations_),
      debug_context_(std::move(other.debug_context_)),
      lazy_(std::move(other.lazy_)) {
    if (lazy_) {
        std::lock_guard<std::mutex> lock(provider_modules_mutex());
        std::replace(provider_modules().begin(), provider_modules().end(), &other, this);
    }
}

Module::Module(std::string_view name)
    : arena_(), string_pool_(), name_(string_pool_.intern(name)) {}

Module::~Module() {
    if (lazy_) forget_provider_module(this);
    // Runtime registries hold raw pointers into this module; drop them before
    // the arena frees the functions, so a new module reusing the addresses is
    // never mistaken for this one.
    runtime::forget_module(*this);
}

Module& Module::operator=(Module&& other) noexcept {
    if (this == &other) return *this;
    if (lazy_) forget_provider_module(this);
    runtime::forget_module(*this);
    if (other.lazy_) {
        std::lock_guard<std::mutex> lock(provider_modules_mutex());
        std::replace(provider_modules().begin(), provider_modules().end(), &other, this);
    }
    arena_ = std::move(other.arena_);
    string_pool_ = std::move(other.string_pool_);
    name_ = other.name_;
    functions_ = std::move(other.functions_);
    function_map_ = std::move(other.function_map_);
    external_symbols_ = std::move(other.external_symbols_);
    symbol_roles_ = std::move(other.symbol_roles_);
    string_symbols_ = std::move(other.string_symbols_);
    allow_fp_reassociation_ = other.allow_fp_reassociation_;
    deterministic_fp_ = other.deterministic_fp_;
    pinned_tls_register_ = other.pinned_tls_register_;
    has_loop_optimizations_ = other.has_loop_optimizations_;
    debug_context_ = std::move(other.debug_context_);
    lazy_ = std::move(other.lazy_);
    return *this;
}

void Module::set_body_provider(BodyProvider provider) {
    if (!lazy_) {
        lazy_ = std::make_unique<LazyBodies>();
        std::lock_guard<std::mutex> lock(provider_modules_mutex());
        provider_modules().push_back(this);
    }
    lazy_->provider = std::move(provider);
}

void Module::materialize(const Function& fn) const {
    if (!fn.is_lazy()) return;
    if (!lazy_ || !lazy_->provider) {
        std::fprintf(stderr, "brass: fatal: '%.*s' has a lazy body and its module no provider\n",
                     static_cast<int>(fn.name().size()), fn.name().data());
        std::fflush(stderr);
        std::abort();
    }
    std::lock_guard<std::mutex> lock(lazy_->mutex);
    if (!fn.is_lazy()) return;
    auto& body = const_cast<Function&>(fn);
    // Its own arena, so release_bodies can drop it again.
    body.own_body_arena();
    if (!lazy_->provider(body)) {
        std::fprintf(stderr, "brass: fatal: building the body of '%.*s' failed\n",
                     static_cast<int>(fn.name().size()), fn.name().data());
        std::fflush(stderr);
        std::abort();
    }
    // Coroutine lowering runs once over the bodies a module has when it
    // starts executing, and pairs every coro_create with its body: a body
    // built after that cannot join in, so one needing it is an error of the
    // producer, not something to run unlowered.
    for (const BasicBlock* bb : body.blocks()) {
        for (const Instruction* inst : *bb) {
            if (inst && (inst->opcode() == Opcode::coro_create || inst->opcode() == Opcode::coro_suspend)) {
                std::fprintf(stderr, "brass: fatal: lazy body '%.*s' has coroutine ops; build it eagerly\n",
                             static_cast<int>(fn.name().size()), fn.name().data());
                std::fflush(stderr);
                std::abort();
            }
        }
    }
    lazy_->built.fetch_add(1, std::memory_order_relaxed);
    body.note_built();
    body.mark_body_ready();
}

size_t Module::materialized_count() const noexcept {
    return lazy_ ? lazy_->built.load(std::memory_order_relaxed) : 0;
}

Module::BodyRead::BodyRead(const Module& mod) {
    enter(mod);
}

void Module::BodyRead::enter(const Module& mod) {
    LazyBodies* lazy = mod.lazy_.get();
    if (!lazy) return;
    mod_ = &mod;
    lazy->readers.fetch_add(1, std::memory_order_seq_cst);
    // A release under way finishes first (it holds the lock throughout);
    // one that starts after this sees the reader and releases nothing.
    if (lazy->releasing.load(std::memory_order_seq_cst)) {
        std::lock_guard<std::mutex> wait(lazy->mutex);
    }
}

Module::BodyRead::BodyRead(const Function& fn) {
    if (!fn.parent()) return;
    enter(*fn.parent());
    if (fn.is_lazy()) fn.parent()->materialize(fn);
}

namespace {
// BRASS_RELEASE_BODIES_STRESS=N: every N-th time the last reader leaves,
// it releases every body, so a read after that builds the body again (a
// front end's test that its provider builds the same body twice, and
// brass's that nothing keeps pointers into a body past its read). N=1 is
// every time; a larger N keeps a program that deopts on every call (each
// resume reads its guard) from building a body per call.
uint32_t release_stress_period() {
    static const uint32_t period = [] {
        const unsigned long n = std::strtoul(env_value("BRASS_RELEASE_BODIES_STRESS").c_str(), nullptr, 10);
        return n > 0xFFFFu ? 0xFFFFu : static_cast<uint32_t>(n);
    }();
    return period;
}

bool release_stress_now() {
    const uint32_t period = release_stress_period();
    if (period == 0) return false;
    static std::atomic<uint32_t> exits{0};
    return exits.fetch_add(1, std::memory_order_relaxed) % period == 0;
}
} // namespace

Module::BodyRead::~BodyRead() {
    if (!mod_) return;
    const uint32_t before = mod_->lazy_->readers.fetch_sub(1, std::memory_order_seq_cst);
    if (before == 1 && release_stress_now()) const_cast<Module*>(mod_)->release_bodies();
}

size_t Module::release_bodies() {
    LazyBodies* lazy = lazy_.get();
    if (!lazy || !lazy->provider) return 0;
    std::unique_lock<std::mutex> lock(lazy->mutex, std::try_to_lock);
    if (!lock.owns_lock()) return 0;
    lazy->releasing.store(true, std::memory_order_seq_cst);
    size_t bytes = 0;
    if (lazy->readers.load(std::memory_order_seq_cst) == 0) {
        size_t count = 0;
        for (Function* fn : functions_) {
            if (!fn || !fn->body_releasable()) continue;
            bytes += fn->release_body();
            ++count;
        }
        lazy->released.fetch_add(count, std::memory_order_relaxed);
    }
    lazy->releasing.store(false, std::memory_order_seq_cst);
    return bytes;
}

size_t Module::released_count() const noexcept {
    return lazy_ ? lazy_->released.load(std::memory_order_relaxed) : 0;
}

size_t Module::release_idle_bodies() {
    std::lock_guard<std::mutex> lock(provider_modules_mutex());
    size_t bytes = 0;
    for (Module* mod : provider_modules()) bytes += mod->release_bodies();
    return bytes;
}

Module::BodyStats Module::body_stats() {
    std::lock_guard<std::mutex> lock(provider_modules_mutex());
    BodyStats stats;
    for (Module* mod : provider_modules()) {
        stats.built += mod->materialized_count();
        stats.released += mod->released_count();
        // Under the body lock: a body being built or released is not read.
        std::lock_guard<std::mutex> body_lock(mod->lazy_->mutex);
        for (const Function* fn : mod->functions_) {
            if (fn && fn->body_arena()) stats.live_bytes += fn->body_arena()->bytes_capacity();
        }
    }
    return stats;
}

void Module::set_name(std::string_view name) {
    name_ = string_pool_.intern(name);
}

Function* Module::create_function(std::string_view name, Type return_type, Span<const Type> param_types) {
    std::string_view interned_name = string_pool_.intern(name);
    std::vector<Type> params;
    params.reserve(param_types.size());
    for (size_t i = 0; i < param_types.size(); ++i) {
        params.push_back(param_types[i]);
    }

    Function* fn = arena_.make<Function>(interned_name, return_type, std::move(params));
    fn->set_parent(this);
    if (deterministic_fp_) fn->set_deterministic_fp(true);
    functions_.push_back(fn);
    function_map_[interned_name] = fn;
    return fn;
}

Function* Module::create_function(std::string_view name, Type return_type, std::initializer_list<Type> param_types) {
    return create_function(name, return_type, Span<const Type>(param_types.begin(), param_types.size()));
}

Function* Module::create_function(std::string_view name, Type return_type) {
    return create_function(name, return_type, Span<const Type>());
}

Function* Module::get_function(std::string_view name) const noexcept {
    auto it = function_map_.find(name);
    if (it != function_map_.end()) {
        return it->second;
    }
    return nullptr;
}

void Module::rename_function(Function* fn, std::string_view new_name) {
    if (!fn) return;
    function_map_.erase(fn->name());
    std::string_view interned = string_pool_.intern(new_name);
    fn->set_name(interned);
    function_map_[interned] = fn;
}

void Module::add_external_symbol(std::string_view sym) {
    std::string_view interned_sym = string_pool_.intern(sym);
    if (!has_external_symbol(interned_sym)) {
        external_symbols_.push_back(interned_sym);
    }
}

bool Module::has_external_symbol(std::string_view sym) const noexcept {
    return std::find(external_symbols_.begin(), external_symbols_.end(), sym) != external_symbols_.end();
}

namespace {

uint32_t role_bit(SymbolRole role) noexcept {
    return uint32_t{1} << static_cast<uint32_t>(role);
}

} // namespace

void Module::add_symbol_role(std::string_view sym, SymbolRole role) {
    add_external_symbol(sym);
    symbol_roles_[string_pool_.intern(sym)] |= role_bit(role);
}

bool Module::has_symbol_role(std::string_view sym, SymbolRole role) const noexcept {
    auto it = symbol_roles_.find(sym);
    return it != symbol_roles_.end() && (it->second & role_bit(role)) != 0;
}

std::vector<SymbolRole> Module::symbol_roles(std::string_view sym) const {
    std::vector<SymbolRole> roles;
    for (SymbolRole role : kAllSymbolRoles) {
        if (has_symbol_role(sym, role)) roles.push_back(role);
    }
    return roles;
}

void Module::add_allocation_function(std::string_view sym) {
    add_symbol_role(sym, SymbolRole::Allocator);
}

bool Module::is_allocation_function(std::string_view sym) const noexcept {
    return has_symbol_role(sym, SymbolRole::Allocator);
}

void Module::define_string_symbol(std::string_view sym, std::string_view text) {
    for (const auto& [s, t] : string_symbols_) {
        if (s != sym) continue;
        if (t != text) {
            throw std::logic_error("Module::define_string_symbol: '" + std::string(sym) +
                                   "' is already defined with different text");
        }
        return;
    }
    // intern() stores a NUL after the text, so the copy is a C string. An
    // empty string interns to a literal "", which is NUL-terminated too.
    string_symbols_.emplace_back(string_pool_.intern(sym), string_pool_.intern(text));
}

const char* Module::string_symbol(std::string_view sym) const noexcept {
    for (const auto& [s, t] : string_symbols_) {
        if (s == sym) return t.data();
    }
    return nullptr;
}

void Module::copy_declarations_from(const Module& src) {
    for (std::string_view sym : src.external_symbols()) {
        add_external_symbol(sym);
        for (SymbolRole role : src.symbol_roles(sym)) add_symbol_role(sym, role);
    }
    for (const auto& [sym, text] : src.string_symbols()) define_string_symbol(sym, text);
}

} // namespace brass
