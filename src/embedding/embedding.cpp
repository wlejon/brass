#include <brass/embedding/embedding.hpp>
#include <brass/gc/runtime_gc.hpp>
#include <stdexcept>

namespace brass {

CompiledModule::CompiledModule(std::unique_ptr<codegen::JitExecutionEngine> jit_engine)
    : jit_engine_(std::move(jit_engine)) {
}

void* CompiledModule::get_symbol_address(std::string_view name) const {
    if (!jit_engine_) return nullptr;
    return jit_engine_->get_symbol_address(name);
}

const ModuleStackMap& CompiledModule::stack_maps() const noexcept {
    return jit_engine_->stack_maps();
}

ModuleStackMap& CompiledModule::stack_maps() noexcept {
    return jit_engine_->stack_maps();
}

const runtime::ResumeTableRegistry& CompiledModule::resume_tables() const noexcept {
    return jit_engine_->resume_tables();
}

const runtime::FunctionResumeTable* CompiledModule::get_resume_table(std::string_view fn_name) const noexcept {
    if (!jit_engine_) return nullptr;
    return jit_engine_->get_resume_table(fn_name);
}

void* CompiledModule::get_resume_target_address(std::string_view fn_name, uint32_t resume_id) const {
    if (!jit_engine_) return nullptr;
    return jit_engine_->get_resume_target_address(fn_name, resume_id);
}

const runtime::PatchRegistry& CompiledModule::patch_sites() const noexcept {
    return jit_engine_->patch_sites();
}

runtime::PatchRegistry& CompiledModule::patch_sites() noexcept {
    return jit_engine_->patch_sites();
}

bool CompiledModule::patch_constant(std::string_view site_name, int32_t new_val) {
    if (!jit_engine_) return false;
    return jit_engine_->patch_const32(site_name, new_val);
}

bool CompiledModule::patch_constant(std::string_view site_name, int64_t new_val) {
    if (!jit_engine_) return false;
    return jit_engine_->patch_const64(site_name, new_val);
}

bool CompiledModule::patch_call(std::string_view site_name, const void* new_target) {
    if (!jit_engine_) return false;
    return jit_engine_->patch_call(site_name, new_target);
}

bool CompiledModule::patch_call(std::string_view site_name, std::string_view new_target_fn) {
    if (!jit_engine_) return false;
    void* addr = get_symbol_address(new_target_fn);
    if (!addr) return false;
    return jit_engine_->patch_call(site_name, addr);
}

size_t CompiledModule::walk_stack(
    uintptr_t rbp,
    uintptr_t return_ip,
    brass_root_visitor_fn visitor,
    void* user_data
) const {
    if (!jit_engine_) return 0;
    return brass_stack_walk(rbp, return_ip, jit_engine_->stack_maps(), visitor, user_data);
}

size_t CompiledModule::walk_stack(
    uintptr_t rbp,
    uintptr_t return_ip,
    const std::function<void(void**)>& visitor
) const {
    if (!jit_engine_) return 0;
    return brass_stack_walk(rbp, return_ip, jit_engine_->stack_maps(), visitor);
}

RuntimeValue CompiledModule::invoke(std::string_view name, const std::vector<RuntimeValue>& args) {
    if (!jit_engine_) {
        throw std::runtime_error("CompiledModule::invoke: engine is null");
    }
    return jit_engine_->invoke(name, args);
}

RuntimeValue CompiledModule::resume(std::string_view name, uint32_t resume_id, const std::vector<RuntimeValue>& args) {
    if (!jit_engine_) {
        throw std::runtime_error("CompiledModule::resume: engine is null");
    }
    return jit_engine_->resume(name, resume_id, args);
}

HostEngine::HostEngine()
    : options_{Target::host(), true, false} {
}

HostEngine::HostEngine(const Target& target)
    : options_{target, true, false} {
}

HostEngine::HostEngine(const EngineOptions& options)
    : options_(options) {
}

void HostEngine::register_external_symbol(std::string_view name, void* address) {
    registered_symbols_[std::string(name)] = address;
}

void HostEngine::register_host_gc(HostGC* gc) {
    attached_gc_ = gc;
    if (gc) {
        set_active_host_gc(gc);
    }
}

std::unique_ptr<CompiledModule> HostEngine::compile(const Module& mod) {
    auto jit = std::make_unique<codegen::JitExecutionEngine>(options_.target);

    // Register user external symbols
    for (const auto& [name, addr] : registered_symbols_) {
        jit->register_external_symbol(name, addr);
    }

    // Register HostGC and RuntimeGC runtime bridge symbols
    if (attached_gc_ != nullptr) {
        set_active_host_gc(attached_gc_);
        jit->register_external_symbol("host_gc_safepoint", reinterpret_cast<void*>(&host_gc_safepoint));
        jit->register_external_symbol("host_gc_alloc", reinterpret_cast<void*>(&host_gc_alloc));
        jit->register_external_symbol("host_gc_alloc_nanbox", reinterpret_cast<void*>(&host_gc_alloc_nanbox));
        jit->register_external_symbol("host_gc_collect", reinterpret_cast<void*>(&host_gc_collect));
        jit->register_external_symbol("brass_gc_safepoint", reinterpret_cast<void*>(&host_gc_safepoint));
        jit->register_external_symbol("brass_gc_alloc", reinterpret_cast<void*>(&host_gc_alloc));
        jit->register_external_symbol("brass_gc_collect", reinterpret_cast<void*>(&host_gc_collect));
    } else {
        // Standard GC runtime symbols
        jit->register_external_symbol("brass_gc_safepoint", reinterpret_cast<void*>(&brass_gc_safepoint));
        jit->register_external_symbol("brass_gc_alloc", reinterpret_cast<void*>(&brass_gc_alloc));
        jit->register_external_symbol("brass_gc_collect", reinterpret_cast<void*>(&brass_gc_collect));
    }

    bool ok = jit->compile_and_load(mod);
    if (!ok) {
        return nullptr;
    }

    if (attached_gc_ != nullptr) {
        attached_gc_->set_stack_maps(&jit->stack_maps());
    }

    return std::make_unique<CompiledModule>(std::move(jit));
}

std::unique_ptr<CompiledModule> HostEngine::compile(Module& mod) {
    return compile(const_cast<const Module&>(mod));
}

} // namespace brass
