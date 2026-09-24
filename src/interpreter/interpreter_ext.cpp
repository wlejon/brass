#include <brass/interpreter/interpreter.hpp>
#include <brass/gc/generational_gc.hpp>
#include <brass/gc/runtime_gc.hpp>
#include <brass/gc/host_heap.hpp>
#include <brass/runtime/parallel_runtime.hpp>
#include <brass/runtime/code_installer.hpp>
#include <brass/runtime/multi_tier_pipeline.hpp>
#include <iostream>
#include <cmath>

namespace brass {

namespace {
thread_local void* s_interp_red_target = nullptr;
thread_local runtime::ReductionKind s_interp_red_kind = runtime::ReductionKind::None;
thread_local Interpreter* s_active_interp = nullptr;
} // namespace

Interpreter::ActiveScope::ActiveScope(Interpreter* interp) noexcept : prev(s_active_interp) {
    s_active_interp = interp;
}

Interpreter::ActiveScope::~ActiveScope() { s_active_interp = prev; }

Interpreter* Interpreter::active_on_thread() noexcept { return s_active_interp; }

void Interpreter::register_builtin_host_functions() {
    register_external_function("brass_parallel_alloc_context", [](Interpreter&, const std::vector<RuntimeValue>& args) -> RuntimeValue {
        uint64_t bytes = args.empty() ? 64ULL : args[0].as_u64();
        void* ptr = brass_parallel_alloc_context(bytes);
        return RuntimeValue::from_ptr(reinterpret_cast<uintptr_t>(ptr));
    });

    register_external_function("brass_parallel_free_context", [](Interpreter&, const std::vector<RuntimeValue>& args) -> RuntimeValue {
        if (!args.empty()) {
            brass_parallel_free_context(reinterpret_cast<void*>(args[0].as_ptr()));
        }
        return RuntimeValue::from_void();
    });

    register_external_function("brass_parallel_reduce_i64", [](Interpreter&, const std::vector<RuntimeValue>& args) -> RuntimeValue {
        if (!args.empty()) {
            int64_t val = args[0].as_i64();
            if (s_interp_red_target) {
                int64_t* ptr = reinterpret_cast<int64_t*>(s_interp_red_target);
                *ptr = runtime::combine_reduction_i64(s_interp_red_kind, *ptr, val);
            } else {
                brass_parallel_reduce_i64(val);
            }
        }
        return RuntimeValue::from_void();
    });

    register_external_function("brass_parallel_reduce_f64", [](Interpreter&, const std::vector<RuntimeValue>& args) -> RuntimeValue {
        if (!args.empty()) {
            double val = args[0].as_f64();
            if (s_interp_red_target) {
                double* ptr = reinterpret_cast<double*>(s_interp_red_target);
                *ptr = runtime::combine_reduction_f64(s_interp_red_kind, *ptr, val);
            } else {
                brass_parallel_reduce_f64(val);
            }
        }
        return RuntimeValue::from_void();
    });

    register_external_function("brass_parallel_for", [](Interpreter& interp, const std::vector<RuntimeValue>& args) -> RuntimeValue {
        if (args.size() < 6) return RuntimeValue::from_void();
        uint64_t trip_count = args[0].as_u64();
        uint64_t grain_size = args[1].as_u64();
        uintptr_t kernel_ptr = args[2].as_ptr();
        uintptr_t ctx_ptr = args[3].as_ptr();
        auto red_kind = static_cast<runtime::ReductionKind>(args[4].as_u64());
        void* red_target = reinterpret_cast<void*>(args[5].as_ptr());

        const Function* target_fn = interp.find_function_by_pointer(kernel_ptr);
        if (target_fn) {
            void* prev_target = s_interp_red_target;
            runtime::ReductionKind prev_kind = s_interp_red_kind;
            s_interp_red_target = red_target;
            s_interp_red_kind = red_kind;

            std::vector<RuntimeValue> kargs = {
                RuntimeValue::from_u64(0),
                RuntimeValue::from_u64(trip_count),
                RuntimeValue::from_ptr(ctx_ptr)
            };
            interp.run(*target_fn, kargs);

            s_interp_red_target = prev_target;
            s_interp_red_kind = prev_kind;
        } else {
            auto kernel_fn = reinterpret_cast<runtime::ParallelKernelFn>(kernel_ptr);
            brass_parallel_for(trip_count, grain_size, kernel_fn, reinterpret_cast<void*>(ctx_ptr), red_kind, red_target);
        }
        return RuntimeValue::from_void();
    });
    register_external_function("brass_gc_alloc", [](Interpreter& interp, const std::vector<RuntimeValue>& args) -> RuntimeValue {
        if (args.empty()) {
            throw InterpreterException("brass_gc_alloc requires at least 1 argument (size)");
        }
        size_t size = static_cast<size_t>(args[0].is_i32() ? args[0].as_u32() : args[0].as_u64());
        uint64_t pointer_mask = (args.size() > 1) ? args[1].as_u64() : 0ULL;
        uint32_t type_tag = (args.size() > 2) ? args[2].as_u32() : 0U;
        uintptr_t addr = interp.allocate_gc(size, pointer_mask, type_tag);
        return RuntimeValue::from_gcref(addr);
    });

    register_external_function("brass_gc_collect", [](Interpreter& interp, const std::vector<RuntimeValue>&) -> RuntimeValue {
        if (auto* host = host_heap()) {
            host->collect();
            return RuntimeValue::from_void();
        }
        // A borrowed heap sees this interpreter's frames through the
        // borrower's ThreadRootsScope (borrow_generational_gc).
        if (GenerationalGC* gen = interp.borrowed_generational_gc()) {
            gen->collect();
            return RuntimeValue::from_void();
        }
        interp.gc().collect();
        return RuntimeValue::from_void();
    });

    register_external_function("sqrt", [](Interpreter&, const std::vector<RuntimeValue>& args) -> RuntimeValue {
        if (args.empty()) return RuntimeValue::from_f64(0.0);
        return RuntimeValue::from_f64(std::sqrt(args[0].as_f64()));
    });

    register_external_function("fabs", [](Interpreter&, const std::vector<RuntimeValue>& args) -> RuntimeValue {
        if (args.empty()) return RuntimeValue::from_f64(0.0);
        return RuntimeValue::from_f64(std::fabs(args[0].as_f64()));
    });

    register_external_function("floor", [](Interpreter&, const std::vector<RuntimeValue>& args) -> RuntimeValue {
        if (args.empty()) return RuntimeValue::from_f64(0.0);
        return RuntimeValue::from_f64(std::floor(args[0].as_f64()));
    });

    register_external_function("ceil", [](Interpreter&, const std::vector<RuntimeValue>& args) -> RuntimeValue {
        if (args.empty()) return RuntimeValue::from_f64(0.0);
        return RuntimeValue::from_f64(std::ceil(args[0].as_f64()));
    });
}

RuntimeValue Interpreter::resume_from_native(const Function& fn, uint32_t resume_id,
                                             const std::vector<RuntimeValue>& state_values) {
    // resume_after_guard points module_ at fn's module; the frames below
    // keep running in theirs.
    struct ModuleRestore {
        const Module*& slot;
        const Module* saved;
        ~ModuleRestore() { slot = saved; }
    } restore{module_, module_};
    return resume_after_guard(fn, resume_id, state_values, nullptr);
}

RuntimeValue Interpreter::call_in_own_module(const Function& fn, const std::vector<RuntimeValue>& args) {
    struct ModuleRestore {
        const Module*& slot;
        const Module* saved;
        ~ModuleRestore() { slot = saved; }
    } restore{module_, module_};
    if (fn.parent()) module_ = fn.parent();
    return execute_function(fn, args);
}

void Interpreter::collect_all_roots(std::vector<uintptr_t*>& roots) {
    for (InterpreterFrame* f = current_frame_; f != nullptr; f = f->caller()) {
        f->collect_roots(roots);
    }
}

uintptr_t Interpreter::allocate_gc(size_t size, uint64_t pointer_mask, uint32_t type_tag) {
    if (host_heap()) return host_heap_allocate(size, pointer_mask, type_tag);
    // A borrowed heap sees this interpreter's frames through the borrower's
    // ThreadRootsScope, its own heap through its root provider.
    if (borrowed_gen_gc_) return borrowed_gen_gc_->allocate(size, pointer_mask, type_tag);
    return gc().allocate(size, pointer_mask, type_tag);
}

void Interpreter::register_external_function(std::string_view name, HostFn fn) {
    external_functions_[std::string(name)] = std::move(fn);
}

bool Interpreter::has_external_function(std::string_view name) const noexcept {
    return external_functions_.find(std::string(name)) != external_functions_.end();
}

void Interpreter::register_function_pointer(uintptr_t ptr, const Function* fn) {
    function_pointers_[ptr] = fn;
}

void Interpreter::register_function_pointer(uintptr_t ptr, HostFn fn) {
    host_function_pointers_[ptr] = std::move(fn);
}

const Function* Interpreter::find_function_by_pointer(uintptr_t ptr) const noexcept {
    auto it = function_pointers_.find(ptr);
    return it != function_pointers_.end() ? it->second : nullptr;
}

uintptr_t Interpreter::function_address(const Function& fn) {
    if (fn.block_count() == 0) {
        uintptr_t ptr = reinterpret_cast<uintptr_t>(&fn);
        register_function_pointer(ptr, &fn);
        return ptr;
    }
    uintptr_t ptr = reinterpret_cast<uintptr_t>(dispatch_table().pipeline().function_address(fn.name(), &fn));
    function_pointers_[ptr] = &fn;
    return ptr;
}

const Function* Interpreter::function_at(uintptr_t ptr) {
    if (auto it = function_pointers_.find(ptr); it != function_pointers_.end()) return it->second;
    const std::string name = dispatch_table().function_name_at(reinterpret_cast<const void*>(ptr));
    if (name.empty()) return nullptr;
    const Function* fn = nullptr;
    if (runtime::FunctionHandle* handle = dispatch_table().find(name)) fn = handle->mir_function();
    if (!fn && module_) fn = module_->get_function(name);
    if (fn) function_pointers_[ptr] = fn;
    return fn;
}

void Interpreter::patch_const(std::string_view symbol, int64_t val) {
    patched_consts_[std::string(symbol)] = val;
}

int64_t Interpreter::get_patched_const(std::string_view symbol, int64_t default_val) const {
    auto it = patched_consts_.find(std::string(symbol));
    if (it != patched_consts_.end()) {
        return it->second;
    }
    return default_val;
}

void Interpreter::patch_call(std::string_view site, std::string_view target) {
    patched_calls_[std::string(site)] = std::string(target);
}

std::string_view Interpreter::get_patched_call(std::string_view site, std::string_view default_callee) const {
    auto it = patched_calls_.find(std::string(site));
    if (it != patched_calls_.end()) {
        return it->second;
    }
    return default_callee;
}

RuntimeValue Interpreter::run(const Function& fn) {
    std::vector<RuntimeValue> empty_args;
    return run(fn, empty_args);
}

RuntimeValue Interpreter::run(std::string_view fn_name) {
    std::vector<RuntimeValue> empty_args;
    return run(fn_name, empty_args);
}

RuntimeValue Interpreter::run(std::string_view fn_name, const std::vector<RuntimeValue>& args) {
    runtime::ProgramScope program_scope(dispatch_table());
    if (module_) {
        const Function* fn = module_->get_function(fn_name);
        if (fn) {
            return execute_function(*fn, args);
        }
    }
    auto it = external_functions_.find(std::string(fn_name));
    if (it != external_functions_.end()) {
        return it->second(*this, args);
    }
    throw InterpreterException("Function not found: " + std::string(fn_name));
}

RuntimeValue Interpreter::run(const Module& mod, std::string_view entry_name) {
    std::vector<RuntimeValue> empty_args;
    return run(mod, entry_name, empty_args);
}

RuntimeValue Interpreter::run(const Module& mod, std::string_view entry_name, const std::vector<RuntimeValue>& args) {
    runtime::ProgramScope program_scope(dispatch_table());
    module_ = &mod;
    const Function* fn = mod.get_function(entry_name);
    if (!fn) {
        throw InterpreterException("Entry function not found in module: " + std::string(entry_name));
    }
    return execute_function(*fn, args);
}

} // namespace brass
