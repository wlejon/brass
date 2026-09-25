#include <brass/interpreter/interpreter.hpp>
#include <brass/gc/runtime_gc.hpp>
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

Interpreter::ActiveScope::ActiveScope(Interpreter* interp) noexcept
    : prev(s_active_interp), prev_heap(gc::Heap::current()) {
    s_active_interp = interp;
    gc::Heap::set_current(interp->heap_);
}

Interpreter::ActiveScope::~ActiveScope() {
    s_active_interp = prev;
    gc::Heap::set_current(prev_heap);
}

namespace {
void visit_interpreter_roots(gc::Tracer& tracer, void* context) {
    std::vector<uintptr_t*> roots;
    static_cast<Interpreter*>(context)->collect_all_roots(roots);
    // A frame keeps every gcref-typed value it computed, derived ones
    // included (MIR may leave one in a frame across a call it does not
    // outlive): each is kept as an offset into its object.
    for (uintptr_t* slot : roots) tracer.visit_derived(reinterpret_cast<uint64_t*>(slot));
}
} // namespace

void Interpreter::attach_heap(gc::Heap* heap) {
    heap_ = heap;
    root_source_ = heap_->add_root_source(&visit_interpreter_roots, this);
}

void Interpreter::detach_heap() noexcept {
    if (heap_) heap_->remove_root_source(root_source_);
    heap_ = nullptr;
    root_source_ = 0;
}

void Interpreter::use_heap(gc::Heap* heap) {
    if (heap && heap == heap_) return;
    if (!heap && own_heap_ && heap_ == own_heap_.get()) return;
    detach_heap();
    if (heap) {
        own_heap_.reset();
        attach_heap(heap);
    } else {
        if (!own_heap_) own_heap_ = std::make_unique<gc::Heap>();
        attach_heap(own_heap_.get());
    }
}

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
        interp.heap().collect(gc::CollectionKind::Full);
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

namespace {

// Bits a scalar type occupies (0 for void and vectors).
unsigned scalar_bits(Type t) noexcept {
    switch (t.kind()) {
        case TypeKind::I8: return 8;
        case TypeKind::I16: return 16;
        case TypeKind::I32:
        case TypeKind::F32: return 32;
        case TypeKind::I64:
        case TypeKind::F64:
        case TypeKind::Ptr:
        case TypeKind::GCRef:
        case TypeKind::Tagged: return 64;
        default: return 0;
    }
}

} // namespace

RuntimeValue retype_exception_value(RuntimeValue v, Type t) {
    if (t.is_void()) return v;
    const Type from = v.type();
    if (from == t) return v;
    // i64 is the carrier of bare bits, as the return register is in the
    // native tiers: a native throw arrives as i64 and a pad of any scalar
    // type says what the bits are, and an i64 pad takes any scalar's bits.
    // Otherwise a value is reinterpreted only at its own width (f64/ptr/gcref,
    // i32/f32). Anything else (another width, a vector for a scalar or the
    // reverse) is a pad that cannot hold what was thrown.
    const unsigned from_bits = scalar_bits(from);
    const unsigned to_bits = scalar_bits(t);
    const bool legal = to_bits != 0 && from_bits != 0 &&
                       (from.is_i64() || t.is_i64() || from_bits == to_bits);
    if (!legal) {
        throw InterpreterException("landing_pad." + to_string(t) + " caught a thrown " + to_string(from) +
                                   " value: the pad's type cannot hold it");
    }
    uint64_t bits = v.raw_bits();
    switch (t.kind()) {
        case TypeKind::I8: bits &= 0xFFull; break;
        case TypeKind::I16: bits &= 0xFFFFull; break;
        case TypeKind::I32:
        case TypeKind::F32: bits &= 0xFFFFFFFFull; break;
        default: break;
    }
    return RuntimeValue::from_bits(t, bits);
}

void Interpreter::collect_all_roots(std::vector<uintptr_t*>& roots) {
    for (InterpreterFrame* f = current_frame_; f != nullptr; f = f->caller()) {
        f->collect_roots(roots);
    }
    // The value in flight between a throw and its pad (a pad retypes a
    // native callee's throw to gcref: then it moves with its object).
    if (current_exception_.is_gc_root() && !current_exception_.is_null()) {
        roots.push_back(reinterpret_cast<uintptr_t*>(&current_exception_.raw_bits_ref()));
    }
}

uintptr_t Interpreter::allocate_gc(size_t size, uint64_t pointer_mask, uint32_t type_tag) {
    return heap_->allocate_masked(size, pointer_mask, type_tag);
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
