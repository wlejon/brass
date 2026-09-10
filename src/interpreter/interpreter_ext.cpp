#include <brass/interpreter/interpreter.hpp>
#include <brass/gc/generational_gc.hpp>
#include <brass/gc/runtime_gc.hpp>
#include <iostream>
#include <cmath>

namespace brass {

void Interpreter::register_builtin_host_functions() {
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

void Interpreter::collect_all_roots(std::vector<uintptr_t*>& roots) {
    for (InterpreterFrame* f = current_frame_; f != nullptr; f = f->caller()) {
        f->collect_roots(roots);
    }
}

uintptr_t Interpreter::allocate_gc(size_t size, uint64_t pointer_mask, uint32_t type_tag) {
    std::vector<uintptr_t*> roots;
    collect_all_roots(roots);
    return gc_.allocate(size, pointer_mask, type_tag, roots);
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
    module_ = &mod;
    const Function* fn = mod.get_function(entry_name);
    if (!fn) {
        throw InterpreterException("Entry function not found in module: " + std::string(entry_name));
    }
    return execute_function(*fn, args);
}

} // namespace brass
