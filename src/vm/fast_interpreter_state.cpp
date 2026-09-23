// FastInterpreter construction, the per-thread current interpreter, the
// bytecode compile cache, and GC integration (allocation and root scanning).
// The dispatch loop lives in fast_interpreter.cpp.

#include "fast_interpreter_impl.hpp"

namespace brass {

FastInterpreter::FastInterpreter(size_t gc_semispace_size)
    : gc_(gc_semispace_size), alloca_arena_(std::make_unique<FastAllocaArena>()) {
    gc_.set_root_provider([this](std::vector<uintptr_t*>& roots) {
        this->collect_all_roots(roots);
    });
    register_builtin_host_functions();
}

FastInterpreter::~FastInterpreter() = default;

FastInterpreter::FastInterpreter(FastInterpreter&&) noexcept = default;
FastInterpreter& FastInterpreter::operator=(FastInterpreter&&) noexcept = default;

static thread_local FastInterpreter* s_current_fast_interp = nullptr;

FastInterpreter* FastInterpreter::current() noexcept {
    return s_current_fast_interp;
}

void FastInterpreter::set_current(FastInterpreter* interp) noexcept {
    s_current_fast_interp = interp;
}

void FastInterpreter::clear_compile_cache() noexcept {
    try {
        retire_caches();
        release_retired();
    } catch (...) {
        // Out of memory while retiring: nothing was freed, so running
        // frames stay valid; the caches are simply kept.
    }
}

const BytecodeFunction* FastInterpreter::get_or_compile(const Function& fn) {
    auto it = compiled_functions_.find(&fn);
    if (it != compiled_functions_.end()) {
        return it->second.get();
    }
    auto bfn = compiler_.compile(fn);
    const BytecodeFunction* ptr = bfn.get();
    compiled_functions_[&fn] = std::move(bfn);
    return ptr;
}

void FastInterpreter::set_generational_gc(GenerationalGC* gc) noexcept {
    gen_gc_ = gc;
    if (gen_gc_) {
        gen_gc_->set_root_provider([this](std::vector<uintptr_t*>& roots) {
            this->collect_all_roots(roots);
        });
    }
}

uintptr_t FastInterpreter::allocate_gc(size_t size, uint64_t pointer_mask, uint32_t type_tag) {
    // Roots come from the root provider installed in the constructor (and
    // set_generational_gc), which the collector asks only when it collects.
    if (gen_gc_) {
        return gen_gc_->allocate(size, pointer_mask, type_tag);
    }
    return gc_.allocate(size, pointer_mask, type_tag);
}

void FastInterpreter::collect_all_roots(std::vector<uintptr_t*>& roots) {
    // 1. Walk active frames on the call stack
    for (FastFrame* f = current_frame_; f != nullptr; f = f->caller) {
        for (uint32_t i = 0; i < f->num_registers; ++i) {
            if (f->registers[i] == 0) continue;
            bool is_gc = false;
            if (f->bfn && i < f->bfn->register_types.size()) {
                is_gc = f->bfn->register_types[i].is_gcref();
            }
            if (!is_gc) {
                uintptr_t val = static_cast<uintptr_t>(f->registers[i]);
                is_gc = gc_.is_valid_object(val) || (gen_gc_ && gen_gc_->is_valid_object(val));
            }
            if (is_gc) {
                roots.push_back(reinterpret_cast<uintptr_t*>(&f->registers[i]));
            }
        }
    }

    // 2. Scan suspended coroutines
    for (auto& [handle, coro] : active_coros_) {
        if (coro && !coro->is_done) {
            for (size_t i = 0; i < coro->registers.size(); ++i) {
                if (coro->registers[i] == 0) continue;
                bool is_gc = false;
                if (coro->bfn && i < coro->bfn->register_types.size()) {
                    is_gc = coro->bfn->register_types[i].is_gcref();
                }
                if (!is_gc) {
                    uintptr_t val = static_cast<uintptr_t>(coro->registers[i]);
                    is_gc = gc_.is_valid_object(val) || (gen_gc_ && gen_gc_->is_valid_object(val));
                }
                if (is_gc) {
                    roots.push_back(reinterpret_cast<uintptr_t*>(&coro->registers[i]));
                }
            }
        }
    }

    // 3. Scan current_exception_ if gcref
    if (current_exception_.is_gcref() && !current_exception_.is_null()) {
        roots.push_back(reinterpret_cast<uintptr_t*>(&current_exception_.raw_bits_ref()));
    }

    // 4. Scan last_deopt_ state map if gcref
    for (auto& val : last_deopt_.state_map) {
        if (val.is_gcref() && !val.is_null()) {
            roots.push_back(reinterpret_cast<uintptr_t*>(&val.raw_bits_ref()));
        }
    }
}

} // namespace brass
