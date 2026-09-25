// FastInterpreter construction, the per-thread current interpreter, the
// bytecode compile cache, and GC integration (allocation and root scanning).
// The dispatch loop lives in fast_interpreter.cpp.

#include "fast_interpreter_impl.hpp"

namespace brass {

FastInterpreter::FastInterpreter(gc::Heap* heap) : alloca_arena_(std::make_unique<FastAllocaArena>()) {
    use_heap(heap ? heap : gc::Heap::current());
    register_builtin_host_functions();
}

FastInterpreter::FastInterpreter(const gc::HeapConfig& config)
    : own_heap_(std::make_unique<gc::Heap>(config)), alloca_arena_(std::make_unique<FastAllocaArena>()) {
    attach_heap(own_heap_.get());
    register_builtin_host_functions();
}

FastInterpreter::~FastInterpreter() {
    // The host adapter shares this heap; it goes first.
    host_adapter_.reset();
    detach_heap();
}

namespace {
void visit_fast_interpreter_roots(gc::Tracer& tracer, void* context) {
    auto* interp = static_cast<FastInterpreter*>(context);
    std::vector<uintptr_t*> roots;
    interp->collect_all_roots(roots);
    // A gcref register keeps whatever its frame computed last, a derived
    // gcref included: it is kept as an offset into its object.
    for (uintptr_t* slot : roots) tracer.visit_derived(reinterpret_cast<uint64_t*>(slot));
    // A register without a type may hold a reference (a value a host or
    // native callee returned untyped): kept when it names an object.
    roots.clear();
    interp->collect_untyped_registers(roots);
    for (uintptr_t* slot : roots) tracer.visit_conservative(reinterpret_cast<uint64_t*>(slot));
}
} // namespace

void FastInterpreter::attach_heap(gc::Heap* heap) {
    heap_ = heap;
    root_source_ = heap_->add_root_source(&visit_fast_interpreter_roots, this);
}

void FastInterpreter::detach_heap() noexcept {
    if (heap_) heap_->remove_root_source(root_source_);
    heap_ = nullptr;
    root_source_ = 0;
}

void FastInterpreter::use_heap(gc::Heap* heap) {
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
    if (host_adapter_) host_adapter_->use_heap(heap_);
}

static thread_local FastInterpreter* s_current_fast_interp = nullptr;

FastInterpreter* FastInterpreter::current() noexcept {
    return s_current_fast_interp;
}

void FastInterpreter::set_current(FastInterpreter* interp) noexcept {
    s_current_fast_interp = interp;
}

static thread_local FastFrame* s_thread_frame_top = nullptr;

FastFrame*& FastInterpreter::thread_frame_top() noexcept {
    return s_thread_frame_top;
}

void FastInterpreter::for_each_frame_on_thread(const std::function<bool(const InterpretedFrameInfo&)>& visit) {
    for (const FastFrame* f = s_thread_frame_top; f; f = f->thread_prev) {
        InterpretedFrameInfo info;
        info.frame_address = f;
        info.function = f->mir_fn;
        if (f->bfn) info.loc = f->bfn->get_line_info(f->pc);
        if (!visit(info)) return;
    }
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

uintptr_t FastInterpreter::allocate_gc(size_t size, uint64_t pointer_mask, uint32_t type_tag) {
    return heap_->allocate_masked(size, pointer_mask, type_tag);
}

void FastInterpreter::collect_all_roots(std::vector<uintptr_t*>& roots) {
    for_each_register([&roots](uint64_t* slot, bool typed_gcref) {
        if (typed_gcref) roots.push_back(reinterpret_cast<uintptr_t*>(slot));
    });

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

void FastInterpreter::collect_untyped_registers(std::vector<uintptr_t*>& slots) {
    for_each_register([&slots](uint64_t* slot, bool typed_gcref) {
        if (!typed_gcref) slots.push_back(reinterpret_cast<uintptr_t*>(slot));
    });
}

template <typename Fn>
void FastInterpreter::for_each_register(Fn&& fn) {
    // The active frames on the call stack, then the suspended coroutines.
    for (FastFrame* f = current_frame_; f != nullptr; f = f->caller) {
        const size_t typed = f->bfn ? f->bfn->register_types.size() : 0;
        for (uint32_t i = 0; i < f->num_registers; ++i) {
            if (f->registers[i] == 0) continue;
            fn(&f->registers[i], i < typed && f->bfn->register_types[i].is_gcref());
        }
    }
    for (auto& [handle, coro] : active_coros_) {
        if (!coro || coro->is_done) continue;
        const size_t typed = coro->bfn ? coro->bfn->register_types.size() : 0;
        for (size_t i = 0; i < coro->registers.size(); ++i) {
            if (coro->registers[i] == 0) continue;
            fn(&coro->registers[i], i < typed && coro->bfn->register_types[i].is_gcref());
        }
    }
}

} // namespace brass
