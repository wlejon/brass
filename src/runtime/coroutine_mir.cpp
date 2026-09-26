// One resume of a frame with a descriptor (CORO_FLAG_BODY), in its body's
// current best tier: native code once the body's program has compiled it,
// Tier 0 otherwise. Any tier may resume such a frame: an interpreter runs a
// Tier-0 body itself; brass_coro_resume (generated code, the host C API,
// MicrotaskQueue) runs it here.
#include <brass/runtime/coroutine.hpp>
#include <brass/runtime/code_installer.hpp>
#include <brass/runtime/host_symbols.hpp>
#include <brass/runtime/multi_tier_pipeline.hpp>
#include <brass/interpreter/interpreter.hpp>
#include <brass/vm/fast_interpreter.hpp>
#include <brass/gc/runtime_gc.hpp>
#include <brass/gc/heap.hpp>
#include <brass/gc/native_frames.hpp>
#include <brass/mir/coro_transform.hpp>
#include <brass/mir/function.hpp>
#include <cstdio>
#include <cstdlib>
#include <string>

namespace brass::runtime {

namespace {

[[noreturn]] void coro_body_fatal(const std::string& msg) {
    std::fprintf(stderr, "brass: fatal: resuming a coroutine frame: %s\n", msg.c_str());
    std::fflush(stderr);
    std::abort();
}

// The body in Tier 0 on this thread.
uint64_t run_tier0(const CoroBody& desc, const Function& body, uintptr_t& frame_addr) {
    // Re-enter the interpreter whose code called into native code on this
    // thread, as a native-to-Tier-0 call does (call_tier0_from_native): the
    // body then shares its heap, and its frames are that GC's roots. The
    // body runs in its own module, whichever the interpreter was running.
    // A MIR exception propagates as a C++ exception, as through the bridge.
    // The caller roots `frame_addr`; the body's frame parameter is re-read
    // from it afterwards.
    //
    // Only an interpreter of the body's own program: one running another
    // program (a host with several, whose frame of one is resumed while
    // another's code is on the stack) would resolve the body's calls and
    // module state through the wrong program, as call_tier0_from_native
    // declines to.
    FunctionDispatchTable* table = desc.table.load(std::memory_order_acquire);
    Interpreter* active = Interpreter::active_on_thread();
    if (active && (!table || &active->dispatch_table() == table)) {
        return active->call_in_own_module(body, {RuntimeValue::from_gcref(frame_addr)}).raw_bits();
    }
    FastInterpreter* active_fast = FastInterpreter::current();
    if (active_fast && (!table || &active_fast->dispatch_table() == table)) {
        return active_fast->call_from_native(body, {RuntimeValue::from_ptr(frame_addr)}).raw_bits();
    }
    // A host resumed it with no interpreter of its program running. A body
    // of a program runs in a fresh interpreter of that program's Tier-0
    // kind, set up as every other of its Tier-0 runs is (the FastInterpreter,
    // when the program uses it, carries module state the MIR interpreter
    // does not).
    if (table) return table->pipeline().run_coro_body_fresh(*table, body, frame_addr).raw_bits();
    // No program: a fresh MIR interpreter allocates from the heap native
    // code on this thread allocates from, whose objects the frame holds.
    Interpreter interp;  // the thread's current heap, else a private one
    // The host's symbols, as every interpreter brass sets up gets them
    // (run_fresh_tier0): a body calling a host external resolves it.
    install_host_symbols(interp);
    ThreadRootsScope roots([](void* ctx, std::vector<uintptr_t*>& out) {
        static_cast<Interpreter*>(ctx)->collect_all_roots(out);
    }, &interp);
    const RuntimeValue r = interp.call_in_own_module(body, {RuntimeValue::from_gcref(frame_addr)});
    // Without a shared heap the fresh interpreter's heap dies here: a gcref
    // into it, returned or kept in the frame, would dangle.
    if (interp.owns_heap()) {
        const gc::Heap& private_heap = interp.heap();
        auto inside = [&private_heap](uint64_t word) {
            return private_heap.contains(static_cast<uintptr_t>(word & gc::kAddressMask));
        };
        bool escapes = r.is_gcref() && inside(r.raw_bits());
        const auto* frame = reinterpret_cast<const BrassCoroFrame*>(frame_addr);
        for (uint32_t i = 0; i < frame->slot_count && !escapes; ++i) {
            escapes = inside(frame->slots[i]);
        }
        if (escapes) {
            coro_body_fatal("'" + std::string(body.name()) + "' ran in a fresh Tier-0 interpreter and kept a "
                            "gcref into its private heap; bind a gc::Heap for the thread (gc::HeapScope) or "
                            "resume the frame from an interpreter");
        }
    }
    return r.raw_bits();
}

} // namespace

uint64_t resume_coro_body(uintptr_t& frame_addr) {
    auto* frame = reinterpret_cast<BrassCoroFrame*>(frame_addr);
    const CoroBody* desc = coro_body(frame);
    if (!desc) coro_body_fatal("the frame has no body descriptor");
    if (void* entry = coro_body_native_entry(frame)) {
        // Native code entered from this C++ frame: a throw in it searches for
        // pads no further than here and leaves as a C++ exception, which
        // unwinds the scopes above (an SEH exception raised past this frame
        // would skip their destructors, /EHs).
        GeneratedCodeEntryScope entry_scope;
        using CoroFn = uint64_t (*)(BrassCoroFrame*);
        return reinterpret_cast<CoroFn>(entry)(frame);
    }
    const Function* body = desc->mir.load(std::memory_order_acquire);
    if (!body) {
        coro_body_fatal("the module of coroutine body '" + desc->name + "' was destroyed while a frame of it "
                        "was unfinished");
    }
    return run_tier0(*desc, *body, frame_addr);
}

} // namespace brass::runtime
