#include "test_framework.hpp"
#include <brass/brass.hpp>
#include <brass/gc/mini_cheney.hpp>
#include <brass/gc/tlab.hpp>
#include <brass/embedding/host_gc.hpp>
#include <brass/runtime/deopt.hpp>
#include <brass/runtime/shape.hpp>
#include <brass/runtime/object.hpp>
#include <brass/runtime/inline_cache.hpp>
#include <brass/runtime/patcher.hpp>
#include <brass/runtime/exception.hpp>
#include <brass/interpreter/interpreter.hpp>
#include <brass/interpreter/frame.hpp>
#include <brass/interpreter/value.hpp>
#include <brass/codegen/jit_exec.hpp>
#include <brass/mir/builder.hpp>
#include <brass/mir/module.hpp>
#include <brass/mir/function.hpp>
#include <brass/mir/instruction.hpp>
#include <brass/object/object_writer.hpp>

#include <thread>
#include <vector>
#include <set>
#include <atomic>
#include <cmath>
#include <cstring>

using namespace brass;
using namespace brass::runtime;
using namespace brass::codegen;

namespace {

struct Win64DispatcherContextLayout {
    uint64_t ControlPc;
    uint64_t ImageBase;
    void* FunctionEntry;
    uint64_t EstablisherFrame;
    uint64_t TargetIp;
    void* ContextRecord;
    void* LanguageHandler;
    void* HandlerData;
};

} // namespace

// ============================================================================
// Deliverable 6.a: GCRef Preservation Surviving Moving GC Scavenge
// ============================================================================
TEST_CASE("Deopt Hardening - GCRef in InterpreterFrame Survives GC Scavenge") {
    MiniCheneyGC gc(128 * 1024);

    // Allocate an object in Cheney semispace
    uintptr_t orig_addr = gc.allocate(24, 0, 77);
    REQUIRE(orig_addr != 0);
    gc.write_field(orig_addr, 0, 0xCAFEBABEDEADBEEFULL);
    gc.write_field(orig_addr, 1, 0x1234567890ABCDEFULL);

    // 1. Create a DeoptFrame with GCRef kind
    DeoptFrame dframe;
    dframe.resume_id = 1;
    dframe.push_deopt_value(DeoptValue::gcref(orig_addr));
    CHECK_EQ(dframe.get_value(0).kind, DeoptValueKind::GcRef);
    CHECK_EQ(dframe.get_value(0).as_gcref(), orig_addr);

    // 2. Convert to RuntimeValues
    std::vector<RuntimeValue> rvals = dframe.to_runtime_values();
    REQUIRE_EQ(rvals.size(), 1u);
    CHECK(rvals[0].is_gcref());
    CHECK_EQ(rvals[0].as_gcref(), orig_addr);

    // 3. Set into an InterpreterFrame
    Module mod("test_gc_frame_mod");
    Function* fn = mod.create_function("fn_gc", Type::void_type(), {Type::gcref()});
    Builder b(*fn);
    BasicBlock* bb = b.append_block("entry");
    b.position_at_end(bb);
    Value* gcref_val = b.add_param(Type::gcref());

    InterpreterFrame frame(fn);
    frame.set_value(gcref_val, rvals[0]);

    // 4. Collect roots from InterpreterFrame
    std::vector<uintptr_t*> roots;
    frame.collect_roots(roots);
    REQUIRE_EQ(roots.size(), 1u);
    CHECK_EQ(*roots[0], orig_addr);

    // 5. Trigger moving GC scavenge
    gc.collect(roots);
    CHECK_EQ(gc.collection_count(), 1u);

    // 6. Verify pointer was relocated
    uintptr_t relocated_addr = *roots[0];
    CHECK_NE(relocated_addr, orig_addr);
    CHECK(gc.is_valid_object(relocated_addr));
    CHECK(!gc.is_valid_object(orig_addr));

    // 7. Verify the InterpreterFrame now contains the relocated pointer
    RuntimeValue updated_val = frame.get_value(gcref_val);
    CHECK(updated_val.is_gcref());
    CHECK_EQ(updated_val.as_gcref(), relocated_addr);

    // 8. Data integrity in evacuated object
    CHECK_EQ(gc.read_field(relocated_addr, 0), 0xCAFEBABEDEADBEEFULL);
    CHECK_EQ(gc.read_field(relocated_addr, 1), 0x1234567890ABCDEFULL);
    (void)bb;
}

// ============================================================================
// Deliverable 6.b: JIT Deopt Exit Reloading Float Registers (XMM0/D0)
// ============================================================================
TEST_CASE("Deopt Hardening - JIT Deopt Exit Reloads Float Registers (F64 & F32)") {
    Module mod("deopt_float_reload_mod");

    // 1. Function returning F64 with guard exit
    Function* fn_f64 = mod.create_function("guard_exit_f64", Type::f64(), {Type::f64(), Type::i32()});
    Builder b64(*fn_f64);
    BasicBlock* entry64 = b64.append_block("entry");
    b64.position_at_end(entry64);
    Value* in_f64 = b64.add_param(Type::f64());
    Value* cond64 = b64.add_param(Type::i32());
    Instruction* g64 = b64.build_guard(cond64, "", {in_f64});
    g64->set_resume_id(101);
    b64.build_ret(in_f64);

    // 2. Function returning F32 with guard exit
    Function* fn_f32 = mod.create_function("guard_exit_f32", Type::f32(), {Type::f32(), Type::i32()});
    Builder b32(*fn_f32);
    BasicBlock* entry32 = b32.append_block("entry");
    b32.position_at_end(entry32);
    Value* in_f32 = b32.add_param(Type::f32());
    Value* cond32 = b32.add_param(Type::i32());
    Instruction* g32 = b32.build_guard(cond32, "", {in_f32});
    g32->set_resume_id(102);
    b32.build_ret(in_f32);

    JitExecutionEngine engine;
    REQUIRE(engine.compile_and_load(mod));

    // Register deopt handler to return a designated floating-point bit pattern
    constexpr double kExpectedF64 = 9876.54321;
    constexpr float kExpectedF32 = 123.75f;

    register_deopt_handler([](const DeoptFrame& frame) -> void* {
        if (frame.resume_id == 101) {
            double d = kExpectedF64;
            uint64_t bits = 0;
            std::memcpy(&bits, &d, sizeof(double));
            return reinterpret_cast<void*>(static_cast<uintptr_t>(bits));
        } else if (frame.resume_id == 102) {
            float f = kExpectedF32;
            uint32_t bits = 0;
            std::memcpy(&bits, &f, sizeof(float));
            return reinterpret_cast<void*>(static_cast<uintptr_t>(bits));
        }
        return nullptr;
    });

    // Invoke F64: cond = 0 triggers guard deopt exit
    RuntimeValue res_f64 = engine.invoke("guard_exit_f64", {RuntimeValue::from_f64(1.0), RuntimeValue::from_i32(0)});
    CHECK(res_f64.is_f64());
    CHECK(std::abs(res_f64.as_f64() - kExpectedF64) < 1e-6);

    // Invoke F32: cond = 0 triggers guard deopt exit
    RuntimeValue res_f32 = engine.invoke("guard_exit_f32", {RuntimeValue::from_f32(2.0f), RuntimeValue::from_i32(0)});
    CHECK(res_f32.is_f32());
    CHECK(std::abs(res_f32.as_f32() - kExpectedF32) < 1e-4f);

    register_deopt_handler(nullptr);
}

// ============================================================================
// Deliverable 6.c: Multi-threaded Concurrent Shape Transitions and Lookups
// ============================================================================
TEST_CASE("Deopt Hardening - Concurrent Shape Transitions and Lookups") {
    ShapeRegistry registry;
    Shape* root = registry.get_root_shape();
    REQUIRE(root != nullptr);

    // Pre-populate shared base properties
    Shape* base = registry.transition_to(root, "shared_base_0");
    base = registry.transition_to(base, "shared_base_1");
    base = registry.transition_to(base, "shared_base_2");

    constexpr int kNumThreads = 8;
    constexpr int kOpsPerThread = 200;
    std::atomic<bool> start_flag{false};
    std::atomic<size_t> total_transitions{0};
    std::vector<std::thread> threads;
    threads.reserve(kNumThreads);

    for (int t = 0; t < kNumThreads; ++t) {
        threads.emplace_back([&, t]() {
            while (!start_flag.load(std::memory_order_acquire)) {
                std::this_thread::yield();
            }

            Shape* cur = base;
            for (int i = 0; i < kOpsPerThread; ++i) {
                // Concurrent property lookup
                auto slot_opt = cur->find_slot("shared_base_1");
                (void)slot_opt;

                // Concurrent transition lookup
                auto trans = cur->find_transition("shared_base_2");
                (void)trans;

                // Concurrent transition creation
                std::string prop = "th_" + std::to_string(t) + "_p_" + std::to_string(i);
                cur = registry.transition_to(cur, prop);
                total_transitions.fetch_add(1, std::memory_order_relaxed);

                // Read transitions map
                const auto& m = cur->string_transitions();
                (void)m;
            }
        });
    }

    start_flag.store(true, std::memory_order_release);
    for (auto& th : threads) {
        th.join();
    }

    CHECK_EQ(total_transitions.load(), static_cast<size_t>(kNumThreads * kOpsPerThread));
    CHECK(registry.shape_count() >= static_cast<size_t>(kNumThreads * kOpsPerThread));
}

// ============================================================================
// Deliverable 6.d: Monomorphic IC Slot Patching and Property Validation
// ============================================================================
TEST_CASE("Deopt Hardening - Monomorphic IC Slot Patching and Property Validation") {
    ShapeRegistry registry;
    Shape* root = registry.get_root_shape();

    // Shape 1: {target_prop} -> slot 0
    Shape* shape1 = registry.transition_to(root, "target_prop");
    REQUIRE(shape1 != nullptr);
    CHECK_EQ(shape1->find_slot("target_prop").value(), 0u);

    // Shape 2: {other1, other2, target_prop} -> slot 2
    Shape* s2_tmp = registry.transition_to(registry.transition_to(root, "other1"), "other2");
    Shape* shape2 = registry.transition_to(s2_tmp, "target_prop");
    REQUIRE(shape2 != nullptr);
    CHECK_EQ(shape2->find_slot("target_prop").value(), 2u);

    // Shape 3: {unrelated1, unrelated2} -> missing target_prop!
    Shape* shape3 = registry.transition_to(registry.transition_to(root, "unrelated1"), "unrelated2");
    REQUIRE(shape3 != nullptr);
    CHECK(!shape3->find_slot("target_prop").has_value());

    InlineCache ic(88, "target_prop", 0, /*is_load=*/true);
    int64_t shape_patch_dest = 0;
    int32_t slot_patch_dest = -1;
    ic.set_patch_point(&shape_patch_dest);
    ic.set_slot_patch_point(&slot_patch_dest);

    constexpr int32_t kBaseOffset = static_cast<int32_t>(offsetof(DynamicObject, inline_slots));

    // Case 1: Patch with shape1 (slot 0)
    bool ok1 = patch_monomorphic_ic(ic, shape1, 0);
    CHECK(ok1);
    CHECK_EQ(shape_patch_dest, reinterpret_cast<int64_t>(shape1));
    CHECK_EQ(slot_patch_dest, kBaseOffset + 0); // slot 0 offset

    // Case 2: Patch with shape2 (slot 2)
    // Note: Pass a dummy slot = 999 to verify patch_monomorphic_ic extracts the true slot from shape2
    bool ok2 = patch_monomorphic_ic(ic, shape2, 999);
    CHECK(ok2);
    CHECK_EQ(shape_patch_dest, reinterpret_cast<int64_t>(shape2));
    CHECK_EQ(slot_patch_dest, kBaseOffset + 16); // slot 2 * 8 = 16 offset

    // Case 3: Patch with shape3 (does not contain target_prop!)
    shape_patch_dest = 0xAA;
    slot_patch_dest = 0x55;
    bool ok3 = patch_monomorphic_ic(ic, shape3, 0);
    CHECK(!ok3); // Must fail validation!
    CHECK_EQ(shape_patch_dest, 0xAA); // Unmodified
    CHECK_EQ(slot_patch_dest, 0x55);  // Unmodified

    // Case 4: Null shape validation
    bool ok4 = patch_monomorphic_ic(ic, nullptr, 0);
    CHECK(!ok4);
}

// ============================================================================
// Deliverable 6.e: Multi-threaded Concurrent TLAB Allocations
// ============================================================================
TEST_CASE("Deopt Hardening - Concurrent TLAB Thread-Local Disjoint Buffers") {
    HostGC gc(16 * 1024 * 1024);
    set_active_host_gc(&gc);

    constexpr int kNumThreads = 8;
    constexpr int kAllocsPerThread = 50;

    struct WorkerResult {
        bool top_matches = false;
        bool end_matches = false;
        bool all_non_null = true;
        std::vector<uintptr_t> allocs;
    };

    std::vector<WorkerResult> worker_results(kNumThreads);
    std::atomic<bool> start_flag{false};
    std::vector<std::thread> threads;
    threads.reserve(kNumThreads);

    for (int t = 0; t < kNumThreads; ++t) {
        threads.emplace_back([&, t]() {
            ThreadLocalAllocBuffer tlab;
            tlab.init(&gc, 32 * 1024);
            set_active_tlab(&tlab);

            // Verify thread-local pointers point to this thread's tlab
            worker_results[t].top_matches = (brass_current_thread_tlab_top() == &tlab.top);
            worker_results[t].end_matches = (brass_current_thread_tlab_end() == &tlab.end);

            while (!start_flag.load(std::memory_order_acquire)) {
                std::this_thread::yield();
            }

            for (int i = 0; i < kAllocsPerThread; ++i) {
                uintptr_t obj = tlab.allocate_fast(64, 0, 10);
                if (obj == 0) {
                    worker_results[t].all_non_null = false;
                } else {
                    worker_results[t].allocs.push_back(obj);
                }
            }

            gc.unregister_tlab(&tlab);
            set_active_tlab(nullptr);
        });
    }

    start_flag.store(true, std::memory_order_release);
    for (auto& th : threads) {
        th.join();
    }

    set_active_host_gc(nullptr);

    // Verify all thread checks in main thread
    std::set<uintptr_t> all_ptrs;
    for (int t = 0; t < kNumThreads; ++t) {
        CHECK(worker_results[t].top_matches);
        CHECK(worker_results[t].end_matches);
        CHECK(worker_results[t].all_non_null);
        CHECK_EQ(worker_results[t].allocs.size(), static_cast<size_t>(kAllocsPerThread));
        for (uintptr_t ptr : worker_results[t].allocs) {
            CHECK_EQ(ptr % 8, 0u); // 8-byte alignment for HostGC payload
            CHECK(gc.is_valid_object(ptr));
            auto [it, inserted] = all_ptrs.insert(ptr);
            CHECK(inserted); // Disjointness
        }
    }
    CHECK_EQ(all_ptrs.size(), static_cast<size_t>(kNumThreads * kAllocsPerThread));
}

// ============================================================================
// Deliverable 6.f: Exception Unwinding Callee-Saved Regs & SEH Personality
// ============================================================================
TEST_CASE("Deopt Hardening - Win64 SEH Personality Landing Pad Identification") {
    FunctionExceptionTable table("test_seh_fn", 0x1000, 0x2000);
    table.add_scope(0x10, 0x30, 0x100);
    table.add_scope(0x40, 0x60, 0x200);

    object::Section xdata;
    emit_win64_seh_scope_table(xdata, table);
    REQUIRE(!xdata.data.empty());

    Win64DispatcherContextLayout dc{};
    dc.ImageBase = 0x10000;
    dc.HandlerData = xdata.data.data();

    // 1. IP within Scope 0: [0x10, 0x30) -> Landing pad 0x100
    dc.ControlPc = 0x10020;
    int res0 = brass_seh_personality(nullptr, nullptr, nullptr, &dc);
    CHECK_EQ(res0, 0); // Target identified
    CHECK_EQ(dc.TargetIp, 0x10000 + 0x100);

    // 2. IP within Scope 1: [0x40, 0x60) -> Landing pad 0x200
    dc.ControlPc = 0x10050;
    int res1 = brass_seh_personality(nullptr, nullptr, nullptr, &dc);
    CHECK_EQ(res1, 0);
    CHECK_EQ(dc.TargetIp, 0x10000 + 0x200);

    // 3. IP outside any scope: 0x10035
    dc.ControlPc = 0x10035;
    dc.TargetIp = 0;
    int res_none = brass_seh_personality(nullptr, nullptr, nullptr, &dc);
    CHECK_EQ(res_none, 1); // ExceptionContinueSearch
    CHECK_EQ(dc.TargetIp, 0u);

    // 4. Fallback to global registry when HandlerData is null
    FunctionExceptionTable table_global("global_seh_fn", 0x5000, 0x6000);
    table_global.add_scope(0x15, 0x35, 0x350);
    get_global_exception_registry().register_function_mapping(0x5000, 0x1000, table_global);

    Win64DispatcherContextLayout dc_global{};
    dc_global.ImageBase = 0;
    dc_global.HandlerData = nullptr;
    dc_global.ControlPc = 0x5020;

    int res_global = brass_seh_personality(nullptr, nullptr, nullptr, &dc_global);
    CHECK_EQ(res_global, 0);
    CHECK_EQ(dc_global.TargetIp, 0x5000 + 0x350);

    get_global_exception_registry().unregister_function_mapping(0x5000);
}

TEST_CASE("Deopt Hardening - Intervening Frame Callee-Saved Register Restoration") {
    // 3-level function call:
    // @outer calls @middle with invoke (has landing pad)
    // @middle performs heavy computations with multiple local variables (forcing callee-saved registers)
    // @middle calls @thrower with invoke (unwind target rethrows to @outer)
    // @thrower throws an exception
    // Landing pad in @outer catches and returns computed result
    Module mod("intervening_unwind_mod");
    Builder b(mod);

    // 1. thrower: func @thrower(%val: i64) -> i64
    Function* thrower = mod.create_function("thrower", Type::i64(), {Type::i64()});
    b.set_function(thrower);
    BasicBlock* th_bb = b.append_block("entry");
    b.position_at_end(th_bb);
    Value* th_arg = b.add_block_param(th_bb, Type::i64());
    b.build_throw(th_arg);
    thrower->rebuild_cfg_predecessors();

    // 2. middle: func @middle(%x: i64) -> i64
    Function* middle = mod.create_function("middle", Type::i64(), {Type::i64()});
    b.set_function(middle);
    BasicBlock* m_entry = b.append_block("entry");
    BasicBlock* m_norm = b.append_block("m_norm");
    BasicBlock* m_unw = b.append_block("m_unw");

    b.position_at_end(m_entry);
    Value* m_arg = b.add_block_param(m_entry, Type::i64());

    // Use multiple distinct SSA values to ensure callee-saved registers are used/spilled
    Value* c1 = b.build_iconst_i64(10);
    Value* c2 = b.build_iconst_i64(20);
    Value* c3 = b.build_iconst_i64(30);
    Value* c4 = b.build_iconst_i64(40);
    Value* s1 = b.build_add(m_arg, c1);
    Value* s2 = b.build_add(s1, c2);
    Value* s3 = b.build_add(s2, c3);
    Value* s4 = b.build_add(s3, c4);

    Instruction* inv_th = b.build_invoke("thrower", Type::i64(), {s4}, m_norm, m_unw);

    b.position_at_end(m_norm);
    b.build_ret(inv_th->result());

    b.position_at_end(m_unw);
    Value* caught_m = b.build_landing_pad(Type::i64());
    b.build_resume(caught_m); // Rethrow up to outer
    middle->rebuild_cfg_predecessors();

    // 3. outer: func @outer(%base: i64) -> i64
    Function* outer = mod.create_function("outer", Type::i64(), {Type::i64()});
    b.set_function(outer);
    BasicBlock* out_entry = b.append_block("entry");
    BasicBlock* out_norm = b.append_block("out_norm");
    BasicBlock* out_unw = b.append_block("out_unw");

    b.position_at_end(out_entry);
    Value* out_arg = b.add_block_param(out_entry, Type::i64());
    Instruction* inv_m = b.build_invoke("middle", Type::i64(), {out_arg}, out_norm, out_unw);

    b.position_at_end(out_norm);
    b.build_ret(inv_m->result());

    b.position_at_end(out_unw);
    Value* caught_out = b.build_landing_pad(Type::i64());
    Value* out_final = b.build_add(caught_out, b.build_iconst_i64(500));
    b.build_ret(out_final);
    outer->rebuild_cfg_predecessors();

    JitExecutionEngine jit;
    REQUIRE(jit.compile_and_load(mod));

    typedef int64_t (*OuterFn)(int64_t);
    OuterFn outer_fn = jit.get_function_ptr<OuterFn>("outer");
    REQUIRE(outer_fn != nullptr);

    // Call outer with 5:
    // middle computes 5 + 10 + 20 + 30 + 40 = 105
    // thrower throws 105
    // middle catches 105 and resumes (rethrows) to outer
    // outer catches 105 and adds 500 = 605
    int64_t result = outer_fn(5);
    CHECK_EQ(result, 605);
}
