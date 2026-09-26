#include "test_framework.hpp"
#include <brass/brass.hpp>
#include <brass/gc/heap.hpp>
#include <brass/runtime/deopt.hpp>
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

// ============================================================================
// Deliverable 6.a: GCRef Preservation Surviving Moving GC Scavenge
// ============================================================================
TEST_CASE("Deopt Hardening - GCRef in InterpreterFrame Survives GC Scavenge") {
    gc::HeapConfig config;
    config.read_environment = false;
    gc::Heap gc(config);

    // Allocate an object in the young generation
    uintptr_t orig_addr = gc.allocate_masked(24, 0, 77);
    REQUIRE(orig_addr != 0);
    gc.store(orig_addr, 0, 0xCAFEBABEDEADBEEFULL);
    gc.store(orig_addr, 1, 0x1234567890ABCDEFULL);

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
    const auto source = gc.add_root_source([&roots](gc::Tracer& tracer) {
        for (uintptr_t* slot : roots) tracer.visit_ref(slot);
    });
    gc.collect(gc::CollectionKind::Minor);
    gc.remove_root_source(source);
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
    CHECK_EQ(gc::Heap::load(relocated_addr, 0), 0xCAFEBABEDEADBEEFULL);
    CHECK_EQ(gc::Heap::load(relocated_addr, 1), 0x1234567890ABCDEFULL);
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
// Deliverable 6.f: Exception Unwinding Callee-Saved Regs & SEH Personality
// ============================================================================
TEST_CASE("Deopt Hardening - Win64 SEH Personality Landing Pad Identification") {
    FunctionExceptionTable table("test_seh_fn", 0x1000, 0x2000);
    table.add_scope(0x10, 0x30, 0x100);
    table.add_scope(0x40, 0x60, 0x200);

    object::Section xdata;
    emit_win64_seh_scope_table(xdata, table, "test_seh_fn");
    REQUIRE(!xdata.data.empty());

    // Every begin/end/pad entry is image-relative: an ADDR32NB relocation
    // against the function whose addend is the function-relative offset.
    const uint32_t expected_offsets[] = {0x10, 0x30, 0x100, 0x40, 0x60, 0x200};
    REQUIRE_EQ(xdata.relocations.size(), 6u);
    for (size_t i = 0; i < 6; ++i) {
        const auto& r = xdata.relocations[i];
        CHECK(r.kind == object::RelocKind::Addr32NB);
        CHECK_EQ(r.symbol_name, "test_seh_fn");
        CHECK_EQ(r.offset, 4 + i * 4);
        CHECK_EQ(r.addend, static_cast<int64_t>(expected_offsets[i]));
    }
    {
        object::Section unnamed;
        bool threw = false;
        try {
            emit_win64_seh_scope_table(unnamed, table, "");
        } catch (const std::runtime_error&) {
            threw = true;
        }
        CHECK(threw);
    }

    // With the function linked at RVA 0 the relocated table equals the raw
    // one, so ImageBase 0x10000 puts the scopes at 0x10010.. and so on.
    const uint64_t image_base = 0x10000;
    const void* handler_data = xdata.data.data();

    // 1. Return address within Scope 0: [0x10, 0x30) -> Landing pad 0x100
    CHECK_EQ(brass_seh_find_landing_pad(0x10020, image_base, handler_data), 0x10000 + 0x100);

    // 2. Return address within Scope 1: [0x40, 0x60) -> Landing pad 0x200
    CHECK_EQ(brass_seh_find_landing_pad(0x10050, image_base, handler_data), 0x10000 + 0x200);

    // 3. Outside any scope: 0x10035 (the call at 0x10034)
    CHECK_EQ(brass_seh_find_landing_pad(0x10035, image_base, handler_data), 0u);

    // A return address just past a scope's end belongs to the call inside it.
    CHECK_EQ(brass_seh_find_landing_pad(0x10030, image_base, handler_data), 0x10000 + 0x100);

    // The personality claims nothing that is not a brass exception.
    CHECK_EQ(brass_default_seh_personality(nullptr, nullptr, nullptr, nullptr), 1);

    // 4. Fallback to global registry when HandlerData is null
    FunctionExceptionTable table_global("global_seh_fn", 0x5000, 0x6000);
    table_global.add_scope(0x15, 0x35, 0x350);
    get_global_exception_registry().register_function_mapping(0x5000, 0x1000, table_global);

    CHECK_EQ(brass_seh_find_landing_pad(0x5020, 0, nullptr), 0x5000 + 0x350);

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
