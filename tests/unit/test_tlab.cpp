#include "test_framework.hpp"
#include <brass/brass.hpp>
#include <brass/gc/tlab.hpp>
#include <brass/embedding/host_gc.hpp>
#include <brass/runtime/object.hpp>
#include <brass/runtime/shape.hpp>
#include <brass/il_translator/il_alloc_lowering.hpp>
#include <brass/il_translator/il_translator.hpp>
#include <brass/mir/builder.hpp>
#include <brass/mir/module.hpp>
#include <brass/mir/function.hpp>
#include <brass/mir/instruction.hpp>
#include <brass/codegen/jit_exec.hpp>
#include <vector>
#include <string>

using namespace brass;
using namespace brass::il;
using namespace brass::runtime;

TEST_CASE("TLAB - Direct bump allocation and object initialization") {
    HostGC gc(128 * 1024);
    set_active_host_gc(&gc);

    ThreadLocalAllocBuffer tlab;
    tlab.init(&gc, 16 * 1024);
    set_active_tlab(&tlab);

    CHECK_EQ(tlab.top, 0ULL);
    CHECK_EQ(tlab.end, 0ULL);
    CHECK_EQ(tlab.total_allocated, 0ULL);

    // Initial allocation triggers refill
    constexpr size_t PAYLOAD = sizeof(DynamicObject); // 104 bytes
    constexpr uint64_t MASK = DynamicObject::POINTER_MASK;
    constexpr uint32_t TAG = DynamicObject::TYPE_TAG_DYNAMIC_OBJECT;

    uintptr_t obj1 = tlab.allocate_fast(PAYLOAD, MASK, TAG);
    REQUIRE(obj1 != 0);
    CHECK(gc.is_valid_object(obj1));
    CHECK(tlab.top > 0);
    CHECK(tlab.end > tlab.top);
    CHECK_EQ(tlab.total_allocated, 104ULL);

    // Verify HostGcHeader
    const HostGcHeader* hdr1 = gc.get_header(obj1);
    REQUIRE(hdr1 != nullptr);
    CHECK_EQ(hdr1->size, 104U);
    CHECK_EQ(hdr1->type_tag, TAG);
    CHECK_EQ(hdr1->pointer_mask, MASK);
    CHECK_EQ(hdr1->forwarding_address, 0ULL);

    // Second allocation bumps pointer within the same chunk
    uintptr_t top_before = tlab.top;
    uintptr_t obj2 = tlab.allocate_fast(PAYLOAD, MASK, TAG);
    REQUIRE(obj2 != 0);
    CHECK(gc.is_valid_object(obj2));
    CHECK_EQ(tlab.total_allocated, 208ULL);

    constexpr size_t TOTAL_SIZE = sizeof(HostGcHeader) + PAYLOAD; // 24 + 104 = 128
    CHECK_EQ(tlab.top, top_before + TOTAL_SIZE);
    CHECK_EQ(obj2, obj1 + TOTAL_SIZE);

    set_active_tlab(nullptr);
    set_active_host_gc(nullptr);
}

TEST_CASE("TLAB - Exhaustion and automatic refill") {
    // 64 KB semispace, 1 KB TLAB chunk (holds ~7 objects of 128 bytes each)
    HostGC gc(64 * 1024);
    set_active_host_gc(&gc);

    ThreadLocalAllocBuffer tlab;
    tlab.init(&gc, 1024);
    set_active_tlab(&tlab);

    std::vector<uintptr_t> allocated_objs;
    constexpr size_t NUM_OBJS = 20;

    for (size_t i = 0; i < NUM_OBJS; ++i) {
        uintptr_t obj = tlab.allocate_fast(sizeof(DynamicObject), DynamicObject::POINTER_MASK, DynamicObject::TYPE_TAG_DYNAMIC_OBJECT);
        REQUIRE(obj != 0);
        CHECK(gc.is_valid_object(obj));
        allocated_objs.push_back(obj);
    }

    CHECK_EQ(allocated_objs.size(), NUM_OBJS);
    CHECK_EQ(tlab.total_allocated, NUM_OBJS * 104);

    // Verify all objects have valid headers
    for (uintptr_t obj : allocated_objs) {
        const HostGcHeader* hdr = gc.get_header(obj);
        REQUIRE(hdr != nullptr);
        CHECK_EQ(hdr->size, 104U);
        CHECK_EQ(hdr->type_tag, DynamicObject::TYPE_TAG_DYNAMIC_OBJECT);
    }

    set_active_tlab(nullptr);
    set_active_host_gc(nullptr);
}

TEST_CASE("TLAB - Collection during TLAB retirement and object evacuation") {
    HostGC gc(128 * 1024);
    set_active_host_gc(&gc);

    ThreadLocalAllocBuffer tlab;
    tlab.init(&gc, 16 * 1024);
    set_active_tlab(&tlab);

    // Allocate parent and child in TLAB
    uintptr_t parent = tlab.allocate_fast(sizeof(DynamicObject), DynamicObject::POINTER_MASK, DynamicObject::TYPE_TAG_DYNAMIC_OBJECT);
    uintptr_t child = tlab.allocate_fast(sizeof(DynamicObject), DynamicObject::POINTER_MASK, DynamicObject::TYPE_TAG_DYNAMIC_OBJECT);
    REQUIRE(parent != 0);
    REQUIRE(child != 0);

    // Field 2 (out_of_line_slots) in parent points to child (bit 2 in DynamicObject::POINTER_MASK)
    gc.write_field(parent, 2, child);
    // Write marker in child field 3
    gc.write_field(child, 3, 0xCAFEBABE12345678ULL);

    CHECK(tlab.top != 0);
    CHECK(tlab.end != 0);

    // Register parent as root
    uintptr_t root_parent = parent;
    gc.register_root(&root_parent);

    // Trigger collection
    gc.collect();

    // Verify TLAB was retired/reset
    CHECK_EQ(tlab.top, 0ULL);
    CHECK_EQ(tlab.end, 0ULL);

    // Verify parent was evacuated to new semispace
    CHECK(root_parent != parent);
    CHECK(gc.is_valid_object(root_parent));

    // Verify child pointer in parent was updated to new child address
    uintptr_t new_child = gc.read_field(root_parent, 2);
    CHECK(new_child != child);
    CHECK(gc.is_valid_object(new_child));
    CHECK_EQ(gc.read_field(new_child, 3), 0xCAFEBABE12345678ULL);

    // Subsequent allocation in TLAB works in new active semispace
    uintptr_t after_gc = tlab.allocate_fast(sizeof(DynamicObject), DynamicObject::POINTER_MASK, DynamicObject::TYPE_TAG_DYNAMIC_OBJECT);
    REQUIRE(after_gc != 0);
    CHECK(gc.is_valid_object(after_gc));

    gc.unregister_root(&root_parent);
    set_active_tlab(nullptr);
    set_active_host_gc(nullptr);
}

TEST_CASE("TLAB - Stress mode compliance") {
    HostGC gc(128 * 1024);
    gc.set_stress_mode(true);
    set_active_host_gc(&gc);

    ThreadLocalAllocBuffer tlab;
    tlab.init(&gc, 16 * 1024);
    set_active_tlab(&tlab);

    // In stress mode, allocate_fast must return 0 to force out-of-line GC
    uintptr_t obj = tlab.allocate_fast(sizeof(DynamicObject), DynamicObject::POINTER_MASK, DynamicObject::TYPE_TAG_DYNAMIC_OBJECT);
    CHECK_EQ(obj, 0ULL);
    CHECK_EQ(tlab.top, 0ULL);
    CHECK_EQ(tlab.end, 0ULL);

    // Refill also leaves top and end at 0 in stress mode
    tlab.refill(128);
    CHECK_EQ(tlab.top, 0ULL);
    CHECK_EQ(tlab.end, 0ULL);

    // host_gc_alloc runs and forces collection on each call in stress mode
    size_t col_before = gc.collection_count();
    uintptr_t direct_obj = host_gc_alloc(sizeof(DynamicObject), DynamicObject::POINTER_MASK, DynamicObject::TYPE_TAG_DYNAMIC_OBJECT);
    REQUIRE(direct_obj != 0);
    CHECK(gc.is_valid_object(direct_obj));
    CHECK_EQ(gc.collection_count(), col_before + 1);

    set_active_tlab(nullptr);
    set_active_host_gc(nullptr);
}

TEST_CASE("TLAB - Inlined SSA Diamond Generation in AllocLoweringHelper") {
    Module mod("test_alloc_lowering");
    Function* fn = mod.create_function("test_create_obj", Type::i64(), {});
    Builder b(mod);
    b.set_function(fn);

    BasicBlock* b0 = b.append_block("entry");
    b.position_at_end(b0);

    AllocLoweringHelper helper(/*enable_tlab=*/true);
    CHECK(helper.enable_tlab());

    Value* obj = helper.lower_create_object(b);
    b.build_ret(obj);

    bool has_top_load = false;
    bool has_end_load = false;
    bool has_ule = false;
    bool has_fallback_call = false;

    for (BasicBlock* bb : fn->blocks()) {
        for (Instruction* inst : *bb) {
            if (inst->opcode() == Opcode::func_addr) {
                if (inst->symbol() == "brass_tlab_top") has_top_load = true;
                if (inst->symbol() == "brass_tlab_end") has_end_load = true;
            }
            if (inst->opcode() == Opcode::ule) {
                has_ule = true;
            }
            if (inst->opcode() == Opcode::call) {
                if (inst->symbol() == "bronze_create_object") has_fallback_call = true;
            }
        }
    }

    CHECK(has_top_load);
    CHECK(has_end_load);
    CHECK(has_ule);
    CHECK(has_fallback_call);
}

TEST_CASE("TLAB - Bronze TLS SSA Diamond Generation in AllocLoweringHelper") {
    Module mod("test_alloc_lowering_bronze");
    Function* fn = mod.create_function("test_create_obj_bronze", Type::i64(), {});
    Builder b(mod);
    b.set_function(fn);

    BasicBlock* b0 = b.append_block("entry");
    b.position_at_end(b0);

    AllocLoweringHelper helper(/*enable_tlab=*/true, AllocLoweringHelper::Model::BronzeTLS);
    CHECK(helper.enable_tlab());
    CHECK(helper.model() == AllocLoweringHelper::Model::BronzeTLS);

    Value* obj = helper.lower_create_object(b);
    b.build_ret(obj);

    bool has_tls_call = false;
    bool has_cursor_load = false;
    bool has_limit_load = false;
    bool has_shape_load = false;
    bool has_fallback_call = false;

    for (BasicBlock* bb : fn->blocks()) {
        for (Instruction* inst : *bb) {
            if (inst->opcode() == Opcode::call) {
                if (inst->symbol() == "bronze_tls_block_addr") has_tls_call = true;
                if (inst->symbol() == "bronze_create_object") has_fallback_call = true;
            }
            if (inst->opcode() == Opcode::load) {
                if (inst->offset() == 24) has_cursor_load = true;
                if (inst->offset() == 32) has_limit_load = true;
                if (inst->offset() == 40) has_shape_load = true;
            }
        }
    }

    CHECK(has_tls_call);
    CHECK(has_cursor_load);
    CHECK(has_limit_load);
    CHECK(has_shape_load);
    CHECK(has_fallback_call);
}

TEST_CASE("TLAB - JIT execution with inlined TLAB allocations") {
    std::string il_src = R"(
module test_tlab_jit.js

func main() -> f64 {
  b0:
    %0: dynamic = create.array 4
    %1: dynamic = create.object
    %2: f64 = const.f64 42
    %3: dynamic = box.f64 %2
    prop.set %1, "val", %3
    %4: dynamic = prop.get %1, "val"
    %5: f64 = unbox.f64 %4
    ret %5
}
)";

    TranslatorOptions opts;
    opts.enable_tlab = true;
    opts.enable_optimizations = true;

    DiagnosticReporter diag;
    TranslationResult res = translate_bronze_il(il_src, opts, &diag);
    REQUIRE(res.success);
    REQUIRE(res.module != nullptr);

    HostGC gc(1024 * 1024);
    set_active_host_gc(&gc);

    // Ensure active TLAB is bound to this GC
    auto* tlab = get_active_tlab();
    REQUIRE(tlab != nullptr);
    tlab->init(&gc);

    codegen::JitExecutionEngine jit(Target::host());
    register_bronze_runtime_symbols(&jit);
    REQUIRE(jit.compile_and_load(*res.module));

    auto fn_ptr = jit.get_function_ptr<double(*)()>("main");
    REQUIRE(fn_ptr != nullptr);
    double r = fn_ptr();
    CHECK_EQ(r, 42.0);

    // Verify allocations took place via TLAB
    CHECK(tlab->total_allocated > 0);

    tlab->reset();
    set_active_host_gc(nullptr);
}
