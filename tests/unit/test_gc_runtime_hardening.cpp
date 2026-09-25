#include "test_framework.hpp"
#include <brass/brass.hpp>
#include <brass/gc/heap.hpp>
#include <brass/gc/runtime_gc.hpp>
#include <brass/gc/stack_walker.hpp>
#include <brass/gc/stack_map.hpp>
#include <brass/runtime/coroutine.hpp>
#include <brass/embedding/nanbox.hpp>
#include <brass/mir/builder.hpp>
#include <brass/mir/module.hpp>
#include <brass/mir/function.hpp>
#include <brass/mir/instruction.hpp>
#include <brass/mir/write_barrier_elim.hpp>
#include <brass/interpreter/interpreter.hpp>
#include <brass/codegen/lir.hpp>
#include <brass/codegen/emit_context.hpp>
#include <brass/codegen/linear_scan.hpp>
#include <vector>
#include <cstring>

using namespace brass;
using namespace brass::runtime;
using namespace brass::codegen;

namespace {

constexpr uint64_t BRONZE_OBJECT_TAG = 0xFFF1000000000000ULL;

// A heap that ignores BRASS_GC_* (these tests place objects by generation).
gc::HeapConfig plain_config() {
    gc::HeapConfig config;
    config.read_environment = false;
    return config;
}

size_t count_barriers(const Function& fn) {
    size_t count = 0;
    for (const BasicBlock* bb : fn.blocks()) {
        if (!bb) continue;
        for (const Instruction* inst : *bb) {
            if (inst && inst->opcode() == Opcode::write_barrier) {
                count++;
            }
        }
    }
    return count;
}

} // namespace

TEST_CASE("Hardening - Generational GC Stack Walker Bridge with Forwarded Roots") {
    gc::Heap gc(plain_config());

    // Allocate two young objects in eden
    uintptr_t young_obj1 = gc.allocate_masked(24, 0, 10);
    uintptr_t young_obj2 = gc.allocate_masked(24, 0, 11);
    REQUIRE(young_obj1 != 0);
    REQUIRE(young_obj2 != 0);
    CHECK(gc.is_young(young_obj1));
    CHECK(gc.is_young(young_obj2));

    gc.store(young_obj1, 0, 0x1111222233334444ULL);
    gc.store(young_obj2, 0, 0x5555666677778888ULL);

    // Set up synthetic stack frame simulating JIT execution
    alignas(16) uint64_t stack_mem[16] = {0};
    uint64_t* rbp_ptr = &stack_mem[8];
    uintptr_t rbp_addr = reinterpret_cast<uintptr_t>(rbp_ptr);
    rbp_ptr[0] = 0; // saved caller RBP
    rbp_ptr[1] = 0; // return IP

    // Place young objects in frame spill slots [RBP - 8] and [RBP - 16]
    *reinterpret_cast<void**>(rbp_addr - 8) = reinterpret_cast<void*>(young_obj1);
    *reinterpret_cast<void**>(rbp_addr - 16) = reinterpret_cast<void*>(young_obj2);

    // Build stack map for return address 0x600020
    uintptr_t return_ip = 0x600020;
    ModuleStackMap stack_maps;
    FunctionStackMap fn_map;
    fn_map.function_name = "test_jit_gen_gc_frame";
    fn_map.function_address = 0x600000;
    fn_map.code_size = 0x100;

    StackMapRecord rec;
    rec.instruction_offset = 0x20;
    rec.frame_size = 32;
    rec.add_root(StackMapRootLocation::frame_slot(-8));
    rec.add_root(StackMapRootLocation::frame_slot(-16));
    fn_map.add_record(rec);
    stack_maps.add_function(fn_map);

    // A young collection requested by the synthetic frame: the heap walks it
    // through the thread's stack maps.
    const ModuleStackMap* previous_maps = brass_get_active_stack_maps();
    brass_set_active_stack_maps(&stack_maps);
    gc.collect_at(gc::CollectionKind::Minor, rbp_addr, return_ip);
    brass_set_active_stack_maps(previous_maps);

    CHECK_EQ(gc.stats().minor_collections, 1ULL);

    // Read forwarded pointers from stack frame
    void* forwarded_slot1 = *reinterpret_cast<void**>(rbp_addr - 8);
    void* forwarded_slot2 = *reinterpret_cast<void**>(rbp_addr - 16);
    uintptr_t new_addr1 = reinterpret_cast<uintptr_t>(forwarded_slot1);
    uintptr_t new_addr2 = reinterpret_cast<uintptr_t>(forwarded_slot2);

    // Verify objects were evacuated to Survivor space and stack slots updated
    CHECK_NE(new_addr1, young_obj1);
    CHECK_NE(new_addr2, young_obj2);
    CHECK(gc.is_young(new_addr1));
    CHECK(gc.is_young(new_addr2));
    CHECK(gc.is_valid_object(new_addr1));
    CHECK(gc.is_valid_object(new_addr2));

    // Verify payload integrity at new addresses
    CHECK_EQ(gc::Heap::load(new_addr1, 0), 0x1111222233334444ULL);
    CHECK_EQ(gc::Heap::load(new_addr2, 0), 0x5555666677778888ULL);
}

TEST_CASE("Hardening - Suspended Coroutines Rooting and Lifecycle Across Collections") {
    gc::Heap gen_gc(plain_config());
    gc::HeapScope bind(gen_gc);

    // Create coroutine frame with 4 slots, slot 0 and 1 are pointer slots
    // (mask = 0x3). No generated caller: the test is not generated code.
    uintptr_t coro_addr = brass_coro_create_at(nullptr, 4, 0x3ULL, 0, 0);
    REQUIRE(coro_addr != 0);
    CHECK(is_active_coro_frame(coro_addr));
    CHECK(gen_gc.is_young(coro_addr));

    auto* frame = reinterpret_cast<BrassCoroFrame*>(coro_addr);

    // Allocate young children and store them in the coroutine's slots (the
    // frame is young: no barrier needed)
    uintptr_t child1 = gen_gc.allocate_masked(16, 0, 21);
    uintptr_t child2 = gen_gc.allocate_masked(16, 0, 22);
    gen_gc.store(child1, 0, 0xAAAAAAAAULL);
    gen_gc.store(child2, 0, 0xBBBBBBBBULL);

    frame->slots[0] = child1;
    frame->slots[1] = child2;

    uint64_t coro_root = coro_addr;
    gen_gc.add_root(&coro_root);
    // Trigger minor collection - coroutine is active so it must be rooted and evacuated!
    gen_gc.collect(gc::CollectionKind::Minor);
    CHECK_EQ(gen_gc.stats().minor_collections, 1ULL);

    // Frame must have been evacuated to a survivor space
    uintptr_t new_coro_addr = static_cast<uintptr_t>(coro_root);
    CHECK(new_coro_addr != coro_addr);
    CHECK(gen_gc.is_young(new_coro_addr));
    CHECK(is_active_coro_frame(new_coro_addr));
    // The registry follows the frame even though a root had already moved it
    // (the old copy's is_done then holds the forwarding address).
    CHECK(!is_active_coro_frame(coro_addr));
    size_t registered = 0;
    uintptr_t registered_addr = 0;
    visit_active_coro_frames([&](uintptr_t* slot) {
        ++registered;
        registered_addr = *slot;
    });
    CHECK_EQ(registered, size_t{1});
    CHECK_EQ(registered_addr, new_coro_addr);

    auto* new_frame = reinterpret_cast<BrassCoroFrame*>(new_coro_addr);
    uintptr_t new_child1 = static_cast<uintptr_t>(new_frame->slots[0]);
    uintptr_t new_child2 = static_cast<uintptr_t>(new_frame->slots[1]);

    CHECK(new_child1 != child1);
    CHECK(new_child2 != child2);
    CHECK(gen_gc.is_young(new_child1));
    CHECK(gen_gc.is_young(new_child2));
    CHECK_EQ(gc::Heap::load(new_child1, 0), 0xAAAAAAAAULL);
    CHECK_EQ(gc::Heap::load(new_child2, 0), 0xBBBBBBBBULL);

    // Destroy coroutine and ensure unregistered
    brass_coro_destroy(new_coro_addr);
    CHECK(!is_active_coro_frame(new_coro_addr));

    gen_gc.remove_root(&coro_root);
}

TEST_CASE("Hardening - Card Table Multi-Card Object Scanning") {
    gc::HeapConfig config = plain_config();
    config.tenure_age = 2;
    gc::Heap gc(config);

    // Allocate an object that spans multiple cards (512 bytes each) in the
    // old generation: 192 fields = 1536 bytes = 3 cards.
    constexpr size_t NUM_FIELDS = 192;
    constexpr size_t OBJ_SIZE = NUM_FIELDS * sizeof(uint64_t);

    // Allocated young, promoted by its survival of two minor collections:
    uint64_t root = gc.allocate_masked(OBJ_SIZE, 1ULL << 63, 77);
    gc.add_root(&root);
    gc.collect(gc::CollectionKind::Minor);
    gc.collect(gc::CollectionKind::Minor);
    CHECK(gc.is_old(root));
    uintptr_t large_obj = static_cast<uintptr_t>(root);

    // Allocate a small young object
    uintptr_t young_child = gc.allocate_masked(16, 0, 88);
    gc.store(young_child, 0, 0x12345678ULL);

    // Field 180 is at offset 180 * 8 = 1440 bytes into large_obj,
    // which falls onto a subsequent card of the multi-card object!
    gc.store(large_obj, 180, young_child);

    // Collect minor: multi-card scanning must detect young_child from dirty card
    gc.collect(gc::CollectionKind::Minor);
    CHECK_EQ(root, large_obj);  // old objects never move

    // Verify young_child was evacuated and field 180 updated
    uintptr_t forwarded_child = static_cast<uintptr_t>(gc::Heap::load(large_obj, 180));
    CHECK_NE(forwarded_child, young_child);
    CHECK(gc.is_young(forwarded_child));
    CHECK_EQ(gc::Heap::load(forwarded_child, 0), 0x12345678ULL);
    gc.remove_root(&root);
}

TEST_CASE("Hardening - Large Tenured Allocation Write Barrier Retention in WBE") {
    Module mod("test_wbe_large_tenured");
    Builder b(mod);

    Function* fn = mod.create_function("test_fn", Type::void_type(), {});
    b.set_function(fn);
    BasicBlock* entry = b.append_block("entry");
    Value* child = b.add_block_param(entry, Type::gcref());

    // 1. Small allocation (16 bytes) -> goes to nursery -> write barrier can be eliminated by Rule 2
    Value* sz16 = b.build_iconst_i64(16);
    Value* mask0 = b.build_iconst_i64(0);
    Value* tag1 = b.build_iconst_i32(1);
    Value* small_obj = b.build_call("brass_gc_alloc", Type::gcref(), {sz16, mask0, tag1});
    b.build_write_barrier(small_obj, child);

    // 2. Large allocation (300 KB > 256 KB) -> allocated directly in tenured -> write barrier MUST be retained!
    Value* sz300k = b.build_iconst_i64(300 * 1024);
    Value* large_obj = b.build_call("brass_gc_alloc", Type::gcref(), {sz300k, mask0, tag1});
    b.build_write_barrier(large_obj, child);

    b.build_ret_void();

    CHECK_EQ(count_barriers(*fn), 2ULL);

    WriteBarrierElimination wbe;
    bool changed = wbe.run_on_function(*fn);
    CHECK(changed);

    // Exactly 1 barrier eliminated (small nursery object), 1 barrier retained (large tenured object)
    CHECK_EQ(count_barriers(*fn), 1ULL);
    CHECK_EQ(wbe.stats().eliminated_young_provenance, 1U);
    CHECK_EQ(wbe.stats().remaining_barriers, 1U);
}

TEST_CASE("Hardening - Tagged Pointer Unmasking in Interpreter Write Barrier") {
    gc::Heap gen_gc(plain_config());

    // Allocate an object in the old generation
    uintptr_t tenured_obj = gen_gc.allocate(24, gc::mask_layout(0, 1), gc::kAllocOld);
    CHECK(gen_gc.is_old(tenured_obj));
    uint8_t& card = gen_gc.card_table_base()[(tenured_obj - gen_gc.old_base()) >> gc::kCardShift];

    // Allocate young object
    uintptr_t young_obj = gen_gc.allocate_masked(24, 0, 2);
    CHECK(gen_gc.is_young(young_obj));

    // Initially card should be clean
    card = gc::kCardClean;

    // Construct MIR with write_barrier on NaN-tagged pointers
    Module mod("test_tagged_interp_wb");
    Builder b(mod);
    Function* fn = mod.create_function("wb_test", Type::void_type(), {Type::i64(), Type::i64()});
    b.set_function(fn);
    BasicBlock* entry = b.append_block("entry");
    Value* p0 = b.add_block_param(entry, Type::i64());
    Value* p1 = b.add_block_param(entry, Type::i64());
    b.build_write_barrier(p0, p1);
    b.build_ret_void();

    Interpreter interp(&gen_gc);

    // Tag the pointers with NaN-box object tag
    RuntimeValue tagged_obj = RuntimeValue::from_bits(Type::i64(), BRONZE_OBJECT_TAG | tenured_obj);
    RuntimeValue tagged_young = RuntimeValue::from_bits(Type::i64(), BRONZE_OBJECT_TAG | young_obj);

    interp.run(*fn, {tagged_obj, tagged_young});

    // Verify card is now marked dirty despite tagged NaN bits!
    CHECK(card == gc::kCardDirty);
}

TEST_CASE("Hardening - Callee-Saved Register Invariant and Stack Walker Root Offset") {
    // Construct LIR function with a Call instruction holding a live GCRef in an assigned GPR
    LirFunction fn;
    fn.name = "test_callee_saved_gcref";
    Target target = Target::x64_linux();
    fn.calling_conv = CallingConvention::for_target(target);

    auto block = std::make_unique<LirBlock>(0, "entry");

    // VReg 0 is a GCRef
    VReg vr0 = fn.allocate_vreg(RegClass::GPR, 8, true);
    VRegInfo& info0 = fn.get_vreg_info(vr0);
    info0.is_spilled = false;
    info0.assigned_preg = codegen::PReg::gpr(x64::GPR::RBX); // Callee-saved RBX

    auto call_inst = std::make_unique<LirInst>(LirOpcode::Call);
    call_inst->callee_symbol = "brass_gc_safepoint";
    call_inst->live_gcrefs.push_back(vr0);
    call_inst->safepoint_id = 42;
    block->instructions.push_back(std::move(call_inst));

    auto ret_inst = std::make_unique<LirInst>(LirOpcode::Ret);
    block->instructions.push_back(std::move(ret_inst));
    fn.blocks.push_back(std::move(block));

    // Compile LIR to x64
    codegen::EmitContext emitter(fn, target);
    codegen::CompilationResult res = emitter.compile();

    // Verify stack map was generated for safepoint
    CHECK(res.stack_map.records.size() >= 1ULL);
    if (!res.stack_map.records.empty()) {
        const auto& rec = res.stack_map.records[0];
        CHECK_EQ(rec.safepoint_id, 42U);
        CHECK(rec.roots.size() >= 1ULL);
        if (!rec.roots.empty()) {
            // Designated spill slot offset must be negative from RBP
            CHECK(rec.roots[0].offset_from_rbp < 0);
            CHECK_EQ(rec.roots[0].kind, StackMapRootKind::CalleeSavedReg);
        }
    }
}
