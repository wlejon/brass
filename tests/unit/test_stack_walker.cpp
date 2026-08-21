#include "test_framework.hpp"
#include <brass/gc/stack_walker.hpp>
#include <vector>
#include <cstdint>

using namespace brass;

TEST_CASE("Stack Walker - Single Synthetic Frame Root Extraction and Mutation") {
    // Construct a synthetic stack frame:
    // RBP points to saved caller RBP.
    // [RBP + 0] = 0 (saved caller RBP: bottom of stack)
    // [RBP + 8] = 0 (saved caller return IP)
    // [RBP - 8]  = root1
    // [RBP - 16] = root2
    // [RBP - 24] = root3
    alignas(16) uint64_t stack_mem[16] = {0};

    // Let RBP be at stack_mem[8]
    uint64_t* rbp_ptr = &stack_mem[8];
    uintptr_t rbp_addr = reinterpret_cast<uintptr_t>(rbp_ptr);

    // Frame setup
    rbp_ptr[0] = 0; // saved caller RBP
    rbp_ptr[1] = 0; // saved caller return IP

    // Put mock GC pointers in slots
    void* objA = reinterpret_cast<void*>(0x11111111ULL);
    void* objB = reinterpret_cast<void*>(0x22222222ULL);
    void* objC = reinterpret_cast<void*>(0x33333333ULL);

    *reinterpret_cast<void**>(rbp_addr - 8)  = objA;
    *reinterpret_cast<void**>(rbp_addr - 16) = objB;
    *reinterpret_cast<void**>(rbp_addr - 24) = objC;

    // Build stack map for return address 0x500020
    uintptr_t top_return_ip = 0x500020;
    ModuleStackMap stack_maps;
    FunctionStackMap fn;
    fn.function_name = "test_single_frame";
    fn.function_address = 0x500000;
    fn.code_size = 0x100;

    StackMapRecord rec;
    rec.instruction_offset = 0x20; // 0x500020 - 0x500000
    rec.frame_size = 32;
    rec.add_root(StackMapRootLocation::frame_slot(-8));
    rec.add_root(StackMapRootLocation::frame_slot(-16));
    rec.add_root(StackMapRootLocation::frame_slot(-24));
    fn.add_record(rec);
    stack_maps.add_function(fn);

    // Walk the stack
    std::vector<void**> visited_slots;
    std::vector<void*> original_values;

    size_t frames = brass_stack_walk(rbp_addr, top_return_ip, stack_maps, [&](void** slot) {
        visited_slots.push_back(slot);
        original_values.push_back(*slot);
        // Mutate root pointer in-place (simulating Cheney evacuation to new address)
        uintptr_t old_v = reinterpret_cast<uintptr_t>(*slot);
        *slot = reinterpret_cast<void*>(old_v + 0x1000);
    });

    CHECK_EQ(frames, 1ULL);
    REQUIRE_EQ(visited_slots.size(), 3ULL);
    CHECK_EQ(original_values[0], objA);
    CHECK_EQ(original_values[1], objB);
    CHECK_EQ(original_values[2], objC);

    // Verify stack memory was mutated in place!
    CHECK_EQ(*reinterpret_cast<void**>(rbp_addr - 8),  reinterpret_cast<void*>(0x11111111ULL + 0x1000));
    CHECK_EQ(*reinterpret_cast<void**>(rbp_addr - 16), reinterpret_cast<void*>(0x22222222ULL + 0x1000));
    CHECK_EQ(*reinterpret_cast<void**>(rbp_addr - 24), reinterpret_cast<void*>(0x33333333ULL + 0x1000));
}

TEST_CASE("Stack Walker - Multi-Level Call Stack Unwinding (4 Frames)") {
    // Create 4 chained synthetic stack frames in a single contiguous stack memory buffer:
    // Frame 0 (topmost / callee caller): RBP0 -> Frame 1 (RBP1) -> Frame 2 (RBP2) -> Frame 3 (RBP3) -> 0
    alignas(16) uint64_t stack_mem[64] = {0};

    uint64_t* rbp0 = &stack_mem[8];
    uint64_t* rbp1 = &stack_mem[20];
    uint64_t* rbp2 = &stack_mem[36];
    uint64_t* rbp3 = &stack_mem[52];

    uintptr_t rbp0_addr = reinterpret_cast<uintptr_t>(rbp0);
    uintptr_t rbp1_addr = reinterpret_cast<uintptr_t>(rbp1);
    uintptr_t rbp2_addr = reinterpret_cast<uintptr_t>(rbp2);
    uintptr_t rbp3_addr = reinterpret_cast<uintptr_t>(rbp3);

    // Link frames
    // Frame 0 links to Frame 1
    rbp0[0] = rbp1_addr;
    rbp0[1] = 0x601030; // Return IP inside Frame 1 (func_1)

    // Frame 1 links to Frame 2
    rbp1[0] = rbp2_addr;
    rbp1[1] = 0x602040; // Return IP inside Frame 2 (func_2)

    // Frame 2 links to Frame 3
    rbp2[0] = rbp3_addr;
    rbp2[1] = 0x603050; // Return IP inside Frame 3 (func_3)

    // Frame 3 is bottom
    rbp3[0] = 0;
    rbp3[1] = 0;

    // Populate live roots on each frame
    void* root_f0 = reinterpret_cast<void*>(0xAAAA0000ULL);
    void* root_f1_a = reinterpret_cast<void*>(0xBBBB0001ULL);
    void* root_f1_b = reinterpret_cast<void*>(0xBBBB0002ULL);
    void* root_f2 = reinterpret_cast<void*>(0xCCCC0000ULL);
    void* root_f3 = reinterpret_cast<void*>(0xDDDD0000ULL);

    *reinterpret_cast<void**>(rbp0_addr - 8) = root_f0;
    *reinterpret_cast<void**>(rbp1_addr - 8) = root_f1_a;
    *reinterpret_cast<void**>(rbp1_addr - 16) = root_f1_b;
    *reinterpret_cast<void**>(rbp2_addr - 24) = root_f2;
    *reinterpret_cast<void**>(rbp3_addr - 8) = root_f3;

    // Build ModuleStackMap for all 4 functions
    ModuleStackMap stack_maps;

    // Top frame (func_0): current return IP is 0x600020
    uintptr_t top_return_ip = 0x600020;
    {
        FunctionStackMap fn0;
        fn0.function_name = "func_0";
        fn0.function_address = 0x600000;
        fn0.code_size = 0x100;
        StackMapRecord rec0;
        rec0.instruction_offset = 0x20;
        rec0.frame_size = 32;
        rec0.add_root(StackMapRootLocation::frame_slot(-8));
        fn0.add_record(rec0);
        stack_maps.add_function(fn0);
    }
    // func_1
    {
        FunctionStackMap fn1;
        fn1.function_name = "func_1";
        fn1.function_address = 0x601000;
        fn1.code_size = 0x100;
        StackMapRecord rec1;
        rec1.instruction_offset = 0x30;
        rec1.frame_size = 48;
        rec1.add_root(StackMapRootLocation::frame_slot(-8));
        rec1.add_root(StackMapRootLocation::frame_slot(-16));
        fn1.add_record(rec1);
        stack_maps.add_function(fn1);
    }
    // func_2
    {
        FunctionStackMap fn2;
        fn2.function_name = "func_2";
        fn2.function_address = 0x602000;
        fn2.code_size = 0x100;
        StackMapRecord rec2;
        rec2.instruction_offset = 0x40;
        rec2.frame_size = 64;
        rec2.add_root(StackMapRootLocation::frame_slot(-24));
        fn2.add_record(rec2);
        stack_maps.add_function(fn2);
    }
    // func_3
    {
        FunctionStackMap fn3;
        fn3.function_name = "func_3";
        fn3.function_address = 0x603000;
        fn3.code_size = 0x100;
        StackMapRecord rec3;
        rec3.instruction_offset = 0x50;
        rec3.frame_size = 32;
        rec3.add_root(StackMapRootLocation::frame_slot(-8));
        fn3.add_record(rec3);
        stack_maps.add_function(fn3);
    }

    std::vector<void*> visited_roots;
    size_t frames_visited = brass_stack_walk(rbp0_addr, top_return_ip, stack_maps, [&](void** slot) {
        visited_roots.push_back(*slot);
        *slot = reinterpret_cast<void*>(reinterpret_cast<uintptr_t>(*slot) | 0x1);
    });

    CHECK_EQ(frames_visited, 4ULL);
    REQUIRE_EQ(visited_roots.size(), 5ULL);
    CHECK_EQ(visited_roots[0], root_f0);
    CHECK_EQ(visited_roots[1], root_f1_a);
    CHECK_EQ(visited_roots[2], root_f1_b);
    CHECK_EQ(visited_roots[3], root_f2);
    CHECK_EQ(visited_roots[4], root_f3);

    // Verify in-place updates on each stack frame
    CHECK_EQ(*reinterpret_cast<uintptr_t*>(rbp0_addr - 8),  0xAAAA0001ULL);
    CHECK_EQ(*reinterpret_cast<uintptr_t*>(rbp1_addr - 8),  0xBBBB0001ULL | 0x1);
    CHECK_EQ(*reinterpret_cast<uintptr_t*>(rbp1_addr - 16), 0xBBBB0002ULL | 0x1);
    CHECK_EQ(*reinterpret_cast<uintptr_t*>(rbp2_addr - 24), 0xCCCC0001ULL);
    CHECK_EQ(*reinterpret_cast<uintptr_t*>(rbp3_addr - 8),  0xDDDD0001ULL);
}

TEST_CASE("Stack Walker - Safety Guards (Misaligned, Cycles, Nulls)") {
    ModuleStackMap empty_maps;

    // 1. Null RBP
    CHECK_EQ(brass_stack_walk(0, 0x1000, empty_maps, nullptr, nullptr), 0ULL);

    // 2. Null Return IP
    alignas(16) uint64_t dummy[4] = {0};
    CHECK_EQ(brass_stack_walk(reinterpret_cast<uintptr_t>(&dummy[2]), 0, empty_maps, nullptr, nullptr), 0ULL);

    // 3. Misaligned RBP (not divisible by 8)
    CHECK_EQ(brass_stack_walk(reinterpret_cast<uintptr_t>(&dummy[2]) + 1, 0x1000, empty_maps, nullptr, nullptr), 0ULL);

    // 4. Self-cyclic RBP: rbp[0] == rbp
    dummy[2] = reinterpret_cast<uintptr_t>(&dummy[2]);
    dummy[3] = 0x1000;
    CHECK_EQ(brass_stack_walk(reinterpret_cast<uintptr_t>(&dummy[2]), 0x1000, empty_maps, nullptr, nullptr), 0ULL);
}
