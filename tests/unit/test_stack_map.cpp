#include "test_framework.hpp"
#include <brass/gc/stack_map.hpp>
#include <vector>

using namespace brass;

TEST_CASE("GC Stack Map - RootLocation and Record Construction") {
    // 1. Frame slot root
    StackMapRootLocation loc1 = StackMapRootLocation::frame_slot(-24);
    CHECK_EQ(static_cast<uint8_t>(loc1.kind), static_cast<uint8_t>(StackMapRootKind::FrameSlot));
    CHECK_EQ(loc1.offset_from_rbp, -24);
    CHECK_EQ(loc1.reg.is_valid(), false);

    // 2. Callee-saved register root
    codegen::PReg rbx = codegen::PReg::gpr(x64::GPR::RBX);
    StackMapRootLocation loc2 = StackMapRootLocation::callee_saved(-16, rbx);
    CHECK_EQ(static_cast<uint8_t>(loc2.kind), static_cast<uint8_t>(StackMapRootKind::CalleeSavedReg));
    CHECK_EQ(loc2.offset_from_rbp, -16);
    CHECK_EQ(loc2.reg.is_valid(), true);
    CHECK_EQ(loc2.reg.code, rbx.code);

    CHECK_NE(loc1, loc2);

    // 3. StackMapRecord
    StackMapRecord rec;
    rec.instruction_offset = 42;
    rec.frame_size = 64;
    rec.safepoint_id = 101;
    rec.add_root(loc1);
    rec.add_root(loc2);
    rec.add_root(loc1); // Duplicate addition should be ignored

    CHECK_EQ(rec.instruction_offset, 42U);
    CHECK_EQ(rec.frame_size, 64U);
    CHECK_EQ(rec.safepoint_id, 101U);
    CHECK_EQ(rec.roots.size(), 2ULL);
    CHECK_EQ(rec.roots[0], loc1);
    CHECK_EQ(rec.roots[1], loc2);
}

TEST_CASE("GC Stack Map - Function and Module Stack Maps IP Lookup") {
    ModuleStackMap mod_map;

    FunctionStackMap fn1;
    fn1.function_name = "func_alpha";
    fn1.code_offset = 0;
    fn1.code_size = 100;

    StackMapRecord rec1;
    rec1.instruction_offset = 25; // return address 25 bytes into func_alpha
    rec1.frame_size = 48;
    rec1.add_root(StackMapRootLocation::frame_slot(-16));
    rec1.add_root(StackMapRootLocation::frame_slot(-24));
    fn1.add_record(rec1);

    StackMapRecord rec2;
    rec2.instruction_offset = 60;
    rec2.frame_size = 48;
    rec2.add_root(StackMapRootLocation::frame_slot(-32));
    fn1.add_record(rec2);

    FunctionStackMap fn2;
    fn2.function_name = "func_beta";
    fn2.code_offset = 128;
    fn2.code_size = 80;

    StackMapRecord rec3;
    rec3.instruction_offset = 40;
    rec3.frame_size = 32;
    rec3.add_root(StackMapRootLocation::callee_saved(-8, codegen::PReg::gpr(x64::GPR::R12)));
    fn2.add_record(rec3);

    mod_map.add_function(std::move(fn1));
    mod_map.add_function(std::move(fn2));

    CHECK_EQ(mod_map.size(), 2ULL);
    CHECK(!mod_map.empty());

    // Before relocation (function_address == 0)
    CHECK(mod_map.find_function_by_ip(0x1000) == nullptr);
    CHECK(mod_map.find_record(0x1000) == nullptr);

    // Relocate to base address 0x100000
    uintptr_t base_addr = 0x100000;
    mod_map.relocate(base_addr);

    // func_alpha is at [0x100000 .. 0x100064]
    const FunctionStackMap* found_fn1 = mod_map.find_function_by_ip(0x100025);
    REQUIRE(found_fn1 != nullptr);
    CHECK_EQ(found_fn1->function_name, "func_alpha");

    // Lookup records by return IP
    const StackMapRecord* found_rec1 = mod_map.find_record(0x100000 + 25);
    REQUIRE(found_rec1 != nullptr);
    CHECK_EQ(found_rec1->instruction_offset, 25U);
    CHECK_EQ(found_rec1->frame_size, 48U);
    CHECK_EQ(found_rec1->roots.size(), 2ULL);
    CHECK_EQ(found_rec1->roots[0].offset_from_rbp, -16);
    CHECK_EQ(found_rec1->roots[1].offset_from_rbp, -24);

    const StackMapRecord* found_rec2 = mod_map.find_record(0x100000 + 60);
    REQUIRE(found_rec2 != nullptr);
    CHECK_EQ(found_rec2->instruction_offset, 60U);
    CHECK_EQ(found_rec2->roots.size(), 1ULL);
    CHECK_EQ(found_rec2->roots[0].offset_from_rbp, -32);

    // func_beta is at [0x100080 .. 0x1000D0]
    const StackMapRecord* found_rec3 = mod_map.find_record(0x100000 + 128 + 40);
    REQUIRE(found_rec3 != nullptr);
    CHECK_EQ(found_rec3->instruction_offset, 40U);
    CHECK_EQ(found_rec3->frame_size, 32U);
    CHECK_EQ(found_rec3->roots.size(), 1ULL);
    CHECK_EQ(found_rec3->roots[0].offset_from_rbp, -8);
    CHECK_EQ(static_cast<uint8_t>(found_rec3->roots[0].kind), static_cast<uint8_t>(StackMapRootKind::CalleeSavedReg));

    // Non-existent IP
    CHECK(mod_map.find_record(0x100000 + 99) == nullptr);
    CHECK(mod_map.find_record(0x200000) == nullptr);
}

TEST_CASE("GC Stack Map - Binary Serialization and Deserialization Roundtrip") {
    ModuleStackMap original;

    for (int f = 0; f < 5; ++f) {
        FunctionStackMap fn;
        fn.function_name = "test_fn_" + std::to_string(f);
        fn.code_offset = static_cast<uint32_t>(f * 256);
        fn.code_size = 200;

        for (int r = 0; r < 3; ++r) {
            StackMapRecord rec;
            rec.instruction_offset = static_cast<uint32_t>(16 + r * 32);
            rec.frame_size = static_cast<uint32_t>(32 + f * 16);
            rec.safepoint_id = static_cast<uint32_t>(1000 + f * 10 + r);

            for (int k = 0; k < r + 1; ++k) {
                if (k % 2 == 0) {
                    rec.add_root(StackMapRootLocation::frame_slot(-static_cast<int32_t>(8 + k * 8)));
                } else {
                    rec.add_root(StackMapRootLocation::callee_saved(
                        -static_cast<int32_t>(24 + k * 8),
                        codegen::PReg::gpr(static_cast<x64::GPR>(static_cast<uint8_t>(x64::GPR::RBX) + k))
                    ));
                }
            }
            fn.add_record(std::move(rec));
        }
        original.add_function(std::move(fn));
    }

    // 1. Encode
    std::vector<uint8_t> encoded = encode_stack_maps(original);
    CHECK(!encoded.empty());

    // Verify magic and version at start of binary buffer
    REQUIRE(encoded.size() >= 8);
    uint32_t magic = *reinterpret_cast<const uint32_t*>(encoded.data());
    uint32_t version = *reinterpret_cast<const uint32_t*>(encoded.data() + 4);
    CHECK_EQ(magic, STACK_MAP_MAGIC);
    CHECK_EQ(version, STACK_MAP_VERSION);

    // 2. Decode
    uintptr_t text_base = 0x400000;
    ModuleStackMap decoded = decode_stack_maps(encoded, text_base);

    REQUIRE_EQ(decoded.size(), original.size());

    for (size_t f = 0; f < original.functions().size(); ++f) {
        const auto& orig_fn = original.functions()[f];
        const auto* dec_fn = decoded.find_function_by_name(orig_fn.function_name);
        REQUIRE(dec_fn != nullptr);
        CHECK_EQ(dec_fn->code_offset, orig_fn.code_offset);
        CHECK_EQ(dec_fn->code_size, orig_fn.code_size);
        CHECK_EQ(dec_fn->function_address, text_base + orig_fn.code_offset);
        REQUIRE_EQ(dec_fn->records.size(), orig_fn.records.size());

        for (size_t r = 0; r < orig_fn.records.size(); ++r) {
            const auto& orig_rec = orig_fn.records[r];
            const auto& dec_rec = dec_fn->records[r];
            CHECK_EQ(dec_rec.instruction_offset, orig_rec.instruction_offset);
            CHECK_EQ(dec_rec.frame_size, orig_rec.frame_size);
            CHECK_EQ(dec_rec.safepoint_id, orig_rec.safepoint_id);
            REQUIRE_EQ(dec_rec.roots.size(), orig_rec.roots.size());

            for (size_t k = 0; k < orig_rec.roots.size(); ++k) {
                CHECK_EQ(dec_rec.roots[k].offset_from_rbp, orig_rec.roots[k].offset_from_rbp);
                CHECK_EQ(static_cast<uint8_t>(dec_rec.roots[k].kind), static_cast<uint8_t>(orig_rec.roots[k].kind));
                CHECK_EQ(dec_rec.roots[k].reg.code, orig_rec.roots[k].reg.code);
            }
        }
    }
}

TEST_CASE("GC Stack Map - Binary Deserialization Robustness and Errors") {
    // 1. Empty buffer
    ModuleStackMap empty_map = decode_stack_maps({});
    CHECK(empty_map.empty());

    // 2. Corrupt Magic
    std::vector<uint8_t> corrupt_magic = {0x00, 0x11, 0x22, 0x33, 0x01, 0x00, 0x00, 0x00};
    ModuleStackMap bad_magic_map = decode_stack_maps(corrupt_magic);
    CHECK(bad_magic_map.empty());

    // 3. Corrupt Version
    std::vector<uint8_t> corrupt_ver = {0x42, 0x53, 0x43, 0x4D, 0x99, 0x00, 0x00, 0x00};
    ModuleStackMap bad_ver_map = decode_stack_maps(corrupt_ver);
    CHECK(bad_ver_map.empty());

    // 4. Truncated payload
    std::vector<uint8_t> truncated = {0x42, 0x53, 0x43, 0x4D, 0x01, 0x00, 0x00, 0x00, 0x05, 0x00, 0x00, 0x00};
    ModuleStackMap trunc_map = decode_stack_maps(truncated);
    CHECK(trunc_map.empty());
}
