// The process-wide code-address index of stack maps (code_stack_maps.hpp):
// registering and unregistering tens of thousands of functions one at a time
// (a program tiering up function by function) must stay cheap. Each
// registration used to copy the whole index, O(n^2) over a run.
#include "test_framework.hpp"
#include <brass/gc/code_stack_maps.hpp>
#include <chrono>
#include <cstdint>
#include <memory>
#include <vector>

using namespace brass;

namespace {

constexpr size_t kFunctions = 50000;
constexpr uintptr_t kCodeSize = 16;

FunctionStackMap fake_function(uintptr_t addr, size_t i) {
    FunctionStackMap fn;
    fn.function_name = "csm_fn_" + std::to_string(i);
    fn.function_address = addr;
    fn.code_size = static_cast<uint32_t>(kCodeSize);
    StackMapRecord rec;
    rec.instruction_offset = 4;
    fn.add_record(rec);
    return fn;
}

} // namespace

TEST_CASE("Code stack maps - 50k registrations and unregistrations stay cheap") {
    // Heap addresses, which no loaded code can overlap.
    std::vector<uint8_t> code(kFunctions * kCodeSize);
    const uintptr_t base = reinterpret_cast<uintptr_t>(code.data());
    auto addr = [&](size_t i) { return base + i * kCodeSize; };

    const auto start = std::chrono::steady_clock::now();
    std::vector<std::shared_ptr<const void>> tokens(kFunctions);
    for (size_t i = 0; i < kFunctions; ++i) {
        tokens[i] = register_code_stack_map(fake_function(addr(i), i));
        REQUIRE(tokens[i] != nullptr);
    }
    CHECK(code_stack_map_segment_count() <= 40);
    {
        auto snap = code_stack_map_snapshot();
        for (size_t i = 0; i < kFunctions; i += 97) {
            const FunctionStackMap* m = find_code_stack_map(snap.get(), addr(i) + 5);
            REQUIRE(m != nullptr);
            CHECK_EQ(m->function_address, addr(i));
        }
    }

    // Unregister every other function; the rest stay found, the dropped ones
    // do not, and their addresses can be registered again.
    for (size_t i = 0; i < kFunctions; i += 2) tokens[i].reset();
    {
        auto snap = code_stack_map_snapshot();
        for (size_t i = 0; i < 2000; ++i) {
            const FunctionStackMap* m = find_code_stack_map(snap.get(), addr(i) + 1);
            if (i % 2 == 0) {
                CHECK(m == nullptr);
            } else {
                REQUIRE(m != nullptr);
                CHECK_EQ(m->function_address, addr(i));
            }
        }
    }
    for (size_t i = 0; i < kFunctions; i += 2) {
        tokens[i] = register_code_stack_map(fake_function(addr(i), i));
        REQUIRE(tokens[i] != nullptr);
    }
    CHECK(code_stack_maps_cover(addr(0)));
    CHECK(code_stack_maps_cover(addr(kFunctions - 1) + kCodeSize - 1));
    CHECK(!code_stack_maps_cover(addr(kFunctions - 1) + kCodeSize));

    for (auto& t : tokens) t.reset();
    const auto elapsed = std::chrono::steady_clock::now() - start;
    for (size_t i = 0; i < kFunctions; i += 1013) CHECK(!code_stack_maps_cover(addr(i)));
    CHECK(code_stack_map_segment_count() <= 40);
    // Amortized O(log n) per operation takes well under a second here; the
    // old index copied and re-sorted every entry on each registration.
    const auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(elapsed).count();
    CHECK(ms < 5000);
}

TEST_CASE("Code stack maps - encoded maps are indexed now and decoded on first lookup") {
    constexpr size_t n = 64;
    std::vector<uint8_t> code(n * kCodeSize);
    const uintptr_t base = reinterpret_cast<uintptr_t>(code.data());
    auto addr = [&](size_t i) { return base + i * kCodeSize; };

    // One function's maps each, encoded as an image carries them; every
    // eighth has no records and is not indexed, like a decoded one.
    std::vector<std::vector<uint8_t>> blobs(n);
    std::vector<EncodedCodeStackMap> ranges;
    for (size_t i = 0; i < n; ++i) {
        ModuleStackMap one;
        FunctionStackMap fn = fake_function(0, i);
        fn.code_offset = 0;
        if (i % 8 == 7) fn.records.clear();
        else {
            StackMapRecord rec;
            rec.instruction_offset = 9;
            rec.frame_size = 48;
            rec.add_root(StackMapRootLocation::frame_slot(-16));
            fn.add_record(rec);
        }
        one.add_function(fn);
        blobs[i] = encode_stack_maps(one);
        ranges.push_back({addr(i), static_cast<uint32_t>(kCodeSize), blobs[i].data(),
                          static_cast<uint32_t>(blobs[i].size())});
    }
    auto token = register_encoded_code_stack_maps(ranges);
    REQUIRE(token != nullptr);
    for (size_t i = 0; i < n; ++i) CHECK_EQ(code_stack_maps_cover(addr(i) + 3), i % 8 != 7);

    auto snap = code_stack_map_snapshot();
    const FunctionStackMap* m = find_code_stack_map(snap.get(), addr(5) + 2);
    REQUIRE(m != nullptr);
    CHECK_EQ(m->function_address, addr(5));
    CHECK_EQ(m->code_size, static_cast<uint32_t>(kCodeSize));
    CHECK_EQ(m->function_name, std::string("csm_fn_5"));
    REQUIRE(m->records.size() == size_t{2});
    const StackMapRecord* rec = m->find_record_by_ip(addr(5) + 9);
    REQUIRE(rec != nullptr);
    CHECK_EQ(rec->frame_size, uint32_t{48});
    REQUIRE(rec->roots.size() == size_t{1});
    CHECK_EQ(rec->roots[0].offset_from_rbp, -16);
    // Decoded once: a second lookup finds the same map.
    CHECK(find_code_stack_map(snap.get(), addr(5) + 12) == m);
    CHECK(find_code_stack_map(snap.get(), addr(7) + 1) == nullptr);

    snap.reset();
    token.reset();
    for (size_t i = 0; i < n; ++i) CHECK(!code_stack_maps_cover(addr(i)));
}
