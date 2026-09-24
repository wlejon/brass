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
