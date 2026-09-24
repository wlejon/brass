#include "test_framework.hpp"
#include <brass/brass.hpp>
#include <brass/gc/host_heap.hpp>
#include <brass/gc/runtime_gc.hpp>
#include <brass/runtime/coroutine.hpp>
#include <brass/interpreter/interpreter.hpp>
#include <brass/vm/fast_interpreter.hpp>
#include <brass/codegen/jit_exec.hpp>
#include <brass/mir/builder.hpp>
#include <brass/mir/module.hpp>
#include <memory>
#include <vector>

using namespace brass;
using namespace brass::runtime;

namespace {

// A host heap that remembers what brass asked of it. Its objects are plain
// zeroed blocks it owns; a real host would trace them by pointer_mask.
class RecordingHeap final : public HostHeap {
public:
    struct Allocation {
        uintptr_t payload;
        size_t size;
        uint64_t pointer_mask;
        uint32_t type_tag;
    };

    uintptr_t allocate(size_t size, uint64_t pointer_mask, uint32_t type_tag) override {
        blocks_.push_back(std::make_unique<uint64_t[]>((size + 7) / 8 + 1));
        const auto payload = reinterpret_cast<uintptr_t>(blocks_.back().get());
        allocations.push_back({payload, size, pointer_mask, type_tag});
        return payload;
    }
    void collect() override { ++collects; }
    void safepoint() override { ++safepoints; }

    bool owns(uintptr_t addr) const {
        for (const Allocation& a : allocations) {
            if (addr == a.payload) return true;
        }
        return false;
    }

    std::vector<Allocation> allocations;
    int collects = 0;
    int safepoints = 0;

private:
    std::vector<std::unique_ptr<uint64_t[]>> blocks_;
};

// Installs a heap for the scope of a test, so a failing CHECK cannot leave
// it installed for the next one.
struct HeapScope {
    explicit HeapScope(HostHeap* heap) { set_host_heap(heap); }
    ~HeapScope() { set_host_heap(nullptr); }
};

// alloc_and_collect() -> i64: brass_gc_alloc(24, 0b10, 7), then
// brass_gc_collect(), answering the allocation.
std::unique_ptr<Module> build_alloc_module() {
    auto mod = std::make_unique<Module>("host_heap_alloc");
    Builder b(*mod);
    Function* fn = mod->create_function("alloc_and_collect", Type::i64(), {});
    b.set_function(fn);
    BasicBlock* entry = b.append_block("entry");
    b.position_at_end(entry);
    Value* obj = b.build_call("brass_gc_alloc", Type::i64(),
                              {b.build_iconst_i64(24), b.build_iconst_i64(2), b.build_iconst_i32(7)});
    b.build_call("brass_gc_collect", Type::void_type(), {});
    b.build_ret(obj);
    fn->rebuild_cfg_predecessors();
    return mod;
}

} // namespace

TEST_CASE("HostHeap - none installed by default") {
    CHECK(host_heap() == nullptr);
}

TEST_CASE("HostHeap - brass_gc_alloc and safepoints from C++ reach the host heap") {
    RecordingHeap heap;
    HeapScope scope(&heap);

    const uintptr_t obj = brass_gc_alloc(40, 0x5, 11);
    REQUIRE(obj != 0);
    REQUIRE_EQ(heap.allocations.size(), 1u);
    CHECK_EQ(heap.allocations[0].payload, obj);
    CHECK_EQ(heap.allocations[0].size, 40u);
    CHECK_EQ(heap.allocations[0].pointer_mask, 0x5ULL);
    CHECK_EQ(heap.allocations[0].type_tag, 11u);

    brass_gc_safepoint();
    brass_gc_collect();
    CHECK_EQ(heap.safepoints, 2);
}

TEST_CASE("HostHeap - coroutine frames are host objects") {
    RecordingHeap heap;
    HeapScope scope(&heap);

    const uintptr_t frame = brass_coro_create(nullptr, 3, 0x1);
    REQUIRE(frame != 0);
    CHECK(heap.owns(frame));
    REQUIRE_EQ(heap.allocations.size(), 1u);
    CHECK_EQ(heap.allocations[0].type_tag, TYPE_TAG_CORO_FRAME);
    brass_coro_destroy(frame);
}

TEST_CASE("HostHeap - reference and fast interpreters allocate and collect through it") {
    RecordingHeap heap;
    HeapScope scope(&heap);
    auto mod = build_alloc_module();

    Interpreter oracle;
    const RuntimeValue o = oracle.run(*mod, "alloc_and_collect", {});
    CHECK(heap.owns(static_cast<uintptr_t>(o.raw_bits())));

    FastInterpreter fast;
    const RuntimeValue f = fast.run(*mod, "alloc_and_collect", {});
    CHECK(heap.owns(static_cast<uintptr_t>(f.raw_bits())));

    CHECK_EQ(heap.allocations.size(), 2u);
    CHECK_EQ(heap.collects, 2);
    for (const auto& a : heap.allocations) {
        CHECK_EQ(a.size, 24u);
        CHECK_EQ(a.pointer_mask, 2ULL);
        CHECK_EQ(a.type_tag, 7u);
    }
}

TEST_CASE("HostHeap - JIT code allocates through it") {
    RecordingHeap heap;
    HeapScope scope(&heap);
    auto mod = build_alloc_module();

    codegen::JitExecutionEngine jit(Target::host());
    REQUIRE(jit.compile_and_load(*mod));
    auto fn = jit.get_function_ptr<int64_t (*)()>("alloc_and_collect");
    REQUIRE(fn != nullptr);

    const auto obj = static_cast<uintptr_t>(fn());
    CHECK(heap.owns(obj));
    REQUIRE_EQ(heap.allocations.size(), 1u);
    CHECK_EQ(heap.allocations[0].type_tag, 7u);
    // brass_gc_collect from generated code is a safepoint, as it always was.
    CHECK_EQ(heap.safepoints, 1);
}
