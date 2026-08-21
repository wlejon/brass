#include "test_framework.hpp"
#include <brass/gc/mini_cheney.hpp>
#include <vector>
#include <cstring>

using namespace brass;

TEST_CASE("MiniCheneyGC - Basic allocation and field read/write") {
    MiniCheneyGC gc(128 * 1024); // 128 KB semispace

    CHECK_EQ(gc.collection_count(), 0ULL);
    CHECK_EQ(gc.total_allocations(), 0ULL);

    // Allocate an object of 24 bytes payload (3 fields of 8 bytes)
    uintptr_t obj = gc.allocate(24, 0, 42);
    REQUIRE(obj != 0);
    CHECK(gc.is_valid_object(obj));
    CHECK_EQ(gc.total_allocations(), 1ULL);

    const GcHeader* hdr = gc.get_header(obj);
    REQUIRE(hdr != nullptr);
    CHECK_EQ(hdr->size, 24U);
    CHECK_EQ(hdr->type_tag, 42U);
    CHECK_EQ(hdr->pointer_mask, 0ULL);

    // Write and read fields
    gc.write_field(obj, 0, 100ULL);
    gc.write_field(obj, 1, 200ULL);
    gc.write_field(obj, 2, 300ULL);

    CHECK_EQ(gc.read_field(obj, 0), 100ULL);
    CHECK_EQ(gc.read_field(obj, 1), 200ULL);
    CHECK_EQ(gc.read_field(obj, 2), 300ULL);
}

TEST_CASE("MiniCheneyGC - Single object evacuation and poison verification") {
    MiniCheneyGC gc(64 * 1024);

    uintptr_t obj = gc.allocate(16, 0, 1);
    gc.write_field(obj, 0, 0x123456789ABCDEF0ULL);
    gc.write_field(obj, 1, 0x0FEDCBA987654321ULL);

    uintptr_t old_addr = obj;
    uintptr_t root = obj;
    std::vector<uintptr_t*> roots = { &root };

    gc.collect(roots);

    CHECK_EQ(gc.collection_count(), 1ULL);
    uintptr_t new_addr = root;
    CHECK_NE(old_addr, new_addr);
    CHECK(gc.is_valid_object(new_addr));
    CHECK(!gc.is_valid_object(old_addr));

    // Verify old space is poisoned
    const uint64_t* old_mem = reinterpret_cast<const uint64_t*>(old_addr);
    CHECK_EQ(old_mem[0], MiniCheneyGC::POISON_PATTERN);
    CHECK_EQ(old_mem[1], MiniCheneyGC::POISON_PATTERN);

    // Verify new object retained data
    CHECK_EQ(gc.read_field(new_addr, 0), 0x123456789ABCDEF0ULL);
    CHECK_EQ(gc.read_field(new_addr, 1), 0x0FEDCBA987654321ULL);
}

TEST_CASE("MiniCheneyGC - Linked list relocation") {
    MiniCheneyGC gc(128 * 1024);

    // Create linked list of 20 nodes
    // Node layout: field 0 = value (i64), field 1 = next pointer (gcref)
    // pointer_mask: bit 1 is 1 -> 0b10 = 2
    uint64_t pointer_mask = (1ULL << 1);

    uintptr_t head = 0;
    std::vector<uintptr_t> old_addresses;

    for (int64_t i = 0; i < 20; ++i) {
        uintptr_t node = gc.allocate(16, pointer_mask, 10);
        gc.write_field(node, 0, static_cast<uint64_t>(i));
        gc.write_field(node, 1, head);
        head = node;
        old_addresses.push_back(node);
    }

    uintptr_t root = head;
    std::vector<uintptr_t*> roots = { &root };

    gc.collect(roots);

    CHECK_EQ(gc.collection_count(), 1ULL);
    uintptr_t new_head = root;
    CHECK_NE(new_head, head);

    // Verify all old addresses are poisoned
    for (uintptr_t old_a : old_addresses) {
        CHECK(!gc.is_valid_object(old_a));
        const uint64_t* old_mem = reinterpret_cast<const uint64_t*>(old_a);
        CHECK_EQ(old_mem[0], MiniCheneyGC::POISON_PATTERN);
    }

    // Traverse new linked list
    uintptr_t curr = new_head;
    int expected_val = 19;
    int count = 0;
    while (curr != 0) {
        CHECK(gc.is_valid_object(curr));
        uint64_t val = gc.read_field(curr, 0);
        CHECK_EQ(val, static_cast<uint64_t>(expected_val));
        expected_val--;
        count++;
        curr = static_cast<uintptr_t>(gc.read_field(curr, 1));
    }
    CHECK_EQ(count, 20);
}

TEST_CASE("MiniCheneyGC - Binary tree relocation") {
    MiniCheneyGC gc(256 * 1024);

    // Node layout: field 0 = value, field 1 = left, field 2 = right
    // pointer_mask = (1 << 1) | (1 << 2) = 0x6
    uint64_t tree_mask = (1ULL << 1) | (1ULL << 2);

    auto make_node = [&](uint64_t val, uintptr_t left, uintptr_t right) -> uintptr_t {
        uintptr_t n = gc.allocate(24, tree_mask, 20);
        gc.write_field(n, 0, val);
        gc.write_field(n, 1, left);
        gc.write_field(n, 2, right);
        return n;
    };

    // Build tree: root=4, left child 2 (children 1, 3), right child 6 (children 5, 7)
    uintptr_t n1 = make_node(1, 0, 0);
    uintptr_t n3 = make_node(3, 0, 0);
    uintptr_t n2 = make_node(2, n1, n3);

    uintptr_t n5 = make_node(5, 0, 0);
    uintptr_t n7 = make_node(7, 0, 0);
    uintptr_t n6 = make_node(6, n5, n7);

    uintptr_t root = make_node(4, n2, n6);
    std::vector<uintptr_t*> roots = { &root };

    gc.collect(roots);

    // In-order traversal verification
    std::vector<uint64_t> in_order;
    std::function<void(uintptr_t)> traverse = [&](uintptr_t node) {
        if (node == 0) return;
        CHECK(gc.is_valid_object(node));
        traverse(static_cast<uintptr_t>(gc.read_field(node, 1)));
        in_order.push_back(gc.read_field(node, 0));
        traverse(static_cast<uintptr_t>(gc.read_field(node, 2)));
    };

    traverse(root);

    std::vector<uint64_t> expected = {1, 2, 3, 4, 5, 6, 7};
    CHECK_EQ(in_order.size(), expected.size());
    for (size_t i = 0; i < expected.size(); ++i) {
        CHECK_EQ(in_order[i], expected[i]);
    }
}

TEST_CASE("MiniCheneyGC - Circular references") {
    MiniCheneyGC gc(128 * 1024);

    // 1. Self-reference: node.next = node
    uint64_t mask = (1ULL << 1);
    uintptr_t self_node = gc.allocate(16, mask, 30);
    gc.write_field(self_node, 0, 42ULL);
    gc.write_field(self_node, 1, self_node);

    uintptr_t root1 = self_node;
    std::vector<uintptr_t*> roots = { &root1 };

    gc.collect(roots);
    CHECK(gc.is_valid_object(root1));
    CHECK_EQ(gc.read_field(root1, 0), 42ULL);
    CHECK_EQ(gc.read_field(root1, 1), root1); // Still points to itself at new address!

    // 2. Two-node cycle: A -> B -> A
    uintptr_t nodeA = gc.allocate(16, mask, 31);
    uintptr_t nodeB = gc.allocate(16, mask, 32);
    gc.write_field(nodeA, 0, 100ULL);
    gc.write_field(nodeA, 1, nodeB);
    gc.write_field(nodeB, 0, 200ULL);
    gc.write_field(nodeB, 1, nodeA);

    uintptr_t root2 = nodeA;
    roots = { &root2 };

    gc.collect(roots);
    CHECK(gc.is_valid_object(root2));
    CHECK_EQ(gc.read_field(root2, 0), 100ULL);
    uintptr_t new_nodeB = static_cast<uintptr_t>(gc.read_field(root2, 1));
    CHECK(gc.is_valid_object(new_nodeB));
    CHECK_EQ(gc.read_field(new_nodeB, 0), 200ULL);
    CHECK_EQ(gc.read_field(new_nodeB, 1), root2); // Points back to new A!
}

TEST_CASE("MiniCheneyGC - Unreachable objects are reclaimed") {
    MiniCheneyGC gc(64 * 1024);

    // Allocate 50 dead objects
    for (int i = 0; i < 50; ++i) {
        gc.allocate(32, 0, 1);
    }
    size_t bytes_before = gc.bytes_allocated_active();
    CHECK(bytes_before > 50 * 32);

    // Allocate 1 live object and register as root
    uintptr_t live_obj = gc.allocate(16, 0, 99);
    gc.write_field(live_obj, 0, 777ULL);

    uintptr_t root = live_obj;
    std::vector<uintptr_t*> roots = { &root };

    gc.collect(roots);

    size_t bytes_after = gc.bytes_allocated_active();
    // Only 1 object remains in active space (sizeof(GcHeader) + 16 = 40 bytes)
    CHECK_EQ(bytes_after, sizeof(GcHeader) + 16);
    CHECK(gc.is_valid_object(root));
    CHECK_EQ(gc.read_field(root, 0), 777ULL);
}

TEST_CASE("MiniCheneyGC - Stress mode allocations") {
    MiniCheneyGC gc(64 * 1024);
    gc.set_stress_mode(true);

    uint64_t mask = (1ULL << 1);
    uintptr_t head = 0;
    gc.register_root(&head);

    // In stress mode, every single allocate() triggers GC collection!
    for (int i = 0; i < 20; ++i) {
        uintptr_t node = gc.allocate(16, mask, 50);
        gc.write_field(node, 0, static_cast<uint64_t>(i));
        gc.write_field(node, 1, head);
        head = node;
    }

    CHECK(gc.collection_count() >= 20ULL);

    // Verify all 20 elements in linked list remain intact
    uintptr_t curr = head;
    int expected = 19;
    int count = 0;
    while (curr != 0) {
        CHECK(gc.is_valid_object(curr));
        uint64_t val = gc.read_field(curr, 0);
        CHECK_EQ(val, static_cast<uint64_t>(expected));
        expected--;
        count++;
        curr = static_cast<uintptr_t>(gc.read_field(curr, 1));
    }
    CHECK_EQ(count, 20);
}
