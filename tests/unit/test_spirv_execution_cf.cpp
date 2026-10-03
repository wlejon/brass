// Control-flow shapes through the structurizer, executed on a Vulkan device
// and checked against the same computation on the host: if / if-else with a
// phi, loops with carried scalars and 256-bit vectors, nested and unrolled
// loops, early return and break inside a loop, nested ifs sharing a join, both
// arms returning, a do-while latch, a loop left by return, a loop at the entry
// block, a br_if with one target, and divergent trip counts per thread.
// Each kernel runs one thread per input. [SKIP] without a Vulkan device.

#include "spirv_exec_support.hpp"

#include <numeric>

using namespace brass;
using namespace spvexec;
using codegen::KernelBuilder;

namespace {

// append_block without moving the insertion point.
BasicBlock* blk(Builder& b, const char* name) {
    BasicBlock* cur = b.current_block();
    BasicBlock* bb = b.append_block(name);
    if (cur) b.position_at_end(cur);
    return bb;
}

// k(in, out, n): `body` builds from the entry block with x = in[tid] (i32)
// and returns the i32 written to out[tid]; threads >= n do nothing. `body`
// may create blocks; it must leave the builder in a block without a
// terminator. Runs over `in`, one thread each, block 64.
std::vector<int32_t> run_per_thread(const std::vector<int32_t>& in,
                                    const std::function<Value*(KernelBuilder&, Value* x, Value* in_ptr)>& body) {
    spvtest::Kernel k({Type::ptr(), Type::ptr(), Type::i32()});
    KernelBuilder kb(k.b);
    Builder& b = k.b;
    Value* tid = kb.global_tid_x();
    BasicBlock* work = blk(b, "work");
    BasicBlock* done = blk(b, "done");
    b.build_br_if(b.build_slt(tid, k.p(2)), work, done);
    b.position_at_end(work);
    Value* x = kb.load_i32_indexed(k.p(0), tid);
    Value* r = body(kb, x, k.p(0));
    kb.store_i32_indexed(k.p(1), tid, r);
    b.build_br(done);
    b.position_at_end(done);
    b.build_ret_void();
    VulkanModule m = load(*k.fn);
    VulkanBuffer din = upload(in);
    VulkanBuffer dout = zeros<int32_t>(in.size());
    launch(m, static_cast<uint32_t>((in.size() + 63) / 64), 64, {din, dout, static_cast<int32_t>(in.size())});
    return download<int32_t>(dout, in.size());
}

std::vector<int32_t> iota_inputs(int32_t n, int32_t start = -20) {
    std::vector<int32_t> v(static_cast<size_t>(n));
    std::iota(v.begin(), v.end(), start);
    return v;
}

void expect(const char* what, const std::vector<int32_t>& in, const std::vector<int32_t>& got,
            const std::function<int32_t(int32_t)>& host) {
    for (size_t i = 0; i < in.size(); ++i) {
        int32_t want = host(in[i]);
        if (got[i] != want) std::cerr << what << ": x=" << in[i] << " device " << got[i] << " host " << want << "\n";
        CHECK_EQ(got[i], want);
    }
}

} // namespace

TEST_CASE("SPIR-V exec cf - if-then and if-else joined by a block parameter") {
    if (!vk_ready()) return;
    std::vector<int32_t> in = iota_inputs(70);
    // r = x; if (x > 3) r = x * 5;  then  x & 1 ? r + 1 : r * 2 through a phi
    std::vector<int32_t> got = run_per_thread(in, [](KernelBuilder& kb, Value* x, Value*) {
        Builder& b = kb.builder();
        BasicBlock* t = blk(b, "then");
        BasicBlock* e = blk(b, "else");
        BasicBlock* j = blk(b, "join");
        Value* jr = b.add_block_param(j, Type::i32());
        BasicBlock* big = blk(b, "big");
        BasicBlock* after = blk(b, "after");
        Value* r = b.add_block_param(after, Type::i32());
        b.build_br_if(b.build_sgt(x, kb.const_i32(3)), big, {}, after, {x});
        b.position_at_end(big);
        b.build_br(after, {b.build_mul(x, kb.const_i32(5))});
        b.position_at_end(after);
        b.build_br_if(b.build_ne(b.build_and(x, kb.const_i32(1)), kb.const_i32(0)), t, e);
        b.position_at_end(t);
        b.build_br(j, {b.build_add(r, kb.const_i32(1))});
        b.position_at_end(e);
        b.build_br(j, {b.build_mul(r, kb.const_i32(2))});
        b.position_at_end(j);
        return jr;
    });
    expect("if-else", in, got, [](int32_t x) {
        int32_t r = x > 3 ? x * 5 : x;
        return (x & 1) ? r + 1 : r * 2;
    });
}

TEST_CASE("SPIR-V exec cf - loops with divergent trip counts, nested and unrolled") {
    if (!vk_ready()) return;
    std::vector<int32_t> in = iota_inputs(100, 0);
    // sum_{i < x % 13} sum_{j < x % 7 (inner unrolled 4)} (i * j + 1)
    std::vector<int32_t> got = run_per_thread(in, [](KernelBuilder& kb, Value* x, Value*) {
        Builder& b = kb.builder();
        Value* n = b.build_smod(x, kb.const_i32(13));
        Value* m = b.build_smod(x, kb.const_i32(7));
        return kb.for_range_reduce(kb.const_i32(0), n, kb.const_i32(1), kb.const_i32(0), [&](Value* i, Value* acc) {
            return kb.for_range_reduce(kb.const_i32(0), m, kb.const_i32(1), acc, [&](Value* j, Value* a) {
                return b.build_add(a, b.build_add(b.build_mul(i, j), kb.const_i32(1)));
            }, 4);
        });
    });
    expect("nested loops", in, got, [](int32_t x) {
        int32_t s = 0;
        for (int32_t i = 0; i < x % 13; ++i)
            for (int32_t j = 0; j < x % 7; ++j) s += i * j + 1;
        return s;
    });
}

TEST_CASE("SPIR-V exec cf - early return and break inside a loop") {
    if (!vk_ready()) return;
    // for (i = 0; i < 20; ++i) { v = in[(x + i) % n]; if (v < -15) return out stays 0
    // (the thread exits before storing); if (v == 7) break; s += v } -> s * 10 + i
    std::vector<int32_t> in = iota_inputs(40);
    const int32_t n = static_cast<int32_t>(in.size());
    spvtest::Kernel k({Type::ptr(), Type::ptr(), Type::i32()});
    KernelBuilder kb(k.b);
    Builder& b = k.b;
    BasicBlock* head = blk(b, "head");
    BasicBlock* body = blk(b, "body");
    BasicBlock* early = blk(b, "early_ret");
    BasicBlock* check = blk(b, "check");
    BasicBlock* latch = blk(b, "latch");
    BasicBlock* exit = blk(b, "exit");
    Value* i = b.add_block_param(head, Type::i32());
    Value* s = b.add_block_param(head, Type::i32());
    Value* ei = b.add_block_param(exit, Type::i32());
    Value* es = b.add_block_param(exit, Type::i32());
    Value* tid = kb.global_tid_x();
    BasicBlock* start = blk(b, "start");
    BasicBlock* skip = blk(b, "skip");
    b.build_br_if(b.build_slt(tid, k.p(2)), start, skip);
    b.position_at_end(skip);
    b.build_ret_void();
    b.position_at_end(start);
    Value* x = kb.load_i32_indexed(k.p(0), tid);
    b.build_br(head, {kb.const_i32(0), kb.const_i32(0)});
    b.position_at_end(head);
    b.build_br_if(b.build_slt(i, kb.const_i32(20)), body, {}, exit, {i, s});
    b.position_at_end(body);
    Value* idx = b.build_smod(b.build_add(b.build_add(x, kb.const_i32(20)), i), k.p(2));
    Value* v = kb.load_i32_indexed(k.p(0), idx);
    b.build_br_if(b.build_slt(v, kb.const_i32(-15)), early, check);
    b.position_at_end(early);
    b.build_ret_void();
    b.position_at_end(check);
    b.build_br_if(b.build_eq(v, kb.const_i32(7)), exit, {i, s}, latch, {});
    b.position_at_end(latch);
    b.build_br(head, {b.build_add(i, kb.const_i32(1)), b.build_add(s, v)});
    b.position_at_end(exit);
    kb.store_i32_indexed(k.p(1), tid, b.build_add(b.build_mul(es, kb.const_i32(10)), ei));
    b.build_ret_void();

    VulkanModule m = load(*k.fn);
    VulkanBuffer din = upload(in);
    VulkanBuffer dout = VulkanBuffer::alloc(in.size() * 4);
    REQUIRE(dout.fill(0xFFFFFFFFu));
    launch(m, 1, 64, {din, dout, n});
    std::vector<int32_t> got = download<int32_t>(dout, in.size());
    for (int32_t t = 0; t < n; ++t) {
        int32_t want = -1, si = 0, ii = 0;
        bool returned = false;
        for (ii = 0; ii < 20; ++ii) {
            int32_t vv = in[static_cast<size_t>((in[static_cast<size_t>(t)] + 20 + ii) % n)];
            if (vv < -15) { returned = true; break; }
            if (vv == 7) break;
            si += vv;
        }
        if (!returned) want = si * 10 + ii;
        if (got[static_cast<size_t>(t)] != want) std::cerr << "thread " << t << " device " << got[static_cast<size_t>(t)] << " host " << want << "\n";
        CHECK_EQ(got[static_cast<size_t>(t)], want);
    }
}

TEST_CASE("SPIR-V exec cf - nested ifs sharing a join, both arms returning") {
    if (!vk_ready()) return;
    std::vector<int32_t> in = iota_inputs(64);
    // join(x > 0 ? (x > 10 ? 3 : 2) : 1) with the inner arms jumping to the outer join
    std::vector<int32_t> got = run_per_thread(in, [](KernelBuilder& kb, Value* x, Value*) {
        Builder& b = kb.builder();
        BasicBlock* inner = blk(b, "inner");
        BasicBlock* deep = blk(b, "deep");
        BasicBlock* join = blk(b, "join");
        Value* r = b.add_block_param(join, Type::i32());
        b.build_br_if(b.build_sgt(x, kb.const_i32(0)), inner, {}, join, {kb.const_i32(1)});
        b.position_at_end(inner);
        b.build_br_if(b.build_sgt(x, kb.const_i32(10)), deep, {}, join, {kb.const_i32(2)});
        b.position_at_end(deep);
        b.build_br(join, {kb.const_i32(3)});
        b.position_at_end(join);
        return b.build_add(b.build_mul(r, kb.const_i32(100)), x);
    });
    expect("shared join", in, got, [](int32_t x) { return (x > 0 ? (x > 10 ? 3 : 2) : 1) * 100 + x; });

    // Both arms store and return.
    spvtest::Kernel k({Type::ptr(), Type::ptr()});
    KernelBuilder kb(k.b);
    Builder& b = k.b;
    Value* tid = kb.global_tid_x();
    Value* x = kb.load_i32_indexed(k.p(0), tid);
    BasicBlock* t = blk(b, "t");
    BasicBlock* e = blk(b, "e");
    b.build_br_if(b.build_slt(x, kb.const_i32(0)), t, e);
    b.position_at_end(t);
    kb.store_i32_indexed(k.p(1), tid, b.build_neg(x));
    b.build_ret_void();
    b.position_at_end(e);
    kb.store_i32_indexed(k.p(1), tid, b.build_add(x, kb.const_i32(1000)));
    b.build_ret_void();
    VulkanModule m = load(*k.fn);
    VulkanBuffer din = upload(in);
    VulkanBuffer dout = zeros<int32_t>(in.size());
    launch(m, 1, 64, {din, dout});
    expect("both arms return", in, download<int32_t>(dout, in.size()), [](int32_t x) { return x < 0 ? -x : x + 1000; });
}

TEST_CASE("SPIR-V exec cf - do-while latch, loop left by return, loop at the entry block") {
    if (!vk_ready()) return;
    std::vector<int32_t> in = iota_inputs(64, 1);
    // do { s += i * x; i++ } while (i < x % 9)  (runs at least once)
    std::vector<int32_t> got = run_per_thread(in, [](KernelBuilder& kb, Value* x, Value*) {
        Builder& b = kb.builder();
        BasicBlock* body = blk(b, "body");
        BasicBlock* exit = blk(b, "exit");
        Value* i = b.add_block_param(body, Type::i32());
        Value* s = b.add_block_param(body, Type::i32());
        Value* r = b.add_block_param(exit, Type::i32());
        b.build_br(body, {kb.const_i32(0), kb.const_i32(0)});
        b.position_at_end(body);
        Value* s2 = b.build_add(s, b.build_mul(i, x));
        Value* i2 = b.build_add(i, kb.const_i32(1));
        b.build_br_if(b.build_slt(i2, b.build_smod(x, kb.const_i32(9))), body, {i2, s2}, exit, {s2});
        b.position_at_end(exit);
        return r;
    });
    expect("do-while", in, got, [](int32_t x) {
        int32_t s = 0, i = 0;
        do { s += i * x; ++i; } while (i < x % 9);
        return s;
    });

    // A loop left only through a block that stores and returns: count up
    // from x until a multiple of 8.
    {
        spvtest::Kernel k({Type::ptr(), Type::ptr()});
        KernelBuilder kb(k.b);
        Builder& b = k.b;
        Value* tid = kb.global_tid_x();
        Value* x = kb.load_i32_indexed(k.p(0), tid);
        BasicBlock* body = blk(b, "body");
        BasicBlock* done = blk(b, "done");
        Value* v = b.add_block_param(body, Type::i32());
        b.build_br(body, {x});
        b.position_at_end(body);
        Value* nv = b.build_add(v, kb.const_i32(1));
        b.build_br_if(b.build_eq(b.build_and(nv, kb.const_i32(7)), kb.const_i32(0)), done, {}, body, {nv});
        b.position_at_end(done);
        kb.store_i32_indexed(k.p(1), tid, nv);
        b.build_ret_void();
        VulkanModule m = load(*k.fn);
        VulkanBuffer din = upload(in);
        VulkanBuffer dout = zeros<int32_t>(in.size());
        launch(m, 1, 64, {din, dout});
        expect("loop left by return", in, download<int32_t>(dout, in.size()), [](int32_t x) {
            int32_t v = x;
            do { ++v; } while (v & 7);
            return v;
        });
    }

    // The entry block is the loop head: k(out, n) recurses on its own
    // parameters (out + 4, n - 1) and a br_if sends both edges to one block.
    {
        spvtest::Kernel k({Type::ptr(), Type::i32()});
        Builder& b = k.b;
        BasicBlock* body = blk(b, "body");
        BasicBlock* same = blk(b, "same");
        BasicBlock* exit = blk(b, "exit");
        Value* s = b.add_block_param(same, Type::i32());
        b.position_at_end(k.entry);
        b.build_br_if(b.build_sgt(k.p(1), b.build_iconst_i32(0)), body, exit);
        b.position_at_end(body);
        b.build_br_if(b.build_slt(k.p(1), b.build_iconst_i32(5)), same, {b.build_iconst_i32(10)}, same,
                      {b.build_iconst_i32(20)});
        b.position_at_end(same);
        b.build_store(Type::i32(), k.p(0), 0, b.build_add(s, k.p(1)));
        b.build_br(k.entry, {b.build_add(k.p(0), b.build_iconst_i64(4)), b.build_sub(k.p(1), b.build_iconst_i32(1))});
        b.position_at_end(exit);
        b.build_ret_void();
        VulkanModule m = load(*k.fn);
        VulkanBuffer out = zeros<int32_t>(9);
        launch(m, 1, 1, {out, int32_t{8}});
        std::vector<int32_t> got = download<int32_t>(out, 9);
        for (int32_t j = 0; j < 8; ++j) {
            int32_t n = 8 - j;
            CHECK_EQ(got[static_cast<size_t>(j)], (n < 5 ? 10 : 20) + n);
        }
        CHECK_EQ(got[8], 0);
    }
}

TEST_CASE("SPIR-V exec cf - loop-carried 256-bit vectors and a grid-stride loop") {
    if (!vk_ready()) return;
    // acc_f32x8 += in[i] (8 floats), acc_i64x4 += in[i] reinterpreted; n rows of 32 bytes.
    const int32_t n = 37;
    std::vector<float> rows(static_cast<size_t>(n) * 8);
    for (size_t i = 0; i < rows.size(); ++i) rows[i] = static_cast<float>(i % 17) * 0.5f - 3.0f;
    spvtest::Kernel k({Type::ptr(), Type::ptr(), Type::i32()});
    KernelBuilder kb(k.b);
    Builder& b = k.b;
    auto accs = kb.for_range_reduce_n(kb.const_i32(0), k.p(2), kb.const_i32(1),
        {b.build_vzero(Type::f32x8()), b.build_vzero(Type::i64x4())}, [&](Value* i, const std::vector<Value*>& a) {
            Value* p = b.build_add(k.p(0), b.build_mul(b.build_sext_i64(i), b.build_iconst_i64(32)));
            return std::vector<Value*>{b.build_vadd(a[0], b.build_vload(Type::f32x8(), p, 0)),
                                       b.build_vadd(a[1], b.build_vload(Type::i64x4(), p, 0))};
        });
    b.build_vstore(Type::f32x8(), k.p(1), 0, accs[0]);
    b.build_vstore(Type::i64x4(), k.p(1), 32, accs[1]);
    b.build_ret_void();
    VulkanModule m = load(*k.fn);
    VulkanBuffer din = upload(rows);
    VulkanBuffer dout = zeros<uint64_t>(8);
    launch(m, 1, 1, {din, dout, n});
    std::vector<float> gf = download<float>(dout, 8);
    std::vector<uint64_t> gi = download<uint64_t>(dout, 8);
    for (int lane = 0; lane < 8; ++lane) {
        float s = 0;
        for (int32_t r = 0; r < n; ++r) s += rows[static_cast<size_t>(r) * 8 + static_cast<size_t>(lane)];
        CHECK_EQ(gf[static_cast<size_t>(lane)], s);
    }
    for (int lane = 0; lane < 4; ++lane) {
        uint64_t s = 0;
        for (int32_t r = 0; r < n; ++r) {
            uint64_t v;
            std::memcpy(&v, &rows[static_cast<size_t>(r) * 8 + static_cast<size_t>(lane) * 2], 8);
            s += v;
        }
        CHECK_EQ(gi[4 + static_cast<size_t>(lane)], s);
    }

    // grid-stride: out[i] = in[i] * 2 over n elements with fewer threads than elements
    const uint32_t count = 10007;
    std::vector<int32_t> in(count);
    std::iota(in.begin(), in.end(), -5000);
    spvtest::Kernel g({Type::ptr(), Type::ptr(), Type::i32()});
    KernelBuilder gk(g.b);
    gk.for_range(gk.global_tid_x(), g.p(2), g.b.build_mul(gk.nctaid_x(), gk.ntid_x()), [&](Value* i) {
        gk.store_i32_indexed(g.p(1), i, g.b.build_mul(gk.load_i32_indexed(g.p(0), i), gk.const_i32(2)));
    });
    g.b.build_ret_void();
    VulkanModule gm = load(*g.fn);
    VulkanBuffer gin = upload(in);
    VulkanBuffer gout = zeros<int32_t>(count);
    launch(gm, 7, 128, {gin, gout, static_cast<int32_t>(count)});
    std::vector<int32_t> got = download<int32_t>(gout, count);
    for (uint32_t i = 0; i < count; ++i) CHECK_EQ(got[i], in[i] * 2);
}
