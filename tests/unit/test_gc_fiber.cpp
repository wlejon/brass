// gc::Fiber: switching between a thread's stacks, what a switch keeps (the
// callee-saved registers, the floating-point control state, the stack's own
// bounds for growth and exception dispatch), and what a host's conservative
// scan of a switched-out stack sees (for_each_stack_span).

#include "test_framework.hpp"

#include <brass/gc/fiber.hpp>

#include <cfenv>
#include <cmath>
#include <cstring>
#include <stdexcept>
#include <vector>

using namespace brass::gc;

namespace {

struct Counter {
    int steps = 0;
    int limit = 0;
};

void count_entry(void* arg) {
    auto* c = static_cast<Counter*>(arg);
    for (int i = 0; i < c->limit; ++i) {
        ++c->steps;
        Fiber::suspend();
    }
}

// Recurses `depth` frames of `kFrameBytes` each, suspending at the bottom, and
// returns the sum the frames hold, so none is optimized away.
constexpr size_t kFrameBytes = 4096;
volatile uint64_t g_sink = 0;

uint64_t deep(int depth) {
    volatile unsigned char frame[kFrameBytes];
    for (size_t i = 0; i < kFrameBytes; i += 512) frame[i] = static_cast<unsigned char>(depth + i);
    uint64_t below = 0;
    if (depth > 0) {
        below = deep(depth - 1);
    } else {
        Fiber::suspend();
    }
    uint64_t sum = below;
    for (size_t i = 0; i < kFrameBytes; i += 512) sum += frame[i];
    return sum;
}

struct DeepArgs {
    int depth = 0;
    uint64_t result = 0;
};

void deep_entry(void* arg) {
    auto* a = static_cast<DeepArgs*>(arg);
    a->result = deep(a->depth);
}

void throw_entry(void* arg) {
    auto* caught = static_cast<int*>(arg);
    try {
        Fiber::suspend();
        throw std::runtime_error("inside a fiber");
    } catch (const std::runtime_error& e) {
        *caught = std::strcmp(e.what(), "inside a fiber") == 0 ? 1 : -1;
    }
}

struct FloatArgs {
    double values[8] = {};
    double out = 0;
};

// Keeps doubles live across suspensions, where the callee-saved vector
// registers hold them, and checks the rounding mode it set survives.
void float_entry(void* arg) {
    auto* a = static_cast<FloatArgs*>(arg);
    std::fesetround(FE_UPWARD);
    double v0 = a->values[0], v1 = a->values[1], v2 = a->values[2], v3 = a->values[3];
    double v4 = a->values[4], v5 = a->values[5], v6 = a->values[6], v7 = a->values[7];
    for (int i = 0; i < 4; ++i) {
        Fiber::suspend();
        v0 = v0 * 1.5 + v7;
        v1 = v1 * 1.25 + v6;
        v2 = std::sqrt(v2 + v5);
        v3 = v3 - v4 * 0.5;
    }
    a->out = std::fegetround() == FE_UPWARD ? v0 + v1 + v2 + v3 + v4 + v5 + v6 + v7 : -1.0;
    std::fesetround(FE_TONEAREST);
}

struct Pair {
    std::unique_ptr<Fiber> inner;
    int inner_steps = 0;
    int outer_steps = 0;
};

void inner_entry(void* arg) {
    auto* p = static_cast<Pair*>(arg);
    for (int i = 0; i < 3; ++i) {
        ++p->inner_steps;
        Fiber::suspend();
    }
}

void outer_entry(void* arg) {
    auto* p = static_cast<Pair*>(arg);
    p->inner = Fiber::create(&inner_entry, p, 64 * 1024);
    while (!p->inner->finished()) {
        p->inner->resume();
        ++p->outer_steps;
        Fiber::suspend();
    }
}

struct Marker {
    uint64_t word = 0;
};

void marker_entry(void* arg) {
    auto* m = static_cast<Marker*>(arg);
    volatile uint64_t on_stack = m->word;
    Fiber::suspend();
    m->word = on_stack + 1;
}

struct SpanSearch {
    uint64_t word = 0;
    int spans = 0;
    int found = 0;
};

void search_span(void* ctx, const Fiber::StackSpan& span) {
    auto* s = static_cast<SpanSearch*>(ctx);
    ++s->spans;
    if (span.high == 0) return;  // the thread's own stack: the host's anchor bounds it
    for (uintptr_t p = (span.low + 7) & ~uintptr_t{7}; p + 8 <= span.high; p += 8) {
        if (*reinterpret_cast<const uint64_t*>(p) == s->word) ++s->found;
    }
}

} // namespace

TEST_CASE("gc::Fiber - resume and suspend alternate, and a finished fiber reports it") {
    Counter c;
    c.limit = 1000;
    auto f = Fiber::create(&count_entry, &c, 64 * 1024);
    REQUIRE(f != nullptr);
    CHECK(!f->started());
    CHECK(Fiber::current() == nullptr);
    for (int i = 1; i <= 1000; ++i) {
        f->resume();
        CHECK_EQ(c.steps, i);
    }
    CHECK(!f->finished());
    f->resume();
    CHECK(f->finished());
    CHECK(Fiber::current() == nullptr);
}

TEST_CASE("gc::Fiber - a stack grows past what is first committed, and keeps its frames across switches") {
    // 192 frames of 4 KiB: 768 KiB, far past the first 64 KiB.
    DeepArgs a;
    a.depth = 192;
    auto f = Fiber::create(&deep_entry, &a, 1024 * 1024);
    REQUIRE(f != nullptr);
    f->resume();
    CHECK(!f->finished());
    // The thread's own stack runs in between, with its own bounds in force:
    // a deep frame of its own probes pages below where it is.
    volatile unsigned char own[64 * 1024];
    for (size_t i = 0; i < sizeof own; i += 4096) own[i] = static_cast<unsigned char>(i);
    g_sink = own[4096];
    f->resume();
    CHECK(f->finished());
    uint64_t expect = 0;
    for (int d = 0; d <= 192; ++d) {
        for (size_t i = 0; i < kFrameBytes; i += 512) expect += static_cast<unsigned char>(d + i);
    }
    CHECK_EQ(a.result, expect);
}

TEST_CASE("gc::Fiber - an exception thrown and caught inside a fiber stays there") {
    int caught = 0;
    auto f = Fiber::create(&throw_entry, &caught, 64 * 1024);
    f->resume();
    CHECK_EQ(caught, 0);
    f->resume();
    CHECK(f->finished());
    CHECK_EQ(caught, 1);
    // And on the thread's own stack after it.
    bool own = false;
    try {
        throw std::logic_error("own");
    } catch (const std::logic_error&) {
        own = true;
    }
    CHECK(own);
}

TEST_CASE("gc::Fiber - doubles and the rounding mode survive the switches") {
    FloatArgs a;
    for (int i = 0; i < 8; ++i) a.values[i] = 1.0 + i * 0.125;
    double mine = 3.0;
    auto f = Fiber::create(&float_entry, &a, 64 * 1024);
    while (!f->finished()) {
        f->resume();
        mine = mine * 1.0000001;  // the own stack computes with its rounding mode in between
    }
    CHECK(std::fegetround() == FE_TONEAREST);
    CHECK(a.out > 0.0);
    // The same arithmetic on one stack, rounding upward.
    std::fesetround(FE_UPWARD);
    double v0 = a.values[0], v1 = a.values[1], v2 = a.values[2], v3 = a.values[3];
    double v4 = a.values[4], v5 = a.values[5], v6 = a.values[6], v7 = a.values[7];
    for (int i = 0; i < 4; ++i) {
        v0 = v0 * 1.5 + v7;
        v1 = v1 * 1.25 + v6;
        v2 = std::sqrt(v2 + v5);
        v3 = v3 - v4 * 0.5;
    }
    const double expect = v0 + v1 + v2 + v3 + v4 + v5 + v6 + v7;
    std::fesetround(FE_TONEAREST);
    CHECK_EQ(a.out, expect);
    CHECK(mine > 3.0);
}

TEST_CASE("gc::Fiber - a fiber resumes another and each suspends back to its resumer") {
    Pair p;
    auto outer = Fiber::create(&outer_entry, &p, 64 * 1024);
    while (!outer->finished()) outer->resume();
    CHECK_EQ(p.inner_steps, 3);
    CHECK_EQ(p.outer_steps, 4);
    CHECK(p.inner->finished());
}

TEST_CASE("gc::Fiber - a switched-out stack's words are in its span") {
    Marker m;
    m.word = 0x5EC0DE5EC0DE1234ull;
    auto f = Fiber::create(&marker_entry, &m, 64 * 1024);
    f->resume();
    SpanSearch s;
    s.word = 0x5EC0DE5EC0DE1234ull;
    volatile uint64_t here = 0;
    Fiber::for_each_stack_span(reinterpret_cast<uintptr_t>(&here), &search_span, &s);
    CHECK_EQ(s.spans, 2);  // the running own stack, the switched-out fiber
    CHECK(s.found >= 1);
    f->resume();
    CHECK(f->finished());
    CHECK_EQ(m.word, 0x5EC0DE5EC0DE1235ull);
}

TEST_CASE("gc::Fiber - finished fibers' stacks are reused and many switches stay cheap") {
    for (int round = 0; round < 64; ++round) {
        Counter c;
        c.limit = 100;
        auto f = Fiber::create(&count_entry, &c, 256 * 1024);
        REQUIRE(f != nullptr);
        while (!f->finished()) f->resume();
        CHECK_EQ(c.steps, 100);
    }
}
