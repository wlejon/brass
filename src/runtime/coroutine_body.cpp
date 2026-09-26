// Coroutine bodies (CoroBody): the descriptor a frame points to, its frame
// shape and the collector's trace of it, frame creation, and which tier a
// resume runs in. See coroutine.hpp.
#include <brass/runtime/coroutine.hpp>
#include <brass/runtime/code_installer.hpp>
#include <brass/runtime/multi_tier_pipeline.hpp>
#include <brass/runtime/tiering.hpp>
#include <brass/gc/heap.hpp>
#include <brass/gc/object.hpp>
#include <brass/gc/tracer.hpp>
#include <brass/mir/coro_transform.hpp>
#include <brass/mir/function.hpp>
#include <algorithm>
#include <bit>
#include <map>
#include <mutex>
#include <stdexcept>
#include <string>
#include <utility>

namespace brass::runtime {

// Defined in coroutine.cpp: allocates `total_size` bytes of `layout` on the
// thread's heap (or outside any heap), registers nothing.
uintptr_t allocate_coro_frame_object(size_t total_size, gc::LayoutId layout, uintptr_t caller_fp,
                                     uintptr_t caller_ip);
// Initializes a fresh frame's header and registers it as a root.
void init_coro_frame(BrassCoroFrame* frame, void* fn_ptr, uint32_t slot_count, uint32_t flags);

namespace {

// Descriptors of each (Function, program). Leaked, as the descriptors are:
// a frame of a body may outlive everything but the process.
struct BodyRegistry {
    std::mutex mutex;
    std::map<std::pair<const Function*, const FunctionDispatchTable*>, CoroBody*> bodies;
};
BodyRegistry& body_registry() {
    static auto* r = new BodyRegistry();
    return *r;
}

// The program's handle of a body and its feedback, cached beside the
// descriptor (a handle is never freed while its table lives, and the
// descriptor forgets the table before the table dies).
struct TieredBody : CoroBody {
    std::atomic<FunctionHandle*> handle{nullptr};
    std::atomic<TieringFeedback*> feedback{nullptr};
};

void visit_body_slots(const BrassCoroFrame* frame, const CoroBody& body, size_t payload_bytes, uint32_t first,
                      uint32_t last, gc::Tracer& t) {
    const uint32_t in_object = static_cast<uint32_t>((payload_bytes - CORO_FRAME_HEADER_SIZE) / 8);
    last = std::min({last, frame->slot_count, body.slot_count, in_object});
    auto* slots = const_cast<uint64_t*>(frame->slots);
    for (uint32_t w = first / 64; w < body.ref_bits.size() && w * 64 < last; ++w) {
        uint64_t bits = body.ref_bits[w];
        while (bits) {
            const uint32_t i = w * 64 + static_cast<uint32_t>(std::countr_zero(bits));
            bits &= bits - 1;
            if (i < first) continue;
            if (i >= last) break;
            t.visit(&slots[i]);
        }
    }
}

// The Custom layout of every frame with a descriptor: the awaiter link and
// the body's reference slots, however many.
void trace_coro_frame(uintptr_t payload, size_t payload_bytes, gc::Tracer& t) {
    auto* frame = reinterpret_cast<BrassCoroFrame*>(payload);
    t.visit(&frame->awaiter);
    if (const CoroBody* body = coro_body(frame)) {
        visit_body_slots(frame, *body, payload_bytes, 0, UINT32_MAX, t);
    }
}

void trace_coro_frame_range(uintptr_t payload, size_t payload_bytes, size_t begin, size_t end, gc::Tracer& t) {
    auto* frame = reinterpret_cast<BrassCoroFrame*>(payload);
    if (begin <= static_cast<size_t>(CORO_OFFSET_AWAITER) && static_cast<size_t>(CORO_OFFSET_AWAITER) < end) {
        t.visit(&frame->awaiter);
    }
    const CoroBody* body = coro_body(frame);
    if (!body || end <= CORO_FRAME_HEADER_SIZE) return;
    const size_t lo = begin > CORO_FRAME_HEADER_SIZE ? begin - CORO_FRAME_HEADER_SIZE : 0;
    const size_t hi = end - CORO_FRAME_HEADER_SIZE;
    visit_body_slots(frame, *body, payload_bytes, static_cast<uint32_t>((lo + 7) / 8),
                     static_cast<uint32_t>(std::min<size_t>((hi + 7) / 8, UINT32_MAX)), t);
}

gc::LayoutId body_frame_layout() {
    static const gc::LayoutId id = [] {
        gc::LayoutDescriptor d;
        d.kind = gc::LayoutKind::Custom;
        d.type_tag = TYPE_TAG_CORO_FRAME;
        d.trace = &trace_coro_frame;
        d.trace_range = &trace_coro_frame_range;
        d.name = "coroutine frame";
        return gc::register_layout(d);
    }();
    return id;
}

TieredBody& make_body(const Function& fn, FunctionDispatchTable* table, uint32_t min_slots, uint64_t extra_mask) {
    if (!is_lowered_coro_body(fn)) {
        throw std::logic_error("coroutine body '" + std::string(fn.name()) +
                               "' has not been lowered by CoroTransformPass");
    }
    const CoroFrameLayout layout = compute_coro_frame_layout(fn);
    auto* body = new TieredBody();
    body->name = std::string(fn.name());
    body->slot_count = std::max({layout.slot_count, min_slots, 1u});
    body->ref_bits = layout.ref_bits;
    if (extra_mask) {
        body->slot_count = std::max<uint32_t>(body->slot_count, 64 - std::countl_zero(extra_mask));
        if (body->ref_bits.empty()) body->ref_bits.push_back(0);
        body->ref_bits[0] |= extra_mask;
    }
    body->mir.store(&fn, std::memory_order_release);
    body->table.store(table, std::memory_order_release);
    return *body;
}

} // namespace

const CoroBody& coro_body_of(const Function& fn, FunctionDispatchTable* table) {
    BodyRegistry& r = body_registry();
    {
        std::lock_guard<std::mutex> lock(r.mutex);
        auto it = r.bodies.find({&fn, table});
        if (it != r.bodies.end()) return *it->second;
    }
    TieredBody& body = make_body(fn, table, 0, 0);
    std::lock_guard<std::mutex> lock(r.mutex);
    auto [it, fresh] = r.bodies.emplace(std::make_pair(&fn, table), &body);
    if (!fresh) {
        delete &body;  // another thread made it first; nothing points to this one
        return *it->second;
    }
    return body;
}

void forget_coro_body_function(const Function* fn) noexcept {
    BodyRegistry& r = body_registry();
    std::lock_guard<std::mutex> lock(r.mutex);
    for (auto it = r.bodies.begin(); it != r.bodies.end();) {
        if (it->first.first == fn) {
            it->second->mir.store(nullptr, std::memory_order_release);
            it = r.bodies.erase(it);
        } else {
            ++it;
        }
    }
}

void forget_coro_body_program(const FunctionDispatchTable* table) noexcept {
    if (!table) return;
    BodyRegistry& r = body_registry();
    std::lock_guard<std::mutex> lock(r.mutex);
    for (auto it = r.bodies.begin(); it != r.bodies.end();) {
        if (it->first.second == table) {
            auto* body = static_cast<TieredBody*>(it->second);
            body->table.store(nullptr, std::memory_order_release);
            body->handle.store(nullptr, std::memory_order_release);
            body->feedback.store(nullptr, std::memory_order_release);
            it = r.bodies.erase(it);
        } else {
            ++it;
        }
    }
}

uintptr_t create_coro_frame_at(const CoroBody& body, uint32_t min_slots, uintptr_t caller_fp, uintptr_t caller_ip) {
    const uint32_t slots = std::max(body.slot_count, min_slots);
    const size_t total = CORO_FRAME_HEADER_SIZE + static_cast<size_t>(slots) * sizeof(uint64_t);
    const uintptr_t addr = allocate_coro_frame_object(total, body_frame_layout(), caller_fp, caller_ip);
    if (!addr) return 0;
    init_coro_frame(reinterpret_cast<BrassCoroFrame*>(addr), const_cast<CoroBody*>(&body), slots, CORO_FLAG_BODY);
    return addr;
}

uintptr_t create_coro_frame(const CoroBody& body, uint32_t min_slots) {
    return create_coro_frame_at(body, min_slots, 0, 0);
}

uintptr_t create_mir_coro_frame(const Function& fn, uint32_t slot_count, uint64_t pointer_mask) {
    // Outside any program: one descriptor per call is wasteful, so frames of
    // the same shape share one (its Function is `fn` either way).
    if (pointer_mask == 0 && slot_count <= compute_coro_frame_layout(fn).slot_count) {
        return create_coro_frame(coro_body_of(fn, nullptr));
    }
    TieredBody& body = make_body(fn, nullptr, slot_count, pointer_mask);
    return create_coro_frame(body);
}

void* coro_body_native_entry(const BrassCoroFrame* frame) {
    const CoroBody* cbody = coro_body(frame);
    if (!cbody) return frame->fn_ptr;
    auto* body = const_cast<TieredBody*>(static_cast<const TieredBody*>(cbody));
    FunctionDispatchTable* table = body->table.load(std::memory_order_acquire);
    const Function* fn = body->mir.load(std::memory_order_acquire);
    if (!table || !fn) return nullptr;
    MultiTierPipeline& pipeline = table->pipeline();
    if (!pipeline.is_initialized() || pipeline.config().max_tier < TierLevel::Tier1_Baseline) return nullptr;
    FunctionHandle* handle = body->handle.load(std::memory_order_acquire);
    if (!handle) {
        handle = table->find(body->name);
        if (!handle) handle = table->get_or_create(body->name, fn);
        body->handle.store(handle, std::memory_order_release);
    }
    // Code compiled from another Function of that name does not know this
    // frame's layout: the frame stays in Tier 0.
    if (handle->mir_function() != fn) return nullptr;
    if (void* entry = handle->native_entry()) return entry;
    TieringFeedback* fb = body->feedback.load(std::memory_order_acquire);
    if (!fb) {
        fb = &table->tiering().get_feedback(body->name);
        body->feedback.store(fb, std::memory_order_release);
    }
    // Counts the resume as a call of the body: at the threshold it compiles
    // Tier 1, which then counts its own entries toward Tier 2.
    pipeline.on_invocation(*fb);
    return handle->mir_function() == fn ? handle->native_entry() : nullptr;
}

} // namespace brass::runtime
