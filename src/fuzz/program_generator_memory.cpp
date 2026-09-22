// Memory statements of the structured program generator: buffers from
// alloca and brass_gc_alloc, constant-offset, indexed, guarded and derived
// pointer accesses, GC objects holding gcrefs behind a write barrier, and
// groups of adjacent loops over the same buffers (the fusion, distribution
// and array-contraction shapes).

#include "program_generator_impl.hpp"

namespace brass::fuzz::detail {

namespace {

Value* gc_alloc(Builder& b, int64_t bytes, int64_t pointer_mask, int32_t tag) {
    return b.build_call("brass_gc_alloc", Type::gcref(),
                        {b.build_iconst_i64(bytes), b.build_iconst_i64(pointer_mask), b.build_iconst_i32(tag)});
}

} // namespace

Value* GenState::in_bounds_index(const Buffer& buf) {
    Value* raw = to_i64(pick(rng_.coin_flip(0.7) ? Type::i64() : Type::i32()));
    return b_.build_and(raw, konst(Type::i64(), static_cast<int64_t>(buf.len) - 1));
}

Value* GenState::elem_value(Type elem) {
    return int_expr(elem, 1);
}

void GenState::setup_buffers() {
    const uint32_t nb = 1 + rng_.next_u32() % 3;
    for (uint32_t k = 0; k < nb; ++k) {
        Buffer buf;
        buf.elem = rng_.coin_flip(0.7) ? Type::i64() : Type::i32();
        buf.scale = buf.elem == Type::i64() ? 8 : 4;
        buf.len = 1u << rng_.range_u64(2, 6);
        buf.gc = opts_.enable_gc && rng_.coin_flip(0.6);
        const int64_t bytes = static_cast<int64_t>(buf.len) * buf.scale;
        buf.base = buf.gc ? gc_alloc(b_, bytes, 0, static_cast<int32_t>(1 + k))
                          : b_.build_alloca(static_cast<uint32_t>(bytes), 8);
        buffers_.push_back(buf);
    }

    // Holders: slot 0 is a pointer (mask bit 0), so the GC relocates it.
    for (size_t k = 0; k < buffers_.size(); ++k) {
        const Buffer& buf = buffers_[k];
        if (!buf.gc || !rng_.coin_flip(0.5)) continue;
        Holder h;
        h.obj = gc_alloc(b_, 16, 1, static_cast<int32_t>(50 + k));
        h.elem = buf.elem;
        h.scale = buf.scale;
        h.len = buf.len;
        b_.build_store(Type::gcref(), h.obj, 0, buf.base);
        b_.build_write_barrier(h.obj, buf.base);
        holders_.push_back(h);
    }

    // Write every element before any statement can read one.
    for (const Buffer& buf : buffers_) {
        Value* seed_v = operand(Type::i64());
        const int64_t mul = rng_.range_i64(1, 1 << 20) | 1;
        LoopCtx ctx;
        emit_loop(konst(Type::i64(), buf.len), buf.len, ctx, [&](Value* iv, LoopCtx&) {
            Value* v = b_.build_xor(b_.build_mul(iv, konst(Type::i64(), mul)), seed_v);
            if (buf.elem == Type::i32()) v = b_.build_trunc_i32(v);
            b_.build_store_indexed(buf.elem, buf.base, iv, buf.scale, v);
        });
    }
}

void GenState::stmt_memory(int depth) {
    const Buffer buf = buffers_[rng_.pick_index(buffers_.size())];
    const int32_t scale = buf.scale;
    uint32_t kind = rng_.next_u32() % 10;
    if ((kind == 7 || kind == 8) && holders_.empty()) kind = 3;
    if (kind == 9 && trip_budget() < buf.len) kind = 2;

    switch (kind) {
        case 0: {
            const int32_t k = static_cast<int32_t>(rng_.pick_index(buf.len));
            Value* v = elem_value(buf.elem);
            b_.build_store(buf.elem, buf.base, k * scale, v);
            break;
        }
        case 1: {
            const int32_t k = static_cast<int32_t>(rng_.pick_index(buf.len));
            Value* v = b_.build_load(buf.elem, buf.base, k * scale);
            add_value(v);
            mix(v);
            break;
        }
        case 2: {
            Value* idx = in_bounds_index(buf);
            Value* v = elem_value(buf.elem);
            b_.build_store_indexed(buf.elem, buf.base, idx, buf.scale, v);
            break;
        }
        case 3: {
            Value* idx = in_bounds_index(buf);
            Value* v = b_.build_load_indexed(buf.elem, buf.base, idx, buf.scale);
            add_value(v);
            mix(v);
            break;
        }
        case 4: {
            // The bounds-check shape: an index that may be out of range,
            // used only under a compare against the length.
            Value* raw = to_i64(pick(Type::i64()));
            Value* idx = b_.build_and(raw, konst(Type::i64(), 2 * static_cast<int64_t>(buf.len) - 1));
            Value* len = konst(Type::i64(), buf.len);
            Value* ok = rng_.coin_flip(0.5) ? b_.build_slt(idx, len) : b_.build_ult(idx, len);
            BasicBlock* t_bb = new_block("bc_in");
            BasicBlock* f_bb = new_block("bc_out");
            BasicBlock* m_bb = new_block("bc_merge");
            b_.build_br_if(ok, t_bb, {}, f_bb, {});
            Value* acc0 = acc_;
            const Mark m = mark();
            b_.position_at_end(t_bb);
            if (rng_.coin_flip(0.5)) {
                b_.build_store_indexed(buf.elem, buf.base, idx, buf.scale, elem_value(buf.elem));
            } else {
                mix(b_.build_load_indexed(buf.elem, buf.base, idx, buf.scale));
            }
            if (rng_.coin_flip(0.3)) stmts(depth + 1, 1);
            b_.build_br(m_bb, {acc_});
            restore(m);
            b_.position_at_end(f_bb);
            acc_ = acc0;
            mix(idx);
            b_.build_br(m_bb, {acc_});
            b_.position_at_end(m_bb);
            acc_ = b_.add_block_param(m_bb, Type::i64());
            break;
        }
        case 5: {
            // A derived pointer, used right away in its own block.
            Value* off = nullptr;
            if (rng_.coin_flip(0.4)) {
                off = konst(Type::i64(), static_cast<int64_t>(rng_.pick_index(buf.len)) * scale);
            } else {
                off = b_.build_mul(in_bounds_index(buf), konst(Type::i64(), scale));
            }
            const bool store = rng_.coin_flip(0.5);
            Value* v = store ? elem_value(buf.elem) : nullptr;
            Value* p = b_.build_add(buf.base, off);
            if (store) {
                b_.build_store(buf.elem, p, 0, v);
            } else {
                Value* x = b_.build_load(buf.elem, p, 0);
                add_value(x);
                mix(x);
            }
            break;
        }
        case 6: {
            // Two derived pointers that may or may not be the same address,
            // with stores between the loads.
            Value* idx1 = in_bounds_index(buf);
            Value* idx2 = nullptr;
            switch (rng_.next_u32() % 3) {
                case 0: idx2 = idx1; break;
                case 1: idx2 = in_bounds_index(buf); break;
                default:
                    idx2 = b_.build_and(b_.build_add(idx1, konst(Type::i64(), 1)),
                                        konst(Type::i64(), static_cast<int64_t>(buf.len) - 1));
                    break;
            }
            Value* off1 = b_.build_mul(idx1, konst(Type::i64(), scale));
            Value* off2 = b_.build_mul(idx2, konst(Type::i64(), scale));
            Value* v1 = elem_value(buf.elem);
            Value* v2 = elem_value(buf.elem);
            Value* p1 = b_.build_add(buf.base, off1);
            Value* p2 = b_.build_add(buf.base, off2);
            b_.build_store(buf.elem, p1, 0, v1);
            Value* x = b_.build_load(buf.elem, p2, 0);
            b_.build_store(buf.elem, p2, 0, v2);
            Value* y = b_.build_load(buf.elem, p1, 0);
            mix(x);
            mix(y);
            break;
        }
        case 7: {
            const Holder h = holders_[rng_.pick_index(holders_.size())];
            Buffer view;
            view.len = h.len;
            Value* idx = in_bounds_index(view);
            Value* g = b_.build_load(Type::gcref(), h.obj, 0);
            Value* x = b_.build_load_indexed(h.elem, g, idx, h.scale);
            add_value(x);
            mix(x);
            break;
        }
        case 8: {
            const Holder h = holders_[rng_.pick_index(holders_.size())];
            std::vector<const Buffer*> same;
            for (const Buffer& other : buffers_) {
                if (other.gc && other.len == h.len && other.elem == h.elem) same.push_back(&other);
            }
            Value* target = same[rng_.pick_index(same.size())]->base;
            b_.build_store(Type::gcref(), h.obj, 0, target);
            b_.build_write_barrier(h.obj, target);
            break;
        }
        default: {
            Value* n = rng_.coin_flip(0.5)
                ? konst(Type::i64(), buf.len)
                : b_.build_and(to_i64(pick(Type::i64())), konst(Type::i64(), static_cast<int64_t>(buf.len) - 1));
            Value* x = operand(buf.elem);
            const uint32_t op = rng_.next_u32() % 3;
            const bool observe = rng_.coin_flip(0.5);
            LoopCtx ctx;
            emit_loop(n, buf.len, ctx, [&](Value* iv, LoopCtx&) {
                Value* e = b_.build_load_indexed(buf.elem, buf.base, iv, buf.scale);
                Value* e2 = op == 0 ? b_.build_add(e, x) : op == 1 ? b_.build_xor(e, x) : b_.build_mul(e, x);
                b_.build_store_indexed(buf.elem, buf.base, iv, buf.scale, e2);
                if (observe) mix(e);
            });
            break;
        }
    }
}

void GenState::stmt_loop_group() {
    Buffer a = buffers_[rng_.pick_index(buffers_.size())];
    // A scratch buffer written by one loop and read by the next, dead
    // afterwards: what array contraction looks for. Only outside loops, so
    // it is allocated at most once per run of the enclosing region.
    const bool scratch = loops_.empty() && rng_.coin_flip(0.3);
    if (scratch) {
        a.elem = Type::i64();
        a.scale = 8;
        a.len = 1u << rng_.range_u64(2, 5);
        a.gc = opts_.enable_gc && rng_.coin_flip(0.5);
        const int64_t bytes = static_cast<int64_t>(a.len) * 8;
        a.base = a.gc ? gc_alloc(b_, bytes, 0, 90) : b_.build_alloca(static_cast<uint32_t>(bytes), 8);
    }
    if (trip_budget() < a.len) {
        stmt_arith();
        return;
    }
    const Buffer* partner = nullptr;
    for (const Buffer& other : buffers_) {
        if (other.base != a.base && other.len == a.len && other.elem == a.elem) partner = &other;
    }
    const Type et = a.elem;
    // Scratch buffers are read only where the first loop wrote them.
    Value* n = (scratch || rng_.coin_flip(0.5))
        ? konst(Type::i64(), a.len)
        : b_.build_and(to_i64(pick(Type::i64())), konst(Type::i64(), static_cast<int64_t>(a.len) - 1));
    Value* x = operand(et);
    Value* kmul = konst(et, rng_.range_i64(1, 999) | 1);
    auto elem_iv = [&](Value* iv) { return et == Type::i32() ? b_.build_trunc_i32(iv) : iv; };

    LoopCtx producer;
    emit_loop(n, a.len, producer, [&](Value* iv, LoopCtx&) {
        Value* v = b_.build_add(b_.build_mul(elem_iv(iv), kmul), x);
        b_.build_store_indexed(et, a.base, iv, a.scale, v);
    });

    uint32_t variant = rng_.next_u32() % 4;
    if (variant == 1 && !partner) variant = 0;
    if (variant == 2 && scratch) variant = 0;
    LoopCtx consumer;
    emit_loop(n, a.len, consumer, [&](Value* iv, LoopCtx&) {
        switch (variant) {
            case 0:
                mix(b_.build_load_indexed(et, a.base, iv, a.scale));
                break;
            case 1: {
                Value* e = b_.build_load_indexed(et, a.base, iv, a.scale);
                Value* v = b_.build_add(b_.build_mul(e, kmul), elem_iv(iv));
                b_.build_store_indexed(et, partner->base, iv, partner->scale, v);
                break;
            }
            case 2: {
                Value* j = b_.build_and(b_.build_add(iv, konst(Type::i64(), 1)),
                                        konst(Type::i64(), static_cast<int64_t>(a.len) - 1));
                mix(b_.build_load_indexed(et, a.base, j, a.scale));
                break;
            }
            default: {
                Value* e = b_.build_load_indexed(et, a.base, iv, a.scale);
                b_.build_store_indexed(et, a.base, iv, a.scale, b_.build_add(e, x));
                break;
            }
        }
    });

    if (rng_.coin_flip(0.4)) {
        LoopCtx reader;
        emit_loop(konst(Type::i64(), a.len), a.len, reader, [&](Value* iv, LoopCtx&) {
            mix(b_.build_load_indexed(et, a.base, iv, a.scale));
        });
    }
}

void GenState::checksum_buffers() {
    for (const Buffer& buf : buffers_) {
        LoopCtx ctx;
        emit_loop(konst(Type::i64(), buf.len), buf.len, ctx, [&](Value* iv, LoopCtx&) {
            mix(b_.build_load_indexed(buf.elem, buf.base, iv, buf.scale));
        });
    }
}

} // namespace brass::fuzz::detail
