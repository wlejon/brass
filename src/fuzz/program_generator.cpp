// Structured random MIR programs for the differential fuzzer: nested
// regions of arithmetic, branches, switches, loops, memory and calls, all
// folded into one i64 checksum. Every construct is chosen so the program is
// verifier-clean, deterministic and terminates quickly:
//   - loops count up from 0 by 1 to a trip count bounded by construction,
//     and the product of nested trip counts is capped;
//   - divisors are `x | 1` (never zero);
//   - shift amounts are masked to the operand width;
//   - memory indices are masked by power-of-two lengths or guarded by a
//     compare against the length, and every buffer is fully written before
//     anything reads it;
//   - addresses never reach the checksum, and a derived gcref (`add` on a
//     gcref) is used in its defining block with no GC point in between.

#include "program_generator_impl.hpp"
#include <brass/mir/verifier.hpp>
#include <algorithm>
#include <limits>

namespace brass::fuzz {

namespace detail {

namespace {
constexpr int64_t kMixMul = 0x100000001B3LL;  // FNV-1a prime: odd, spreads bits
}

GenState::GenState(Module& mod, const ProgramGeneratorOptions& opts, uint64_t seed)
    : mod_(mod), opts_(opts), rng_(seed), b_(mod) {}

void GenState::add_value(Value* v) {
    if (!v) return;
    if (v->type() == Type::i64()) i64s_.push_back(v);
    else if (v->type() == Type::i32()) i32s_.push_back(v);
}

BasicBlock* GenState::new_block(const char* name) {
    // Unique names: reproducers go through the printer and parser, which
    // identify blocks by name.
    const std::string unique = std::string(name) + std::to_string(fn_->blocks().size());
    BasicBlock* bb = b_.create_block(unique);
    fn_->append_block(bb);
    return bb;
}

Value* GenState::konst(Type t, int64_t v) {
    if (t == Type::i32()) return b_.build_iconst_i32(static_cast<int32_t>(v));
    return b_.build_iconst_i64(v);
}

Value* GenState::interesting_const(Type t) {
    const bool is64 = t == Type::i64();
    switch (rng_.next_u32() % 4) {
        case 0: return is64 ? konst(t, rng_.boundary_i64()) : konst(t, rng_.boundary_i32());
        case 1: return konst(t, rng_.range_i64(-8, 16));
        case 2: {
            const int64_t bits = is64 ? 63 : 31;
            const int64_t p = static_cast<int64_t>(uint64_t{1} << rng_.range_i64(0, bits - 1));
            return konst(t, p + rng_.range_i64(-1, 1));
        }
        default: return konst(t, is64 ? rng_.next_i64() : static_cast<int64_t>(rng_.next_i32()));
    }
}

Value* GenState::pick(Type t) {
    std::vector<Value*>& pool = t == Type::i64() ? i64s_ : i32s_;
    // Prefer recent values: they carry the most computation.
    const size_t n = pool.size();
    if (rng_.coin_flip(0.5) && n > 4) return pool[n - 1 - rng_.pick_index(4)];
    return pool[rng_.pick_index(n)];
}

Value* GenState::operand(Type t) {
    return rng_.coin_flip(0.75) ? pick(t) : interesting_const(t);
}

Value* GenState::to_i64(Value* v) {
    if (v->type() == Type::i64()) return v;
    return rng_.coin_flip(0.5) ? b_.build_sext_i64(v) : b_.build_zext_i64(v);
}

Value* GenState::safe_divisor(Value* d) {
    Value* nz = b_.build_or(d, konst(d->type(), 1));
    if (opts_.enable_div_overflow) return nz;
    Value* is_m1 = b_.build_eq(nz, konst(d->type(), -1));
    return b_.build_select(is_m1, konst(d->type(), 3), nz);
}

namespace {

Value* build_cmp(Builder& b, uint32_t pred, Value* x, Value* y) {
    switch (pred % 10) {
        case 0: return b.build_eq(x, y);
        case 1: return b.build_ne(x, y);
        case 2: return b.build_slt(x, y);
        case 3: return b.build_sle(x, y);
        case 4: return b.build_sgt(x, y);
        case 5: return b.build_sge(x, y);
        case 6: return b.build_ult(x, y);
        case 7: return b.build_ule(x, y);
        case 8: return b.build_ugt(x, y);
        default: return b.build_uge(x, y);
    }
}

Value* build_overflow(Builder& b, uint32_t which, Value* x, Value* y) {
    switch (which % 6) {
        case 0: return b.build_sadd_overflow(x, y);
        case 1: return b.build_ssub_overflow(x, y);
        case 2: return b.build_smul_overflow(x, y);
        case 3: return b.build_uadd_overflow(x, y);
        case 4: return b.build_usub_overflow(x, y);
        default: return b.build_umul_overflow(x, y);
    }
}

Value* build_div(Builder& b, uint32_t which, Value* x, Value* d) {
    switch (which % 4) {
        case 0: return b.build_sdiv(x, d);
        case 1: return b.build_udiv(x, d);
        case 2: return b.build_smod(x, d);
        default: return b.build_umod(x, d);
    }
}

} // namespace

Value* GenState::cond() {
    const Type t = rng_.coin_flip(0.6) ? Type::i64() : Type::i32();
    switch (rng_.next_u32() % 5) {
        case 0: {
            Value* x = operand(t);
            Value* y = operand(t);
            return build_overflow(b_, rng_.next_u32(), x, y);
        }
        case 1: {
            const int64_t bit = static_cast<int64_t>(uint64_t{1} << rng_.range_i64(0, t == Type::i64() ? 62 : 30));
            Value* masked = b_.build_and(pick(t), konst(t, bit));
            return b_.build_ne(masked, konst(t, 0));
        }
        default: {
            Value* x = int_expr(t, 1);
            Value* y = operand(t);
            return build_cmp(b_, rng_.next_u32(), x, y);
        }
    }
}

Value* GenState::int_expr(Type t, int depth) {
    if (depth <= 0 || rng_.coin_flip(0.2)) return operand(t);
    const bool is64 = t == Type::i64();
    const int64_t width = is64 ? 64 : 32;
    const uint32_t op = rng_.next_u32() % 19;
    switch (op) {
        case 0: case 1: case 2: case 3: case 4: case 5: {
            Value* x = int_expr(t, depth - 1);
            Value* y = int_expr(t, depth - 1);
            switch (op) {
                case 0: return b_.build_add(x, y);
                case 1: return b_.build_sub(x, y);
                case 2: return b_.build_mul(x, y);
                case 3: return b_.build_and(x, y);
                case 4: return b_.build_or(x, y);
                default: return b_.build_xor(x, y);
            }
        }
        case 6: case 7: case 8: {
            Value* x = int_expr(t, depth - 1);
            Value* amt = nullptr;
            if (rng_.coin_flip(0.5)) {
                amt = konst(t, rng_.range_i64(0, width - 1));
            } else {
                Value* raw = int_expr(t, depth - 1);
                amt = b_.build_and(raw, konst(t, width - 1));
            }
            if (op == 6) return b_.build_shl(x, amt);
            if (op == 7) return b_.build_lshr(x, amt);
            return b_.build_ashr(x, amt);
        }
        case 9: return b_.build_not(int_expr(t, depth - 1));
        case 10: return b_.build_neg(int_expr(t, depth - 1));
        case 11: {
            Value* x = int_expr(t, depth - 1);
            switch (rng_.next_u32() % 3) {
                case 0: return b_.build_clz(x);
                case 1: return b_.build_ctz(x);
                default: return b_.build_popcnt(x);
            }
        }
        case 12: {
            Value* c = cond();
            Value* x = int_expr(t, depth - 1);
            Value* y = int_expr(t, depth - 1);
            return b_.build_select(c, x, y);
        }
        case 13: {
            Value* c = cond();
            return is64 ? b_.build_zext_i64(c) : c;
        }
        case 14: {
            if (is64) {
                Value* narrow = int_expr(Type::i32(), depth - 1);
                return rng_.coin_flip(0.5) ? b_.build_sext_i64(narrow) : b_.build_zext_i64(narrow);
            }
            return b_.build_trunc_i32(int_expr(Type::i64(), depth - 1));
        }
        case 15: {
            Value* x = int_expr(t, depth - 1);
            Value* d = safe_divisor(int_expr(t, depth - 1));
            return build_div(b_, rng_.next_u32(), x, d);
        }
        case 16: {
            Value* x = int_expr(t, depth - 1);
            return b_.build_mul(x, interesting_const(t));
        }
        default: {
            Value* x = int_expr(t, depth - 1);
            return b_.build_add(x, konst(t, rng_.range_i64(-3, 7)));
        }
    }
}

void GenState::mix(Value* v) {
    Value* v64 = to_i64(v);
    Value* scaled = b_.build_mul(acc_, konst(Type::i64(), kMixMul));
    acc_ = b_.build_xor(scaled, v64);
}

uint32_t GenState::trip_budget() const {
    return iter_product_ >= opts_.max_iteration_product ? 1u : opts_.max_iteration_product / iter_product_;
}

Value* GenState::trip_count(uint32_t max_trip) {
    if (rng_.coin_flip(0.5)) {
        return konst(Type::i64(), static_cast<int64_t>(rng_.range_u64(rng_.coin_flip(0.1) ? 0 : 1, max_trip)));
    }
    uint64_t mask = 1;
    while (mask * 2 + 1 <= max_trip) mask = mask * 2 + 1;
    return b_.build_and(to_i64(pick(rng_.coin_flip(0.7) ? Type::i64() : Type::i32())),
                        konst(Type::i64(), static_cast<int64_t>(mask)));
}

void GenState::emit_loop(Value* n, uint32_t max_trip, LoopCtx& ctx,
                         const std::function<void(Value* iv, LoopCtx& ctx)>& body) {
    BasicBlock* hdr = new_block("loop_hdr");
    BasicBlock* body_bb = new_block("loop_body");
    BasicBlock* exit = new_block("loop_exit");

    std::vector<Type> carried_types;
    for (Value* v : ctx.carried) carried_types.push_back(v->type());

    std::vector<Value*> init{konst(Type::i64(), 0), acc_};
    init.insert(init.end(), ctx.carried.begin(), ctx.carried.end());
    b_.build_br(hdr, init);

    b_.position_at_end(hdr);
    const Mark m = mark();
    Value* iv = b_.add_block_param(hdr, Type::i64());
    acc_ = b_.add_block_param(hdr, Type::i64());
    for (size_t i = 0; i < ctx.carried.size(); ++i) {
        ctx.carried[i] = b_.add_block_param(hdr, carried_types[i]);
        add_value(ctx.carried[i]);
    }
    add_value(iv);
    Value* in_range = b_.build_slt(iv, n);
    std::vector<Value*> exit_args{acc_};
    exit_args.insert(exit_args.end(), ctx.carried.begin(), ctx.carried.end());
    b_.build_br_if(in_range, body_bb, Span<Value* const>(), exit, Span<Value* const>(exit_args));

    b_.position_at_end(body_bb);
    ctx.exit = exit;
    loops_.push_back(&ctx);
    iter_product_ *= std::max<uint32_t>(max_trip, 1);
    body(iv, ctx);
    iter_product_ /= std::max<uint32_t>(max_trip, 1);
    loops_.pop_back();

    Value* next = b_.build_add(iv, konst(Type::i64(), 1));
    std::vector<Value*> latch{next, acc_};
    latch.insert(latch.end(), ctx.carried.begin(), ctx.carried.end());
    b_.build_br(hdr, latch);
    restore(m);

    b_.position_at_end(exit);
    acc_ = b_.add_block_param(exit, Type::i64());
    for (size_t i = 0; i < ctx.carried.size(); ++i) {
        ctx.carried[i] = b_.add_block_param(exit, carried_types[i]);
    }
}

void GenState::emit_break_if(Value* c) {
    LoopCtx& ctx = *loops_.back();
    BasicBlock* cont = new_block("loop_cont");
    std::vector<Value*> args{acc_};
    args.insert(args.end(), ctx.carried.begin(), ctx.carried.end());
    b_.build_br_if(c, ctx.exit, Span<Value* const>(args), cont, Span<Value* const>());
    b_.position_at_end(cont);
}

void GenState::stmts(int depth, uint32_t count) {
    for (uint32_t i = 0; i < count && stmts_left_ > 0; ++i) stmt(depth);
}

void GenState::stmt(int depth) {
    if (stmts_left_ > 0) --stmts_left_;
    const bool can_nest = depth < static_cast<int>(opts_.max_depth);
    const bool has_mem = !buffers_.empty();
    struct Choice { uint32_t weight; int kind; };
    const Choice table[] = {
        {18, 0},                                            // arithmetic
        {5, 1},                                             // division
        {can_nest ? 9u : 0u, 2},                            // diamond
        {can_nest && !helper_mode_ ? 5u : 0u, 3},           // switch
        {can_nest ? 9u : 0u, 4},                            // phi of constants
        {can_nest && trip_budget() >= 2 ? 12u : 0u, 5},     // loop
        {has_mem ? 18u : 0u, 6},                            // memory
        {has_mem && can_nest && trip_budget() >= 4 ? 7u : 0u, 7},  // loop group
        {!helpers_.empty() || !buf_helpers_.empty() ? 5u : 0u, 8}, // call
        {opts_.enable_f64 ? 4u : 0u, 9},                    // f64
        {opts_.enable_vectors && !helper_mode_ ? 2u : 0u, 10},     // vector
        {helper_mode_ ? 0u : 2u, 11},                       // safepoint
    };
    uint32_t total = 0;
    for (const Choice& c : table) total += c.weight;
    uint32_t r = static_cast<uint32_t>(rng_.next_u64() % total);
    int kind = 0;
    for (const Choice& c : table) {
        if (r < c.weight) { kind = c.kind; break; }
        r -= c.weight;
    }
    switch (kind) {
        case 1: stmt_div(); break;
        case 2: stmt_diamond(depth); break;
        case 3: stmt_switch(depth); break;
        case 4: stmt_phi_consts(depth); break;
        case 5: stmt_loop(depth); break;
        case 6: stmt_memory(depth); break;
        case 7: stmt_loop_group(); break;
        case 8: stmt_call(); break;
        case 9: stmt_f64(); break;
        case 10: stmt_vector(); break;
        case 11: b_.build_safepoint(); break;
        default: stmt_arith(); break;
    }
}

void GenState::stmt_arith() {
    const uint32_t n = 1 + rng_.next_u32() % 3;
    for (uint32_t i = 0; i < n; ++i) {
        const Type t = rng_.coin_flip(0.6) ? Type::i64() : Type::i32();
        Value* v = int_expr(t, 1 + static_cast<int>(rng_.next_u32() % 3));
        add_value(v);
        // Some results stay unused so DCE has something to remove.
        if (rng_.coin_flip(0.8)) mix(v);
    }
}

void GenState::stmt_div() {
    const Type t = rng_.coin_flip(0.5) ? Type::i64() : Type::i32();
    const int64_t min_v = t == Type::i64() ? std::numeric_limits<int64_t>::min()
                                           : static_cast<int64_t>(std::numeric_limits<int32_t>::min());
    Value* x = nullptr;
    Value* d = nullptr;
    if (opts_.enable_div_overflow && rng_.coin_flip(0.3)) {
        // Reaches INT_MIN / -1 on some inputs.
        Value* c1 = cond();
        Value* other = int_expr(t, 1);
        x = b_.build_select(c1, konst(t, min_v), other);
        Value* c2 = cond();
        Value* dv = safe_divisor(int_expr(t, 1));
        d = b_.build_select(c2, konst(t, -1), dv);
    } else {
        x = int_expr(t, 2);
        d = safe_divisor(int_expr(t, 1));
    }
    Value* q = build_div(b_, rng_.next_u32(), x, d);
    add_value(q);
    mix(q);
}

void GenState::stmt_diamond(int depth) {
    Value* c = cond();
    const Type et = rng_.coin_flip(0.5) ? Type::i64() : Type::i32();
    const bool extra = rng_.coin_flip(0.6);
    BasicBlock* t_bb = new_block("then");
    BasicBlock* f_bb = new_block("else");
    BasicBlock* m_bb = new_block("merge");
    b_.build_br_if(c, t_bb, {}, f_bb, {});

    Value* acc0 = acc_;
    const Mark m = mark();
    for (BasicBlock* arm : {t_bb, f_bb}) {
        b_.position_at_end(arm);
        acc_ = acc0;
        stmts(depth + 1, static_cast<uint32_t>(rng_.next_u32() % 3));
        std::vector<Value*> out{acc_};
        if (extra) out.push_back(int_expr(et, 1));
        b_.build_br(m_bb, out);
        restore(m);
    }
    b_.position_at_end(m_bb);
    acc_ = b_.add_block_param(m_bb, Type::i64());
    if (extra) add_value(b_.add_block_param(m_bb, et));
}

void GenState::stmt_switch(int depth) {
    const Type st = rng_.coin_flip(0.7) ? Type::i64() : Type::i32();
    Value* sel = b_.build_and(pick(st), konst(st, 7));
    const Type et = rng_.coin_flip(0.5) ? Type::i64() : Type::i32();
    BasicBlock* m_bb = new_block("sw_merge");

    const uint32_t ncases = 2 + rng_.next_u32() % 4;
    std::vector<int64_t> values;
    while (values.size() < ncases) {
        const int64_t v = rng_.range_i64(0, 7);
        if (std::find(values.begin(), values.end(), v) == values.end()) values.push_back(v);
    }

    Value* acc0 = acc_;
    std::vector<SwitchCase> cases;
    std::vector<BasicBlock*> arms;
    for (int64_t v : values) {
        if (rng_.coin_flip(0.35)) {
            // Straight to the merge with a constant: a phi of constants.
            cases.emplace_back(v, m_bb, std::vector<Value*>{acc0, konst(et, rng_.range_i64(-4, 9))});
        } else {
            BasicBlock* arm = new_block("sw_case");
            arms.push_back(arm);
            cases.emplace_back(v, arm);
        }
    }
    BasicBlock* def = new_block("sw_default");
    arms.push_back(def);
    b_.build_switch(sel, def, Span<Value* const>(), Span<const SwitchCase>(cases));

    const Mark m = mark();
    for (BasicBlock* arm : arms) {
        b_.position_at_end(arm);
        acc_ = acc0;
        stmts(depth + 1, static_cast<uint32_t>(rng_.next_u32() % 2));
        std::vector<Value*> out{acc_, int_expr(et, 1)};
        b_.build_br(m_bb, out);
        restore(m);
    }
    b_.position_at_end(m_bb);
    acc_ = b_.add_block_param(m_bb, Type::i64());
    Value* merged = b_.add_block_param(m_bb, et);
    add_value(merged);
    mix(merged);
}

void GenState::stmt_phi_consts(int depth) {
    const Type t = rng_.coin_flip(0.6) ? Type::i64() : Type::i32();
    std::vector<int64_t> consts;
    auto fresh_const = [&]() {
        const int64_t c = rng_.coin_flip(0.8) ? rng_.range_i64(-3, 9)
                                              : (t == Type::i64() ? rng_.boundary_i64() : rng_.boundary_i32());
        consts.push_back(c);
        return c;
    };

    BasicBlock* m_bb = new_block("pc_merge");
    Value* acc0 = acc_;
    const Mark m = mark();
    if (rng_.coin_flip(0.5)) {
        Value* c = cond();
        BasicBlock* p1 = new_block("pc_pred");
        BasicBlock* p2 = new_block("pc_pred");
        b_.build_br_if(c, p1, {}, p2, {});
        for (BasicBlock* p : {p1, p2}) {
            b_.position_at_end(p);
            acc_ = acc0;
            stmts(depth + 1, static_cast<uint32_t>(rng_.next_u32() % 2));
            Value* k = konst(t, fresh_const());
            b_.build_br(m_bb, {acc_, k});
            restore(m);
        }
    } else {
        Value* sel = b_.build_and(pick(Type::i64()), konst(Type::i64(), 3));
        std::vector<SwitchCase> cases;
        for (int64_t v = 0; v < 3; ++v) {
            cases.emplace_back(v, m_bb, std::vector<Value*>{acc0, konst(t, fresh_const())});
        }
        std::vector<Value*> def_args{acc0, konst(t, fresh_const())};
        b_.build_switch(sel, m_bb, Span<Value* const>(def_args), Span<const SwitchCase>(cases));
    }
    b_.position_at_end(m_bb);
    acc_ = b_.add_block_param(m_bb, Type::i64());
    Value* p = b_.add_block_param(m_bb, t);

    // Compare the merged constant and branch; both successors use it. Chains
    // repeat the shape on a constant chosen in each successor.
    const uint32_t chain = 1 + rng_.next_u32() % 3;
    for (uint32_t i = 0; i < chain; ++i) {
        const int64_t k = rng_.coin_flip(0.8) ? consts[rng_.pick_index(consts.size())] : rng_.range_i64(-3, 9);
        Value* c = build_cmp(b_, rng_.next_u32(), p, konst(t, k));
        BasicBlock* t_bb = new_block("pc_true");
        BasicBlock* f_bb = new_block("pc_false");
        BasicBlock* j_bb = new_block("pc_join");
        b_.build_br_if(c, t_bb, {}, f_bb, {});
        Value* accj = acc_;
        const Mark mj = mark();
        consts.clear();
        for (BasicBlock* arm : {t_bb, f_bb}) {
            b_.position_at_end(arm);
            acc_ = accj;
            add_value(p);
            Value* other = operand(t);
            Value* v = rng_.coin_flip(0.5) ? b_.build_add(p, other) : b_.build_xor(p, other);
            mix(v);
            if (rng_.coin_flip(0.3)) stmts(depth + 1, 1);
            Value* next = rng_.coin_flip(0.75) ? konst(t, fresh_const()) : p;
            b_.build_br(j_bb, {acc_, next});
            restore(mj);
        }
        b_.position_at_end(j_bb);
        acc_ = b_.add_block_param(j_bb, Type::i64());
        p = b_.add_block_param(j_bb, t);
        if (consts.empty()) consts.push_back(0);
    }
    add_value(p);
    mix(p);
}

void GenState::stmt_loop(int depth) {
    const uint32_t budget = trip_budget();
    const uint32_t cap = std::min<uint32_t>(budget, depth == 0 ? 24u : 8u);
    const uint32_t max_trip = static_cast<uint32_t>(rng_.range_u64(2, std::max<uint32_t>(cap, 2)));
    Value* n = trip_count(max_trip);
    const uint32_t variant = rng_.next_u32() % 5;

    LoopCtx ctx;
    const uint32_t ncarried = rng_.next_u32() % 3;
    for (uint32_t i = 0; i < ncarried; ++i) {
        ctx.carried.push_back(operand(rng_.coin_flip(0.6) ? Type::i64() : Type::i32()));
    }
    // Unswitch pattern: a condition computed outside the loop.
    Value* invariant = variant == 1 ? cond() : nullptr;
    size_t state_idx = 0;
    if (variant == 2) {
        state_idx = ctx.carried.size();
        ctx.carried.push_back(konst(Type::i64(), rng_.range_i64(0, 3)));
    }

    emit_loop(n, max_trip, ctx, [&](Value* iv, LoopCtx& c) {
        if (!helper_mode_ && rng_.coin_flip(0.4)) b_.build_safepoint();
        if (invariant) {
            BasicBlock* t_bb = new_block("unsw_then");
            BasicBlock* f_bb = new_block("unsw_else");
            BasicBlock* m_bb = new_block("unsw_merge");
            b_.build_br_if(invariant, t_bb, {}, f_bb, {});
            Value* acc0 = acc_;
            const Mark m = mark();
            for (BasicBlock* arm : {t_bb, f_bb}) {
                b_.position_at_end(arm);
                acc_ = acc0;
                stmts(depth + 1, 1 + rng_.next_u32() % 2);
                b_.build_br(m_bb, {acc_});
                restore(m);
            }
            b_.position_at_end(m_bb);
            acc_ = b_.add_block_param(m_bb, Type::i64());
        }
        if (variant == 2) {
            // A state machine: the state arrives as a constant from each arm.
            Value* sel = b_.build_and(c.carried[state_idx], konst(Type::i64(), 3));
            BasicBlock* m_bb = new_block("sm_merge");
            std::vector<SwitchCase> cases;
            std::vector<BasicBlock*> arms;
            for (int64_t v = 0; v < 3; ++v) {
                arms.push_back(new_block("sm_state"));
                cases.emplace_back(v, arms.back());
            }
            arms.push_back(new_block("sm_default"));
            b_.build_switch(sel, arms.back(), Span<Value* const>(), Span<const SwitchCase>(cases));
            Value* acc0 = acc_;
            const Mark m = mark();
            for (BasicBlock* arm : arms) {
                b_.position_at_end(arm);
                acc_ = acc0;
                mix(konst(Type::i64(), rng_.range_i64(1, 99)));
                if (rng_.coin_flip(0.4)) stmts(depth + 1, 1);
                b_.build_br(m_bb, {acc_, konst(Type::i64(), rng_.range_i64(0, 3))});
                restore(m);
            }
            b_.position_at_end(m_bb);
            acc_ = b_.add_block_param(m_bb, Type::i64());
            c.carried[state_idx] = b_.add_block_param(m_bb, Type::i64());
        }
        stmts(depth + 1, 1 + rng_.next_u32() % 3);
        for (size_t i = 0; i < c.carried.size(); ++i) {
            if (variant == 2 && i == state_idx) continue;
            Value* step = int_expr(c.carried[i]->type(), 1);
            c.carried[i] = rng_.coin_flip(0.5) ? b_.build_add(c.carried[i], step) : b_.build_xor(c.carried[i], step);
        }
        if (variant == 3 || rng_.coin_flip(0.15)) {
            Value* bits = b_.build_and(rng_.coin_flip(0.5) ? acc_ : iv, konst(Type::i64(), 7));
            emit_break_if(b_.build_eq(bits, konst(Type::i64(), rng_.range_i64(0, 7))));
        }
        if (rng_.coin_flip(0.5)) mix(iv);
    });
    for (Value* v : ctx.carried) {
        add_value(v);
        mix(v);
    }
}

void GenState::stmt_call() {
    const bool use_buf = !buf_helpers_.empty() && (helpers_.empty() || rng_.coin_flip(0.4));
    if (use_buf) {
        std::vector<const Buffer*> eligible;
        for (const Buffer& buf : buffers_) {
            if (buf.gc && buf.elem == Type::i64() && buf.len >= 4) eligible.push_back(&buf);
        }
        if (!eligible.empty()) {
            const Buffer& buf = *eligible[rng_.pick_index(eligible.size())];
            Value* i = operand(Type::i64());
            Value* r = b_.build_call(buf_helpers_[rng_.pick_index(buf_helpers_.size())], Type::i64(), {buf.base, i});
            add_value(r);
            mix(r);
            return;
        }
    }
    if (helpers_.empty()) {
        stmt_arith();
        return;
    }
    Value* x = operand(Type::i64());
    Value* y = operand(Type::i64());
    Value* r = b_.build_call(helpers_[rng_.pick_index(helpers_.size())], Type::i64(), {x, y});
    add_value(r);
    mix(r);
}

void GenState::stmt_f64() {
    // Integer-valued inputs below 2^20 and at most four bounded steps keep
    // every value finite, non-NaN and far inside the i64 range.
    auto fresh = [&]() {
        Value* raw = to_i64(int_expr(Type::i64(), 1));
        return b_.build_sitofp_f64_i64(b_.build_and(raw, konst(Type::i64(), 0xFFFFF)));
    };
    Value* f = fresh();
    Value* g = fresh();
    bool zero_sign_unspecified = false;
    const uint32_t steps = 1 + rng_.next_u32() % 4;
    static const double kFactors[] = {-8.0, -3.0, -1.0, -0.5, 0.25, 0.5, 2.0, 3.0, 7.0, 0.1, 1.1};
    for (uint32_t i = 0; i < steps; ++i) {
        switch (rng_.next_u32() % 8) {
            case 0: f = b_.build_fadd(f, g); break;
            case 1: f = b_.build_sub(f, g); break;
            case 2: {
                const size_t n = sizeof(kFactors) / sizeof(kFactors[0]);
                // The fractional factors are rare: they are the ones whose
                // products round, which is what FMA contraction changes.
                const size_t idx = rng_.coin_flip(0.85) ? rng_.pick_index(n - 2) : n - 2 + rng_.pick_index(2);
                f = b_.build_mul(f, b_.build_fconst_f64(kFactors[idx]));
                break;
            }
            case 3:
                f = rng_.coin_flip(0.5) ? b_.build_fmin_f64(f, g) : b_.build_fmax_f64(f, g);
                zero_sign_unspecified = true;
                break;
            case 4: f = b_.build_fabs_f64(f); break;
            case 5: f = rng_.coin_flip(0.5) ? b_.build_floor_f64(f) : b_.build_ceil_f64(f); break;
            case 6: f = b_.build_sqrt_f64(b_.build_fabs_f64(f)); break;
            default: {
                Value* sum = b_.build_fadd(b_.build_mul(f, b_.build_fconst_f64(0.5)), g);
                f = sum;
                break;
            }
        }
    }
    // fmin/fmax may return either zero for (-0, +0); fptosi erases the sign.
    Value* out = (zero_sign_unspecified || rng_.coin_flip(0.5)) ? b_.build_fptosi_i64(f)
                                                              : b_.build_bitcast_i64_f64(f);
    add_value(out);
    mix(out);
}

void GenState::stmt_vector() {
    Value* x = int_expr(Type::i32(), 1);
    Value* y = int_expr(Type::i32(), 1);
    Value* vx = b_.build_vbroadcast(Type::i32x4(), x);
    Value* vy = b_.build_vbroadcast(Type::i32x4(), y);
    if (rng_.coin_flip(0.5)) {
        vx = b_.build_vinsert_lane(vx, operand(Type::i32()), static_cast<uint32_t>(rng_.next_u32() % 4));
    }
    Value* r = nullptr;
    switch (rng_.next_u32() % 6) {
        case 0: r = b_.build_vadd(vx, vy); break;
        case 1: r = b_.build_vsub(vx, vy); break;
        case 2: r = b_.build_vmul(vx, vy); break;
        case 3: r = b_.build_vand(vx, vy); break;
        case 4: r = b_.build_vor(vx, vy); break;
        default: r = b_.build_vxor(vx, vy); break;
    }
    Value* lane = b_.build_vextract_lane(r, static_cast<uint32_t>(rng_.next_u32() % 4));
    add_value(lane);
    mix(lane);
}

Function* GenState::generate_entry(std::string_view name) {
    fn_ = mod_.create_function(name, Type::i64(), {Type::i64(), Type::i64()});
    b_.set_function(fn_);
    BasicBlock* entry = b_.append_block("entry");
    b_.position_at_end(entry);
    Value* a = b_.add_block_param(entry, Type::i64());
    Value* c = b_.add_block_param(entry, Type::i64());
    add_value(a);
    add_value(c);
    add_value(b_.build_trunc_i32(a));
    add_value(b_.build_trunc_i32(c));
    acc_ = b_.build_xor(a, konst(Type::i64(), rng_.next_i64()));
    mix(c);

    stmts_left_ = helper_mode_ ? 2 + rng_.next_u32() % 4 : opts_.max_statements;
    if (!helper_mode_ && opts_.enable_memory) setup_buffers();
    while (stmts_left_ > 0) stmt(0);
    if (!helper_mode_) checksum_buffers();
    b_.build_ret(acc_);
    fn_->rebuild_cfg_predecessors();
    return fn_;
}

} // namespace detail

namespace {

// (gcref buf, i64 i) -> i64 over a gc buffer of at least four i64s; some
// variants also write the buffer, so callers see the store.
void build_buffer_helper(Module& mod, const std::string& name, FuzzRng& rng) {
    Function* fn = mod.create_function(name, Type::i64(), {Type::gcref(), Type::i64()});
    Builder b(mod);
    b.set_function(fn);
    BasicBlock* entry = b.append_block("entry");
    b.position_at_end(entry);
    Value* buf = b.add_block_param(entry, Type::gcref());
    Value* i = b.add_block_param(entry, Type::i64());
    Value* idx = b.build_and(i, b.build_iconst_i64(3));
    Value* v = b.build_load_indexed(Type::i64(), buf, idx, 8);
    if (rng.coin_flip(0.6)) {
        Value* next = b.build_and(b.build_add(i, b.build_iconst_i64(1)), b.build_iconst_i64(3));
        Value* stored = b.build_add(v, b.build_iconst_i64(rng.range_i64(1, 1000)));
        b.build_store_indexed(Type::i64(), buf, next, 8, stored);
    }
    Value* r = b.build_xor(b.build_mul(v, b.build_iconst_i64(rng.range_i64(3, 99) | 1)), i);
    b.build_ret(r);
    fn->rebuild_cfg_predecessors();
}

// Lays blocks out in reverse post-order, the order a frontend emits, instead
// of creation order (where a loop's exit precedes its body's inner blocks).
void order_blocks_rpo(Function& fn) {
    std::vector<BasicBlock*>& blocks = fn.blocks();
    if (blocks.empty()) return;
    std::vector<BasicBlock*> post;
    std::vector<BasicBlock*> seen;
    std::vector<std::pair<BasicBlock*, size_t>> stack{{blocks.front(), 0}};
    seen.push_back(blocks.front());
    while (!stack.empty()) {
        auto& [bb, next] = stack.back();
        const std::vector<BasicBlock*> succs = bb->successors();
        if (next < succs.size()) {
            BasicBlock* s = succs[next++];
            if (std::find(seen.begin(), seen.end(), s) == seen.end()) {
                seen.push_back(s);
                stack.emplace_back(s, 0);
            }
            continue;
        }
        post.push_back(bb);
        stack.pop_back();
    }
    std::vector<BasicBlock*> order(post.rbegin(), post.rend());
    for (BasicBlock* bb : blocks) {
        if (std::find(order.begin(), order.end(), bb) == order.end()) order.push_back(bb);
    }
    blocks = std::move(order);
}

void apply_safe_mutations(Function& fn, FuzzRng& rng) {
    const uint32_t count = rng.next_u32() % 3;
    for (uint32_t i = 0; i < count; ++i) {
        switch (rng.next_u32() % 3) {
            case 0: IrMutator::swap_commutative_operands(fn, rng); break;
            case 1: IrMutator::split_basic_blocks(fn, rng); break;
            default: IrMutator::inject_speculative_guards(fn, rng); break;
        }
        fn.rebuild_cfg_predecessors();
    }
}

} // namespace

Function* ProgramGenerator::generate(Module& mod, std::string_view name, uint64_t seed) {
    FuzzRng top(seed ^ 0x5bd1e995ULL);
    std::vector<std::string> helpers;
    std::vector<std::string> buf_helpers;
    if (options_.enable_calls) {
        const uint32_t nh = top.next_u32() % 3;
        for (uint32_t k = 0; k < nh; ++k) {
            ProgramGeneratorOptions ho = options_;
            ho.max_depth = 1;
            ho.max_iteration_product = 16;
            ho.enable_memory = false;
            ho.enable_calls = false;
            detail::GenState h(mod, ho, top.next_u64());
            h.helper_mode_ = true;
            const std::string hname = std::string(name) + "_helper" + std::to_string(k);
            h.generate_entry(hname);
            helpers.push_back(hname);
        }
        if (options_.enable_memory && options_.enable_gc) {
            const uint32_t nb = top.next_u32() % 2;
            for (uint32_t k = 0; k < nb; ++k) {
                const std::string bname = std::string(name) + "_bufhelper" + std::to_string(k);
                build_buffer_helper(mod, bname, top);
                buf_helpers.push_back(bname);
            }
        }
    }

    detail::GenState g(mod, options_, seed);
    g.helpers_ = helpers;
    g.buf_helpers_ = buf_helpers;
    Function* fn = g.generate_entry(name);
    // Mostly frontend order; sometimes creation order, which passes must
    // handle just the same.
    if (top.coin_flip(0.85)) {
        for (Function* f : mod.functions()) {
            if (f) order_blocks_rpo(*f);
        }
    }
    if (options_.apply_mutations) {
        FuzzRng mrng(seed ^ 0xDEADBEEFULL);
        apply_safe_mutations(*fn, mrng);
    }
    return fn;
}

} // namespace brass::fuzz
