#include <brass/mir/egraph.hpp>
#include <algorithm>
#include <bit>
#include <vector>

namespace brass {

namespace {

constexpr uint64_t mask_w(uint64_t val, uint32_t w) noexcept {
    return (w < 64) ? (val & ((1ULL << w) - 1ULL)) : val;
}

constexpr int64_t sext_w(uint64_t val, uint32_t w) noexcept {
    return (w == 32) ? static_cast<int64_t>(static_cast<int32_t>(val & 0xFFFFFFFFULL)) : static_cast<int64_t>(val);
}

bool try_fold_const(const EGraph& eg, const ENode& n, uint64_t& out_bits) noexcept {
    const uint32_t w = n.width, ar = eop_arity(n.op);
    uint64_t ca = 0, cb = 0, cc = 0;
    if (ar == 0 || !eg.is_const(n.a, ca)) return false;
    if (ar == 1) {
        if (n.op == EOp::Neg)     { out_bits = mask_w(0ULL - ca, w); return true; }
        if (n.op == EOp::Not)     { out_bits = mask_w(~ca, w); return true; }
        if (n.op == EOp::SExt64)  { out_bits = static_cast<uint64_t>(sext_w(ca, 32)); return true; }
        if (n.op == EOp::ZExt64 || n.op == EOp::Trunc32) { out_bits = ca & 0xFFFFFFFFULL; return true; }
        return false;
    }
    if (!eg.is_const(n.b, cb)) return false;
    if (ar == 3) {
        if (!eg.is_const(n.c, cc) || n.op != EOp::Select) return false;
        out_bits = (mask_w(ca, eg.class_width(n.a)) != 0) ? mask_w(cb, w) : mask_w(cc, w);
        return true;
    }
    const uint32_t op_w = eg.class_width(n.a), sh_mask = (w == 32) ? 31u : 63u;
    ca = mask_w(ca, op_w); cb = mask_w(cb, eg.class_width(n.b));
    switch (n.op) {
        case EOp::Add:  out_bits = mask_w(ca + cb, w); return true;
        case EOp::Sub:  out_bits = mask_w(ca - cb, w); return true;
        case EOp::Mul:  out_bits = mask_w(ca * cb, w); return true;
        case EOp::And:  out_bits = mask_w(ca & cb, w); return true;
        case EOp::Or:   out_bits = mask_w(ca | cb, w); return true;
        case EOp::Xor:  out_bits = mask_w(ca ^ cb, w); return true;
        case EOp::Shl:  out_bits = mask_w(ca << (cb & sh_mask), w); return true;
        case EOp::LShr: out_bits = mask_w(ca >> (cb & sh_mask), w); return true;
        case EOp::AShr: out_bits = mask_w(static_cast<uint64_t>(sext_w(ca, w) >> (cb & sh_mask)), w); return true;
        case EOp::UDiv: if (cb == 0) return false; out_bits = mask_w(ca / cb, w); return true;
        case EOp::UMod: if (cb == 0) return false; out_bits = mask_w(ca % cb, w); return true;
        case EOp::SDiv: case EOp::SMod: {
            const int64_t sa = sext_w(ca, w), sb = sext_w(cb, w), smin = (w == 32) ? static_cast<int64_t>(INT32_MIN) : INT64_MIN;
            if (sb == 0 || (sa == smin && sb == -1)) return false;
            out_bits = mask_w(static_cast<uint64_t>((n.op == EOp::SDiv) ? (sa / sb) : (sa % sb)), w);
            return true;
        }
        case EOp::Eq:  out_bits = (ca == cb) ? 1ULL : 0ULL; return true;
        case EOp::Ne:  out_bits = (ca != cb) ? 1ULL : 0ULL; return true;
        case EOp::Ult: out_bits = (ca < cb)  ? 1ULL : 0ULL; return true;
        case EOp::Ule: out_bits = (ca <= cb) ? 1ULL : 0ULL; return true;
        case EOp::Slt: out_bits = (sext_w(ca, op_w) < sext_w(cb, op_w))  ? 1ULL : 0ULL; return true;
        case EOp::Sle: out_bits = (sext_w(ca, op_w) <= sext_w(cb, op_w)) ? 1ULL : 0ULL; return true;
        default:       return false;
    }
}

bool match_shared_pair(
    EClassId a0, EClassId a1, EClassId b0, EClassId b1,
    EClassId& out_shared, EClassId& out_r0, EClassId& out_r1) noexcept {
    if (a0 == b0) { out_shared = a0; out_r0 = a1; out_r1 = b1; return true; }
    if (a0 == b1) { out_shared = a0; out_r0 = a1; out_r1 = b0; return true; }
    if (a1 == b0) { out_shared = a1; out_r0 = a0; out_r1 = b1; return true; }
    if (a1 == b1) { out_shared = a1; out_r0 = a0; out_r1 = b0; return true; }
    return false;
}

} // namespace

uint32_t EGraph::saturate(const EGraphLimits& limits, EGraphStats* stats) {
    rebuild();
    uint32_t total_rewrites = 0, iter_rewrites = 0;
    bool budget_exceeded = false;

    auto can_alloc = [&]() noexcept -> bool {
        if (budget_exceeded) return false;
        if (num_enodes() >= limits.max_enodes || num_eclasses() >= limits.max_eclasses) {
            budget_exceeded = true;
            if (stats) stats->budget_hit++;
            return false;
        }
        return true;
    };

    auto make_node_checked = [&](const ENode& raw) -> EClassId {
        const ENode canon = canonicalize(raw);
        auto it = memo_.find(canon);
        if (it != memo_.end()) return find(it->second);
        if (!can_alloc()) return kInvalidEClass;
        const EClassId cid = add_node(canon);
        if (stats) stats->enodes_added++;
        return cid;
    };

    auto make_const_checked = [&](uint64_t imm, uint32_t w) -> EClassId {
        return make_node_checked(ENode{EOp::Const, w, kInvalidEClass, kInvalidEClass, kInvalidEClass, mask_w(imm, w)});
    };
    auto make_unary_checked = [&](EOp op, uint32_t w, EClassId a) -> EClassId {
        return (a == kInvalidEClass) ? kInvalidEClass : make_node_checked(ENode{op, w, a, kInvalidEClass, kInvalidEClass, 0});
    };
    auto make_bin_checked = [&](EOp op, uint32_t w, EClassId a, EClassId b) -> EClassId {
        return (a == kInvalidEClass || b == kInvalidEClass) ? kInvalidEClass : make_node_checked(ENode{op, w, a, b, kInvalidEClass, 0});
    };

    auto apply_merge = [&](EClassId c1, EClassId c2) -> bool {
        if (c1 == kInvalidEClass || c2 == kInvalidEClass || !merge(c1, c2)) return false;
        ++iter_rewrites;
        ++total_rewrites;
        if (stats) { stats->eclasses_merged++; stats->rewrites_applied++; }
        return true;
    };

    auto class_has_unary = [&](EClassId cid, EOp op, EClassId& out_x) noexcept -> bool {
        for (const ENode& m : class_nodes(cid)) {
            if (m.op == op) { out_x = m.a; return true; }
        }
        return false;
    };
    auto class_is_not_of = [&](EClassId cid, EClassId target) noexcept -> bool {
        const EClassId t = find(target);
        for (const ENode& m : class_nodes(cid)) {
            if (m.op == EOp::Not && find(m.a) == t) return true;
        }
        return false;
    };

    for (uint32_t iter = 0; iter < limits.max_iters; ++iter) {
        if (!can_alloc()) break;
        iter_rewrites = 0;
        if (stats) stats->saturation_iters++;

        struct SnapshotClass { EClassId cid; std::vector<ENode> nodes; };
        std::vector<SnapshotClass> snap;
        snap.reserve(classes_.size());
        for (size_t i = 0; i < classes_.size(); ++i) {
            if (parents_uf_[i] == static_cast<EClassId>(i)) snap.push_back({static_cast<EClassId>(i), classes_[i].nodes});
        }

        // Phase 1: Contractions, Identities, Cancellations, Factoring
        for (const auto& sc : snap) {
            if (budget_exceeded) break;
            const EClassId cid = find(sc.cid);
            for (const ENode& raw_n : sc.nodes) {
                if (budget_exceeded) break;
                const ENode n = canonicalize(raw_n);
                const uint32_t w = n.width;
                const uint64_t all_ones = mask_w(~0ULL, w);

                uint64_t folded_imm = 0;
                if (try_fold_const(*this, n, folded_imm)) {
                    apply_merge(cid, make_const_checked(folded_imm, w));
                    continue;
                }

                uint64_t ca = 0, cb = 0;
                const bool is_ca = (eop_arity(n.op) >= 1) && is_const(n.a, ca);
                const bool is_cb = (eop_arity(n.op) >= 2) && is_const(n.b, cb);
                const EClassId x_nonconst = is_ca ? n.b : n.a;
                const uint64_t c_const = is_ca ? ca : cb;
                const bool has_one_const = (is_ca != is_cb);

                switch (n.op) {
                    case EOp::Neg: {
                        EClassId inner = kInvalidEClass;
                        if (class_has_unary(n.a, EOp::Neg, inner)) apply_merge(cid, inner);
                        if (class_has_unary(n.a, EOp::Not, inner)) apply_merge(cid, make_bin_checked(EOp::Add, w, inner, make_const_checked(1, w)));
                        for (const ENode& ma : class_nodes(n.a)) {
                            uint64_t mc = 0;
                            if (ma.op == EOp::Sub) apply_merge(cid, make_bin_checked(EOp::Sub, w, ma.b, ma.a));
                            else if (ma.op == EOp::Add && ((is_const(ma.a, mc) && mc == 1) || (is_const(ma.b, mc) && mc == 1))) {
                                apply_merge(cid, make_unary_checked(EOp::Not, w, (is_const(ma.a, mc) && mc == 1) ? ma.b : ma.a));
                            }
                        }
                        break;
                    }
                    case EOp::Not: {
                        EClassId inner = kInvalidEClass;
                        if (class_has_unary(n.a, EOp::Not, inner)) apply_merge(cid, inner);
                        break;
                    }
                    case EOp::Trunc32: {
                        EClassId inner = kInvalidEClass;
                        if ((class_has_unary(n.a, EOp::ZExt64, inner) || class_has_unary(n.a, EOp::SExt64, inner)) && class_width(inner) == 32) apply_merge(cid, inner);
                        break;
                    }
                    case EOp::Add: {
                        if (has_one_const && c_const == 0) { apply_merge(cid, x_nonconst); break; }
                        if (n.a == n.b) apply_merge(cid, make_bin_checked(EOp::Shl, w, n.a, make_const_checked(1, w)));
                        if (has_one_const && c_const == 1) {
                            EClassId not_inner = kInvalidEClass;
                            if (class_has_unary(x_nonconst, EOp::Not, not_inner)) apply_merge(cid, make_unary_checked(EOp::Neg, w, not_inner));
                        }
                        if (class_is_not_of(n.a, n.b) || class_is_not_of(n.b, n.a)) { apply_merge(cid, make_const_checked(all_ones, w)); break; }
                        EClassId neg_inner = kInvalidEClass;
                        if (class_has_unary(n.a, EOp::Neg, neg_inner)) apply_merge(cid, (find(neg_inner) == n.b) ? make_const_checked(0, w) : make_bin_checked(EOp::Sub, w, n.b, neg_inner));
                        if (class_has_unary(n.b, EOp::Neg, neg_inner)) apply_merge(cid, (find(neg_inner) == n.a) ? make_const_checked(0, w) : make_bin_checked(EOp::Sub, w, n.a, neg_inner));
                        auto check_add_child = [&](EClassId ch, EClassId other) {
                            for (const ENode& mc : class_nodes(ch)) {
                                if (mc.op == EOp::Sub && mc.b == other) apply_merge(cid, mc.a);
                                uint64_t c2 = 0, c1 = 0;
                                if (is_const(other, c2)) {
                                    if (mc.op == EOp::Add && (is_const(mc.a, c1) || is_const(mc.b, c1))) {
                                        apply_merge(cid, make_bin_checked(EOp::Add, w, is_const(mc.a, c1) ? mc.b : mc.a, make_const_checked(c1 + c2, w)));
                                    } else if (mc.op == EOp::Sub && is_const(mc.b, c1)) {
                                        apply_merge(cid, make_bin_checked(EOp::Add, w, mc.a, make_const_checked(c2 - c1, w)));
                                    }
                                }
                                if (mc.op == EOp::Mul && (mc.a == other || mc.b == other)) {
                                    apply_merge(cid, make_bin_checked(EOp::Mul, w, other, make_bin_checked(EOp::Add, w, (mc.a == other) ? mc.b : mc.a, make_const_checked(1, w))));
                                }
                            }
                        };
                        check_add_child(n.a, n.b);
                        check_add_child(n.b, n.a);
                        for (const ENode& ma : class_nodes(n.a)) {
                            for (const ENode& mb : class_nodes(n.b)) {
                                if (((ma.op == EOp::Or && mb.op == EOp::And) || (ma.op == EOp::And && mb.op == EOp::Or)) && ma.a == mb.a && ma.b == mb.b) {
                                    apply_merge(cid, make_bin_checked(EOp::Add, w, ma.a, ma.b));
                                }
                                auto try_xor_and_shl = [&](const ENode& xor_n, const ENode& shl_n) {
                                    uint64_t sh_c = 0;
                                    if (xor_n.op == EOp::Xor && shl_n.op == EOp::Shl && is_const(shl_n.b, sh_c) && sh_c == 1) {
                                        for (const ENode& mand : class_nodes(shl_n.a)) {
                                            if (mand.op == EOp::And && mand.a == xor_n.a && mand.b == xor_n.b) apply_merge(cid, make_bin_checked(EOp::Add, w, xor_n.a, xor_n.b));
                                        }
                                    }
                                };
                                try_xor_and_shl(ma, mb);
                                try_xor_and_shl(mb, ma);
                                EClassId sh = kInvalidEClass, r0 = kInvalidEClass, r1 = kInvalidEClass;
                                if (ma.op == EOp::Mul && mb.op == EOp::Mul && match_shared_pair(ma.a, ma.b, mb.a, mb.b, sh, r0, r1)) {
                                    uint64_t k0 = 0, k1 = 0;
                                    apply_merge(cid, make_bin_checked(EOp::Mul, w, sh, (is_const(r0, k0) && is_const(r1, k1)) ? make_const_checked(k0 + k1, w) : make_bin_checked(EOp::Add, w, r0, r1)));
                                }
                                if (ma.op == EOp::Shl && mb.op == EOp::Shl && ma.b == mb.b) {
                                    apply_merge(cid, make_bin_checked(EOp::Shl, w, make_bin_checked(EOp::Add, w, ma.a, mb.a), ma.b));
                                }
                            }
                        }
                        break;
                    }
                    case EOp::Sub: {
                        if (is_cb && cb == 0) { apply_merge(cid, n.a); break; }
                        if (is_ca && ca == 0) { apply_merge(cid, make_unary_checked(EOp::Neg, w, n.b)); break; }
                        if (n.a == n.b) { apply_merge(cid, make_const_checked(0, w)); break; }
                        if (class_is_not_of(n.b, n.a)) {
                            apply_merge(cid, make_bin_checked(EOp::Add, w, make_bin_checked(EOp::Shl, w, n.a, make_const_checked(1, w)), make_const_checked(1, w)));
                        }
                        EClassId neg_inner = kInvalidEClass;
                        if (class_has_unary(n.b, EOp::Neg, neg_inner)) apply_merge(cid, make_bin_checked(EOp::Add, w, n.a, neg_inner));
                        for (const ENode& ma : class_nodes(n.a)) {
                            if (ma.op == EOp::Add) {
                                if (ma.a == n.b) apply_merge(cid, ma.b);
                                else if (ma.b == n.b) apply_merge(cid, ma.a);
                                uint64_t c1 = 0;
                                if (is_cb && (is_const(ma.a, c1) || is_const(ma.b, c1))) {
                                    apply_merge(cid, make_bin_checked(EOp::Add, w, is_const(ma.a, c1) ? ma.b : ma.a, make_const_checked(c1 - cb, w)));
                                }
                            } else if (ma.op == EOp::Sub) {
                                uint64_t c1 = 0;
                                if (is_cb && is_const(ma.b, c1)) apply_merge(cid, make_bin_checked(EOp::Sub, w, ma.a, make_const_checked(c1 + cb, w)));
                            } else if (ma.op == EOp::Mul && (ma.a == n.b || ma.b == n.b)) {
                                const EClassId yv = (ma.a == n.b) ? ma.b : ma.a;
                                apply_merge(cid, make_bin_checked(EOp::Mul, w, n.b, make_bin_checked(EOp::Sub, w, yv, make_const_checked(1, w))));
                            }
                        }
                        for (const ENode& mb : class_nodes(n.b)) {
                            if (mb.op == EOp::Sub && mb.a == n.a) apply_merge(cid, mb.b);
                            else if (mb.op == EOp::Add) {
                                if (mb.a == n.a) apply_merge(cid, make_unary_checked(EOp::Neg, w, mb.b));
                                else if (mb.b == n.a) apply_merge(cid, make_unary_checked(EOp::Neg, w, mb.a));
                            }
                        }
                        for (const ENode& ma : class_nodes(n.a)) {
                            for (const ENode& mb : class_nodes(n.b)) {
                                EClassId sh = kInvalidEClass, r0 = kInvalidEClass, r1 = kInvalidEClass;
                                if (ma.op == EOp::Add && mb.op == EOp::Add && match_shared_pair(ma.a, ma.b, mb.a, mb.b, sh, r0, r1)) {
                                    apply_merge(cid, make_bin_checked(EOp::Sub, w, r0, r1));
                                }
                                if (ma.op == EOp::Or && mb.op == EOp::And && ma.a == mb.a && ma.b == mb.b) apply_merge(cid, make_bin_checked(EOp::Xor, w, ma.a, ma.b));
                                if (ma.op == EOp::Or && mb.op == EOp::Xor && ma.a == mb.a && ma.b == mb.b) apply_merge(cid, make_bin_checked(EOp::And, w, ma.a, ma.b));
                                if (ma.op == EOp::Mul && mb.op == EOp::Mul && match_shared_pair(ma.a, ma.b, mb.a, mb.b, sh, r0, r1)) {
                                    apply_merge(cid, make_bin_checked(EOp::Mul, w, sh, make_bin_checked(EOp::Sub, w, r0, r1)));
                                }
                            }
                        }
                        break;
                    }
                    case EOp::Mul: {
                        if (has_one_const) {
                            if (c_const == 0) { apply_merge(cid, make_const_checked(0, w)); break; }
                            if (c_const == 1) { apply_merge(cid, x_nonconst); break; }
                            if (c_const == all_ones) { apply_merge(cid, make_unary_checked(EOp::Neg, w, x_nonconst)); break; }
                            if (std::has_single_bit(c_const)) {
                                const uint32_t k = static_cast<uint32_t>(std::countr_zero(c_const));
                                if (k < w) apply_merge(cid, make_bin_checked(EOp::Shl, w, x_nonconst, make_const_checked(k, w)));
                            }
                            for (const ENode& mx : class_nodes(x_nonconst)) {
                                uint64_t c1 = 0;
                                if (mx.op == EOp::Mul && (is_const(mx.a, c1) || is_const(mx.b, c1))) {
                                    apply_merge(cid, make_bin_checked(EOp::Mul, w, is_const(mx.a, c1) ? mx.b : mx.a, make_const_checked(c1 * c_const, w)));
                                }
                            }
                        }
                        break;
                    }
                    case EOp::SDiv:
                    case EOp::UDiv: if (is_cb && cb == 1) apply_merge(cid, n.a); break;
                    case EOp::SMod:
                    case EOp::UMod: if (is_cb && cb == 1) apply_merge(cid, make_const_checked(0, w)); break;
                    case EOp::And:
                    case EOp::Or:
                    case EOp::Xor: {
                        if (has_one_const) {
                            if (n.op == EOp::And && c_const == 0) { apply_merge(cid, make_const_checked(0, w)); break; }
                            if (n.op == EOp::And && c_const == all_ones) { apply_merge(cid, x_nonconst); break; }
                            if (n.op == EOp::Or && c_const == 0) { apply_merge(cid, x_nonconst); break; }
                            if (n.op == EOp::Or && c_const == all_ones) { apply_merge(cid, make_const_checked(all_ones, w)); break; }
                            if (n.op == EOp::Xor && c_const == 0) { apply_merge(cid, x_nonconst); break; }
                            if (n.op == EOp::Xor && c_const == all_ones) { apply_merge(cid, make_unary_checked(EOp::Not, w, x_nonconst)); break; }
                        }
                        if (n.a == n.b) { apply_merge(cid, (n.op == EOp::Xor) ? make_const_checked(0, w) : n.a); break; }
                        if (class_is_not_of(n.a, n.b) || class_is_not_of(n.b, n.a)) {
                            apply_merge(cid, make_const_checked((n.op == EOp::And) ? 0ULL : all_ones, w));
                            break;
                        }
                        if (n.op == EOp::And || n.op == EOp::Or) {
                            EClassId na = kInvalidEClass, nb = kInvalidEClass;
                            if (class_has_unary(n.a, EOp::Not, na) && class_has_unary(n.b, EOp::Not, nb)) {
                                apply_merge(cid, make_unary_checked(EOp::Not, w, make_bin_checked((n.op == EOp::And) ? EOp::Or : EOp::And, w, na, nb)));
                            }
                        }
                        auto check_bit_child = [&](EClassId ch, EClassId other) {
                            for (const ENode& mc : class_nodes(ch)) {
                                if (n.op == EOp::And) {
                                    if (mc.op == EOp::Or && (mc.a == other || mc.b == other)) apply_merge(cid, other);
                                    else if (mc.op == EOp::Or && (class_is_not_of(other, mc.a) || class_is_not_of(other, mc.b))) {
                                        apply_merge(cid, make_bin_checked(EOp::And, w, class_is_not_of(other, mc.a) ? mc.b : mc.a, other));
                                    } else if (mc.op == EOp::And && (mc.a == other || mc.b == other)) apply_merge(cid, ch);
                                    else if (mc.op == EOp::And && (class_is_not_of(other, mc.a) || class_is_not_of(other, mc.b))) apply_merge(cid, make_const_checked(0, w));
                                    else if (mc.op == EOp::Xor && (mc.a == other || mc.b == other)) {
                                        apply_merge(cid, make_bin_checked(EOp::And, w, other, make_unary_checked(EOp::Not, w, (mc.a == other) ? mc.b : mc.a)));
                                    }
                                } else if (n.op == EOp::Or) {
                                    if (mc.op == EOp::And && (mc.a == other || mc.b == other)) apply_merge(cid, other);
                                    else if (mc.op == EOp::And && (class_is_not_of(other, mc.a) || class_is_not_of(other, mc.b))) {
                                        apply_merge(cid, make_bin_checked(EOp::Or, w, class_is_not_of(other, mc.a) ? mc.b : mc.a, other));
                                    } else if (mc.op == EOp::Or && (mc.a == other || mc.b == other)) apply_merge(cid, ch);
                                    else if (mc.op == EOp::Or && (class_is_not_of(other, mc.a) || class_is_not_of(other, mc.b))) apply_merge(cid, make_const_checked(all_ones, w));
                                    else if (mc.op == EOp::Xor && (mc.a == other || mc.b == other)) apply_merge(cid, make_bin_checked(EOp::Or, w, mc.a, mc.b));
                                } else if (n.op == EOp::Xor && mc.op == EOp::Xor) {
                                    if (mc.a == other) apply_merge(cid, mc.b);
                                    else if (mc.b == other) apply_merge(cid, mc.a);
                                }
                                uint64_t c1 = 0, c2 = 0;
                                if (mc.op == n.op && is_const(other, c2) && (is_const(mc.a, c1) || is_const(mc.b, c1))) {
                                    const uint64_t comb = (n.op == EOp::And) ? (c1 & c2) : (n.op == EOp::Or) ? (c1 | c2) : (c1 ^ c2);
                                    apply_merge(cid, make_bin_checked(n.op, w, is_const(mc.a, c1) ? mc.b : mc.a, make_const_checked(comb, w)));
                                }
                            }
                        };
                        check_bit_child(n.a, n.b);
                        check_bit_child(n.b, n.a);

                        for (const ENode& ma : class_nodes(n.a)) {
                            for (const ENode& mb : class_nodes(n.b)) {
                                EClassId sh = kInvalidEClass, r0 = kInvalidEClass, r1 = kInvalidEClass;
                                if ((n.op == EOp::Or || n.op == EOp::Xor) && ma.op == EOp::And && mb.op == EOp::And &&
                                    match_shared_pair(ma.a, ma.b, mb.a, mb.b, sh, r0, r1)) {
                                    if (class_is_not_of(r0, r1) || class_is_not_of(r1, r0)) apply_merge(cid, sh);
                                    else apply_merge(cid, make_bin_checked(EOp::And, w, sh, make_bin_checked(n.op, w, r0, r1)));
                                }
                                if (n.op == EOp::And && ma.op == EOp::Or && mb.op == EOp::Or &&
                                    match_shared_pair(ma.a, ma.b, mb.a, mb.b, sh, r0, r1)) {
                                    if (class_is_not_of(r0, r1) || class_is_not_of(r1, r0)) apply_merge(cid, sh);
                                    else apply_merge(cid, make_bin_checked(EOp::Or, w, sh, make_bin_checked(EOp::And, w, r0, r1)));
                                }
                                if (n.op == EOp::Xor && ma.op == EOp::Xor && mb.op == EOp::Xor &&
                                    match_shared_pair(ma.a, ma.b, mb.a, mb.b, sh, r0, r1)) {
                                    apply_merge(cid, make_bin_checked(EOp::Xor, w, r0, r1));
                                }
                                if (n.op == EOp::Or && ((ma.op == EOp::Xor && mb.op == EOp::And) || (ma.op == EOp::And && mb.op == EOp::Xor)) &&
                                    ma.a == mb.a && ma.b == mb.b) {
                                    apply_merge(cid, make_bin_checked(EOp::Or, w, ma.a, ma.b));
                                }
                                if (n.op == EOp::Xor && ((ma.op == EOp::Or && mb.op == EOp::And) || (ma.op == EOp::And && mb.op == EOp::Or)) &&
                                    ma.a == mb.a && ma.b == mb.b) {
                                    apply_merge(cid, make_bin_checked(EOp::Xor, w, ma.a, ma.b));
                                }
                            }
                        }
                        break;
                    }
                    case EOp::Shl:
                    case EOp::LShr:
                    case EOp::AShr: {
                        if (is_cb && cb == 0) { apply_merge(cid, n.a); break; }
                        if (is_ca && ca == 0) { apply_merge(cid, make_const_checked(0, w)); break; }
                        if (is_cb && cb < w) {
                            for (const ENode& ma : class_nodes(n.a)) {
                                uint64_t c1 = 0;
                                if (ma.op == n.op && is_const(ma.b, c1) && c1 < w && (c1 + cb) < w) {
                                    apply_merge(cid, make_bin_checked(n.op, w, ma.a, make_const_checked(c1 + cb, w)));
                                }
                            }
                        }
                        break;
                    }
                    case EOp::Eq: case EOp::Sle: case EOp::Ule: if (n.a == n.b) apply_merge(cid, make_const_checked(1, w)); break;
                    case EOp::Ne: case EOp::Slt: case EOp::Ult: if (n.a == n.b) apply_merge(cid, make_const_checked(0, w)); break;
                    case EOp::Select:
                        if (n.b == n.c) apply_merge(cid, n.b);
                        else if (is_ca) apply_merge(cid, (mask_w(ca, class_width(n.a)) != 0) ? n.b : n.c);
                        break;
                    default: break;
                }
            }
        }

        // Phase 2: Bridge / Expansion Rewrites
        for (const auto& sc : snap) {
            if (budget_exceeded) break;
            const EClassId cid = find(sc.cid);
            for (const ENode& raw_n : sc.nodes) {
                if (budget_exceeded) break;
                const ENode n = canonicalize(raw_n);
                const uint32_t w = n.width;
                switch (n.op) {
                    case EOp::Neg:
                        apply_merge(cid, make_bin_checked(EOp::Add, w, make_unary_checked(EOp::Not, w, n.a), make_const_checked(1, w)));
                        break;
                    case EOp::Not:
                        for (const ENode& ma : class_nodes(n.a)) {
                            if (ma.op == EOp::And || ma.op == EOp::Or) {
                                const EOp dual = (ma.op == EOp::And) ? EOp::Or : EOp::And;
                                apply_merge(cid, make_bin_checked(dual, w, make_unary_checked(EOp::Not, w, ma.a), make_unary_checked(EOp::Not, w, ma.b)));
                            }
                        }
                        break;
                    case EOp::Sub:
                        apply_merge(cid, make_bin_checked(EOp::Add, w, n.a, make_unary_checked(EOp::Neg, w, n.b)));
                        break;
                    case EOp::Shl: {
                        uint64_t sh = 0;
                        if (is_const(n.b, sh) && sh > 0 && sh < w) apply_merge(cid, make_bin_checked(EOp::Mul, w, n.a, make_const_checked(1ULL << sh, w)));
                        break;
                    }
                    case EOp::And: {
                        auto try_dist = [&](EClassId x_cls, EClassId yz_cls) {
                            for (const ENode& myz : class_nodes(yz_cls)) {
                                if (myz.op == EOp::Or || myz.op == EOp::Xor) {
                                    apply_merge(cid, make_bin_checked(myz.op, w, make_bin_checked(EOp::And, w, x_cls, myz.a),
                                                                                  make_bin_checked(EOp::And, w, x_cls, myz.b)));
                                }
                            }
                        };
                        try_dist(n.a, n.b);
                        try_dist(n.b, n.a);
                        [[fallthrough]];
                    }
                    case EOp::Add: case EOp::Mul: case EOp::Or: case EOp::Xor: {
                        auto reassociate = [&](EClassId ch, EClassId z_cls) {
                            for (const ENode& mc : class_nodes(ch)) {
                                if (mc.op == n.op && mc.a != mc.b) {
                                    apply_merge(cid, make_bin_checked(n.op, w, mc.a, make_bin_checked(n.op, w, mc.b, z_cls)));
                                    apply_merge(cid, make_bin_checked(n.op, w, mc.b, make_bin_checked(n.op, w, mc.a, z_cls)));
                                }
                            }
                        };
                        reassociate(n.a, n.b);
                        reassociate(n.b, n.a);
                        break;
                    }
                    default: break;
                }
            }
        }
        rebuild();
        if (iter_rewrites == 0 || budget_exceeded) break;
    }
    return total_rewrites;
}

} // namespace brass
