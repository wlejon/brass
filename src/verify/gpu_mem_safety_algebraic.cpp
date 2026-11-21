#include "gpu_mem_safety_internal.hpp"
#include <algorithm>
#include <bit>
#include <cstdint>
#include <sstream>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace brass::verify {

namespace {

bool is_disjoint_add_like_bv(const BVContext& ctx, const BVNode& n) noexcept {
    if (n.op == BVOp::Add) return true;
    if (n.op == BVOp::Or) {
        const uint64_t m = width_mask(n.width);
        return ((ctx.interval(n.a).known_zeros | ctx.interval(n.b).known_zeros) & m) == m;
    }
    return false;
}


struct Monomial {
    uint64_t const_mul = 1;
    std::vector<BVNodeId> sym_factors;
};

void flatten_monomial(const BVContext& ctx, BVNodeId node, Monomial& out, int depth = 0) {
    if (depth > 16) {
        out.sym_factors.push_back(node);
        return;
    }
    const BVNode& n = bv_node(ctx, node);
    if (n.op == BVOp::Const) {
        out.const_mul *= n.imm;
        return;
    }
    if (n.op == BVOp::Mul) {
        flatten_monomial(ctx, n.a, out, depth + 1);
        flatten_monomial(ctx, n.b, out, depth + 1);
        return;
    }
    if (n.op == BVOp::Shl && bv_node(ctx, n.b).op == BVOp::Const && bv_node(ctx, n.b).imm < 30) {
        flatten_monomial(ctx, n.a, out, depth + 1);
        out.const_mul <<= bv_node(ctx, n.b).imm;
        return;
    }
    out.sym_factors.push_back(node);
}

BVNodeId build_product_32(BVContext& ctx, const Monomial& m) {
    if (m.sym_factors.empty()) {
        return bv_const(ctx, 32, m.const_mul);
    }
    BVNodeId acc = m.sym_factors[0];
    for (size_t i = 1; i < m.sym_factors.size(); ++i) {
        acc = bv_bin(ctx, BVOp::Mul, 32, acc, m.sym_factors[i]);
    }
    if (m.const_mul != 1) {
        acc = bv_bin(ctx, BVOp::Mul, 32, acc, bv_const(ctx, 32, m.const_mul));
    }
    return acc;
}

bool match_monomial_div(
    BVContext& ctx,
    BVNodeId term,
    const std::vector<BVNodeId>& div_factors,
    BVNodeId& out_q) {
    if (div_factors.empty()) return false;
    Monomial tm;
    flatten_monomial(ctx, term, tm);
    if (tm.const_mul == 0) return false;

    uint64_t need_const = 1;
    std::vector<BVNodeId> need_syms;
    for (BVNodeId f : div_factors) {
        Monomial fm;
        flatten_monomial(ctx, f, fm);
        if (fm.const_mul == 0) return false;
        if (need_const > 0x100000000ULL / std::max<uint64_t>(1, fm.const_mul)) return false;
        need_const *= fm.const_mul;
        need_syms.insert(need_syms.end(), fm.sym_factors.begin(), fm.sym_factors.end());
    }
    if (need_const == 0 || (tm.const_mul % need_const) != 0) return false;

    std::vector<BVNodeId> rem_syms = tm.sym_factors;
    for (BVNodeId nf : need_syms) {
        auto it = std::find(rem_syms.begin(), rem_syms.end(), nf);
        if (it == rem_syms.end()) return false;
        rem_syms.erase(it);
    }
    Monomial rem;
    rem.const_mul = tm.const_mul / need_const;
    rem.sym_factors = std::move(rem_syms);
    out_q = build_product_32(ctx, rem);
    return true;
}

void flatten_add32_into(const BVContext& ctx, BVNodeId expr32, std::vector<BVNodeId>& out) {
    const BVNode& n = bv_node(ctx, expr32);
    if (is_disjoint_add_like_bv(ctx, n)) {
        flatten_add32_into(ctx, n.a, out);
        flatten_add32_into(ctx, n.b, out);
        return;
    }
    if (n.op == BVOp::Const && n.imm == 0) {
        return;
    }
    out.push_back(expr32);
}

BVNodeId sum_terms_32(BVContext& ctx, const std::vector<BVNodeId>& terms) {
    if (terms.empty()) return bv_const(ctx, 32, 0);
    BVNodeId acc = terms[0];
    for (size_t i = 1; i < terms.size(); ++i) {
        acc = bv_bin(ctx, BVOp::Add, 32, acc, terms[i]);
    }
    return acc;
}

BVNodeId sum_terms_64(BVContext& ctx, const std::vector<BVNodeId>& terms) {
    if (terms.empty()) return bv_const(ctx, 64, 0);
    BVNodeId acc = bv_ext(ctx, false, 64, terms[0]);
    for (size_t i = 1; i < terms.size(); ++i) {
        acc = bv_bin(ctx, BVOp::Add, 64, acc, bv_ext(ctx, false, 64, terms[i]));
    }
    return acc;
}

bool peel_single_64_to_32(
    BVContext& ctx,
    BVNodeId node64,
    uint32_t elem_bytes,
    BVNodeId& out_e32,
    bool& out_saw_sext) {
    const BVNode& n = bv_node(ctx, node64);
    if (n.op == BVOp::Const) {
        if (n.imm == 0) {
            out_e32 = bv_const(ctx, 32, 0);
            return true;
        }
        if (elem_bytes > 0 && n.imm % elem_bytes == 0 && (n.imm / elem_bytes) < 0x40000000ULL) {
            out_e32 = bv_const(ctx, 32, n.imm / elem_bytes);
            return true;
        }
        return false;
    }

    auto unwrap_ext32 = [&](BVNodeId id, BVNodeId& out_id) -> bool {
        const BVNode& z = bv_node(ctx, id);
        if ((z.op == BVOp::ZExt || z.op == BVOp::SExt) && ctx.width(z.a) == 32) {
            out_id = z.a;
            if (z.op == BVOp::SExt) out_saw_sext = true;
            return true;
        }
        return false;
    };

    if (elem_bytes == 1 && unwrap_ext32(node64, out_e32)) {
        return true;
    }

    if (n.op == BVOp::Shl && bv_node(ctx, n.b).op == BVOp::Const) {
        uint64_t sh = bv_node(ctx, n.b).imm;
        if (sh < 31 && (1ULL << sh) == elem_bytes && unwrap_ext32(n.a, out_e32)) {
            return true;
        }
    }

    if (n.op == BVOp::Mul) {
        const BVNode& na = bv_node(ctx, n.a);
        const BVNode& nb = bv_node(ctx, n.b);
        if (nb.op == BVOp::Const && nb.imm == elem_bytes && unwrap_ext32(n.a, out_e32)) {
            return true;
        }
        if (na.op == BVOp::Const && na.imm == elem_bytes && unwrap_ext32(n.b, out_e32)) {
            return true;
        }
    }

    return false;
}

bool peel_byte_offset_64(
    BVContext& ctx,
    BVNodeId node64,
    uint32_t elem_bytes,
    std::vector<BVNodeId>& out_terms32,
    BVNodeId& out_single32,
    bool& out_from_add64,
    bool& out_saw_sext,
    int depth = 0) {
    if (depth > 12) return false;
    BVNodeId e32 = 0;
    if (peel_single_64_to_32(ctx, node64, elem_bytes, e32, out_saw_sext)) {
        flatten_add32_into(ctx, e32, out_terms32);
        if (out_single32 == 0) {
            out_single32 = e32;
        } else {
            out_from_add64 = true;
            out_single32 = 0;
        }
        return true;
    }
    const BVNode& n = bv_node(ctx, node64);
    if (is_disjoint_add_like_bv(ctx, n)) {
        std::vector<BVNodeId> lhs_terms, rhs_terms;
        BVNodeId lhs_single = 0, rhs_single = 0;
        bool lhs_add64 = false, rhs_add64 = false;
        if (!peel_byte_offset_64(ctx, n.a, elem_bytes, lhs_terms, lhs_single, lhs_add64, out_saw_sext, depth + 1) ||
            !peel_byte_offset_64(ctx, n.b, elem_bytes, rhs_terms, rhs_single, rhs_add64, out_saw_sext, depth + 1)) {
            return false;
        }
        out_terms32.insert(out_terms32.end(), lhs_terms.begin(), lhs_terms.end());
        out_terms32.insert(out_terms32.end(), rhs_terms.begin(), rhs_terms.end());
        if (lhs_terms.empty() && !rhs_add64 && rhs_single != 0) {
            out_single32 = rhs_single;
            return true;
        }
        if (rhs_terms.empty() && !lhs_add64 && lhs_single != 0) {
            out_single32 = lhs_single;
            return true;
        }
        out_from_add64 = true;
        out_single32 = 0;
        return true;
    }
    return false;
}

struct AlgebraicCtx {
    BVNodeId light_domain = 0;
    BVNodeId guard_domain = 0;
    BVNodeId active = 0;
    const std::unordered_map<BVNodeId, uint64_t>* umin_map = nullptr;
    const std::unordered_map<BVNodeId, uint64_t>* umax_map = nullptr;
    const std::vector<DivConstraint>* div_pairs = nullptr;
    uint32_t max_conflicts = 4000;
};

bool solve_unsat_staged(
    BVContext& ctx,
    const AlgebraicCtx& ac,
    BVNodeId bad_pred) {
    if (bv_node(ctx, bad_pred).op == BVOp::Const && bv_node(ctx, bad_pred).imm == 0) {
        return true;
    }
    SolveOutcome o = solve_nonzero(
        ctx, bv_bin(ctx, BVOp::And, 1, ac.light_domain, bad_pred), ac.max_conflicts);
    if (o.status == SolveOutcome::Unsat) return true;
    if (o.status == SolveOutcome::Sat && ac.guard_domain != ac.light_domain) {
        o = solve_nonzero(
            ctx, bv_bin(ctx, BVOp::And, 1, ac.guard_domain, bad_pred), ac.max_conflicts);
        if (o.status == SolveOutcome::Unsat) return true;
    }
    if (o.status == SolveOutcome::Sat && ac.active != ac.guard_domain) {
        o = solve_nonzero(
            ctx, bv_bin(ctx, BVOp::And, 1, ac.active, bad_pred), ac.max_conflicts);
        if (o.status == SolveOutcome::Unsat) return true;
    }
    return false;
}

void collect_active_bounds(
    const BVContext& ctx,
    BVNodeId root,
    std::unordered_map<BVNodeId, uint64_t>& umin_map,
    std::unordered_map<BVNodeId, uint64_t>& umax_map) {
    auto record_ub = [&](BVNodeId v, uint64_t ub) {
        auto it = umax_map.find(v);
        if (it == umax_map.end() || ub < it->second) umax_map[v] = ub;
    };
    auto record_lb = [&](BVNodeId v, uint64_t lb) {
        auto it = umin_map.find(v);
        if (it == umin_map.end() || lb > it->second) umin_map[v] = lb;
    };

    std::vector<BVNodeId> work = {root};
    std::unordered_set<BVNodeId> seen;
    while (!work.empty() && seen.size() < 2048) {
        BVNodeId node = work.back();
        work.pop_back();
        if (node >= ctx.nodes().size() || !seen.insert(node).second) continue;
        const BVNode& n = bv_node(ctx, node);
        if (n.op == BVOp::And) {
            work.push_back(n.a);
            work.push_back(n.b);
            continue;
        }
        if (n.op == BVOp::Ult) {
            if (n.b < ctx.nodes().size() && bv_node(ctx, n.b).op == BVOp::Const &&
                bv_node(ctx, n.b).imm > 0) {
                record_ub(n.a, bv_node(ctx, n.b).imm - 1);
            }
            if (n.a < ctx.nodes().size() && bv_node(ctx, n.a).op == BVOp::Const) {
                uint64_t c = bv_node(ctx, n.a).imm;
                if (c < UINT64_MAX) record_lb(n.b, c + 1);
            }
        } else if (n.op == BVOp::Ule && n.b < ctx.nodes().size() &&
                   bv_node(ctx, n.b).op == BVOp::Const) {
            record_ub(n.a, bv_node(ctx, n.b).imm);
        } else if (n.op == BVOp::Uge && n.b < ctx.nodes().size() &&
                   bv_node(ctx, n.b).op == BVOp::Const) {
            record_lb(n.a, bv_node(ctx, n.b).imm);
        } else if (n.op == BVOp::Ugt && n.b < ctx.nodes().size() &&
                   bv_node(ctx, n.b).op == BVOp::Const) {
            uint64_t c = bv_node(ctx, n.b).imm;
            if (c < UINT64_MAX) record_lb(n.a, c + 1);
        } else if (n.op == BVOp::Eq) {
            if (n.b < ctx.nodes().size() && bv_node(ctx, n.b).op == BVOp::Const) {
                uint64_t c = bv_node(ctx, n.b).imm;
                record_ub(n.a, c);
                record_lb(n.a, c);
            } else if (n.a < ctx.nodes().size() && bv_node(ctx, n.a).op == BVOp::Const) {
                uint64_t c = bv_node(ctx, n.a).imm;
                record_ub(n.b, c);
                record_lb(n.b, c);
            }
        } else if (n.op == BVOp::Not && n.a < ctx.nodes().size()) {
            const BVNode& in = bv_node(ctx, n.a);
            if (in.op == BVOp::Ult) {
                // Canonical Ule(v, C) is Not(Ult(C, v)); canonical Uge(v, C) is Not(Ult(v, C))
                if (in.a < ctx.nodes().size() && bv_node(ctx, in.a).op == BVOp::Const) {
                    record_ub(in.b, bv_node(ctx, in.a).imm);
                }
                if (in.b < ctx.nodes().size() && bv_node(ctx, in.b).op == BVOp::Const) {
                    record_lb(in.a, bv_node(ctx, in.b).imm);
                }
            } else if (in.op == BVOp::Eq) {
                if (in.b < ctx.nodes().size() && bv_node(ctx, in.b).op == BVOp::Const &&
                    bv_node(ctx, in.b).imm == 0) {
                    record_lb(in.a, 1);
                } else if (in.a < ctx.nodes().size() && bv_node(ctx, in.a).op == BVOp::Const &&
                           bv_node(ctx, in.a).imm == 0) {
                    record_lb(in.b, 1);
                }
            }
        }
    }
}

uint64_t eval_umax_32(
    const BVContext& ctx,
    BVNodeId node,
    const std::unordered_map<BVNodeId, uint64_t>& umax_map,
    int depth = 0) {
    if (depth > 8 || node >= ctx.nodes().size()) return 0xFFFFFFFFULL;
    uint64_t ub = ctx.interval(node).umax & 0xFFFFFFFFULL;
    if (auto it = umax_map.find(node); it != umax_map.end()) {
        ub = std::min<uint64_t>(ub, it->second & 0xFFFFFFFFULL);
    }
    const BVNode& n = bv_node(ctx, node);
    if (n.op == BVOp::Const) return n.imm & 0xFFFFFFFFULL;
    if ((n.op == BVOp::ZExt || n.op == BVOp::Extract) && ctx.width(n.a) >= 32) {
        uint64_t sub = eval_umax_32(ctx, n.a, umax_map, depth + 1);
        if (n.op == BVOp::Extract && n.imm >= n.extra && (n.imm - n.extra + 1) < 32) {
            uint64_t wbits = n.imm - n.extra + 1;
            uint64_t mask = (1ULL << wbits) - 1ULL;
            if (n.extra == 0) return std::min<uint64_t>({ub, sub, mask});
            if (ctx.width(n.a) <= 32) return std::min<uint64_t>({ub, sub >> n.extra, mask});
            return std::min<uint64_t>(ub, mask);
        }
        return std::min<uint64_t>(ub, sub);
    }
    if (n.op == BVOp::LShr && n.b < ctx.nodes().size() &&
        bv_node(ctx, n.b).op == BVOp::Const && bv_node(ctx, n.b).imm < 32) {
        return std::min<uint64_t>(ub, eval_umax_32(ctx, n.a, umax_map, depth + 1) >> bv_node(ctx, n.b).imm);
    }
    if (n.op == BVOp::UDiv && n.b < ctx.nodes().size() && bv_node(ctx, n.b).op == BVOp::Const &&
        bv_node(ctx, n.b).imm > 0) {
        return std::min<uint64_t>(ub, eval_umax_32(ctx, n.a, umax_map, depth + 1) / bv_node(ctx, n.b).imm);
    }
    if (n.op == BVOp::And && n.b < ctx.nodes().size() && bv_node(ctx, n.b).op == BVOp::Const) {
        return std::min<uint64_t>({ub, eval_umax_32(ctx, n.a, umax_map, depth + 1), bv_node(ctx, n.b).imm});
    }
    if (n.op == BVOp::Mul) {
        uint64_t a = eval_umax_32(ctx, n.a, umax_map, depth + 1);
        uint64_t b = eval_umax_32(ctx, n.b, umax_map, depth + 1);
        if (b == 0) return 0ULL;
        if (a <= 0xFFFFFFFFULL / b) return std::min<uint64_t>(ub, a * b);
    }
    return ub;
}

uint64_t eval_umin_32(
    const BVContext& ctx,
    BVNodeId node,
    const std::unordered_map<BVNodeId, uint64_t>& umin_map,
    const std::unordered_map<BVNodeId, uint64_t>& umax_map,
    int depth = 0) {
    if (depth > 8 || node >= ctx.nodes().size()) return 0ULL;
    uint64_t lb = ctx.interval(node).umin & 0xFFFFFFFFULL;
    if (auto it = umin_map.find(node); it != umin_map.end()) {
        lb = std::max<uint64_t>(lb, it->second & 0xFFFFFFFFULL);
    }
    const BVNode& n = bv_node(ctx, node);
    if (n.op == BVOp::Const) return n.imm & 0xFFFFFFFFULL;
    if (n.op == BVOp::ZExt && ctx.width(n.a) <= 32) {
        lb = std::max(lb, eval_umin_32(ctx, n.a, umin_map, umax_map, depth + 1));
    } else if (n.op == BVOp::LShr && n.b < ctx.nodes().size() &&
               bv_node(ctx, n.b).op == BVOp::Const && bv_node(ctx, n.b).imm < 32) {
        uint64_t sub = eval_umin_32(ctx, n.a, umin_map, umax_map, depth + 1);
        lb = std::max(lb, sub >> bv_node(ctx, n.b).imm);
    } else if (n.op == BVOp::UDiv && n.b < ctx.nodes().size() &&
               bv_node(ctx, n.b).op == BVOp::Const && bv_node(ctx, n.b).imm > 0) {
        uint64_t sub = eval_umin_32(ctx, n.a, umin_map, umax_map, depth + 1);
        lb = std::max(lb, sub / bv_node(ctx, n.b).imm);
    } else if (n.op == BVOp::Add && ctx.width(node) <= 32) {
        uint64_t ua = eval_umax_32(ctx, n.a, umax_map, depth + 1);
        uint64_t ub = eval_umax_32(ctx, n.b, umax_map, depth + 1);
        if (ua + ub <= 0xFFFFFFFFULL) {
            uint64_t la = eval_umin_32(ctx, n.a, umin_map, umax_map, depth + 1);
            uint64_t lb2 = eval_umin_32(ctx, n.b, umin_map, umax_map, depth + 1);
            lb = std::max(lb, la + lb2);
        }
    } else if (n.op == BVOp::Mul && ctx.width(node) <= 32) {
        uint64_t ua = eval_umax_32(ctx, n.a, umax_map, depth + 1);
        uint64_t ub = eval_umax_32(ctx, n.b, umax_map, depth + 1);
        if (ua == 0 || ub <= 0xFFFFFFFFULL / ua) {
            uint64_t la = eval_umin_32(ctx, n.a, umin_map, umax_map, depth + 1);
            uint64_t lb2 = eval_umin_32(ctx, n.b, umin_map, umax_map, depth + 1);
            lb = std::max(lb, la * lb2);
        }
    }
    return lb;
}

bool is_proven_nonzero_32(BVContext& ctx, const AlgebraicCtx& ac, BVNodeId f) {
    if (ac.umin_map && ac.umax_map &&
        eval_umin_32(ctx, f, *ac.umin_map, *ac.umax_map) > 0) {
        return true;
    }
    BVNodeId eq0 = bv_bin(ctx, BVOp::Eq, 1, f, bv_const(ctx, 32, 0));
    return solve_unsat_staged(ctx, ac, eq0);
}

bool prove_all_nonzero(BVContext& ctx, const AlgebraicCtx& ac, const std::vector<BVNodeId>& factors) {
    for (BVNodeId f : factors) {
        if (!is_proven_nonzero_32(ctx, ac, f)) return false;
    }
    return true;
}

bool match_factors_multiset(
    BVContext& ctx,
    BVNodeId rhs,
    const std::vector<BVNodeId>& factors) {
    Monomial rm;
    flatten_monomial(ctx, rhs, rm);
    Monomial fm;
    for (BVNodeId f : factors) flatten_monomial(ctx, f, fm);
    if (rm.const_mul != fm.const_mul || rm.sym_factors.size() != fm.sym_factors.size()) return false;
    std::vector<BVNodeId> rem = rm.sym_factors;
    for (BVNodeId f : fm.sym_factors) {
        auto it = std::find(rem.begin(), rem.end(), f);
        if (it == rem.end()) return false;
        rem.erase(it);
    }
    return true;
}

bool guard_implies_expr_lt_factors(
    BVContext& ctx,
    const AlgebraicCtx& ac,
    BVNodeId expr32,
    const std::vector<BVNodeId>& factors) {
    std::unordered_map<BVNodeId, uint64_t> local_ub;
    const auto& umax_map = ac.umax_map ? *ac.umax_map : local_ub;
    uint64_t prod_ub = 1;
    for (BVNodeId f : factors) {
        uint64_t fub = eval_umax_32(ctx, f, umax_map);
        if (fub > 0 && prod_ub > 0xFFFFFFFFULL / fub) return false;
        prod_ub *= std::max<uint64_t>(1, fub);
    }

    std::vector<BVNodeId> work = {ac.guard_domain};
    std::unordered_set<BVNodeId> seen;
    while (!work.empty() && seen.size() < 1024) {
        BVNodeId cur = work.back();
        work.pop_back();
        if (cur >= ctx.nodes().size() || !seen.insert(cur).second) continue;
        const BVNode& n = bv_node(ctx, cur);
        if (n.op == BVOp::And) {
            work.push_back(n.a);
            work.push_back(n.b);
        } else if ((n.op == BVOp::Ult && n.a == expr32 && match_factors_multiset(ctx, n.b, factors)) ||
                   (n.op == BVOp::Ugt && n.b == expr32 && match_factors_multiset(ctx, n.a, factors))) {
            return true;
        } else if (n.op == BVOp::Not && n.a < ctx.nodes().size()) {
            const BVNode& in = bv_node(ctx, n.a);
            if ((in.op == BVOp::Uge && in.a == expr32 && match_factors_multiset(ctx, in.b, factors)) ||
                (in.op == BVOp::Ule && in.b == expr32 && match_factors_multiset(ctx, in.a, factors))) {
                return true;
            }
        }
    }
    return false;
}

BVNodeId build_factors_prod_64(BVContext& ctx, const std::vector<BVNodeId>& factors) {
    if (factors.empty()) return bv_const(ctx, 64, 1);
    BVNodeId acc = bv_ext(ctx, false, 64, factors[0]);
    for (size_t i = 1; i < factors.size(); ++i) {
        acc = gpu_mul_or_shl(ctx, 64, acc, bv_ext(ctx, false, 64, factors[i]));
    }
    return acc;
}

bool contains_div_or_rem(const BVContext& ctx, BVNodeId node, int depth = 0) {
    if (depth > 8 || node >= ctx.nodes().size()) return false;
    const BVNode& n = bv_node(ctx, node);
    if (n.op == BVOp::UDiv || n.op == BVOp::URem || n.op == BVOp::SDiv || n.op == BVOp::SRem) {
        return true;
    }
    if (contains_div_or_rem(ctx, n.a, depth + 1)) return true;
    if (n.op == BVOp::ZExt || n.op == BVOp::SExt || n.op == BVOp::Extract ||
        n.op == BVOp::Not || n.op == BVOp::Neg) {
        return false;
    }
    if (contains_div_or_rem(ctx, n.b, depth + 1)) return true;
    return n.op == BVOp::Ite && contains_div_or_rem(ctx, n.c, depth + 1);
}

bool prove_via_rem_ub(
    BVContext& ctx,
    const AlgebraicCtx& ac,
    const std::vector<BVNodeId>& terms,
    const std::vector<BVNodeId>& factors) {
    bool saw_urem = false;
    std::vector<BVNodeId> ub_terms64;
    ub_terms64.reserve(terms.size());
    for (BVNodeId t : terms) {
        Monomial tm;
        flatten_monomial(ctx, t, tm);
        BVNodeId rem_denom = 0;
        std::vector<BVNodeId> other_syms;
        bool bad_mult_rem = false;
        for (BVNodeId sf : tm.sym_factors) {
            const BVNode& sn = bv_node(ctx, sf);
            if (sn.op == BVOp::URem) {
                if (rem_denom != 0) { bad_mult_rem = true; break; }
                rem_denom = sn.b;
            } else if (sn.op == BVOp::Ite && sn.a < ctx.nodes().size() && sn.c < ctx.nodes().size() &&
                       bv_node(ctx, sn.a).op == BVOp::Eq && bv_node(ctx, sn.c).op == BVOp::URem) {
                const BVNode& eqn = bv_node(ctx, sn.a);
                const BVNode& rn = bv_node(ctx, sn.c);
                const bool zero_g =
                    (eqn.a == rn.b && bv_node(ctx, eqn.b).op == BVOp::Const && bv_node(ctx, eqn.b).imm == 0) ||
                    (eqn.b == rn.b && bv_node(ctx, eqn.a).op == BVOp::Const && bv_node(ctx, eqn.a).imm == 0);
                if (!zero_g) {
                    other_syms.push_back(sf);
                } else {
                    if (rem_denom != 0) { bad_mult_rem = true; break; }
                    rem_denom = rn.b;
                }
            } else {
                other_syms.push_back(sf);
            }
        }
        if (bad_mult_rem) return false;
        if (rem_denom == 0) {
            if (contains_div_or_rem(ctx, t)) return false;
            ub_terms64.push_back(bv_ext(ctx, false, 64, t));
        } else {
            saw_urem = true;
            if (!is_proven_nonzero_32(ctx, ac, rem_denom)) return false;
            BVNodeId d_minus_1 = bv_bin(ctx, BVOp::Sub, 64, bv_ext(ctx, false, 64, rem_denom), bv_const(ctx, 64, 1));
            BVNodeId acc = gpu_mul_or_shl(ctx, 64, d_minus_1, bv_const(ctx, 64, tm.const_mul));
            for (BVNodeId os : other_syms) {
                acc = bv_bin(ctx, BVOp::Mul, 64, acc, bv_ext(ctx, false, 64, os));
            }
            ub_terms64.push_back(acc);
        }
    }
    if (!saw_urem) return false;
    BVNodeId sum64 = ub_terms64.empty() ? bv_const(ctx, 64, 0) : ub_terms64[0];
    for (size_t i = 1; i < ub_terms64.size(); ++i) sum64 = bv_bin(ctx, BVOp::Add, 64, sum64, ub_terms64[i]);
    BVNodeId prod64 = build_factors_prod_64(ctx, factors);
    return solve_unsat_staged(ctx, ac, bv_bin(ctx, BVOp::Uge, 1, sum64, prod64));
}

bool try_prove_terms_lt_product(
    BVContext& ctx,
    const AlgebraicCtx& ac,
    const std::vector<BVNodeId>& terms,
    BVNodeId single_e32,
    bool from_add64,
    bool saw_sext,
    const std::vector<BVNodeId>& factors,
    int depth) {
    if (depth > 7 || factors.empty()) return false;

    if (terms.empty()) {
        return prove_all_nonzero(ctx, ac, factors);
    }

    if (terms.size() == 1 && bv_node(ctx, terms[0]).op == BVOp::Ite) {
        const BVNode in = bv_node(ctx, terms[0]);
        auto check_arm = [&](BVNodeId arm_cond, BVNodeId arm_val) -> bool {
            if (solve_unsat_staged(ctx, ac, arm_cond)) return true; // arm unreachable
            std::unordered_map<BVNodeId, uint64_t> sub_umin =
                ac.umin_map ? *ac.umin_map : std::unordered_map<BVNodeId, uint64_t>{};
            std::unordered_map<BVNodeId, uint64_t> sub_umax =
                ac.umax_map ? *ac.umax_map : std::unordered_map<BVNodeId, uint64_t>{};
            collect_active_bounds(ctx, arm_cond, sub_umin, sub_umax);
            AlgebraicCtx sub_ac = ac;
            sub_ac.guard_domain = bv_bin(ctx, BVOp::And, 1, ac.guard_domain, arm_cond);
            sub_ac.active = bv_bin(ctx, BVOp::And, 1, ac.active, arm_cond);
            sub_ac.umin_map = &sub_umin;
            sub_ac.umax_map = &sub_umax;
            std::vector<BVNodeId> arm_terms;
            flatten_add32_into(ctx, arm_val, arm_terms);
            return try_prove_terms_lt_product(
                ctx, sub_ac, arm_terms, arm_val, false, saw_sext, factors, depth + 1);
        };
        if (check_arm(in.a, in.b) && check_arm(bv_un(ctx, BVOp::Not, 1, in.a), in.c)) {
            return true;
        }
    }

    if (factors.size() == 1) {
        BVNodeId D = factors[0];
        if (from_add64 && terms.size() > 1) {
            if (saw_sext) return false;
            BVNodeId sum64 = sum_terms_64(ctx, terms);
            BVNodeId d64 = bv_ext(ctx, false, 64, D);
            return solve_unsat_staged(ctx, ac, bv_bin(ctx, BVOp::Uge, 1, sum64, d64));
        }
        BVNodeId expr = (terms.size() == 1) ? terms[0] : (single_e32 ? single_e32 : sum_terms_32(ctx, terms));
        const BVNode& en = bv_node(ctx, expr);
        if (en.op == BVOp::URem) {
            BVNodeId denom = en.b;
            if (denom == D && is_proven_nonzero_32(ctx, ac, denom)) {
                return true;
            }
            BVNodeId bad_mod = bv_bin(ctx, BVOp::Or, 1,
                bv_bin(ctx, BVOp::Eq, 1, denom, bv_const(ctx, 32, 0)),
                bv_bin(ctx, BVOp::Ugt, 1, denom, D));
            if (solve_unsat_staged(ctx, ac, bad_mod)) {
                return true;
            }
        }
        if (prove_via_rem_ub(ctx, ac, terms, factors)) return true;
        if (en.op != BVOp::UDiv && en.op != BVOp::URem && !contains_div_or_rem(ctx, expr)) {
            if (ac.umax_map && ac.umin_map &&
                eval_umax_32(ctx, expr, *ac.umax_map) < eval_umin_32(ctx, D, *ac.umin_map, *ac.umax_map)) {
                return true;
            }
            if (solve_unsat_staged(ctx, ac, bv_bin(ctx, BVOp::Uge, 1, expr, D))) return true;
        }
    }

    // Fast Euclidean UDiv reduction: (x / denom) < prod(F) <==> x < denom * prod(F) (for denom > 0)
    if (terms.size() == 1) {
        const BVNode& dn = bv_node(ctx, terms[0]);
        const bool is_udiv = (dn.op == BVOp::UDiv);
        const bool is_lshr_const = (dn.op == BVOp::LShr && dn.b < ctx.nodes().size() &&
            bv_node(ctx, dn.b).op == BVOp::Const && bv_node(ctx, dn.b).imm > 0 && bv_node(ctx, dn.b).imm < 31);
        if (is_udiv || is_lshr_const) {
            BVNodeId x = dn.a;
            uint32_t sh = is_lshr_const ? static_cast<uint32_t>(bv_node(ctx, dn.b).imm) : 0u;
            BVNodeId denom = is_udiv ? dn.b : bv_const(ctx, 32, 1ULL << sh);
            if (is_lshr_const || is_proven_nonzero_32(ctx, ac, denom)) {
                std::vector<BVNodeId> next_factors = factors;
                bool combined = false;
                for (size_t i = 0; i < next_factors.size(); ++i) {
                    const BVNode& fn = bv_node(ctx, next_factors[i]);
                    const bool match_udiv = (fn.op == BVOp::UDiv && fn.b == denom);
                    const bool match_lshr = (is_lshr_const && fn.op == BVOp::LShr && fn.b < ctx.nodes().size() &&
                        bv_node(ctx, fn.b).op == BVOp::Const && bv_node(ctx, fn.b).imm == sh);
                    if ((match_udiv || match_lshr) && ac.div_pairs) {
                        for (const auto& dc : *ac.div_pairs) {
                            if (dc.dividend == fn.a && dc.divisor == denom) {
                                next_factors[i] = dc.dividend;
                                combined = true;
                                break;
                            }
                        }
                        if (combined) break;
                    }
                }
                if (!combined) next_factors.push_back(denom);
                std::vector<BVNodeId> x_terms;
                flatten_add32_into(ctx, x, x_terms);
                if (try_prove_terms_lt_product(
                        ctx, ac, x_terms, x, false, false, next_factors, depth + 1)) {
                    return true;
                }
            }
        }
    }

    BVNodeId check_expr = (terms.size() == 1) ? terms[0] : single_e32;
    if (factors.size() >= 2 && check_expr != 0 &&
        guard_implies_expr_lt_factors(ctx, ac, check_expr, factors)) {
        return true;
    }

    // Inductive case: match any subset of factors F_lo
    auto try_partition = [&](const std::vector<BVNodeId>& F_lo, const std::vector<BVNodeId>& F_hi) -> bool {
        if (F_lo.empty() || F_hi.empty()) return false;
        std::vector<BVNodeId> q_parts;
        std::vector<BVNodeId> rem_terms;
        for (BVNodeId t : terms) {
            BVNodeId q = 0;
            if (match_monomial_div(ctx, t, F_lo, q)) {
                flatten_add32_into(ctx, q, q_parts);
            } else {
                rem_terms.push_back(t);
            }
        }
        if (q_parts.empty()) return false;
        if (from_add64 && q_parts.size() > 1) return false;
        if (!try_prove_terms_lt_product(
                ctx, ac, q_parts, 0, false, false, F_hi, depth + 1)) {
            return false;
        }
        return try_prove_terms_lt_product(
            ctx, ac, rem_terms, 0, from_add64, saw_sext, F_lo, depth + 1);
    };

    if (factors.size() >= 3) {
        for (size_t i = 0; i < factors.size(); ++i) {
            for (size_t j = i + 1; j < factors.size(); ++j) {
                std::vector<BVNodeId> F_lo = {factors[i], factors[j]};
                std::vector<BVNodeId> F_hi;
                for (size_t k = 0; k < factors.size(); ++k) {
                    if (k != i && k != j) F_hi.push_back(factors[k]);
                }
                if (try_partition(F_lo, F_hi)) return true;
            }
        }
    }

    for (size_t i = 0; i < factors.size(); ++i) {
        std::vector<BVNodeId> F_lo = {factors[i]};
        std::vector<BVNodeId> F_hi;
        for (size_t k = 0; k < factors.size(); ++k) {
            if (k != i) F_hi.push_back(factors[k]);
        }
        if (try_partition(F_lo, F_hi)) return true;
    }

    if (factors.size() >= 2 && prove_via_rem_ub(ctx, ac, terms, factors)) {
        return true;
    }

    // Fallback when leading index is 0 (e.g. inner loop r = 0 simplifies r * cols + c -> c < rows * cols)
    for (size_t i = 0; i < factors.size(); ++i) {
        if (!is_proven_nonzero_32(ctx, ac, factors[i])) {
            continue;
        }
        std::vector<BVNodeId> F_lo;
        for (size_t k = 0; k < factors.size(); ++k) {
            if (k != i) F_lo.push_back(factors[k]);
        }
        if (try_prove_terms_lt_product(
                ctx, ac, terms, single_e32, from_add64, saw_sext, F_lo, depth + 1)) {
            return true;
        }
    }

    // Divisibility factor split: when domain guarantees P % D == 0 and D > 0, split P into {P / D, D}
    if (depth == 0 && ac.div_pairs && !ac.div_pairs->empty()) {
        for (size_t i = 0; i < factors.size(); ++i) {
            for (const auto& dc : *ac.div_pairs) {
                if (dc.dividend == factors[i] && is_proven_nonzero_32(ctx, ac, dc.divisor)) {
                    std::vector<BVNodeId> split_factors;
                    split_factors.reserve(factors.size() + 1);
                    for (size_t k = 0; k < factors.size(); ++k) {
                        if (k == i) {
                            split_factors.push_back(bv_bin(ctx, BVOp::UDiv, 32, dc.dividend, dc.divisor));
                            split_factors.push_back(dc.divisor);
                        } else {
                            split_factors.push_back(factors[k]);
                        }
                    }
                    if (try_prove_terms_lt_product(
                            ctx, ac, terms, single_e32, from_add64, saw_sext, split_factors, depth + 1)) {
                        return true;
                    }
                }
            }
        }
    }

    return false;
}

} // namespace

std::vector<uint32_t> collect_buffer_dim_params(const GpuBufferSpec& bspec, const GpuSymEnv& env) {
    std::vector<uint32_t> dims;
    const int32_t raw[3] = {bspec.length_param_index, bspec.length_param_index2, bspec.length_param_index3};
    for (int32_t idx : raw) {
        if (idx >= 0 && static_cast<size_t>(idx) < env.param_vars.size()) {
            dims.push_back(static_cast<uint32_t>(idx));
        }
    }
    return dims;
}

bool try_prove_global_access_algebraic(
    BVContext& ctx,
    BVNodeId light_domain,
    BVNodeId guard_domain,
    BVNodeId active,
    BVNodeId byte_off,
    uint32_t access_bytes,
    const GpuBufferSpec* bspec,
    const GpuSymEnv& env,
    const std::vector<DivConstraint>& div_pairs,
    uint32_t max_conflicts) {
    if (!bspec || bspec->static_byte_size > 0 || bspec->elem_bytes == 0 ||
        access_bytes == 0 || (access_bytes % bspec->elem_bytes) != 0) {
        return false;
    }
    std::vector<uint32_t> dims = collect_buffer_dim_params(*bspec, env);
    if (dims.empty()) return false;

    const uint32_t ratio = access_bytes / bspec->elem_bytes;
    std::vector<BVNodeId> cap_factors;
    cap_factors.reserve(dims.size() + 1);
    for (uint32_t idx : dims) {
        cap_factors.push_back(gpu_coerce_width(ctx, env.param_vars[idx], 32, false));
    }
    bool scaled_down = (ratio == 1);
    if (!scaled_down && bspec->elem_scale > 1 && (bspec->elem_scale % ratio) == 0) {
        uint32_t rem_scale = bspec->elem_scale / ratio;
        if (rem_scale > 1) cap_factors.push_back(bv_const(ctx, 32, rem_scale));
        scaled_down = true;
    } else if (bspec->elem_scale > 1) {
        cap_factors.push_back(bv_const(ctx, 32, bspec->elem_scale));
    }
    if (!scaled_down) {
        BVNodeId rconst = bv_const(ctx, 32, ratio);
        for (size_t i = 0; i < cap_factors.size() && !scaled_down; ++i) {
            const BVNode& fn = bv_node(ctx, cap_factors[i]);
            if (fn.op == BVOp::Const && (fn.imm % ratio) == 0) {
                cap_factors[i] = bv_const(ctx, 32, fn.imm / ratio);
                scaled_down = true;
            } else {
                for (const auto& dc : div_pairs) {
                    if (dc.dividend == cap_factors[i] && dc.divisor == rconst) {
                        cap_factors[i] = bv_bin(ctx, BVOp::UDiv, 32, dc.dividend, rconst);
                        scaled_down = true;
                        break;
                    }
                }
            }
        }
        if (!scaled_down) return false;
    }

    std::vector<BVNodeId> terms32;
    BVNodeId single32 = 0;
    bool from_add64 = false;
    bool saw_sext = false;
    if (!peel_byte_offset_64(ctx, byte_off, access_bytes, terms32, single32, from_add64, saw_sext)) {
        return false;
    }

    std::unordered_map<BVNodeId, uint64_t> umin_map;
    std::unordered_map<BVNodeId, uint64_t> umax_map;
    collect_active_bounds(ctx, guard_domain, umin_map, umax_map);
    AlgebraicCtx ac{light_domain, guard_domain, active, &umin_map, &umax_map, &div_pairs, max_conflicts};
    return try_prove_terms_lt_product(
        ctx, ac, terms32, single32, from_add64, saw_sext, cap_factors, 0);
}

BVNodeId build_buffer_capacity_64(BVContext& ctx, const GpuBufferSpec& bspec, const GpuSymEnv& env) {
    if (bspec.static_byte_size > 0) {
        return bv_const(ctx, 64, bspec.static_byte_size);
    }
    std::vector<uint32_t> dims = collect_buffer_dim_params(bspec, env);
    if (dims.empty()) return 0;

    BVNodeId cap = bv_const(ctx, 64, std::max(1u, bspec.elem_scale));
    for (uint32_t idx : dims) {
        BVNodeId l64 = gpu_coerce_width(ctx, env.param_vars[idx], 64, false);
        cap = (bv_node(ctx, cap).op == BVOp::Const && bv_node(ctx, cap).imm == 1)
            ? l64
            : bv_bin(ctx, BVOp::Mul, 64, cap, l64);
    }
    return gpu_mul_or_shl(ctx, 64, cap, bv_const(ctx, 64, std::max(1u, bspec.elem_bytes)));
}


} // namespace brass::verify
