#include "bv_solver.hpp"
#include "bv_cdcl_solver.hpp"
#include <algorithm>
#include <atomic>
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <vector>

namespace brass::verify {

namespace {

// ---------------------------------------------------------------------------
// Stage 2: Bit-Blaster from BVContext DAG to CDCL SAT
// ---------------------------------------------------------------------------
struct BitBlaster {
    const BVContext& ctx;
    CDCLSolver sat;
    std::vector<std::vector<CDCLSolver::Lit>> bits;
    bool has_unsupported_op = false;

    explicit BitBlaster(const BVContext& c) : ctx(c), bits(c.nodes().size()) {}

    using Lit = CDCLSolver::Lit;

    Lit bit_at(const std::vector<Lit>& v, size_t i) const noexcept {
        return (i < v.size()) ? v[i] : sat.false_lit();
    }

    std::vector<uint64_t> extract_model() const {
        std::vector<uint64_t> model(ctx.vars().size(), 0);
        for (size_t vi = 0; vi < ctx.vars().size(); ++vi) {
            BVNodeId vid = ctx.vars()[vi].node_id;
            uint64_t val = 0;
            const auto& vbits = bits[vid];
            for (size_t bi = 0; bi < vbits.size(); ++bi) {
                if (sat.value_lit(vbits[bi]) == 1) val |= (1ULL << bi);
            }
            model[vi] = val;
        }
        return model;
    }

    void add_adder(const std::vector<Lit>& a, const std::vector<Lit>& b,
                   Lit cin, std::vector<Lit>& sum, Lit* cout_last = nullptr, Lit* cin_last = nullptr) {
        const size_t w = std::max(a.size(), b.size());
        sum.resize(w);
        Lit carry = cin;
        for (size_t i = 0; i < w; ++i) {
            Lit ai = bit_at(a, i);
            Lit bi = bit_at(b, i);
            if (i + 1 == w && cin_last) *cin_last = carry;
            Lit axb = sat.mk_xor(ai, bi);
            sum[i] = sat.mk_xor(axb, carry);
            if (i + 1 < w || cout_last) {
                Lit ab = sat.mk_and(ai, bi);
                Lit c_axb = sat.mk_and(carry, axb);
                carry = sat.mk_or(ab, c_axb);
            }
        }
        if (cout_last) *cout_last = carry;
    }

    void blast_cond_neg(const std::vector<Lit>& a, Lit cond, std::vector<Lit>& out) {
        const size_t w = a.size();
        std::vector<Lit> flipped(w);
        std::vector<Lit> zero(w, sat.false_lit());
        for (size_t i = 0; i < w; ++i) {
            flipped[i] = sat.mk_xor(a[i], cond);
        }
        add_adder(flipped, zero, cond, out);
    }

    void blast_umul(const std::vector<Lit>& ba, const std::vector<Lit>& bb,
                    std::vector<Lit>& out, Lit* out_ovf = nullptr) {
        const size_t w = std::max(ba.size(), bb.size());
        std::vector<Lit> acc(w, sat.false_lit());
        Lit ovf = sat.false_lit();
        for (size_t i = 0; i < w; ++i) {
            Lit bi = bit_at(bb, i);
            if (bi == sat.false_lit()) continue;
            std::vector<Lit> partial(w, sat.false_lit());
            for (size_t j = 0; j < w; ++j) {
                if (i + j < w) {
                    partial[i + j] = sat.mk_and(bit_at(ba, j), bi);
                } else if (out_ovf) {
                    ovf = sat.mk_or(ovf, sat.mk_and(bit_at(ba, j), bi));
                }
            }
            std::vector<Lit> next_acc;
            Lit cout = sat.false_lit();
            add_adder(acc, partial, sat.false_lit(), next_acc, out_ovf ? &cout : nullptr, nullptr);
            if (out_ovf) ovf = sat.mk_or(ovf, cout);
            acc = std::move(next_acc);
        }
        out = std::move(acc);
        if (out_ovf) *out_ovf = ovf;
    }

    void blast_udiv_urem(const std::vector<Lit>& ba, const std::vector<Lit>& bb,
                         std::vector<Lit>& quot, std::vector<Lit>& rem) {
        const size_t w = std::max(ba.size(), bb.size());
        quot.assign(w, sat.false_lit());
        rem.assign(w, sat.false_lit());
        if (w == 0) return;

        Lit b_nz = sat.false_lit();
        for (size_t j = 0; j < w; ++j) b_nz = sat.mk_or(b_nz, bit_at(bb, j));

        std::vector<Lit> nb_ext(w + 1);
        for (size_t j = 0; j < w; ++j) nb_ext[j] = CDCLSolver::neg(bit_at(bb, j));
        nb_ext[w] = sat.true_lit();

        for (size_t step = 0; step < w; ++step) {
            size_t i = w - 1 - step;
            std::vector<Lit> r_sh(w + 1);
            r_sh[0] = bit_at(ba, i);
            for (size_t j = 0; j < w; ++j) r_sh[j + 1] = rem[j];

            std::vector<Lit> diff;
            Lit cout = sat.false_lit();
            add_adder(r_sh, nb_ext, sat.true_lit(), diff, &cout, nullptr);

            quot[i] = sat.mk_and(b_nz, cout);
            for (size_t j = 0; j < w; ++j) {
                rem[j] = sat.mk_ite(cout, diff[j], r_sh[j]);
            }
        }
        for (size_t j = 0; j < w; ++j) {
            rem[j] = sat.mk_and(b_nz, rem[j]);
        }
    }

    Lit blast_ult(const std::vector<Lit>& a, const std::vector<Lit>& b) {
        const size_t w = std::max(a.size(), b.size());
        std::vector<Lit> nb(w);
        for (size_t i = 0; i < w; ++i) nb[i] = CDCLSolver::neg(bit_at(b, i));
        std::vector<Lit> diff;
        Lit cout = sat.false_lit();
        add_adder(a, nb, sat.true_lit(), diff, &cout, nullptr);
        return CDCLSolver::neg(cout);
    }

    Lit blast_slt(const std::vector<Lit>& a, const std::vector<Lit>& b) {
        const size_t w = std::max(a.size(), b.size());
        if (w == 0) return sat.false_lit();
        std::vector<Lit> sa(w), sb(w);
        for (size_t i = 0; i < w; ++i) {
            sa[i] = bit_at(a, i);
            sb[i] = bit_at(b, i);
        }
        sa[w - 1] = CDCLSolver::neg(sa[w - 1]);
        sb[w - 1] = CDCLSolver::neg(sb[w - 1]);
        return blast_ult(sa, sb);
    }

    Lit blast_eq(const std::vector<Lit>& a, const std::vector<Lit>& b) {
        const size_t w = std::max(a.size(), b.size());
        Lit acc = sat.true_lit();
        for (size_t i = 0; i < w; ++i) {
            Lit eq_i = CDCLSolver::neg(sat.mk_xor(bit_at(a, i), bit_at(b, i)));
            acc = sat.mk_and(acc, eq_i);
        }
        return acc;
    }

    void blast_all(BVNodeId root) {
        const auto& nodes = ctx.nodes();
        std::vector<uint8_t> reachable(nodes.size(), 0);
        std::vector<BVNodeId> stack = {root};
        reachable[root] = 1;
        while (!stack.empty()) {
            BVNodeId u = stack.back();
            stack.pop_back();
            const BVNode& n = nodes[u];
            for (BVNodeId ch : {n.a, n.b, n.c}) {
                if (ch < nodes.size() && !reachable[ch]) {
                    reachable[ch] = 1;
                    stack.push_back(ch);
                }
            }
        }

        for (size_t id = 0; id < nodes.size(); ++id) {
            if (!reachable[id]) continue;
            const BVNode& n = nodes[id];
            const uint32_t w = n.width;
            auto& out = bits[id];
            out.resize(w, sat.false_lit());

            switch (n.op) {
                case BVOp::Const:
                    for (uint32_t i = 0; i < w; ++i) {
                        out[i] = ((n.imm >> i) & 1ULL) ? sat.true_lit() : sat.false_lit();
                    }
                    break;

                case BVOp::Var:
                    for (uint32_t i = 0; i < w; ++i) {
                        out[i] = CDCLSolver::mk_lit(sat.new_var());
                    }
                    break;

                case BVOp::Extract: {
                    const auto& ba = bits[n.a];
                    for (uint32_t i = 0; i < w; ++i) {
                        out[i] = bit_at(ba, n.extra + i);
                    }
                    break;
                }

                case BVOp::Concat: {
                    const auto& bhi = bits[n.a];
                    const auto& blo = bits[n.b];
                    uint32_t lo_w = static_cast<uint32_t>(blo.size());
                    for (uint32_t i = 0; i < lo_w && i < w; ++i) out[i] = blo[i];
                    for (uint32_t i = 0; i < bhi.size() && lo_w + i < w; ++i) out[lo_w + i] = bhi[i];
                    break;
                }

                case BVOp::ZExt: {
                    const auto& ba = bits[n.a];
                    for (size_t i = 0; i < ba.size() && i < w; ++i) out[i] = ba[i];
                    for (size_t i = ba.size(); i < w; ++i) out[i] = sat.false_lit();
                    break;
                }
                case BVOp::SExt: {
                    const auto& ba = bits[n.a];
                    Lit sign_b = ba.empty() ? sat.false_lit() : ba.back();
                    for (size_t i = 0; i < ba.size() && i < w; ++i) out[i] = ba[i];
                    for (size_t i = ba.size(); i < w; ++i) out[i] = sign_b;
                    break;
                }
                case BVOp::Not: {
                    for (uint32_t i = 0; i < w; ++i) out[i] = CDCLSolver::neg(bit_at(bits[n.a], i));
                    break;
                }
                case BVOp::Neg: {
                    blast_cond_neg(bits[n.a], sat.true_lit(), out);
                    break;
                }
                case BVOp::And: {
                    for (uint32_t i = 0; i < w; ++i) out[i] = sat.mk_and(bit_at(bits[n.a], i), bit_at(bits[n.b], i));
                    break;
                }
                case BVOp::Or: {
                    for (uint32_t i = 0; i < w; ++i) out[i] = sat.mk_or(bit_at(bits[n.a], i), bit_at(bits[n.b], i));
                    break;
                }
                case BVOp::Xor: {
                    for (uint32_t i = 0; i < w; ++i) out[i] = sat.mk_xor(bit_at(bits[n.a], i), bit_at(bits[n.b], i));
                    break;
                }
                case BVOp::Add: {
                    add_adder(bits[n.a], bits[n.b], sat.false_lit(), out);
                    break;
                }
                case BVOp::Sub: {
                    std::vector<Lit> nb(w);
                    for (uint32_t i = 0; i < w; ++i) nb[i] = CDCLSolver::neg(bit_at(bits[n.b], i));
                    add_adder(bits[n.a], nb, sat.true_lit(), out);
                    break;
                }
                case BVOp::Mul: {
                    const auto& ba = bits[n.a];
                    const auto& bb = bits[n.b];
                    auto mul_half = [&](uint32_t ja_off, uint32_t ja_len, uint32_t ib_lim) {
                        std::vector<Lit> acc(w, sat.false_lit());
                        for (uint32_t i = 0; i < ib_lim; ++i) {
                            if (bb[i] == sat.false_lit()) continue;
                            std::vector<Lit> partial(w, sat.false_lit());
                            for (uint32_t j = 0; j < ja_len && i + j < w; ++j)
                                partial[i + j] = sat.mk_and(ba[ja_off + j], bb[i]);
                            std::vector<Lit> next_acc;
                            add_adder(acc, partial, sat.false_lit(), next_acc);
                            acc = std::move(next_acc);
                        }
                        return acc;
                    };
                    bool has_hi = false;
                    if (w == 32) {
                        for (uint32_t j = 16; j < 32; ++j)
                            if (ba[j] != sat.false_lit()) { has_hi = true; break; }
                    }
                    if (!has_hi) {
                        out = mul_half(0, w, w);
                    } else {
                        std::vector<Lit> lo = mul_half(0, 16, 32), hi = mul_half(16, 16, 16), hi_sh(32, sat.false_lit());
                        for (uint32_t j = 0; j < 16; ++j) hi_sh[16 + j] = hi[j];
                        add_adder(lo, hi_sh, sat.false_lit(), out);
                    }
                    break;
                }
                case BVOp::UDiv:
                case BVOp::URem: {
                    std::vector<Lit> quot, rem;
                    blast_udiv_urem(bits[n.a], bits[n.b], quot, rem);
                    out = (n.op == BVOp::UDiv) ? std::move(quot) : std::move(rem);
                    break;
                }
                case BVOp::SDiv:
                case BVOp::SRem: {
                    const auto& ba = bits[n.a];
                    const auto& bb = bits[n.b];
                    Lit sign_a = bit_at(ba, w - 1);
                    Lit sign_b = bit_at(bb, w - 1);
                    std::vector<Lit> abs_a, abs_b, uq, ur;
                    blast_cond_neg(ba, sign_a, abs_a);
                    blast_cond_neg(bb, sign_b, abs_b);
                    blast_udiv_urem(abs_a, abs_b, uq, ur);
                    if (n.op == BVOp::SDiv) {
                        blast_cond_neg(uq, sat.mk_xor(sign_a, sign_b), out);
                    } else {
                        blast_cond_neg(ur, sign_a, out);
                    }
                    break;
                }

                case BVOp::Shl:
                case BVOp::LShr:
                case BVOp::AShr: {
                    std::vector<Lit> cur = bits[n.a];
                    const auto& sh = bits[n.b];
                    uint32_t sh_mask = shift_mask(w);
                    for (uint32_t s = 0; s < 6 && s < sh.size(); ++s) {
                        uint32_t step = 1u << s;
                        if ((sh_mask & step) == 0u) continue;
                        Lit sel = sh[s];
                        if (sel == sat.false_lit()) continue;
                        std::vector<Lit> next(w);
                        Lit fill = (n.op == BVOp::AShr) ? cur[w - 1] : sat.false_lit();
                        for (uint32_t i = 0; i < w; ++i) {
                            Lit shifted = sat.false_lit();
                            if (n.op == BVOp::Shl) {
                                shifted = (i >= step) ? cur[i - step] : sat.false_lit();
                            } else {
                                shifted = (i + step < w) ? cur[i + step] : fill;
                            }
                            next[i] = sat.mk_ite(sel, shifted, cur[i]);
                        }
                        cur = std::move(next);
                    }
                    out = std::move(cur);
                    break;
                }

                case BVOp::Eq:  out[0] = blast_eq(bits[n.a], bits[n.b]); break;
                case BVOp::Ne:  out[0] = CDCLSolver::neg(blast_eq(bits[n.a], bits[n.b])); break;
                case BVOp::Ult: out[0] = blast_ult(bits[n.a], bits[n.b]); break;
                case BVOp::Ule: out[0] = CDCLSolver::neg(blast_ult(bits[n.b], bits[n.a])); break;
                case BVOp::Ugt: out[0] = blast_ult(bits[n.b], bits[n.a]); break;
                case BVOp::Uge: out[0] = CDCLSolver::neg(blast_ult(bits[n.a], bits[n.b])); break;
                case BVOp::Slt: out[0] = blast_slt(bits[n.a], bits[n.b]); break;
                case BVOp::Sle: out[0] = CDCLSolver::neg(blast_slt(bits[n.b], bits[n.a])); break;
                case BVOp::Sgt: out[0] = blast_slt(bits[n.b], bits[n.a]); break;
                case BVOp::Sge: out[0] = CDCLSolver::neg(blast_slt(bits[n.a], bits[n.b])); break;

                case BVOp::Ite: {
                    Lit c = bit_at(bits[n.a], 0);
                    const auto& bt = bits[n.b];
                    const auto& bf = bits[n.c];
                    for (uint32_t i = 0; i < w; ++i) {
                        out[i] = sat.mk_ite(c, bit_at(bt, i), bit_at(bf, i));
                    }
                    break;
                }

                case BVOp::Clz: {
                    const auto& ba = bits[n.a];
                    std::vector<Lit> cur(w);
                    for (uint32_t j = 0; j < w; ++j) {
                        cur[j] = ((static_cast<uint64_t>(w) >> j) & 1ULL) ? sat.true_lit() : sat.false_lit();
                    }
                    for (uint32_t bit = 0; bit < w; ++bit) {
                        uint64_t lz = static_cast<uint64_t>(w - 1u - bit);
                        Lit b_lit = bit_at(ba, bit);
                        for (uint32_t j = 0; j < w; ++j) {
                            Lit lz_j = ((lz >> j) & 1ULL) ? sat.true_lit() : sat.false_lit();
                            cur[j] = sat.mk_ite(b_lit, lz_j, cur[j]);
                        }
                    }
                    out = std::move(cur);
                    break;
                }

                case BVOp::Ctz: {
                    const auto& ba = bits[n.a];
                    std::vector<Lit> cur(w);
                    for (uint32_t j = 0; j < w; ++j) {
                        cur[j] = ((static_cast<uint64_t>(w) >> j) & 1ULL) ? sat.true_lit() : sat.false_lit();
                    }
                    for (uint32_t rev = 0; rev < w; ++rev) {
                        uint64_t bit = static_cast<uint64_t>(w - 1u - rev);
                        Lit b_lit = bit_at(ba, static_cast<size_t>(bit));
                        for (uint32_t j = 0; j < w; ++j) {
                            Lit tz_j = ((bit >> j) & 1ULL) ? sat.true_lit() : sat.false_lit();
                            cur[j] = sat.mk_ite(b_lit, tz_j, cur[j]);
                        }
                    }
                    out = std::move(cur);
                    break;
                }

                case BVOp::Popcnt: {
                    const auto& ba = bits[n.a];
                    std::vector<Lit> acc(w, sat.false_lit());
                    for (uint32_t bit = 0; bit < w; ++bit) {
                        std::vector<Lit> one_bit(w, sat.false_lit());
                        one_bit[0] = bit_at(ba, bit);
                        std::vector<Lit> next_acc;
                        add_adder(acc, one_bit, sat.false_lit(), next_acc);
                        acc = std::move(next_acc);
                    }
                    out = std::move(acc);
                    break;
                }

                case BVOp::UAddOvf: {
                    std::vector<Lit> sum;
                    Lit cout = sat.false_lit();
                    add_adder(bits[n.a], bits[n.b], sat.false_lit(), sum, &cout, nullptr);
                    out[0] = cout;
                    break;
                }

                case BVOp::USubOvf:
                    out[0] = blast_ult(bits[n.a], bits[n.b]);
                    break;

                case BVOp::UMulOvf: {
                    std::vector<Lit> prod;
                    blast_umul(bits[n.a], bits[n.b], prod, &out[0]);
                    break;
                }

                case BVOp::SAddOvf: {
                    std::vector<Lit> sum;
                    Lit cout = sat.false_lit();
                    Lit cin_msb = sat.false_lit();
                    add_adder(bits[n.a], bits[n.b], sat.false_lit(), sum, &cout, &cin_msb);
                    out[0] = sat.mk_xor(cout, cin_msb);
                    break;
                }

                case BVOp::SSubOvf: {
                    const auto& ba = bits[n.a];
                    const auto& bb = bits[n.b];
                    std::vector<Lit> nb(ba.size());
                    for (size_t i = 0; i < ba.size(); ++i) nb[i] = CDCLSolver::neg(bb[i]);
                    std::vector<Lit> sum;
                    Lit cout = sat.false_lit();
                    Lit cin_msb = sat.false_lit();
                    add_adder(ba, nb, sat.true_lit(), sum, &cout, &cin_msb);
                    out[0] = sat.mk_xor(cout, cin_msb);
                    break;
                }

                case BVOp::SMulOvf: {
                    const auto& ba = bits[n.a];
                    const auto& bb = bits[n.b];
                    const size_t aw = ba.size();
                    Lit sign_a = bit_at(ba, aw - 1);
                    Lit sign_b = bit_at(bb, aw - 1);
                    std::vector<Lit> abs_a, abs_b, u_prod;
                    blast_cond_neg(ba, sign_a, abs_a);
                    blast_cond_neg(bb, sign_b, abs_b);
                    Lit umul_ovf = sat.false_lit();
                    blast_umul(abs_a, abs_b, u_prod, &umul_ovf);
                    Lit same_sign = CDCLSolver::neg(sat.mk_xor(sign_a, sign_b));
                    Lit low_nz = sat.false_lit();
                    for (size_t i = 0; i + 1 < aw; ++i) {
                        low_nz = sat.mk_or(low_nz, u_prod[i]);
                    }
                    Lit msb_bad = sat.mk_and(u_prod[aw - 1], sat.mk_or(same_sign, low_nz));
                    out[0] = sat.mk_or(umul_ovf, msb_bad);
                    break;
                }

                default: {
                    // Uninterpreted bitvector abstraction for floating-point ops
                    has_unsupported_op = true;
                    for (uint32_t i = 0; i < w; ++i) {
                        out[i] = CDCLSolver::mk_lit(sat.new_var());
                    }
                    break;
                }
            }
        }
    }
};

SolveOutcome solve_nonzero_impl(
    const BVContext& ctx,
    BVNodeId bad_cond,
    uint32_t max_sat_conflicts
) {
    SolveOutcome out;
    const auto& nodes = ctx.nodes();
    const auto& root = nodes[bad_cond];
    if (root.op == BVOp::Const) {
        if (root.imm == 0) {
            out.status = SolveOutcome::Unsat;
        } else {
            out.status = SolveOutcome::Sat;
            out.model.assign(ctx.vars().size(), 0);
        }
        return out;
    }

    // Fast level-0 structural contradiction check on top-level 1-bit And conjuncts
    if (root.width == 1 && root.op == BVOp::And) {
        std::vector<BVNodeId> stack = {bad_cond};
        std::vector<BVNodeId> pos_lits;
        std::vector<BVNodeId> neg_lits;
        while (!stack.empty()) {
            BVNodeId u = stack.back();
            stack.pop_back();
            const BVNode& n = nodes[u];
            if (n.op == BVOp::Const && n.imm == 0) {
                out.status = SolveOutcome::Unsat;
                return out;
            }
            if (n.op == BVOp::And && n.width == 1) {
                stack.push_back(n.a);
                stack.push_back(n.b);
            } else if (n.op == BVOp::Not && n.width == 1) {
                neg_lits.push_back(n.a);
            } else {
                pos_lits.push_back(u);
            }
        }
        for (BVNodeId p : pos_lits) {
            if (std::find(neg_lits.begin(), neg_lits.end(), p) != neg_lits.end()) {
                out.status = SolveOutcome::Unsat;
                return out;
            }
        }
    }

    // Stage 1: Fast concrete boundary & algebraic constant search
    if (try_concrete_search(ctx, bad_cond, out.model)) {
        out.status = SolveOutcome::Sat;
        return out;
    }

    // Stage 2: Exact Bit-Blasted CDCL SAT
    BitBlaster blaster(ctx);
    blaster.blast_all(bad_cond);

    // Inductive LSB-to-MSB bit-sliced equivalence check when bad_cond is (lhs != rhs) && domain
    if (root.width == 1) {
        BVNodeId neq_eq_node = UINT32_MAX;
        std::vector<BVNodeId> dom_nodes, stack = {bad_cond};
        while (!stack.empty()) {
            BVNodeId u = stack.back();
            stack.pop_back();
            const BVNode& n = nodes[u];
            if (n.op == BVOp::And && n.width == 1) {
                stack.push_back(n.a);
                stack.push_back(n.b);
            } else if (neq_eq_node == UINT32_MAX && n.op == BVOp::Not && n.width == 1 &&
                       nodes[n.a].op == BVOp::Eq && nodes[nodes[n.a].a].width > 1) {
                neq_eq_node = n.a;
            } else {
                dom_nodes.push_back(u);
            }
        }
        if (neq_eq_node != UINT32_MAX) {
            for (BVNodeId d : dom_nodes) {
                if (!blaster.sat.add_clause(blaster.bits[d][0])) {
                    out.status = SolveOutcome::Unsat;
                    return out;
                }
            }
            const auto& ba = blaster.bits[nodes[neq_eq_node].a];
            const auto& bb = blaster.bits[nodes[neq_eq_node].b];
            bool all_bits_unsat = true;
            for (size_t bi = 0; bi < ba.size(); ++bi) {
                CDCLSolver::Lit diff_i = blaster.sat.mk_xor(ba[bi], bb[bi]);
                SolveOutcome::Status bst = blaster.sat.solve_assuming(diff_i, max_sat_conflicts);
                if (bst == SolveOutcome::Unsat) {
                    if (!blaster.sat.add_clause(CDCLSolver::neg(diff_i))) {
                        out.status = SolveOutcome::Unsat;
                        return out;
                    }
                } else {
                    all_bits_unsat = false;
                    break;
                }
            }
            if (all_bits_unsat) {
                out.status = SolveOutcome::Unsat;
                return out;
            }
        }
    }

    CDCLSolver::Lit root_lit = blaster.sat.false_lit();
    for (CDCLSolver::Lit b : blaster.bits[bad_cond]) {
        root_lit = blaster.sat.mk_or(root_lit, b);
    }
    if (!blaster.sat.add_clause(root_lit)) {
        out.status = SolveOutcome::Unsat;
        return out;
    }

    SolveOutcome::Status st = blaster.sat.solve(max_sat_conflicts);
    if (st == SolveOutcome::Unsat) {
        out.status = SolveOutcome::Unsat;
        return out;
    }
    if (st == SolveOutcome::Sat) {
        std::vector<uint64_t> model = blaster.extract_model();
        if (ctx.eval_concrete(bad_cond, model) != 0) {
            out.status = SolveOutcome::Sat;
            out.model = std::move(model);
            return out;
        }
        if (!blaster.has_unsupported_op) {
            if (strict_solver_check_enabled()) {
                assert(false && "BVSolver soundness bug: SAT model from bit-blaster failed concrete evaluation");
                std::abort();
            }
            static std::atomic<bool> s_logged_mismatch{false};
            if (!s_logged_mismatch.exchange(true, std::memory_order_relaxed)) {
                std::fprintf(stderr,
                    "[brass::verify] warning: BVSolver bit-blaster SAT model failed concrete evaluation; "
                    "falling back to SolveOutcome::Unknown\n");
            }
        }
        out.status = SolveOutcome::Unknown;
        return out;
    }

    out.status = SolveOutcome::Unknown;
    return out;
}

} // namespace

SolveOutcome solve_nonzero(
    const BVContext& ctx,
    BVNodeId bad_cond,
    uint32_t max_sat_conflicts
) {
    SolveOutcome out = solve_nonzero_impl(ctx, bad_cond, max_sat_conflicts);
    if (out.status == SolveOutcome::Unsat) {
        debug_verify_unsat_samples(ctx, bad_cond);
    }
    return out;
}

} // namespace brass::verify
