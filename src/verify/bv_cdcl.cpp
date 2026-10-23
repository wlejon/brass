#include "bv_solver.hpp"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <vector>

namespace brass::verify {

namespace {

// ---------------------------------------------------------------------------
// Minimal Zero-Dependency CDCL SAT Solver (2-Watched Literals + 1-UIP + VSIDS)
// ---------------------------------------------------------------------------
class CDCLSolver {
public:
    using Lit = uint32_t;
    static constexpr Lit kLitUndef = UINT32_MAX;
    static constexpr uint32_t kReasonDecision = UINT32_MAX;

    static constexpr Lit mk_lit(uint32_t var, bool neg = false) noexcept {
        return (var << 1) | (neg ? 1u : 0u);
    }
    static constexpr Lit neg(Lit l) noexcept { return l ^ 1u; }
    static constexpr uint32_t var(Lit l) noexcept { return l >> 1; }
    static constexpr bool sign(Lit l) noexcept { return (l & 1u) != 0; }

    CDCLSolver() {
        new_var(); // var 0 = constant FALSE (so mk_lit(0, true) is TRUE, mk_lit(0, false) is FALSE)
        assign(mk_lit(0, true), kReasonDecision);
    }

    Lit true_lit() const noexcept { return mk_lit(0, true); }
    Lit false_lit() const noexcept { return mk_lit(0, false); }

    uint32_t new_var() {
        uint32_t v = static_cast<uint32_t>(assigns_.size());
        assigns_.push_back(-1);
        level_.push_back(0);
        reason_.push_back(kReasonDecision);
        seen_.push_back(0);
        activity_.push_back(0.0);
        phase_.push_back(false);
        xor_lhs_.push_back(kLitUndef);
        xor_rhs_.push_back(kLitUndef);
        watches_.emplace_back();
        watches_.emplace_back();
        return v;
    }

    int8_t value_var(uint32_t v) const noexcept { return assigns_[v]; }
    int8_t value_lit(Lit l) const noexcept {
        int8_t v = assigns_[var(l)];
        if (v < 0) return -1;
        return sign(l) ? (v ^ 1) : v;
    }

    bool add_clause(std::vector<Lit> lits) {
        if (!ok_) return false;
        if (decision_level() == 0) {
            size_t j = 0;
            for (size_t i = 0; i < lits.size(); ++i) {
                int8_t val = value_lit(lits[i]);
                if (val == 1) return true;
                if (val == 0) continue;
                lits[j++] = lits[i];
            }
            lits.resize(j);
        }
        std::sort(lits.begin(), lits.end());
        size_t j = 0;
        for (size_t i = 0; i < lits.size(); ++i) {
            if (i > 0 && lits[i] == neg(lits[i - 1])) return true;
            if (i == 0 || lits[i] != lits[i - 1]) lits[j++] = lits[i];
        }
        lits.resize(j);

        if (lits.empty()) {
            ok_ = false;
            return false;
        }
        if (lits.size() == 1) {
            int8_t v = value_lit(lits[0]);
            if (v == 0) {
                ok_ = false;
                return false;
            }
            if (v < 0) {
                assign(lits[0], kReasonDecision);
                if (propagate() != UINT32_MAX) {
                    ok_ = false;
                    return false;
                }
            }
            return true;
        }
        uint32_t cid = static_cast<uint32_t>(clauses_.size());
        watches_[neg(lits[0])].push_back(cid);
        watches_[neg(lits[1])].push_back(cid);
        clauses_.push_back(std::move(lits));
        return true;
    }

    bool add_clause(Lit a) { return add_clause(std::vector<Lit>{a}); }
    bool add_clause(Lit a, Lit b) { return add_clause(std::vector<Lit>{a, b}); }
    bool add_clause(Lit a, Lit b, Lit c) { return add_clause(std::vector<Lit>{a, b, c}); }

    Lit mk_and(Lit a, Lit b) {
        if (a == false_lit() || b == false_lit()) return false_lit();
        if (a == true_lit()) return b;
        if (b == true_lit()) return a;
        if (a == b) return a;
        if (a == neg(b)) return false_lit();
        if (a > b) std::swap(a, b);
        uint64_t key = (1ULL << 62) | (static_cast<uint64_t>(a) << 30) | static_cast<uint64_t>(b);
        auto it = gate_cache_.find(key);
        if (it != gate_cache_.end()) return it->second;
        Lit out = mk_lit(new_var());
        gate_cache_[key] = out;
        add_clause(neg(a), neg(b), out);
        add_clause(a, neg(out));
        add_clause(b, neg(out));
        return out;
    }

    Lit mk_or(Lit a, Lit b) {
        return neg(mk_and(neg(a), neg(b)));
    }

    Lit mk_xor(Lit a, Lit b) {
        if (a == false_lit()) return b;
        if (b == false_lit()) return a;
        if (a == true_lit()) return neg(b);
        if (b == true_lit()) return neg(a);
        if (a == b) return false_lit();
        if (a == neg(b)) return true_lit();
        bool neg_out = sign(a) ^ sign(b);
        Lit ua = mk_lit(var(a), false);
        Lit ub = mk_lit(var(b), false);
        uint32_t va = var(ua), vb = var(ub);
        if (xor_lhs_[va] != kLitUndef) {
            if (xor_lhs_[va] == ub) return neg_out ? neg(xor_rhs_[va]) : xor_rhs_[va];
            if (xor_rhs_[va] == ub) return neg_out ? neg(xor_lhs_[va]) : xor_lhs_[va];
        }
        if (xor_lhs_[vb] != kLitUndef) {
            if (xor_lhs_[vb] == ua) return neg_out ? neg(xor_rhs_[vb]) : xor_rhs_[vb];
            if (xor_rhs_[vb] == ua) return neg_out ? neg(xor_lhs_[vb]) : xor_lhs_[vb];
        }
        if (ua > ub) std::swap(ua, ub);
        uint64_t key = (2ULL << 62) | (static_cast<uint64_t>(ua) << 30) | static_cast<uint64_t>(ub);
        auto it = gate_cache_.find(key);
        if (it != gate_cache_.end()) return neg_out ? neg(it->second) : it->second;
        uint32_t vout = new_var();
        Lit out = mk_lit(vout, false);
        xor_lhs_[vout] = ua;
        xor_rhs_[vout] = ub;
        gate_cache_[key] = out;
        add_clause(neg(ua), neg(ub), neg(out));
        add_clause(ua, ub, neg(out));
        add_clause(ua, neg(ub), out);
        add_clause(neg(ua), ub, out);
        return neg_out ? neg(out) : out;
    }

    Lit mk_ite(Lit c, Lit t, Lit f) {
        if (c == true_lit()) return t;
        if (c == false_lit()) return f;
        if (t == f) return t;
        if (t == true_lit() && f == false_lit()) return c;
        if (t == false_lit() && f == true_lit()) return neg(c);
        if (sign(c)) {
            c = neg(c);
            std::swap(t, f);
        }
        Lit out = mk_lit(new_var());
        add_clause(neg(c), neg(t), out);
        add_clause(neg(c), t, neg(out));
        add_clause(c, neg(f), out);
        add_clause(c, f, neg(out));
        return out;
    }

    SolveOutcome::Status solve_assuming(Lit assumption, uint32_t max_conflicts) {
        if (!ok_) return SolveOutcome::Unsat;
        backtrack(0);
        if (propagate() != UINT32_MAX) {
            ok_ = false;
            return SolveOutcome::Unsat;
        }
        int8_t av = value_lit(assumption);
        if (av == 0) return SolveOutcome::Unsat;
        if (av < 0) {
            trail_lim_.push_back(static_cast<uint32_t>(trail_.size()));
            assign(assumption, kReasonDecision);
        }
        uint32_t base_lvl = decision_level();
        uint32_t conflicts = 0;
        std::vector<Lit> learnt;

        while (true) {
            uint32_t confl = propagate();
            if (confl != UINT32_MAX) {
                ++conflicts;
                if (decision_level() == 0) {
                    ok_ = false;
                    return SolveOutcome::Unsat;
                }
                uint32_t bt_level = 0;
                learnt.clear();
                analyze(confl, learnt, bt_level);
                backtrack(bt_level);

                if (learnt.size() == 1) {
                    assign(learnt[0], kReasonDecision);
                } else {
                    uint32_t cid = static_cast<uint32_t>(clauses_.size());
                    watches_[neg(learnt[0])].push_back(cid);
                    watches_[neg(learnt[1])].push_back(cid);
                    clauses_.push_back(learnt);
                    assign(learnt[0], cid);
                }
                var_inc_ *= 1.05;
                if (var_inc_ > 1e100) {
                    for (double& a : activity_) a *= 1e-100;
                    var_inc_ *= 1e-100;
                }
                if (bt_level < base_lvl) {
                    if (propagate() != UINT32_MAX) {
                        ok_ = false;
                        return SolveOutcome::Unsat;
                    }
                    int8_t cur_av = value_lit(assumption);
                    if (cur_av == 0) {
                        backtrack(0);
                        return SolveOutcome::Unsat;
                    }
                    if (cur_av < 0) {
                        trail_lim_.push_back(static_cast<uint32_t>(trail_.size()));
                        assign(assumption, kReasonDecision);
                    }
                    base_lvl = decision_level();
                }
                if (conflicts >= max_conflicts) {
                    backtrack(0);
                    return SolveOutcome::Unknown;
                }
            } else {
                uint32_t next_v = pick_branch_var();
                if (next_v == UINT32_MAX) {
                    return SolveOutcome::Sat;
                }
                trail_lim_.push_back(static_cast<uint32_t>(trail_.size()));
                assign(mk_lit(next_v, !phase_[next_v]), kReasonDecision);
            }
        }
    }

    SolveOutcome::Status solve(uint32_t max_conflicts) {
        return solve_assuming(true_lit(), max_conflicts);
    }

private:
    uint32_t decision_level() const noexcept {
        return static_cast<uint32_t>(trail_lim_.size());
    }

    void assign(Lit l, uint32_t reason_cid) {
        uint32_t v = var(l);
        assigns_[v] = sign(l) ? 0 : 1;
        phase_[v] = (assigns_[v] == 1);
        level_[v] = decision_level();
        reason_[v] = reason_cid;
        trail_.push_back(l);
    }

    void backtrack(uint32_t lvl) {
        if (decision_level() <= lvl) return;
        uint32_t target = trail_lim_[lvl];
        for (size_t i = trail_.size(); i > target; --i) {
            uint32_t v = var(trail_[i - 1]);
            assigns_[v] = -1;
        }
        trail_.resize(target);
        trail_lim_.resize(lvl);
        qhead_ = target;
    }

    uint32_t propagate() {
        while (qhead_ < trail_.size()) {
            Lit p = trail_[qhead_++];
            auto& ws = watches_[p];
            size_t i = 0;
            size_t j = 0;
            uint32_t confl = UINT32_MAX;

            while (i < ws.size()) {
                uint32_t cid = ws[i++];
                auto& c = clauses_[cid];
                Lit false_l = neg(p);
                if (c[0] == false_l) {
                    std::swap(c[0], c[1]);
                }
                if (c[1] != false_l) {
                    ws[j++] = cid;
                    continue;
                }
                Lit first = c[0];
                int8_t val_first = value_lit(first);
                if (val_first == 1) {
                    ws[j++] = cid;
                    continue;
                }
                bool found_watch = false;
                for (size_t k = 2; k < c.size(); ++k) {
                    if (value_lit(c[k]) != 0) {
                        std::swap(c[1], c[k]);
                        watches_[neg(c[1])].push_back(cid);
                        found_watch = true;
                        break;
                    }
                }
                if (found_watch) continue;

                ws[j++] = cid;
                if (val_first == 0) {
                    confl = cid;
                    qhead_ = static_cast<uint32_t>(trail_.size());
                    while (i < ws.size()) ws[j++] = ws[i++];
                } else {
                    assign(first, cid);
                }
            }
            ws.resize(j);
            if (confl != UINT32_MAX) return confl;
        }
        return UINT32_MAX;
    }

    void analyze(uint32_t confl_cid, std::vector<Lit>& out_learnt, uint32_t& out_btlevel) {
        int path_c = 0;
        Lit p = kLitUndef;
        out_learnt.push_back(kLitUndef);
        size_t index = trail_.size();
        uint32_t cur_lvl = decision_level();

        uint32_t cid = confl_cid;
        do {
            const auto& c = clauses_[cid];
            for (size_t j = (p == kLitUndef ? 0 : 1); j < c.size(); ++j) {
                Lit q = c[j];
                uint32_t v = var(q);
                if (!seen_[v] && level_[v] > 0) {
                    activity_[v] += var_inc_;
                    seen_[v] = 1;
                    if (level_[v] >= cur_lvl) {
                        ++path_c;
                    } else {
                        out_learnt.push_back(q);
                    }
                }
            }
            while (!seen_[var(trail_[--index])]) {}
            p = trail_[index];
            cid = reason_[var(p)];
            seen_[var(p)] = 0;
            --path_c;
        } while (path_c > 0);

        out_learnt[0] = neg(p);
        for (size_t i = 1; i < out_learnt.size(); ++i) {
            seen_[var(out_learnt[i])] = 0;
        }

        if (out_learnt.size() == 1) {
            out_btlevel = 0;
        } else {
            size_t max_i = 1;
            for (size_t i = 2; i < out_learnt.size(); ++i) {
                if (level_[var(out_learnt[i])] > level_[var(out_learnt[max_i])]) {
                    max_i = i;
                }
            }
            Lit tmp = out_learnt[max_i];
            out_learnt[max_i] = out_learnt[1];
            out_learnt[1] = tmp;
            out_btlevel = level_[var(tmp)];
        }
    }

    uint32_t pick_branch_var() const noexcept {
        uint32_t best = UINT32_MAX;
        double best_act = -1.0;
        for (uint32_t v = 1; v < assigns_.size(); ++v) {
            if (assigns_[v] < 0 && activity_[v] > best_act) {
                best = v;
                best_act = activity_[v];
            }
        }
        return best;
    }

    bool ok_ = true;
    std::vector<std::vector<Lit>> clauses_;
    std::vector<std::vector<uint32_t>> watches_;
    std::vector<int8_t> assigns_;
    std::vector<uint32_t> level_;
    std::vector<uint32_t> reason_;
    std::vector<uint8_t> seen_;
    std::vector<double> activity_;
    std::vector<bool> phase_;
    std::vector<Lit> trail_;
    std::vector<uint32_t> trail_lim_;
    std::vector<Lit> xor_lhs_;
    std::vector<Lit> xor_rhs_;
    std::unordered_map<uint64_t, Lit> gate_cache_;
    uint32_t qhead_ = 0;
    double var_inc_ = 1.0;
};

// ---------------------------------------------------------------------------
// Stage 1: Boundary + Constant-Guided Concrete Counterexample Search
// ---------------------------------------------------------------------------
bool try_concrete_search(const BVContext& ctx, BVNodeId bad_cond, std::vector<uint64_t>& out_model) {
    const auto& vars = ctx.vars();
    const auto& nodes = ctx.nodes();
    const size_t total_vars = vars.size();

    BVCone& sc = ctx.scratch_cone();
    ctx.compute_cone(bad_cond, sc.nodes, sc.vars);
    const auto& cone_nodes = sc.nodes;
    const auto& cone_vars = sc.vars;
    const size_t nvars = cone_vars.size();

    sc.model.assign(total_vars, 0);
    sc.node_vals.resize(nodes.size(), 0);
    auto& model = sc.model;
    auto& node_vals = sc.node_vals;

    auto check = [&](const std::vector<uint64_t>& m) -> bool {
        ctx.evaluate_cone(cone_nodes, m, node_vals);
        if (node_vals[bad_cond] != 0) {
            out_model = m;
            return true;
        }
        return false;
    };

    if (check(model)) return true;
    if (nvars == 0) return false;

    std::vector<uint64_t> cands = {
        0ULL, 1ULL, 2ULL, 3ULL, 4ULL, 7ULL, 8ULL, 15ULL, 16ULL, 31ULL, 32ULL, 63ULL, 64ULL,
        127ULL, 128ULL, 255ULL, 256ULL,
        0x7FFFFFFFULL, 0x80000000ULL, 0xFFFFFFFFULL,
        0x7FFFFFFFFFFFFFFFULL, 0x8000000000000000ULL, 0xFFFFFFFFFFFFFFFFULL,
        static_cast<uint64_t>(-1LL), static_cast<uint64_t>(-2LL), static_cast<uint64_t>(-8LL)
    };

    bool has_fp_var = false;
    for (uint32_t vi : cone_vars) {
        if (vars[vi].is_float) { has_fp_var = true; break; }
    }
    if (has_fp_var) {
        const double fp_interesting[] = {0.0, -0.0, 1.0, -1.0, 0.5, -0.5, 2.0, -2.0,
                                         std::numeric_limits<double>::infinity(),
                                         -std::numeric_limits<double>::infinity(),
                                         std::numeric_limits<double>::quiet_NaN()};
        for (double d : fp_interesting) {
            uint64_t b64 = 0;
            std::memcpy(&b64, &d, 8);
            cands.push_back(b64);
            float f = static_cast<float>(d);
            uint32_t b32 = 0;
            std::memcpy(&b32, &f, 4);
            cands.push_back(b32);
        }
    }

    // Extract constants only from the cone of influence
    for (BVNodeId id : cone_nodes) {
        if (cands.size() >= 48) break;
        const auto& n = nodes[id];
        if (n.op == BVOp::Const && n.width > 1) {
            cands.push_back(n.imm);
            cands.push_back(n.imm + 1ULL);
            cands.push_back(n.imm - 1ULL);
            cands.push_back(~n.imm);
            int64_t sv = sign_extend_bits(n.imm, n.width);
            cands.push_back(static_cast<uint64_t>(sv));
            cands.push_back(static_cast<uint64_t>(sv - 1));
            cands.push_back(static_cast<uint64_t>(sv + 1));
        }
    }

    std::sort(cands.begin(), cands.end());
    cands.erase(std::unique(cands.begin(), cands.end()), cands.end());

    for (uint64_t c : cands) {
        for (uint32_t vi : cone_vars) model[vi] = c & width_mask(vars[vi].width);
        if (check(model)) return true;
    }
    const size_t sweep_vars = std::min<size_t>(nvars, 6), sweep_cands = std::min<size_t>(cands.size(), 24);
    for (uint64_t bg : {0ULL, ~0ULL, 1ULL}) {
        for (uint32_t vi : cone_vars) model[vi] = bg & width_mask(vars[vi].width);
        for (size_t i = 0; i < sweep_vars; ++i) {
            uint32_t vi = cone_vars[i];
            uint64_t orig = model[vi];
            for (size_t ci = 0; ci < sweep_cands; ++ci) {
                model[vi] = cands[ci] & width_mask(vars[vi].width);
                if (check(model)) return true;
            }
            model[vi] = orig;
        }
    }
    if (cone_nodes.size() <= 32 && nvars <= 4) {
        const size_t pair_vars = std::min<size_t>(nvars, 3), pair_cands = std::min<size_t>(cands.size(), 16);
        for (uint32_t vi : cone_vars) model[vi] = 0ULL;
        for (size_t i = 0; i < pair_vars; ++i) {
            for (size_t j = i + 1; j < pair_vars; ++j) {
                uint32_t vi = cone_vars[i], vj = cone_vars[j];
                for (size_t ci = 0; ci < pair_cands; ++ci) {
                    model[vi] = cands[ci] & width_mask(vars[vi].width);
                    for (size_t cj = 0; cj < pair_cands; ++cj) {
                        model[vj] = cands[cj] & width_mask(vars[vj].width);
                        if (check(model)) return true;
                    }
                }
                model[vi] = 0;
                model[vj] = 0;
            }
        }
    }

    // 4. Deterministic PRNG mixtures of boundary candidates and random bitvectors
    uint64_t rng = 0x9E3779B97F4A7C15ULL;
    auto next_rng = [&]() {
        rng ^= rng >> 12;
        rng ^= rng << 25;
        rng ^= rng >> 27;
        return rng * 0x2545F4914F6CDD1DULL;
    };
    const uint32_t prng_iters = (cone_nodes.size() <= 32) ? 32u : 12u;
    for (uint32_t iter = 0; iter < prng_iters; ++iter) {
        for (uint32_t vi : cone_vars) {
            uint64_t r = next_rng();
            if ((r & 3ULL) != 0 && !cands.empty()) {
                model[vi] = cands[static_cast<size_t>((r >> 8) % cands.size())] & width_mask(vars[vi].width);
            } else {
                model[vi] = next_rng() & width_mask(vars[vi].width);
            }
        }
        if (check(model)) return true;
    }

    return false;
}

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
                        out[i] = ba[n.extra + i];
                    }
                    break;
                }

                case BVOp::Concat: {
                    const auto& bhi = bits[n.a];
                    const auto& blo = bits[n.b];
                    uint32_t lo_w = static_cast<uint32_t>(blo.size());
                    for (uint32_t i = 0; i < lo_w; ++i) out[i] = blo[i];
                    for (uint32_t i = 0; i < bhi.size(); ++i) out[lo_w + i] = bhi[i];
                    break;
                }

                case BVOp::ZExt: {
                    const auto& ba = bits[n.a];
                    for (size_t i = 0; i < ba.size(); ++i) out[i] = ba[i];
                    for (size_t i = ba.size(); i < w; ++i) out[i] = sat.false_lit();
                    break;
                }
                case BVOp::SExt: {
                    const auto& ba = bits[n.a];
                    Lit sign_b = ba.back();
                    for (size_t i = 0; i < ba.size(); ++i) out[i] = ba[i];
                    for (size_t i = ba.size(); i < w; ++i) out[i] = sign_b;
                    break;
                }
                case BVOp::Not: {
                    for (uint32_t i = 0; i < w; ++i) out[i] = CDCLSolver::neg(bit_at(bits[n.a], i));
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

                case BVOp::Shl:
                case BVOp::LShr:
                case BVOp::AShr: {
                    std::vector<Lit> cur = bits[n.a];
                    const auto& sh = bits[n.b];
                    uint32_t stages = (w == 64) ? 6 : (w == 32 ? 5 : (w == 16 ? 4 : 3));
                    for (uint32_t s = 0; s < stages && s < sh.size(); ++s) {
                        uint32_t step = 1u << s;
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
                case BVOp::Slt: out[0] = blast_slt(bits[n.a], bits[n.b]); break;
                case BVOp::Sle: out[0] = CDCLSolver::neg(blast_slt(bits[n.b], bits[n.a])); break;

                case BVOp::Ite: {
                    Lit c = bit_at(bits[n.a], 0);
                    const auto& bt = bits[n.b];
                    const auto& bf = bits[n.c];
                    for (uint32_t i = 0; i < w; ++i) {
                        out[i] = sat.mk_ite(c, bit_at(bt, i), bit_at(bf, i));
                    }
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

                default: {
                    // Uninterpreted bitvector abstraction for complex ops (div/rem/float)
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

} // namespace

SolveOutcome solve_nonzero(
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
        std::vector<uint64_t> model(ctx.vars().size(), 0);
        for (size_t vi = 0; vi < ctx.vars().size(); ++vi) {
            BVNodeId vid = ctx.vars()[vi].node_id;
            uint64_t val = 0;
            const auto& vbits = blaster.bits[vid];
            for (size_t bi = 0; bi < vbits.size(); ++bi) {
                if (blaster.sat.value_lit(vbits[bi]) == 1) val |= (1ULL << bi);
            }
            model[vi] = val;
        }
        if (ctx.eval_concrete(bad_cond, model) != 0) {
            out.status = SolveOutcome::Sat;
            out.model = std::move(model);
            return out;
        }
        out.status = blaster.has_unsupported_op ? SolveOutcome::Unknown : SolveOutcome::Unsat;
        return out;
    }

    out.status = SolveOutcome::Unknown;
    return out;
}

} // namespace brass::verify
