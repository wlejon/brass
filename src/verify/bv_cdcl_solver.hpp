#pragma once

#include <brass/verify/bv_solver.hpp>
#include <algorithm>
#include <cstdint>
#include <unordered_map>
#include <vector>

namespace brass::verify {

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

    void reset_to_root() {
        if (decision_level() > 0) {
            backtrack(0);
        }
    }

    uint32_t new_var() {
        reset_to_root();
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
        reset_to_root();

        size_t j = 0;
        for (size_t i = 0; i < lits.size(); ++i) {
            int8_t val = value_lit(lits[i]);
            if (val == 1) return true;
            if (val == 0) continue;
            lits[j++] = lits[i];
        }
        lits.resize(j);

        std::sort(lits.begin(), lits.end());
        j = 0;
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

} // namespace brass::verify
