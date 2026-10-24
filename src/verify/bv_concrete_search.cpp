#include "bv_solver.hpp"
#include <brass/interpreter/float_arith.hpp>
#include <algorithm>
#include <atomic>
#include <bit>
#include <cassert>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <sstream>
#include <vector>

namespace brass::verify {

namespace {

bool default_strict_solver_check() noexcept {
#if !defined(NDEBUG) || defined(BRASS_VERIFY_UNSAT_SAMPLES)
    return true;
#else
    const char* env = std::getenv("BRASS_VERIFY_UNSAT_SAMPLES");
    return env != nullptr && env[0] != '\0' && env[0] != '0';
#endif
}

std::atomic<bool> g_strict_solver_check{default_strict_solver_check()};
std::atomic<uint64_t> g_unsat_sample_checks{0};

uint64_t f32_to_bits(float f) noexcept { uint32_t u = 0; std::memcpy(&u, &f, sizeof(float)); return u; }
float bits_to_f32(uint64_t b) noexcept {
    uint32_t u = static_cast<uint32_t>(b & 0xFFFFFFFFULL);
    float f = 0.0f;
    std::memcpy(&f, &u, sizeof(float));
    return f;
}
uint64_t f64_to_bits(double d) noexcept { uint64_t u = 0; std::memcpy(&u, &d, sizeof(double)); return u; }
double bits_to_f64(uint64_t b) noexcept { double d = 0.0; std::memcpy(&d, &b, sizeof(double)); return d; }

bool mul_u64_overflow(uint64_t a, uint64_t b) noexcept {
    return a != 0 && b != 0 && ((a * b) / a) != b;
}

bool mul_s64_overflow(int64_t a, int64_t b) noexcept {
    if (a == 0 || b == 0) return false;
    if ((a == -1 && b == INT64_MIN) || (b == -1 && a == INT64_MIN)) return true;
    if (a > 0) return b > 0 ? (a > INT64_MAX / b) : (b < INT64_MIN / a);
    return b > 0 ? (a < INT64_MIN / b) : (a < INT64_MAX / b);
}

} // namespace

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

void set_strict_solver_check(bool enabled) noexcept {
    g_strict_solver_check.store(enabled, std::memory_order_relaxed);
}

bool strict_solver_check_enabled() noexcept {
    return g_strict_solver_check.load(std::memory_order_relaxed);
}

uint64_t unsat_sample_checks_performed() noexcept {
    return g_unsat_sample_checks.load(std::memory_order_relaxed);
}

void debug_verify_unsat_samples(const BVContext& ctx, BVNodeId bad_cond) {
    if (!strict_solver_check_enabled()) return;
    const auto& vars = ctx.vars();
    const auto& nodes = ctx.nodes();
    if (bad_cond >= nodes.size()) return;
    g_unsat_sample_checks.fetch_add(1, std::memory_order_relaxed);

    BVCone& sc = ctx.scratch_cone();
    ctx.compute_cone(bad_cond, sc.nodes, sc.vars);
    const auto& cone_nodes = sc.nodes;
    const auto& cone_vars = sc.vars;
    sc.model.assign(vars.size(), 0);
    sc.node_vals.resize(nodes.size(), 0);
    auto& model = sc.model;
    auto& node_vals = sc.node_vals;

    auto check_unsat = [&]() {
        ctx.evaluate_cone(cone_nodes, model, node_vals);
        if (node_vals[bad_cond] != 0) {
            assert(false && "Unsat soundness check failed: concrete sample satisfied bad_cond!");
            std::abort();
        }
    };

    check_unsat();
    if (cone_vars.empty()) return;

    static constexpr uint64_t kSamples[] = {
        0ULL, 1ULL, 2ULL, 3ULL, 4ULL, 7ULL, 8ULL, 15ULL, 16ULL, 31ULL, 32ULL, 63ULL, 64ULL,
        127ULL, 128ULL, 255ULL, 0x7FFFFFFFULL, 0x80000000ULL, 0xFFFFFFFFULL,
        0x7FFFFFFFFFFFFFFFULL, 0x8000000000000000ULL, ~0ULL
    };
    uint64_t rng = 0xD1B54A32D192ED03ULL ^ (static_cast<uint64_t>(bad_cond) * 0x9E3779B97F4A7C15ULL);
    auto next_rng = [&]() {
        rng ^= rng >> 12;
        rng ^= rng << 25;
        rng ^= rng >> 27;
        return rng * 0x2545F4914F6CDD1DULL;
    };
    constexpr size_t kNumSamples = sizeof(kSamples) / sizeof(kSamples[0]);
    for (uint32_t iter = 0; iter < 32; ++iter) {
        for (uint32_t vi : cone_vars) {
            uint64_t r = next_rng();
            uint64_t val = ((r & 1ULL) != 0)
                ? kSamples[static_cast<size_t>((r >> 8) % kNumSamples)]
                : next_rng();
            model[vi] = val & width_mask(vars[vi].width);
        }
        check_unsat();
    }
}

uint64_t BVContext::eval_node_concrete(const BVNode& n, uint64_t va, uint64_t vb, uint64_t vc) const noexcept {
    const uint32_t w = n.width;
    const uint64_t m = width_mask(w);
    const uint32_t aw = (n.a < nodes_.size()) ? nodes_[n.a].width : w;
    const uint64_t am = width_mask(aw);
    va &= am;
    vb &= (n.b < nodes_.size()) ? width_mask(nodes_[n.b].width) : m;
    vc &= (n.c < nodes_.size()) ? width_mask(nodes_[n.c].width) : m;

    switch (n.op) {
        case BVOp::Const:   return n.imm & m;
        case BVOp::Var:     return va & m;
        case BVOp::Extract: return (va >> n.extra) & m;
        case BVOp::Concat: {
            uint32_t lo_w = nodes_[n.b].width;
            return ((va << lo_w) | (vb & width_mask(lo_w))) & m;
        }
        case BVOp::ZExt:    return va & m;
        case BVOp::SExt:    return static_cast<uint64_t>(sign_extend_bits(va, aw)) & m;
        case BVOp::Not:     return (~va) & m;
        case BVOp::Neg:     return (~va + 1ULL) & m;
        case BVOp::And:     return (va & vb) & m;
        case BVOp::Or:      return (va | vb) & m;
        case BVOp::Xor:     return (va ^ vb) & m;
        case BVOp::Add:     return (va + vb) & m;
        case BVOp::Sub:     return (va - vb) & m;
        case BVOp::Mul:     return (va * vb) & m;
        case BVOp::UDiv:    return (vb == 0) ? 0 : ((va / vb) & m);
        case BVOp::URem:    return (vb == 0) ? 0 : ((va % vb) & m);
        case BVOp::SDiv: {
            if (vb == 0) return 0;
            int64_t sa = sign_extend_bits(va, w);
            int64_t sb = sign_extend_bits(vb, w);
            int64_t min_s = (w >= 64) ? INT64_MIN : -(1LL << (w - 1));
            if (sa == min_s && sb == -1) return static_cast<uint64_t>(min_s) & m;
            return static_cast<uint64_t>(sa / sb) & m;
        }
        case BVOp::SRem: {
            if (vb == 0) return 0;
            int64_t sa = sign_extend_bits(va, w);
            int64_t sb = sign_extend_bits(vb, w);
            int64_t min_s = (w >= 64) ? INT64_MIN : -(1LL << (w - 1));
            if (sa == min_s && sb == -1) return 0;
            return static_cast<uint64_t>(sa % sb) & m;
        }
        case BVOp::Shl: {
            uint32_t sh = static_cast<uint32_t>(vb & shift_mask(w));
            return (va << sh) & m;
        }
        case BVOp::LShr: {
            uint32_t sh = static_cast<uint32_t>(vb & shift_mask(w));
            return (va >> sh) & m;
        }
        case BVOp::AShr: {
            uint32_t sh = static_cast<uint32_t>(vb & shift_mask(w));
            int64_t sa = sign_extend_bits(va, w);
            return static_cast<uint64_t>(sa >> sh) & m;
        }
        case BVOp::Eq:  return (va == vb) ? 1 : 0;
        case BVOp::Ne:  return (va != vb) ? 1 : 0;
        case BVOp::Ult: return (va < vb) ? 1 : 0;
        case BVOp::Ule: return (va <= vb) ? 1 : 0;
        case BVOp::Ugt: return (va > vb) ? 1 : 0;
        case BVOp::Uge: return (va >= vb) ? 1 : 0;
        case BVOp::Slt: return (sign_extend_bits(va, aw) < sign_extend_bits(vb, aw)) ? 1 : 0;
        case BVOp::Sle: return (sign_extend_bits(va, aw) <= sign_extend_bits(vb, aw)) ? 1 : 0;
        case BVOp::Sgt: return (sign_extend_bits(va, aw) > sign_extend_bits(vb, aw)) ? 1 : 0;
        case BVOp::Sge: return (sign_extend_bits(va, aw) >= sign_extend_bits(vb, aw)) ? 1 : 0;
        case BVOp::Ite: return (va != 0) ? (vb & m) : (vc & m);
        case BVOp::Clz: return ((va == 0) ? w : static_cast<uint64_t>(std::countl_zero(va) - (64 - w))) & m;
        case BVOp::Ctz: return ((va == 0) ? w : static_cast<uint64_t>(std::countr_zero(va))) & m;
        case BVOp::Popcnt: return static_cast<uint64_t>(std::popcount(va & m)) & m;
        case BVOp::UAddOvf: return (((va + vb) & am) < va) ? 1 : 0;
        case BVOp::USubOvf: return (va < vb) ? 1 : 0;
        case BVOp::UMulOvf: return (aw <= 32) ? (((va * vb) > am) ? 1 : 0) : (mul_u64_overflow(va, vb) ? 1 : 0);
        case BVOp::SAddOvf: {
            int64_t sa = sign_extend_bits(va, aw), sb = sign_extend_bits(vb, aw), sr = sign_extend_bits(va + vb, aw);
            return (((sa >= 0 && sb >= 0 && sr < 0) || (sa < 0 && sb < 0 && sr >= 0))) ? 1 : 0;
        }
        case BVOp::SSubOvf: {
            int64_t sa = sign_extend_bits(va, aw), sb = sign_extend_bits(vb, aw), sr = sign_extend_bits(va - vb, aw);
            return (((sa >= 0 && sb < 0 && sr < 0) || (sa < 0 && sb >= 0 && sr >= 0))) ? 1 : 0;
        }
        case BVOp::SMulOvf: {
            int64_t sa = sign_extend_bits(va, aw), sb = sign_extend_bits(vb, aw);
            if (aw <= 32) {
                int64_t p = sa * sb, min_s = -(1LL << (aw - 1)), max_s = (1LL << (aw - 1)) - 1LL;
                return (p < min_s || p > max_s) ? 1 : 0;
            }
            return mul_s64_overflow(sa, sb) ? 1 : 0;
        }
        case BVOp::FAdd: return (w == 32) ? f32_to_bits(fparith::add(bits_to_f32(va), bits_to_f32(vb))) : f64_to_bits(fparith::add(bits_to_f64(va), bits_to_f64(vb)));
        case BVOp::FSub: return (w == 32) ? f32_to_bits(fparith::sub(bits_to_f32(va), bits_to_f32(vb))) : f64_to_bits(fparith::sub(bits_to_f64(va), bits_to_f64(vb)));
        case BVOp::FMul: return (w == 32) ? f32_to_bits(fparith::mul(bits_to_f32(va), bits_to_f32(vb))) : f64_to_bits(fparith::mul(bits_to_f64(va), bits_to_f64(vb)));
        case BVOp::FDiv: return (w == 32) ? f32_to_bits(fparith::div(bits_to_f32(va), bits_to_f32(vb))) : f64_to_bits(fparith::div(bits_to_f64(va), bits_to_f64(vb)));
        case BVOp::FSqrt: return (w == 32) ? f32_to_bits(std::sqrt(bits_to_f32(va))) : f64_to_bits(std::sqrt(bits_to_f64(va)));
        case BVOp::FAbs:  return (w == 32) ? f32_to_bits(std::fabs(bits_to_f32(va))) : f64_to_bits(std::fabs(bits_to_f64(va)));
        case BVOp::FNeg:  return (w == 32) ? f32_to_bits(-bits_to_f32(va)) : f64_to_bits(-bits_to_f64(va));
        case BVOp::FFloor: return (w == 32) ? f32_to_bits(std::floor(bits_to_f32(va))) : f64_to_bits(std::floor(bits_to_f64(va)));
        case BVOp::FCeil:  return (w == 32) ? f32_to_bits(std::ceil(bits_to_f32(va)))  : f64_to_bits(std::ceil(bits_to_f64(va)));
        case BVOp::FRound: return (w == 32) ? f32_to_bits(std::round(bits_to_f32(va))) : f64_to_bits(std::round(bits_to_f64(va)));
        case BVOp::FMin:   return (w == 32) ? f32_to_bits(std::fmin(bits_to_f32(va), bits_to_f32(vb))) : f64_to_bits(std::fmin(bits_to_f64(va), bits_to_f64(vb)));
        case BVOp::FMax:   return (w == 32) ? f32_to_bits(std::fmax(bits_to_f32(va), bits_to_f32(vb))) : f64_to_bits(std::fmax(bits_to_f64(va), bits_to_f64(vb)));
        case BVOp::FEq:    return (aw == 32) ? (bits_to_f32(va) == bits_to_f32(vb) ? 1 : 0) : (bits_to_f64(va) == bits_to_f64(vb) ? 1 : 0);
        case BVOp::FNe:    return (aw == 32) ? (bits_to_f32(va) != bits_to_f32(vb) ? 1 : 0) : (bits_to_f64(va) != bits_to_f64(vb) ? 1 : 0);
        case BVOp::FLt:    return (aw == 32) ? (bits_to_f32(va) < bits_to_f32(vb) ? 1 : 0)  : (bits_to_f64(va) < bits_to_f64(vb) ? 1 : 0);
        case BVOp::FLe:    return (aw == 32) ? (bits_to_f32(va) <= bits_to_f32(vb) ? 1 : 0) : (bits_to_f64(va) <= bits_to_f64(vb) ? 1 : 0);
        case BVOp::FGt:    return (aw == 32) ? (bits_to_f32(va) > bits_to_f32(vb) ? 1 : 0)  : (bits_to_f64(va) > bits_to_f64(vb) ? 1 : 0);
        case BVOp::FGe:    return (aw == 32) ? (bits_to_f32(va) >= bits_to_f32(vb) ? 1 : 0) : (bits_to_f64(va) >= bits_to_f64(vb) ? 1 : 0);
        case BVOp::SiToFp: {
            int64_t sv = sign_extend_bits(va, aw);
            return (w == 32) ? f32_to_bits(static_cast<float>(sv)) : f64_to_bits(static_cast<double>(sv));
        }
        case BVOp::FpToSi: {
            double d = (aw == 32) ? static_cast<double>(bits_to_f32(va)) : bits_to_f64(va);
            if (!std::isfinite(d)) return 0;
            if (w == 32) {
                if (d <= static_cast<double>(INT32_MIN)) return static_cast<uint32_t>(INT32_MIN);
                if (d >= static_cast<double>(INT32_MAX)) return static_cast<uint32_t>(INT32_MAX);
                return static_cast<uint32_t>(static_cast<int32_t>(d));
            }
            if (d <= -9223372036854775808.0) return static_cast<uint64_t>(INT64_MIN);
            if (d >= 9223372036854775807.0) return static_cast<uint64_t>(INT64_MAX);
            return static_cast<uint64_t>(static_cast<int64_t>(d)) & m;
        }
        case BVOp::FpExt:   return f64_to_bits(static_cast<double>(bits_to_f32(va)));
        case BVOp::FpTrunc: return f32_to_bits(static_cast<float>(bits_to_f64(va)));
    }
    return 0;
}

void BVContext::compute_cone(BVNodeId root, std::vector<BVNodeId>& out_nodes, std::vector<uint32_t>& out_vars) const {
    out_nodes.clear();
    out_vars.clear();
    if (root >= nodes_.size()) return;
    auto& reachable = scratch_cone_.reachable;
    auto& stack = scratch_cone_.stack;
    if (reachable.size() < nodes_.size()) reachable.resize(nodes_.size(), 0);
    stack.assign(1, root);
    reachable[root] = 1;
    for (size_t idx = 0; idx < stack.size(); ++idx) {
        const BVNode& n = nodes_[stack[idx]];
        for (BVNodeId ch : {n.a, n.b, n.c}) {
            if (ch < nodes_.size() && !reachable[ch]) { reachable[ch] = 1; stack.push_back(ch); }
        }
    }
    std::sort(stack.begin(), stack.end());
    for (BVNodeId id : stack) {
        reachable[id] = 0;
        out_nodes.push_back(id);
        if (nodes_[id].op == BVOp::Var) out_vars.push_back(static_cast<uint32_t>(nodes_[id].imm));
    }
}

void BVContext::evaluate_cone(const std::vector<BVNodeId>& cone_nodes, const std::vector<uint64_t>& var_values, std::vector<uint64_t>& node_vals) const {
    if (node_vals.size() < nodes_.size()) node_vals.resize(nodes_.size(), 0);
    for (BVNodeId id : cone_nodes) {
        const BVNode& n = nodes_[id];
        if (n.op == BVOp::Var) {
            node_vals[id] = ((n.imm < var_values.size()) ? var_values[static_cast<size_t>(n.imm)] : 0) & width_mask(n.width);
        } else if (n.op == BVOp::Const) {
            node_vals[id] = n.imm & width_mask(n.width);
        } else {
            node_vals[id] = eval_node_concrete(n, node_vals[n.a], node_vals[n.b], node_vals[n.c]);
        }
    }
}

void BVContext::evaluate_all(const std::vector<uint64_t>& var_values, std::vector<uint64_t>& node_vals) const {
    node_vals.resize(nodes_.size(), 0);
    for (size_t i = 0; i < nodes_.size(); ++i) {
        const BVNode& n = nodes_[i];
        if (n.op == BVOp::Var) {
            node_vals[i] = ((n.imm < var_values.size()) ? var_values[static_cast<size_t>(n.imm)] : 0) & width_mask(n.width);
        } else if (n.op == BVOp::Const) {
            node_vals[i] = n.imm & width_mask(n.width);
        } else {
            node_vals[i] = eval_node_concrete(n, node_vals[n.a], node_vals[n.b], node_vals[n.c]);
        }
    }
}

uint64_t BVContext::eval_concrete(BVNodeId root, const std::vector<uint64_t>& var_values) const {
    std::vector<uint64_t> node_vals;
    evaluate_all(var_values, node_vals);
    return (root < node_vals.size()) ? node_vals[root] : 0;
}

std::string BVContext::export_smt2(BVNodeId bad_cond, std::string_view comment) const {
    std::ostringstream os;
    os << "; Brass Symbolic Bitvector Equivalence Query\n";
    if (!comment.empty()) os << "; " << comment << "\n";
    os << "(set-logic QF_BV)\n";
    for (const auto& v : vars_) os << "(declare-fun " << v.name << " () (_ BitVec " << v.width << "))\n";
    for (size_t i = 0; i < nodes_.size(); ++i) {
        const BVNode& n = nodes_[i];
        os << "(define-fun t" << i << " () (_ BitVec " << n.width << ") ";
        auto r = [](BVNodeId id) { return "t" + std::to_string(id); };
        switch (n.op) {
            case BVOp::Const:   os << "(_ bv" << n.imm << " " << n.width << ")"; break;
            case BVOp::Var:     os << vars_[static_cast<size_t>(n.imm)].name; break;
            case BVOp::Extract: os << "((_ extract " << n.imm << " " << n.extra << ") " << r(n.a) << ")"; break;
            case BVOp::Concat:  os << "(concat " << r(n.a) << " " << r(n.b) << ")"; break;
            case BVOp::ZExt:    os << "((_ zero_extend " << n.extra << ") " << r(n.a) << ")"; break;
            case BVOp::SExt:    os << "((_ sign_extend " << n.extra << ") " << r(n.a) << ")"; break;
            case BVOp::Not: os << "(bvnot " << r(n.a) << ")"; break;
            case BVOp::Neg: os << "(bvneg " << r(n.a) << ")"; break;
            case BVOp::And: os << "(bvand " << r(n.a) << " " << r(n.b) << ")"; break;
            case BVOp::Or:  os << "(bvor "  << r(n.a) << " " << r(n.b) << ")"; break;
            case BVOp::Xor: os << "(bvxor " << r(n.a) << " " << r(n.b) << ")"; break;
            case BVOp::Add: os << "(bvadd " << r(n.a) << " " << r(n.b) << ")"; break;
            case BVOp::Sub: os << "(bvsub " << r(n.a) << " " << r(n.b) << ")"; break;
            case BVOp::Mul: os << "(bvmul " << r(n.a) << " " << r(n.b) << ")"; break;
            case BVOp::UDiv:
            case BVOp::SDiv:
            case BVOp::URem:
            case BVOp::SRem: {
                const char* op_s = (n.op == BVOp::UDiv) ? "bvudiv"
                                 : (n.op == BVOp::SDiv) ? "bvsdiv"
                                 : (n.op == BVOp::URem) ? "bvurem" : "bvsrem";
                os << "(ite (= " << r(n.b) << " (_ bv0 " << n.width << ")) (_ bv0 " << n.width << ") ("
                   << op_s << " " << r(n.a) << " " << r(n.b) << "))";
                break;
            }
            case BVOp::Shl:
            case BVOp::LShr:
            case BVOp::AShr: {
                const char* op_s = (n.op == BVOp::Shl) ? "bvshl" : (n.op == BVOp::LShr ? "bvlshr" : "bvashr");
                os << "(" << op_s << " " << r(n.a) << " (bvand " << r(n.b)
                   << " (_ bv" << shift_mask(n.width) << " " << n.width << ")))";
                break;
            }
            case BVOp::Eq:  os << "(ite (= "     << r(n.a) << " " << r(n.b) << ") #b1 #b0)"; break;
            case BVOp::Ne:  os << "(ite (distinct " << r(n.a) << " " << r(n.b) << ") #b1 #b0)"; break;
            case BVOp::Ult: os << "(ite (bvult " << r(n.a) << " " << r(n.b) << ") #b1 #b0)"; break;
            case BVOp::Ule: os << "(ite (bvule " << r(n.a) << " " << r(n.b) << ") #b1 #b0)"; break;
            case BVOp::Ugt: os << "(ite (bvugt " << r(n.a) << " " << r(n.b) << ") #b1 #b0)"; break;
            case BVOp::Uge: os << "(ite (bvuge " << r(n.a) << " " << r(n.b) << ") #b1 #b0)"; break;
            case BVOp::Slt: os << "(ite (bvslt " << r(n.a) << " " << r(n.b) << ") #b1 #b0)"; break;
            case BVOp::Sle: os << "(ite (bvsle " << r(n.a) << " " << r(n.b) << ") #b1 #b0)"; break;
            case BVOp::Sgt: os << "(ite (bvsgt " << r(n.a) << " " << r(n.b) << ") #b1 #b0)"; break;
            case BVOp::Sge: os << "(ite (bvsge " << r(n.a) << " " << r(n.b) << ") #b1 #b0)"; break;
            case BVOp::Ite: os << "(ite (= " << r(n.a) << " #b1) " << r(n.b) << " " << r(n.c) << ")"; break;
            default:        os << "(_ bv0 " << n.width << ")"; break;
        }
        os << ")\n";
    }
    os << "(assert (= t" << bad_cond << " #b1))\n(check-sat)\n(get-model)\n";
    return os.str();
}

} // namespace brass::verify
