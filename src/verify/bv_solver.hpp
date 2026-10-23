#pragma once

#include <cstdint>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace brass::verify {

using BVNodeId = uint32_t;

enum class BVOp : uint8_t {
    Const,
    Var,
    Extract,
    Concat,
    ZExt,
    SExt,
    Not,
    Neg,
    And,
    Or,
    Xor,
    Add,
    Sub,
    Mul,
    UDiv,
    SDiv,
    URem,
    SRem,
    Shl,
    LShr,
    AShr,
    Eq,
    Ne,
    Ult,
    Ule,
    Ugt,
    Uge,
    Slt,
    Sle,
    Sgt,
    Sge,
    Ite,
    Clz,
    Ctz,
    Popcnt,
    SAddOvf,
    SSubOvf,
    SMulOvf,
    UAddOvf,
    USubOvf,
    UMulOvf,
    // IEEE-754 floating-point ops (evaluated bit-precisely via float_arith)
    FAdd,
    FSub,
    FMul,
    FDiv,
    FSqrt,
    FAbs,
    FNeg,
    FFloor,
    FCeil,
    FRound,
    FMin,
    FMax,
    FEq,
    FNe,
    FLt,
    FLe,
    FGt,
    FGe,
    SiToFp,
    FpToSi,
    FpExt,
    FpTrunc
};

struct BVNode {
    BVOp op = BVOp::Const;
    uint32_t width = 64;
    BVNodeId a = 0;
    BVNodeId b = 0;
    BVNodeId c = 0;
    uint64_t imm = 0;   // constant value (for Const), var index (for Var), or hi_bit (for Extract)
    uint32_t extra = 0; // lo_bit (for Extract) or extend amount (for ZExt/SExt)

    bool operator==(const BVNode& o) const noexcept {
        return op == o.op && width == o.width && a == o.a && b == o.b &&
               c == o.c && imm == o.imm && extra == o.extra;
    }
};

struct BVNodeHasher {
    size_t operator()(const BVNode& n) const noexcept {
        size_t h = static_cast<size_t>(n.op) * 1099511628211ULL;
        h ^= static_cast<size_t>(n.width) + 0x9e3779b9 + (h << 6) + (h >> 2);
        h ^= static_cast<size_t>(n.a) + 0x9e3779b9 + (h << 6) + (h >> 2);
        h ^= static_cast<size_t>(n.b) + 0x9e3779b9 + (h << 6) + (h >> 2);
        h ^= static_cast<size_t>(n.c) + 0x9e3779b9 + (h << 6) + (h >> 2);
        h ^= static_cast<size_t>(n.imm) + 0x9e3779b9 + (h << 6) + (h >> 2);
        h ^= static_cast<size_t>(n.extra) + 0x9e3779b9 + (h << 6) + (h >> 2);
        return h;
    }
};

struct BVVarInfo {
    std::string name;
    uint32_t width = 64;
    BVNodeId node_id = 0;
    bool is_float = false;
};

struct BVInterval {
    uint64_t umin = 0;
    uint64_t umax = ~0ULL;
    int64_t smin = INT64_MIN;
    int64_t smax = INT64_MAX;
    uint64_t known_zeros = 0;
    uint64_t known_ones = 0;
};

inline constexpr uint64_t width_mask(uint32_t w) noexcept {
    if (w == 0) return 0;
    if (w >= 64) return ~0ULL;
    return (1ULL << w) - 1ULL;
}

inline constexpr int64_t sign_extend_bits(uint64_t v, uint32_t w) noexcept {
    if (w == 0) return 0;
    if (w >= 64) return static_cast<int64_t>(v);
    uint64_t m = width_mask(w);
    v &= m;
    uint64_t sign_bit = 1ULL << (w - 1);
    if (v & sign_bit) {
        v |= ~m;
    }
    return static_cast<int64_t>(v);
}

class BVContext {
public:
    BVContext();

    BVNodeId make_const(uint64_t val, uint32_t width);
    BVNodeId make_var(std::string name, uint32_t width, bool is_float = false);
    BVNodeId make_extract(BVNodeId a, uint32_t hi, uint32_t lo);
    BVNodeId make_concat(BVNodeId hi, BVNodeId lo);
    BVNodeId make_zext(BVNodeId a, uint32_t ext_bits);
    BVNodeId make_sext(BVNodeId a, uint32_t ext_bits);
    BVNodeId make_unary(BVOp op, BVNodeId a, uint32_t res_width = 0);
    BVNodeId make_binary(BVOp op, BVNodeId a, BVNodeId b, uint32_t res_width = 0);
    BVNodeId make_ite(BVNodeId cond, BVNodeId t_val, BVNodeId f_val);

    const std::vector<BVNode>& nodes() const noexcept { return nodes_; }
    const std::vector<BVVarInfo>& vars() const noexcept { return vars_; }
    const BVInterval& interval(BVNodeId id) const noexcept { return intervals_[id]; }
    uint32_t width(BVNodeId id) const noexcept { return nodes_[id].width; }

    uint64_t eval_concrete(BVNodeId root, const std::vector<uint64_t>& var_values) const;
    void evaluate_all(const std::vector<uint64_t>& var_values, std::vector<uint64_t>& node_vals) const;

    std::string export_smt2(BVNodeId bad_cond, std::string_view comment = "") const;

private:
    BVNodeId intern(const BVNode& node);
    BVInterval compute_interval(const BVNode& n) const noexcept;
    uint64_t eval_node_concrete(const BVNode& n, uint64_t va, uint64_t vb, uint64_t vc) const noexcept;

    std::vector<BVNode> nodes_;
    std::vector<BVInterval> intervals_;
    std::vector<BVVarInfo> vars_;
    std::unordered_map<BVNode, BVNodeId, BVNodeHasher> memo_;
};

struct SolveOutcome {
    enum Status : uint8_t {
        Unsat,      // Condition is unsatisfiable (no input makes root != 0)
        Sat,        // Found concrete assignment making root != 0
        Unknown     // Budget exhausted
    };
    Status status = Unknown;
    std::vector<uint64_t> model;
};

/// Solve whether there exists an assignment to `ctx.vars()` making `bad_cond != 0`.
SolveOutcome solve_nonzero(
    const BVContext& ctx,
    BVNodeId bad_cond,
    uint32_t max_sat_conflicts = 10000
);

} // namespace brass::verify
