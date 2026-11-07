#pragma once

#include <brass/mir/function.hpp>
#include <brass/mir/instruction.hpp>
#include <brass/mir/module.hpp>
#include <brass/mir/superopt.hpp>
#include <cstddef>
#include <cstdint>
#include <unordered_map>
#include <utility>
#include <vector>

namespace brass {

using EClassId = uint32_t;
inline constexpr EClassId kInvalidEClass = UINT32_MAX;

enum class EOp : uint8_t {
    Var,
    Const,
    Add,
    Sub,
    Mul,
    SDiv,
    UDiv,
    SMod,
    UMod,
    And,
    Or,
    Xor,
    Shl,
    LShr,
    AShr,
    Neg,
    Not,
    Eq,
    Ne,
    Slt,
    Ult,
    Sle,
    Ule,
    Select,
    SExt64,
    ZExt64,
    Trunc32
};

constexpr uint32_t eop_arity(EOp op) noexcept {
    switch (op) {
        case EOp::Var:
        case EOp::Const:
            return 0;
        case EOp::Neg:
        case EOp::Not:
        case EOp::SExt64:
        case EOp::ZExt64:
        case EOp::Trunc32:
            return 1;
        case EOp::Select:
            return 3;
        default:
            return 2;
    }
}

constexpr bool eop_is_commutative(EOp op) noexcept {
    switch (op) {
        case EOp::Add:
        case EOp::Mul:
        case EOp::And:
        case EOp::Or:
        case EOp::Xor:
        case EOp::Eq:
        case EOp::Ne:
            return true;
        default:
            return false;
    }
}

constexpr uint32_t eop_cost(EOp op) noexcept {
    switch (op) {
        case EOp::Var:
        case EOp::Const:
            return 0;
        case EOp::Add:
        case EOp::Sub:
        case EOp::And:
        case EOp::Or:
        case EOp::Xor:
        case EOp::Shl:
        case EOp::LShr:
        case EOp::AShr:
        case EOp::Neg:
        case EOp::Not:
        case EOp::Eq:
        case EOp::Ne:
        case EOp::Slt:
        case EOp::Ult:
        case EOp::Sle:
        case EOp::Ule:
        case EOp::SExt64:
        case EOp::ZExt64:
        case EOp::Trunc32:
            return 1;
        case EOp::Select:
            return 2;
        case EOp::Mul:
            return 3;
        case EOp::SDiv:
        case EOp::UDiv:
        case EOp::SMod:
        case EOp::UMod:
            return 8;
    }
    return 1;
}

struct ENode {
    EOp op = EOp::Const;
    uint32_t width = 64;
    EClassId a = kInvalidEClass;
    EClassId b = kInvalidEClass;
    EClassId c = kInvalidEClass;
    uint64_t imm = 0;

    constexpr void canonicalize_operands() noexcept {
        const uint32_t ar = eop_arity(op);
        if (ar == 0) {
            a = kInvalidEClass;
            b = kInvalidEClass;
            c = kInvalidEClass;
            if (op == EOp::Const && width < 64) {
                imm &= (1ULL << width) - 1ULL;
            }
        } else if (ar == 1) {
            b = kInvalidEClass;
            c = kInvalidEClass;
            imm = 0;
        } else if (ar == 2) {
            c = kInvalidEClass;
            imm = 0;
            if (eop_is_commutative(op) && a > b) {
                const EClassId tmp = a;
                a = b;
                b = tmp;
            }
        } else {
            imm = 0;
        }
    }

    constexpr bool operator==(const ENode& o) const noexcept {
        return op == o.op && width == o.width && a == o.a && b == o.b &&
               c == o.c && imm == o.imm;
    }

    constexpr bool operator!=(const ENode& o) const noexcept {
        return !(*this == o);
    }
};

struct ENodeHasher {
    size_t operator()(const ENode& n) const noexcept {
        size_t h = static_cast<size_t>(n.op) * 1099511628211ULL;
        h ^= static_cast<size_t>(n.width) + 0x9e3779b9u + (h << 6) + (h >> 2);
        h ^= static_cast<size_t>(n.a) + 0x9e3779b9u + (h << 6) + (h >> 2);
        h ^= static_cast<size_t>(n.b) + 0x9e3779b9u + (h << 6) + (h >> 2);
        h ^= static_cast<size_t>(n.c) + 0x9e3779b9u + (h << 6) + (h >> 2);
        h ^= static_cast<size_t>(n.imm) + 0x9e3779b9u + (h << 6) + (h >> 2);
        return h;
    }
};

struct EGraphLimits {
    uint32_t max_enodes = 512;
    uint32_t max_eclasses = 256;
    uint32_t max_iters = 4;
    uint32_t max_sat_conflicts = 2000;
};

struct EGraphStats {
    uint32_t enodes_added = 0;
    uint32_t eclasses_merged = 0;
    uint32_t saturation_iters = 0;
    uint32_t rewrites_applied = 0;
    uint32_t budget_hit = 0;
};

class EGraph {
public:
    struct EClass {
        EClassId id = kInvalidEClass;
        uint32_t width = 64;
        Value* ssa_val = nullptr;
        uint32_t ssa_order = UINT32_MAX;
        bool has_const = false;
        uint64_t const_imm = 0;
        std::vector<ENode> nodes;
        std::vector<std::pair<ENode, EClassId>> parents;
    };

    EGraph() = default;

    EClassId find(EClassId id) const noexcept {
        if (id == kInvalidEClass || id >= parents_uf_.size()) {
            return kInvalidEClass;
        }
        EClassId cur = id;
        while (parents_uf_[cur] != cur) {
            parents_uf_[cur] = parents_uf_[parents_uf_[cur]];
            cur = parents_uf_[cur];
        }
        return cur;
    }

    ENode canonicalize(ENode n) const noexcept {
        const uint32_t ar = eop_arity(n.op);
        if (ar >= 1 && n.a != kInvalidEClass) n.a = find(n.a);
        if (ar >= 2 && n.b != kInvalidEClass) n.b = find(n.b);
        if (ar >= 3 && n.c != kInvalidEClass) n.c = find(n.c);
        n.canonicalize_operands();
        return n;
    }

    EClassId add_var(uint32_t var_id, uint32_t width, Value* ssa_val = nullptr) {
        if (ssa_val != nullptr) {
            var_ssa_[var_id] = ssa_val;
        }
        ENode n;
        n.op = EOp::Var;
        n.width = width;
        n.imm = var_id;
        return add_node(n, ssa_val);
    }

    Value* var_value(uint32_t var_id) const noexcept {
        auto it = var_ssa_.find(var_id);
        return (it != var_ssa_.end()) ? it->second : nullptr;
    }

    EClassId add_const(uint64_t imm, uint32_t width, Value* ssa_val = nullptr) {
        ENode n;
        n.op = EOp::Const;
        n.width = width;
        n.imm = (width < 64) ? (imm & ((1ULL << width) - 1ULL)) : imm;
        return add_node(n, ssa_val);
    }

    EClassId add_node(const ENode& raw_node, Value* ssa_val = nullptr) {
        ENode n = canonicalize(raw_node);
        auto it = memo_.find(n);
        if (it != memo_.end()) {
            const EClassId cid = find(it->second);
            if (ssa_val != nullptr && classes_[cid].ssa_val == nullptr) {
                classes_[cid].ssa_val = ssa_val;
                classes_[cid].ssa_order = next_ssa_order_++;
            }
            return cid;
        }

        const EClassId cid = static_cast<EClassId>(classes_.size());
        parents_uf_.push_back(cid);

        EClass cls;
        cls.id = cid;
        cls.width = n.width;
        if (ssa_val != nullptr) {
            cls.ssa_val = ssa_val;
            cls.ssa_order = next_ssa_order_++;
        }
        if (n.op == EOp::Const) {
            cls.has_const = true;
            cls.const_imm = n.imm;
        }
        cls.nodes.push_back(n);
        classes_.push_back(std::move(cls));

        const uint32_t ar = eop_arity(n.op);
        if (ar >= 1 && n.a < classes_.size()) {
            classes_[n.a].parents.emplace_back(n, cid);
        }
        if (ar >= 2 && n.b < classes_.size() && n.b != n.a) {
            classes_[n.b].parents.emplace_back(n, cid);
        }
        if (ar >= 3 && n.c < classes_.size() && n.c != n.a && n.c != n.b) {
            classes_[n.c].parents.emplace_back(n, cid);
        }

        memo_.emplace(n, cid);
        ++total_enodes_;
        return cid;
    }

    bool merge(EClassId c1, EClassId c2) {
        c1 = find(c1);
        c2 = find(c2);
        if (c1 == kInvalidEClass || c2 == kInvalidEClass || c1 == c2) {
            return false;
        }
        if (classes_[c1].parents.size() < classes_[c2].parents.size()) {
            const EClassId tmp = c1;
            c1 = c2;
            c2 = tmp;
        }
        parents_uf_[c2] = c1;

        EClass& dst = classes_[c1];
        EClass& src = classes_[c2];
        if (src.ssa_val != nullptr) {
            if (dst.ssa_val == nullptr || src.ssa_order < dst.ssa_order) {
                dst.ssa_val = src.ssa_val;
                dst.ssa_order = src.ssa_order;
            }
        }
        if (src.has_const && !dst.has_const) {
            dst.has_const = true;
            dst.const_imm = src.const_imm;
        }
        dst.nodes.insert(dst.nodes.end(), src.nodes.begin(), src.nodes.end());
        src.nodes.clear();
        dst.parents.insert(dst.parents.end(), src.parents.begin(), src.parents.end());
        src.parents.clear();

        worklist_.push_back(c1);
        ++total_merges_;
        return true;
    }

    void rebuild() {
        while (!worklist_.empty()) {
            std::vector<EClassId> todo;
            todo.swap(worklist_);
            for (EClassId cid : todo) {
                const EClassId leader = find(cid);
                if (leader == kInvalidEClass) continue;
                std::vector<std::pair<ENode, EClassId>> old_parents = std::move(classes_[leader].parents);
                classes_[leader].parents.clear();

                std::vector<std::pair<ENode, EClassId>> new_parents;
                new_parents.reserve(old_parents.size());
                for (const auto& [old_node, pid] : old_parents) {
                    auto it_old = memo_.find(old_node);
                    if (it_old != memo_.end() && find(it_old->second) == find(pid)) {
                        memo_.erase(it_old);
                    }
                    const ENode new_node = canonicalize(old_node);
                    EClassId canon_pid = find(pid);
                    auto it = memo_.find(new_node);
                    if (it != memo_.end()) {
                        const EClassId existing_pid = find(it->second);
                        if (existing_pid != canon_pid) {
                            merge(existing_pid, canon_pid);
                            canon_pid = find(existing_pid);
                        }
                        it->second = canon_pid;
                    } else {
                        memo_.emplace(new_node, canon_pid);
                        new_parents.emplace_back(new_node, canon_pid);
                    }
                }
                auto& dst_parents = classes_[find(leader)].parents;
                dst_parents.insert(dst_parents.end(), new_parents.begin(), new_parents.end());
            }
        }

        for (size_t i = 0; i < classes_.size(); ++i) {
            if (parents_uf_[i] != static_cast<EClassId>(i)) continue;
            auto& nodes = classes_[i].nodes;
            for (ENode& n : nodes) {
                n = canonicalize(n);
                memo_[n] = static_cast<EClassId>(i);
            }
            size_t w_idx = 0;
            for (size_t r_idx = 0; r_idx < nodes.size(); ++r_idx) {
                bool dup = false;
                for (size_t j = 0; j < w_idx; ++j) {
                    if (nodes[j] == nodes[r_idx]) {
                        dup = true;
                        break;
                    }
                }
                if (!dup) {
                    nodes[w_idx++] = nodes[r_idx];
                }
            }
            nodes.resize(w_idx);
        }
    }

    uint32_t saturate(const EGraphLimits& limits = {}, EGraphStats* stats = nullptr);

    uint32_t num_enodes() const noexcept { return total_enodes_; }
    uint32_t num_eclasses() const noexcept { return static_cast<uint32_t>(classes_.size()); }
    uint32_t total_merges() const noexcept { return total_merges_; }

    bool contains_node(const ENode& raw_node) const noexcept {
        return memo_.find(canonicalize(raw_node)) != memo_.end();
    }

    Value* class_value(EClassId id) const noexcept {
        const EClassId cid = find(id);
        return (cid < classes_.size()) ? classes_[cid].ssa_val : nullptr;
    }

    uint32_t class_width(EClassId id) const noexcept {
        const EClassId cid = find(id);
        return (cid < classes_.size()) ? classes_[cid].width : 0;
    }

    bool is_const(EClassId id, uint64_t& out_imm) const noexcept {
        const EClassId cid = find(id);
        if (cid < classes_.size() && classes_[cid].has_const) {
            out_imm = classes_[cid].const_imm;
            return true;
        }
        return false;
    }

    std::vector<ENode> class_nodes(EClassId id) const {
        const EClassId cid = find(id);
        return (cid < classes_.size()) ? classes_[cid].nodes : std::vector<ENode>{};
    }

    const std::vector<EClass>& classes() const noexcept { return classes_; }

private:
    std::vector<EClass> classes_;
    mutable std::vector<EClassId> parents_uf_;
    std::unordered_map<ENode, EClassId, ENodeHasher> memo_;
    std::unordered_map<uint32_t, Value*> var_ssa_;
    std::vector<EClassId> worklist_;
    uint32_t total_enodes_ = 0;
    uint32_t total_merges_ = 0;
    uint32_t next_ssa_order_ = 0;
};

struct ExtractedExpr {
    EClassId eclass = kInvalidEClass;
    uint32_t cost = UINT32_MAX;
    ENode node{};
    Value* leaf_value = nullptr;
    std::vector<ExtractedExpr> children;
};

bool extract_min_cost(const EGraph& eg, EClassId root, ExtractedExpr& out);

bool run_mir_egraph_superopt(
    Function& fn,
    Module& mod,
    const SuperoptOptions& opts,
    const EGraphLimits& limits = {});

} // namespace brass
