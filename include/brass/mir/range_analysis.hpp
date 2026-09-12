#pragma once

#include <brass/mir/types.hpp>
#include <brass/mir/function.hpp>
#include <brass/mir/block.hpp>
#include <brass/mir/instruction.hpp>
#include <brass/mir/dominators.hpp>
#include <brass/mir/loop_analysis.hpp>
#include <cstdint>
#include <algorithm>
#include <iosfwd>
#include <string>
#include <unordered_map>
#include <vector>

namespace brass {

struct ValueRange {
    int64_t min_val = INT64_MIN;
    int64_t max_val = INT64_MAX;

    constexpr ValueRange() noexcept = default;
    constexpr ValueRange(int64_t min_v, int64_t max_v) noexcept : min_val(min_v), max_val(max_v) {}

    [[nodiscard]] constexpr bool is_constant() const noexcept { return min_val == max_val; }
    [[nodiscard]] constexpr bool is_non_negative() const noexcept { return min_val >= 0; }
    [[nodiscard]] constexpr bool is_empty() const noexcept { return min_val > max_val; }
    [[nodiscard]] constexpr bool is_full() const noexcept { return min_val == INT64_MIN && max_val == INT64_MAX; }
    [[nodiscard]] constexpr bool contains(int64_t val) const noexcept { return !is_empty() && val >= min_val && val <= max_val; }

    void intersect_with(const ValueRange& other) noexcept {
        if (is_empty()) return;
        if (other.is_empty()) {
            min_val = 1;
            max_val = 0;
            return;
        }
        min_val = std::max(min_val, other.min_val);
        max_val = std::min(max_val, other.max_val);
    }

    void union_with(const ValueRange& other) noexcept {
        if (other.is_empty()) return;
        if (is_empty()) {
            *this = other;
            return;
        }
        min_val = std::min(min_val, other.min_val);
        max_val = std::max(max_val, other.max_val);
    }

    [[nodiscard]] bool is_subrange_of(const ValueRange& other) const noexcept {
        if (is_empty()) return true;
        if (other.is_empty()) return false;
        return min_val >= other.min_val && max_val <= other.max_val;
    }

    static constexpr ValueRange constant(int64_t val) noexcept {
        return ValueRange(val, val);
    }

    static constexpr ValueRange range(int64_t min_v, int64_t max_v) noexcept {
        return ValueRange(min_v, max_v);
    }

    static constexpr ValueRange non_negative() noexcept {
        return ValueRange(0, INT64_MAX);
    }

    static constexpr ValueRange full() noexcept {
        return ValueRange(INT64_MIN, INT64_MAX);
    }

    static constexpr ValueRange empty() noexcept {
        return ValueRange(1, 0);
    }

    bool operator==(const ValueRange& other) const noexcept {
        if (is_empty() && other.is_empty()) return true;
        return min_val == other.min_val && max_val == other.max_val;
    }

    bool operator!=(const ValueRange& other) const noexcept {
        return !(*this == other);
    }

    std::string to_string() const;

    // Static interval arithmetic helpers
    static ValueRange add(const ValueRange& a, const ValueRange& b) noexcept;
    static ValueRange sub(const ValueRange& a, const ValueRange& b) noexcept;
    static ValueRange mul(const ValueRange& a, const ValueRange& b) noexcept;
    static ValueRange and_(const ValueRange& a, const ValueRange& b) noexcept;
    static ValueRange or_(const ValueRange& a, const ValueRange& b) noexcept;
    static ValueRange xor_(const ValueRange& a, const ValueRange& b) noexcept;
    static ValueRange shl(const ValueRange& a, const ValueRange& b) noexcept;
    static ValueRange lshr(const ValueRange& a, const ValueRange& b) noexcept;
    static ValueRange ashr(const ValueRange& a, const ValueRange& b) noexcept;
    static ValueRange zext(const ValueRange& a) noexcept;
    static ValueRange sext(const ValueRange& a) noexcept;
    static ValueRange trunc(const ValueRange& a) noexcept;
    static ValueRange select(const ValueRange& cond, const ValueRange& then_r, const ValueRange& else_r) noexcept;
};

std::ostream& operator<<(std::ostream& os, const ValueRange& r);

class RangeAnalysis {
public:
    explicit RangeAnalysis(Function& fn);
    RangeAnalysis(Function& fn, const DominatorTree& dom, const LoopAnalysis& loops);

    [[nodiscard]] ValueRange get_range(const Value* v) const;
    [[nodiscard]] ValueRange get_range_at(const Value* v, const BasicBlock* bb) const;
    [[nodiscard]] ValueRange get_range(const Value* v, const BasicBlock* bb) const {
        return bb ? get_range_at(v, bb) : get_range(v);
    }

    [[nodiscard]] const std::unordered_map<const Value*, ValueRange>& all_ranges() const noexcept {
        return global_ranges_;
    }

    void dump(std::ostream& os) const;

private:
    void run_analysis(Function& fn, const DominatorTree& dom, const LoopAnalysis& loops);
    void infer_loop_induction_variables(const LoopAnalysis& loops);
    void propagate_path_sensitive(Function& fn, const DominatorTree& dom);
    void visit_dominator_block(
        const BasicBlock* bb,
        const DominatorTree& dom,
        std::unordered_map<const Value*, ValueRange> current_ranges
    );
    void apply_branch_condition(
        const Value* cond,
        bool is_true_edge,
        std::unordered_map<const Value*, ValueRange>& ranges
    );

    ValueRange evaluate_instruction(
        const Instruction* inst,
        const std::unordered_map<const Value*, ValueRange>& context_ranges
    ) const;

    ValueRange get_context_range(
        const Value* v,
        const std::unordered_map<const Value*, ValueRange>& context_ranges
    ) const;

    std::unordered_map<const Value*, ValueRange> global_ranges_;
    std::unordered_map<const BasicBlock*, std::unordered_map<const Value*, ValueRange>> block_ranges_;
};

} // namespace brass
