#pragma once

#include <brass/mir/function.hpp>
#include <brass/mir/block.hpp>
#include <brass/pgo/profile_data.hpp>
#include <unordered_map>
#include <memory>
#include <cstdint>

namespace brass::mir {

class BlockFrequencyInfo {
public:
    BlockFrequencyInfo() = default;

    uint64_t entry_count() const noexcept { return entry_count_; }
    void set_entry_count(uint64_t count) noexcept { entry_count_ = count; }

    uint64_t get_block_count(const BasicBlock* bb) const;
    void set_block_count(const BasicBlock* bb, uint64_t count);

    double get_block_frequency(const BasicBlock* bb) const; // normalized to entry count (1.0 = entry)

    bool is_hot_block(const BasicBlock* bb, double threshold = 0.10) const;
    bool is_cold_block(const BasicBlock* bb, double threshold = 0.001) const;

private:
    uint64_t entry_count_ = 0;
    std::unordered_map<const BasicBlock*, uint64_t> block_counts_;
};

class BranchProbabilityInfo {
public:
    BranchProbabilityInfo() = default;

    double get_edge_probability(const BasicBlock* src, const BasicBlock* dst) const; // 0.0 .. 1.0
    uint64_t get_edge_count(const BasicBlock* src, const BasicBlock* dst) const;

    void set_edge_info(const BasicBlock* src, const BasicBlock* dst, uint64_t count, double prob);

private:
    struct EdgeKey {
        const BasicBlock* src = nullptr;
        const BasicBlock* dst = nullptr;

        bool operator==(const EdgeKey& o) const noexcept {
            return src == o.src && dst == o.dst;
        }
    };

    struct EdgeKeyHash {
        size_t operator()(const EdgeKey& k) const noexcept {
            return std::hash<const BasicBlock*>()(k.src) ^ (std::hash<const BasicBlock*>()(k.dst) << 1);
        }
    };

    struct EdgeData {
        uint64_t count = 0;
        double probability = 0.0;
    };

    std::unordered_map<EdgeKey, EdgeData, EdgeKeyHash> edges_;
};

class BranchProbabilityAnalysis {
public:
    BranchProbabilityAnalysis() = default;
    BranchProbabilityAnalysis(const Function& fn, const pgo::FunctionProfile& prof);

    const BlockFrequencyInfo& block_frequency_info() const noexcept { return bfi_; }
    const BranchProbabilityInfo& branch_probability_info() const noexcept { return bpi_; }

    static BranchProbabilityAnalysis run(const Function& fn, const pgo::FunctionProfile& prof) {
        return BranchProbabilityAnalysis(fn, prof);
    }

private:
    BlockFrequencyInfo bfi_;
    BranchProbabilityInfo bpi_;

    void solve_flows(const Function& fn, const pgo::FunctionProfile& prof);
};

} // namespace brass::mir
