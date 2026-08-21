#pragma once

#include <brass/mir/function.hpp>
#include <brass/mir/block.hpp>
#include <brass/mir/dominators.hpp>
#include <vector>
#include <unordered_set>
#include <memory>

namespace brass {

class LoopInfo {
public:
    explicit LoopInfo(BasicBlock* header) : header_(header) {}

    BasicBlock* header() const noexcept { return header_; }
    BasicBlock* preheader() const noexcept { return preheader_; }
    void set_preheader(BasicBlock* ph) noexcept { preheader_ = ph; }

    const std::vector<BasicBlock*>& latches() const noexcept { return latches_; }
    std::vector<BasicBlock*>& latches() noexcept { return latches_; }
    void add_latch(BasicBlock* latch);

    const std::vector<BasicBlock*>& blocks() const noexcept { return blocks_; }
    std::vector<BasicBlock*>& blocks() noexcept { return blocks_; }
    void add_block(BasicBlock* bb);

    bool contains(const BasicBlock* bb) const noexcept;
    bool contains(const Instruction* inst) const noexcept;
    bool is_loop_invariant(const Value* val) const noexcept;

    LoopInfo* parent() const noexcept { return parent_; }
    void set_parent(LoopInfo* p) noexcept { parent_ = p; }

    const std::vector<std::unique_ptr<LoopInfo>>& sub_loops() const noexcept { return sub_loops_; }
    std::vector<std::unique_ptr<LoopInfo>>& sub_loops() noexcept { return sub_loops_; }
    void add_sub_loop(std::unique_ptr<LoopInfo> sub) {
        sub->set_parent(this);
        sub_loops_.push_back(std::move(sub));
    }

    size_t depth() const noexcept;

private:
    BasicBlock* header_ = nullptr;
    BasicBlock* preheader_ = nullptr;
    std::vector<BasicBlock*> latches_;
    std::vector<BasicBlock*> blocks_;
    std::unordered_set<const BasicBlock*> block_set_;
    LoopInfo* parent_ = nullptr;
    std::vector<std::unique_ptr<LoopInfo>> sub_loops_;
};

class LoopAnalysis {
public:
    explicit LoopAnalysis(Function& fn, const DominatorTree& dom);

    const std::vector<std::unique_ptr<LoopInfo>>& top_level_loops() const noexcept { return top_level_loops_; }
    
    // Returns all loops in post-order (innermost loops first)
    std::vector<LoopInfo*> post_order_loops() const;

    // Helper to ensure loop has a dedicated preheader
    static BasicBlock* ensure_preheader(Function& fn, LoopInfo& loop);

private:
    void discover_loops(Function& fn, const DominatorTree& dom);

    std::vector<std::unique_ptr<LoopInfo>> top_level_loops_;
};

} // namespace brass
