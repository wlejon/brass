#pragma once

#include <brass/mir/function.hpp>
#include <brass/mir/block.hpp>
#include <brass/mir/loop_analysis.hpp>
#include <brass/mir/dominators.hpp>
#include <brass/mir/instruction.hpp>
#include <vector>
#include <memory>
#include <cstdint>

namespace brass {

struct LoopNestLevel {
    LoopInfo* loop = nullptr;
    BasicBlock* header = nullptr;
    BasicBlock* preheader = nullptr;
    BasicBlock* body = nullptr;
    BasicBlock* latch = nullptr;
    BasicBlock* exit_bb = nullptr;
    std::vector<BasicBlock*> blocks;

    size_t iv_param_index = 0;
    Value* iv_param = nullptr;
    Type iv_type = Type::i64();
    Value* init_val = nullptr;
    Value* limit_val = nullptr;
    int64_t step = 1;
    Opcode cmp_opcode = Opcode::slt;
    bool exit_on_false = true;
};

enum class DependenceDirection : uint8_t {
    Equal,       // distance == 0
    Forward,     // distance > 0
    Backward,    // distance < 0
    Any          // variable / unknown
};

struct DependenceVector {
    std::vector<DependenceDirection> directions;
    std::vector<int64_t> distances;
    bool has_distance = false;
    bool is_loop_independent = false;
};

struct AccessIndexTerm {
    size_t level_index = 0;
    int64_t const_stride = 1;
    Value* symbolic_stride = nullptr;
};

struct NestMemoryAccess {
    Instruction* inst = nullptr;
    bool is_store = false;
    Type elem_type = Type::void_type();
    Value* base = nullptr;
    std::vector<AccessIndexTerm> terms;
    int32_t const_offset = 0;
    uint8_t scale = 1;
};

class LoopNest {
public:
    LoopNest() = default;

    size_t depth() const noexcept { return levels_.size(); }
    const std::vector<LoopNestLevel>& levels() const noexcept { return levels_; }
    std::vector<LoopNestLevel>& levels() noexcept { return levels_; }

    const LoopNestLevel& level(size_t idx) const { return levels_[idx]; }
    LoopNestLevel& level(size_t idx) { return levels_[idx]; }

    const std::vector<NestMemoryAccess>& memory_accesses() const noexcept { return accesses_; }
    std::vector<NestMemoryAccess>& memory_accesses() noexcept { return accesses_; }

    const std::vector<DependenceVector>& dependences() const noexcept { return dependences_; }
    std::vector<DependenceVector>& dependences() noexcept { return dependences_; }

    bool is_tileable() const noexcept;
    bool is_interchange_legal(size_t level_a, size_t level_b) const noexcept;
    bool is_matrix_multiply() const noexcept;

    // Matmul reduction details
    bool has_reduction() const noexcept { return has_reduction_; }
    Value* reduction_init() const noexcept { return reduction_init_; }
    Instruction* reduction_inst() const noexcept { return reduction_inst_; }
    Instruction* reduction_store() const noexcept { return reduction_store_; }
    BasicBlock* reduction_store_block() const noexcept { return reduction_store_block_; }

    void set_reduction(Value* init, Instruction* inst, Instruction* store, BasicBlock* store_bb) noexcept {
        has_reduction_ = true;
        reduction_init_ = init;
        reduction_inst_ = inst;
        reduction_store_ = store;
        reduction_store_block_ = store_bb;
    }

private:
    std::vector<LoopNestLevel> levels_;
    std::vector<NestMemoryAccess> accesses_;
    std::vector<DependenceVector> dependences_;
    bool has_reduction_ = false;
    Value* reduction_init_ = nullptr;
    Instruction* reduction_inst_ = nullptr;
    Instruction* reduction_store_ = nullptr;
    BasicBlock* reduction_store_block_ = nullptr;
};

class LoopNestAnalysis {
public:
    explicit LoopNestAnalysis(Function& fn, const DominatorTree& dom);

    const std::vector<std::unique_ptr<LoopNest>>& nests() const noexcept { return nests_; }

    static std::unique_ptr<LoopNest> analyze_nest(Function& fn, LoopInfo& outer_loop, const DominatorTree& dom);

private:
    void discover_nests(Function& fn, const DominatorTree& dom);
    std::unique_ptr<LoopAnalysis> loop_analysis_;
    std::vector<std::unique_ptr<LoopNest>> nests_;
};

} // namespace brass
