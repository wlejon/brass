#pragma once

#include <brass/mir/function.hpp>
#include <brass/mir/block.hpp>
#include <brass/mir/instruction.hpp>
#include <brass/mir/dominators.hpp>
#include <brass/mir/alias_analysis.hpp>
#include <cstdint>
#include <memory>
#include <vector>
#include <unordered_map>
#include <string_view>
#include <iosfwd>

namespace brass {

enum class MemoryAccessKind : uint8_t {
    LiveOnEntry = 0,
    Use = 1,
    Def = 2,
    Phi = 3
};

std::string_view memory_access_kind_name(MemoryAccessKind kind) noexcept;

class MemoryUse;
class MemoryDef;
class MemoryPhi;

class MemoryAccess {
public:
    virtual ~MemoryAccess() = default;

    uint32_t id() const noexcept { return id_; }
    MemoryAccessKind kind() const noexcept { return kind_; }
    BasicBlock* block() const noexcept { return block_; }
    Instruction* origin_instruction() const noexcept { return origin_inst_; }

    bool is_live_on_entry() const noexcept { return kind_ == MemoryAccessKind::LiveOnEntry; }
    bool is_use() const noexcept { return kind_ == MemoryAccessKind::Use; }
    bool is_def() const noexcept { return kind_ == MemoryAccessKind::Def; }
    bool is_phi() const noexcept { return kind_ == MemoryAccessKind::Phi; }

    MemoryUse* as_use() noexcept;
    const MemoryUse* as_use() const noexcept;
    MemoryDef* as_def() noexcept;
    const MemoryDef* as_def() const noexcept;
    MemoryPhi* as_phi() noexcept;
    const MemoryPhi* as_phi() const noexcept;

protected:
    MemoryAccess(uint32_t id, MemoryAccessKind kind, BasicBlock* bb, Instruction* inst) noexcept
        : id_(id), kind_(kind), block_(bb), origin_inst_(inst) {}

    uint32_t id_ = 0;
    MemoryAccessKind kind_ = MemoryAccessKind::LiveOnEntry;
    BasicBlock* block_ = nullptr;
    Instruction* origin_inst_ = nullptr;
};

class MemoryLiveOnEntry final : public MemoryAccess {
public:
    explicit MemoryLiveOnEntry(uint32_t id) noexcept
        : MemoryAccess(id, MemoryAccessKind::LiveOnEntry, nullptr, nullptr) {}
};

class MemoryUse final : public MemoryAccess {
public:
    MemoryUse(uint32_t id, BasicBlock* bb, Instruction* inst, MemoryAccess* defining_access) noexcept
        : MemoryAccess(id, MemoryAccessKind::Use, bb, inst), defining_access_(defining_access) {}

    MemoryAccess* defining_access() const noexcept { return defining_access_; }
    void set_defining_access(MemoryAccess* acc) noexcept { defining_access_ = acc; }

private:
    MemoryAccess* defining_access_ = nullptr;
};

class MemoryDef final : public MemoryAccess {
public:
    MemoryDef(uint32_t id, BasicBlock* bb, Instruction* inst, MemoryAccess* defining_access) noexcept
        : MemoryAccess(id, MemoryAccessKind::Def, bb, inst), defining_access_(defining_access) {}

    MemoryAccess* defining_access() const noexcept { return defining_access_; }
    void set_defining_access(MemoryAccess* acc) noexcept { defining_access_ = acc; }

private:
    MemoryAccess* defining_access_ = nullptr;
};

class MemoryPhi final : public MemoryAccess {
public:
    MemoryPhi(uint32_t id, BasicBlock* bb) noexcept
        : MemoryAccess(id, MemoryAccessKind::Phi, bb, nullptr) {}

    void add_incoming(BasicBlock* pred, MemoryAccess* access);
    MemoryAccess* get_incoming(const BasicBlock* pred) const;
    void set_incoming(BasicBlock* pred, MemoryAccess* access);

    const std::vector<std::pair<BasicBlock*, MemoryAccess*>>& incoming() const noexcept { return incoming_; }
    size_t incoming_count() const noexcept { return incoming_.size(); }

private:
    std::vector<std::pair<BasicBlock*, MemoryAccess*>> incoming_;
};

class MemorySSA {
public:
    MemorySSA(const Function& fn, const DominatorTree& dom, const AliasAnalysis& aa);
    ~MemorySSA();

    MemorySSA(const MemorySSA&) = delete;
    MemorySSA& operator=(const MemorySSA&) = delete;
    MemorySSA(MemorySSA&&) noexcept;
    MemorySSA& operator=(MemorySSA&&) noexcept;

    // Queries
    MemoryAccess* get_memory_access(const Instruction* inst) const;
    MemoryUse* get_memory_use(const Instruction* inst) const;
    MemoryDef* get_memory_def(const Instruction* inst) const;
    MemoryPhi* get_memory_phi(const BasicBlock* bb) const;

    MemoryAccess* live_on_entry() const noexcept { return live_on_entry_.get(); }
    const Function& function() const noexcept { return *fn_; }
    const DominatorTree& dom_tree() const noexcept { return *dom_; }
    const AliasAnalysis& alias_analysis() const noexcept { return *aa_; }

    // Use optimization: finds the nearest clobbering MemoryDef or MemoryPhi/LiveOnEntry
    MemoryAccess* find_clobbering_access(const Instruction* load_inst, MemoryAccess* starting_access) const;

private:
    void build();
    void compute_dominance_frontiers(std::unordered_map<const BasicBlock*, std::vector<BasicBlock*>>& df);

    const Function* fn_ = nullptr;
    const DominatorTree* dom_ = nullptr;
    const AliasAnalysis* aa_ = nullptr;

    uint32_t next_access_id_ = 1;
    std::unique_ptr<MemoryLiveOnEntry> live_on_entry_;
    std::vector<std::unique_ptr<MemoryAccess>> all_accesses_;

    std::unordered_map<const Instruction*, MemoryAccess*> inst_to_access_;
    std::unordered_map<const BasicBlock*, MemoryPhi*> block_to_phi_;
};

} // namespace brass
