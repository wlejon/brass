#pragma once

#include <brass/mir/module.hpp>
#include <brass/mir/function.hpp>
#include <brass/mir/block.hpp>
#include <brass/mir/instruction.hpp>
#include <cstdint>
#include <iosfwd>

namespace brass {

struct WbeStats {
    uint32_t total_barriers = 0;
    uint32_t eliminated_non_pointer = 0;
    uint32_t eliminated_young_provenance = 0;
    uint32_t eliminated_redundant = 0;
    uint32_t remaining_barriers = 0;

    [[nodiscard]] uint32_t total_eliminated() const noexcept {
        return eliminated_non_pointer + eliminated_young_provenance + eliminated_redundant;
    }
};

class WriteBarrierElimination {
public:
    explicit WriteBarrierElimination(bool dump_stats = false) noexcept
        : dump_stats_(dump_stats) {}

    bool run_on_function(Function& fn);
    bool run_on_module(Module& mod);

    [[nodiscard]] const WbeStats& stats() const noexcept { return stats_; }
    void reset_stats() noexcept { stats_ = WbeStats{}; }
    void dump_stats(std::ostream& os) const;

private:
    bool is_non_pointer_value(const Value* val) const noexcept;
    bool is_allocation_inst(const Instruction* inst) const noexcept;

    bool dump_stats_ = false;
    WbeStats stats_;
};

} // namespace brass
