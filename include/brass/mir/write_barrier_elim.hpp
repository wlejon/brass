#pragma once

#include <brass/mir/module.hpp>
#include <brass/mir/function.hpp>
#include <brass/mir/block.hpp>
#include <brass/mir/instruction.hpp>
#include <atomic>
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

    WbeStats& operator+=(const WbeStats& o) noexcept {
        total_barriers += o.total_barriers;
        eliminated_non_pointer += o.eliminated_non_pointer;
        eliminated_young_provenance += o.eliminated_young_provenance;
        eliminated_redundant += o.eliminated_redundant;
        remaining_barriers += o.remaining_barriers;
        return *this;
    }

    void dump(std::ostream& os) const;
};

class WriteBarrierElimination {
public:
    explicit WriteBarrierElimination(bool dump_stats = false) noexcept
        : dump_stats_(dump_stats) {}

    bool run_on_function(Function& fn);
    bool run_on_module(Module& mod);

    [[nodiscard]] WbeStats stats() const noexcept {
        return WbeStats{
            total_barriers_.load(std::memory_order_relaxed),
            eliminated_non_pointer_.load(std::memory_order_relaxed),
            eliminated_young_provenance_.load(std::memory_order_relaxed),
            eliminated_redundant_.load(std::memory_order_relaxed),
            remaining_barriers_.load(std::memory_order_relaxed),
        };
    }
    void reset_stats() noexcept {
        total_barriers_.store(0, std::memory_order_relaxed);
        eliminated_non_pointer_.store(0, std::memory_order_relaxed);
        eliminated_young_provenance_.store(0, std::memory_order_relaxed);
        eliminated_redundant_.store(0, std::memory_order_relaxed);
        remaining_barriers_.store(0, std::memory_order_relaxed);
    }
    void dump_stats(std::ostream& os) const;

    bool is_non_pointer_value(const Value* val) const noexcept;

private:
    bool is_allocation_inst(const Instruction* inst) const noexcept;
    // The payload bytes of a young allocation `inst` makes, or -1 when it
    // is none.
    int64_t young_allocation_bytes(const Instruction* inst) const noexcept;

    bool dump_stats_ = false;
    std::atomic<uint32_t> total_barriers_{0};
    std::atomic<uint32_t> eliminated_non_pointer_{0};
    std::atomic<uint32_t> eliminated_young_provenance_{0};
    std::atomic<uint32_t> eliminated_redundant_{0};
    std::atomic<uint32_t> remaining_barriers_{0};
};

} // namespace brass
