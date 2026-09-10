#pragma once

#include <brass/mir/function.hpp>
#include <brass/mir/block.hpp>
#include <brass/mir/instruction.hpp>
#include <brass/embedding/nanbox.hpp>
#include <vector>
#include <unordered_map>
#include <cstdint>
#include <cstddef>
#include <string>

namespace brass {

struct OsrSlotValue {
    uint32_t slot_idx = 0;
    uint32_t padding = 0;
    HostValue val{};

    constexpr OsrSlotValue() noexcept = default;
    constexpr OsrSlotValue(uint32_t slot, HostValue v) noexcept : slot_idx(slot), padding(0), val(v) {}
    constexpr OsrSlotValue(uint32_t slot, uint64_t raw) noexcept : slot_idx(slot), padding(0), val(HostValue::from_raw(raw)) {}
};

// ABI-compatible memory descriptor holding count and array of (slot_idx, HostValue)
struct OsrMigrationFrame {
    static constexpr size_t kMaxInlineSlots = 64;
    uint32_t count = 0;
    uint32_t loop_header_id = 0;
    OsrSlotValue slots[kMaxInlineSlots]{};

    constexpr OsrMigrationFrame() noexcept = default;

    void add_slot(uint32_t slot_idx, HostValue val) noexcept {
        if (count < kMaxInlineSlots) {
            slots[count++] = OsrSlotValue(slot_idx, val);
        }
    }

    void add_slot(uint32_t slot_idx, uint64_t raw) noexcept {
        add_slot(slot_idx, HostValue::from_raw(raw));
    }

    HostValue get_value(uint32_t slot_idx) const noexcept {
        for (size_t i = 0; i < count; ++i) {
            if (slots[i].slot_idx == slot_idx) return slots[i].val;
        }
        return HostValue::undefined_val();
    }

    uint64_t get_raw_value(uint32_t slot_idx) const noexcept {
        return get_value(slot_idx).raw();
    }
};

struct OsrLiveIn {
    Value* val = nullptr;
    uint32_t slot_index = 0;
    bool is_loop_param = false;
    uint32_t loop_param_index = 0;
};

// Metadata identifying candidate loop header and its live-in SSA mapping
struct OsrTarget {
    BasicBlock* loop_header = nullptr;
    uint32_t loop_header_id = 0;
    std::vector<Value*> live_ins;
    std::vector<OsrLiveIn> live_in_details;
    std::unordered_map<const Value*, uint32_t> val_to_slot;

    bool is_valid() const noexcept { return loop_header != nullptr; }
    size_t live_in_count() const noexcept { return live_ins.size(); }
    uint32_t get_slot_index(const Value* v) const noexcept {
        auto it = val_to_slot.find(v);
        return (it != val_to_slot.end()) ? it->second : UINT32_MAX;
    }
};

// FunctionMetadata holding compiled function metadata including OSR entry offset
struct FunctionMetadata {
    std::string name;
    size_t text_offset = 0;
    size_t text_size = 0;
    size_t osr_entry_offset = 0;
    bool has_osr = false;
    uint32_t osr_loop_header_id = 0;
};

// Analysis to find candidate loop headers and compute live-ins entering the loop
OsrTarget analyze_osr_target(Function& fn, BasicBlock* loop_header);
std::vector<OsrTarget> find_all_osr_targets(Function& fn);

} // namespace brass
