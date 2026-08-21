#pragma once

#include <brass/codegen/lir.hpp>
#include <brass/target/x64/x64_registers.hpp>
#include <cstdint>
#include <cstddef>
#include <string>
#include <string_view>
#include <vector>
#include <span>
#include <iosfwd>

namespace brass {

enum class StackMapRootKind : uint8_t {
    FrameSlot = 0,
    CalleeSavedReg = 1
};

std::string_view to_string(StackMapRootKind kind) noexcept;
std::ostream& operator<<(std::ostream& os, StackMapRootKind kind);

struct StackMapRootLocation {
    StackMapRootKind kind = StackMapRootKind::FrameSlot;
    int32_t offset_from_rbp = 0; // Relative to RBP (e.g. -24)
    codegen::PReg reg;           // Callee-saved physical register (if applicable)

    static StackMapRootLocation frame_slot(int32_t offset) noexcept {
        StackMapRootLocation loc;
        loc.kind = StackMapRootKind::FrameSlot;
        loc.offset_from_rbp = offset;
        loc.reg = codegen::PReg();
        return loc;
    }

    static StackMapRootLocation callee_saved(int32_t offset, codegen::PReg r) noexcept {
        StackMapRootLocation loc;
        loc.kind = StackMapRootKind::CalleeSavedReg;
        loc.offset_from_rbp = offset;
        loc.reg = r;
        return loc;
    }

    bool operator==(const StackMapRootLocation& other) const noexcept = default;
};

std::ostream& operator<<(std::ostream& os, const StackMapRootLocation& loc);

struct StackMapRecord {
    uint32_t instruction_offset = 0; // Return address offset from function entry
    uint32_t frame_size = 0;         // Total frame size in bytes
    std::vector<StackMapRootLocation> roots;
    uint32_t safepoint_id = 0;

    void add_root(StackMapRootLocation loc) {
        for (const auto& r : roots) {
            if (r == loc) return;
        }
        roots.push_back(loc);
    }

    bool operator==(const StackMapRecord& other) const noexcept = default;
};

std::ostream& operator<<(std::ostream& os, const StackMapRecord& rec);

struct FunctionStackMap {
    std::string function_name;
    uintptr_t function_address = 0; // Absolute runtime code address (or 0)
    uint32_t code_offset = 0;       // Offset in .text section
    uint32_t code_size = 0;         // Function size in bytes
    std::vector<StackMapRecord> records;

    const StackMapRecord* find_record_by_offset(uint32_t call_offset) const noexcept;
    const StackMapRecord* find_record_by_ip(uintptr_t return_ip) const noexcept;

    void add_record(StackMapRecord rec) {
        records.push_back(std::move(rec));
    }
};

class ModuleStackMap {
public:
    ModuleStackMap() = default;

    void add_function(FunctionStackMap fn_map);
    const FunctionStackMap* find_function_by_name(std::string_view name) const noexcept;
    const FunctionStackMap* find_function_by_ip(uintptr_t ip) const noexcept;
    const StackMapRecord* find_record(uintptr_t return_ip) const noexcept;

    void relocate(uintptr_t text_base_address) noexcept;
    void register_function_address(std::string_view name, uintptr_t address, uint32_t code_size = 0) noexcept;

    const std::vector<FunctionStackMap>& functions() const noexcept { return functions_; }
    std::vector<FunctionStackMap>& functions() noexcept { return functions_; }
    size_t size() const noexcept { return functions_.size(); }
    bool empty() const noexcept { return functions_.empty(); }
    void clear() noexcept { functions_.clear(); }

private:
    std::vector<FunctionStackMap> functions_;
};

// Binary serialization format
// Magic: 0x4D435342 ("BSCM")
// Version: 1
constexpr uint32_t STACK_MAP_MAGIC = 0x4D435342;
constexpr uint32_t STACK_MAP_VERSION = 1;

std::vector<uint8_t> encode_stack_maps(const ModuleStackMap& stack_maps);
ModuleStackMap decode_stack_maps(std::span<const uint8_t> data, uintptr_t text_base_address = 0);
bool decode_stack_maps_into(std::span<const uint8_t> data, ModuleStackMap& out, uintptr_t text_base_address = 0);

} // namespace brass
