#pragma once

#include <brass/codegen/baseline_jit.hpp>
#include <brass/target/x64/x64_encoder.hpp>
#include <brass/target/calling_conv.hpp>
#include <unordered_map>
#include <string_view>
#include <vector>

namespace brass::codegen {

using namespace brass::x64;

struct X64BaselineEmitter {
    CodeBuffer& buffer;
    X64Encoder& enc;
    Target target;
    const Function& fn;
    const std::unordered_map<const Value*, int32_t>& slot_map;
    const std::unordered_map<const Instruction*, int32_t>& alloca_offsets;
    const std::unordered_map<uint32_t, Label>& block_labels;
    FunctionStackMap& fn_stack_map;
    BaselineSymbolResolver resolver;
    int32_t frame_size = 0;

    MemAddress slot_addr(const Value* val) const {
        auto it = slot_map.find(val);
        int32_t off = (it != slot_map.end()) ? it->second : 0;
        return MemAddress::base_disp(GPR::RBP, -off);
    }

    MemAddress slot_off_addr(int32_t off) const {
        return MemAddress::base_disp(GPR::RBP, -off);
    }

    void* resolve_sym(std::string_view name) const;
    void copy_block_args(const BranchTarget& target_branch);
};

// Returns true if the opcode was handled.
bool emit_baseline_x64_op(X64BaselineEmitter& emitter, const Instruction& inst);

} // namespace brass::codegen
