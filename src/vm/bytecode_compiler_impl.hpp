#pragma once

#include <brass/vm/bytecode.hpp>
#include <brass/vm/bytecode_compiler.hpp>
#include <brass/mir/function.hpp>
#include <brass/mir/block.hpp>
#include <brass/mir/instruction.hpp>
#include <unordered_map>
#include <vector>
#include <stdexcept>

namespace brass::detail {

struct JumpFixup {
    size_t inst_idx = 0;
    const BasicBlock* target = nullptr;
    uint8_t cond_reg = 0;
    BytecodeOp op = BytecodeOp::jump;
};

struct SwitchFixup {
    size_t table_idx = 0;
    size_t case_idx = 0;
    const BasicBlock* target = nullptr;
    bool is_default = false;
};

struct ExceptionFixup {
    size_t ee_idx = 0;
    const BasicBlock* unwind_target = nullptr;
};

class FunctionCompilerContext {
public:
    const Function& fn;
    BytecodeFunction& out;
    std::unordered_map<uint32_t, uint8_t> reg_map;
    std::unordered_map<const BasicBlock*, uint32_t> block_pc_map;
    std::vector<JumpFixup> jump_fixups;
    std::vector<SwitchFixup> switch_fixups;
    std::vector<ExceptionFixup> exception_fixups;

    uint8_t scratch_reg = 0;
    uint8_t scratch_reg2 = 0;

    FunctionCompilerContext(const Function& f, BytecodeFunction& o)
        : fn(f), out(o) {}

    uint8_t get_reg(const Value* v) const {
        if (!v) {
            throw std::runtime_error("Attempted to get register for null Value");
        }
        auto it = reg_map.find(v->id());
        if (it == reg_map.end()) {
            throw std::runtime_error("Value id " + std::to_string(v->id()) + " has no assigned register");
        }
        return it->second;
    }

    uint8_t get_result_reg(const Instruction& inst) const {
        if (!inst.result()) {
            throw std::runtime_error("Instruction produces no result");
        }
        return get_reg(inst.result());
    }

    void emit(BytecodeOp op, uint8_t dst, uint8_t src1, uint8_t src2) {
        out.emit(op, dst, src1, src2);
    }

    void emit_ad(BytecodeOp op, uint8_t dst, uint16_t imm16) {
        out.emit(op, dst, imm16);
    }

    void emit_ad(BytecodeOp op, uint8_t dst, int16_t imm16) {
        out.emit(op, dst, imm16);
    }

    void emit_parallel_moves(const BranchTarget& target);

    // Defined in bytecode_compiler_ops.cpp
    void lower_instruction(const Instruction& inst);
    void lower_constant(const Instruction& inst);
    void lower_conversion(const Instruction& inst);
    void lower_arithmetic(const Instruction& inst);
    void lower_bitwise(const Instruction& inst);
    void lower_comparison(const Instruction& inst);
    void lower_memory(const Instruction& inst);
    void lower_call(const Instruction& inst);
    void lower_terminator(const Instruction& inst);
    void lower_runtime_gc(const Instruction& inst);
    void lower_coroutine(const Instruction& inst);
    void lower_vector(const Instruction& inst);
};

} // namespace brass::detail
