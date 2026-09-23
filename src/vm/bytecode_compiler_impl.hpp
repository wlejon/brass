#pragma once

#include "bytecode_regalloc.hpp"
#include <brass/vm/bytecode.hpp>
#include <brass/vm/bytecode_compiler.hpp>
#include <brass/mir/function.hpp>
#include <brass/mir/block.hpp>
#include <brass/mir/instruction.hpp>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace brass::detail {

// A jump whose target pc is patched once every block has been placed.
// `imm24` jumps are the fused compare-branches (ABI24), the rest AI32.
struct JumpFixup {
    size_t inst_idx = 0;
    const BasicBlock* target = nullptr;
    bool imm24 = false;
};

struct SwitchFixup {
    size_t table_idx = 0;
    size_t case_idx = 0;
    const BasicBlock* target = nullptr;
    bool is_default = false;
    // Set when the edge carries arguments: the pc of its move trampoline.
    int64_t trampoline_pc = -1;
};

struct ExceptionFixup {
    size_t ee_idx = 0;
    const BasicBlock* unwind_target = nullptr;
    int64_t trampoline_pc = -1;
};

class FunctionCompilerContext {
public:
    const Function& fn;
    BytecodeFunction& out;
    const BlockLayout& layout;
    const RegisterAssignment& regs;
    std::unordered_map<const BasicBlock*, uint32_t> block_pc_map;
    std::vector<JumpFixup> jump_fixups;
    std::vector<SwitchFixup> switch_fixups;
    std::vector<ExceptionFixup> exception_fixups;
    std::unordered_map<uint64_t, uint32_t> constant_index;
    std::unordered_map<std::string, uint32_t> string_index;

    BcReg scratch_reg = 0;
    BcReg scratch_reg2 = 0;
    // The block laid out after the one being lowered (fall-through target).
    const BasicBlock* next_block = nullptr;

    FunctionCompilerContext(const Function& f, BytecodeFunction& o, const BlockLayout& l,
                            const RegisterAssignment& r)
        : fn(f), out(o), layout(l), regs(r) {}

    [[noreturn]] void fail(const std::string& what) const {
        throw std::runtime_error("Bytecode compiler: function @" + std::string(fn.name()) + ": " + what);
    }

    BcReg get_reg(const Value* v) const {
        if (!v) fail("operand is null");
        auto it = regs.reg.find(v);
        if (it == regs.reg.end()) fail("value %" + std::to_string(v->id()) + " has no register");
        return it->second;
    }

    // The result register, or kNoReg for an instruction without a value.
    BcReg result_reg_or_none(const Instruction& inst) const {
        return (inst.produces_value() && inst.result()) ? get_reg(inst.result()) : kNoReg;
    }

    BcReg get_result_reg(const Instruction& inst) const {
        if (!inst.produces_value() || !inst.result()) fail(std::string(opcode_name(inst.opcode())) + " has no result");
        return get_reg(inst.result());
    }

    Type reg_type(BcReg r) const { return out.register_types.at(r); }

    void emit(BytecodeOp op, uint32_t a, uint32_t b = 0, uint32_t c = 0, uint32_t d = 0) {
        out.emit(encode_abc(op, a, b, c, d));
    }
    void emit_ai(BytecodeOp op, uint32_t a, int32_t imm) { out.emit(encode_ai(op, a, imm)); }
    void emit_abi(BytecodeOp op, uint32_t a, uint32_t b, int32_t imm24) { out.emit(encode_abi(op, a, b, imm24)); }

    uint32_t add_constant(uint64_t bits);
    uint32_t add_string(std::string_view s);
    // `dst` <- the 64-bit pattern `bits`, by the shortest form.
    void emit_const64(BcReg dst, uint64_t bits);

    void emit_jump_to(const BasicBlock* target);
    void emit_parallel_moves(const BranchTarget& target);
    // Moves for `target`, then a jump to it; returns the pc of the first
    // instruction.
    uint32_t emit_edge(const BranchTarget& target);

    // Defined in bytecode_compiler_ops.cpp
    void lower_instruction(const Instruction& inst);
    void lower_constant(const Instruction& inst);
    void lower_conversion(const Instruction& inst);
    void lower_arithmetic(const Instruction& inst);
    void lower_bitwise(const Instruction& inst);
    void lower_comparison(const Instruction& inst);
    void lower_memory(const Instruction& inst);
    void lower_call(const Instruction& inst);
    void lower_runtime_gc(const Instruction& inst);
    void lower_coroutine(const Instruction& inst);
    void lower_vector(const Instruction& inst);

    // Defined in bytecode_compiler_branches.cpp
    void lower_terminator(const Instruction& inst);
    void lower_br_if(const Instruction& inst);
    void lower_switch(const Instruction& inst);
    void lower_invoke(const Instruction& inst);
    // True when the integer comparison `cmp` can be folded into the br_if
    // that follows it.
    bool can_fuse_compare_branch(const Instruction& cmp) const;

    // The comparison fused into the next br_if (its lowering is skipped).
    const Instruction* fused_compare = nullptr;
};

} // namespace brass::detail
