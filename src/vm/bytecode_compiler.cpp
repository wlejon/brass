#include "bytecode_compiler_impl.hpp"
#include <brass/mir/module.hpp>
#include <stdexcept>
#include <iostream>

namespace brass {

std::unique_ptr<BytecodeModule> BytecodeCompiler::compile(const Module& mod) {
    auto bmod = std::make_unique<BytecodeModule>(mod.name());
    for (const auto& fn : mod.functions()) {
        if (fn) {
            auto bfn = compile(*fn);
            bmod->add_function(std::move(bfn));
        }
    }
    return bmod;
}

std::unique_ptr<BytecodeFunction> BytecodeCompiler::compile(const Function& fn) {
    auto bfn = std::make_unique<BytecodeFunction>(fn.name(), fn.return_type(), fn.param_types());
    detail::FunctionCompilerContext ctx(fn, *bfn);

    // Step 1: Assign Register Slots
    const BasicBlock* entry = fn.entry_block();
    uint32_t next_reg = 0;

    if (entry) {
        bfn->num_params = static_cast<uint32_t>(entry->param_count());
        for (size_t i = 0; i < entry->param_count(); ++i) {
            const Value* p = entry->param(i);
            if (p) {
                uint8_t r = static_cast<uint8_t>(next_reg++);
                ctx.reg_map[p->id()] = r;
                bfn->ssa_to_reg[p->id()] = r;
            }
        }
    } else {
        bfn->num_params = static_cast<uint32_t>(fn.param_count());
        next_reg = bfn->num_params;
    }

    // Allocate for non-entry block parameters
    for (const auto* bb : fn.blocks()) {
        if (!bb || bb == entry) continue;
        for (size_t i = 0; i < bb->param_count(); ++i) {
            const Value* p = bb->param(i);
            if (p) {
                uint8_t r = static_cast<uint8_t>(next_reg++);
                ctx.reg_map[p->id()] = r;
                bfn->ssa_to_reg[p->id()] = r;
            }
        }
    }

    // Allocate for instruction results
    for (const auto* bb : fn.blocks()) {
        if (!bb) continue;
        for (const auto* inst : *bb) {
            if (inst && inst->produces_value() && inst->result()) {
                uint8_t r = static_cast<uint8_t>(next_reg++);
                ctx.reg_map[inst->result()->id()] = r;
                bfn->ssa_to_reg[inst->result()->id()] = r;
            }
        }
    }

    // Reserve scratch registers for parallel copies and address calculations
    ctx.scratch_reg = static_cast<uint8_t>(next_reg++);
    ctx.scratch_reg2 = static_cast<uint8_t>(next_reg++);
    bfn->num_registers = next_reg;

    bfn->register_types.assign(next_reg, Type::i64());
    if (entry) {
        for (size_t i = 0; i < entry->param_count(); ++i) {
            const Value* p = entry->param(i);
            if (p) bfn->register_types[ctx.get_reg(p)] = p->type();
        }
    }
    for (const auto* bb : fn.blocks()) {
        if (!bb || bb == entry) continue;
        for (size_t i = 0; i < bb->param_count(); ++i) {
            const Value* p = bb->param(i);
            if (p) bfn->register_types[ctx.get_reg(p)] = p->type();
        }
    }
    for (const auto* bb : fn.blocks()) {
        if (!bb) continue;
        for (const auto* inst : *bb) {
            if (inst && inst->produces_value() && inst->result()) {
                bfn->register_types[ctx.get_reg(inst->result())] = inst->result()->type();
            }
        }
    }

    if (next_reg > 256) {
        throw std::runtime_error("Function @" + std::string(fn.name()) +
                                 " exceeds maximum 256 virtual registers (required " +
                                 std::to_string(next_reg) + ")");
    }

    // Step 2: Linearize Blocks & Lower Instructions
    for (const auto* bb : fn.blocks()) {
        if (!bb) continue;
        uint32_t b_pc = static_cast<uint32_t>(bfn->current_pc());
        ctx.block_pc_map[bb] = b_pc;
        bfn->pc_block_map[b_pc] = bb;

        for (const auto* inst : *bb) {
            if (!inst) continue;
            if (inst->loc().is_valid()) {
                bfn->set_line_info(static_cast<uint32_t>(bfn->current_pc()), inst->loc());
            }
            ctx.lower_instruction(*inst);
        }
    }

    // Step 3: Backpatch Relative Jump Offsets
    for (const auto& fixup : ctx.jump_fixups) {
        auto it = ctx.block_pc_map.find(fixup.target);
        if (it == ctx.block_pc_map.end()) {
            throw std::runtime_error("Jump fixup target block not found in PC map");
        }
        uint32_t target_pc = it->second;
        int32_t rel_offset = static_cast<int32_t>(target_pc) - static_cast<int32_t>(fixup.inst_idx);
        if (rel_offset < -32768 || rel_offset > 32767) {
            throw std::runtime_error("Jump relative offset out of range: " + std::to_string(rel_offset));
        }
        bfn->code[fixup.inst_idx] = encode_ad(fixup.op, fixup.cond_reg, static_cast<int16_t>(rel_offset));
    }

    // Backpatch Switch Tables
    for (const auto& sfixup : ctx.switch_fixups) {
        auto it = ctx.block_pc_map.find(sfixup.target);
        if (it == ctx.block_pc_map.end()) {
            throw std::runtime_error("Switch fixup target block not found in PC map");
        }
        uint32_t target_pc = it->second;
        // The table stores target_pc directly or relative to switch inst
        if (sfixup.is_default) {
            bfn->switch_tables[sfixup.table_idx].default_offset = static_cast<int32_t>(target_pc);
        } else {
            bfn->switch_tables[sfixup.table_idx].cases[sfixup.case_idx].second = static_cast<int32_t>(target_pc);
        }
    }

    // Backpatch Exception Table Handlers
    for (const auto& ef : ctx.exception_fixups) {
        auto it = ctx.block_pc_map.find(ef.unwind_target);
        if (it != ctx.block_pc_map.end()) {
            bfn->exception_table[ef.ee_idx].handler_pc = it->second;
        }
    }

    // Step 4: Resume Points Table
    for (const auto& [resume_id, target_bb] : fn.resume_points()) {
        auto it = ctx.block_pc_map.find(target_bb);
        if (it != ctx.block_pc_map.end()) {
            ResumePointEntry rpe;
            rpe.resume_id = resume_id;
            rpe.target_pc = it->second;
            for (size_t p = 0; p < target_bb->param_count(); ++p) {
                const Value* pv = target_bb->param(p);
                if (pv) {
                    rpe.param_regs.push_back(ctx.get_reg(pv));
                }
            }
            bfn->resume_points.push_back(std::move(rpe));
        }
    }

    return bfn;
}

namespace detail {

void FunctionCompilerContext::emit_parallel_moves(const BranchTarget& target) {
    if (target.args.empty()) return;
    const BasicBlock* target_bb = target.block;
    if (!target_bb) return;

    struct Move {
        uint8_t src;
        uint8_t dst;
    };
    std::vector<Move> moves;
    moves.reserve(target.args.size());

    for (size_t i = 0; i < target.args.size() && i < target_bb->param_count(); ++i) {
        const Value* arg_val = target.args[i];
        const Value* param_val = target_bb->param(i);
        if (!arg_val || !param_val) continue;

        uint8_t src = get_reg(arg_val);
        uint8_t dst = get_reg(param_val);
        if (src != dst) {
            moves.push_back({src, dst});
        }
    }

    while (!moves.empty()) {
        // Step 1: Find a move whose dst is not a src of any other pending move
        int free_idx = -1;
        for (size_t i = 0; i < moves.size(); ++i) {
            bool dst_is_src = false;
            for (size_t j = 0; j < moves.size(); ++j) {
                if (i != j && moves[i].dst == moves[j].src) {
                    dst_is_src = true;
                    break;
                }
            }
            if (!dst_is_src) {
                free_idx = static_cast<int>(i);
                break;
            }
        }

        if (free_idx != -1) {
            emit(BytecodeOp::mov, moves[free_idx].dst, moves[free_idx].src, 0);
            moves.erase(moves.begin() + free_idx);
        } else {
            // Cycle detected! Break it with scratch_reg
            Move m = moves[0];
            emit(BytecodeOp::mov, scratch_reg, m.src, 0);
            for (auto& pending : moves) {
                if (pending.src == m.src) {
                    pending.src = scratch_reg;
                }
            }
        }
    }
}

} // namespace detail

} // namespace brass
