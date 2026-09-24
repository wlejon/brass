#include "bytecode_compiler_impl.hpp"
#include <brass/mir/module.hpp>
#include <algorithm>
#include <cstring>
#include <stdexcept>

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

namespace {

// Use counts of every value, for compare-branch fusion.
std::unordered_map<const Value*, uint32_t> count_uses(const detail::BlockLayout& layout) {
    std::unordered_map<const Value*, uint32_t> uses;
    for (const BasicBlock* bb : layout.order) {
        for (const Instruction* inst : *bb) {
            if (!inst) continue;
            for (size_t i = 0; i < inst->operand_count(); ++i) {
                if (inst->operand(i)) ++uses[inst->operand(i)];
            }
            for (const Value* v : inst->state_map()) {
                if (v) ++uses[v];
            }
            for (const BranchTarget* t : detail::branch_targets_of(*inst)) {
                for (const Value* v : t->args) {
                    if (v) ++uses[v];
                }
            }
        }
    }
    return uses;
}

} // namespace

std::unique_ptr<BytecodeFunction> BytecodeCompiler::compile(const Function& fn) {
    auto bfn = std::make_unique<BytecodeFunction>(fn.name(), fn.return_type(), fn.param_types());

    // Step 1: register allocation over the emission order.
    detail::BlockLayout layout = detail::build_block_layout(fn);
    detail::RegisterAssignment regs = detail::allocate_bytecode_registers(fn, layout);

    detail::FunctionCompilerContext ctx(fn, *bfn, layout, regs);
    const BasicBlock* entry = fn.entry_block();
    bfn->num_params = static_cast<uint32_t>(entry ? entry->param_count() : fn.param_count());
    bfn->num_ssa_values = regs.num_values;
    bfn->register_types = regs.register_types;
    ctx.scratch_reg = static_cast<BcReg>(regs.num_registers);
    ctx.scratch_reg2 = static_cast<BcReg>(regs.num_registers + 1);
    bfn->register_types.push_back(Type::i64());
    bfn->register_types.push_back(Type::i64());
    bfn->num_registers = regs.num_registers + 2;
    for (const auto& [v, r] : regs.reg) bfn->ssa_to_reg[v->id()] = r;

    // Step 2: lower the blocks in layout order.
    const auto uses = count_uses(layout);
    for (size_t bi = 0; bi < layout.order.size(); ++bi) {
        const BasicBlock* bb = layout.order[bi];
        ctx.next_block = bi + 1 < layout.order.size() ? layout.order[bi + 1] : nullptr;
        uint32_t b_pc = static_cast<uint32_t>(bfn->current_pc());
        ctx.block_pc_map[bb] = b_pc;
        bfn->pc_block_map[b_pc] = bb;
        // Falling off a block into its layout successor would be silent
        // misexecution; malformed blocks are rejected.
        if (!bb->terminator()) ctx.fail("block does not end in a terminator");

        for (const Instruction* inst : *bb) {
            if (!inst) continue;
            if (inst->is_terminator() && inst != bb->tail()) ctx.fail("terminator in the middle of a block");
            if (inst->loc().is_valid()) {
                bfn->set_line_info(static_cast<uint32_t>(bfn->current_pc()), inst->loc());
            }
            // A comparison used only by the br_if right after it becomes
            // part of that branch.
            const Instruction* next = inst->next();
            if (next && next->opcode() == Opcode::br_if && next->operand(0) == inst->result() &&
                inst->result()) {
                auto it = uses.find(inst->result());
                if (it != uses.end() && it->second == 1 && ctx.can_fuse_compare_branch(*inst)) {
                    ctx.fused_compare = inst;
                    continue;
                }
            }
            ctx.lower_instruction(*inst);
        }
        ctx.fused_compare = nullptr;
    }

    // Step 3: patch jump targets.
    for (const auto& fixup : ctx.jump_fixups) {
        auto it = ctx.block_pc_map.find(fixup.target);
        if (it == ctx.block_pc_map.end()) ctx.fail("jump target block was never placed");
        int64_t rel = static_cast<int64_t>(it->second) - static_cast<int64_t>(fixup.inst_idx);
        BytecodeWord& w = bfn->code[fixup.inst_idx];
        if (fixup.imm24) {
            if (!fits_imm24(rel)) ctx.fail("compare-branch offset out of range: " + std::to_string(rel));
            w = encode_abi(decode_op(w), decode_a(w), decode_b(w), static_cast<int32_t>(rel));
        } else {
            if (rel < INT32_MIN || rel > INT32_MAX) ctx.fail("jump offset out of range");
            w = encode_ai(decode_op(w), decode_a(w), static_cast<int32_t>(rel));
        }
    }

    for (const auto& sf : ctx.switch_fixups) {
        int64_t target_pc = sf.trampoline_pc;
        if (target_pc < 0) {
            auto it = ctx.block_pc_map.find(sf.target);
            if (it == ctx.block_pc_map.end()) ctx.fail("switch target block was never placed");
            target_pc = it->second;
        }
        SwitchTable& table = bfn->switch_tables[sf.table_idx];
        if (sf.is_default) {
            table.default_offset = static_cast<int32_t>(target_pc);
        } else {
            table.cases[sf.case_idx].second = static_cast<int32_t>(target_pc);
        }
    }
    for (SwitchTable& table : bfn->switch_tables) {
        std::stable_sort(table.cases.begin(), table.cases.end(),
                         [](const auto& a, const auto& b) { return a.first < b.first; });
    }

    for (const auto& ef : ctx.exception_fixups) {
        int64_t handler = ef.trampoline_pc;
        if (handler < 0) {
            auto it = ctx.block_pc_map.find(ef.unwind_target);
            if (it == ctx.block_pc_map.end()) ctx.fail("unwind block was never placed");
            handler = it->second;
        }
        bfn->exception_table[ef.ee_idx].handler_pc = static_cast<uint32_t>(handler);
    }

    // Step 4: resume points.
    for (const auto& [resume_id, target_bb] : fn.resume_points()) {
        auto it = ctx.block_pc_map.find(target_bb);
        if (it == ctx.block_pc_map.end()) continue;
        ResumePointEntry rpe;
        rpe.resume_id = resume_id;
        rpe.target_pc = it->second;
        for (size_t p = 0; p < target_bb->param_count(); ++p) {
            if (const Value* pv = target_bb->param(p)) rpe.param_regs.push_back(ctx.get_reg(pv));
        }
        if (const Instruction* g = fn.find_guard(resume_id)) {
            for (const Value* sv : g->state_map()) rpe.state_regs.push_back(sv ? ctx.get_reg(sv) : kNoReg);
        }
        bfn->resume_points.push_back(std::move(rpe));
    }

    return bfn;
}

namespace detail {

uint32_t FunctionCompilerContext::add_constant(uint64_t bits) {
    auto [it, inserted] = constant_index.emplace(bits, static_cast<uint32_t>(out.constants.size()));
    if (inserted) out.constants.push_back(bits);
    return it->second;
}

uint32_t FunctionCompilerContext::add_string(std::string_view s) {
    auto [it, inserted] = string_index.emplace(std::string(s), static_cast<uint32_t>(out.string_pool.size()));
    if (inserted) out.string_pool.emplace_back(s);
    return it->second;
}

void FunctionCompilerContext::emit_const64(BcReg dst, uint64_t bits) {
    int64_t v = static_cast<int64_t>(bits);
    if (v >= INT32_MIN && v <= INT32_MAX) {
        emit_ai(BytecodeOp::mov_imm, dst, static_cast<int32_t>(v));
    } else {
        emit_ai(BytecodeOp::load_const, dst, static_cast<int32_t>(add_constant(bits)));
    }
}

} // namespace detail

} // namespace brass
