#include "ir_clone.hpp"
#include <brass/mir/module.hpp>
#include <string>

namespace brass::ir {

namespace {

std::string_view intern(Function& fn, std::string_view s) {
    if (s.empty() || !fn.parent()) return s;
    return fn.parent()->string_pool().intern(s);
}

Value* map_value(const ValueMap& values, Value* v) {
    if (!v) return nullptr;
    auto it = values.find(v);
    return it != values.end() ? it->second : v;
}

BranchTarget map_target(const BranchTarget& t, const ValueMap& values, const BlockMap& blocks) {
    BranchTarget out;
    auto it = blocks.find(t.block);
    out.block = it != blocks.end() ? it->second : t.block;
    out.args.reserve(t.args.size());
    for (Value* a : t.args) out.args.push_back(map_value(values, a));
    return out;
}

} // namespace

BasicBlock* new_block(Function& fn, std::string_view name) {
    BasicBlock* bb = fn.parent()
        ? fn.parent()->arena().make<BasicBlock>(fn.next_block_id(), intern(fn, name))
        : new BasicBlock(fn.next_block_id(), name);
    bb->set_parent(&fn);
    fn.append_block(bb);
    return bb;
}

Value* new_block_param(Function& fn, BasicBlock* bb, Type type) {
    Value* p = fn.parent()
        ? fn.parent()->arena().make<Value>(fn.next_value_id(), type, ValueKind::BlockParam)
        : new Value(fn.next_value_id(), type, ValueKind::BlockParam);
    bb->add_param(p);
    return p;
}

Instruction* clone_shell(Function& fn, const Instruction& src, ValueMap& values) {
    Instruction* dst = fn.parent()
        ? fn.parent()->arena().make<Instruction>(src.opcode(), src.type())
        : new Instruction(src.opcode(), src.type());
    dst->set_imm_i64(src.imm_i64());
    dst->set_imm_f64(src.imm_f64());
    dst->set_scale(src.scale());
    dst->set_offset(src.offset());
    dst->set_memory_type(src.memory_type());
    dst->set_symbol(intern(fn, src.symbol()));
    dst->set_extra_symbol(intern(fn, src.extra_symbol()));
    dst->set_loc(src.loc());
    if (src.result()) {
        Value* res = fn.parent()
            ? fn.parent()->arena().make<Value>(fn.next_value_id(), src.result()->type(), ValueKind::InstructionResult)
            : new Value(fn.next_value_id(), src.result()->type(), ValueKind::InstructionResult);
        res->set_defining_instruction(dst);
        res->set_noalias(src.result()->is_noalias());
        dst->set_result(res);
        values[src.result()] = res;
    }
    return dst;
}

void clone_uses(const Instruction& src, Instruction& dst, const ValueMap& values, const BlockMap& blocks) {
    // Operand positions are significant, so a null operand stays null.
    for (Value* op : src.operands()) dst.add_operand(map_value(values, op));
    for (Value* sv : src.state_map()) dst.add_state_value(map_value(values, sv));
    if (src.branch_target().block || !src.branch_target().args.empty()) {
        dst.set_branch_target(map_target(src.branch_target(), values, blocks));
    }
    if (src.true_target().block || !src.true_target().args.empty()) {
        dst.set_true_target(map_target(src.true_target(), values, blocks));
    }
    if (src.false_target().block || !src.false_target().args.empty()) {
        dst.set_false_target(map_target(src.false_target(), values, blocks));
    }
    for (const SwitchCase& sc : src.switch_cases()) {
        dst.add_switch_case(sc.value, map_target(sc.target, values, blocks));
    }
}

Instruction* clone_instruction(Function& fn, const Instruction& src, ValueMap& values, const BlockMap& blocks) {
    Instruction* dst = clone_shell(fn, src, values);
    clone_uses(src, *dst, values, blocks);
    return dst;
}

} // namespace brass::ir
