#include <brass/mir/fp_load.hpp>

#include <brass/mir/block.hpp>
#include <brass/mir/instruction.hpp>
#include <brass/mir/uses.hpp>

#include <unordered_map>
#include <vector>

namespace brass {

size_t read_doubles_as_doubles(Function& fn) {
    // Each i64 load's uses: how many, and how many of them are the operand
    // of a bitcast to f64.
    struct Uses {
        Instruction* load = nullptr;
        uint32_t all = 0;
        uint32_t as_double = 0;
    };
    std::unordered_map<const Value*, Uses> loads;
    for (BasicBlock* bb : fn.blocks()) {
        if (!bb) continue;
        for (Instruction* inst : *bb) {
            const Opcode op = inst->opcode();
            if ((op == Opcode::load || op == Opcode::load_indexed) && inst->result() &&
                inst->type() == Type::i64() && inst->memory_type() == Type::i64()) {
                loads[inst->result()].load = inst;
            }
        }
    }
    if (loads.empty()) return 0;
    for (BasicBlock* bb : fn.blocks()) {
        if (!bb) continue;
        for (Instruction* inst : *bb) {
            for_each_use(*inst, [&](const Value* v) {
                if (auto it = loads.find(v); it != loads.end()) ++it->second.all;
            });
            if (inst->opcode() == Opcode::bitcast_f64_i64 && inst->operand_count() == 1) {
                if (auto it = loads.find(inst->operand(0)); it != loads.end()) ++it->second.as_double;
            }
        }
    }

    std::unordered_map<const Value*, Value*> stands_for;
    std::vector<Instruction*> casts;
    size_t retyped = 0;
    for (auto& [value, uses] : loads) {
        if (uses.all == 0 || uses.all != uses.as_double) continue;
        uses.load->set_type(Type::f64());
        uses.load->set_memory_type(Type::f64());
        uses.load->result()->set_type(Type::f64());
        ++retyped;
    }
    if (retyped == 0) return 0;
    for (BasicBlock* bb : fn.blocks()) {
        if (!bb) continue;
        for (Instruction* inst : *bb) {
            if (inst->opcode() != Opcode::bitcast_f64_i64 || inst->operand_count() != 1) continue;
            Value* src = inst->operand(0);
            if (loads.count(src) && src->type() == Type::f64()) {
                stands_for[inst->result()] = src;
                casts.push_back(inst);
            }
        }
    }
    for (BasicBlock* bb : fn.blocks()) {
        if (!bb) continue;
        for (Instruction* inst : *bb) {
            for_each_use_slot(*inst, [&](Value*& v) {
                if (auto it = stands_for.find(v); it != stands_for.end()) v = it->second;
            });
        }
    }
    for (Instruction* cast : casts) {
        // A certificate that named the cast names the load it stood for.
        fn.retarget_guard_elim_witness(cast->result(), stands_for[cast->result()]);
        cast->parent()->remove_instruction(cast);
    }
    return retyped;
}

}  // namespace brass
