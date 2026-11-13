// The write barrier's inline young-target filter
// (include/brass/mir/write_barrier_lowering.hpp).
#include <brass/mir/write_barrier_lowering.hpp>
#include <brass/gc/object.hpp>
#include <brass/mir/builder.hpp>
#include <brass/mir/module.hpp>
#include <brass/mir/uses.hpp>

#include <algorithm>

#include <vector>

namespace brass {

namespace {

constexpr std::string_view kBarrierSymbol = "brass_gc_write_barrier";

bool filtered(const Function& fn) {
    const Module* mod = fn.parent();
    return mod && mod->young_range() && mod->pinned_tls_register();
}

bool expandable(const Instruction& inst) {
    return inst.opcode() == Opcode::write_barrier && inst.operand_count() == 2 && inst.operand(0) &&
           inst.operand(1) && inst.operand(0)->type().kind() == TypeKind::I64;
}

// The address `v` is an offset from, where that is one with no tag bits above
// gc::kAddressMask (a value masked with exactly that, which an index masked
// with something smaller is not); null where `v`
// is no such sum. A slot's address is in the generation its object is in, so
// the filter can ask the object's.
Value* clean_base(Value* v, int depth = 0) {
    const Instruction* def = v ? v->defining_instruction() : nullptr;
    if (!def || depth > 4 || def->operand_count() != 2) return nullptr;
    if (def->opcode() == Opcode::and_) {
        for (size_t i = 0; i < 2; ++i) {
            const Instruction* c = def->operand(i) ? def->operand(i)->defining_instruction() : nullptr;
            if (c && c->opcode() == Opcode::iconst_i64 &&
                static_cast<uint64_t>(c->imm_i64()) == gc::kAddressMask) {
                return v;
            }
        }
        return nullptr;
    }
    if (def->opcode() != Opcode::add) return nullptr;
    for (size_t i = 0; i < 2; ++i) {
        if (Value* base = clean_base(def->operand(i), depth + 1)) return base;
    }
    return nullptr;
}

// The arithmetic in `bb` that computes `v` for the barrier alone, `base`
// apart: with the filter asking `base`, only the call reads it.
void address_chain(Value* v, const Value* base, const BasicBlock* bb, const UseCounts& uses,
                   std::vector<Instruction*>& chain, int depth = 0) {
    Instruction* def = v ? v->defining_instruction() : nullptr;
    if (!def || v == base || depth > 6 || def->parent() != bb) return;
    const Opcode op = def->opcode();
    if (op != Opcode::add && op != Opcode::shl && op != Opcode::iconst_i64) return;
    const uint32_t* n = uses.find(v);
    if (!n || *n != 1) return;
    chain.push_back(def);
    for (size_t i = 0; i < def->operand_count(); ++i) address_chain(def->operand(i), base, bb, uses, chain, depth + 1);
}

} // namespace

bool needs_write_barrier_expansion(const Function& fn) {
    if (!filtered(fn)) return false;
    for (const BasicBlock* bb : fn.blocks()) {
        if (!bb) continue;
        for (const Instruction* inst : *bb) {
            if (expandable(*inst)) return true;
        }
    }
    return false;
}

size_t expand_write_barriers(Function& fn) {
    if (!filtered(fn)) return 0;
    std::vector<Instruction*> sites;
    for (BasicBlock* bb : fn.blocks()) {
        if (!bb) continue;
        for (Instruction* inst : *bb) {
            if (expandable(*inst)) sites.push_back(inst);
        }
    }
    if (sites.empty()) return 0;

    Module* mod = fn.parent();
    const UseCounts uses = compute_use_counts(fn);
    const Module::YoungRange range = mod->young_range();
    mod->add_external_symbol(kBarrierSymbol);
    Builder b(fn);
    for (Instruction* barrier : sites) {
        BasicBlock* bb = barrier->parent();
        BasicBlock* cold = b.create_block();
        cold->set_cold();
        fn.append_block(cold);
        BasicBlock* cont = b.create_block();
        fn.append_block(cont);
        while (Instruction* moved = barrier->next()) {
            bb->remove_instruction(moved);
            cont->append_instruction(moved);
        }
        Value* obj = barrier->operand(0);
        Value* val = barrier->operand(1);
        const DebugLoc loc = barrier->loc();
        bb->remove_instruction(barrier);
        // A slot address made of a clean base and offsets: the filter asks the
        // base, and the sum is computed where the call is.
        Value* clean = clean_base(obj);
        std::vector<Instruction*> chain;
        if (clean && clean != obj) address_chain(obj, clean, bb, uses, chain);
        std::vector<Instruction*> sunk;
        for (Instruction* inst = bb->head(); inst && !chain.empty(); inst = inst->next()) {
            if (std::find(chain.begin(), chain.end(), inst) != chain.end()) sunk.push_back(inst);
        }
        for (Instruction* inst : sunk) {
            bb->remove_instruction(inst);
            cold->append_instruction(inst);
        }

        b.position_at_end(bb);
        b.set_current_loc(loc);
        Value* tls = b.build_pinned_tls_read();
        Value* address =
            clean ? clean : b.build_and(obj, b.build_iconst_i64(static_cast<int64_t>(gc::kAddressMask)));
        Value* base = b.build_load(Type::i64(), tls, range.base_offset);
        Value* span = b.build_load(Type::i64(), tls, range.span_offset);
        b.build_br_if(b.build_ult(b.build_sub(address, base), span), cont, cold);

        b.position_at_end(cold);
        b.set_current_loc(loc);
        b.build_call(kBarrierSymbol, Type::void_type(), {obj, val});
        b.build_br(cont);
    }
    fn.rebuild_cfg_predecessors();
    return sites.size();
}

} // namespace brass
