// The tagged allocator's inline window
// (include/brass/mir/tagged_alloc_lowering.hpp).
#include <brass/mir/tagged_alloc_lowering.hpp>
#include <brass/mir/builder.hpp>
#include <brass/mir/gc_refs.hpp>
#include <brass/mir/module.hpp>
#include <brass/mir/tagged_alloc.hpp>
#include <brass/mir/uses.hpp>

#include <unordered_set>
#include <vector>

namespace brass {

namespace {

constexpr int64_t kCollectorHeaderBytes = 8;
constexpr uint64_t kAddressMask = 0x0000FFFFFFFFFFFFULL;

bool is_address_mask(const Value* v) {
    const Instruction* def = v ? v->defining_instruction() : nullptr;
    return def && def->opcode() == Opcode::iconst_i64 && static_cast<uint64_t>(def->imm_i64()) == kAddressMask;
}

bool windowed(const Function& fn) {
    const Module* mod = fn.parent();
    return mod && mod->tagged_allocator().inline_window() && mod->pinned_tls_register();
}

bool expandable(const Instruction& inst) {
    return is_tagged_allocation(&inst) && inst.result() &&
           (inst.type() == Type::tagged() || inst.type() == Type::i64());
}

} // namespace

bool needs_tagged_allocation_expansion(const Function& fn) {
    if (!windowed(fn)) return false;
    for (const BasicBlock* bb : fn.blocks()) {
        if (!bb) continue;
        for (const Instruction* inst : *bb) {
            if (expandable(*inst)) return true;
        }
    }
    return false;
}

namespace {

// One allocation's expansion. `extra` bytes past its own are reserved with
// it (the later members of its group); `reserved`, when set, is the address
// an earlier member reserved for this one, or 0 when that member took the
// cold path. Answers the address after this object's bytes when the next
// member's space follows them, else 0 (the value its continuation gets as
// `*rest`).
class Expander {
public:
    explicit Expander(Function& fn) : fn_(fn), alloc_(fn.parent()->tagged_allocator()), b_(fn) {}

    void expand(Instruction* call, int64_t extra, Value* reserved, Value** rest);

private:
    BasicBlock* block(bool cold) {
        BasicBlock* bb = b_.create_block();
        fn_.append_block(bb);
        if (cold) bb->set_cold();
        return bb;
    }

    Function& fn_;
    const Module::TaggedAllocator alloc_;
    Builder b_;
};

void Expander::expand(Instruction* call, int64_t extra, Value* reserved, Value** rest) {
    BasicBlock* bb = call->parent();
    const bool grouped = rest != nullptr;
    BasicBlock* cont = block(false);
    while (Instruction* moved = call->next()) {
        bb->remove_instruction(moved);
        cont->append_instruction(moved);
    }
    Value* obj = b_.create_value(call->type());
    cont->add_param(obj);
    Value* rest_out = nullptr;
    if (grouped) {
        rest_out = b_.create_value(Type::i64());
        cont->add_param(rest_out);
        *rest = rest_out;
    }
    replace_all_uses(fn_, call->result(), obj);

    // Up to the continuation's first GC point the object has not moved: its
    // bits and its address there are the ones computed here, passed in
    // rather than re-read from the tagged value, which lives in a root slot.
    std::vector<Instruction*> casts;
    std::vector<Instruction*> masks;
    if (call->type() == Type::tagged()) {
        std::unordered_set<const Value*> cast_values;
        for (Instruction* inst = cont->head(); inst && !inst->is_terminator() && !may_trigger_gc(*inst);
             inst = inst->next()) {
            if (inst->opcode() == Opcode::bitcast_i64_tagged && inst->operand(0) == obj && inst->result()) {
                casts.push_back(inst);
                cast_values.insert(inst->result());
            } else if (inst->opcode() == Opcode::and_ && inst->result() && inst->operand_count() == 2 &&
                       cast_values.count(inst->operand(0)) && is_address_mask(inst->operand(1))) {
                masks.push_back(inst);
            }
        }
    }
    Value* bits_in = nullptr;
    Value* addr_in = nullptr;
    if (!casts.empty()) {
        bits_in = b_.create_value(Type::i64());
        cont->add_param(bits_in);
        for (Instruction* c : casts) replace_all_uses(fn_, c->result(), bits_in);
        for (Instruction* c : casts) cont->remove_instruction(c);
    }
    if (!masks.empty()) {
        addr_in = b_.create_value(Type::i64());
        cont->add_param(addr_in);
        for (Instruction* m : masks) replace_all_uses(fn_, m->result(), addr_in);
        for (Instruction* m : masks) cont->remove_instruction(m);
    }
    Value* bytes = call->operand(0);
    Value* header = call->operand(1);
    const DebugLoc loc = call->loc();
    const Type t = call->type();
    bb->remove_instruction(call);
    auto after = [&](Value* at) {
        return b_.build_add(b_.build_add(at, bytes), b_.build_iconst_i64(kCollectorHeaderBytes));
    };
    // `addr` the payload address, `v` the object (as the call's type) and
    // `vbits` its bits.
    auto edge = [&](Value* v, Value* vbits, Value* addr, Value* r) {
        std::vector<Value*> args{v};
        if (grouped) args.push_back(r);
        if (bits_in) args.push_back(vbits);
        if (addr_in) args.push_back(addr ? addr : b_.build_and(vbits, b_.build_iconst_i64(static_cast<int64_t>(kAddressMask))));
        b_.build_br(cont, args);
    };
    auto box = [&](Value* at, Value** vbits) {
        *vbits = b_.build_or(at, b_.build_iconst_i64(static_cast<int64_t>(alloc_.tag)));
        return t == Type::tagged() ? b_.build_bitcast_tagged_i64(*vbits) : *vbits;
    };

    b_.position_at_end(bb);
    b_.set_current_loc(loc);
    if (reserved) {
        // The space an earlier member reserved: its collector header, no
        // test of the window.
        BasicBlock* resv = block(false);
        BasicBlock* own = block(false);
        b_.build_br_if(b_.build_ne(reserved, b_.build_iconst_i64(0)), resv, own);
        b_.position_at_end(resv);
        b_.set_current_loc(loc);
        b_.build_store(Type::i64(), reserved, 0, b_.build_or(header, bytes));
        Value* at = b_.build_add(reserved, b_.build_iconst_i64(kCollectorHeaderBytes));
        Value* vbits = nullptr;
        Value* v = box(at, &vbits);
        edge(v, vbits, at, after(reserved));
        bb = own;
        b_.position_at_end(bb);
        b_.set_current_loc(loc);
    }
    BasicBlock* fast = block(false);
    BasicBlock* cold = block(true);
    Value* tls = b_.build_pinned_tls_read();
    Value* top = b_.build_load(Type::i64(), tls, alloc_.top_offset);
    Value* next = after(top);
    // The first member of a group takes the window for all of them.
    Value* claim = extra > 0 ? b_.build_add(next, b_.build_iconst_i64(extra)) : next;
    Value* end = b_.build_load(Type::i64(), tls, alloc_.end_offset);
    b_.build_br_if(b_.build_ule(claim, end), fast, cold);

    b_.position_at_end(fast);
    b_.set_current_loc(loc);
    b_.build_store(Type::i64(), tls, alloc_.top_offset, claim);
    b_.build_store(Type::i64(), top, 0, b_.build_or(header, bytes));
    Value* at = b_.build_add(top, b_.build_iconst_i64(kCollectorHeaderBytes));
    Value* vbits = nullptr;
    Value* v = box(at, &vbits);
    edge(v, vbits, at, extra > 0 ? next : b_.build_iconst_i64(0));

    b_.position_at_end(cold);
    cold->append_instruction(call);
    b_.position_at_end(cold);
    b_.set_current_loc(loc);
    Value* cold_bits = t == Type::tagged() ? b_.build_bitcast_i64_tagged(call->result()) : call->result();
    edge(call->result(), cold_bits, nullptr, b_.build_iconst_i64(0));
}

} // namespace

size_t expand_tagged_allocations(Function& fn) {
    if (!windowed(fn)) return 0;
    // Groups: in one block, allocations of constant size with nothing that
    // may collect between one and the next. The first takes the window for
    // the group and the others their reserved space, with one test of the
    // window between them all; a member after a cold first tests its own.
    std::vector<std::vector<Instruction*>> groups;
    for (BasicBlock* bb : fn.blocks()) {
        if (!bb) continue;
        bool open = false;
        for (Instruction* inst : *bb) {
            if (expandable(*inst)) {
                if (open && tagged_allocation_bytes(inst) >= 0 && tagged_allocation_bytes(groups.back().back()) >= 0) {
                    groups.back().push_back(inst);
                } else {
                    groups.push_back({inst});
                }
                open = true;
            } else if (may_trigger_gc(*inst)) {
                open = false;
            }
        }
    }
    if (groups.empty()) return 0;

    Expander ex(fn);
    size_t n = 0;
    for (const std::vector<Instruction*>& g : groups) {
        int64_t extra = 0;
        for (size_t i = 1; i < g.size(); ++i) extra += tagged_allocation_bytes(g[i]) + kCollectorHeaderBytes;
        Value* rest = nullptr;
        for (size_t i = 0; i < g.size(); ++i) {
            const bool more = i + 1 < g.size();
            Value* reserved = rest;
            ex.expand(g[i], i == 0 ? extra : 0, reserved, more ? &rest : nullptr);
            ++n;
        }
    }
    fn.rebuild_cfg_predecessors();
    return n;
}

} // namespace brass
