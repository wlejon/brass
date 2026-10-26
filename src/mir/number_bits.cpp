#include "number_bits.hpp"

#include <brass/mir/block.hpp>
#include <cmath>

namespace brass::number_bits {

namespace {

constexpr unsigned kMaxDepth = 24;

bool const_bits(const Value* v, uint64_t& out) {
    if (!v || !v->is_instruction()) return false;
    const Instruction* d = v->defining_instruction();
    if (!d || (d->opcode() != Opcode::iconst_i64)) return false;
    out = static_cast<uint64_t>(d->imm_i64());
    return true;
}

// A double that is never a NaN.
bool never_nan(const Value* f) {
    if (!f || !f->is_instruction()) return false;
    const Instruction* d = f->defining_instruction();
    if (!d) return false;
    switch (d->opcode()) {
        case Opcode::fconst_f64: return !std::isnan(d->imm_f64());
        case Opcode::sitofp_f64_i32:
        case Opcode::sitofp_f64_i64: return true;
        default: return false;
    }
}

// The double `c` tests for a NaN (`f != f`), and the one it tests for not
// being one (`f == f`).
const Value* nan_test(const Value* c, bool& is_nan) {
    if (!c || !c->is_instruction()) return nullptr;
    const Instruction* d = c->defining_instruction();
    if (!d || d->operand_count() != 2 || d->operand(0) != d->operand(1)) return nullptr;
    const Value* f = d->operand(0);
    if (!f || f->type() != Type::f64()) return nullptr;
    if (d->opcode() == Opcode::ne) {
        is_nan = true;
        return f;
    }
    if (d->opcode() == Opcode::eq) {
        is_nan = false;
        return f;
    }
    return nullptr;
}

} // namespace

Facts::Facts(const Function& fn) : fn_(fn) {
    for (const auto& rp : fn.resume_points()) {
        if (rp.second) resume_targets_.insert(rp.second);
    }
}

bool Facts::at_most_ordered_max(const Value* v) {
    auto it = known_.find(v);
    if (it != known_.end()) return it->second;
    const bool r = holds(v, nullptr, 0);
    known_[v] = r;
    return r;
}

bool Facts::holds(const Value* v, const Value* ordered, unsigned depth) {
    if (!v || v->type() != Type::i64() || depth > kMaxDepth) return false;
    uint64_t c = 0;
    if (const_bits(v, c)) return c <= kMaxOrderedBits;
    if (v->is_block_param()) return param_holds(v, depth);
    if (!v->is_instruction()) return false;
    const Instruction* d = v->defining_instruction();
    if (!d) return false;
    switch (d->opcode()) {
        case Opcode::bitcast_i64_f64: {
            const Value* f = d->operand(0);
            return f && (f == ordered || never_nan(f));
        }
        case Opcode::select: {
            bool is_nan = false;
            const Value* f = nan_test(d->operand(0), is_nan);
            const Value* t_ordered = f && !is_nan ? f : ordered;
            const Value* f_ordered = f && is_nan ? f : ordered;
            return holds(d->operand(1), t_ordered, depth + 1) && holds(d->operand(2), f_ordered, depth + 1);
        }
        case Opcode::and_: {
            uint64_t m = 0;
            if (const_bits(d->operand(1), m) || const_bits(d->operand(0), m)) {
                if (m <= kMaxOrderedBits) return true;
            }
            return holds(d->operand(0), ordered, depth + 1) || holds(d->operand(1), ordered, depth + 1);
        }
        case Opcode::lshr: {
            uint64_t s = 0;
            return const_bits(d->operand(1), s) && (s & 63) != 0;
        }
        default:
            return false;
    }
}

bool Facts::param_holds(const Value* param, unsigned depth) {
    const BasicBlock* bb = param->defining_block();
    if (!bb || bb == fn_.entry_block() || resume_targets_.count(bb)) return false;
    auto it = known_.find(param);
    if (it != known_.end()) return it->second;
    if (assumed_.count(param)) return true;
    const size_t idx = param->param_index();
    assumed_.insert(param);
    bool ok = !bb->predecessors().empty();
    for (const BasicBlock* pred : bb->predecessors()) {
        if (!ok) break;
        const Instruction* term = pred ? pred->terminator() : nullptr;
        if (!term) {
            ok = false;
            break;
        }
        auto check = [&](const BranchTarget& bt) {
            if (!ok || bt.block != bb) return;
            ok = idx < bt.args.size() && holds(bt.args[idx], nullptr, depth + 1);
        };
        switch (term->opcode()) {
            case Opcode::br: check(term->branch_target()); break;
            case Opcode::br_if:
                check(term->true_target());
                check(term->false_target());
                break;
            case Opcode::switch_:
                check(term->default_target());
                for (const auto& sc : term->switch_cases()) check(sc.target);
                break;
            default:
                // An invoke's targets take the call's own results.
                ok = false;
                break;
        }
    }
    assumed_.erase(param);
    // A parameter decided while an outer one was assumed rests on it.
    if (assumed_.empty()) known_[param] = ok;
    return ok;
}

} // namespace brass::number_bits
