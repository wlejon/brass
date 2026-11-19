// EXIT-ONLY STATE. A NaN-boxing front end hands a guard's exit the boxed form
// of a value its guarded code holds unboxed: a double as its bits with NaN
// canonicalized, an int32 as the bits of its double, a boolean's 0/1 or'd
// into its tag:
//
//     guard %c, @twin, [..., select(ne f, f, K, bitcast_i64_f64 f), ...]
//     guard %c, @twin, [..., bitcast_i64_f64(sitofp_f64_i32 x), ...]
//     guard %c, @twin, [..., or(zext_i64(and x, 1), K), ...]
//
// (each possibly under tagged bitcasts). When only guards read such a value,
// the GuardExit converts it out-of-line in the exit record (`aarch64_emit_guard.cpp`).

#include <brass/target/aarch64/aarch64_isel.hpp>
#include <brass/codegen/lir.hpp>
#include <brass/mir/instruction.hpp>

namespace brass::aarch64 {

using namespace brass::codegen;

namespace {

const Instruction* def_of(const Value* v) {
    return v && v->is_instruction() ? v->defining_instruction() : nullptr;
}

bool is_tagged_bitcast(Opcode op) {
    return op == Opcode::bitcast_i64_tagged || op == Opcode::bitcast_tagged_i64;
}

}  // namespace

void AArch64ISel::find_guard_state_folds(const Function& mir_fn, std::unordered_map<const Value*, uint32_t>& folded_uses) {
    guard_state_folds_.clear();
    std::unordered_map<const Value*, uint32_t> state_uses;
    for (const auto* bb : mir_fn.blocks()) {
        for (const auto* inst : *bb) {
            if (inst->opcode() != Opcode::guard) continue;
            for (const auto* sv : inst->state_map()) {
                if (sv) state_uses[sv]++;
            }
        }
    }
    auto uses = [&](const Value* v) {
        const auto it = use_count_.find(v);
        return it == use_count_.end() ? 0u : it->second;
    };
    for (const auto& [top, n] : state_uses) {
        if (uses(top) != n) continue;
        std::vector<const Instruction*> chain;
        const Value* v = top;
        const Instruction* d = def_of(v);
        bool shared = false;
        while (d && is_tagged_bitcast(d->opcode()) && !skipped_insts_.count(d)) {
            chain.push_back(d);
            v = d->operand(0);
            d = def_of(v);
            if (uses(v) != 1) {
                shared = true;
                break;
            }
        }
        std::vector<const Instruction*> conv_chain;
        GuardStateFold fold;
        auto conversion = [&]() -> bool {
            if (shared || !d || skipped_insts_.count(d)) return false;
            if (d->opcode() == Opcode::select && v->type().size_in_bytes() == 8 && !v->type().is_float()) {
                const Instruction* cmp = def_of(d->operand(0));
                if (!cmp || !is_self_nan_test(*cmp) || cmp->operand(0)->type() != Type::f64()) return false;
                if (uses(d->operand(0)) != 1) return false;
                const Value* f = cmp->operand(0);
                const bool nan_selects_first = cmp->opcode() == Opcode::ne;
                const Value* k = d->operand(nan_selects_first ? 1 : 2);
                const Value* bits = d->operand(nan_selects_first ? 2 : 1);
                const ImmIntInfo ki = get_imm_int_info(k);
                if (!ki.is_imm) return false;
                const Instruction* bc = def_of(bits);
                const Instruction* fd = def_of(f);
                const bool bits_of_f = bc && bc->opcode() == Opcode::bitcast_i64_f64 && bc->operand(0) == f;
                const bool f_of_bits = fd && fd->opcode() == Opcode::bitcast_f64_i64 && fd->operand(0) == bits;
                if (!bits_of_f && !f_of_bits) return false;
                fold = GuardStateFold{f, GuardStateConv::CanonF64Bits, ki.val};
                conv_chain = {d, cmp};
                if (bits_of_f && uses(bits) == 1) conv_chain.push_back(bc);
                folded_uses[k]++;
                return true;
            }
            if (d->opcode() == Opcode::bitcast_i64_f64) {
                const Value* conv = d->operand(0);
                const Instruction* cv = def_of(conv);
                if (!cv || cv->opcode() != Opcode::sitofp_f64_i32 || uses(conv) != 1) return false;
                fold = GuardStateFold{cv->operand(0), GuardStateConv::I32ToF64Bits, 0};
                conv_chain = {d, cv};
                return true;
            }
            if (d->opcode() == Opcode::or_ && v->type() == Type::i64()) {
                for (size_t side = 0; side < 2; ++side) {
                    const ImmIntInfo ki = get_imm_int_info(d->operand(1 - side));
                    const Value* z = d->operand(side);
                    const Instruction* zx = def_of(z);
                    if (!ki.is_imm || !zx || zx->opcode() != Opcode::zext_i64 || uses(z) != 1) continue;
                    const Value* b = zx->operand(0);
                    const Instruction* bd = def_of(b);
                    if (b->type() != Type::i32() || !bd) continue;
                    const Value* src = b;
                    conv_chain = {d, zx};
                    if (bd->opcode() == Opcode::and_ && uses(b) == 1 &&
                        get_imm_int_info(bd->operand(1)).is_imm &&
                        get_imm_int_info(bd->operand(1)).val == 1) {
                        src = bd->operand(0);
                        conv_chain.push_back(bd);
                        folded_uses[bd->operand(1)]++;
                    } else if (!is_comparison(bd->opcode())) {
                        continue;
                    }
                    fold = GuardStateFold{src, GuardStateConv::BoolTag, ki.val};
                    folded_uses[d->operand(1 - side)]++;
                    return true;
                }
            }
            return false;
        };
        if (conversion()) {
            chain.insert(chain.end(), conv_chain.begin(), conv_chain.end());
        } else {
            size_t read = chain.size();
            const Value* at = v;
            while (read > 0 && !at->type().is_tagged()) {
                --read;
                at = read == 0 ? top : chain[read - 1]->operand(0);
            }
            if (!at->type().is_tagged()) {
                at = v;
                read = chain.size();
            }
            chain.resize(read);
            if (chain.empty()) continue;
            fold = GuardStateFold{at, GuardStateConv::None, 0};
        }
        for (const Instruction* c : chain) skipped_insts_.insert(c);
        guard_state_folds_[top] = fold;
    }
}

} // namespace brass::aarch64
