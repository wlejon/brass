#include "f64_demote_internal.hpp"
#include "ir_clone.hpp"
#include <brass/mir/builder.hpp>
#include <brass/mir/opcodes.hpp>
#include <brass/mir/verifier.hpp>
#include <functional>
#include <string>
#include <unordered_map>
#include <vector>

// The demotion's one unchecked assumption, made checked. Its dataflow takes
// every f64 entry parameter to hold an integer (a parameter is whatever the
// caller passed), and its bounds take a loop limit's parameters to be sizes
// in [0, 2^26); the rewrite then converts such a parameter to i64 with a bare
// fptosi. A width of 10.5 would index with 10. So a function that relies on
// it gets two bodies: the one the pass demotes, entered when a test of the
// parameters at entry holds, and a copy of the body as it was before, entered
// otherwise.
namespace brass {

namespace {

void collect_args_into(const BranchTarget& bt, const BasicBlock* target, size_t idx, std::vector<const Value*>& out) {
    if (bt.block == target && idx < bt.args.size() && bt.args[idx]) out.push_back(bt.args[idx]);
}

} // namespace

std::vector<Value*> assumed_entry_params(const Function& fn, const std::unordered_set<Value*>& demote_set,
                                         const std::unordered_set<const Value*>& exact_ints,
                                         const std::unordered_set<const Value*>& sized) {
    std::vector<Value*> out;
    const BasicBlock* entry = fn.entry_block();
    if (!entry) return out;
    std::unordered_set<const Value*> entry_f64;
    for (size_t i = 0; i < entry->param_count(); ++i) {
        const Value* p = entry->param(i);
        if (p && p->type() == Type::f64()) entry_f64.insert(p);
    }
    if (entry_f64.empty()) return out;

    // A value's exactness comes from its operands (an instruction) or its
    // incoming arguments (a block parameter); walk those back from every
    // demoted value, through values the dataflow called exact.
    std::unordered_set<const Value*> seen;
    std::unordered_set<const Value*> found(sized.begin(), sized.end());
    std::vector<const Value*> stack(demote_set.begin(), demote_set.end());
    while (!stack.empty()) {
        const Value* v = stack.back();
        stack.pop_back();
        if (!v || !seen.insert(v).second) continue;
        if (entry_f64.count(v)) {
            found.insert(v);
            continue;
        }
        if (v->type() != Type::f64()) continue;
        if (exact_ints.count(v) == 0 && demote_set.count(const_cast<Value*>(v)) == 0) continue;
        if (v->is_instruction()) {
            const Instruction* d = v->defining_instruction();
            for (size_t i = 0; i < d->operand_count(); ++i) stack.push_back(d->operand(i));
        } else if (v->is_block_param()) {
            const BasicBlock* bb = v->defining_block();
            if (!bb) continue;
            const size_t idx = v->param_index();
            for (const BasicBlock* pred : bb->predecessors()) {
                const Instruction* term = pred ? pred->terminator() : nullptr;
                if (!term) continue;
                if (term->opcode() == Opcode::br) collect_args_into(term->branch_target(), bb, idx, stack);
                else if (term->opcode() == Opcode::br_if) {
                    collect_args_into(term->true_target(), bb, idx, stack);
                    collect_args_into(term->false_target(), bb, idx, stack);
                } else if (term->opcode() == Opcode::switch_) {
                    collect_args_into(term->default_target(), bb, idx, stack);
                    for (const auto& sc : term->switch_cases()) collect_args_into(sc.target, bb, idx, stack);
                }
            }
        }
    }
    // In parameter order, so the check reads the same way every time.
    for (size_t i = 0; i < entry->param_count(); ++i) {
        if (found.count(entry->param(i))) out.push_back(entry->param(i));
    }
    return out;
}

bool version_loop_on_clamp(Function& fn, const ClampedLimit& site, const std::vector<ClampedLimit>& all) {
    constexpr size_t kMaxCopiedInstructions = 4000;
    Instruction* ph_term = site.preheader ? site.preheader->terminator() : nullptr;
    if (!ph_term || ph_term->opcode() != Opcode::br || ph_term->branch_target().block != site.header) return false;

    // Everything reachable from the header short of the blocks that dominate
    // it (an enclosing loop's header and body before this loop): the copy
    // branches back to those, which choose again on their next way in, and
    // every value they define dominates the copy as it does the loop.
    fn.rebuild_cfg_predecessors();
    const DominatorTree dom(fn);
    std::vector<BasicBlock*> region;
    std::unordered_set<const BasicBlock*> in_region;
    std::vector<BasicBlock*> work = {site.header};
    size_t instructions = 0;
    while (!work.empty()) {
        BasicBlock* bb = work.back();
        work.pop_back();
        if (!bb || in_region.count(bb)) continue;
        if (bb != site.header && dom.dominates(bb, site.header)) continue;
        in_region.insert(bb);
        region.push_back(bb);
        for (const Instruction* inst : *bb) {
            if (inst) ++instructions;
        }
        if (instructions > kMaxCopiedInstructions) return false;
        const Instruction* term = bb->terminator();
        if (!term) continue;
        auto push = [&](const BranchTarget& bt) { if (bt.block) work.push_back(bt.block); };
        if (term->opcode() == Opcode::br) push(term->branch_target());
        else if (term->opcode() == Opcode::br_if) { push(term->true_target()); push(term->false_target()); }
        else if (term->opcode() == Opcode::switch_) {
            push(term->default_target());
            for (const auto& sc : term->switch_cases()) push(sc.target);
        } else if (term->opcode() == Opcode::invoke) {
            return false;  // an unwind edge this walk does not follow
        }
    }
    // Keep the body's own order, so the copy reads like the code it copies.
    std::vector<BasicBlock*> ordered;
    for (BasicBlock* bb : fn.blocks()) {
        if (bb && in_region.count(bb)) ordered.push_back(bb);
    }

    ir::BlockMap blocks;
    ir::ValueMap values;
    for (BasicBlock* bb : ordered) {
        BasicBlock* copy = ir::new_block(fn, std::string(bb->name()) + "_f64c");
        blocks[bb] = copy;
        for (size_t i = 0; i < bb->param_count(); ++i) {
            values[bb->param(i)] = ir::new_block_param(fn, copy, bb->param(i)->type());
        }
    }
    std::vector<std::pair<const Instruction*, Instruction*>> pairs;
    std::unordered_map<const Instruction*, Instruction*> copies;
    for (BasicBlock* bb : ordered) {
        for (const Instruction* inst : *bb) {
            if (!inst) continue;
            Instruction* shell = ir::clone_shell(fn, *inst, values);
            blocks[bb]->append_instruction(shell);
            pairs.emplace_back(inst, shell);
            copies[inst] = shell;
        }
    }
    for (auto& [orig, copy] : pairs) ir::clone_uses(*orig, *copy, values, blocks);
    for (const ClampedLimit& c : all) {
        auto it = copies.find(c.cmp);
        if (it == copies.end()) continue;
        auto vit = values.find(c.original);
        it->second->set_operand(c.idx, vit != values.end() ? vit->second : c.original);
    }

    const std::vector<Value*> args = ph_term->branch_target().args;
    Builder b(*fn.parent());
    b.set_function(&fn);
    b.position_before(ph_term);
    Instruction* choose = b.build_br_if(site.fits, site.header, Span<Value* const>(args.data(), args.size()),
                                        blocks[site.header], Span<Value* const>(args.data(), args.size()));
    site.preheader->remove_instruction(ph_term);
    fn.rebuild_cfg_predecessors();
    if (verify_function(fn)) return true;

    // A shape the region rule did not foresee (a value of the region used
    // past a join it does not dominate): take the copy back out.
    b.position_before(choose);
    b.build_br(site.header, Span<Value* const>(args.data(), args.size()));
    site.preheader->remove_instruction(choose);
    for (BasicBlock* bb : ordered) fn.remove_block(blocks[bb]);
    fn.rebuild_cfg_predecessors();
    return false;
}

BasicBlock* version_on_entry_params(Function& fn, const std::vector<Value*>& assumed,
                                    const std::unordered_set<const Value*>& sized,
                                    const std::vector<ClampedLimit>& clamped) {
    BasicBlock* old_entry = fn.entry_block();
    const std::vector<BasicBlock*> body(fn.blocks().begin(), fn.blocks().end());
    const double sized_max = sized_limit_max(fn);

    // The fallback: every block, shells first so each result exists before
    // any operand is remapped.
    ir::BlockMap blocks;
    ir::ValueMap values;
    for (BasicBlock* bb : body) {
        if (!bb) continue;
        BasicBlock* copy = ir::new_block(fn, std::string(bb->name()) + "_f64v");
        blocks[bb] = copy;
        for (size_t i = 0; i < bb->param_count(); ++i) {
            values[bb->param(i)] = ir::new_block_param(fn, copy, bb->param(i)->type());
        }
    }
    std::vector<std::pair<const Instruction*, Instruction*>> pairs;
    std::unordered_map<const Instruction*, Instruction*> copies;
    for (BasicBlock* bb : body) {
        if (!bb) continue;
        for (const Instruction* inst : *bb) {
            if (!inst) continue;
            Instruction* shell = ir::clone_shell(fn, *inst, values);
            blocks[bb]->append_instruction(shell);
            pairs.emplace_back(inst, shell);
            copies[inst] = shell;
        }
    }
    for (auto& [orig, copy] : pairs) ir::clone_uses(*orig, *copy, values, blocks);
    // The fallback reads each loop's own limit, so its loops run as written.
    for (const ClampedLimit& c : clamped) {
        auto it = copies.find(c.cmp);
        if (it == copies.end()) continue;
        auto vit = values.find(c.original);
        it->second->set_operand(c.idx, vit != values.end() ? vit->second : c.original);
    }

    // The new entry takes the function's parameters and tests the assumed
    // ones: inside the range (false for NaN), then equal bit for bit to the
    // integer they convert to (false for a fraction and for -0). The range
    // test comes first and picks what is converted, so the conversion never
    // sees a value out of an i64's reach.
    BasicBlock* entry = ir::new_block(fn, kF64VersionEntryName);
    fn.blocks().pop_back();
    fn.prepend_block(entry);
    std::vector<Value*> args;
    for (size_t i = 0; i < old_entry->param_count(); ++i) {
        args.push_back(ir::new_block_param(fn, entry, old_entry->param(i)->type()));
    }

    Builder b(*fn.parent());
    b.set_function(&fn);
    b.position_at_end(entry);
    Value* ok = nullptr;
    auto require = [&](Value* cond) { ok = ok ? b.build_and(ok, cond) : cond; };
    for (Value* p : assumed) {
        const size_t idx = p->param_index();
        Value* a = args[idx];
        const double hi = sized.count(p) ? sized_max : 9007199254740992.0;
        const double lo = sized.count(p) ? 0.0 : -9007199254740992.0;
        Value* in_range = b.build_and(b.build_sge(a, b.build_fconst_f64(lo)), b.build_sle(a, b.build_fconst_f64(hi)));
        Value* safe = b.build_select(in_range, a, b.build_fconst_f64(0.0));
        Value* back = b.build_sitofp_f64_i64(b.build_fptosi_i64(safe));
        Value* same = b.build_eq(b.build_bitcast_i64_f64(a), b.build_bitcast_i64_f64(back));
        require(b.build_and(in_range, same));
    }
    // Each clamped limit computed again from the parameters: its ceiling
    // within the clamp (false for NaN) is a loop the clamp leaves as it is.
    ir::ValueMap at_entry;
    for (size_t i = 0; i < old_entry->param_count(); ++i) at_entry[old_entry->param(i)] = args[i];
    std::function<Value*(const Value*)> rebuild = [&](const Value* v) -> Value* {
        if (auto it = at_entry.find(v); it != at_entry.end()) return it->second;
        const Instruction* d = v->defining_instruction();
        Value* r = nullptr;
        if (d->opcode() == Opcode::fconst_f64) r = b.build_fconst_f64(d->imm_f64());
        else if (d->opcode() == Opcode::add) r = b.build_add(rebuild(d->operand(0)), rebuild(d->operand(1)));
        else if (d->opcode() == Opcode::sub) r = b.build_sub(rebuild(d->operand(0)), rebuild(d->operand(1)));
        else r = b.build_mul(rebuild(d->operand(0)), rebuild(d->operand(1)));
        at_entry[v] = r;
        return r;
    };
    for (const ClampedLimit& c : clamped) {
        require(b.build_sle(b.build_ceil_f64(rebuild(c.limit)), b.build_fconst_f64(c.max)));
    }
    b.build_br_if(ok, old_entry, Span<Value* const>(args.data(), args.size()), blocks[old_entry],
                  Span<Value* const>(args.data(), args.size()));
    fn.rebuild_cfg_predecessors();
    return old_entry;
}

} // namespace brass
