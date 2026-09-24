#include <brass/mir/loop_fusion.hpp>
#include <brass/mir/alias_analysis.hpp>
#include <brass/mir/escape_analysis.hpp>
#include <brass/mir/opcodes.hpp>
#include <brass/mir/range_analysis.hpp>
#include <brass/mir/runtime_symbols.hpp>
#include "int_fold.hpp"
#include "ir_clone.hpp"
#include <cmath>
#include <cstring>
#include <memory>
#include <optional>
#include <sstream>
#include <unordered_map>
#include <unordered_set>
#include <vector>

// Loop fusion runs two adjacent loops with the same iteration space as one.
// Iteration i of the fused loop executes the first loop's body for i and
// then the second's, so every pair of operations that the original order
// kept apart (all of loop 1 before any of loop 2) must commute unless the
// second loop only ever looks at what the first wrote in the same or an
// earlier iteration. The analysis below accepts only the shapes where that
// is provable:
//   - both loops are a header plus one body/latch block, counted by an
//     induction variable with identical start, limit, step and exit test;
//   - the code between them is pure, does not depend on loop 1, and can run
//     before loop 1;
//   - loop 2 reads nothing loop 1 computes except through memory;
//   - neither loop calls anything but a declared allocator, deoptimizes, or
//     may trap; and every memory access pair where one side writes is either
//     provably disjoint or indexes the same buffer by the induction variable
//     with identical, non-overlapping element geometry.
namespace brass {

std::string LoopFusionStats::format_report() const {
    std::ostringstream ss;
    ss << "=== Loop Fusion Statistics ===\n"
       << "  Loops fused: " << loops_fused << "\n"
       << "  Candidates checked: " << candidates_checked << "\n"
       << "  Rejected (non-adjacent): " << rejected_non_adjacent << "\n"
       << "  Rejected (domain mismatch): " << rejected_domain_mismatch << "\n"
       << "  Rejected (dependencies): " << rejected_dependencies << "\n";
    return ss.str();
}

namespace {

// The exact integer a constant denotes, without reinterpreting bits:
// integer constants, their widenings, and integral float constants.
std::optional<int64_t> exact_int_constant(const Value* val) {
    if (!val || !val->is_instruction()) return std::nullopt;
    const Instruction* def = val->defining_instruction();
    if (!def) return std::nullopt;
    switch (def->opcode()) {
        case Opcode::iconst_i32: return static_cast<int64_t>(def->imm_i32());
        case Opcode::iconst_i64: return def->imm_i64();
        case Opcode::fconst_f64: {
            const double d = def->imm_f64();
            if (!std::isfinite(d) || d != std::trunc(d) || std::fabs(d) > 9007199254740992.0) return std::nullopt;
            return static_cast<int64_t>(d);
        }
        case Opcode::sext_i64:
        case Opcode::zext_i64:
            if (def->operand_count() < 1 || !def->operand(0)) return std::nullopt;
            if (auto c = exact_int_constant(def->operand(0))) {
                return int_fold::convert(def->opcode(), def->operand(0)->type(), *c);
            }
            return std::nullopt;
        default:
            return std::nullopt;
    }
}

// Same SSA value, or two constants that denote the identical bit pattern
// of the same type.
bool same_value(const Value* a, const Value* b) {
    if (a == b) return true;
    if (!a || !b || a->type() != b->type() || !a->is_instruction() || !b->is_instruction()) return false;
    const Instruction* da = a->defining_instruction();
    const Instruction* db = b->defining_instruction();
    if (!da || !db || da->opcode() != db->opcode()) return false;
    switch (da->opcode()) {
        case Opcode::iconst_i32:
        case Opcode::iconst_i64:
            return da->imm_i64() == db->imm_i64();
        case Opcode::fconst_f64: {
            const double x = da->imm_f64();
            const double y = db->imm_f64();
            return std::memcmp(&x, &y, sizeof(double)) == 0;
        }
        default:
            return false;
    }
}

bool is_iv_conversion(Opcode op) {
    return op == Opcode::sitofp_f64_i64 || op == Opcode::sitofp_f64_i32 ||
           op == Opcode::sext_i64 || op == Opcode::zext_i64;
}

// Pure, reads no memory, and cannot trap: safe to execute at a different
// point relative to the loops' memory operations.
bool is_movable(const Instruction* inst) {
    if (!inst || inst->is_terminator()) return false;
    if (inst->has_side_effects() || inst->is_call()) return false;
    switch (inst->opcode()) {
        case Opcode::load:
        case Opcode::load_indexed:
        case Opcode::vload:
        case Opcode::alloca_:
        case Opcode::landing_pad:
            return false;
        case Opcode::sdiv:
        case Opcode::udiv:
        case Opcode::smod:
        case Opcode::umod: {
            if (inst->type().is_float()) return true;
            auto divisor = exact_int_constant(inst->operand(1));
            const unsigned width = int_fold::width_of(inst->type());
            return divisor && width != 0 && !int_fold::division_may_trap(inst->opcode(), width, *divisor);
        }
        default:
            return true;
    }
}

struct FusibleLoopInfo {
    LoopInfo* loop = nullptr;
    BasicBlock* preheader = nullptr;
    BasicBlock* header = nullptr;
    BasicBlock* latch = nullptr;
    BasicBlock* exit_bb = nullptr;
    size_t iv_index = 0;
    Value* iv_param = nullptr;
    Value* init_val = nullptr;
    Value* limit_val = nullptr;
    Instruction* cmp_inst = nullptr;
    std::optional<Opcode> cmp_conversion;
    int64_t step = 1;
    Opcode cmp_opcode = Opcode::slt;
    bool exit_on_false = true;
};

// The loop's sole outside predecessor when it ends in an unconditional branch
// to the header. Found rather than created, so analysis never edits the CFG.
BasicBlock* find_preheader(const LoopInfo& loop) {
    BasicBlock* header = loop.header();
    if (!header) return nullptr;
    BasicBlock* found = nullptr;
    for (BasicBlock* pred : header->predecessors()) {
        if (!pred || loop.contains(pred)) continue;
        if (found && found != pred) return nullptr;
        found = pred;
    }
    if (!found) return nullptr;
    const Instruction* term = found->terminator();
    if (!term || term->opcode() != Opcode::br || term->branch_target().block != header) return nullptr;
    return found;
}

bool extract_fusible_loop(LoopInfo& loop, FusibleLoopInfo& info) {
    BasicBlock* header = loop.header();
    BasicBlock* preheader = loop.preheader() ? loop.preheader() : find_preheader(loop);
    if (!header || !preheader || loop.latches().size() != 1 || loop.blocks().size() != 2) return false;
    BasicBlock* latch = loop.latches()[0];
    if (!latch || latch == header) return false;

    Instruction* ph_term = preheader->terminator();
    Instruction* latch_term = latch->terminator();
    Instruction* hdr_term = header->terminator();
    if (!ph_term || ph_term->opcode() != Opcode::br || ph_term->branch_target().block != header) return false;
    if (!latch_term || latch_term->opcode() != Opcode::br || latch_term->branch_target().block != header) return false;
    if (!hdr_term || hdr_term->opcode() != Opcode::br_if) return false;
    const BranchTarget& ph_bt = ph_term->branch_target();
    const BranchTarget& latch_bt = latch_term->branch_target();
    if (ph_bt.args.size() != header->param_count() || latch_bt.args.size() != header->param_count()) return false;

    bool exit_on_false = true;
    BasicBlock* exit_bb = nullptr;
    if (hdr_term->true_target().block == latch && !loop.contains(hdr_term->false_target().block)) {
        exit_bb = hdr_term->false_target().block;
        if (!hdr_term->true_target().args.empty()) return false;
    } else if (hdr_term->false_target().block == latch && !loop.contains(hdr_term->true_target().block)) {
        exit_bb = hdr_term->true_target().block;
        exit_on_false = false;
        if (!hdr_term->false_target().args.empty()) return false;
    } else {
        return false;
    }
    if (!exit_bb) return false;

    Value* cond = hdr_term->operand(0);
    Instruction* cmp = (cond && cond->is_instruction()) ? cond->defining_instruction() : nullptr;
    if (!cmp || cmp->parent() != header) return false;
    const Opcode cmp_op = cmp->opcode();
    if (cmp_op != Opcode::slt && cmp_op != Opcode::ult && cmp_op != Opcode::sle && cmp_op != Opcode::ule) return false;
    Value* lhs = cmp->operand(0);
    Value* rhs = cmp->operand(1);
    if (!lhs || !rhs || !loop.is_loop_invariant(rhs)) return false;

    Value* iv_candidate = lhs;
    std::optional<Opcode> conversion;
    if (lhs->is_instruction() && lhs->defining_instruction()->parent() == header &&
        is_iv_conversion(lhs->defining_instruction()->opcode())) {
        conversion = lhs->defining_instruction()->opcode();
        iv_candidate = lhs->defining_instruction()->operand(0);
    }
    if (!iv_candidate || !iv_candidate->is_block_param() || iv_candidate->defining_block() != header) return false;

    const size_t idx = iv_candidate->param_index();
    Value* next = latch_bt.args[idx];
    Instruction* inc = (next && next->is_instruction()) ? next->defining_instruction() : nullptr;
    if (!inc || inc->opcode() != Opcode::add || inc->parent() != latch) return false;
    Value* step_op = inc->operand(0) == iv_candidate ? inc->operand(1)
                   : (inc->operand(1) == iv_candidate ? inc->operand(0) : nullptr);
    auto step = exact_int_constant(step_op);
    if (!step || *step <= 0) return false;

    info.loop = &loop;
    info.preheader = preheader;
    info.header = header;
    info.latch = latch;
    info.exit_bb = exit_bb;
    info.iv_index = idx;
    info.iv_param = iv_candidate;
    info.init_val = ph_bt.args[idx];
    info.limit_val = rhs;
    info.cmp_inst = cmp;
    info.cmp_conversion = conversion;
    info.step = *step;
    info.cmp_opcode = cmp_op;
    info.exit_on_false = exit_on_false;
    return true;
}

bool check_domain_congruence(const FusibleLoopInfo& l1, const FusibleLoopInfo& l2) {
    return l1.iv_param->type() == l2.iv_param->type() &&
           l1.step == l2.step &&
           l1.cmp_opcode == l2.cmp_opcode &&
           l1.exit_on_false == l2.exit_on_false &&
           l1.cmp_conversion == l2.cmp_conversion &&
           l1.cmp_inst->operand(0)->type() == l2.cmp_inst->operand(0)->type() &&
           same_value(l1.init_val, l2.init_val) &&
           same_value(l1.limit_val, l2.limit_val);
}

// The blocks from loop 1's exit to loop 2's header: each entered only from
// the previous one, ending in the preheader of loop 2.
bool collect_bridge(const FusibleLoopInfo& l1, const FusibleLoopInfo& l2, std::vector<BasicBlock*>& bridge) {
    if (l1.loop == l2.loop || l1.loop->parent() != l2.loop->parent()) return false;
    const auto& exit_preds = l1.exit_bb->predecessors();
    if (exit_preds.size() != 1 || exit_preds[0] != l1.header) return false;
    for (const BasicBlock* pred : l2.header->predecessors()) {
        if (pred != l2.preheader && pred != l2.latch) return false;
    }

    BasicBlock* cur = l1.exit_bb;
    while (cur && cur != l2.header) {
        if (bridge.size() > 16) return false;
        if (cur != l1.exit_bb) {
            if (cur->param_count() != 0) return false;
            const auto& preds = cur->predecessors();
            if (preds.size() != 1 || preds[0] != bridge.back()) return false;
        }
        Instruction* term = cur->terminator();
        if (!term || term->opcode() != Opcode::br) return false;
        bridge.push_back(cur);
        cur = term->branch_target().block;
    }
    return cur == l2.header && !bridge.empty() && bridge.back() == l2.preheader;
}

bool uses_any(const Instruction& inst, const std::unordered_set<const Value*>& defs) {
    bool found = false;
    for_each_use(inst, [&](Value* v) { if (defs.count(v)) found = true; });
    return found;
}

struct MemAccess {
    Value* base = nullptr;
    Value* index = nullptr;
    uint8_t scale = 1;
    int32_t offset = 0;
    size_t size = 0;
    bool is_store = false;
    // An element access of a fresh runtime array, addressed only through the
    // runtime's element calls.
    bool runtime_element = false;
};

bool is_elem_set_call(const Instruction& inst) {
    return callee_has_role(inst, SymbolRole::ArraySet) && inst.operand_count() >= 3;
}

bool is_runtime_elem_call(const Instruction& inst) {
    return (callee_has_role(inst, SymbolRole::ArrayGet) && inst.operand_count() == 2) ||
           is_elem_set_call(inst);
}

// A runtime array (a declared `array_new` result) used only as the receiver
// of element calls and write barriers: no other code can reach its elements
// or give it accessors, so its element calls act as plain reads and writes
// of their slot. Array contraction relies on the same property.
bool is_private_runtime_array(const Function& fn, const Value* arr) {
    if (!arr || !arr->is_instruction()) return false;
    const Instruction* def = arr->defining_instruction();
    if (!def || !callee_has_role(*def, SymbolRole::ArrayNew)) return false;
    for (const BasicBlock* bb : fn.blocks()) {
        if (!bb) continue;
        for (const Instruction* inst : *bb) {
            bool as_receiver = inst->operand_count() > 0 && inst->operand(0) == arr &&
                               (is_runtime_elem_call(*inst) || inst->opcode() == Opcode::write_barrier);
            bool other_use = false;
            for_each_use(*inst, [&](Value* v) { if (v == arr) other_use = true; });
            for (size_t i = 1; i < inst->operand_count(); ++i) {
                if (inst->operand(i) == arr) as_receiver = false;
            }
            if (other_use && !as_receiver) return false;
        }
    }
    return true;
}

// The runtime answers a read with the last write to the same slot only for
// plain element indices (see array_contraction.cpp).
bool is_plain_element_index(const Value* idx, const BasicBlock* bb, const RangeAnalysis& ra) {
    if (!idx || (idx->type() != Type::i64() && idx->type() != Type::i32())) return false;
    const ValueRange r = ra.get_range_at(idx, bb);
    return !r.is_empty() && r.min_val >= 0 && r.max_val < INT32_MAX;
}

// Describes the memory `inst` touches, or returns false when it is not a
// plain load or store (those are handled by the effect check).
bool describe_access(const Instruction& inst, MemAccess& acc) {
    switch (inst.opcode()) {
        case Opcode::load:
        case Opcode::vload:
            acc.base = inst.operand(0);
            acc.size = inst.type().size_in_bytes();
            break;
        case Opcode::store:
        case Opcode::vstore:
            acc.base = inst.operand(0);
            acc.size = inst.operand(1) ? inst.operand(1)->type().size_in_bytes() : 0;
            acc.is_store = true;
            break;
        case Opcode::load_indexed:
            acc.base = inst.operand(0);
            acc.index = inst.operand(1);
            acc.size = inst.type().size_in_bytes();
            break;
        case Opcode::store_indexed:
            acc.base = inst.operand(0);
            acc.index = inst.operand(1);
            acc.size = inst.operand(2) ? inst.operand(2)->type().size_in_bytes() : 0;
            acc.is_store = true;
            break;
        default:
            return false;
    }
    acc.scale = inst.scale();
    acc.offset = inst.offset();
    return true;
}

// Rejects anything whose relative order with the other loop's operations
// is observable: calls other than declared allocators, deoptimization,
// exceptions, coroutine switches, and divisions that may trap.
bool effects_reorderable(const Function& fn, const LoopInfo& loop, const RangeAnalysis& ra,
                         std::vector<MemAccess>& accesses) {
    for (BasicBlock* bb : loop.blocks()) {
        for (Instruction* inst : *bb) {
            if (inst->is_terminator()) continue;
            MemAccess acc;
            if (describe_access(*inst, acc)) {
                accesses.push_back(acc);
                continue;
            }
            if (is_runtime_elem_call(*inst)) {
                // Any other element call may run accessors or reach a
                // shared array, and so does not commute with anything.
                if (!is_private_runtime_array(fn, inst->operand(0)) ||
                    !is_plain_element_index(inst->operand(1), bb, ra)) {
                    return false;
                }
                acc.base = inst->operand(0);
                acc.index = inst->operand(1);
                acc.scale = 8;
                acc.size = 8;
                acc.is_store = is_elem_set_call(*inst);
                acc.runtime_element = true;
                accesses.push_back(acc);
                continue;
            }
            const Opcode op = inst->opcode();
            if (op == Opcode::safepoint || op == Opcode::write_barrier) continue;
            if (is_allocation_call(inst)) continue;
            if (inst->has_side_effects() || inst->is_call()) return false;
            if ((op == Opcode::sdiv || op == Opcode::udiv || op == Opcode::smod || op == Opcode::umod) &&
                !is_movable(inst)) {
                return false;
            }
        }
    }
    return true;
}

bool accesses_commute(const MemAccess& a, const MemAccess& b, const FusibleLoopInfo& l1,
                      const FusibleLoopInfo& l2, const AliasAnalysis& aa) {
    if (!a.is_store && !b.is_store) return true;
    if (!a.base || !b.base) return false;
    // A private runtime array is reachable only through its own element
    // calls, so it is disjoint from every other access.
    if ((a.runtime_element || b.runtime_element) && a.base != b.base) return true;
    if (a.runtime_element != b.runtime_element) return true;
    if (!a.runtime_element && aa.alias(a.base, b.base) == AliasResult::NoAlias) return true;
    // Element i of the same buffer on both sides: the fused loop touches it
    // in iteration i only, loop 1 first, exactly as the original order did.
    return a.index == l1.iv_param && b.index == l2.iv_param && a.base == b.base &&
           a.scale == b.scale && a.offset == b.offset && a.size == b.size && a.size != 0 &&
           static_cast<size_t>(a.scale) >= a.size;
}

bool check_dependencies(Function& fn, const FusibleLoopInfo& l1, const FusibleLoopInfo& l2,
                        const std::vector<BasicBlock*>& bridge) {
    // Values that exist only once loop 1 has run (or is running).
    std::unordered_set<const Value*> loop1_defs;
    for (BasicBlock* bb : l1.loop->blocks()) {
        for (Value* p : bb->params()) loop1_defs.insert(p);
        for (Instruction* inst : *bb) if (inst->result()) loop1_defs.insert(inst->result());
    }
    for (Value* p : l1.exit_bb->params()) loop1_defs.insert(p);

    // The bridge moves in front of loop 1.
    for (BasicBlock* bb : bridge) {
        for (Instruction* inst : *bb) {
            if (uses_any(*inst, loop1_defs)) return false;
            if (!inst->is_terminator() && !is_movable(inst)) return false;
        }
    }
    // Loop 2 runs interleaved with loop 1: it may not read loop 1's values.
    for (BasicBlock* bb : l2.loop->blocks()) {
        for (Instruction* inst : *bb) {
            if (uses_any(*inst, loop1_defs)) return false;
        }
    }
    // Loop 2's header joins loop 1's header, which runs once more than a body.
    for (Instruction* inst : *l2.header) {
        if (!inst->is_terminator() && !is_movable(inst)) return false;
    }

    std::vector<MemAccess> acc1;
    std::vector<MemAccess> acc2;
    RangeAnalysis ra(fn);
    if (!effects_reorderable(fn, *l1.loop, ra, acc1) || !effects_reorderable(fn, *l2.loop, ra, acc2)) return false;
    if (acc1.empty() || acc2.empty()) return true;
    AliasAnalysis aa(fn);
    for (const MemAccess& a : acc1) {
        for (const MemAccess& b : acc2) {
            if (!accesses_commute(a, b, l1, l2, aa)) return false;
        }
    }
    return true;
}

bool analyze_pair(Function& fn, LoopInfo& loop1, LoopInfo& loop2, const LoopFusionOptions& options,
                  FusibleLoopInfo& info1, FusibleLoopInfo& info2, std::vector<BasicBlock*>& bridge) {
    if (options.stats) options.stats->candidates_checked++;
    if (!extract_fusible_loop(loop1, info1) || !extract_fusible_loop(loop2, info2)) return false;
    if (!collect_bridge(info1, info2, bridge)) {
        if (options.stats) options.stats->rejected_non_adjacent++;
        return false;
    }
    if (!check_domain_congruence(info1, info2)) {
        if (options.stats) options.stats->rejected_domain_mismatch++;
        return false;
    }
    if (!check_dependencies(fn, info1, info2, bridge)) {
        if (options.stats) options.stats->rejected_dependencies++;
        return false;
    }
    // Loop 1's exit values move to loop 2's exit, which must be its own.
    if (info1.exit_bb->param_count() != 0) {
        const auto& preds = info2.exit_bb->predecessors();
        if (preds.size() != 1 || preds[0] != info2.header) return false;
    }
    return true;
}

void remap_uses(Instruction& inst, const std::unordered_map<const Value*, Value*>& repl) {
    auto sub = [&](Value*& v) {
        auto it = repl.find(v);
        if (it != repl.end()) v = it->second;
    };
    for_each_use_slot(inst, sub);
}

void move_before(BasicBlock* from, BasicBlock* to, Instruction* before,
                 const std::unordered_map<const Value*, Value*>& repl) {
    std::vector<Instruction*> insts;
    for (Instruction* inst : *from) {
        if (!inst->is_terminator()) insts.push_back(inst);
    }
    for (Instruction* inst : insts) {
        from->remove_instruction(inst);
        remap_uses(*inst, repl);
        to->insert_before(inst, before);
    }
}

void do_fuse(Function& fn, const FusibleLoopInfo& info1, const FusibleLoopInfo& info2,
             const std::vector<BasicBlock*>& bridge) {
    BasicBlock* hdr1 = info1.header;
    BasicBlock* hdr2 = info2.header;
    Instruction* ph1_term = info1.preheader->terminator();
    Instruction* hdr1_term = hdr1->terminator();
    Instruction* latch1_term = info1.latch->terminator();
    Instruction* hdr2_term = hdr2->terminator();
    Instruction* ph2_term = info2.preheader->terminator();
    Instruction* latch2_term = info2.latch->terminator();

    // Loop 2's carried values become extra parameters of loop 1's header;
    // its induction variable is loop 1's.
    std::unordered_map<const Value*, Value*> repl;
    repl[info2.iv_param] = info1.iv_param;
    std::vector<size_t> carried;
    for (size_t i = 0; i < hdr2->param_count(); ++i) {
        if (i == info2.iv_index) continue;
        repl[hdr2->param(i)] = ir::new_block_param(fn, hdr1, hdr2->param(i)->type());
        carried.push_back(i);
    }

    // The bridge (pure, independent of loop 1) runs before loop 1 now.
    for (BasicBlock* bb : bridge) move_before(bb, info1.preheader, ph1_term, repl);
    for (size_t i : carried) ph1_term->branch_target().args.push_back(ph2_term->branch_target().args[i]);

    move_before(hdr2, hdr1, hdr1_term, repl);
    move_before(info2.latch, info1.latch, latch1_term, repl);
    for (size_t i : carried) {
        Value* next = latch2_term->branch_target().args[i];
        auto it = repl.find(next);
        latch1_term->branch_target().args.push_back(it != repl.end() ? it->second : next);
    }

    // The fused header exits to loop 2's exit, carrying both loops' results.
    BranchTarget& exit1 = info1.exit_on_false ? hdr1_term->false_target() : hdr1_term->true_target();
    const BranchTarget& exit2 = info2.exit_on_false ? hdr2_term->false_target() : hdr2_term->true_target();
    std::vector<Value*> loop1_exit_args = exit1.args;
    std::vector<Value*> fused_args;
    for (Value* a : exit2.args) {
        auto it = repl.find(a);
        fused_args.push_back(it != repl.end() ? it->second : a);
    }
    for (size_t k = 0; k < info1.exit_bb->param_count(); ++k) {
        repl[info1.exit_bb->param(k)] = ir::new_block_param(fn, info2.exit_bb, info1.exit_bb->param(k)->type());
        fused_args.push_back(loop1_exit_args[k]);
    }
    exit1 = BranchTarget(info2.exit_bb, std::move(fused_args));

    // Uses after the loops of loop 2's parameters and loop 1's exit values.
    for (BasicBlock* bb : fn.blocks()) {
        if (!bb) continue;
        for (Instruction* inst : *bb) remap_uses(*inst, repl);
    }

    for (BasicBlock* bb : bridge) fn.remove_block(bb);
    fn.remove_block(info2.latch);
    fn.remove_block(hdr2);
    fn.rebuild_cfg_predecessors();
}

} // namespace

bool can_fuse_loops(
    Function& fn,
    LoopInfo& loop1,
    LoopInfo& loop2,
    const DominatorTree& dom,
    const LoopFusionOptions& options
) {
    (void)dom;
    FusibleLoopInfo info1, info2;
    std::vector<BasicBlock*> bridge;
    return analyze_pair(fn, loop1, loop2, options, info1, info2, bridge);
}

bool fuse_loops(
    Function& fn,
    LoopInfo& loop1,
    LoopInfo& loop2,
    const DominatorTree& dom,
    const LoopFusionOptions& options
) {
    (void)dom;
    FusibleLoopInfo info1, info2;
    std::vector<BasicBlock*> bridge;
    if (!analyze_pair(fn, loop1, loop2, options, info1, info2, bridge)) return false;
    do_fuse(fn, info1, info2, bridge);
    if (options.stats) options.stats->loops_fused++;
    return true;
}

bool loop_fusion_pass(
    Function& fn,
    const DominatorTree& dom,
    const LoopFusionOptions& options
) {
    (void)dom;
    bool any_fused = false;
    bool changed = true;

    while (changed) {
        changed = false;
        fn.rebuild_cfg_predecessors();
        DominatorTree current_dom(fn);
        LoopAnalysis la(fn, current_dom);

        const auto& loops = la.top_level_loops();
        for (size_t i = 0; i < loops.size() && !changed; ++i) {
            for (size_t j = 0; j < loops.size() && !changed; ++j) {
                if (i == j) continue;
                if (fuse_loops(fn, *loops[i], *loops[j], current_dom, options)) {
                    changed = true;
                    any_fused = true;
                }
            }
        }
    }

    return any_fused;
}

} // namespace brass
