#include "gvn_pre_dataflow.hpp"
#include "int_fold.hpp"
#include "gvn_table.hpp"
#include <brass/mir/opcodes.hpp>
#include <brass/mir/gc_refs.hpp>
#include <algorithm>
#include <cstring>

namespace brass {

bool is_pre_commutative_op(Opcode op, Type type) noexcept {
    return is_commutative_op(op, type);
}

bool is_pre_candidate_op(const Instruction* inst) noexcept {
    if (!inst || !inst->produces_value()) return false;
    // PRE would carry a derived gcref through a block parameter (gc_refs.hpp).
    if (is_derived_gcref(inst->result())) return false;
    Opcode op = inst->opcode();
    if (op == Opcode::load || op == Opcode::vload) {
        return true;
    }
    switch (op) {
        case Opcode::sext_i64:
        case Opcode::zext_i64:
        case Opcode::trunc_i32:
        case Opcode::fptosi_i32:
        case Opcode::fptosi_i64:
        case Opcode::sitofp_f64_i32:
        case Opcode::sitofp_f64_i64:
        case Opcode::bitcast_i64_f64:
        case Opcode::bitcast_f64_i64:
        case Opcode::add:
        case Opcode::sub:
        case Opcode::mul:
        case Opcode::neg:
        case Opcode::and_:
        case Opcode::or_:
        case Opcode::xor_:
        case Opcode::shl:
        case Opcode::lshr:
        case Opcode::ashr:
        case Opcode::not_:
        case Opcode::clz:
        case Opcode::ctz:
        case Opcode::popcnt:
        case Opcode::eq:
        case Opcode::ne:
        case Opcode::slt:
        case Opcode::ult:
        case Opcode::sle:
        case Opcode::ule:
        case Opcode::sgt:
        case Opcode::ugt:
        case Opcode::sge:
        case Opcode::uge:
        case Opcode::select:
        case Opcode::vadd:
        case Opcode::vsub:
        case Opcode::vmul:
        case Opcode::vdiv:
        case Opcode::vneg:
        case Opcode::vmin:
        case Opcode::vmax:
        case Opcode::vsqrt:
        case Opcode::vand:
        case Opcode::vor:
        case Opcode::vxor:
        case Opcode::vnot:
        case Opcode::vbroadcast:
        case Opcode::vextract_lane:
        case Opcode::vinsert_lane:
        case Opcode::vshuffle:
        case Opcode::vzero:
            return true;
        case Opcode::sdiv:
        case Opcode::udiv:
        case Opcode::smod:
        case Opcode::umod: {
            // PRE may insert the division on a path that did not evaluate it,
            // so only a constant divisor that rules out every trap qualifies.
            if (inst->operand_count() < 2 || !inst->operand(1)) return false;
            const Value* denom = inst->operand(1);
            if (denom->is_instruction()) {
                const Instruction* ddef = denom->defining_instruction();
                if (ddef && (ddef->opcode() == Opcode::iconst_i32 || ddef->opcode() == Opcode::iconst_i64)) {
                    const int64_t divisor = ddef->opcode() == Opcode::iconst_i32
                        ? static_cast<int64_t>(ddef->imm_i32()) : ddef->imm_i64();
                    const unsigned width = int_fold::width_of(inst->type());
                    return width != 0 && !int_fold::division_may_trap(op, width, divisor);
                }
            }
            return false;
        }
        default:
            return false;
    }
}

PreExpression PreExpression::from_instruction(
    const Instruction* inst,
    const std::unordered_map<const Value*, const Value*>& leaders
) {
    PreExpression expr;
    if (!inst) return expr;

    expr.opcode = inst->opcode();
    expr.type = inst->type();
    expr.offset = inst->offset();
    expr.scale = inst->scale();
    expr.memory_type = inst->memory_type();
    expr.symbol = inst->symbol();

    if (inst->opcode() == Opcode::fconst_f64) {
        double f = inst->imm_f64();
        std::memcpy(&expr.imm_bits, &f, sizeof(double));
    } else {
        expr.imm_bits = static_cast<uint64_t>(inst->imm_i64());
    }

    auto resolve = [&](const Value* v) -> const Value* {
        if (!v) return nullptr;
        auto it = leaders.find(v);
        return (it != leaders.end()) ? it->second : v;
    };

    size_t op_count = inst->operand_count();
    if (op_count >= 1) expr.op0 = resolve(inst->operand(0));
    if (op_count >= 2) expr.op1 = resolve(inst->operand(1));
    if (op_count >= 3) expr.op2 = resolve(inst->operand(2));

    // Canonicalize commutative operations. Operands of different types
    // (ptr + i64 offset) keep their order: the expression is rebuilt from
    // it when hoisted, and only the pointer may come first.
    if (is_pre_commutative_op(expr.opcode, expr.type) && expr.op0 && expr.op1 &&
        expr.op0->type() == expr.op1->type()) {
        if (expr.op0->id() > expr.op1->id()) {
            std::swap(expr.op0, expr.op1);
        }
    }

    return expr;
}

bool PreExpression::operator==(const PreExpression& other) const noexcept {
    return opcode == other.opcode &&
           type == other.type &&
           op0 == other.op0 &&
           op1 == other.op1 &&
           op2 == other.op2 &&
           imm_bits == other.imm_bits &&
           offset == other.offset &&
           scale == other.scale &&
           memory_type == other.memory_type &&
           symbol == other.symbol;
}

size_t PreExprHash::operator()(const PreExpression& k) const noexcept {
    size_t h = static_cast<size_t>(k.opcode);
    h = h * 31 + static_cast<size_t>(k.type.kind());
    h = h * 31 + std::hash<const void*>()(k.op0);
    h = h * 31 + std::hash<const void*>()(k.op1);
    h = h * 31 + std::hash<const void*>()(k.op2);
    h = h * 31 + static_cast<size_t>(k.imm_bits);
    h = h * 31 + static_cast<size_t>(k.offset);
    h = h * 31 + static_cast<size_t>(k.scale);
    h = h * 31 + static_cast<size_t>(k.memory_type.kind());
    return h;
}

// ---- PreFunctionIndex ------------------------------------------------------

void PreFunctionIndex::build(
    Function& fn,
    const std::unordered_map<const Value*, const Value*>& leaders,
    bool include_load_candidates
) {
    blocks.clear();
    block_ordinal.clear();
    positions.clear();
    memory_writers.clear();
    evaluations.clear();
    candidates.clear();
    exemplars.clear();
    stored_memory_types.clear();

    for (BasicBlock* bb : fn.blocks()) {
        if (!bb) continue;
        const uint32_t ordinal = static_cast<uint32_t>(blocks.size());
        blocks.push_back(bb);
        block_ordinal[bb] = ordinal;
        memory_writers.emplace_back();

        uint32_t index = 0;
        for (Instruction* inst : *bb) {
            if (!inst) continue;
            positions[inst] = Position{ordinal, index};

            const Opcode op = inst->opcode();
            if (op == Opcode::store || op == Opcode::store_indexed || op == Opcode::vstore || is_call(op) ||
                is_coro_op(op)) {
                memory_writers.back().push_back(Writer{inst, index});
            }
            if (op == Opcode::store || op == Opcode::vstore) {
                const Type mt = inst->memory_type();
                if (std::find(stored_memory_types.begin(), stored_memory_types.end(), mt) ==
                    stored_memory_types.end()) {
                    stored_memory_types.push_back(mt);
                }
            }
            ++index;

            if (!is_pre_candidate_op(inst)) continue;
            const bool is_load = (op == Opcode::load || op == Opcode::vload);
            if (is_load && !include_load_candidates) continue;

            // Every instruction whose expression equals a candidate's is a
            // candidate itself — candidacy is a function of opcode and, for
            // division, of an operand the expression carries — so the
            // evaluation lists collected here are exactly what a scan
            // comparing every instruction's expression would have found.
            PreExpression expr = PreExpression::from_instruction(inst, leaders);
            auto ev = evaluations.find(expr);
            if (ev == evaluations.end()) {
                evaluations.emplace(expr, std::vector<Instruction*>{inst});
                candidates.push_back(expr);
                exemplars[expr] = inst;
            } else {
                ev->second.push_back(inst);
            }
        }
    }
}

bool PreFunctionIndex::has_store_of_type(Type memory_type) const noexcept {
    return std::find(stored_memory_types.begin(), stored_memory_types.end(), memory_type) !=
           stored_memory_types.end();
}

const BasicBlock* PreFunctionIndex::block_of(const Instruction* inst) const noexcept {
    auto pos = positions.find(inst);
    if (pos == positions.end()) return nullptr;
    return blocks[pos->second.block];
}

// ---- PreDataflow -----------------------------------------------------------

PreDataflow::PreDataflow(
    Function& fn,
    const DominatorTree& dom,
    const AliasAnalysis& aa,
    const PreFunctionIndex& index
) : fn_(fn), dom_(dom), aa_(aa), index_(index) {
    const size_t n = index_.blocks.size();
    local_info_.resize(n);
    is_touched_.assign(n, 0);
    opacity_.assign(n, kOpacityUnknown);
    ant_in_.assign(n, 0);
    avail_at_exit_.assign(n, nullptr);
    for (uint32_t o = 0; o < n; ++o) {
        for (const auto& w : index_.memory_writers[o]) {
            const Opcode op = w.inst->opcode();
            if (op != Opcode::store && op != Opcode::vstore) continue;
            int64_t ignored = 0;
            const Value* base = aa_.get_underlying_base(w.inst->operand(0), ignored);
            if (!base) continue;
            auto& blocks = stores_by_base_[base];
            if (blocks.empty() || blocks.back() != o) blocks.push_back(o);
        }
    }
    succs_.resize(n);
    preds_.resize(n);
    pred_has_unknown_.assign(n, false);
    for (size_t i = 0; i < n; ++i) {
        const BasicBlock* bb = index_.blocks[i];
        for (const BasicBlock* succ : bb->successors()) {
            if (!succ) continue;  // skipped by the sweep, exactly as before
            const uint32_t o = ordinal(succ);
            if (o != kNoBlock) succs_[i].push_back(o);
        }
        for (const BasicBlock* pred : bb->predecessors()) {
            const uint32_t o = pred ? ordinal(pred) : kNoBlock;
            if (o == kNoBlock) {
                pred_has_unknown_[i] = true;
            } else {
                preds_[i].push_back(o);
            }
        }
    }

    // Which blocks reach an exit along the edges anticipation reads
    // (`succs_`), found backwards from the exits over those same edges.
    std::vector<std::vector<uint32_t>>& into = succ_of_;
    into.resize(n);
    for (uint32_t i = 0; i < n; ++i) {
        for (uint32_t s : succs_[i]) into[s].push_back(i);
    }
    std::vector<uint8_t> reaches_exit(n, 0);
    std::vector<uint32_t> work;
    for (uint32_t i = 0; i < n; ++i) {
        if (index_.blocks[i]->successors().empty()) {
            reaches_exit[i] = 1;
            work.push_back(i);
        }
    }
    while (!work.empty()) {
        const uint32_t b = work.back();
        work.pop_back();
        for (uint32_t p : into[b]) {
            if (!reaches_exit[p]) {
                reaches_exit[p] = 1;
                work.push_back(p);
            }
        }
    }
    for (uint32_t i = 0; i < n; ++i) {
        if (!reaches_exit[i]) no_exit_blocks_.push_back(i);
    }
}

uint32_t PreDataflow::ordinal(const BasicBlock* bb) const noexcept {
    if (!bb) return kNoBlock;
    auto it = index_.block_ordinal.find(bb);
    return it == index_.block_ordinal.end() ? kNoBlock : it->second;
}

void PreDataflow::analyze_expression(const PreExpression& expr, const Instruction* exemplar) {
    for (uint32_t o : touched_) {
        local_info_[o] = BlockLocalInfo{};
        is_touched_[o] = 0;
    }
    touched_.clear();
    for (uint32_t o : opacity_set_) opacity_[o] = kOpacityUnknown;
    opacity_set_.clear();

    compute_local_info(expr, exemplar);
    compute_anticipation();
    compute_availability();
}

// The block-local facts about one expression, from its events alone: its
// evaluations, the instructions defining its operands, and — for a load —
// the instructions that may write memory. Each block's events are replayed
// in position order through the state machine a full scan of the block
// would have run; an instruction that both defines an operand and may write
// memory (a call whose result the expression uses) is a kill first and a
// clobber second, the order the scan applied its two checks in. Blocks with
// no events keep the default: transparent, nothing anticipated or available.
void PreDataflow::compute_local_info(const PreExpression& expr, const Instruction* exemplar) {
    enum class EventKind : uint8_t { Evaluation, Kill, Writer };
    struct Event {
        uint32_t block;
        uint32_t position;
        EventKind kind;
        Instruction* inst;
    };

    // The few events every expression has: evaluations and operand kills.
    std::vector<Event> sparse;
    auto add_sparse = [&](Instruction* inst, EventKind kind) {
        auto pos = index_.positions.find(inst);
        if (pos == index_.positions.end()) return;
        sparse.push_back(Event{pos->second.block, pos->second.index, kind, inst});
    };
    if (auto ev = index_.evaluations.find(expr); ev != index_.evaluations.end()) {
        for (Instruction* inst : ev->second) add_sparse(inst, EventKind::Evaluation);
    }
    for (const Value* op : {expr.op0, expr.op1, expr.op2}) {
        if (op && op->is_instruction() && op->defining_instruction()) {
            add_sparse(op->defining_instruction(), EventKind::Kill);
        }
    }
    std::sort(sparse.begin(), sparse.end(), [](const Event& a, const Event& b) {
        if (a.block != b.block) return a.block < b.block;
        if (a.position != b.position) return a.position < b.position;
        return static_cast<uint8_t>(a.kind) < static_cast<uint8_t>(b.kind);
    });

    const bool track_memory = expr.is_load() && exemplar;
    track_memory_ = track_memory;
    exemplar_ = exemplar;

    // One block's events, replayed.
    BlockLocalInfo info;
    bool operand_killed_before_eval = false;
    bool memory_clobbered_before_eval = false;
    auto begin_block = [&] {
        info = BlockLocalInfo{};
        operand_killed_before_eval = false;
        memory_clobbered_before_eval = false;
    };
    auto apply = [&](EventKind kind, Instruction* inst) {
        switch (kind) {
            case EventKind::Evaluation:
                info.evaluations.push_back(inst);
                if (!info.ant_loc && !operand_killed_before_eval && !memory_clobbered_before_eval) {
                    info.ant_loc = true;
                }
                info.avail_loc = true;
                info.avail_val = inst->result();
                break;
            case EventKind::Kill:
                operand_killed_before_eval = true;
                info.transp = false;
                info.avail_loc = false;
                info.avail_val = nullptr;
                break;
            case EventKind::Writer:
                if (aa_.can_clobber(inst, exemplar)) {
                    Opcode op = inst->opcode();
                    if ((op == Opcode::store || op == Opcode::vstore) &&
                        inst->memory_type() == expr.memory_type &&
                        expr.op0 != nullptr &&
                        aa_.alias(inst->operand(0), inst->offset(), inst->memory_type(),
                                  expr.op0, expr.offset, expr.memory_type) == AliasResult::MustAlias) {
                        // The stored value is what the load reads from here
                        // on, but it replaces the location's old value: the
                        // block is not transparent and a load after the
                        // store is not anticipated at the block's entry.
                        memory_clobbered_before_eval = true;
                        info.transp = false;
                        info.avail_loc = true;
                        info.avail_val = inst->operand(1);
                    } else {
                        memory_clobbered_before_eval = true;
                        info.transp = false;
                        info.avail_loc = false;
                        info.avail_val = nullptr;
                    }
                }
                break;
        }
    };
    auto end_block = [&](uint32_t block) {
        local_info_[block] = std::move(info);
        touched_.push_back(block);
        is_touched_[block] = 1;
    };

    size_t s = 0;  // cursor into `sparse`
    if (!track_memory) {
        while (s < sparse.size()) {
            const uint32_t block = sparse[s].block;
            begin_block();
            for (; s < sparse.size() && sparse[s].block == block; ++s) apply(sparse[s].kind, sparse[s].inst);
            end_block(block);
        }
        return;
    }

    // A load. The blocks replayed in full are the ones with an event and the
    // ones holding a store that can MUST-alias it — a store off the same
    // underlying base, since that is the only way `alias` answers MustAlias —
    // because a must-alias store makes the load available without evaluating
    // it. Any other writer can only clobber, which makes its block opaque and
    // nothing else; that is asked lazily (`is_transparent`), for the blocks
    // the dataflow actually reaches. Replaying every block with a writer made
    // each load cost the whole function's writers: in a large function, where
    // nearly every block calls something, that was the pass's entire cost.
    //
    // Writers merge with a block's sparse events by position (sparse first on
    // a tie, which is the kill-before-clobber rule above; an evaluation never
    // shares a position with a writer).
    std::vector<uint32_t> active;
    for (const Event& e : sparse) {
        if (active.empty() || active.back() != e.block) active.push_back(e.block);
    }
    if (expr.op0 != nullptr) {
        int64_t ignored = 0;
        const Value* base = aa_.get_underlying_base(expr.op0, ignored);
        if (const auto it = stores_by_base_.find(base); base && it != stores_by_base_.end()) {
            active.insert(active.end(), it->second.begin(), it->second.end());
            std::sort(active.begin(), active.end());
            active.erase(std::unique(active.begin(), active.end()), active.end());
        }
    }
    for (const uint32_t block : active) {
        const auto& writers = index_.memory_writers[block];
        begin_block();
        size_t w = 0;
        while (w < writers.size() || (s < sparse.size() && sparse[s].block == block)) {
            const bool sparse_next = s < sparse.size() && sparse[s].block == block &&
                                     (w >= writers.size() || sparse[s].position <= writers[w].index);
            if (sparse_next) {
                apply(sparse[s].kind, sparse[s].inst);
                ++s;
            } else {
                apply(EventKind::Writer, writers[w].inst);
                ++w;
            }
        }
        end_block(block);
    }
}

// AntIn(b)  = ant_loc(b) || (transp(b) && AntOut(b))
// AntOut(b) = b has successors && every successor has AntIn
// as a GREATEST fixpoint, computed on a superset of the blocks that can come
// out true rather than on every block.
//
// The superset: a block anticipating the expression without evaluating it is
// transparent, and so is every block on a path from it until that path meets
// an evaluation — which it must, before any exit, because an exit anticipates
// only what it evaluates. So the block is found walking BACKWARDS from the
// evaluating blocks through transparent ones. The one other way to be true is
// vacuous: a transparent block with no path to an exit at all (an endless
// loop), which the walk may miss; those are `no_exit_blocks_`, added outright.
// Every block outside the superset is false, exactly as the dense sweep found
// it, and inside it the fixpoint runs as before, from true downwards.
void PreDataflow::compute_anticipation() {
    for (uint32_t o : ant_set_) ant_in_[o] = 0;
    ant_set_.clear();

    std::vector<uint32_t> work;
    auto admit = [&](uint32_t o) {
        if (ant_in_[o]) return;
        ant_in_[o] = 1;
        ant_set_.push_back(o);
        work.push_back(o);
    };
    for (uint32_t o : touched_) {
        if (local_info_[o].ant_loc) admit(o);
    }
    for (uint32_t o : no_exit_blocks_) {
        if (local_info_[o].ant_loc || transparent(o)) admit(o);
    }
    while (!work.empty()) {
        const uint32_t b = work.back();
        work.pop_back();
        for (uint32_t p : succ_of_[b]) {
            if (local_info_[p].ant_loc || transparent(p)) admit(p);
        }
    }

    // Downwards from true: a block falls when an exit or a false successor
    // leaves nothing to anticipate below it, and its predecessors in the set
    // are looked at again.
    work = ant_set_;
    while (!work.empty()) {
        const uint32_t b = work.back();
        work.pop_back();
        if (!ant_in_[b] || local_info_[b].ant_loc) continue;
        bool out = !index_.blocks[b]->successors().empty();
        for (uint32_t s : succs_[b]) {
            if (!out) break;
            out = ant_in_[s] != 0;
        }
        if (out) continue;
        ant_in_[b] = 0;
        for (uint32_t p : succ_of_[b]) {
            if (ant_in_[p] && !local_info_[p].ant_loc) work.push_back(p);
        }
    }
    ant_set_.erase(std::remove_if(ant_set_.begin(), ant_set_.end(),
                                  [&](uint32_t o) { return ant_in_[o] == 0; }),
                   ant_set_.end());
    std::sort(ant_set_.begin(), ant_set_.end());
}

// Forward across transparent blocks: a block without its own value takes the
// one every predecessor agrees on. Only blocks downstream of a block that
// makes the value available can take one, so the walk starts there. A block's
// value, once set, never changes (it would need a predecessor's to change
// first), so the order blocks are visited in cannot matter.
void PreDataflow::compute_availability() {
    for (uint32_t o : avail_set_) avail_at_exit_[o] = nullptr;
    avail_set_.clear();

    std::vector<uint32_t> work;
    for (uint32_t o : touched_) {
        const BlockLocalInfo& info = local_info_[o];
        if (info.avail_loc && info.avail_val) {
            avail_at_exit_[o] = info.avail_val;
            avail_set_.push_back(o);
            work.insert(work.end(), succs_[o].begin(), succs_[o].end());
        }
    }
    while (!work.empty()) {
        const uint32_t i = work.back();
        work.pop_back();
        if (local_info_[i].avail_loc) continue;
        if (!transparent(i)) continue;
        if (index_.blocks[i]->predecessors().empty()) continue;
        if (pred_has_unknown_[i]) continue;

        Value* common_val = nullptr;
        bool all_same = true;
        for (uint32_t pred : preds_[i]) {
            Value* v = avail_at_exit_[pred];
            if (v == nullptr) {
                all_same = false;
                break;
            }
            if (!common_val) {
                common_val = v;
            } else if (common_val != v) {
                all_same = false;
                break;
            }
        }

        if (all_same && common_val && avail_at_exit_[i] != common_val) {
            if (avail_at_exit_[i] == nullptr) avail_set_.push_back(i);
            avail_at_exit_[i] = common_val;
            work.insert(work.end(), succs_[i].begin(), succs_[i].end());
        }
    }
}

bool PreDataflow::transparent(uint32_t o) const {
    if (is_touched_[o]) return local_info_[o].transp;
    if (!track_memory_) return true;
    if (opacity_[o] == kOpacityUnknown) {
        bool clear = true;
        for (const auto& w : index_.memory_writers[o]) {
            if (aa_.can_clobber(w.inst, exemplar_)) {
                clear = false;
                break;
            }
        }
        opacity_[o] = clear ? kTransparent : kOpaque;
        opacity_set_.push_back(o);
    }
    return opacity_[o] == kTransparent;
}

bool PreDataflow::is_transparent(const BasicBlock* bb) const {
    const uint32_t o = ordinal(bb);
    return o == kNoBlock || transparent(o);
}

bool PreDataflow::is_anticipated_at_entry(const BasicBlock* bb) const {
    const uint32_t o = ordinal(bb);
    return o != kNoBlock && ant_in_[o] != 0;
}

bool PreDataflow::is_anticipated_at_exit(const BasicBlock* bb) const {
    const uint32_t o = ordinal(bb);
    if (o == kNoBlock || index_.blocks[o]->successors().empty()) return false;
    for (uint32_t s : succs_[o]) {
        if (!ant_in_[s]) return false;
    }
    return true;
}

std::vector<BasicBlock*> PreDataflow::anticipated_blocks() const {
    std::vector<BasicBlock*> out;
    out.reserve(ant_set_.size());
    for (uint32_t o : ant_set_) out.push_back(index_.blocks[o]);
    return out;
}

std::vector<BasicBlock*> PreDataflow::evaluation_blocks() const {
    std::vector<BasicBlock*> out;
    for (uint32_t o : touched_) {
        if (!local_info_[o].evaluations.empty()) out.push_back(index_.blocks[o]);
    }
    return out;
}

Value* PreDataflow::available_at_exit(const BasicBlock* bb) const {
    const uint32_t o = ordinal(bb);
    return o == kNoBlock ? nullptr : avail_at_exit_[o];
}

bool PreDataflow::can_evaluate_at_end(
    const BasicBlock* bb,
    const PreExpression& expr,
    const Instruction* /* exemplar */
) const {
    if (!bb) return false;

    // Verify all operands dominate the end of bb
    auto check_operand = [&](const Value* val) -> bool {
        if (!val) return true;
        if (val->is_block_param()) {
            const BasicBlock* def_bb = val->defining_block();
            return def_bb && dom_.dominates(def_bb, bb);
        } else if (val->is_instruction()) {
            const Instruction* def_inst = val->defining_instruction();
            if (!def_inst) return false;
            if (def_inst->opcode() == Opcode::iconst_i32 ||
                def_inst->opcode() == Opcode::iconst_i64 ||
                def_inst->opcode() == Opcode::fconst_f64) {
                return true;
            }
            const BasicBlock* def_bb = def_inst->parent();
            if (!def_bb) return false;
            if (def_bb == bb) {
                // Defined in bb before the terminator
                return true;
            }
            return dom_.dominates(def_bb, bb);
        }
        return true;
    };

    if (!check_operand(expr.op0)) return false;
    if (!check_operand(expr.op1)) return false;
    if (!check_operand(expr.op2)) return false;

    return true;
}

const BlockLocalInfo& PreDataflow::get_local_info(const BasicBlock* bb) const {
    const uint32_t o = ordinal(bb);
    return o == kNoBlock ? default_local_info_ : local_info_[o];
}

} // namespace brass
