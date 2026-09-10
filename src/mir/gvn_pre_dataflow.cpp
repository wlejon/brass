#include "gvn_pre_dataflow.hpp"
#include <brass/mir/opcodes.hpp>
#include <algorithm>
#include <cstring>

namespace brass {

bool is_pre_commutative_op(Opcode op) noexcept {
    switch (op) {
        case Opcode::add:
        case Opcode::mul:
        case Opcode::and_:
        case Opcode::or_:
        case Opcode::xor_:
        case Opcode::eq:
        case Opcode::ne:
        case Opcode::vadd:
        case Opcode::vmul:
        case Opcode::vand:
        case Opcode::vor:
        case Opcode::vxor:
            return true;
        default:
            return false;
    }
}

bool is_pre_candidate_op(const Instruction* inst) noexcept {
    if (!inst || !inst->produces_value()) return false;
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
            if (inst->operand_count() < 2 || !inst->operand(1)) return false;
            const Value* denom = inst->operand(1);
            if (denom->is_instruction()) {
                const Instruction* ddef = denom->defining_instruction();
                if (ddef && (ddef->opcode() == Opcode::iconst_i32 || ddef->opcode() == Opcode::iconst_i64)) {
                    return ddef->imm_i64() != 0;
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

    // Canonicalize commutative operations
    if (is_pre_commutative_op(expr.opcode) && expr.op0 && expr.op1) {
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

PreDataflow::PreDataflow(
    Function& fn,
    const DominatorTree& dom,
    const AliasAnalysis& aa
) : fn_(fn), dom_(dom), aa_(aa) {}

void PreDataflow::analyze_expression(
    const PreExpression& expr,
    const Instruction* exemplar,
    const std::unordered_map<const Value*, const Value*>& leaders
) {
    local_info_.clear();
    ant_in_.clear();
    ant_out_.clear();
    avail_at_exit_.clear();

    compute_local_info(expr, exemplar, leaders);
    compute_anticipation();
    compute_availability();
}

void PreDataflow::compute_local_info(
    const PreExpression& expr,
    const Instruction* exemplar,
    const std::unordered_map<const Value*, const Value*>& leaders
) {
    for (BasicBlock* bb : fn_.blocks()) {
        if (!bb) continue;
        BlockLocalInfo info;
        info.transp = true;
        info.ant_loc = false;
        info.avail_loc = false;
        info.avail_val = nullptr;

        bool operand_killed_before_eval = false;
        bool memory_clobbered_before_eval = false;

        for (Instruction* inst : *bb) {
            if (!inst) continue;

            // Check if inst evaluates expr
            PreExpression inst_expr = PreExpression::from_instruction(inst, leaders);
            if (inst_expr == expr) {
                info.evaluations.push_back(inst);
                if (!info.ant_loc && !operand_killed_before_eval && !memory_clobbered_before_eval) {
                    info.ant_loc = true;
                }
                info.avail_loc = true;
                info.avail_val = inst->result();
                continue;
            }

            // Check if inst defines any operand of expr
            Value* res = inst->result();
            if (res && (res == expr.op0 || res == expr.op1 || res == expr.op2)) {
                operand_killed_before_eval = true;
                info.transp = false;
                info.avail_loc = false;
                info.avail_val = nullptr;
            }

            // For load expressions, check if inst clobbers memory
            if (expr.is_load() && exemplar) {
                // Check if inst can clobber exemplar
                if (aa_.can_clobber(inst, exemplar)) {
                    // Check for must-alias store-to-load forwarding
                    Opcode op = inst->opcode();
                    if ((op == Opcode::store || op == Opcode::vstore) &&
                        inst->memory_type() == expr.memory_type &&
                        expr.op0 != nullptr &&
                        aa_.alias(inst->operand(0), inst->offset(), inst->memory_type(),
                                  expr.op0, expr.offset, expr.memory_type) == AliasResult::MustAlias) {
                        info.avail_loc = true;
                        info.avail_val = inst->operand(1);
                    } else {
                        memory_clobbered_before_eval = true;
                        info.transp = false;
                        info.avail_loc = false;
                        info.avail_val = nullptr;
                    }
                }
            }
        }

        local_info_[bb] = std::move(info);
    }
}

void PreDataflow::compute_anticipation() {
    // Initialize: AntIn and AntOut default to true, except exit blocks
    for (const BasicBlock* bb : fn_.blocks()) {
        if (!bb) continue;
        ant_in_[bb] = true;
        ant_out_[bb] = true;
    }

    bool changed = true;
    while (changed) {
        changed = false;
        const auto& blocks = fn_.blocks();
        for (auto it = blocks.rbegin(); it != blocks.rend(); ++it) {
            const BasicBlock* bb = *it;
            if (!bb) continue;

            bool new_out = true;
            auto succs = bb->successors();
            if (succs.empty()) {
                new_out = false;
            } else {
                for (const BasicBlock* succ : succs) {
                    if (succ) {
                        auto s_it = ant_in_.find(succ);
                        if (s_it != ant_in_.end() && !s_it->second) {
                            new_out = false;
                            break;
                        }
                    }
                }
            }

            const auto& info = local_info_[bb];
            bool new_in = info.ant_loc || (info.transp && new_out);

            if (new_in != ant_in_[bb] || new_out != ant_out_[bb]) {
                ant_in_[bb] = new_in;
                ant_out_[bb] = new_out;
                changed = true;
            }
        }
    }
}

void PreDataflow::compute_availability() {
    for (const BasicBlock* bb : fn_.blocks()) {
        if (!bb) continue;
        const auto& info = local_info_[bb];
        if (info.avail_loc && info.avail_val) {
            avail_at_exit_[bb] = info.avail_val;
        } else {
            avail_at_exit_[bb] = nullptr;
        }
    }

    // Forward propagation across transparent blocks
    bool changed = true;
    while (changed) {
        changed = false;
        for (const BasicBlock* bb : fn_.blocks()) {
            if (!bb) continue;
            const auto& info = local_info_[bb];
            if (info.avail_loc) continue;
            if (!info.transp) continue;
            if (bb->predecessors().empty()) continue;

            Value* common_val = nullptr;
            bool all_same = true;
            for (const BasicBlock* pred : bb->predecessors()) {
                if (!pred) {
                    all_same = false;
                    break;
                }
                auto it = avail_at_exit_.find(pred);
                if (it == avail_at_exit_.end() || it->second == nullptr) {
                    all_same = false;
                    break;
                }
                if (!common_val) {
                    common_val = it->second;
                } else if (common_val != it->second) {
                    all_same = false;
                    break;
                }
            }

            if (all_same && common_val && avail_at_exit_[bb] != common_val) {
                avail_at_exit_[bb] = common_val;
                changed = true;
            }
        }
    }
}

bool PreDataflow::is_anticipated_at_entry(const BasicBlock* bb) const {
    if (!bb) return false;
    auto it = ant_in_.find(bb);
    return (it != ant_in_.end()) ? it->second : false;
}

bool PreDataflow::is_anticipated_at_exit(const BasicBlock* bb) const {
    if (!bb) return false;
    auto it = ant_out_.find(bb);
    return (it != ant_out_.end()) ? it->second : false;
}

Value* PreDataflow::available_at_exit(const BasicBlock* bb) const {
    if (!bb) return nullptr;
    auto it = avail_at_exit_.find(bb);
    return (it != avail_at_exit_.end()) ? it->second : nullptr;
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
    if (!bb) return default_local_info_;
    auto it = local_info_.find(bb);
    return (it != local_info_.end()) ? it->second : default_local_info_;
}

} // namespace brass
