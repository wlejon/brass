#include <brass/mir/f64_demote.hpp>
#include <brass/mir/function.hpp>
#include <brass/mir/instruction.hpp>
#include <brass/mir/block.hpp>
#include <brass/mir/loop_analysis.hpp>
#include <brass/mir/runtime_symbols.hpp>
#include "f64_demote_internal.hpp"
#include <sstream>
#include <iomanip>
#include <algorithm>
#include <vector>
#include <unordered_map>
#include <cmath>

namespace brass {

const char* demote_refusal_reason_string(DemoteRefusalReason reason) {
    switch (reason) {
        case DemoteRefusalReason::None:
            return "none";
        case DemoteRefusalReason::NonIntegralStep:
            return "non-integral step / induction variable";
        case DemoteRefusalReason::EscapingValue:
            return "escaping value / phi used outside loop";
        case DemoteRefusalReason::UnprovenRangeOrOverflow:
            return "unproven range / value >= 2^53 bound";
        case DemoteRefusalReason::UnprovenDivision:
            return "unproven division";
        case DemoteRefusalReason::CallInBody:
            return "call in body";
        case DemoteRefusalReason::NonIntegralConstantOrFloatOp:
            return "non-integral constant / float operations";
        case DemoteRefusalReason::DynamicOrNonF64State:
            return "dynamic / non-f64 state";
        case DemoteRefusalReason::Unprofitable:
            return "unprofitable";
        case DemoteRefusalReason::Other:
            return "other";
    }
    return "other";
}

void DemoteStats::add_record(LoopDemoteRecord rec) {
    if (rec.demoted) {
        loops_demoted++;
    } else {
        loops_refused++;
        refusal_counts[rec.refusal_reason]++;
    }
    total_loops_considered++;
    loop_records.push_back(std::move(rec));
}

std::string DemoteStats::format_report() const {
    std::ostringstream os;
    os << "=== F64 Demotion Statistics ===\n\n";
    if (!module_name.empty()) {
        os << "Module: " << module_name << "\n";
    }

    std::unordered_map<std::string, std::vector<const LoopDemoteRecord*>> fn_to_records;
    std::vector<std::string> fn_order;
    for (const auto& rec : loop_records) {
        if (fn_to_records[rec.function_name].empty()) {
            fn_order.push_back(rec.function_name);
        }
        fn_to_records[rec.function_name].push_back(&rec);
    }

    for (const auto& fn_name : fn_order) {
        os << "  Function " << fn_name << ":\n";
        for (const auto* r : fn_to_records[fn_name]) {
            os << "    Loop " << r->loop_header_block << ": ";
            if (r->demoted) {
                os << "DEMOTED (i64 loop)\n";
            } else {
                os << "REFUSED (" << demote_refusal_reason_string(r->refusal_reason) << ")\n";
            }
        }
    }

    os << "\nTotal:\n";
    double demote_pct = total_loops_considered > 0 ? (100.0 * static_cast<double>(loops_demoted) / static_cast<double>(total_loops_considered)) : 0.0;
    double refuse_pct = total_loops_considered > 0 ? (100.0 * static_cast<double>(loops_refused) / static_cast<double>(total_loops_considered)) : 0.0;
    os << "  Loops Considered:  " << total_loops_considered << "\n";
    os << "  Demoted to i64:    " << loops_demoted << " (" << std::fixed << std::setprecision(1) << demote_pct << "%)\n";
    os << "  Refused:           " << loops_refused << " (" << std::fixed << std::setprecision(1) << refuse_pct << "%)\n";

    if (!refusal_counts.empty()) {
        os << "  Top refusal reasons:\n";
        std::vector<std::pair<DemoteRefusalReason, size_t>> sorted_reasons(refusal_counts.begin(), refusal_counts.end());
        std::sort(sorted_reasons.begin(), sorted_reasons.end(), [](const auto& a, const auto& b) {
            return a.second > b.second;
        });
        for (const auto& [reason, count] : sorted_reasons) {
            os << "    " << demote_refusal_reason_string(reason) << ": " << count << "\n";
        }
    }

    return os.str();
}

void record_loop_stats(
    Function& fn,
    const LoopAnalysis& loops,
    const std::unordered_set<Value*>& demote_set,
    DemoteStats* stats
) {
    if (!stats) return;

    for (LoopInfo* loop : loops.post_order_loops()) {
        if (!loop || !loop->header()) continue;
        BasicBlock* header = loop->header();

        LoopDemoteRecord rec;
        rec.function_name = fn.name();
        rec.loop_header_block = header->name();

        size_t f64_param_count = 0;
        size_t demoted_param_count = 0;
        bool has_dynamic_or_ptr_param = false;

        for (size_t p_i = 0; p_i < header->param_count(); ++p_i) {
            Value* p = header->param(p_i);
            if (!p) continue;
            if (p->type() == Type::f64()) {
                f64_param_count++;
                if (demote_set.count(p)) {
                    demoted_param_count++;
                }
            } else if (p->type().is_pointer_or_gcref()) {
                has_dynamic_or_ptr_param = true;
            }
        }

        if (f64_param_count > 0 && demoted_param_count == f64_param_count && f64_param_count == header->param_count()) {
            rec.demoted = true;
            rec.refusal_reason = DemoteRefusalReason::None;
        } else {
            rec.demoted = false;

            bool has_overflow = false;
            bool has_non_int_const = false;
            bool has_unproven_div = false;
            bool has_call = false;
            bool has_dynamic_inst = false;

            for (BasicBlock* bb : loop->blocks()) {
                if (!bb) continue;
                for (Instruction* inst : *bb) {
                    if (!inst) continue;
                    Opcode op = inst->opcode();
                    if (op == Opcode::fconst_f64) {
                        double f = inst->imm_f64();
                        if (std::abs(f) >= 9007199254740992.0) {
                            has_overflow = true;
                        } else if (!is_safe_integer_f64(f)) {
                            has_non_int_const = true;
                        }
                    } else if (op == Opcode::sdiv || op == Opcode::udiv) {
                        has_unproven_div = true;
                    } else if (op == Opcode::call) {
                        if (callee_has_role(*inst, SymbolRole::ArrayGet) || callee_has_role(*inst, SymbolRole::ArraySet)) {
                            has_dynamic_inst = true;
                        } else if (!callee_has_role(*inst, SymbolRole::FloatRem)) {
                            has_call = true;
                        }
                    } else if (inst->type().is_pointer_or_gcref()) {
                        has_dynamic_inst = true;
                    }
                }
            }

            for (const BasicBlock* pred : header->predecessors()) {
                if (!pred || loop->contains(pred)) continue;
                const Instruction* term = pred->terminator();
                if (!term) continue;
                auto check_target = [&](const BranchTarget& bt) {
                    if (bt.block != header) return;
                    for (const Value* arg : bt.args) {
                        if (!arg || !arg->is_instruction()) continue;
                        const Instruction* def = arg->defining_instruction();
                        if (def && def->opcode() == Opcode::fconst_f64) {
                            double f = def->imm_f64();
                            if (std::abs(f) >= 9007199254740992.0) {
                                has_overflow = true;
                            } else if (!is_safe_integer_f64(f)) {
                                has_non_int_const = true;
                            }
                        }
                    }
                };
                if (term->opcode() == Opcode::br) check_target(term->branch_target());
                else if (term->opcode() == Opcode::br_if) {
                    check_target(term->true_target());
                    check_target(term->false_target());
                }
            }

            if (has_overflow) {
                rec.refusal_reason = DemoteRefusalReason::UnprovenRangeOrOverflow;
            } else if (has_dynamic_or_ptr_param || has_dynamic_inst || f64_param_count < header->param_count()) {
                rec.refusal_reason = DemoteRefusalReason::DynamicOrNonF64State;
            } else if (has_non_int_const) {
                rec.refusal_reason = DemoteRefusalReason::NonIntegralConstantOrFloatOp;
            } else if (has_call) {
                rec.refusal_reason = DemoteRefusalReason::CallInBody;
            } else if (is_unprofitable_loop(loop, fn)) {
                rec.refusal_reason = DemoteRefusalReason::Unprofitable;
            } else if (has_unproven_div) {
                rec.refusal_reason = DemoteRefusalReason::UnprovenDivision;
            } else {
                rec.refusal_reason = DemoteRefusalReason::NonIntegralStep;
            }
        }

        stats->add_record(std::move(rec));
    }
}

} // namespace brass
