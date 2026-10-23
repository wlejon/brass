#include "verifier_certs.hpp"
#include <brass/gc/gc_limits.hpp>
#include <brass/mir/block.hpp>
#include <brass/mir/dominators.hpp>
#include <brass/mir/gc_refs.hpp>
#include <brass/mir/instruction.hpp>
#include <brass/mir/opcodes.hpp>
#include <brass/mir/proof_cert.hpp>
#include <set>
#include <unordered_map>
#include <unordered_set>

namespace brass {

namespace {

constexpr int kMaxCertNonPointerDepth = 8;

bool cert_const_int(const Value* val, int64_t& out) noexcept {
    if (!val || !val->is_instruction()) return false;
    const Instruction* def = val->defining_instruction();
    if (!def) return false;
    if (def->opcode() == Opcode::iconst_i64 || def->opcode() == Opcode::patchable_const_i64) {
        out = def->imm_i64();
        return true;
    }
    if (def->opcode() == Opcode::iconst_i32 || def->opcode() == Opcode::patchable_const_i32) {
        out = def->imm_i32();
        return true;
    }
    return false;
}

bool cert_is_non_pointer_value(const Value* val, int depth) noexcept {
    if (!val) return true;
    const Type t = val->type();
    if (t.is_gc_root() || t.is_pointer()) return false;
    if (t.is_float() || t.is_vector() || t.is_void()) return true;
    if (t.kind() != TypeKind::I64) return true;

    if (!val->is_instruction()) return false;
    const Instruction* def = val->defining_instruction();
    if (!def) return false;

    const Opcode op = def->opcode();
    if (op == Opcode::iconst_i64 || op == Opcode::iconst_i32 || is_comparison(op)) return true;
    switch (op) {
        case Opcode::clz: case Opcode::ctz: case Opcode::popcnt:
        case Opcode::sext_i64: case Opcode::zext_i64:
        case Opcode::fptosi_i64: case Opcode::fptosi_i64_f32:
        case Opcode::bitcast_f64_i64:
            return true;
        default:
            break;
    }
    if (depth >= kMaxCertNonPointerDepth) return false;
    if (is_arithmetic(op) || is_bitwise(op) || op == Opcode::select) {
        const size_t first = (op == Opcode::select) ? 1 : 0;
        for (size_t i = first; i < def->operand_count(); ++i) {
            if (!cert_is_non_pointer_value(def->operand(i), depth + 1)) return false;
        }
        return true;
    }
    return false;
}

bool cert_is_young_alloc(const Instruction* inst) noexcept {
    if (!inst || inst->opcode() != Opcode::call) return false;
    if (inst->symbol() != "brass_gc_alloc" || inst->operand_count() < 1) return false;
    int64_t size = 0;
    if (!cert_const_int(inst->operand(0), size)) return false;
    return size >= 0 && static_cast<uint64_t>(size) <= kMaxAlwaysYoungPayloadBytes;
}

bool cert_traces_to_alloca(const Value* val, int depth = 0) noexcept {
    if (!val || depth > 8 || val->type().is_gcref()) return false;
    if (!val->is_instruction()) return false;
    const Instruction* def = val->defining_instruction();
    if (!def) return false;
    if (def->opcode() == Opcode::alloca_) return true;
    if ((def->opcode() == Opcode::add || def->opcode() == Opcode::sub) && def->operand_count() >= 1) {
        return cert_traces_to_alloca(def->operand(0), depth + 1);
    }
    return false;
}

bool has_barrier_in_epoch(const Instruction* store_inst, const Value* obj, const Value* val) noexcept {
    for (const Instruction* cur = store_inst->prev(); cur != nullptr; cur = cur->prev()) {
        if (may_trigger_gc(*cur)) break;
        if (cur->opcode() == Opcode::write_barrier &&
            cur->operand(0) == obj && cur->operand(1) == val) {
            return true;
        }
    }
    for (const Instruction* cur = store_inst->next(); cur != nullptr; cur = cur->next()) {
        if (may_trigger_gc(*cur)) break;
        if (cur->opcode() == Opcode::write_barrier &&
            cur->operand(0) == obj && cur->operand(1) == val) {
            return true;
        }
    }
    return false;
}

void verify_barrier_certificates(
    const Function& fn,
    const std::string& fn_prefix,
    const std::function<void(const std::string&)>& report_error
) {
    const bool strict_mode = fn.requires_barrier_certs();

    for (const BasicBlock* bb : fn.blocks()) {
        if (!bb) continue;

        std::unordered_set<const Value*> young_in_epoch;
        std::set<std::pair<const Value*, const Value*>> barriered_in_epoch;

        for (const Instruction* inst : *bb) {
            if (!inst) continue;

            if (may_trigger_gc(*inst)) {
                young_in_epoch.clear();
                barriered_in_epoch.clear();
            }

            if (cert_is_young_alloc(inst) && inst->result()) {
                young_in_epoch.insert(inst->result());
            }

            if (inst->opcode() == Opcode::write_barrier) {
                if (inst->operand(0)) {
                    barriered_in_epoch.insert({inst->operand(0), inst->operand(1)});
                }
                continue;
            }

            if (inst->opcode() != Opcode::store && inst->opcode() != Opcode::store_indexed) {
                continue;
            }

            const Value* obj = inst->operand(0);
            const Value* val = (inst->opcode() == Opcode::store) ? inst->operand(1) : inst->operand(2);
            const BarrierCertKind cert = inst->barrier_cert();
            const std::string where = fn_prefix + "Block '" + std::string(bb->name()) +
                                      "': Instruction '" + std::string(opcode_name(inst->opcode())) + "': ";

            if (cert != BarrierCertKind::None) {
                switch (cert) {
                    case BarrierCertKind::YoungGenBumpAlloc:
                        if (!obj || young_in_epoch.count(obj) == 0) {
                            report_error(where + "Invalid BarrierCertKind::YoungGenBumpAlloc: target object is not a "
                                                 "young nursery allocation in the current GC epoch (either not allocated "
                                                 "here or a GC safepoint/call intervened).");
                        }
                        break;
                    case BarrierCertKind::StackOrNonGC:
                        if (!cert_traces_to_alloca(obj)) {
                            report_error(where + "Invalid BarrierCertKind::StackOrNonGC: target base does not trace to a stack alloca.");
                        }
                        break;
                    case BarrierCertKind::NonRefStore:
                        if (!cert_is_non_pointer_value(val, 0)) {
                            report_error(where + "Invalid BarrierCertKind::NonRefStore: stored value may carry a heap reference.");
                        }
                        break;
                    case BarrierCertKind::ConstantOrNullRef: {
                        int64_t c_imm = -1;
                        if (!cert_const_int(val, c_imm) || c_imm != 0) {
                            report_error(where + "Invalid BarrierCertKind::ConstantOrNullRef: stored value is not constant 0 / null.");
                        }
                        break;
                    }
                    case BarrierCertKind::RedundantSameEpoch:
                        if (!obj || barriered_in_epoch.count({obj, val}) == 0) {
                            report_error(where + "Invalid BarrierCertKind::RedundantSameEpoch: no preceding write_barrier(obj, val) "
                                                 "in the current GC epoch.");
                        }
                        break;
                    case BarrierCertKind::None:
                        break;
                }
            } else if (strict_mode) {
                const bool stores_potential_ref =
                    inst->memory_type().is_gc_root() ||
                    (val && (val->type().is_gc_root() || !cert_is_non_pointer_value(val, 0)));
                const bool into_gc_heap = obj && (obj->type().is_gcref() || !cert_traces_to_alloca(obj));
                if (stores_potential_ref && into_gc_heap && !has_barrier_in_epoch(inst, obj, val)) {
                    report_error(where + "Unbarriered GC store lacks a write_barrier or BarrierCertKind proof certificate.");
                }
            }
        }
    }
}

void verify_guard_elim_certificates(
    const Function& fn,
    const std::string& fn_prefix,
    const std::function<void(const std::string&)>& report_error
) {
    if (fn.guard_elim_certs().empty()) return;
    if (!fn.entry_block()) return;

    struct ValLoc {
        const Value* val = nullptr;
        const BasicBlock* bb = nullptr;
    };
    std::unordered_map<uint32_t, ValLoc> val_map;
    for (const BasicBlock* bb : fn.blocks()) {
        if (!bb) continue;
        for (size_t i = 0; i < bb->param_count(); ++i) {
            if (const Value* p = bb->param(i)) val_map[p->id()] = {p, bb};
        }
        for (const Instruction* inst : *bb) {
            if (inst && inst->result()) val_map[inst->result()->id()] = {inst->result(), bb};
        }
    }

    DominatorTree dom(fn);

    for (const GuardElimCert& cert : fn.guard_elim_certs()) {
        const std::string cert_prefix = fn_prefix + "GuardElimCert (resume_id " +
                                        std::to_string(cert.eliminated_resume_id) + "): ";
        const BasicBlock* bb = fn.get_block_by_id(cert.block_id);
        if (!bb) {
            report_error(cert_prefix + "references non-existent block id " + std::to_string(cert.block_id) + ".");
            continue;
        }

        for (const Instruction* inst : *bb) {
            if (inst && inst->opcode() == Opcode::guard && inst->resume_id() == cert.eliminated_resume_id) {
                report_error(cert_prefix + "claims guard was eliminated in block '" + std::string(bb->name()) +
                             "', but guard is still present.");
                break;
            }
        }

        if (cert.bound_min > cert.bound_max) {
            report_error(cert_prefix + "has invalid empty range [" + std::to_string(cert.bound_min) +
                         ", " + std::to_string(cert.bound_max) + "].");
            continue;
        }

        const Value* w_val = nullptr;
        const BasicBlock* w_bb = nullptr;
        if (cert.witness_inst_or_val_id != 0) {
            auto it = val_map.find(cert.witness_inst_or_val_id);
            if (it == val_map.end()) {
                report_error(cert_prefix + "references non-existent witness value %" +
                             std::to_string(cert.witness_inst_or_val_id) + ".");
                continue;
            }
            w_val = it->second.val;
            w_bb = it->second.bb;
            if (!dom.dominates(w_bb, bb)) {
                report_error(cert_prefix + "witness value %" + std::to_string(cert.witness_inst_or_val_id) +
                             " in block '" + std::string(w_bb->name()) + "' does not dominate block '" +
                             std::string(bb->name()) + "'.");
                continue;
            }
        }

        switch (cert.reason) {
            case GuardElimCert::Reason::DominatedSameCond: {
                if (!w_val) {
                    report_error(cert_prefix + "DominatedSameCond requires a non-zero witness value id.");
                    break;
                }
                int64_t c_imm = -1;
                if (cert_const_int(w_val, c_imm) && c_imm == 0) {
                    report_error(cert_prefix + "DominatedSameCond witness is constant 0.");
                    break;
                }
                // Verify that some dominating block has a guard or conditional branch establishing a fact
                bool has_dom_guard_or_branch = false;
                for (const BasicBlock* dbb : fn.blocks()) {
                    if (!dbb || !dom.dominates(dbb, bb)) continue;
                    for (const Instruction* inst : *dbb) {
                        if (!inst) continue;
                        if (inst->opcode() == Opcode::guard && inst->operand(0) == w_val) {
                            has_dom_guard_or_branch = true;
                            break;
                        }
                        if (inst->opcode() == Opcode::br_if && dbb != bb) {
                            has_dom_guard_or_branch = true;
                            break;
                        }
                    }
                    if (has_dom_guard_or_branch) break;
                }
                if (!has_dom_guard_or_branch) {
                    report_error(cert_prefix + "DominatedSameCond witness %" +
                                 std::to_string(cert.witness_inst_or_val_id) +
                                 " is not established by any dominating guard or conditional branch.");
                }
                break;
            }

            case GuardElimCert::Reason::ConstantFolded: {
                int64_t c_imm = 0;
                if (!w_val || !cert_const_int(w_val, c_imm) || c_imm == 0) {
                    report_error(cert_prefix + "ConstantFolded witness must be a non-zero integer constant.");
                }
                break;
            }

            case GuardElimCert::Reason::NonNegativeMask: {
                if (!w_val || !w_val->is_instruction() || !w_val->defining_instruction() ||
                    w_val->defining_instruction()->opcode() != Opcode::and_) {
                    report_error(cert_prefix + "NonNegativeMask witness must be defined by an 'and' instruction.");
                    break;
                }
                const Instruction* and_inst = w_val->defining_instruction();
                int64_t m0 = -1, m1 = -1;
                const bool has0 = cert_const_int(and_inst->operand(0), m0) && m0 >= 0;
                const bool has1 = cert_const_int(and_inst->operand(1), m1) && m1 >= 0;
                if (!has0 && !has1) {
                    report_error(cert_prefix + "NonNegativeMask 'and' instruction has no non-negative constant mask.");
                    break;
                }
                const int64_t mask = has1 ? m1 : m0;
                if (cert.bound_min < 0 || cert.bound_max < mask) {
                    report_error(cert_prefix + "NonNegativeMask range [" + std::to_string(cert.bound_min) +
                                 ", " + std::to_string(cert.bound_max) + "] does not cover mask " +
                                 std::to_string(mask) + ".");
                }
                break;
            }

            case GuardElimCert::Reason::RangeProven: {
                int64_t c_imm = 0;
                if (w_val && cert_const_int(w_val, c_imm)) {
                    if (c_imm < cert.bound_min || c_imm > cert.bound_max) {
                        report_error(cert_prefix + "RangeProven constant witness " + std::to_string(c_imm) +
                                     " lies outside certified range [" + std::to_string(cert.bound_min) +
                                     ", " + std::to_string(cert.bound_max) + "].");
                    } else if (cert.check_op == Opcode::unreachable && c_imm == 0) {
                        report_error(cert_prefix + "RangeProven witness is constant 0.");
                    }
                }
                break;
            }

            case GuardElimCert::Reason::ImpliedRelational:
            case GuardElimCert::Reason::SmtProven: {
                int64_t c_imm = -1;
                if (w_val && cert_const_int(w_val, c_imm) && c_imm == 0) {
                    report_error(cert_prefix + "witness value is constant 0.");
                }
                break;
            }
        }
    }
}

} // namespace

void verify_proof_certificates(
    const Function& fn,
    const std::string& fn_prefix,
    const std::function<void(const std::string&)>& report_error
) {
    verify_barrier_certificates(fn, fn_prefix, report_error);
    verify_guard_elim_certificates(fn, fn_prefix, report_error);
}

} // namespace brass
