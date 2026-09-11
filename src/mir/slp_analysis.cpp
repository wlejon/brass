#include "slp_analysis.hpp"
#include <brass/mir/opcodes.hpp>
#include <algorithm>

namespace brass {

bool get_slp_type_info(
    Type elem_type,
    const SlpOptions& options,
    uint32_t& out_width,
    int32_t& out_stride,
    Type& out_vec_type
) {
    if (elem_type == Type::f32() && options.enable_f32x4) {
        out_width = 4;
        out_stride = 4;
        out_vec_type = Type::f32x4();
        return true;
    }
    if (elem_type == Type::i32() && options.enable_i32x4) {
        out_width = 4;
        out_stride = 4;
        out_vec_type = Type::i32x4();
        return true;
    }
    if (elem_type == Type::f64() && options.enable_f64x2) {
        out_width = 2;
        out_stride = 8;
        out_vec_type = Type::f64x2();
        return true;
    }
    return false;
}

static bool may_alias_or_barrier(
    const Instruction* inst,
    const Value* base,
    int32_t range_start,
    int32_t range_end,
    bool check_writes
) {
    if (!inst) return false;
    if (inst->is_call() || inst->opcode() == Opcode::safepoint ||
        inst->opcode() == Opcode::guard || inst->opcode() == Opcode::resume_point) {
        return true;
    }

    Opcode op = inst->opcode();
    if (check_writes) {
        if (op == Opcode::store || op == Opcode::vstore) {
            if (inst->operand(0) == base) {
                int32_t off = inst->offset();
                int32_t size = static_cast<int32_t>(inst->memory_type().size_in_bytes());
                if (off < range_end && (off + size) > range_start) {
                    return true;
                }
            } else {
                // Different or unknown base store might alias
                return true;
            }
        } else if (op == Opcode::store_indexed) {
            return true;
        }
    } else {
        // Checking reads against stores
        if (op == Opcode::load || op == Opcode::vload) {
            if (inst->operand(0) == base) {
                int32_t off = inst->offset();
                int32_t size = static_cast<int32_t>(inst->memory_type().size_in_bytes());
                if (off < range_end && (off + size) > range_start) {
                    return true;
                }
            } else {
                return true;
            }
        } else if (op == Opcode::load_indexed) {
            return true;
        }
    }
    return false;
}

std::vector<SlpStoreBundle> find_slp_store_bundles(
    BasicBlock& bb,
    const SlpOptions& options
) {
    std::vector<SlpStoreBundle> bundles;
    std::unordered_set<Instruction*> visited;

    std::vector<Instruction*> stores;
    for (Instruction* inst = bb.head(); inst != nullptr; inst = inst->next()) {
        if (inst->opcode() == Opcode::store && inst->operand_count() >= 2) {
            stores.push_back(inst);
        }
    }

    for (size_t i = 0; i < stores.size(); ++i) {
        Instruction* s0 = stores[i];
        if (visited.count(s0)) continue;

        Type elem_type = s0->memory_type();
        uint32_t width = 0;
        int32_t stride = 0;
        Type vec_type;
        if (!get_slp_type_info(elem_type, options, width, stride, vec_type)) {
            continue;
        }

        Value* base = s0->operand(0);
        int32_t base_offset = s0->offset();

        std::vector<Instruction*> bundle_stores;
        bundle_stores.push_back(s0);

        bool bundle_complete = false;
        int32_t range_start = base_offset;
        int32_t range_end = base_offset + static_cast<int32_t>(width) * stride;

        Instruction* scan = s0->next();
        bool barrier_hit = false;

        while (scan && !barrier_hit && bundle_stores.size() < width) {
            if (scan->opcode() == Opcode::store && scan->operand(0) == base &&
                scan->memory_type() == elem_type && !visited.count(scan)) {
                int32_t expected_offset = base_offset + static_cast<int32_t>(bundle_stores.size()) * stride;
                if (scan->offset() == expected_offset) {
                    bundle_stores.push_back(scan);
                    if (bundle_stores.size() == width) {
                        bundle_complete = true;
                        break;
                    }
                    scan = scan->next();
                    continue;
                }
            }

            if (may_alias_or_barrier(scan, base, range_start, range_end, /*check_writes=*/false)) {
                barrier_hit = true;
                break;
            }
            if (scan->opcode() == Opcode::store && scan->operand(0) == base) {
                int32_t off = scan->offset();
                if (off >= range_start && off < range_end) {
                    barrier_hit = true;
                    break;
                }
            }

            scan = scan->next();
        }

        if (bundle_complete && bundle_stores.size() == width) {
            SlpStoreBundle bundle;
            bundle.stores = bundle_stores;
            bundle.base = base;
            bundle.base_offset = base_offset;
            bundle.elem_type = elem_type;
            bundle.vec_type = vec_type;
            bundle.width = width;
            for (Instruction* s : bundle_stores) {
                bundle.stored_values.push_back(s->operand(1));
                visited.insert(s);
            }
            bundles.push_back(std::move(bundle));
        }
    }

    return bundles;
}

std::vector<SlpLoadBundle> find_slp_load_bundles(
    BasicBlock& bb,
    const SlpOptions& options
) {
    std::vector<SlpLoadBundle> bundles;
    std::unordered_set<Instruction*> visited;

    std::vector<Instruction*> loads;
    for (Instruction* inst = bb.head(); inst != nullptr; inst = inst->next()) {
        if (inst->opcode() == Opcode::load && inst->operand_count() >= 1 && inst->produces_value()) {
            loads.push_back(inst);
        }
    }

    for (size_t i = 0; i < loads.size(); ++i) {
        Instruction* l0 = loads[i];
        if (visited.count(l0)) continue;

        Type elem_type = l0->memory_type();
        uint32_t width = 0;
        int32_t stride = 0;
        Type vec_type;
        if (!get_slp_type_info(elem_type, options, width, stride, vec_type)) {
            continue;
        }

        Value* base = l0->operand(0);
        int32_t base_offset = l0->offset();

        std::vector<Instruction*> bundle_loads;
        bundle_loads.push_back(l0);

        bool bundle_complete = false;
        int32_t range_start = base_offset;
        int32_t range_end = base_offset + static_cast<int32_t>(width) * stride;

        Instruction* scan = l0->next();
        bool barrier_hit = false;

        while (scan && !barrier_hit && bundle_loads.size() < width) {
            if (scan->opcode() == Opcode::load && scan->operand(0) == base &&
                scan->memory_type() == elem_type && !visited.count(scan)) {
                int32_t expected_offset = base_offset + static_cast<int32_t>(bundle_loads.size()) * stride;
                if (scan->offset() == expected_offset) {
                    bundle_loads.push_back(scan);
                    if (bundle_loads.size() == width) {
                        bundle_complete = true;
                        break;
                    }
                    scan = scan->next();
                    continue;
                }
            }

            if (may_alias_or_barrier(scan, base, range_start, range_end, /*check_writes=*/true)) {
                barrier_hit = true;
                break;
            }

            scan = scan->next();
        }

        if (bundle_complete && bundle_loads.size() == width) {
            SlpLoadBundle bundle;
            bundle.loads = bundle_loads;
            bundle.base = base;
            bundle.base_offset = base_offset;
            bundle.elem_type = elem_type;
            bundle.vec_type = vec_type;
            bundle.width = width;
            for (Instruction* l : bundle_loads) {
                bundle.loaded_values.push_back(l->result());
                visited.insert(l);
            }
            bundles.push_back(std::move(bundle));
        }
    }

    return bundles;
}

static bool is_slp_arith_opcode(Opcode op) {
    switch (op) {
        case Opcode::add:
        case Opcode::sub:
        case Opcode::mul:
        case Opcode::sdiv:
        case Opcode::udiv:
        case Opcode::neg:
        case Opcode::and_:
        case Opcode::or_:
        case Opcode::xor_:
        case Opcode::not_:
            return true;
        default:
            return false;
    }
}

std::vector<SlpArithBundle> find_slp_arith_bundles(
    BasicBlock& bb,
    const SlpOptions& options,
    const std::unordered_set<Instruction*>& ignored_insts
) {
    std::vector<SlpArithBundle> bundles;
    std::unordered_set<Instruction*> visited = ignored_insts;

    std::vector<Instruction*> candidate_insts;
    for (Instruction* inst = bb.head(); inst != nullptr; inst = inst->next()) {
        if (visited.count(inst)) continue;
        if (is_slp_arith_opcode(inst->opcode()) && inst->produces_value()) {
            candidate_insts.push_back(inst);
        }
    }

    for (size_t i = 0; i < candidate_insts.size(); ++i) {
        Instruction* inst0 = candidate_insts[i];
        if (visited.count(inst0)) continue;

        Opcode op = inst0->opcode();
        Type elem_type = inst0->type();
        uint32_t width = 0;
        int32_t stride = 0;
        Type vec_type;
        if (!get_slp_type_info(elem_type, options, width, stride, vec_type)) {
            continue;
        }

        std::vector<Instruction*> bundle_insts;
        bundle_insts.push_back(inst0);
        std::unordered_set<Value*> bundle_results;
        if (inst0->produces_value() && inst0->result()) {
            bundle_results.insert(inst0->result());
        }

        bool hazard = false;
        for (Instruction* cur = inst0->next(); cur != nullptr; cur = cur->next()) {
            if (bundle_insts.size() == width) break;

            for (Value* opnd : cur->operands()) {
                if (bundle_results.count(opnd)) {
                    hazard = true;
                    break;
                }
            }
            if (hazard) break;

            if (visited.count(cur)) continue;

            if (cur->opcode() == op && cur->type() == elem_type) {
                bundle_insts.push_back(cur);
                if (cur->produces_value() && cur->result()) {
                    bundle_results.insert(cur->result());
                }
            }
        }

        if (bundle_insts.size() == width) {
            SlpArithBundle bundle;
            bundle.insts = bundle_insts;
            bundle.op = op;
            bundle.elem_type = elem_type;
            bundle.vec_type = vec_type;
            bundle.width = width;
            for (Instruction* inst : bundle_insts) {
                visited.insert(inst);
            }
            bundles.push_back(std::move(bundle));
        }
    }

    return bundles;
}

} // namespace brass