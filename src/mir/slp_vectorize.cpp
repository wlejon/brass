#include <brass/mir/slp_vectorize.hpp>
#include "slp_analysis.hpp"
#include <brass/mir/builder.hpp>
#include <brass/mir/uses.hpp>
#include <unordered_set>
#include <unordered_map>

namespace brass {

namespace {

static bool is_extract_lane_sequence(const std::vector<Value*>& values, Value*& out_source_vec) {
    if (values.empty()) return false;
    Value* v0 = values[0];
    if (!v0 || !v0->is_instruction()) return false;
    Instruction* def0 = v0->defining_instruction();
    if (!def0 || def0->opcode() != Opcode::vextract_lane || def0->lane() != 0) return false;

    Value* src_vec = def0->operand(0);
    if (!src_vec || !src_vec->type().is_vector()) return false;
    if (src_vec->type().vector_lanes() != values.size()) return false;

    for (uint32_t i = 1; i < values.size(); ++i) {
        Value* vi = values[i];
        if (!vi || !vi->is_instruction()) return false;
        Instruction* defi = vi->defining_instruction();
        if (!defi || defi->opcode() != Opcode::vextract_lane) return false;
        if (defi->operand(0) != src_vec || defi->lane() != i) return false;
    }

    out_source_vec = src_vec;
    return true;
}

static bool are_all_values_identical(const std::vector<Value*>& values) {
    if (values.empty()) return false;
    Value* v0 = values[0];
    for (size_t i = 1; i < values.size(); ++i) {
        if (values[i] != v0) return false;
    }
    return true;
}

static bool are_contiguous_loads(
    const std::vector<Value*>& values,
    Value*& out_base,
    int32_t& out_offset,
    Type& out_elem_type
) {
    if (values.empty()) return false;
    Value* v0 = values[0];
    if (!v0 || !v0->is_instruction()) return false;
    Instruction* def0 = v0->defining_instruction();
    if (!def0 || def0->opcode() != Opcode::load || def0->operand_count() < 1) return false;

    Value* base = def0->operand(0);
    int32_t base_offset = def0->offset();
    Type elem_type = def0->memory_type();
    int32_t stride = static_cast<int32_t>(elem_type.size_in_bytes());

    for (uint32_t i = 1; i < values.size(); ++i) {
        Value* vi = values[i];
        if (!vi || !vi->is_instruction()) return false;
        Instruction* defi = vi->defining_instruction();
        if (!defi || defi->opcode() != Opcode::load) return false;
        if (defi->operand(0) != base || defi->memory_type() != elem_type) return false;
        if (defi->offset() != base_offset + static_cast<int32_t>(i) * stride) return false;
    }

    out_base = base;
    out_offset = base_offset;
    out_elem_type = elem_type;
    return true;
}

static bool check_isomorphic_select_min_max(
    const std::vector<Instruction*>& defs,
    bool& is_min,
    std::vector<Value*>& out_lhs,
    std::vector<Value*>& out_rhs
) {
    if (defs.empty()) return false;
    out_lhs.resize(defs.size());
    out_rhs.resize(defs.size());

    bool first_is_min = false;
    for (size_t i = 0; i < defs.size(); ++i) {
        Instruction* sel = defs[i];
        if (sel->opcode() != Opcode::select || sel->operand_count() < 3) return false;
        Value* cond_v = sel->operand(0);
        Value* true_v = sel->operand(1);
        Value* false_v = sel->operand(2);
        if (!cond_v || !cond_v->is_instruction()) return false;
        Instruction* cmp = cond_v->defining_instruction();
        if (!cmp || cmp->operand_count() < 2) return false;

        Value* cmp_lhs = cmp->operand(0);
        Value* cmp_rhs = cmp->operand(1);
        Opcode cmp_op = cmp->opcode();

        // Check if true_v == cmp_lhs and false_v == cmp_rhs (or vice-versa)
        if (cmp_op == Opcode::slt || cmp_op == Opcode::ult || cmp_op == Opcode::sle || cmp_op == Opcode::ule) {
            if (true_v == cmp_lhs && false_v == cmp_rhs) {
                // select (lhs < rhs) ? lhs : rhs => min(lhs, rhs)
                if (i == 0) first_is_min = true;
                else if (!first_is_min) return false;
                out_lhs[i] = cmp_lhs;
                out_rhs[i] = cmp_rhs;
            } else if (true_v == cmp_rhs && false_v == cmp_lhs) {
                // select (lhs < rhs) ? rhs : lhs => max(lhs, rhs)
                if (i == 0) first_is_min = false;
                else if (first_is_min) return false;
                out_lhs[i] = cmp_lhs;
                out_rhs[i] = cmp_rhs;
            } else {
                return false;
            }
        } else if (cmp_op == Opcode::sgt || cmp_op == Opcode::ugt || cmp_op == Opcode::sge || cmp_op == Opcode::uge) {
            if (true_v == cmp_lhs && false_v == cmp_rhs) {
                // select (lhs > rhs) ? lhs : rhs => max(lhs, rhs)
                if (i == 0) first_is_min = false;
                else if (first_is_min) return false;
                out_lhs[i] = cmp_lhs;
                out_rhs[i] = cmp_rhs;
            } else if (true_v == cmp_rhs && false_v == cmp_lhs) {
                // select (lhs > rhs) ? rhs : lhs => min(lhs, rhs)
                if (i == 0) first_is_min = true;
                else if (!first_is_min) return false;
                out_lhs[i] = cmp_lhs;
                out_rhs[i] = cmp_rhs;
            } else {
                return false;
            }
        } else {
            return false;
        }
    }

    is_min = first_is_min;
    return true;
}

Value* vectorize_lanes(
    const std::vector<Value*>& values,
    Type vec_type,
    Builder& b,
    BasicBlock& current_bb,
    Instruction* at,   // where the builder stands, in current_bb
    const std::unordered_set<const Instruction*>& replaced_stores,   // the scalar stores the result replaces
    std::unordered_set<Instruction*>& visited,
    std::unordered_set<Instruction*>& out_vectorized_defs,
    bool allow_fallback = true,
    bool allow_root_fallback = true
) {
    if (values.empty()) return nullptr;

    // 1. Direct vector extract sequence?
    Value* src_vec = nullptr;
    if (is_extract_lane_sequence(values, src_vec)) {
        return src_vec;
    }

    // 2. Broadcast scalar?
    if (are_all_values_identical(values)) {
        return b.build_vbroadcast(vec_type, values[0]);
    }

    // 3. Contiguous loads?
    // One vector load where the builder stands, if it reads there what the
    // scalar loads read where they are.
    Value* base = nullptr;
    int32_t base_offset = 0;
    Type elem_type;
    if (are_contiguous_loads(values, base, base_offset, elem_type)) {
        std::vector<Instruction*> loads;
        for (Value* v : values) loads.push_back(v->defining_instruction());
        if (slp_loads_readable_at(loads, current_bb, at, replaced_stores)) {
            for (Instruction* l : loads) out_vectorized_defs.insert(l);
            return b.build_vload(vec_type, base, base_offset);
        }
    }

    // 4. Recursive Isomorphic Tree?
    std::vector<Instruction*> defs;
    bool all_in_block = true;
    for (Value* v : values) {
        if (!v || !v->is_instruction()) {
            all_in_block = false;
            break;
        }
        Instruction* def = v->defining_instruction();
        if (!def || def->parent() != &current_bb || visited.count(def)) {
            all_in_block = false;
            break;
        }
        defs.push_back(def);
    }

    if (all_in_block && defs.size() == values.size()) {
        Opcode op0 = defs[0]->opcode();
        bool same_opcode = true;
        for (size_t i = 1; i < defs.size(); ++i) {
            if (defs[i]->opcode() != op0) {
                same_opcode = false;
                break;
            }
        }

        // Ensure no internal dependence
        bool has_internal_dep = false;
        for (size_t i = 0; i < defs.size(); ++i) {
            for (Value* opnd : defs[i]->operands()) {
                for (size_t j = 0; j < values.size(); ++j) {
                    if (opnd == values[j]) {
                        has_internal_dep = true;
                        break;
                    }
                }
                if (has_internal_dep) break;
            }
            if (has_internal_dep) break;
        }

        if (same_opcode && !has_internal_dep) {
            for (Instruction* def : defs) visited.insert(def);

            if (op0 == Opcode::add || op0 == Opcode::sub || op0 == Opcode::mul ||
                op0 == Opcode::sdiv || op0 == Opcode::udiv || op0 == Opcode::and_ ||
                op0 == Opcode::or_ || op0 == Opcode::xor_) {
                std::vector<Value*> lhs_ops(defs.size());
                std::vector<Value*> rhs_ops(defs.size());
                for (size_t i = 0; i < defs.size(); ++i) {
                    lhs_ops[i] = defs[i]->operand(0);
                    rhs_ops[i] = defs[i]->operand(1);
                }

                Value* vec_lhs = vectorize_lanes(lhs_ops, vec_type, b, current_bb, at, replaced_stores, visited, out_vectorized_defs, allow_fallback);
                Value* vec_rhs = vectorize_lanes(rhs_ops, vec_type, b, current_bb, at, replaced_stores, visited, out_vectorized_defs, allow_fallback);

                Value* res = nullptr;
                if (vec_lhs && vec_rhs) {
                    switch (op0) {
                        case Opcode::add: res = b.build_vadd(vec_lhs, vec_rhs); break;
                        case Opcode::sub: res = b.build_vsub(vec_lhs, vec_rhs); break;
                        case Opcode::mul: res = b.build_vmul(vec_lhs, vec_rhs); break;
                        case Opcode::sdiv:
                        case Opcode::udiv: res = b.build_vdiv(vec_lhs, vec_rhs); break;
                        case Opcode::and_: res = b.build_vand(vec_lhs, vec_rhs); break;
                        case Opcode::or_: res = b.build_vor(vec_lhs, vec_rhs); break;
                        case Opcode::xor_: res = b.build_vxor(vec_lhs, vec_rhs); break;
                        default: break;
                    }
                }

                if (res) {
                    for (Instruction* def : defs) out_vectorized_defs.insert(def);
                    return res;
                }
            } else if (op0 == Opcode::neg || op0 == Opcode::not_) {
                std::vector<Value*> arg_ops(defs.size());
                for (size_t i = 0; i < defs.size(); ++i) {
                    arg_ops[i] = defs[i]->operand(0);
                }
                Value* vec_arg = vectorize_lanes(arg_ops, vec_type, b, current_bb, at, replaced_stores, visited, out_vectorized_defs, allow_fallback);
                Value* res = nullptr;
                if (vec_arg) {
                    res = (op0 == Opcode::neg) ? b.build_vneg(vec_arg) : b.build_vnot(vec_arg);
                }
                if (res) {
                    for (Instruction* def : defs) out_vectorized_defs.insert(def);
                    return res;
                }
            } else if (op0 == Opcode::select) {
                bool is_min = false;
                std::vector<Value*> sel_lhs;
                std::vector<Value*> sel_rhs;
                if (check_isomorphic_select_min_max(defs, is_min, sel_lhs, sel_rhs)) {
                    Value* vec_lhs = vectorize_lanes(sel_lhs, vec_type, b, current_bb, at, replaced_stores, visited, out_vectorized_defs, allow_fallback);
                    Value* vec_rhs = vectorize_lanes(sel_rhs, vec_type, b, current_bb, at, replaced_stores, visited, out_vectorized_defs, allow_fallback);
                    Value* res = nullptr;
                    if (vec_lhs && vec_rhs) {
                        res = is_min ? b.build_vmin(vec_lhs, vec_rhs) : b.build_vmax(vec_lhs, vec_rhs);
                    }
                    if (res) {
                        for (Instruction* def : defs) out_vectorized_defs.insert(def);
                        return res;
                    }
                }
            }

            for (Instruction* def : defs) visited.erase(def);
        }
    }

    // 5. Fallback: pack via vbroadcast + vinsert_lane
    if (!allow_fallback || !allow_root_fallback) {
        return nullptr;
    }

    Value* packed = b.build_vbroadcast(vec_type, values[0]);
    for (uint32_t lane = 1; lane < values.size(); ++lane) {
        packed = b.build_vinsert_lane(packed, values[lane], lane);
    }
    return packed;
}

// The instructions of a function that read each value, kept as the pass
// rewrites: every question the pass has about a value's readers, and every
// replacement of one, walks that value's readers and not the function. (A
// function of one block and 100,000 instructions forms thousands of bundles;
// a walk of the function for each lane of each took most of its compile.)
// A list may name an instruction that has since been removed or no longer
// reads the value, so each is checked as it is visited.
class Users {
public:
    explicit Users(Function& fn) {
        for (BasicBlock* bb : fn.blocks()) {
            if (!bb) continue;
            for (Instruction* inst = bb->head(); inst != nullptr; inst = inst->next()) add(inst);
        }
    }

    // `inst` reads its values from now on (built, or put back in its block).
    void add(Instruction* inst) {
        for_each_use(*inst, [&](const Value* v) {
            std::vector<Instruction*>& list = readers_[v];
            if (list.empty() || list.back() != inst) list.push_back(inst);
        });
    }

    // The instructions built before `at` since `mark` was the one before it.
    void add_built(Instruction* mark, Instruction* at) {
        for (Instruction* inst = at->prev(); inst != nullptr && inst != mark; inst = inst->prev()) add(inst);
    }

    // True when `f(inst)` is for some instruction in a block that reads `v`.
    template <typename F>
    bool any(const Value* v, F&& f) const {
        const auto it = readers_.find(v);
        if (it == readers_.end()) return false;
        for (Instruction* inst : it->second) {
            if (inst->parent() && uses_value(*inst, v) && f(inst)) return true;
        }
        return false;
    }

    // `new_val` for `old_val` in every slot of each reader `where` accepts.
    // Every slot: a guard's deopt state map reads a value as surely as an
    // operand does.
    template <typename F>
    void replace(const Value* old_val, Value* new_val, F&& where) {
        const auto it = readers_.find(old_val);
        if (it == readers_.end()) return;
        const std::vector<Instruction*> readers = it->second;   // add() below may move the map
        for (Instruction* inst : readers) {
            if (!inst->parent() || !where(inst)) continue;
            if (replace_uses_in(*inst, old_val, new_val) != 0) add(inst);
        }
    }

private:
    std::unordered_map<const Value*, std::vector<Instruction*>> readers_;
};

// True when `a` stands before `b` in the block both are in. Walks out from
// `a` both ways at once, so it costs their distance and not the block.
static bool stands_before(const Instruction* a, const Instruction* b) {
    const Instruction* fwd = a->next();
    const Instruction* back = a->prev();
    for (;;) {
        if (fwd == b) return true;
        if (back == b) return false;
        if (!fwd) return false;
        if (!back) return true;
        fwd = fwd->next();
        back = back->prev();
    }
}

static void replace_scalar_uses(
    Users& users,
    Value* old_val,
    Value* new_val,
    const std::unordered_set<Instruction*>& dead_insts
) {
    Instruction* def_inst = new_val ? new_val->defining_instruction() : nullptr;
    BasicBlock* def_bb = def_inst ? def_inst->parent() : nullptr;
    // A reader above the new value's definition in its block keeps the scalar.
    users.replace(old_val, new_val, [&](Instruction* inst) {
        if (inst == def_inst || dead_insts.count(inst)) return false;
        return !(def_inst && inst->parent() == def_bb && stands_before(inst, def_inst));
    });
}

// True when a result of `insts` (other than as an operand of another of
// them) is read in `bb` before `last`: the lanes' vector value is built just
// before `last`, so such a use would be left reading a removed scalar.
static bool used_before(const BasicBlock& bb, const std::vector<Instruction*>& insts, const Instruction* last) {
    std::unordered_set<const Value*> results;
    std::unordered_set<const Instruction*> members(insts.begin(), insts.end());
    for (const Instruction* inst : insts) {
        if (inst && inst->produces_value()) results.insert(inst->result());
    }
    for (const Instruction* inst = bb.head(); inst != nullptr && inst != last; inst = inst->next()) {
        if (members.count(inst)) continue;
        bool hit = false;
        for_each_use(*inst, [&](const Value* v) { hit = hit || results.count(v) != 0; });
        if (hit) return true;
    }
    return false;
}

bool vectorize_block(BasicBlock& bb, const SlpOptions& options, Users& users) {
    Function* fn = bb.parent();
    Module* mod = fn->parent();

    bool any_changed = false;
    Builder b(*mod);
    b.set_function(fn);

    // =========================================================================
    // 1. Vectorize Store Bundles (Bottom-Up from Stores)
    // =========================================================================
    std::vector<SlpStoreBundle> store_bundles = find_slp_store_bundles(bb, options);
    for (const auto& sb : store_bundles) {
        Instruction* last_store = sb.stores.back();
        b.position_before(last_store);
        Instruction* mark = last_store->prev();

        std::unordered_set<Instruction*> visited;
        std::unordered_set<Instruction*> vectorized_defs;
        // Stored lanes that only a gather would pack stay scalar stores: the
        // broadcast and inserts cost what the stores they replace did, and
        // the vector store comes on top (x86: unpcklpd and movups for two
        // movsd, which also take an immediate or a register as they stand).
        const std::unordered_set<const Instruction*> replaced(sb.stores.begin(), sb.stores.end());
        Value* vec_val = vectorize_lanes(sb.stored_values, sb.vec_type, b, bb, last_store, replaced, visited, vectorized_defs,
                                         /*allow_fallback=*/true, /*allow_root_fallback=*/false);
        if (!vec_val) {
            users.add_built(mark, last_store);   // what a lane that failed left built
            continue;
        }

        // Emit vector store right before the last store in bundle
        b.build_vstore(sb.vec_type, sb.base, sb.base_offset, vec_val);
        users.add_built(mark, last_store);

        // For any vectorized def that has uses outside the vectorized tree, extract lane
        std::unordered_set<Instruction*> all_dead_stores(sb.stores.begin(), sb.stores.end());
        for (Instruction* def : vectorized_defs) {
            if (def && def->produces_value()) {
                Value* res = def->result();
                const bool has_external_use = users.any(res, [&](Instruction* inst) {
                    return !vectorized_defs.count(inst) && !all_dead_stores.count(inst);
                });

                if (has_external_use) {
                    // Find lane index for res
                    for (uint32_t lane = 0; lane < sb.stored_values.size(); ++lane) {
                        if (sb.stored_values[lane] == res) {
                            b.position_before(last_store);
                            Value* extracted = b.build_vextract_lane(vec_val, lane);
                            users.add(extracted->defining_instruction());
                            replace_scalar_uses(users, res, extracted, all_dead_stores);
                            break;
                        }
                    }
                }
            }
        }

        // Remove the scalar stores
        for (Instruction* s : sb.stores) {
            bb.remove_instruction(s);
        }

        any_changed = true;
    }

    // =========================================================================
    // 2. Vectorize Standalone Contiguous Load Bundles
    // =========================================================================
    // A bundle is formed here so that step 3 can read its lanes as one
    // vector; one that no vector operation ends up reading is put back as
    // its scalar loads afterwards (see below).
    struct FormedLoadBundle {
        Value* vload;
        std::vector<Value*> extracts;
        std::vector<Instruction*> loads;
    };
    std::vector<FormedLoadBundle> formed_loads;
    std::vector<SlpLoadBundle> load_bundles = find_slp_load_bundles(bb, options);
    for (const auto& lb : load_bundles) {
        Instruction* first_load = lb.loads.front();
        b.position_before(first_load);

        Value* vload = b.build_vload(lb.vec_type, lb.base, lb.base_offset);
        users.add(vload->defining_instruction());
        std::unordered_set<Instruction*> dead_loads(lb.loads.begin(), lb.loads.end());

        FormedLoadBundle formed{vload, {}, lb.loads};
        for (uint32_t lane = 0; lane < lb.width; ++lane) {
            Value* extracted = b.build_vextract_lane(vload, lane);
            users.add(extracted->defining_instruction());
            replace_scalar_uses(users, lb.loaded_values[lane], extracted, dead_loads);
            formed.extracts.push_back(extracted);
        }

        for (Instruction* l : lb.loads) {
            bb.remove_instruction(l);
        }
        formed_loads.push_back(std::move(formed));

        any_changed = true;
    }

    // =========================================================================
    // 3. Vectorize Standalone Arithmetic Bundles
    // =========================================================================
    std::unordered_set<Instruction*> empty_ignored;
    std::vector<SlpArithBundle> arith_bundles = find_slp_arith_bundles(bb, options, empty_ignored);
    for (const auto& ab : arith_bundles) {
        std::vector<Value*> inst_results;
        for (Instruction* inst : ab.insts) {
            inst_results.push_back(inst->result());
        }

        // Check if all operands across the bundle are from extracts or broadcasts
        Instruction* last_inst = ab.insts.back();
        if (used_before(bb, ab.insts, last_inst)) continue;
        b.position_before(last_inst);
        Instruction* mark = last_inst->prev();

        std::unordered_set<Instruction*> visited;
        std::unordered_set<Instruction*> vectorized_defs;
        Value* vec_res = vectorize_lanes(inst_results, ab.vec_type, b, bb, last_inst, {}, visited, vectorized_defs, /*allow_fallback=*/false);
        users.add_built(mark, last_inst);
        if (!vec_res) continue;

        std::unordered_set<Instruction*> dead_arith(ab.insts.begin(), ab.insts.end());
        for (uint32_t lane = 0; lane < ab.width; ++lane) {
            Value* extracted = b.build_vextract_lane(vec_res, lane);
            users.add(extracted->defining_instruction());
            replace_scalar_uses(users, ab.insts[lane]->result(), extracted, dead_arith);
        }

        for (Instruction* inst : ab.insts) {
            bb.remove_instruction(inst);
        }

        any_changed = true;
    }

    // =========================================================================
    // 4. Unpack Load Bundles No Vector Operation Reads
    // =========================================================================
    // Read only lane by lane, a bundle is a vector load and an extract per
    // lane (x86: movups, then a movaps and a shufpd for the high lane) where
    // the scalars were one load each, and its lanes leave the vector register
    // only through those copies. The scalar loads go back where the vector
    // load stands: the bundle was formed only if nothing between its first
    // and last load could write what they read, so each reads what it read.
    for (const FormedLoadBundle& fl : formed_loads) {
        Instruction* vdef = fl.vload->defining_instruction();
        if (!vdef || vdef->parent() != &bb) continue;
        std::unordered_set<const Value*> extract_set(fl.extracts.begin(), fl.extracts.end());
        const bool vector_use = users.any(fl.vload, [&](Instruction* inst) {
            if (inst == vdef) return false;
            return !(inst->produces_value() && extract_set.count(inst->result()) != 0 &&
                     inst->opcode() == Opcode::vextract_lane && inst->parent() == &bb);
        });
        if (vector_use) continue;
        for (size_t lane = 0; lane < fl.loads.size(); ++lane) {
            Instruction* load = fl.loads[lane];
            bb.insert_before(load, vdef);
            users.add(load);
            Instruction* ext = fl.extracts[lane]->defining_instruction();
            if (ext && ext->parent() == &bb) {
                users.replace(fl.extracts[lane], load->result(), [](Instruction*) { return true; });
                bb.remove_instruction(ext);
            }
        }
        bb.remove_instruction(vdef);
    }

    return any_changed;
}

} // namespace

bool slp_vectorize_block(BasicBlock& bb, const SlpOptions& options) {
    Function* fn = bb.parent();
    if (!fn || !fn->parent()) return false;
    Users users(*fn);
    return vectorize_block(bb, options, users);
}

bool slp_vectorize_block(BasicBlock& bb) {
    return slp_vectorize_block(bb, SlpOptions());
}

bool slp_vectorize_function(Function& fn, const SlpOptions& options) {
    if (!fn.parent()) return false;
    bool changed = false;
    Users users(fn);
    for (BasicBlock* bb : fn.blocks()) {
        if (bb) {
            changed |= vectorize_block(*bb, options, users);
        }
    }
    return changed;
}

bool slp_vectorize_function(Function& fn) {
    return slp_vectorize_function(fn, SlpOptions());
}

} // namespace brass