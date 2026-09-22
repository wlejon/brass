#include "loop_dependence.hpp"
#include <brass/mir/opcodes.hpp>
#include <brass/mir/alias_analysis.hpp>
#include <algorithm>
#include <map>
#include <optional>

// Dependences between the memory operations of a loop nest, for tiling and
// interchange. An access is described exactly — a sum of induction-variable
// terms and constants — or not at all; anything the parser cannot account
// for, and every operation with effects it does not model (calls, plain
// loads and stores, deopts), makes the access unanalyzable and every pair it
// is part of a dependence of unknown direction.
namespace brass {

namespace {

bool get_const_int(const Value* val, int64_t& out_val) {
    if (!val || !val->is_instruction()) return false;
    const Instruction* def = val->defining_instruction();
    if (!def) return false;
    if (def->opcode() == Opcode::iconst_i32) {
        out_val = static_cast<int64_t>(def->imm_i32());
        return true;
    }
    if (def->opcode() == Opcode::iconst_i64) {
        out_val = def->imm_i64();
        return true;
    }
    return false;
}

bool find_iv_level(const Value* val, const LoopNest& nest, size_t& out_lvl) {
    if (!val) return false;
    for (size_t l = 0; l < nest.depth(); ++l) {
        if (nest.level(l).iv_param == val) {
            out_lvl = l;
            return true;
        }
    }
    return false;
}

bool invariant_in_nest(const Value* v, const LoopNest& nest) {
    return nest.depth() > 0 && nest.level(0).loop && nest.level(0).loop->is_loop_invariant(v);
}

void add_term(std::vector<AccessIndexTerm>& terms, size_t level, int64_t stride, Value* symbolic, bool& ok) {
    for (const auto& t : terms) {
        if (t.level_index == level) {
            ok = false;
            return;
        }
    }
    AccessIndexTerm term;
    term.level_index = level;
    term.const_stride = stride;
    term.symbolic_stride = symbolic;
    terms.push_back(term);
}

void parse_index_expr(const Value* val, const LoopNest& nest, std::vector<AccessIndexTerm>& terms,
                      int64_t& const_offset, bool& ok, int depth = 0) {
    if (!ok) return;
    if (!val || depth > 16) {
        ok = false;
        return;
    }
    size_t lvl = 0;
    if (find_iv_level(val, nest, lvl)) {
        add_term(terms, lvl, 1, nullptr, ok);
        return;
    }
    int64_t c = 0;
    if (get_const_int(val, c)) {
        const_offset += c;
        return;
    }
    const Instruction* inst = val->is_instruction() ? val->defining_instruction() : nullptr;
    if (!inst) {
        ok = false;
        return;
    }
    switch (inst->opcode()) {
        case Opcode::add:
            parse_index_expr(inst->operand(0), nest, terms, const_offset, ok, depth + 1);
            parse_index_expr(inst->operand(1), nest, terms, const_offset, ok, depth + 1);
            return;
        case Opcode::sub: {
            parse_index_expr(inst->operand(0), nest, terms, const_offset, ok, depth + 1);
            int64_t sub_c = 0;
            if (get_const_int(inst->operand(1), sub_c)) const_offset -= sub_c;
            else ok = false;
            return;
        }
        case Opcode::sext_i64:
            parse_index_expr(inst->operand(0), nest, terms, const_offset, ok, depth + 1);
            return;
        case Opcode::mul: {
            const Value* op0 = inst->operand(0);
            const Value* op1 = inst->operand(1);
            size_t mul_lvl = 0;
            const Value* other = nullptr;
            if (find_iv_level(op0, nest, mul_lvl)) other = op1;
            else if (find_iv_level(op1, nest, mul_lvl)) other = op0;
            int64_t factor = 0;
            if (other && get_const_int(other, factor)) {
                if (factor != 0) add_term(terms, mul_lvl, factor, nullptr, ok);
            } else if (other && invariant_in_nest(other, nest)) {
                add_term(terms, mul_lvl, 0, const_cast<Value*>(other), ok);
            } else {
                ok = false;
            }
            return;
        }
        case Opcode::shl: {
            int64_t shift = 0;
            size_t shl_lvl = 0;
            if (get_const_int(inst->operand(1), shift) && shift >= 0 && shift < 62 &&
                find_iv_level(inst->operand(0), nest, shl_lvl)) {
                add_term(terms, shl_lvl, int64_t{1} << shift, nullptr, ok);
            } else {
                ok = false;
            }
            return;
        }
        default:
            ok = false;
            return;
    }
}

// Operations whose memory behaviour the access model cannot describe.
bool has_unmodeled_effect(const Instruction& inst) {
    const Opcode op = inst.opcode();
    if (op == Opcode::load_indexed || op == Opcode::store_indexed || op == Opcode::safepoint) return false;
    if (op == Opcode::throw_ || op == Opcode::invoke || op == Opcode::resume) return true;
    if (inst.is_terminator()) return false;
    if (is_call(op) || inst.has_side_effects()) return true;
    return op == Opcode::load || op == Opcode::vload || op == Opcode::alloca_;
}

const AccessIndexTerm* term_at(const NestMemoryAccess& a, size_t level) {
    for (const auto& t : a.terms) {
        if (t.level_index == level) return &t;
    }
    return nullptr;
}

bool same_stride(const AccessIndexTerm& x, const AccessIndexTerm& y) {
    return x.const_stride == y.const_stride && x.symbolic_stride == y.symbolic_stride;
}

// index = S * iv_a + iv_b where iv_b runs over [0, S): distinct (iv_a, iv_b)
// give distinct indices.
bool row_major_injective(const LoopNest& nest, const AccessIndexTerm& t1, const AccessIndexTerm& t2) {
    auto check = [&](const AccessIndexTerm& scaled, const AccessIndexTerm& unit) {
        if (unit.symbolic_stride || unit.const_stride != 1) return false;
        const LoopNestLevel& lvl = nest.level(unit.level_index);
        int64_t init = 0;
        if (!get_const_int(lvl.init_val, init) || init != 0 || lvl.step != 1) return false;
        if (lvl.cmp_opcode != Opcode::slt || !lvl.exit_on_false) return false;
        if (scaled.symbolic_stride) return scaled.symbolic_stride == lvl.limit_val;
        int64_t limit = 0;
        return get_const_int(lvl.limit_val, limit) && limit == scaled.const_stride;
    };
    return check(t1, t2) || check(t2, t1);
}

DependenceVector unknown_dependence(size_t depth) {
    DependenceVector dep;
    dep.directions.assign(depth, DependenceDirection::Any);
    dep.distances.assign(depth, 0);
    return dep;
}

// The dependence between two accesses to the same base (possibly the same
// access in two iterations), or nothing if they never touch the same bytes.
// Vectors are normalized so the earlier iteration is the source.
std::optional<DependenceVector> classify_same_base(const LoopNest& nest, const NestMemoryAccess& a,
                                                  const NestMemoryAccess& b) {
    const size_t depth = nest.depth();
    if (!a.analyzable || !b.analyzable || a.scale != b.scale || a.elem_type != b.elem_type) {
        return unknown_dependence(depth);
    }
    std::vector<size_t> with_terms;
    std::vector<size_t> without_terms;
    for (size_t l = 0; l < depth; ++l) {
        const AccessIndexTerm* ta = term_at(a, l);
        const AccessIndexTerm* tb = term_at(b, l);
        if (!ta && !tb) {
            without_terms.push_back(l);
        } else if (ta && tb && same_stride(*ta, *tb)) {
            with_terms.push_back(l);
        } else {
            return unknown_dependence(depth);
        }
    }

    const int64_t elem = std::max<int64_t>(1, a.elem_type.is_void() ? a.scale : a.elem_type.size_in_bytes());
    const int64_t diff = static_cast<int64_t>(b.const_offset) - static_cast<int64_t>(a.const_offset);

    DependenceVector dep;
    dep.directions.assign(depth, DependenceDirection::Equal);
    dep.distances.assign(depth, 0);
    dep.has_distance = true;

    bool carried_by_term = false;
    if (with_terms.empty()) {
        if (diff >= elem || -diff >= elem) return std::nullopt;
    } else if (with_terms.size() == 1) {
        const size_t l = with_terms[0];
        const AccessIndexTerm& t = *term_at(a, l);
        if (t.symbolic_stride) {
            if (diff != 0) return unknown_dependence(depth);
        } else {
            const int64_t unit = t.const_stride * static_cast<int64_t>(a.scale) * nest.level(l).step;
            if (unit == 0 || diff % unit != 0) return unknown_dependence(depth);
            const int64_t dist = diff / unit;
            if (dist != 0) {
                dep.directions[l] = DependenceDirection::Forward;
                dep.distances[l] = dist < 0 ? -dist : dist;
                carried_by_term = true;
            }
        }
    } else if (with_terms.size() == 2 && diff == 0 &&
               row_major_injective(nest, *term_at(a, with_terms[0]), *term_at(a, with_terms[1]))) {
        // Same address only in the same (i, j): nothing carried by these levels.
    } else {
        return unknown_dependence(depth);
    }

    // Levels the address does not depend on see the same bytes in every
    // iteration. One such level carries the dependence forward; with two or
    // more, or together with a distance on another level, iterations can be
    // related in both directions.
    if (without_terms.size() >= 2 || (!without_terms.empty() && carried_by_term)) {
        return unknown_dependence(depth);
    }
    if (without_terms.size() == 1) {
        dep.directions[without_terms[0]] = DependenceDirection::Forward;
        dep.has_distance = false;
    }
    dep.is_loop_independent = std::all_of(dep.directions.begin(), dep.directions.end(),
                                          [](DependenceDirection d) { return d == DependenceDirection::Equal; });
    return dep;
}

} // namespace

void analyze_nest_memory_accesses(Function& fn, LoopNest& nest) {
    (void)fn;
    nest.memory_accesses().clear();
    if (nest.depth() == 0 || !nest.level(0).loop) return;

    // The outermost loop's blocks include every inner loop's.
    for (BasicBlock* bb : nest.level(0).loop->blocks()) {
        if (!bb) continue;
        for (Instruction* inst = bb->head(); inst != nullptr; inst = inst->next()) {
            const Opcode op = inst->opcode();
            if (op == Opcode::load_indexed || op == Opcode::store_indexed) {
                NestMemoryAccess acc;
                acc.inst = inst;
                acc.is_store = (op == Opcode::store_indexed);
                acc.elem_type = inst->memory_type();
                acc.base = inst->operand(0);
                acc.scale = inst->scale();

                int64_t extra_offset = 0;
                bool ok = true;
                parse_index_expr(inst->operand(1), nest, acc.terms, extra_offset, ok);
                const int64_t elem_sz = acc.scale > 0 ? acc.scale
                                      : (acc.elem_type.is_void() ? 1 : static_cast<int64_t>(acc.elem_type.size_in_bytes()));
                const int64_t offset = static_cast<int64_t>(inst->offset()) + extra_offset * elem_sz;
                if (offset < INT32_MIN || offset > INT32_MAX) ok = false;
                acc.const_offset = ok ? static_cast<int32_t>(offset) : 0;
                acc.analyzable = ok;
                nest.memory_accesses().push_back(acc);
            } else if (has_unmodeled_effect(*inst)) {
                NestMemoryAccess acc;
                acc.inst = inst;
                acc.is_store = true;
                acc.analyzable = false;
                nest.memory_accesses().push_back(acc);
            }
        }
    }
}

void compute_nest_dependences(Function& fn, LoopNest& nest) {
    nest.dependences().clear();
    const auto& accesses = nest.memory_accesses();
    AliasAnalysis aa(fn);

    // Every pair, including an access with itself in another iteration.
    for (size_t i = 0; i < accesses.size(); ++i) {
        for (size_t j = i; j < accesses.size(); ++j) {
            const auto& a1 = accesses[i];
            const auto& a2 = accesses[j];
            if (!a1.is_store && !a2.is_store) continue;

            if (!a1.analyzable || !a2.analyzable) {
                nest.dependences().push_back(unknown_dependence(nest.depth()));
                continue;
            }
            if (a1.base != a2.base) {
                if (a1.base && a2.base && aa.alias(a1.base, a2.base) == AliasResult::NoAlias) continue;
                nest.dependences().push_back(unknown_dependence(nest.depth()));
                continue;
            }
            if (auto dep = classify_same_base(nest, a1, a2)) nest.dependences().push_back(*dep);
        }
    }
}

// Tiling reorders iterations along every level at once, which preserves a
// dependence only if none of its directions points backwards.
bool check_nest_tiling_legality(const LoopNest& nest) {
    for (const auto& dep : nest.dependences()) {
        if (dep.is_loop_independent) continue;
        for (DependenceDirection dir : dep.directions) {
            if (dir == DependenceDirection::Backward || dir == DependenceDirection::Any) return false;
        }
    }
    return true;
}

bool check_nest_interchange_legality(const LoopNest& nest, size_t level_a, size_t level_b) {
    if (level_a == level_b) return true;
    if (level_a >= nest.depth() || level_b >= nest.depth()) return false;

    for (const auto& dep : nest.dependences()) {
        if (dep.is_loop_independent) continue;

        auto dirs = dep.directions;
        std::swap(dirs[level_a], dirs[level_b]);

        for (DependenceDirection dir : dirs) {
            if (dir == DependenceDirection::Equal) continue;
            if (dir == DependenceDirection::Backward || dir == DependenceDirection::Any) {
                return false;
            }
            if (dir == DependenceDirection::Forward) {
                break;
            }
        }
    }
    return true;
}

bool detect_matrix_multiply_pattern(const LoopNest& nest) {
    if (nest.depth() != 3) return false;
    if (!nest.has_reduction()) return false;

    bool has_a = false;
    bool has_b = false;
    bool has_c = false;

    for (const auto& acc : nest.memory_accesses()) {
        if (!acc.analyzable) return false;
        bool has_i = false;
        bool has_j = false;
        bool has_k = false;

        for (const auto& t : acc.terms) {
            if (t.level_index == 0) has_i = true;
            if (t.level_index == 1) has_j = true;
            if (t.level_index == 2) has_k = true;
        }

        if (!acc.is_store && has_i && has_k && !has_j) has_a = true;
        if (!acc.is_store && has_k && has_j && !has_i) has_b = true;
        if (acc.is_store && has_i && has_j && !has_k) has_c = true;
    }

    return has_a && has_b && has_c;
}

} // namespace brass
