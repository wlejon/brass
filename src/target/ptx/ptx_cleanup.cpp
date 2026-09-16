// PtxCleanup: the passes declared in ptx_cleanup.hpp. Each pass recomputes
// def/use counts from scratch (functions are a few hundred instructions) and
// keeps them consistent while it rewrites, so a single sweep resolves copy
// chains and dead chains.

#include <brass/target/ptx/ptx_cleanup.hpp>

#include <algorithm>
#include <array>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace brass::ptx {

namespace {

// ---------------------------------------------------------------------------
// Register enumeration
// ---------------------------------------------------------------------------

template <typename O, typename F>
void for_each_reg_in_operand(O& o, F&& f) {
    switch (o.kind) {
        case OperandKind::Reg:    f(o.reg_val); break;
        case OperandKind::Addr:   if (o.addr_base.valid()) f(o.addr_base); break;
        case OperandKind::Vector: for (auto& r : o.elems) f(r); break;
        default: break;
    }
}

// Registers written by an instruction: every destination operand (a ld.v4
// tuple defines each element).
template <typename I, typename F>
void for_each_def(I& inst, F&& f) {
    for (auto& o : inst.dsts) for_each_reg_in_operand(o, f);
}

// Registers read: every source operand (address bases and vector elements
// included) and the guard predicate.
template <typename I, typename F>
void for_each_use(I& inst, F&& f) {
    if (inst.has_guard) f(inst.guard_reg);
    for (auto& o : inst.srcs) for_each_reg_in_operand(o, f);
}

template <typename Fn, typename F>
void for_each_inst(Fn& fn, F&& f) {
    for (auto& block : fn.blocks) {
        for (auto& inst : block->insts) f(inst);
    }
}

size_t cls_index(RegClass rc) { return static_cast<size_t>(rc); }

// Per-class def and use counts, sized by the highest register index that is
// declared or mentioned.
struct RegCounts {
    std::array<std::vector<uint32_t>, kRegClassCount> defs;
    std::array<std::vector<uint32_t>, kRegClassCount> uses;

    explicit RegCounts(const Function& fn) {
        std::array<uint32_t, kRegClassCount> size = fn.reg_counts;
        auto grow = [&](Reg r) { if (r.valid()) size[cls_index(r.cls)] = std::max(size[cls_index(r.cls)], r.index + 1); };
        for_each_inst(fn, [&](const Inst& inst) { for_each_def(inst, grow); for_each_use(inst, grow); });
        for (size_t c = 0; c < kRegClassCount; ++c) {
            defs[c].assign(size[c], 0);
            uses[c].assign(size[c], 0);
        }
        for_each_inst(fn, [&](const Inst& inst) {
            for_each_def(inst, [&](Reg r) { if (r.valid()) ++def(r); });
            for_each_use(inst, [&](Reg r) { if (r.valid()) ++use(r); });
        });
    }

    uint32_t& def(Reg r) { return defs[cls_index(r.cls)][r.index]; }
    uint32_t& use(Reg r) { return uses[cls_index(r.cls)][r.index]; }
};

// ---------------------------------------------------------------------------
// 1. Copy propagation
// ---------------------------------------------------------------------------

// `mov d, s` between registers of one class with no guard and no modifier.
// Cross-class movs (mov.b32 %r, %f bitcasts) and special/symbol sources are
// not copies.
bool is_plain_copy(const Inst& I) {
    return I.op == Opcode::mov && !I.has_guard && I.dsts.size() == 1 && I.srcs.size() == 1 &&
           I.dsts[0].is_reg() && I.srcs[0].is_reg() &&
           I.dsts[0].reg_val.valid() && I.srcs[0].reg_val.valid() &&
           I.dsts[0].reg_val.cls == I.srcs[0].reg_val.cls &&
           I.rounding == Rounding::none && !I.is_approx && !I.is_ftz && !I.is_sat;
}

// Every read of `from` (sources, address bases, vector elements, guards)
// becomes a read of `to`. Destinations are left alone.
void replace_uses(Function& fn, Reg from, Reg to) {
    for_each_inst(fn, [&](Inst& inst) {
        for_each_use(inst, [&](Reg& r) { if (r == from) r = to; });
    });
}

// The instruction that defines `s` earlier in `bb`, when the mov at `at` can
// be folded into it: the def is unguarded and scalar, and between it and the
// mov nothing defines or reads `d` and no branch leaves the block (a guarded
// `bra` in between would make the early write of `d` visible on the taken
// path, where the mov never executed). Returns nullptr otherwise.
Inst* coalescable_def(Block& bb, size_t at, Reg d, Reg s) {
    for (size_t j = at; j-- > 0;) {
        Inst& J = bb.insts[j];
        bool defines_s = false;
        bool touches_d = false;
        for_each_def(J, [&](Reg r) { if (r == s) defines_s = true; if (r == d) touches_d = true; });
        if (defines_s) {
            if (touches_d || J.has_guard || J.dsts.size() != 1 || !J.dsts[0].is_reg()) return nullptr;
            return &J; // J reading d itself is fine: reads precede the write
        }
        if (J.op == Opcode::bra || J.op == Opcode::ret || J.op == Opcode::exit || J.op == Opcode::trap) return nullptr;
        for_each_use(J, [&](Reg r) { if (r == d) touches_d = true; });
        if (touches_d) return nullptr;
    }
    return nullptr;
}

} // namespace

size_t propagate_copies(Function& fn) {
    RegCounts counts(fn);
    size_t removed = 0;
    for (auto& block : fn.blocks) {
        Block& bb = *block;
        for (size_t i = 0; i < bb.insts.size();) {
            Inst& I = bb.insts[i];
            if (!is_plain_copy(I)) { ++i; continue; }
            Reg d = I.dsts[0].reg_val;
            Reg s = I.srcs[0].reg_val;
            auto erase = [&] { bb.insts.erase(bb.insts.begin() + static_cast<std::ptrdiff_t>(i)); ++removed; };

            if (d == s) {
                --counts.def(d);
                --counts.use(s);
                erase();
                continue;
            }
            // Both single-def: d is s wherever d is read.
            if (counts.def(d) == 1 && counts.def(s) == 1) {
                replace_uses(fn, d, s);
                counts.use(s) = counts.use(s) - 1 + counts.use(d); // d's readers read s; the mov's own read is gone
                counts.use(d) = 0;
                counts.def(d) = 0;
                erase();
                continue;
            }
            // s is a temporary consumed only by this mov: its def writes d directly.
            if (counts.def(s) == 1 && counts.use(s) == 1) {
                if (Inst* J = coalescable_def(bb, i, d, s)) {
                    J->dsts[0] = Operand::reg(d);
                    counts.def(s) = 0;
                    counts.use(s) = 0; // the mov's def of d moves to J: def(d) is unchanged
                    erase();
                    continue;
                }
            }
            ++i;
        }
    }
    return removed;
}

// ---------------------------------------------------------------------------
// 2. Dead instruction elimination
// ---------------------------------------------------------------------------

namespace {

bool has_side_effects(const Inst& I) {
    switch (I.op) {
        case Opcode::st: case Opcode::atom: case Opcode::bar: case Opcode::call:
        case Opcode::ret: case Opcode::bra: case Opcode::trap: case Opcode::exit:
        case Opcode::shfl: // warp-collective: keep even when the result is unused
            return true;
        case Opcode::mov:
            return I.srcs.size() == 1 && I.srcs[0].is_special() && !is_invariant(I.srcs[0].special_reg);
        default:
            return false;
    }
}

} // namespace

size_t eliminate_dead_instructions(Function& fn) {
    RegCounts counts(fn);
    size_t removed = 0;
    bool changed = true;
    while (changed) {
        changed = false;
        for (auto& block : fn.blocks) {
            Block& bb = *block;
            for (size_t i = 0; i < bb.insts.size();) {
                Inst& I = bb.insts[i];
                bool dead = !has_side_effects(I) && !I.dsts.empty();
                if (dead) for_each_def(I, [&](Reg r) { if (!r.valid() || counts.use(r) != 0) dead = false; });
                if (!dead) { ++i; continue; }
                for_each_use(I, [&](Reg r) { if (r.valid()) --counts.use(r); });
                for_each_def(I, [&](Reg r) { if (r.valid()) --counts.def(r); });
                bb.insts.erase(bb.insts.begin() + static_cast<std::ptrdiff_t>(i));
                ++removed;
                changed = true;
            }
        }
    }
    return removed;
}

// ---------------------------------------------------------------------------
// 3. Branch simplification
// ---------------------------------------------------------------------------

namespace {

const std::string* bra_target(const Inst& I) {
    if (I.op != Opcode::bra || I.srcs.size() != 1 || !I.srcs[0].is_label()) return nullptr;
    return &I.srcs[0].name;
}

// A block whose last instruction is not an unguarded bra/ret/trap/exit
// continues into the next block.
bool falls_through(const Block& b) {
    return b.insts.empty() || !b.insts.back().is_terminator();
}

size_t remove_unreachable_blocks(Function& fn) {
    size_t n = fn.blocks.size();
    if (n == 0) return 0;
    std::unordered_map<std::string, size_t> index;
    for (size_t i = 0; i < n; ++i) index[fn.blocks[i]->label] = i;

    std::vector<bool> reachable(n, false);
    std::vector<size_t> work{0};
    reachable[0] = true;
    while (!work.empty()) {
        size_t i = work.back();
        work.pop_back();
        auto visit = [&](size_t j) { if (!reachable[j]) { reachable[j] = true; work.push_back(j); } };
        for (const Inst& inst : fn.blocks[i]->insts) {
            if (const std::string* t = bra_target(inst)) {
                auto it = index.find(*t);
                if (it != index.end()) visit(it->second);
            }
        }
        if (i + 1 < n && falls_through(*fn.blocks[i])) visit(i + 1);
    }

    size_t removed = 0;
    std::vector<std::unique_ptr<Block>> kept;
    kept.reserve(n);
    for (size_t i = 0; i < n; ++i) {
        if (reachable[i]) kept.push_back(std::move(fn.blocks[i]));
        else ++removed;
    }
    fn.blocks = std::move(kept);
    return removed;
}

} // namespace

size_t simplify_branches(Function& fn) {
    size_t removed = remove_unreachable_blocks(fn);
    for (size_t i = 0; i + 1 < fn.blocks.size(); ++i) {
        Block& b = *fn.blocks[i];
        const std::string& next = fn.blocks[i + 1]->label;
        while (!b.insts.empty()) {
            Inst& last = b.insts.back();
            const std::string* target = bra_target(last);
            // `bra next` / `@p bra next` as the last instruction: both paths reach next.
            if (target && *target == next) {
                b.insts.pop_back();
                ++removed;
                continue;
            }
            // `@p bra next; bra B`  ->  `@!p bra B`
            if (target && !last.has_guard && b.insts.size() >= 2) {
                Inst& prev = b.insts[b.insts.size() - 2];
                const std::string* prev_target = bra_target(prev);
                if (prev_target && prev.has_guard && *prev_target == next) {
                    last.guard(prev.guard_reg, !prev.guard_negated);
                    b.insts.erase(b.insts.end() - 2);
                    ++removed;
                    continue;
                }
            }
            break;
        }
    }
    return removed;
}

// ---------------------------------------------------------------------------
// 4. Register renumbering
// ---------------------------------------------------------------------------

void renumber_registers(Function& fn) {
    std::array<std::vector<uint8_t>, kRegClassCount> used;
    auto mark = [&](Reg r) {
        if (!r.valid()) return;
        auto& u = used[cls_index(r.cls)];
        if (u.size() <= r.index) u.resize(r.index + 1, 0);
        u[r.index] = 1;
    };
    for_each_inst(fn, [&](const Inst& inst) { for_each_def(inst, mark); for_each_use(inst, mark); });

    std::array<std::vector<uint32_t>, kRegClassCount> remap;
    for (size_t c = 0; c < kRegClassCount; ++c) {
        remap[c].assign(used[c].size(), Reg::kInvalid);
        uint32_t next = 0;
        for (size_t i = 0; i < used[c].size(); ++i) {
            if (used[c][i]) remap[c][i] = next++;
        }
        fn.reg_counts[c] = next;
    }
    auto rename = [&](Reg& r) { if (r.valid()) r.index = remap[cls_index(r.cls)][r.index]; };
    for_each_inst(fn, [&](Inst& inst) { for_each_def(inst, rename); for_each_use(inst, rename); });
}

// ---------------------------------------------------------------------------
// Pipeline
// ---------------------------------------------------------------------------

size_t instruction_count(const Function& fn) {
    size_t n = 0;
    for (const auto& block : fn.blocks) n += block->insts.size();
    return n;
}

CleanupStats cleanup(Function& fn) {
    CleanupStats stats;
    stats.insts_before = instruction_count(fn);
    stats.branches_removed += simplify_branches(fn);
    for (;;) {
        size_t copies = propagate_copies(fn);
        size_t dead = eliminate_dead_instructions(fn);
        stats.copies_removed += copies;
        stats.dead_removed += dead;
        if (copies == 0 && dead == 0) break;
    }
    renumber_registers(fn);
    stats.insts_after = instruction_count(fn);
    return stats;
}

} // namespace brass::ptx
