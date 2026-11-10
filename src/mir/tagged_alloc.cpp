// Objects from the module's tagged allocator (include/brass/mir/tagged_alloc.hpp).
#include <brass/mir/tagged_alloc.hpp>
#include <brass/gc/object.hpp>
#include <brass/mir/module.hpp>
#include <brass/mir/uses.hpp>

#include <optional>
#include <unordered_set>

namespace brass {

namespace {

constexpr uint64_t kAddr = gc::kAddressMask;

const Module* module_of(const Instruction* inst) noexcept {
    const BasicBlock* bb = inst ? inst->parent() : nullptr;
    const Function* fn = bb ? bb->parent() : nullptr;
    return fn ? fn->parent() : nullptr;
}

bool constant(const Value* v, int64_t& out) noexcept {
    const Instruction* def = v && v->is_instruction() ? v->defining_instruction() : nullptr;
    if (!def) return false;
    if (def->opcode() == Opcode::iconst_i64) {
        out = def->imm_i64();
        return true;
    }
    if (def->opcode() == Opcode::iconst_i32) {
        out = def->imm_i32();
        return true;
    }
    return false;
}

// Where an instruction reads a value: an operand, its state map, or an
// edge's argument (to `target`'s parameter `index`).
enum class Slot : uint8_t { Operand, State, Edge };

struct UseRef {
    Instruction* inst = nullptr;
    Slot slot = Slot::Operand;
    uint32_t index = 0;
    BasicBlock* target = nullptr;
};

class Analyzer {
public:
    Analyzer(const Function& fn, uint64_t tag, std::vector<TaggedAllocSite>& sites,
             std::unordered_map<const Value*, TaggedDerivation>& derived)
        : fn_(fn), tag_(tag), sites_(sites), derived_(derived) {}

    void run();

private:
    void index_uses();
    void escape(uint32_t site, std::string_view why, const Instruction* at) {
        TaggedAllocSite& s = sites_[site];
        if (s.escape.empty()) {
            s.escape = why;
            s.escape_at = at;
        }
    }
    bool add_derived(const Value* v, const TaggedDerivation& d, const Instruction* at);
    // The derivation an operand use makes, if the instruction derives one.
    std::optional<TaggedDerivation> derive(const Instruction& inst, size_t index, const TaggedDerivation& d) const;
    // The constant result of an operand use that folds, if it does.
    std::optional<int64_t> fold(const Instruction& inst, size_t index, const TaggedDerivation& d) const;
    void classify(uint32_t site, const Value* v, const TaggedDerivation& d);

    const Function& fn_;
    uint64_t tag_;
    std::vector<TaggedAllocSite>& sites_;
    std::unordered_map<const Value*, TaggedDerivation>& derived_;
    std::unordered_map<const Value*, std::vector<UseRef>> uses_;
    std::vector<const Value*> work_;
    std::vector<const Value*> params_;
};

void Analyzer::index_uses() {
    for (BasicBlock* bb : fn_.blocks()) {
        if (!bb) continue;
        for (Instruction* inst : *bb) {
            for (size_t i = 0; i < inst->operand_count(); ++i) {
                if (inst->operand(i)) uses_[inst->operand(i)].push_back({inst, Slot::Operand, static_cast<uint32_t>(i)});
            }
            for (size_t i = 0; i < inst->state_map().size(); ++i) {
                if (inst->state_map()[i]) uses_[inst->state_map()[i]].push_back({inst, Slot::State, static_cast<uint32_t>(i)});
            }
            for_each_edge(*inst, [&](BranchTarget& t) {
                for (size_t i = 0; i < t.args.size(); ++i) {
                    if (t.args[i]) uses_[t.args[i]].push_back({inst, Slot::Edge, static_cast<uint32_t>(i), t.block});
                }
            });
        }
    }
}

bool Analyzer::add_derived(const Value* v, const TaggedDerivation& d, const Instruction* at) {
    auto [it, inserted] = derived_.emplace(v, d);
    if (inserted) {
        sites_[d.site].values.push_back(v);
        work_.push_back(v);
        return true;
    }
    const TaggedDerivation& had = it->second;
    if (had.site != d.site || had.form != d.form || had.delta != d.delta) {
        escape(d.site, "merges with another object or form", at);
        escape(had.site, "merges with another object or form", at);
    }
    return false;
}

std::optional<TaggedDerivation> Analyzer::derive(const Instruction& inst, size_t index,
                                                 const TaggedDerivation& d) const {
    const Opcode op = inst.opcode();
    if (op == Opcode::bitcast_i64_tagged || op == Opcode::bitcast_tagged_i64) {
        if (d.form == TaggedForm::Box) return d;
        return std::nullopt;
    }
    if (inst.operand_count() != 2 || !inst.result()) return std::nullopt;
    const bool commutes = op == Opcode::and_ || op == Opcode::or_ || op == Opcode::xor_ || op == Opcode::add;
    if (index == 1 && !commutes) return std::nullopt;
    int64_t c = 0;
    if (!constant(inst.operand(1 - index), c)) return std::nullopt;
    const uint64_t u = static_cast<uint64_t>(c);
    TaggedDerivation out = d;
    if (d.form == TaggedForm::Box) {
        if ((u & kAddr) == kAddr && op == Opcode::and_) {
            if ((tag_ & u) == 0) {
                out.form = TaggedForm::Address;
                out.delta = 0;
                return out;
            }
            if ((tag_ & u) == tag_) return out;
            return std::nullopt;
        }
        if ((u & kAddr) != 0) return std::nullopt;
        if (op == Opcode::or_ && (tag_ | u) == tag_) return out;
        if (op == Opcode::xor_ && (tag_ ^ u) == 0) {
            out.form = TaggedForm::Address;
            out.delta = 0;
            return out;
        }
        return std::nullopt;
    }
    switch (op) {
        case Opcode::add:
            out.delta += c;
            return out;
        case Opcode::sub:
            out.delta -= c;
            return out;
        case Opcode::and_:
            if ((u & kAddr) == kAddr) return out;
            return std::nullopt;
        case Opcode::or_:
        case Opcode::xor_:
            if (d.delta == 0 && u == tag_) {
                out.form = TaggedForm::Box;
                return out;
            }
            return std::nullopt;
        default:
            return std::nullopt;
    }
}

// A comparison of every `address | tag` with `c`, when one answer holds for
// all of them.
std::optional<int64_t> compare_range(Opcode op, uint64_t lo, uint64_t hi, uint64_t c, bool box_on_left) {
    // Normalize to `box OP c`.
    if (!box_on_left) {
        switch (op) {
            case Opcode::ult: op = Opcode::ugt; break;
            case Opcode::ule: op = Opcode::uge; break;
            case Opcode::ugt: op = Opcode::ult; break;
            case Opcode::uge: op = Opcode::ule; break;
            case Opcode::slt: op = Opcode::sgt; break;
            case Opcode::sle: op = Opcode::sge; break;
            case Opcode::sgt: op = Opcode::slt; break;
            case Opcode::sge: op = Opcode::sle; break;
            default: break;
        }
    }
    const int64_t slo = static_cast<int64_t>(lo), shi = static_cast<int64_t>(hi), sc = static_cast<int64_t>(c);
    auto answer = [](bool always, bool never) -> std::optional<int64_t> {
        if (always) return 1;
        if (never) return 0;
        return std::nullopt;
    };
    switch (op) {
        case Opcode::eq: return answer(false, c < lo || c > hi);
        case Opcode::ne: return answer(c < lo || c > hi, false);
        case Opcode::ult: return answer(hi < c, lo >= c);
        case Opcode::ule: return answer(hi <= c, lo > c);
        case Opcode::ugt: return answer(lo > c, hi <= c);
        case Opcode::uge: return answer(lo >= c, hi < c);
        case Opcode::slt: return answer(shi < sc, slo >= sc);
        case Opcode::sle: return answer(shi <= sc, slo > sc);
        case Opcode::sgt: return answer(slo > sc, shi <= sc);
        case Opcode::sge: return answer(slo >= sc, shi < sc);
        default: return std::nullopt;
    }
}

std::optional<int64_t> Analyzer::fold(const Instruction& inst, size_t index, const TaggedDerivation& d) const {
    if (d.form != TaggedForm::Box || inst.operand_count() != 2) return std::nullopt;
    const Opcode op = inst.opcode();
    const Value* other = inst.operand(1 - index);
    int64_t c = 0;
    const bool other_const = constant(other, c);
    const uint64_t u = static_cast<uint64_t>(c);
    if (is_comparison(op)) {
        if (!other || other->type().kind() != TypeKind::I64) return std::nullopt;
        if (op == Opcode::eq || op == Opcode::ne) {
            auto it = derived_.find(other);
            if (it != derived_.end()) {
                const TaggedDerivation& o = it->second;
                if (o.form != TaggedForm::Box) return std::nullopt;
                const bool same = o.site == d.site;
                return (op == Opcode::eq) == same ? 1 : 0;
            }
            if (!other_const) return op == Opcode::eq ? 0 : 1;
        }
        if (!other_const) return std::nullopt;
        return compare_range(op, tag_, tag_ | kAddr, u, index == 0);
    }
    if (index != 0 && op != Opcode::and_) return std::nullopt;
    if (!other_const) return std::nullopt;
    if (op == Opcode::and_ && (u & kAddr) == 0) return static_cast<int64_t>(tag_ & u);
    if ((op == Opcode::lshr || op == Opcode::ashr) && c >= 48 && c < 64) {
        return op == Opcode::lshr ? static_cast<int64_t>(tag_ >> c) : static_cast<int64_t>(tag_) >> c;
    }
    return std::nullopt;
}

void Analyzer::classify(uint32_t site, const Value* v, const TaggedDerivation& d) {
    auto it = uses_.find(v);
    if (it == uses_.end()) return;
    TaggedAllocSite& s = sites_[site];
    for (const UseRef& u : it->second) {
        Instruction& inst = *u.inst;
        if (u.slot == Slot::State) {
            escape(site, "is deopt state", &inst);
            continue;
        }
        if (u.slot == Slot::Edge) continue;  // the parameter's checks
        const Opcode op = inst.opcode();
        TaggedUse use{&inst};
        if (derive(inst, u.index, d)) {
            use.kind = TaggedUseKind::Derive;
        } else if (op == Opcode::load && u.index == 0 && d.form == TaggedForm::Address) {
            use.kind = TaggedUseKind::Load;
            use.offset = d.delta + inst.offset();
        } else if (op == Opcode::store && u.index == 0 && d.form == TaggedForm::Address) {
            use.kind = TaggedUseKind::Store;
            use.offset = d.delta + inst.offset();
        } else if (op == Opcode::write_barrier && u.index == 0) {
            use.kind = TaggedUseKind::Barrier;
        } else if (op == Opcode::keep_alive) {
            use.kind = TaggedUseKind::KeepAlive;
        } else if (auto f = fold(inst, u.index, d)) {
            use.kind = TaggedUseKind::Fold;
            use.folded = *f;
        } else {
            escape(site, op == Opcode::store || op == Opcode::store_indexed ? "is stored"
                         : is_call(op) || op == Opcode::invoke             ? "is passed to a call"
                         : op == Opcode::ret                               ? "is returned"
                                                                           : "has a use the analysis does not follow",
                   &inst);
            continue;
        }
        bool seen = false;
        for (const TaggedUse& prior : s.uses) seen = seen || prior.inst == &inst;
        if (!seen) s.uses.push_back(use);
    }
}

void Analyzer::run() {
    index_uses();
    for (BasicBlock* bb : fn_.blocks()) {
        if (!bb) continue;
        for (Instruction* inst : *bb) {
            if (!is_tagged_allocation(inst) || !inst->result()) continue;
            TaggedAllocSite s;
            s.call = inst;
            s.bytes = tagged_allocation_bytes(inst);
            sites_.push_back(std::move(s));
            add_derived(inst->result(), {static_cast<uint32_t>(sites_.size() - 1), TaggedForm::Box, 0}, inst);
        }
    }
    // The closure of the deriving instructions and the edges.
    std::unordered_set<const Value*> assumed;
    while (!work_.empty()) {
        const Value* v = work_.back();
        work_.pop_back();
        const TaggedDerivation d = derived_.at(v);
        auto it = uses_.find(v);
        if (it == uses_.end()) continue;
        for (const UseRef& u : it->second) {
            if (u.slot == Slot::Edge) {
                if (!u.target || entered_from_outside(fn_, u.target) || u.index >= u.target->param_count()) {
                    escape(d.site, "flows into a block entered from outside the CFG", u.inst);
                    continue;
                }
                const Value* p = u.target->param(u.index);
                if (assumed.insert(p).second) params_.push_back(p);
                add_derived(p, d, u.inst);
            } else if (u.slot == Slot::Operand) {
                if (auto nd = derive(*u.inst, u.index, d)) add_derived(u.inst->result(), *nd, u.inst);
            }
        }
    }
    // A derived parameter takes nothing else on any edge.
    if (!params_.empty()) {
        std::unordered_map<const Value*, std::vector<const Value*>> incoming;
        for (BasicBlock* bb : fn_.blocks()) {
            if (!bb) continue;
            for (Instruction* inst : *bb) {
                for_each_edge(*inst, [&](BranchTarget& t) {
                    for (size_t i = 0; i < t.args.size() && i < t.block->param_count(); ++i) {
                        const Value* p = t.block->param(i);
                        if (assumed.count(p)) incoming[p].push_back(t.args[i]);
                    }
                });
            }
        }
        for (const Value* p : params_) {
            const TaggedDerivation& d = derived_.at(p);
            for (const Value* in : incoming[p]) {
                auto it = derived_.find(in);
                if (it == derived_.end() || it->second.site != d.site || it->second.form != d.form ||
                    it->second.delta != d.delta) {
                    escape(d.site, "flows into a parameter that also takes another value", nullptr);
                    if (it != derived_.end()) escape(it->second.site, "flows into a parameter that also takes another value", nullptr);
                }
            }
        }
    }
    for (uint32_t site = 0; site < sites_.size(); ++site) {
        for (size_t i = 0; i < sites_[site].values.size(); ++i) {
            const Value* v = sites_[site].values[i];
            classify(site, v, derived_.at(v));
        }
    }
}

} // namespace

bool entered_from_outside(const Function& fn, const BasicBlock* bb) {
    if (bb == fn.entry_block()) return true;
    if (bb->head() && bb->head()->opcode() == Opcode::landing_pad) return true;
    for (const auto& rp : fn.resume_points()) {
        if (rp.second == bb) return true;
    }
    return false;
}

bool is_tagged_allocation(const Instruction* inst) noexcept {
    if (!inst || inst->opcode() != Opcode::call || inst->operand_count() != 2) return false;
    const Module* mod = module_of(inst);
    return mod && mod->tagged_allocator() && inst->symbol() == mod->tagged_allocator().symbol;
}

int64_t tagged_allocation_bytes(const Instruction* inst) noexcept {
    if (!is_tagged_allocation(inst)) return -1;
    int64_t bytes = 0;
    if (!constant(inst->operand(0), bytes) || bytes < 0 || bytes % 8 != 0) return -1;
    return bytes;
}

TaggedAllocations::TaggedAllocations(const Function& fn) {
    const Module* mod = fn.parent();
    if (!mod || !mod->tagged_allocator()) return;
    // A tag that touches the address bits leaves nothing to follow.
    const uint64_t tag = mod->tagged_allocator().tag;
    if ((tag & kAddr) != 0) return;
    Analyzer(fn, tag, sites_, derived_).run();
}

const TaggedDerivation* TaggedAllocations::derivation(const Value* v) const noexcept {
    auto it = derived_.find(v);
    return it == derived_.end() ? nullptr : &it->second;
}

size_t TaggedAllocations::non_escaping() const noexcept {
    size_t n = 0;
    for (const TaggedAllocSite& s : sites_) n += s.escapes() ? 0 : 1;
    return n;
}

} // namespace brass
