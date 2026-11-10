// Scalar replacement of the module's tagged allocations (tagged_alloc.hpp).
//
// An object that does not escape and has a constant size becomes one SSA
// value per 8-byte word that some load reads: its stores define the word,
// its loads read it back, and the allocation, the stores, its barriers and
// the tag arithmetic go. The construction is SroaTransformer's (block
// parameters at the iterated dominance frontier of each word's definitions,
// where the word is live, renamed down the dominator tree).
//
// A word is carried as a `tagged` value when it is stored as an integer:
// the heap object's word is visited by the collector, which moves what it
// names and rewrites it, and a word held as an i64 across a GC point would
// keep the address the referent had. As a tagged value it is a root at every
// GC point it is live across, and a load reads its bits as they are then.
// (So an integer word that is no reference must not look like one: a
// reference tag over an address in the heap. The allocator's objects are
// traced as tagged words, which asks the same of them.) A word only ever
// stored as an f64 is carried as one.
//
// A load narrower than a word (an i32 or i8 read of a header, say) reads
// the bits it covers out of the word.
#include <brass/mir/sroa.hpp>
#include <brass/mir/builder.hpp>
#include <brass/mir/dominators.hpp>
#include <brass/mir/tagged_alloc.hpp>
#include <brass/mir/uses.hpp>

#include <algorithm>
#include <map>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace brass {

namespace {

enum class WordKind : uint8_t { Tagged, F64 };

struct Word {
    WordKind kind = WordKind::Tagged;
    bool stored = false;
    bool loaded = false;
};

// Whether every access of `site` is one this pass can rewrite, and the kind
// of each word.
bool plan_words(const TaggedAllocSite& site, std::map<int64_t, Word>& words) {
    const int64_t bytes = site.bytes;
    for (const TaggedUse& u : site.uses) {
        if (u.kind != TaggedUseKind::Load && u.kind != TaggedUseKind::Store) continue;
        const Instruction* inst = u.inst;
        const Type mem = inst->memory_type().is_void() ? inst->type() : inst->memory_type();
        const int64_t width = static_cast<int64_t>(mem.size_in_bytes());
        if (u.offset < 0 || width <= 0 || u.offset + width > bytes) return false;
        const int64_t w = u.offset / 8;
        if (u.kind == TaggedUseKind::Store) {
            const Value* val = inst->operand(1);
            if (!val || val->type() != mem || u.offset % 8 != 0) return false;
            WordKind k;
            if (mem == Type::f64()) {
                k = WordKind::F64;
            } else if (mem == Type::i64() || mem == Type::tagged()) {
                k = WordKind::Tagged;
            } else {
                return false;
            }
            auto [it, inserted] = words.try_emplace(w);
            if (it->second.stored && it->second.kind != k) return false;
            it->second.kind = k;
            it->second.stored = true;
        } else {
            if (inst->type() != mem) return false;
            const bool whole = width == 8 && u.offset % 8 == 0 &&
                               (mem == Type::i64() || mem == Type::f64() || mem == Type::tagged());
            const bool part = (mem == Type::i32() && u.offset % 4 == 0) || mem == Type::i8();
            if (!whole && !part) return false;
            words[w].loaded = true;
        }
    }
    return true;
}

class TaggedReplacer {
public:
    TaggedReplacer(Function& fn, const TaggedAllocSite& site, std::map<int64_t, Word> words, const DominatorTree& dom,
                   const std::unordered_map<const BasicBlock*, std::vector<BasicBlock*>>& df, SroaStats* stats)
        : fn_(fn), site_(site), words_(std::move(words)), dom_(dom), df_(df), stats_(stats), b_(fn) {}

    void run();

private:
    enum class Event : uint8_t { None, Load, Kill };
    const TaggedUse* use_of(const Instruction* inst) const {
        auto it = by_inst_.find(inst);
        return it == by_inst_.end() ? nullptr : it->second;
    }
    // The first event of word `w` in `bb`.
    Event block_event(const BasicBlock* bb, int64_t w) const;
    Type word_type(const Word& wd) const { return wd.kind == WordKind::F64 ? Type::f64() : Type::tagged(); }
    Value* undefined_word(const Word& wd);
    // The bits a load of `inst` reads from the word's value `word`.
    Value* read(Instruction* load, int64_t offset, const Word& wd, Value* word);

    Function& fn_;
    const TaggedAllocSite& site_;
    std::map<int64_t, Word> words_;
    const DominatorTree& dom_;
    const std::unordered_map<const BasicBlock*, std::vector<BasicBlock*>>& df_;
    SroaStats* stats_;
    Builder b_;
    std::unordered_map<const Instruction*, const TaggedUse*> by_inst_;
    Value* undef_tagged_ = nullptr;
    Value* undef_f64_ = nullptr;
};

TaggedReplacer::Event TaggedReplacer::block_event(const BasicBlock* bb, int64_t w) const {
    for (const Instruction* cur = bb->head(); cur; cur = cur->next()) {
        if (cur == site_.call) return Event::Kill;
        const TaggedUse* u = use_of(cur);
        if (!u || u->offset / 8 != w) continue;
        if (u->kind == TaggedUseKind::Load) return Event::Load;
        if (u->kind == TaggedUseKind::Store) return Event::Kill;
    }
    return Event::None;
}

Value* TaggedReplacer::undefined_word(const Word& wd) {
    Value*& slot = wd.kind == WordKind::F64 ? undef_f64_ : undef_tagged_;
    if (slot) return slot;
    BasicBlock* entry = fn_.entry_block();
    Instruction* at = entry->head();
    while (at && at->opcode() != Opcode::landing_pad && is_constant(at->opcode())) at = at->next();
    if (at) {
        b_.position_before(at);
    } else {
        b_.position_at_end(entry);
    }
    slot = wd.kind == WordKind::F64 ? b_.build_fconst_f64(0.0) : b_.build_bitcast_tagged_i64(b_.build_iconst_i64(0));
    return slot;
}

Value* TaggedReplacer::read(Instruction* load, int64_t offset, const Word& wd, Value* word) {
    b_.position_before(load);
    const Type t = load->type();
    if (wd.kind == WordKind::F64 && t == Type::f64()) return word;
    if (wd.kind == WordKind::Tagged && t == Type::tagged()) return word;
    Value* bits = wd.kind == WordKind::F64 ? b_.build_bitcast_i64_f64(word) : b_.build_bitcast_i64_tagged(word);
    if (t == Type::i64()) return bits;
    if (t == Type::f64()) return b_.build_bitcast_f64_i64(bits);
    if (t == Type::tagged()) return b_.build_bitcast_tagged_i64(bits);
    const int64_t shift = (offset % 8) * 8;
    if (shift != 0) bits = b_.build_lshr(bits, b_.build_iconst_i64(shift));
    return t == Type::i32() ? b_.build_trunc_i32(bits) : b_.build_trunc_i8(bits);
}

void TaggedReplacer::run() {
    for (const TaggedUse& u : site_.uses) by_inst_[u.inst] = &u;
    std::unordered_set<const Value*> derived(site_.values.begin(), site_.values.end());

    // Each loaded word's liveness: live into a block when a load of it is
    // reachable from there before a store to it or the allocation.
    std::vector<int64_t> live_words;
    for (const auto& [w, wd] : words_) {
        if (wd.loaded) live_words.push_back(w);
    }
    std::unordered_map<int64_t, std::unordered_set<const BasicBlock*>> live_in;
    std::unordered_map<int64_t, std::unordered_set<BasicBlock*>> param_blocks;
    for (int64_t w : live_words) {
        auto& live = live_in[w];
        std::vector<const BasicBlock*> work;
        for (const TaggedUse& u : site_.uses) {
            if (u.kind != TaggedUseKind::Load || u.offset / 8 != w) continue;
            const BasicBlock* bb = u.inst->parent();
            if (block_event(bb, w) == Event::Load && live.insert(bb).second) work.push_back(bb);
        }
        while (!work.empty()) {
            const BasicBlock* bb = work.back();
            work.pop_back();
            for (const BasicBlock* p : bb->predecessors()) {
                if (p && dom_.is_reachable(p) && !live.count(p) && block_event(p, w) == Event::None) {
                    live.insert(p);
                    work.push_back(p);
                }
            }
        }
        // The iterated dominance frontier of the word's definitions.
        std::vector<BasicBlock*> defs{site_.call->parent()};
        for (const TaggedUse& u : site_.uses) {
            if (u.kind == TaggedUseKind::Store && u.offset / 8 == w) defs.push_back(u.inst->parent());
        }
        std::unordered_set<BasicBlock*> idf;
        std::unordered_set<BasicBlock*> queued(defs.begin(), defs.end());
        for (size_t i = 0; i < defs.size(); ++i) {
            auto it = df_.find(defs[i]);
            if (it == df_.end()) continue;
            for (BasicBlock* y : it->second) {
                if (idf.insert(y).second && queued.insert(y).second) defs.push_back(y);
            }
        }
        for (BasicBlock* y : idf) {
            if (live.count(y)) param_blocks[w].insert(y);
        }
    }

    // Parameters, in word order per block.
    std::unordered_map<BasicBlock*, std::vector<std::pair<int64_t, Value*>>> params;
    for (int64_t w : live_words) {
        for (BasicBlock* bb : param_blocks[w]) {
            Value* p = b_.create_value(word_type(words_.at(w)));
            bb->add_param(p);
            params[bb].push_back({w, p});
            if (stats_) stats_->block_params_created++;
        }
    }

    // Rename down the dominator tree.
    std::unordered_map<const Value*, Value*> replaced;
    std::vector<Instruction*> dead;
    std::vector<std::unordered_map<int64_t, Value*>> stack;
    stack.emplace_back();
    auto enter = [&](const BasicBlock* cbb) {
        BasicBlock* bb = const_cast<BasicBlock*>(cbb);
        std::unordered_map<int64_t, Value*> cur = stack.back();
        auto pit = params.find(bb);
        if (pit != params.end()) {
            for (const auto& [w, p] : pit->second) cur[w] = p;
        }
        for (Instruction* inst = bb->head(); inst; inst = inst->next()) {
            if (inst == site_.call) {
                for (int64_t w : live_words) cur[w] = undefined_word(words_.at(w));
                dead.push_back(inst);
                continue;
            }
            const TaggedUse* u = use_of(inst);
            if (!u) continue;
            const int64_t w = u->offset / 8;
            switch (u->kind) {
                case TaggedUseKind::Store: {
                    auto wit = words_.find(w);
                    if (wit->second.loaded) {
                        Value* v = inst->operand(1);
                        // A value read from this object before: its word.
                        if (auto rit = replaced.find(v); rit != replaced.end()) v = rit->second;
                        if (v->type() == Type::i64()) {
                            b_.position_before(inst);
                            v = b_.build_bitcast_tagged_i64(v);
                        }
                        cur[w] = v;
                    }
                    dead.push_back(inst);
                    if (stats_) stats_->stores_eliminated++;
                    break;
                }
                case TaggedUseKind::Load: {
                    const Word& wd = words_.at(w);
                    Value* word = cur.count(w) ? cur[w] : nullptr;
                    if (!word) word = undefined_word(wd);
                    replaced[inst->result()] = read(inst, u->offset, wd, word);
                    dead.push_back(inst);
                    if (stats_) stats_->loads_eliminated++;
                    break;
                }
                case TaggedUseKind::Fold: {
                    b_.position_before(inst);
                    replaced[inst->result()] = inst->type() == Type::i32()
                                                   ? b_.build_iconst_i32(static_cast<int32_t>(u->folded))
                                                   : b_.build_iconst_i64(u->folded);
                    dead.push_back(inst);
                    break;
                }
                case TaggedUseKind::Barrier:
                case TaggedUseKind::KeepAlive:
                case TaggedUseKind::Derive:
                    dead.push_back(inst);
                    break;
            }
        }
        // The words each successor's parameters take, on every edge to it.
        if (Instruction* term = bb->terminator()) {
            for_each_edge(*term, [&](BranchTarget& t) {
                auto sit = params.find(t.block);
                if (sit == params.end()) return;
                for (const auto& [w, p] : sit->second) {
                    Value* v = cur.count(w) ? cur[w] : nullptr;
                    t.args.push_back(v ? v : undefined_word(words_.at(w)));
                }
            });
        }
        stack.push_back(std::move(cur));
    };
    walk_dominator_tree(dom_, fn_.entry_block(), enter, [&](const BasicBlock*) { stack.pop_back(); });
    // An unreachable block's edge into a block given parameters passes
    // undefined words, so every edge keeps its arity.
    for (BasicBlock* bb : fn_.blocks()) {
        if (!bb || dom_.is_reachable(bb) || !bb->terminator()) continue;
        for_each_edge(*bb->terminator(), [&](BranchTarget& t) {
            auto sit = params.find(t.block);
            if (sit == params.end()) return;
            for (const auto& [w, p] : sit->second) t.args.push_back(undefined_word(words_.at(w)));
        });
    }

    // Every read of a removed load or fold takes its replacement. Loads of
    // one word read each other's values only through stores, which the
    // replacement already resolved, so one level is enough.
    if (!replaced.empty()) {
        for (BasicBlock* bb : fn_.blocks()) {
            if (!bb) continue;
            for (Instruction* inst : *bb) {
                for_each_use_slot(*inst, [&](Value*& v) {
                    auto it = v ? replaced.find(v) : replaced.end();
                    if (it != replaced.end()) v = it->second;
                });
            }
        }
    }
    for (Instruction* inst : dead) {
        if (inst->parent()) inst->parent()->remove_instruction(inst);
    }

    // The derived parameters and what their edges pass them.
    std::unordered_map<BasicBlock*, std::vector<uint32_t>> drop;
    for (const Value* v : site_.values) {
        if (v->is_block_param() && v->defining_block()) drop[v->defining_block()].push_back(v->param_index());
    }
    for (auto& [bb, idxs] : drop) {
        std::sort(idxs.begin(), idxs.end(), std::greater<uint32_t>());
        for (BasicBlock* pred : fn_.blocks()) {
            if (!pred || !pred->terminator()) continue;
            for_each_edge(*pred->terminator(), [&, target = bb](BranchTarget& t) {
                if (t.block != target) return;
                for (uint32_t i : idxs) {
                    if (i < t.args.size()) t.args.erase(t.args.begin() + i);
                }
            });
        }
        for (uint32_t i : idxs) bb->params().erase(bb->params().begin() + i);
        for (size_t k = 0; k < bb->param_count(); ++k) bb->params()[k]->set_block_param(bb, static_cast<uint32_t>(k));
    }
    if (stats_) stats_->allocations_eliminated++;
}

} // namespace

// True when a block entered from outside the CFG other than the entry is
// dominated by `at`: reachable while an object made in `at` is.
static bool outside_after(const Function& fn, const DominatorTree& dom, const BasicBlock* at) {
    for (const BasicBlock* bb : fn.blocks()) {
        if (bb && bb != fn.entry_block() && dom.is_reachable(bb) && entered_from_outside(fn, bb) &&
            dom.dominates(at, bb)) {
            return true;
        }
    }
    return false;
}

size_t scalar_replace_tagged_allocations(Function& fn, const SroaOptions& options) {
    const Module* mod = fn.parent();
    if (!mod || !mod->tagged_allocator()) return 0;
    size_t replaced = 0;
    // One object a round: each changes the function the next is found in.
    for (size_t round = 0; round < 256; ++round) {
        TaggedAllocations allocs(fn);
        if (allocs.non_escaping() == 0) break;
        fn.rebuild_cfg_predecessors();
        DominatorTree dom(fn);
        const TaggedAllocSite* pick = nullptr;
        std::map<int64_t, Word> words;
        for (const TaggedAllocSite& s : allocs.sites()) {
            if (s.escapes() || s.bytes < 0 || static_cast<uint64_t>(s.bytes / 8) > options.max_fields_per_object * 2) continue;
            // Everything it touches is renamed down the dominator tree, and
            // a word may need a parameter wherever paths from the
            // allocation meet: never a block entered from outside the CFG
            // (a landing pad, a resume target), which no edge passes values.
            if (outside_after(fn, dom, s.call->parent())) continue;
            bool reachable = dom.is_reachable(s.call->parent());
            for (const TaggedUse& u : s.uses) reachable = reachable && dom.is_reachable(u.inst->parent());
            words.clear();
            if (reachable && plan_words(s, words)) {
                pick = &s;
                break;
            }
        }
        if (!pick) break;
        std::unordered_map<const BasicBlock*, std::vector<BasicBlock*>> df;
        for (BasicBlock* bb : fn.blocks()) {
            if (!bb || bb->predecessors().size() < 2 || !dom.is_reachable(bb)) continue;
            const BasicBlock* idom = dom.immediate_dominator(bb);
            for (BasicBlock* pred : bb->predecessors()) {
                const BasicBlock* runner = pred;
                while (runner && runner != idom && dom.is_reachable(runner)) {
                    auto& list = df[runner];
                    if (std::find(list.begin(), list.end(), bb) == list.end()) list.push_back(bb);
                    runner = dom.immediate_dominator(runner);
                }
            }
        }
        TaggedReplacer(fn, *pick, std::move(words), dom, df, options.stats).run();
        fn.rebuild_cfg_predecessors();
        ++replaced;
    }
    return replaced;
}

} // namespace brass
