#include <brass/pgo/instrument.hpp>
#include <brass/mir/builder.hpp>
#include <brass/mir/function.hpp>
#include <brass/mir/block.hpp>
#include <brass/mir/instruction.hpp>
#include <algorithm>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace brass::pgo {

namespace {

static std::vector<uint64_t> g_pgo_counters;

struct DisjointSet {
    std::vector<uint32_t> parent;
    std::vector<uint32_t> rank;

    explicit DisjointSet(size_t n) : parent(n), rank(n, 0) {
        for (uint32_t i = 0; i < n; ++i) {
            parent[i] = i;
        }
    }

    uint32_t find(uint32_t x) {
        if (parent[x] != x) {
            parent[x] = find(parent[x]);
        }
        return parent[x];
    }

    bool unite(uint32_t x, uint32_t y) {
        uint32_t rx = find(x);
        uint32_t ry = find(y);
        if (rx == ry) return false;
        if (rank[rx] < rank[ry]) {
            parent[rx] = ry;
        } else if (rank[rx] > rank[ry]) {
            parent[ry] = rx;
        } else {
            parent[ry] = rx;
            rank[rx]++;
        }
        return true;
    }
};

struct EdgeCandidate {
    BasicBlock* src = nullptr;
    BasicBlock* dst = nullptr; // nullptr represents Exit node
    uint32_t u = 0;
    uint32_t v = 0;
    int weight = 0;
    bool is_exit_edge = false;
};

} // namespace

void brass_pgo_init_counters(size_t num_counters) {
    g_pgo_counters.assign(num_counters, 0);
}

void brass_pgo_inc_counter(uint32_t counter_index) {
    if (counter_index < g_pgo_counters.size()) {
        g_pgo_counters[counter_index]++;
    }
}

const uint64_t* brass_pgo_get_counters(size_t* num_counters) {
    if (num_counters) *num_counters = g_pgo_counters.size();
    return g_pgo_counters.data();
}

void brass_pgo_reset_counters() {
    std::fill(g_pgo_counters.begin(), g_pgo_counters.end(), 0);
}

PgoInstrumentResult instrument_function(Function& fn, uint32_t counter_base_index) {
    PgoInstrumentResult result;
    result.function = &fn;
    result.counter_base_index = counter_base_index;

    if (fn.blocks().empty() || !fn.entry_block()) {
        return result;
    }

    fn.rebuild_cfg_predecessors();

    const size_t num_blocks = fn.blocks().size();
    const uint32_t exit_node = static_cast<uint32_t>(num_blocks);
    const uint32_t total_nodes = exit_node + 1;

    std::unordered_map<const BasicBlock*, uint32_t> block_to_idx;
    for (size_t i = 0; i < num_blocks; ++i) {
        block_to_idx[fn.blocks()[i]] = static_cast<uint32_t>(i);
    }

    std::vector<EdgeCandidate> candidates;

    for (size_t i = 0; i < num_blocks; ++i) {
        BasicBlock* u_bb = fn.blocks()[i];
        uint32_t u = static_cast<uint32_t>(i);
        auto succs = u_bb->successors();

        if (succs.empty()) {
            EdgeCandidate ec;
            ec.src = u_bb;
            ec.dst = nullptr;
            ec.u = u;
            ec.v = exit_node;
            ec.weight = 20;
            ec.is_exit_edge = true;
            candidates.push_back(ec);
        } else {
            Instruction* term = u_bb->terminator();
            bool has_branch = (term && term->opcode() == Opcode::br_if);
            for (BasicBlock* v_bb : succs) {
                if (!v_bb) continue;
                auto it = block_to_idx.find(v_bb);
                if (it == block_to_idx.end()) continue;
                uint32_t v = it->second;

                EdgeCandidate ec;
                ec.src = u_bb;
                ec.dst = v_bb;
                ec.u = u;
                ec.v = v;

                if (v <= u) {
                    ec.weight = 10; // back-edge
                } else if (has_branch) {
                    if (term->true_target().block == v_bb) {
                        ec.weight = 50; // taken branch
                    } else {
                        ec.weight = 100; // fall-through branch
                    }
                } else {
                    ec.weight = 100; // unconditional jump
                }
                candidates.push_back(ec);
            }
        }
    }

    // Kruskal's algorithm to compute spanning tree
    // Higher weight edges are preferred in the spanning tree, leaving back-edges as chords.
    std::sort(candidates.begin(), candidates.end(), [](const EdgeCandidate& a, const EdgeCandidate& b) {
        return a.weight > b.weight;
    });

    DisjointSet dsu(total_nodes);
    std::vector<EdgeCandidate> tree_candidates;
    std::vector<EdgeCandidate> chord_candidates;

    for (const auto& ec : candidates) {
        if (dsu.unite(ec.u, ec.v)) {
            tree_candidates.push_back(ec);
            result.spanning_tree_edges.push_back(SpanningTreeEdge{ec.src, ec.dst});
        } else {
            chord_candidates.push_back(ec);
        }
    }

    // Ensure exit_node is connected if any disconnected parts exist
    for (uint32_t i = 0; i < num_blocks; ++i) {
        if (dsu.unite(i, exit_node)) {
            result.spanning_tree_edges.push_back(SpanningTreeEdge{fn.blocks()[i], nullptr});
        }
    }

    // Circulation edge (Exit -> Entry) is chord 0
    ChordEdge entry_chord;
    entry_chord.src = nullptr; // Exit
    entry_chord.dst = fn.entry_block();
    entry_chord.counter_index = counter_base_index;
    result.chords.push_back(entry_chord);
    result.entry_counter_index = counter_base_index;

    uint32_t next_counter = counter_base_index + 1;
    for (const auto& ec : chord_candidates) {
        ChordEdge c;
        c.src = ec.src;
        c.dst = ec.dst;
        c.counter_index = next_counter++;
        result.chords.push_back(c);
    }
    result.total_counters = next_counter - counter_base_index;

    // Instrument Function Entry block
    {
        BasicBlock* entry_bb = fn.entry_block();
        Builder b(fn);
        if (entry_bb->head()) {
            b.position_before(entry_bb->head());
        } else {
            b.position_at_end(entry_bb);
        }
        b.build_call("brass_pgo_inc", Type::void_type(),
                     {b.build_iconst_i32(static_cast<int32_t>(result.entry_counter_index))});
    }

    // Instrument CFG Chords
    for (size_t i = 1; i < result.chords.size(); ++i) {
        ChordEdge& c = result.chords[i];
        BasicBlock* u = c.src;
        BasicBlock* v = c.dst;
        if (!u) continue;

        if (v == nullptr) {
            // Exit chord: insert before terminator in u
            Instruction* term = u->terminator();
            Builder b(fn);
            if (term) {
                b.position_before(term);
            } else {
                b.position_at_end(u);
            }
            b.build_call("brass_pgo_inc", Type::void_type(),
                         {b.build_iconst_i32(static_cast<int32_t>(c.counter_index))});
            continue;
        }

        Instruction* term = u->terminator();
        if (!term) continue;

        // Check if edge is critical (u has multiple successors)
        if (u->successors().size() > 1) {
            c.is_critical = true;
            Builder b_create(fn);
            BasicBlock* chord_bb = b_create.create_block(std::string(u->name()) + "_pgo_chord");
            fn.append_block(chord_bb);
            c.split_block = chord_bb;

            if (term->opcode() == Opcode::br_if) {
                if (term->true_target().block == v) {
                    auto args = term->true_target().args;
                    term->set_true_target(BranchTarget(chord_bb));
                    Builder b(fn);
                    b.position_at_end(chord_bb);
                    b.build_call("brass_pgo_inc", Type::void_type(),
                                 {b.build_iconst_i32(static_cast<int32_t>(c.counter_index))});
                    b.build_br(v, Span<Value* const>(args.data(), args.size()));
                } else if (term->false_target().block == v) {
                    auto args = term->false_target().args;
                    term->set_false_target(BranchTarget(chord_bb));
                    Builder b(fn);
                    b.position_at_end(chord_bb);
                    b.build_call("brass_pgo_inc", Type::void_type(),
                                 {b.build_iconst_i32(static_cast<int32_t>(c.counter_index))});
                    b.build_br(v, Span<Value* const>(args.data(), args.size()));
                }
            } else if (term->opcode() == Opcode::switch_) {
                bool redirected = false;
                for (auto& sc : term->switch_cases()) {
                    if (sc.target.block == v) {
                        auto args = sc.target.args;
                        sc.target = BranchTarget(chord_bb);
                        Builder b(fn);
                        b.position_at_end(chord_bb);
                        b.build_call("brass_pgo_inc", Type::void_type(),
                                     {b.build_iconst_i32(static_cast<int32_t>(c.counter_index))});
                        b.build_br(v, Span<Value* const>(args.data(), args.size()));
                        redirected = true;
                        break;
                    }
                }
                if (!redirected && term->default_target().block == v) {
                    auto args = term->default_target().args;
                    term->set_default_target(BranchTarget(chord_bb));
                    Builder b(fn);
                    b.position_at_end(chord_bb);
                    b.build_call("brass_pgo_inc", Type::void_type(),
                                 {b.build_iconst_i32(static_cast<int32_t>(c.counter_index))});
                    b.build_br(v, Span<Value* const>(args.data(), args.size()));
                }
            } else if (term->opcode() == Opcode::br) {
                auto args = term->branch_target().args;
                term->set_branch_target(BranchTarget(chord_bb));
                Builder b(fn);
                b.position_at_end(chord_bb);
                b.build_call("brass_pgo_inc", Type::void_type(),
                             {b.build_iconst_i32(static_cast<int32_t>(c.counter_index))});
                b.build_br(v, Span<Value* const>(args.data(), args.size()));
            }
        } else {
            // Single successor edge: insert before terminator in u
            Builder b(fn);
            b.position_before(term);
            b.build_call("brass_pgo_inc", Type::void_type(),
                         {b.build_iconst_i32(static_cast<int32_t>(c.counter_index))});
        }
    }

    fn.rebuild_cfg_predecessors();

    // Populate metadata
    PgoFunctionMetadata fn_meta;
    fn_meta.name = std::string(fn.name());
    fn_meta.counter_base_index = result.counter_base_index;
    fn_meta.entry_counter_index = result.entry_counter_index;
    fn_meta.num_edge_counters = (result.total_counters > 0) ? (result.total_counters - 1) : 0;
    for (size_t i = 1; i < result.chords.size(); ++i) {
        const auto& c = result.chords[i];
        uint32_t u_id = c.src ? c.src->id() : UINT32_MAX;
        uint32_t v_id = c.dst ? c.dst->id() : UINT32_MAX;
        fn_meta.chord_block_ids.push_back({u_id, v_id});
    }
    result.metadata.functions.push_back(fn_meta);

    return result;
}

PgoInstrumentResult instrument_module(Module& mod) {
    PgoInstrumentResult mod_result;
    mod_result.metadata.module_name = std::string(mod.name());

    uint32_t next_counter = 0;
    for (Function* fn : mod.functions()) {
        if (!fn) continue;
        PgoInstrumentResult fn_res = instrument_function(*fn, next_counter);
        next_counter += fn_res.total_counters;
        if (!fn_res.metadata.functions.empty()) {
            mod_result.metadata.functions.push_back(fn_res.metadata.functions[0]);
        }
        for (auto& c : fn_res.chords) {
            mod_result.chords.push_back(std::move(c));
        }
        for (auto& st : fn_res.spanning_tree_edges) {
            mod_result.spanning_tree_edges.push_back(std::move(st));
        }
    }

    mod_result.total_counters = next_counter;
    mod_result.metadata.total_counters = next_counter;
    brass_pgo_init_counters(next_counter);

    return mod_result;
}

bool brass_pgo_dump(const char* path, const uint64_t* counters, size_t num_counters, const PgoMetadata& meta) {
    if (!path || !counters) return false;

    ProfileData prof;
    prof.set_module_name(meta.module_name);
    prof.set_module_hash(meta.module_hash);

    for (const auto& fn_meta : meta.functions) {
        FunctionProfile fp;
        fp.name = fn_meta.name;
        if (fn_meta.entry_counter_index < num_counters) {
            fp.entry_count = counters[fn_meta.entry_counter_index];
        }
        fp.edge_counters.resize(fn_meta.num_edge_counters);
        for (uint32_t i = 0; i < fn_meta.num_edge_counters; ++i) {
            uint32_t c_idx = fn_meta.counter_base_index + 1 + i;
            if (c_idx < num_counters) {
                fp.edge_counters[i] = counters[c_idx];
            }
        }
        prof.add_function(std::move(fp));
    }

    return prof.write_to_file(path);
}

} // namespace brass::pgo

extern "C" {
void brass_pgo_inc(uint32_t counter_index) {
    brass::pgo::brass_pgo_inc_counter(counter_index);
}
}
