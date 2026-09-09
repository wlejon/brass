#include <brass/mir/branch_probability.hpp>
#include <brass/mir/function.hpp>
#include <brass/mir/block.hpp>
#include <brass/mir/instruction.hpp>
#include <algorithm>
#include <queue>
#include <vector>

namespace brass::mir {

uint64_t BlockFrequencyInfo::get_block_count(const BasicBlock* bb) const {
    if (!bb) return 0;
    auto it = block_counts_.find(bb);
    if (it != block_counts_.end()) {
        return it->second;
    }
    return 0;
}

void BlockFrequencyInfo::set_block_count(const BasicBlock* bb, uint64_t count) {
    if (bb) {
        block_counts_[bb] = count;
    }
}

double BlockFrequencyInfo::get_block_frequency(const BasicBlock* bb) const {
    if (!bb || entry_count_ == 0) return 0.0;
    uint64_t count = get_block_count(bb);
    return static_cast<double>(count) / static_cast<double>(entry_count_);
}

bool BlockFrequencyInfo::is_hot_block(const BasicBlock* bb, double threshold) const {
    return get_block_frequency(bb) >= threshold;
}

bool BlockFrequencyInfo::is_cold_block(const BasicBlock* bb, double threshold) const {
    return get_block_frequency(bb) <= threshold;
}

double BranchProbabilityInfo::get_edge_probability(const BasicBlock* src, const BasicBlock* dst) const {
    EdgeKey key{src, dst};
    auto it = edges_.find(key);
    if (it != edges_.end()) {
        return it->second.probability;
    }
    return 0.0;
}

uint64_t BranchProbabilityInfo::get_edge_count(const BasicBlock* src, const BasicBlock* dst) const {
    EdgeKey key{src, dst};
    auto it = edges_.find(key);
    if (it != edges_.end()) {
        return it->second.count;
    }
    return 0;
}

void BranchProbabilityInfo::set_edge_info(const BasicBlock* src, const BasicBlock* dst, uint64_t count, double prob) {
    EdgeKey key{src, dst};
    edges_[key] = EdgeData{count, prob};
}

namespace {

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

struct Edge {
    BasicBlock* src_bb = nullptr;
    BasicBlock* dst_bb = nullptr;
    uint32_t u = 0;
    uint32_t v = 0;
    int weight = 0;
    bool is_tree = false;
    bool is_solved = false;
    int64_t flow = 0;
};

} // namespace

BranchProbabilityAnalysis::BranchProbabilityAnalysis(const Function& fn, const pgo::FunctionProfile& prof) {
    solve_flows(fn, prof);
}

void BranchProbabilityAnalysis::solve_flows(const Function& fn, const pgo::FunctionProfile& prof) {
    bfi_.set_entry_count(prof.entry_count);

    if (fn.blocks().empty() || !fn.entry_block()) {
        return;
    }

    const size_t num_blocks = fn.blocks().size();
    const uint32_t exit_node = static_cast<uint32_t>(num_blocks);
    const uint32_t total_nodes = exit_node + 1;

    std::unordered_map<const BasicBlock*, uint32_t> block_to_idx;
    for (size_t i = 0; i < num_blocks; ++i) {
        block_to_idx[fn.blocks()[i]] = static_cast<uint32_t>(i);
    }

    std::vector<Edge> all_edges;

    for (size_t i = 0; i < num_blocks; ++i) {
        BasicBlock* u_bb = fn.blocks()[i];
        uint32_t u = static_cast<uint32_t>(i);
        auto succs = u_bb->successors();

        if (succs.empty()) {
            Edge e;
            e.src_bb = u_bb;
            e.dst_bb = nullptr;
            e.u = u;
            e.v = exit_node;
            e.weight = 20;
            all_edges.push_back(e);
        } else {
            Instruction* term = u_bb->terminator();
            bool has_branch = (term && term->opcode() == Opcode::br_if);
            for (BasicBlock* v_bb : succs) {
                if (!v_bb) continue;
                auto it = block_to_idx.find(v_bb);
                if (it == block_to_idx.end()) continue;
                uint32_t v = it->second;

                Edge e;
                e.src_bb = u_bb;
                e.dst_bb = v_bb;
                e.u = u;
                e.v = v;

                if (v <= u) {
                    e.weight = 10; // back-edge
                } else if (has_branch) {
                    if (term->true_target().block == v_bb) {
                        e.weight = 50;
                    } else {
                        e.weight = 100;
                    }
                } else {
                    e.weight = 100;
                }
                all_edges.push_back(e);
            }
        }
    }

    // Sort edges identically to instrumentation Kruskal pass
    std::sort(all_edges.begin(), all_edges.end(), [](const Edge& a, const Edge& b) {
        return a.weight > b.weight;
    });

    DisjointSet dsu(total_nodes);
    std::vector<size_t> tree_edge_indices;
    std::vector<size_t> chord_edge_indices;

    for (size_t i = 0; i < all_edges.size(); ++i) {
        if (dsu.unite(all_edges[i].u, all_edges[i].v)) {
            all_edges[i].is_tree = true;
            tree_edge_indices.push_back(i);
        } else {
            all_edges[i].is_tree = false;
            chord_edge_indices.push_back(i);
        }
    }

    // Ensure exit_node is connected if needed
    for (uint32_t i = 0; i < num_blocks; ++i) {
        if (dsu.unite(i, exit_node)) {
            Edge e;
            e.src_bb = fn.blocks()[i];
            e.dst_bb = nullptr;
            e.u = i;
            e.v = exit_node;
            e.weight = 0;
            e.is_tree = true;
            tree_edge_indices.push_back(all_edges.size());
            all_edges.push_back(e);
        }
    }

    // Assign known flows to chord edges from FunctionProfile
    for (size_t k = 0; k < chord_edge_indices.size(); ++k) {
        size_t e_idx = chord_edge_indices[k];
        uint64_t cnt = 0;
        if (k < prof.edge_counters.size()) {
            cnt = prof.edge_counters[k];
        }
        all_edges[e_idx].flow = static_cast<int64_t>(cnt);
        all_edges[e_idx].is_solved = true;
    }

    // Circulation edge: (Exit -> Entry)
    Edge circ_edge;
    circ_edge.src_bb = nullptr; // Exit
    circ_edge.dst_bb = fn.entry_block();
    circ_edge.u = exit_node;
    circ_edge.v = 0; // entry block
    circ_edge.weight = 1000;
    circ_edge.is_tree = false;
    circ_edge.is_solved = true;
    circ_edge.flow = static_cast<int64_t>(prof.entry_count);
    all_edges.push_back(circ_edge);

    // Build incident lists for each node
    std::vector<std::vector<size_t>> in_edges(total_nodes);
    std::vector<std::vector<size_t>> out_edges(total_nodes);

    for (size_t i = 0; i < all_edges.size(); ++i) {
        out_edges[all_edges[i].u].push_back(i);
        in_edges[all_edges[i].v].push_back(i);
    }

    // Solve Spanning Tree Edge Flows via Kirchhoff's Law (Tree Leaf Peeling)
    std::vector<uint32_t> unsolved_tree_degree(total_nodes, 0);
    for (size_t e_idx : tree_edge_indices) {
        if (!all_edges[e_idx].is_solved) {
            unsolved_tree_degree[all_edges[e_idx].u]++;
            unsolved_tree_degree[all_edges[e_idx].v]++;
        }
    }

    std::queue<uint32_t> leaves;
    for (uint32_t node = 0; node < total_nodes; ++node) {
        if (unsolved_tree_degree[node] == 1) {
            leaves.push(node);
        }
    }

    while (!leaves.empty()) {
        uint32_t node = leaves.front();
        leaves.pop();

        if (unsolved_tree_degree[node] != 1) continue;

        // Find the single unsolved incident tree edge
        size_t target_edge_idx = SIZE_MAX;
        bool is_incoming = false;

        for (size_t e_idx : in_edges[node]) {
            if (!all_edges[e_idx].is_solved) {
                target_edge_idx = e_idx;
                is_incoming = true;
                break;
            }
        }
        if (target_edge_idx == SIZE_MAX) {
            for (size_t e_idx : out_edges[node]) {
                if (!all_edges[e_idx].is_solved) {
                    target_edge_idx = e_idx;
                    is_incoming = false;
                    break;
                }
            }
        }

        if (target_edge_idx == SIZE_MAX) continue;

        // Calculate flow conservation: sum(in) == sum(out)
        int64_t other_in_flow = 0;
        for (size_t e_idx : in_edges[node]) {
            if (e_idx != target_edge_idx && all_edges[e_idx].is_solved) {
                other_in_flow += all_edges[e_idx].flow;
            }
        }

        int64_t other_out_flow = 0;
        for (size_t e_idx : out_edges[node]) {
            if (e_idx != target_edge_idx && all_edges[e_idx].is_solved) {
                other_out_flow += all_edges[e_idx].flow;
            }
        }

        int64_t solved_flow = 0;
        if (is_incoming) {
            // target_edge + other_in == other_out => target_edge = other_out - other_in
            solved_flow = other_out_flow - other_in_flow;
        } else {
            // other_in == target_edge + other_out => target_edge = other_in - other_out
            solved_flow = other_in_flow - other_out_flow;
        }
        if (solved_flow < 0) solved_flow = 0;

        all_edges[target_edge_idx].flow = solved_flow;
        all_edges[target_edge_idx].is_solved = true;

        unsolved_tree_degree[all_edges[target_edge_idx].u]--;
        unsolved_tree_degree[all_edges[target_edge_idx].v]--;

        uint32_t neighbor = (node == all_edges[target_edge_idx].u) ?
                             all_edges[target_edge_idx].v : all_edges[target_edge_idx].u;

        if (unsolved_tree_degree[neighbor] == 1) {
            leaves.push(neighbor);
        }
    }

    // Set any remaining unsolved edges to 0
    for (auto& edge : all_edges) {
        if (!edge.is_solved) {
            edge.flow = 0;
            edge.is_solved = true;
        }
    }

    // Compute Block Execution Counts
    for (size_t i = 0; i < num_blocks; ++i) {
        BasicBlock* bb = fn.blocks()[i];
        uint32_t u = static_cast<uint32_t>(i);

        int64_t block_flow = 0;
        for (size_t e_idx : out_edges[u]) {
            block_flow += all_edges[e_idx].flow;
        }
        if (block_flow < 0) block_flow = 0;

        bfi_.set_block_count(bb, static_cast<uint64_t>(block_flow));
    }

    // Compute Edge Probabilities & Counts
    for (size_t i = 0; i < num_blocks; ++i) {
        BasicBlock* u_bb = fn.blocks()[i];
        uint32_t u = static_cast<uint32_t>(i);
        uint64_t u_count = bfi_.get_block_count(u_bb);
        auto succs = u_bb->successors();

        for (BasicBlock* v_bb : succs) {
            if (!v_bb) continue;
            auto it = block_to_idx.find(v_bb);
            if (it == block_to_idx.end()) continue;
            uint32_t v = it->second;

            uint64_t edge_count = 0;
            for (size_t e_idx : out_edges[u]) {
                if (all_edges[e_idx].v == v) {
                    edge_count += static_cast<uint64_t>(all_edges[e_idx].flow);
                }
            }

            double prob = 0.0;
            if (u_count > 0) {
                prob = static_cast<double>(edge_count) / static_cast<double>(u_count);
            } else if (!succs.empty()) {
                prob = 1.0 / static_cast<double>(succs.size());
            }
            if (prob > 1.0) prob = 1.0;

            bpi_.set_edge_info(u_bb, v_bb, edge_count, prob);
        }
    }
}

} // namespace brass::mir
