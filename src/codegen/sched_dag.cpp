#include <brass/codegen/sched_dag.hpp>
#include <brass/codegen/lir_flags.hpp>
#include <algorithm>
#include <unordered_map>
#include <unordered_set>

namespace brass::codegen {

std::string_view to_string(EdgeKind kind) noexcept {
    switch (kind) {
        case EdgeKind::RAW:     return "RAW";
        case EdgeKind::WAR:     return "WAR";
        case EdgeKind::WAW:     return "WAW";
        case EdgeKind::MemRAW:  return "MemRAW";
        case EdgeKind::MemWAR:  return "MemWAR";
        case EdgeKind::MemWAW:  return "MemWAW";
        case EdgeKind::Barrier: return "Barrier";
    }
    return "Unknown";
}

bool is_scheduling_barrier(const LirInst& inst) noexcept {
    if (inst.is_call() || inst.is_terminator()) {
        return true;
    }
    switch (inst.opcode) {
        case LirOpcode::Safepoint:
        case LirOpcode::GuardExit:
        case LirOpcode::WriteBarrier:
        case LirOpcode::ParallelCopy:
            return true;
        default:
            return false;
    }
}

static bool operand_reads_memory(const LirOperand& op) noexcept {
    return op.is_mem() || op.is_spill_slot();
}

static bool operand_writes_memory(const LirOperand& op) noexcept {
    return op.is_mem() || op.is_spill_slot();
}

static bool instruction_reads_memory(const LirInst& inst) noexcept {
    for (const auto& u : inst.uses) {
        if (operand_reads_memory(u)) return true;
    }
    return false;
}

static bool instruction_writes_memory(const LirInst& inst) noexcept {
    for (const auto& d : inst.defs) {
        if (operand_writes_memory(d)) return true;
    }
    return false;
}

uint32_t get_instruction_latency(const LirInst& inst) {
    bool has_mem_load = instruction_reads_memory(inst);

    switch (inst.opcode) {
        case LirOpcode::KeepAlive:
            return 0;
        case LirOpcode::Nop:
            return 1;
        case LirOpcode::Mov:
        case LirOpcode::Mov32:
        case LirOpcode::Movabs:
        case LirOpcode::Movsx8:
        case LirOpcode::Movsx16:
        case LirOpcode::Movsxd:
        case LirOpcode::Movzx8:
        case LirOpcode::Movzx16:
        case LirOpcode::Lea:
            return has_mem_load ? 4 : 1;

        case LirOpcode::Add:
        case LirOpcode::Add32:
        case LirOpcode::Sub:
        case LirOpcode::Sub32:
        case LirOpcode::And:
        case LirOpcode::And32:
        case LirOpcode::Or:
        case LirOpcode::Or32:
        case LirOpcode::Xor:
        case LirOpcode::Xor32:
        case LirOpcode::Not:
        case LirOpcode::Not32:
        case LirOpcode::Neg:
        case LirOpcode::Neg32:
        case LirOpcode::Shl:
        case LirOpcode::Shl32:
        case LirOpcode::Shr:
        case LirOpcode::Shr32:
        case LirOpcode::Sar:
        case LirOpcode::Sar32:
        case LirOpcode::Popcnt:
        case LirOpcode::Popcnt32:
        case LirOpcode::Lzcnt:
        case LirOpcode::Lzcnt32:
        case LirOpcode::Tzcnt:
        case LirOpcode::Tzcnt32:
        case LirOpcode::Bsr:
        case LirOpcode::Bsr32:
        case LirOpcode::Bsf:
        case LirOpcode::Bsf32:
        case LirOpcode::Cmp:
        case LirOpcode::Cmp32:
        case LirOpcode::Test:
        case LirOpcode::Test32:
        case LirOpcode::Setcc:
            return has_mem_load ? 5 : 1;

        case LirOpcode::Cmovcc:
            return has_mem_load ? 5 : 2;

        case LirOpcode::Imul:
        case LirOpcode::Imul32:
        case LirOpcode::Smulh:
        case LirOpcode::Umulh:
            return has_mem_load ? 6 : 3;

        case LirOpcode::Idiv:
        case LirOpcode::Idiv32:
        case LirOpcode::Div:
        case LirOpcode::Div32:
            return has_mem_load ? 16 : 13;

        case LirOpcode::Cdq:
        case LirOpcode::Cqo:
            return 1;

        case LirOpcode::Movsd:
        case LirOpcode::Movss:
            return has_mem_load ? 4 : 1;
        case LirOpcode::Movq_gx:
        case LirOpcode::Movq_xg:
            return 2;
        case LirOpcode::Addsd:
        case LirOpcode::Addss:
        case LirOpcode::Subsd:
        case LirOpcode::Subss:
            return has_mem_load ? 6 : 3;
        case LirOpcode::Mulsd:
        case LirOpcode::Mulss:
            return has_mem_load ? 7 : 4;
        case LirOpcode::Divsd:
        case LirOpcode::Divss:
            return has_mem_load ? 16 : 13;
        case LirOpcode::Sqrtsd:
        case LirOpcode::Sqrtss:
            return has_mem_load ? 17 : 14;
        case LirOpcode::Ucomisd:
        case LirOpcode::Ucomiss:
        case LirOpcode::Xorpd:
            return has_mem_load ? 5 : 1;
        case LirOpcode::Cvtsi2sd:
        case LirOpcode::Cvtsi2sd32:
        case LirOpcode::Cvttsd2si:
        case LirOpcode::Cvttsd2si32:
            return has_mem_load ? 6 : 3;

        case LirOpcode::Movaps:
        case LirOpcode::Movups:
            return has_mem_load ? 5 : 1;
        case LirOpcode::Movd_xg:
        case LirOpcode::Movd_gx:
            return 2;
        case LirOpcode::Addps:
        case LirOpcode::Subps:
        case LirOpcode::Addpd:
        case LirOpcode::Subpd:
        case LirOpcode::Minps:
        case LirOpcode::Maxps:
        case LirOpcode::Minpd:
        case LirOpcode::Maxpd:
            return has_mem_load ? 6 : 3;
        case LirOpcode::Mulps:
        case LirOpcode::Mulpd:
        case LirOpcode::Pmulld:
            return has_mem_load ? 7 : 4;
        case LirOpcode::Divps:
        case LirOpcode::Divpd:
            return has_mem_load ? 16 : 13;
        case LirOpcode::Sqrtps:
        case LirOpcode::Sqrtpd:
        case LirOpcode::Vsqrtps:
        case LirOpcode::Vsqrtpd:
            return has_mem_load ? 17 : 14;
        case LirOpcode::Paddd:
        case LirOpcode::Psubd:
        case LirOpcode::Pminsd:
        case LirOpcode::Pmaxsd:
        case LirOpcode::Paddq:
        case LirOpcode::Psubq:
        case LirOpcode::Pand:
        case LirOpcode::Por:
        case LirOpcode::Pxor:
        case LirOpcode::Pandn:
        case LirOpcode::Pcmpeqd:
        case LirOpcode::Xorps:
            return has_mem_load ? 5 : 1;
        case LirOpcode::Pslld:
        case LirOpcode::Psllq:
        case LirOpcode::Shufps:
        case LirOpcode::Shufpd:
        case LirOpcode::Pshufd:
        case LirOpcode::Movddup:
        case LirOpcode::Pinsrd:
        case LirOpcode::Pextrd:
        case LirOpcode::Pinsrq:
        case LirOpcode::Pextrq:
        case LirOpcode::Insertps:
        case LirOpcode::Extractps:
            return has_mem_load ? 6 : 2;

        case LirOpcode::Call:
        case LirOpcode::CallIndirect:
            return 5;
        case LirOpcode::Push:
        case LirOpcode::Pop:
            return 2;
        case LirOpcode::Jmp:
        case LirOpcode::Jcc:
        case LirOpcode::Ret:
        case LirOpcode::Safepoint:
        case LirOpcode::GuardExit:
        case LirOpcode::Trap:
        case LirOpcode::ParallelCopy:
        case LirOpcode::WriteBarrier:
            return 1;
        default:
            return 1;
    }
}

struct MemoryRef {
    bool is_spill_slot = false;
    int32_t spill_slot_idx = -1;
    bool is_rbp_spill = false;
    int32_t rbp_offset = 0;
    bool is_heap_or_other = false;
    bool has_base = false;
    VReg base_vreg;
    PReg base_preg;
    int32_t disp = 0;
    bool has_index = false;
    uint8_t size = 8;
};

static std::vector<MemoryRef> get_memory_refs(const LirInst& inst) {
    std::vector<MemoryRef> refs;

    auto check_op = [&](const LirOperand& op) {
        if (op.is_spill_slot()) {
            MemoryRef r;
            r.is_spill_slot = true;
            r.spill_slot_idx = op.spill_slot;
            r.size = op.size;
            refs.push_back(r);
        } else if (op.is_mem()) {
            MemoryRef r;
            r.size = op.size;
            r.disp = op.mem_val.disp;
            r.has_index = op.mem_val.has_index();
            if (op.mem_val.base_preg == PReg::gpr(x64::GPR::RBP)) {
                r.is_rbp_spill = true;
                r.rbp_offset = op.mem_val.disp;
            } else {
                r.is_heap_or_other = true;
                r.has_base = op.mem_val.has_base();
                r.base_vreg = op.mem_val.base_vreg;
                r.base_preg = op.mem_val.base_preg;
            }
            refs.push_back(r);
        }
    };

    for (const auto& d : inst.defs) check_op(d);
    for (const auto& u : inst.uses) check_op(u);

    return refs;
}

static bool may_memory_alias(const std::vector<MemoryRef>& refs_a, const std::vector<MemoryRef>& refs_b) {
    if (refs_a.empty() || refs_b.empty()) {
        return false;
    }

    for (const auto& ra : refs_a) {
        for (const auto& rb : refs_b) {
            // Case 1: Both are spill slots
            if (ra.is_spill_slot && rb.is_spill_slot) {
                if (ra.spill_slot_idx == rb.spill_slot_idx) return true;
                continue;
            }

            // Case 2: Spill slot vs Heap
            if ((ra.is_spill_slot || ra.is_rbp_spill) && rb.is_heap_or_other) {
                continue; // Stack spill slot never aliases heap pointer
            }
            if ((rb.is_spill_slot || rb.is_rbp_spill) && ra.is_heap_or_other) {
                continue;
            }

            // Case 3: Both are RBP spills
            if (ra.is_rbp_spill && rb.is_rbp_spill) {
                int32_t a_start = ra.rbp_offset;
                int32_t a_end = a_start + static_cast<int32_t>(ra.size);
                int32_t b_start = rb.rbp_offset;
                int32_t b_end = b_start + static_cast<int32_t>(rb.size);
                if (a_end <= b_start || b_end <= a_start) {
                    continue; // Non-overlapping stack offsets
                }
                return true;
            }

            // Case 4: Both are heap accesses with same base and no index
            if (ra.is_heap_or_other && rb.is_heap_or_other) {
                bool same_base = false;
                if (ra.has_base && rb.has_base) {
                    if (ra.base_vreg.is_valid() && ra.base_vreg == rb.base_vreg) {
                        same_base = true;
                    } else if (ra.base_preg.is_valid() && ra.base_preg == rb.base_preg) {
                        same_base = true;
                    }
                }
                if (same_base && !ra.has_index && !rb.has_index) {
                    int32_t a_start = ra.disp;
                    int32_t a_end = a_start + static_cast<int32_t>(ra.size);
                    int32_t b_start = rb.disp;
                    int32_t b_end = b_start + static_cast<int32_t>(rb.size);
                    if (a_end <= b_start || b_end <= a_start) {
                        continue; // Non-overlapping displacements from same base
                    }
                    return true;
                }
            }

            // Conservative fallback: assume alias
            return true;
        }
    }

    return false;
}

SchedDAG::SchedDAG(LirBlock& block, Arch arch, bool is_pre_ra)
    : block_(&block), arch_(arch), is_pre_ra_(is_pre_ra) {}

void SchedDAG::build() {
    build_nodes();
    build_register_dependencies();
    build_memory_dependencies();
    build_barrier_dependencies();
    compute_metrics();
}

void SchedDAG::build_nodes() {
    // Nodes left from an earlier build keep their edge lists' storage.
    const size_t n = block_->instructions.size();
    nodes_.resize(n);
    for (size_t i = 0; i < n; ++i) {
        SchedNode& node = nodes_[i];
        node.id = static_cast<uint32_t>(i);
        node.inst = block_->instructions[i].get();
        node.latency = get_instruction_latency(*node.inst);
        node.height = 0;
        node.depth = 0;
        node.is_pre_ra = is_pre_ra_;
        node.is_load = instruction_reads_memory(*node.inst);
        node.is_store = instruction_writes_memory(*node.inst);
        node.is_barrier = is_scheduling_barrier(*node.inst);
        node.preds.clear();
        node.succs.clear();
        node.unscheduled_preds = 0;
    }
}

void SchedDAG::add_edge(uint32_t from, uint32_t to, EdgeKind kind, uint32_t latency) {
    if (from >= nodes_.size() || to >= nodes_.size() || from == to) {
        return;
    }

    // An existing edge is looked for from the end with fewer edges (a
    // barrier's list can hold most of the block).
    auto& succs = nodes_[from].succs;
    auto& preds = nodes_[to].preds;
    SchedEdge* s = nullptr;
    SchedEdge* p = nullptr;
    if (succs.size() <= preds.size()) {
        for (auto& e : succs) {
            if (e.target_node == to) { s = &e; p = &preds[e.mirror]; break; }
        }
    } else {
        for (auto& e : preds) {
            if (e.target_node == from) { p = &e; s = &succs[e.mirror]; break; }
        }
    }
    if (s) {
        s->latency = p->latency = std::max(s->latency, latency);
        if (kind == EdgeKind::RAW) s->kind = p->kind = EdgeKind::RAW;
        return;
    }

    const auto succ_at = static_cast<uint32_t>(succs.size());
    const auto pred_at = static_cast<uint32_t>(preds.size());
    succs.push_back(SchedEdge{to, kind, latency, pred_at});
    preds.push_back(SchedEdge{from, kind, latency, succ_at});

    nodes_[to].unscheduled_preds++;
}

void SchedDAG::build_register_dependencies() {
    size_t n = nodes_.size();

    // Virtual register tracking
    std::unordered_map<uint32_t, uint32_t> last_def_vreg;
    std::unordered_map<uint32_t, std::vector<uint32_t>> last_uses_vreg;

    // Physical register tracking, indexed by (reg_class << 8 | code)
    constexpr uint32_t kNoDef = UINT32_MAX;
    constexpr size_t kPregKeys = 2u << 8;
    std::vector<uint32_t> last_def_preg(kPregKeys, kNoDef);
    // The reads of each register since its last def, newest first, as lists
    // threaded through use_links.
    struct UseLink {
        uint32_t node;
        uint32_t next;
    };
    std::vector<UseLink> use_links;
    std::vector<uint32_t> last_use_preg(kPregKeys, kNoDef);

    std::vector<uint32_t> uses_vregs;
    std::vector<uint32_t> uses_pregs;
    std::vector<uint32_t> defs_vregs;
    std::vector<uint32_t> defs_pregs;

    for (uint32_t j = 0; j < n; ++j) {
        const auto& inst = *nodes_[j].inst;

        // 1. Collect VReg and PReg uses
        uses_vregs.clear();
        uses_pregs.clear();

        for (const auto& u : inst.uses) {
            if (u.is_vreg() && u.vreg_val.is_valid()) {
                uses_vregs.push_back(u.vreg_val.id);
            }
            if (u.is_preg() && u.preg_val.is_valid()) {
                uses_pregs.push_back((static_cast<uint32_t>(u.preg_val.reg_class) << 8) | u.preg_val.code);
            }
            if (u.is_mem()) {
                if (u.mem_val.base_vreg.is_valid()) uses_vregs.push_back(u.mem_val.base_vreg.id);
                if (u.mem_val.index_vreg.is_valid()) uses_vregs.push_back(u.mem_val.index_vreg.id);
                if (u.mem_val.base_preg.is_valid()) {
                    uses_pregs.push_back((static_cast<uint32_t>(u.mem_val.base_preg.reg_class) << 8) | u.mem_val.base_preg.code);
                }
                if (u.mem_val.index_preg.is_valid()) {
                    uses_pregs.push_back((static_cast<uint32_t>(u.mem_val.index_preg.reg_class) << 8) | u.mem_val.index_preg.code);
                }
            }
        }
        for (const auto& d : inst.defs) {
            if (d.is_mem()) {
                if (d.mem_val.base_vreg.is_valid()) uses_vregs.push_back(d.mem_val.base_vreg.id);
                if (d.mem_val.index_vreg.is_valid()) uses_vregs.push_back(d.mem_val.index_vreg.id);
                if (d.mem_val.base_preg.is_valid()) {
                    uses_pregs.push_back((static_cast<uint32_t>(d.mem_val.base_preg.reg_class) << 8) | d.mem_val.base_preg.code);
                }
                if (d.mem_val.index_preg.is_valid()) {
                    uses_pregs.push_back((static_cast<uint32_t>(d.mem_val.index_preg.reg_class) << 8) | d.mem_val.index_preg.code);
                }
            }
        }
        for (const auto& c : inst.use_constraints) {
            if (c.has_fixed_preg && c.fixed_preg.is_valid()) {
                uses_pregs.push_back((static_cast<uint32_t>(c.fixed_preg.reg_class) << 8) | c.fixed_preg.code);
            }
        }

        // Apply VReg uses
        for (uint32_t vid : uses_vregs) {
            auto def_it = last_def_vreg.find(vid);
            if (def_it != last_def_vreg.end()) {
                add_edge(def_it->second, j, EdgeKind::RAW, nodes_[def_it->second].latency);
            }
            last_uses_vreg[vid].push_back(j);
        }

        // Apply PReg uses
        for (uint32_t pkey : uses_pregs) {
            const uint32_t def = last_def_preg[pkey];
            if (def != kNoDef) {
                add_edge(def, j, EdgeKind::RAW, nodes_[def].latency);
            }
            use_links.push_back({j, last_use_preg[pkey]});
            last_use_preg[pkey] = static_cast<uint32_t>(use_links.size() - 1);
        }

        // 2. Collect VReg and PReg defs
        defs_vregs.clear();
        defs_pregs.clear();

        for (const auto& d : inst.defs) {
            if (d.is_vreg() && d.vreg_val.is_valid()) {
                defs_vregs.push_back(d.vreg_val.id);
            }
            if (d.is_preg() && d.preg_val.is_valid()) {
                defs_pregs.push_back((static_cast<uint32_t>(d.preg_val.reg_class) << 8) | d.preg_val.code);
            }
        }
        for (const auto& c : inst.def_constraints) {
            if (c.has_fixed_preg && c.fixed_preg.is_valid()) {
                defs_pregs.push_back((static_cast<uint32_t>(c.fixed_preg.reg_class) << 8) | c.fixed_preg.code);
            }
        }
        for (uint8_t c = 0; c < 32; ++c) {
            if (inst.clobbered_gprs & (1u << c)) {
                defs_pregs.push_back((static_cast<uint32_t>(RegClass::GPR) << 8) | c);
            }
            if (inst.clobbered_xmms & (1u << c)) {
                defs_pregs.push_back((static_cast<uint32_t>(RegClass::XMM) << 8) | c);
            }
        }

        // Apply VReg defs
        for (uint32_t vid : defs_vregs) {
            auto u_it = last_uses_vreg.find(vid);
            if (u_it != last_uses_vreg.end()) {
                for (uint32_t u_node : u_it->second) {
                    if (u_node != j) {
                        add_edge(u_node, j, EdgeKind::WAR, 0);
                    }
                }
                u_it->second.clear();
            }
            auto def_it = last_def_vreg.find(vid);
            if (def_it != last_def_vreg.end()) {
                add_edge(def_it->second, j, EdgeKind::WAW, 1);
            }
            last_def_vreg[vid] = j;
        }

        // Apply PReg defs
        for (uint32_t pkey : defs_pregs) {
            for (uint32_t l = last_use_preg[pkey]; l != kNoDef; l = use_links[l].next) {
                if (use_links[l].node != j) {
                    add_edge(use_links[l].node, j, EdgeKind::WAR, 0);
                }
            }
            last_use_preg[pkey] = kNoDef;
            if (last_def_preg[pkey] != kNoDef) {
                add_edge(last_def_preg[pkey], j, EdgeKind::WAW, 1);
            }
            last_def_preg[pkey] = j;
        }
    }

    // Condition flags tracking
    int32_t last_flag_def = -1;
    std::vector<uint32_t> last_flag_uses;

    for (uint32_t j = 0; j < n; ++j) {
        const auto& inst = *nodes_[j].inst;
        // Every flag writer is ordered, full definition or partial clobber:
        // moving a reader across either changes what it reads.
        if (lir_reads_flags(inst, arch_)) {
            if (last_flag_def >= 0) {
                add_edge(static_cast<uint32_t>(last_flag_def), j, EdgeKind::RAW, 1);
            }
            last_flag_uses.push_back(j);
        }
        if (lir_writes_flags(inst, arch_)) {
            for (uint32_t u_node : last_flag_uses) {
                if (u_node != j) {
                    add_edge(u_node, j, EdgeKind::WAR, 0);
                }
            }
            last_flag_uses.clear();
            if (last_flag_def >= 0) {
                add_edge(static_cast<uint32_t>(last_flag_def), j, EdgeKind::WAW, 1);
            }
            last_flag_def = static_cast<int32_t>(j);
        }
    }
}

void SchedDAG::build_memory_dependencies() {
    std::vector<uint32_t> mem;
    std::vector<std::vector<MemoryRef>> refs(nodes_.size());
    for (uint32_t i = 0; i < nodes_.size(); ++i) {
        if (!nodes_[i].is_load && !nodes_[i].is_store) continue;
        mem.push_back(i);
        refs[i] = get_memory_refs(*nodes_[i].inst);
    }

    for (size_t a = 0; a < mem.size(); ++a) {
        const uint32_t i = mem[a];
        const bool i_reads = nodes_[i].is_load;
        const bool i_writes = nodes_[i].is_store;

        for (size_t b = a + 1; b < mem.size(); ++b) {
            const uint32_t j = mem[b];
            const bool j_reads = nodes_[j].is_load;
            const bool j_writes = nodes_[j].is_store;
            if (!i_writes && !j_writes) continue; // two reads are never ordered

            if (may_memory_alias(refs[i], refs[j])) {
                if (i_writes && j_reads) {
                    // Store -> Load hazard (RAW)
                    add_edge(i, j, EdgeKind::MemRAW, 4);
                } else if (i_reads && j_writes) {
                    // Load -> Store hazard (WAR)
                    add_edge(i, j, EdgeKind::MemWAR, 0);
                } else if (i_writes && j_writes) {
                    // Store -> Store hazard (WAW)
                    add_edge(i, j, EdgeKind::MemWAW, 1);
                }
            }
        }
    }
}

void SchedDAG::build_barrier_dependencies() {
    size_t n = nodes_.size();
    if (n == 0) return;

    // Each rule orders a set of pairs; the edges below are enough for the
    // scheduler to respect them, since it never runs a node before all of its
    // predecessors, so an order that follows from a chain of edges holds as
    // surely as a direct edge. A rule's all-pairs edges made a block's DAG
    // cubic to build (every edge added was searched for among its node's).

    // 1. Barrier-to-barrier ordering (Calls, Safepoints, GuardExits must
    // strictly preserve relative order): each barrier after the one before.
    int64_t prev_barrier = -1;
    for (uint32_t i = 0; i < n; ++i) {
        if (!nodes_[i].is_barrier) continue;
        if (prev_barrier >= 0) add_edge(static_cast<uint32_t>(prev_barrier), i, EdgeKind::Barrier, 1);
        prev_barrier = i;
    }

    // 2. Full barriers (Safepoints and GuardExits require exact program state;
    // cannot reorder across them). An invoke's call is one too: its landing
    // pad is entered from the call itself, so everything before it must be
    // done when it raises (a value the pad reads, computed after the call,
    // would be computed only on the normal path), and nothing after it may run
    // first (a def moved above it could overwrite a register the pad reads).
    // Everything from the previous full barrier on precedes one, and it
    // precedes everything up to the next; a node beyond reaches it through them.
    auto is_full_barrier = [&](uint32_t i) {
        const LirInst& inst = *nodes_[i].inst;
        return inst.opcode == LirOpcode::Safepoint || inst.opcode == LirOpcode::GuardExit ||
               inst.opcode == LirOpcode::ParallelCopy || inst.is_invoke;
    };
    std::vector<uint32_t> full;
    for (uint32_t i = 0; i < n; ++i) {
        if (is_full_barrier(i)) full.push_back(i);
    }
    // Every node of [lo, i) before i: walking down, a node with a successor
    // in (it, i] already is, through that successor.
    auto all_before = [&](uint32_t lo, uint32_t i) {
        for (uint32_t k = i; k-- > lo;) {
            bool ordered = false;
            for (const SchedEdge& s : nodes_[k].succs) ordered |= s.target_node <= i;
            if (!ordered) add_edge(k, i, EdgeKind::Barrier, 1);
        }
    };
    // i before every node of (i, hi]: walking up, a node with a predecessor
    // in [i, it) already is.
    auto all_after = [&](uint32_t i, uint32_t hi) {
        for (uint32_t k = i + 1; k <= hi; ++k) {
            bool ordered = false;
            for (const SchedEdge& p : nodes_[k].preds) ordered |= p.target_node >= i;
            if (!ordered) add_edge(i, k, EdgeKind::Barrier, 1);
        }
    };
    for (size_t f = 0; f < full.size(); ++f) {
        const uint32_t i = full[f];
        const uint32_t lo = f > 0 ? full[f - 1] : 0;
        const uint32_t hi = f + 1 < full.size() ? full[f + 1] : static_cast<uint32_t>(n - 1);
        all_before(lo, i);
        all_after(i, hi);
    }

    // 3. Call barriers: all memory operations before a call must precede the
    // call; call must precede all subsequent memory operations. Calls are
    // chained by rule 1, so a call is joined to those back to the previous
    // call and on to the next.
    std::vector<uint32_t> calls;
    for (uint32_t i = 0; i < n; ++i) {
        if (nodes_[i].inst->is_call()) calls.push_back(i);
    }
    for (size_t c = 0; c < calls.size(); ++c) {
        const uint32_t i = calls[c];
        const uint32_t lo = c > 0 ? calls[c - 1] + 1 : 0;
        const uint32_t hi = c + 1 < calls.size() ? calls[c + 1] : static_cast<uint32_t>(n);
        for (uint32_t k = lo; k < i; ++k) {
            if (nodes_[k].is_load || nodes_[k].is_store) add_edge(k, i, EdgeKind::Barrier, 1);
        }
        for (uint32_t k = i + 1; k < hi; ++k) {
            if (nodes_[k].is_load || nodes_[k].is_store) add_edge(i, k, EdgeKind::Barrier, 1);
        }
    }

    // 4. Control-flow & Terminator barrier:
    // Once the first terminator (e.g. Jcc, Jmp, Ret) is reached, all subsequent
    // instructions (including intervening parallel copies, argument setups, and jumps)
    // must strictly remain after all preceding instructions and preserve their exact order:
    // everything before the first terminator precedes it, and from it on each
    // instruction follows the one before.
    uint32_t first_term_idx = static_cast<uint32_t>(n);
    for (uint32_t i = 0; i < n; ++i) {
        if (nodes_[i].inst->is_terminator()) {
            first_term_idx = i;
            break;
        }
    }
    if (first_term_idx < n) {
        all_before(0, first_term_idx);
        for (uint32_t i = first_term_idx + 1; i < n; ++i) add_edge(i - 1, i, EdgeKind::Barrier, 1);
    }
}

void SchedDAG::compute_metrics() {
    size_t n = nodes_.size();
    if (n == 0) return;

    // Compute Depth in forward topological order (0..n-1)
    for (size_t i = 0; i < n; ++i) {
        uint32_t d = 0;
        for (const auto& p : nodes_[i].preds) {
            d = std::max(d, nodes_[p.target_node].depth + p.latency);
        }
        nodes_[i].depth = d;
    }

    // Compute Height in reverse topological order (n-1 down to 0)
    for (size_t idx = n; idx > 0; --idx) {
        size_t i = idx - 1;
        uint32_t max_succ_h = 0;
        for (const auto& s : nodes_[i].succs) {
            max_succ_h = std::max(max_succ_h, nodes_[s.target_node].height + s.latency);
        }
        nodes_[i].height = (nodes_[i].succs.empty()) ? nodes_[i].latency : max_succ_h;
    }
}

} // namespace brass::codegen
