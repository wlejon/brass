#include <brass/codegen/sched_dag.hpp>
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

bool instruction_defines_flags(const LirInst& inst) noexcept {
    switch (inst.opcode) {
        case LirOpcode::Cmp:
        case LirOpcode::Cmp32:
        case LirOpcode::Test:
        case LirOpcode::Test32:
        case LirOpcode::Add:
        case LirOpcode::Add32:
        case LirOpcode::Sub:
        case LirOpcode::Sub32:
        case LirOpcode::Imul:
        case LirOpcode::Imul32:
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
        case LirOpcode::Ucomisd:
        case LirOpcode::Ucomiss:
            return true;
        default:
            return false;
    }
}

bool instruction_uses_flags(const LirInst& inst) noexcept {
    switch (inst.opcode) {
        case LirOpcode::Jcc:
        case LirOpcode::Setcc:
        case LirOpcode::Cmovcc:
            return true;
        default:
            return false;
    }
}

bool is_scheduling_barrier(const LirInst& inst) noexcept {
    if (inst.is_call() || inst.is_terminator()) {
        return true;
    }
    switch (inst.opcode) {
        case LirOpcode::Safepoint:
        case LirOpcode::GuardExit:
        case LirOpcode::WriteBarrier:
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
        case LirOpcode::ParallelCopy:
        case LirOpcode::WriteBarrier:
            return 1;
    }
    return 1;
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

static bool may_memory_alias(const LirInst& a, const LirInst& b) {
    std::vector<MemoryRef> refs_a = get_memory_refs(a);
    std::vector<MemoryRef> refs_b = get_memory_refs(b);

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

SchedDAG::SchedDAG(LirBlock& block, bool is_pre_ra)
    : block_(block), is_pre_ra_(is_pre_ra) {}

void SchedDAG::build() {
    nodes_.clear();
    build_nodes();
    build_register_dependencies();
    build_memory_dependencies();
    build_barrier_dependencies();
    compute_metrics();
}

void SchedDAG::build_nodes() {
    nodes_.reserve(block_.instructions.size());
    for (size_t i = 0; i < block_.instructions.size(); ++i) {
        SchedNode node;
        node.id = static_cast<uint32_t>(i);
        node.inst = block_.instructions[i].get();
        node.latency = get_instruction_latency(*node.inst);
        node.is_pre_ra = is_pre_ra_;
        node.is_load = instruction_reads_memory(*node.inst);
        node.is_store = instruction_writes_memory(*node.inst);
        node.is_barrier = is_scheduling_barrier(*node.inst);
        nodes_.push_back(std::move(node));
    }
}

void SchedDAG::add_edge(uint32_t from, uint32_t to, EdgeKind kind, uint32_t latency) {
    if (from >= nodes_.size() || to >= nodes_.size() || from == to) {
        return;
    }

    // Check if edge already exists
    for (auto& s : nodes_[from].succs) {
        if (s.target_node == to) {
            s.latency = std::max(s.latency, latency);
            if (kind == EdgeKind::RAW) s.kind = EdgeKind::RAW;
            for (auto& p : nodes_[to].preds) {
                if (p.target_node == from) {
                    p.latency = std::max(p.latency, latency);
                    if (kind == EdgeKind::RAW) p.kind = EdgeKind::RAW;
                    break;
                }
            }
            return;
        }
    }

    SchedEdge succ_edge;
    succ_edge.target_node = to;
    succ_edge.kind = kind;
    succ_edge.latency = latency;
    nodes_[from].succs.push_back(succ_edge);

    SchedEdge pred_edge;
    pred_edge.target_node = from;
    pred_edge.kind = kind;
    pred_edge.latency = latency;
    nodes_[to].preds.push_back(pred_edge);

    nodes_[to].unscheduled_preds++;
}

void SchedDAG::build_register_dependencies() {
    size_t n = nodes_.size();

    // Virtual register tracking
    std::unordered_map<uint32_t, uint32_t> last_def_vreg;
    std::unordered_map<uint32_t, std::vector<uint32_t>> last_uses_vreg;

    // Physical register tracking: (reg_class << 8 | code) -> node index
    std::unordered_map<uint32_t, uint32_t> last_def_preg;
    std::unordered_map<uint32_t, std::vector<uint32_t>> last_uses_preg;

    for (uint32_t j = 0; j < n; ++j) {
        const auto& inst = *nodes_[j].inst;

        // 1. Collect VReg and PReg uses
        std::vector<uint32_t> uses_vregs;
        std::vector<uint32_t> uses_pregs;

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
            auto def_it = last_def_preg.find(pkey);
            if (def_it != last_def_preg.end()) {
                add_edge(def_it->second, j, EdgeKind::RAW, nodes_[def_it->second].latency);
            }
            last_uses_preg[pkey].push_back(j);
        }

        // 2. Collect VReg and PReg defs
        std::vector<uint32_t> defs_vregs;
        std::vector<uint32_t> defs_pregs;

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
        for (uint8_t c = 0; c < 16; ++c) {
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
            auto u_it = last_uses_preg.find(pkey);
            if (u_it != last_uses_preg.end()) {
                for (uint32_t u_node : u_it->second) {
                    if (u_node != j) {
                        add_edge(u_node, j, EdgeKind::WAR, 0);
                    }
                }
                u_it->second.clear();
            }
            auto def_it = last_def_preg.find(pkey);
            if (def_it != last_def_preg.end()) {
                add_edge(def_it->second, j, EdgeKind::WAW, 1);
            }
            last_def_preg[pkey] = j;
        }
    }

    // Condition flags tracking
    int32_t last_flag_def = -1;
    std::vector<uint32_t> last_flag_uses;

    for (uint32_t j = 0; j < n; ++j) {
        const auto& inst = *nodes_[j].inst;
        if (instruction_uses_flags(inst)) {
            if (last_flag_def >= 0) {
                add_edge(static_cast<uint32_t>(last_flag_def), j, EdgeKind::RAW, 1);
            }
            last_flag_uses.push_back(j);
        }
        if (instruction_defines_flags(inst)) {
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
    size_t n = nodes_.size();

    for (uint32_t i = 0; i < n; ++i) {
        const auto& inst_i = *nodes_[i].inst;
        bool i_reads = instruction_reads_memory(inst_i);
        bool i_writes = instruction_writes_memory(inst_i);
        if (!i_reads && !i_writes) continue;

        for (uint32_t j = i + 1; j < n; ++j) {
            const auto& inst_j = *nodes_[j].inst;
            bool j_reads = instruction_reads_memory(inst_j);
            bool j_writes = instruction_writes_memory(inst_j);
            if (!j_reads && !j_writes) continue;

            if (may_memory_alias(inst_i, inst_j)) {
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

    // 1. Barrier-to-barrier ordering (Calls, Safepoints, GuardExits must strictly preserve relative order)
    for (uint32_t i = 0; i < n; ++i) {
        if (nodes_[i].is_barrier) {
            for (uint32_t k = 0; k < i; ++k) {
                if (nodes_[k].is_barrier) {
                    add_edge(k, i, EdgeKind::Barrier, 1);
                }
            }
        }
    }

    // 2. Full barriers (Safepoints and GuardExits require exact program state; cannot reorder across them)
    for (uint32_t i = 0; i < n; ++i) {
        if (nodes_[i].inst->opcode == LirOpcode::Safepoint ||
            nodes_[i].inst->opcode == LirOpcode::GuardExit) {
            for (uint32_t k = 0; k < i; ++k) {
                add_edge(k, i, EdgeKind::Barrier, 1);
            }
            for (uint32_t k = i + 1; k < n; ++k) {
                add_edge(i, k, EdgeKind::Barrier, 1);
            }
        }
    }

    // 3. Call barriers: all memory operations before a call must precede the call;
    // call must precede all subsequent memory operations.
    for (uint32_t i = 0; i < n; ++i) {
        if (nodes_[i].inst->is_call()) {
            for (uint32_t k = 0; k < i; ++k) {
                if (nodes_[k].is_load || nodes_[k].is_store) {
                    add_edge(k, i, EdgeKind::Barrier, 1);
                }
            }
            for (uint32_t k = i + 1; k < n; ++k) {
                if (nodes_[k].is_load || nodes_[k].is_store) {
                    add_edge(i, k, EdgeKind::Barrier, 1);
                }
            }
        }
    }

    // 4. Control-flow & Terminator barrier:
    // Once the first terminator (e.g. Jcc, Jmp, Ret) is reached, all subsequent
    // instructions (including intervening parallel copies, argument setups, and jumps)
    // must strictly remain after all preceding instructions and preserve their exact order.
    uint32_t first_term_idx = static_cast<uint32_t>(n);
    for (uint32_t i = 0; i < n; ++i) {
        if (nodes_[i].inst->is_terminator()) {
            first_term_idx = i;
            break;
        }
    }
    if (first_term_idx < n) {
        for (uint32_t i = first_term_idx; i < n; ++i) {
            for (uint32_t k = 0; k < i; ++k) {
                add_edge(k, i, EdgeKind::Barrier, 1);
            }
        }
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
