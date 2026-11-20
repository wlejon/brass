#pragma once
#include <brass/verify/smt_equiv.hpp>
#include "bv_solver.hpp"
#include <string>
#include <string_view>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace brass::verify {

uint32_t type_bv_width(Type t) noexcept;

struct SymStore {
    BVNodeId guard = 0, addr = 0;
    uint32_t num_bytes = 0;
    BVNodeId val = 0;
    bool is_local_alloca = false;
};

struct SymGuardExit {
    BVNodeId fail_cond = 0;
    uint32_t resume_id = 0;
    std::string exit_symbol;
    std::vector<Type> state_types;
    std::vector<BVNodeId> state_vals;
};

struct InitMemEnv {
    struct ByteEntry { BVNodeId addr, val; };
    std::vector<ByteEntry> bytes;
    struct CallEntry { std::string sym; std::vector<BVNodeId> args; BVNodeId ret_val; };
    std::vector<CallEntry> calls;
    std::unordered_map<std::string, BVNodeId> patchable_consts;

    BVNodeId read_init_byte(BVContext& ctx, BVNodeId byte_addr);
    BVNodeId eval_uninterpreted_call(BVContext& ctx, std::string_view sym,
                                     const std::vector<BVNodeId>& args, Type ret_type);
    BVNodeId get_patchable_const(BVContext& ctx, std::string_view sym, uint32_t width);
};

using SymVecMap = std::unordered_map<const Value*, std::vector<BVNodeId>>;

struct SymCapturedEdge {
    const BasicBlock* target = nullptr;
    BVNodeId cond = 0;
    std::vector<BVNodeId> args;
    std::vector<std::vector<BVNodeId>> vec_args;
    std::unordered_map<const Value*, BVNodeId> vals;
    SymVecMap vec_vals;
};

struct SymExecRegionConfig {
    const BasicBlock* start_block = nullptr;
    const BasicBlock* capture_block = nullptr;
    std::unordered_set<const BasicBlock*> capture_blocks;
    bool capture_on_first_visit = false;
    std::unordered_map<const Value*, BVNodeId> initial_vals;
    SymVecMap initial_vec_vals;
    std::vector<std::vector<BVNodeId>> param_vec_vars;
};

struct SymExecOutput {
    bool supported = true;
    bool has_type_error = false;
    bool hit_truncation = false;
    std::string error_msg;
    BVNodeId completed = 0;
    BVNodeId guard_failed = 0;
    BVNodeId trapped = 0;
    BVNodeId returned = 0;
    BVNodeId ret_val = 0;
    std::vector<BVNodeId> ret_vec;
    std::vector<SymStore> stores;
    std::vector<SymGuardExit> guard_exits;
    std::vector<SymCapturedEdge> captured_edges;
};

bool symbolically_execute_vec_inst(
    BVContext& ctx,
    InitMemEnv& init_mem,
    SymExecOutput& out,
    BVNodeId cur_cond,
    const Instruction* inst,
    std::unordered_map<const Value*, BVNodeId>& vals,
    SymVecMap& vec_vals,
    const std::unordered_set<uint64_t>& alloca_bases);

BVNodeId read_byte_at(BVContext& ctx, InitMemEnv& init_mem,
                      const std::vector<SymStore>& stores, BVNodeId byte_addr);

BVNodeId read_memory_bv(BVContext& ctx, InitMemEnv& init_mem,
                        const std::vector<SymStore>& stores, BVNodeId addr, uint32_t num_bytes);

SymExecOutput symbolically_execute_mir(
    BVContext& ctx,
    const Function& fn,
    const std::vector<BVNodeId>& param_vars,
    InitMemEnv& init_mem,
    const EquivOptions& opts);

SymExecOutput symbolically_execute_mir_region(
    BVContext& ctx,
    const Function& fn,
    const std::vector<BVNodeId>& param_vars,
    InitMemEnv& init_mem,
    const EquivOptions& opts,
    const SymExecRegionConfig& region_cfg);

} // namespace brass::verify
