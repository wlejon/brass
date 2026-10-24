#pragma once
#include <brass/verify/arm64_lir_equiv.hpp>
#include "bv_solver.hpp"
#include <string>
#include <unordered_map>
#include <vector>

namespace brass::verify {

struct Arm64NzcvState {
    BVNodeId n = 0, z = 0, c = 0, v = 0; // Negative, Zero, Carry (NOT borrow on SUBS/CMP), oVerflow
    bool from_subs = false;
    BVNodeId subs_a = 0, subs_b = 0;
};

struct Arm64SymMemStore { BVNodeId addr = 0; uint32_t size = 8; BVNodeId val = 0; };

struct Arm64InitEnv {
    BVNodeId gpr_init[32]{}, vec_lo_init[32]{}, vec_hi_init[32]{};
    Arm64NzcvState nzcv_init{};
    std::unordered_map<uint32_t, BVNodeId> vreg_init;
    std::unordered_map<int32_t, BVNodeId> spill_init, local_init;
    struct MemReadEntry { BVNodeId addr; uint32_t size; BVNodeId val; };
    std::vector<MemReadEntry> mem_reads;

    void init_defaults(BVContext& ctx);
    BVNodeId get_vreg_init(BVContext& ctx, uint32_t id);
    BVNodeId get_spill_init(BVContext& ctx, int32_t slot);
    BVNodeId get_local_init(BVContext& ctx, int32_t off);
    BVNodeId read_init_mem(BVContext& ctx, BVNodeId addr, uint32_t sz);
};

struct Arm64ExecState {
    bool supported = true;
    std::string error_msg;
    BVNodeId gpr[32]{}, vec_lo[32]{}, vec_hi[32]{};
    Arm64NzcvState nzcv{};
    std::unordered_map<uint32_t, BVNodeId> vregs;
    std::unordered_map<int32_t, BVNodeId> spill_slots, local_slots;
    std::vector<Arm64SymMemStore> mem_stores;
    bool has_cond_branch = false, has_uncond_branch = false, uses_vec128 = false;
    BVNodeId cond_branch_taken = 0;
    uint32_t cond_target_label = UINT32_MAX, uncond_target_label = UINT32_MAX;
};

BVNodeId eval_arm64_cond(BVContext& ctx, const Arm64NzcvState& f, aarch64::Condition cond);
void set_arm64_adds_flags(BVContext& ctx, Arm64NzcvState& f, BVNodeId a, BVNodeId b, BVNodeId res, uint32_t w);
void set_arm64_subs_flags(BVContext& ctx, Arm64NzcvState& f, BVNodeId a, BVNodeId b, BVNodeId res, uint32_t w);
void set_arm64_ands_flags(BVContext& ctx, Arm64NzcvState& f, BVNodeId res, uint32_t w);
void set_arm64_fcmp_flags(BVContext& ctx, Arm64NzcvState& f, BVNodeId a, BVNodeId b);

Arm64ExecState execute_arm64_block(BVContext& ctx, const codegen::LirBlock& block, Arm64InitEnv& env);

} // namespace brass::verify
