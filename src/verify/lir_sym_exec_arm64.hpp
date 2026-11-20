#pragma once
#include <brass/verify/arm64_lir_equiv.hpp>
#include "bv_solver.hpp"
#include "mir_sym_exec.hpp"
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
    std::unordered_map<uint32_t, BVNodeId> vreg_init, vreg_hi_init;
    std::unordered_map<int32_t, BVNodeId> spill_init, spill_hi_init, local_init, local_hi_init;
    struct MemReadEntry { BVNodeId addr; uint32_t size; BVNodeId val; };
    std::vector<MemReadEntry> mem_reads;
    InitMemEnv* shared_mem = nullptr;

    void init_defaults(BVContext& ctx);
    BVNodeId get_vreg_init(BVContext& ctx, uint32_t id);
    BVNodeId get_vreg_hi_init(BVContext& ctx, uint32_t id);
    BVNodeId get_spill_init(BVContext& ctx, int32_t slot);
    BVNodeId get_spill_hi_init(BVContext& ctx, int32_t slot);
    BVNodeId get_local_init(BVContext& ctx, int32_t off);
    BVNodeId get_local_hi_init(BVContext& ctx, int32_t off);
    BVNodeId read_init_mem(BVContext& ctx, BVNodeId addr, uint32_t sz);
};

struct Arm64ExecState {
    bool supported = true;
    std::string error_msg;
    BVNodeId gpr[32]{}, vec_lo[32]{}, vec_hi[32]{};
    Arm64NzcvState nzcv{};
    std::unordered_map<uint32_t, BVNodeId> vregs, vreg_hi;
    std::unordered_map<int32_t, BVNodeId> spill_slots, spill_slots_hi, local_slots, local_slots_hi;
    std::vector<Arm64SymMemStore> mem_stores;
    std::vector<SymStore>* shared_stores = nullptr;
    BVNodeId path_cond = 0;
    uint64_t local_slot_base = 0;
    uint64_t canonical_fp = 0;
    bool has_cond_branch = false, has_uncond_branch = false, uses_vec128 = false;
    BVNodeId cond_branch_taken = 0;
    uint32_t cond_target_label = UINT32_MAX, uncond_target_label = UINT32_MAX;
};

BVNodeId arm64_read_op(BVContext& ctx, Arm64InitEnv& env, Arm64ExecState& st, const codegen::LirOperand& op, uint32_t width_bits);
void arm64_write_op(BVContext& ctx, Arm64InitEnv& env, Arm64ExecState& st, const codegen::LirOperand& dst, BVNodeId val);
std::pair<BVNodeId, BVNodeId> arm64_read_vec128(BVContext& ctx, Arm64InitEnv& env, Arm64ExecState& st, const codegen::LirOperand& op);
void arm64_write_vec128(BVContext& ctx, Arm64InitEnv& env, Arm64ExecState& st, const codegen::LirOperand& dst, BVNodeId lo, BVNodeId hi);
bool execute_arm64_vec_inst(BVContext& ctx, Arm64InitEnv& env, Arm64ExecState& st, const codegen::LirInst& inst);
bool execute_arm64_inst(BVContext& ctx, Arm64InitEnv& env, Arm64ExecState& st, const codegen::LirInst& inst);

BVNodeId eval_arm64_cond(BVContext& ctx, const Arm64NzcvState& f, aarch64::Condition cond);
void set_arm64_adds_flags(BVContext& ctx, Arm64NzcvState& f, BVNodeId a, BVNodeId b, BVNodeId res, uint32_t w);
void set_arm64_subs_flags(BVContext& ctx, Arm64NzcvState& f, BVNodeId a, BVNodeId b, BVNodeId res, uint32_t w);
void set_arm64_ands_flags(BVContext& ctx, Arm64NzcvState& f, BVNodeId res, uint32_t w);
void set_arm64_fcmp_flags(BVContext& ctx, Arm64NzcvState& f, BVNodeId a, BVNodeId b);

Arm64ExecState execute_arm64_block(BVContext& ctx, const codegen::LirBlock& block, Arm64InitEnv& env);

} // namespace brass::verify
