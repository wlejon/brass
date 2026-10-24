// Static compute shader prover (`spirv::audit_compute_shader` and
// `spirv::audit_compute_shader_binary`).

#include "spirv_shader_prover_internal.hpp"

#include <algorithm>
#include <unordered_map>
#include <unordered_set>

namespace brass::spirv {

using namespace detail;

ComputeShaderAuditReport audit_compute_shader(const Module& mod) {
    ComputeShaderAuditReport report;
    report.parse_ok = true;
    report.num_functions = static_cast<uint32_t>(mod.functions.size());

    // 1. Structural & SSA Dominance Invariants
    report.verifier_diagnostics = verify(mod);
    report.structural_ok = report.verifier_diagnostics.empty();
    for (const Diagnostic& d : report.verifier_diagnostics) {
        report.hazards.push_back(ShaderHazard{
            HazardKind::StructuralOrDominance, d.function, d.block, 0, spv::OpNop, d.message});
    }

    ModuleConstants mc = collect_module_constants(mod);

    // Extract LocalSize from OpExecutionMode / OpExecutionModeId or BuiltIn WorkgroupSize.
    for (const Inst& em : mod.execution_modes) {
        if (em.op == spv::OpExecutionMode && em.operands.size() >= 5 &&
            em.operands[1].value == static_cast<uint32_t>(spv::ExecutionModeLocalSize)) {
            report.local_size = {em.operands[2].value, em.operands[3].value, em.operands[4].value};
        } else if (em.op == spv::OpExecutionModeId && em.operands.size() >= 5 &&
                   em.operands[1].value == static_cast<uint32_t>(spv::ExecutionModeLocalSizeId)) {
            for (size_t k = 0; k < 3; ++k) {
                auto it = mc.int_consts.find(em.operands[2 + k].value);
                if (it != mc.int_consts.end() && it->second > 0) report.local_size[k] = static_cast<uint32_t>(it->second);
            }
        }
    }
    for (const Inst& g : mod.globals) {
        if ((g.op == spv::OpConstantComposite || g.op == spv::OpSpecConstantComposite) && g.operands.size() >= 3) {
            auto bit = mc.builtins.find(g.result);
            if (bit != mc.builtins.end() && bit->second == kBuiltInWorkgroupSize) {
                for (size_t k = 0; k < 3; ++k) {
                    auto it = mc.int_consts.find(g.operands[k].value);
                    if (it != mc.int_consts.end() && it->second > 0) report.local_size[k] = static_cast<uint32_t>(it->second);
                }
            }
        }
    }

    for (const auto& [vid, sc] : mc.var_storage) {
        if (sc != spv::StorageClassWorkgroup) continue;
        auto pit = mc.var_pointee.find(vid);
        if (pit != mc.var_pointee.end()) {
            const TypeInfo* ti = mod.type_info(pit->second);
            if (ti && ti->op == spv::OpTypeArray) ++report.num_workgroup_arrays;
        }
    }
    report.dominance_ok = report.barriers_uniform = report.shared_bounds_ok = report.div_sqrt_ok = true;

    for (const Function& fn : mod.functions) {
        report.num_blocks += static_cast<uint32_t>(fn.blocks.size());
        FunctionCFG cfg = build_cfg(fn);
        if (cfg.num_blocks == 0) continue;

        // Collect local SSA definitions and single-store Function variables.
        std::unordered_map<Id, std::pair<size_t, size_t>> def_loc;
        std::unordered_map<Id, const Inst*> def_inst;
        std::unordered_map<Id, spv::StorageClass> local_var_sc = mc.var_storage;
        std::unordered_map<Id, uint32_t> store_count;
        std::unordered_map<Id, Id> single_store_val;
        for (size_t bi = 0; bi < cfg.num_blocks; ++bi) {
            const Block& b = fn.blocks[bi];
            report.num_instructions += static_cast<uint32_t>(b.insts.size());
            for (size_t ii = 0; ii < b.insts.size(); ++ii) {
                const Inst& inst = b.insts[ii];
                if (inst.result != 0) {
                    def_loc[inst.result] = {bi, ii};
                    def_inst[inst.result] = &inst;
                }
                if (inst.op == spv::OpVariable && !inst.operands.empty()) {
                    local_var_sc[inst.result] = static_cast<spv::StorageClass>(inst.operands[0].value);
                } else if (inst.op == spv::OpStore && inst.operands.size() >= 2) {
                    Id ptr = inst.operands[0].value;
                    if (++store_count[ptr] == 1) single_store_val[ptr] = inst.operands[1].value;
                    else single_store_val.erase(ptr);
                }
            }
        }

        // Check SSA dominance across all reachable blocks.
        for (size_t bi = 0; bi < cfg.num_blocks; ++bi) {
            if (!cfg.reachable[bi]) continue;
            const Block& b = fn.blocks[bi];
            for (size_t ii = 0; ii < b.insts.size(); ++ii) {
                const Inst& inst = b.insts[ii];
                if (inst.op == spv::OpPhi) {
                    for (size_t k = 0; k + 1 < inst.operands.size(); k += 2) {
                        Id val_id = inst.operands[k].value;
                        Id pred_label = inst.operands[k + 1].value;
                        auto dit = def_loc.find(val_id);
                        auto pit = cfg.block_index.find(pred_label);
                        if (dit != def_loc.end() && pit != cfg.block_index.end() && cfg.reachable[pit->second]) {
                            size_t def_b = dit->second.first;
                            if (!cfg.dom[pit->second].count(def_b)) {
                                report.dominance_ok = false;
                                report.hazards.push_back(ShaderHazard{
                                    HazardKind::StructuralOrDominance, fn.name, b.label, inst.result, inst.op,
                                    "OpPhi operand %" + std::to_string(val_id) +
                                        " does not dominate predecessor %" + std::to_string(pred_label)});
                            }
                        }
                    }
                } else {
                    for (const Operand& opd : inst.operands) {
                        if (!opd.is_id()) continue;
                        auto dit = def_loc.find(opd.value);
                        if (dit != def_loc.end()) {
                            auto [def_b, def_i] = dit->second;
                            if (!cfg.dom[bi].count(def_b) || (def_b == bi && def_i >= ii)) {
                                report.dominance_ok = false;
                                report.hazards.push_back(ShaderHazard{
                                    HazardKind::StructuralOrDominance, fn.name, b.label, inst.result, inst.op,
                                    "SSA operand %" + std::to_string(opd.value) +
                                        " used before or without dominating definition"});
                            }
                        }
                    }
                }
            }
        }

        // Identify builtin component pointers/values and propagate uniformity taint.
        std::unordered_map<Id, int> lid_comp; // 0:lid.x, 1:lid.y, 2:lid.z, 3:local_idx, 4:sg_local, 5:sg_id, 10:lid_vec
        std::unordered_set<Id> wg_div;
        std::unordered_set<Id> sg_div;
        for (const auto& [vid, bkind] : mc.builtins) {
            if (is_wg_divergent_builtin(bkind)) wg_div.insert(vid);
            if (is_sg_divergent_builtin(bkind)) sg_div.insert(vid);
            if (bkind == kBuiltInLocalInvocationId) lid_comp[vid] = 10;
            else if (bkind == kBuiltInLocalInvocationIndex) lid_comp[vid] = 3;
            else if (bkind == kBuiltInSubgroupLocalInvocationId) lid_comp[vid] = 4;
            else if (bkind == kBuiltInSubgroupId) lid_comp[vid] = 5;
        }

        for (int pass = 0; pass < 3; ++pass) {
            for (size_t bi = 0; bi < cfg.num_blocks; ++bi) {
                if (!cfg.reachable[bi]) continue;
                bool blk_wg_div = false;
                bool blk_sg_div = false;
                for (const auto& [h, reg] : cfg.control_dep_region) {
                    if (reg.count(bi)) {
                        Id c = cfg.branch_cond[h];
                        if (wg_div.count(c)) blk_wg_div = true;
                        if (sg_div.count(c)) blk_sg_div = true;
                    }
                }
                for (const Inst& inst : fn.blocks[bi].insts) {
                    if (inst.op == spv::OpAccessChain || inst.op == spv::OpInBoundsAccessChain) {
                        if (inst.operands.empty()) continue;
                        Id base = inst.operands[0].value;
                        auto bit = lid_comp.find(base);
                        if (bit != lid_comp.end() && bit->second == 10 && inst.operands.size() >= 2) {
                            auto cit = mc.int_consts.find(inst.operands[1].value);
                            if (cit != mc.int_consts.end() && cit->second < 3) {
                                lid_comp[inst.result] = static_cast<int>(cit->second);
                            }
                        }
                        auto pit = mc.var_pointee.find(base);
                        if (pit != mc.var_pointee.end() && inst.operands.size() >= 2) {
                            auto cit = mc.int_consts.find(inst.operands[1].value);
                            if (cit != mc.int_consts.end()) {
                                uint64_t mkey = (static_cast<uint64_t>(pit->second) << 32) | cit->second;
                                auto mbit = mc.member_builtins.find(mkey);
                                if (mbit != mc.member_builtins.end()) {
                                    if (is_wg_divergent_builtin(mbit->second)) wg_div.insert(inst.result);
                                    if (is_sg_divergent_builtin(mbit->second)) sg_div.insert(inst.result);
                                }
                            }
                        }
                        for (const Operand& o : inst.operands) {
                            if (o.is_id() && wg_div.count(o.value)) wg_div.insert(inst.result);
                            if (o.is_id() && sg_div.count(o.value)) sg_div.insert(inst.result);
                        }
                    } else if (inst.op == spv::OpLoad && !inst.operands.empty()) {
                        Id ptr = inst.operands[0].value;
                        auto bit = lid_comp.find(ptr);
                        if (bit != lid_comp.end()) lid_comp[inst.result] = bit->second;
                        if (wg_div.count(ptr)) wg_div.insert(inst.result);
                        if (sg_div.count(ptr)) sg_div.insert(inst.result);
                    } else if (inst.op == spv::OpStore && inst.operands.size() >= 2) {
                        Id ptr = inst.operands[0].value;
                        Id val = inst.operands[1].value;
                        auto scit = local_var_sc.find(ptr);
                        if (scit != local_var_sc.end() &&
                            (scit->second == spv::StorageClassFunction || scit->second == spv::StorageClassPrivate)) {
                            if (wg_div.count(val) || blk_wg_div) wg_div.insert(ptr);
                            if (sg_div.count(val) || blk_sg_div) sg_div.insert(ptr);
                        }
                    } else if (is_subgroup_uniformizing_op(inst)) {
                        wg_div.insert(inst.result);
                        sg_div.erase(inst.result);
                    } else if (inst.op == spv::OpPhi) {
                        for (size_t k = 0; k + 1 < inst.operands.size(); k += 2) {
                            Id val = inst.operands[k].value;
                            Id plab = inst.operands[k + 1].value;
                            if (wg_div.count(val)) wg_div.insert(inst.result);
                            if (sg_div.count(val)) sg_div.insert(inst.result);
                            auto pit = cfg.block_index.find(plab);
                            if (pit != cfg.block_index.end()) {
                                for (const auto& [h, reg] : cfg.control_dep_region) {
                                    if (reg.count(pit->second) && !reg.count(bi)) {
                                        Id c = cfg.branch_cond[h];
                                        if (wg_div.count(c)) wg_div.insert(inst.result);
                                        if (sg_div.count(c)) sg_div.insert(inst.result);
                                    }
                                }
                            }
                        }
                    } else if (inst.result != 0) {
                        for (const Operand& o : inst.operands) {
                            if (o.is_id() && wg_div.count(o.value)) wg_div.insert(inst.result);
                            if (o.is_id() && sg_div.count(o.value)) sg_div.insert(inst.result);
                        }
                        if (inst.op == spv::OpUDiv && inst.operands.size() >= 2) {
                            auto cit = mc.int_consts.find(inst.operands[1].value);
                            if (cit != mc.int_consts.end() && cit->second >= 32 && (cit->second % 32) == 0) {
                                sg_div.erase(inst.result);
                            }
                        }
                    }
                }
            }
        }

        // Helper to build BVSolver context with LocalInvocationId bounds for block `bi`.
        auto make_bv_prover = [&](size_t bi, verify::BVContext& ctx, verify::BVNodeId& out_pc) -> SymbolicBuilder {
            std::array<verify::BVNodeId, 6> bvars;
            bvars[0] = ctx.make_var("lid_x", 32);
            bvars[1] = ctx.make_var("lid_y", 32);
            bvars[2] = ctx.make_var("lid_z", 32);
            uint32_t lx = std::max(1u, report.local_size[0]);
            uint32_t ly = std::max(1u, report.local_size[1]);
            uint32_t lz = std::max(1u, report.local_size[2]);
            uint32_t wg_total = lx * ly * lz;
            uint32_t max_sg = std::max(1u, (wg_total + 31u) / 32u);
            verify::BVNodeId c_lx = ctx.make_const(lx, 32);
            verify::BVNodeId c_ly = ctx.make_const(ly, 32);
            verify::BVNodeId c_lz = ctx.make_const(lz, 32);
            verify::BVNodeId zy = ctx.make_binary(verify::BVOp::Add,
                                                  ctx.make_binary(verify::BVOp::Mul, bvars[2], c_ly, 32), bvars[1], 32);
            bvars[3] = ctx.make_binary(verify::BVOp::Add,
                                       ctx.make_binary(verify::BVOp::Mul, zy, c_lx, 32), bvars[0], 32);
            bvars[4] = ctx.make_var("sg_local_id", 32);
            bvars[5] = ctx.make_var("sg_id", 32);

            verify::BVNodeId pc = ctx.make_binary(verify::BVOp::Ult, bvars[0], c_lx, 1);
            pc = ctx.make_binary(verify::BVOp::And, pc, ctx.make_binary(verify::BVOp::Ult, bvars[1], c_ly, 1), 1);
            pc = ctx.make_binary(verify::BVOp::And, pc, ctx.make_binary(verify::BVOp::Ult, bvars[2], c_lz, 1), 1);
            pc = ctx.make_binary(verify::BVOp::And, pc,
                                 ctx.make_binary(verify::BVOp::Ult, bvars[4], ctx.make_const(std::min(64u, wg_total), 32), 1), 1);
            pc = ctx.make_binary(verify::BVOp::And, pc,
                                 ctx.make_binary(verify::BVOp::Ult, bvars[5], ctx.make_const(max_sg, 32), 1), 1);

            SymbolicBuilder sb{mod, fn, cfg, mc, def_inst, lid_comp, single_store_val, ctx, bvars, {}, {}};
            for (const auto& [h, reg] : cfg.control_dep_region) {
                if (!reg.count(bi)) continue;
                const Inst& hterm = fn.blocks[h].insts.back();
                if (hterm.op == spv::OpBranchConditional && hterm.operands.size() >= 3) {
                    Id cond_id = hterm.operands[0].value;
                    auto tit = cfg.block_index.find(hterm.operands[1].value);
                    auto fit = cfg.block_index.find(hterm.operands[2].value);
                    bool in_t = (tit != cfg.block_index.end() && cfg.dom[bi].count(tit->second));
                    bool in_f = (fit != cfg.block_index.end() && cfg.dom[bi].count(fit->second));
                    if (in_t != in_f) {
                        verify::BVNodeId cnode = sb.build(cond_id);
                        if (ctx.width(cnode) != 1) {
                            cnode = ctx.make_binary(verify::BVOp::Ne, cnode, ctx.make_const(0, ctx.width(cnode)), 1);
                        }
                        if (in_f) cnode = ctx.make_unary(verify::BVOp::Not, cnode, 1);
                        pc = ctx.make_binary(verify::BVOp::And, pc, cnode, 1);
                    }
                }
            }
            for (verify::BVNodeId ec : sb.extra_constraints) pc = ctx.make_binary(verify::BVOp::And, pc, ec, 1);
            out_pc = pc;
            return sb;
        };

        // 2, 3, 4: Inspect instructions in block `bi`.
        for (size_t bi = 0; bi < cfg.num_blocks; ++bi) {
            if (!cfg.reachable[bi]) continue;
            const Block& b = fn.blocks[bi];

            std::unordered_set<Id> select_operands;
            for (const Inst& inst : b.insts) {
                if (inst.op == spv::OpSelect && inst.operands.size() >= 3) {
                    select_operands.insert(inst.operands[1].value);
                    select_operands.insert(inst.operands[2].value);
                } else if (inst.op == spv::OpCompositeConstruct) {
                    for (const Operand& o : inst.operands) select_operands.insert(o.value);
                }
            }

            for (const Inst& inst : b.insts) {
                // Property 2: Barrier & Subgroup Legality
                if (inst.op == spv::OpControlBarrier || is_subgroup_collective_op(inst.op)) {
                    bool is_wg = false;
                    if (inst.op == spv::OpControlBarrier && !inst.operands.empty()) {
                        auto cit = mc.int_consts.find(inst.operands[0].value);
                        uint64_t scope = (cit != mc.int_consts.end()) ? cit->second : static_cast<uint64_t>(spv::ScopeWorkgroup);
                        is_wg = (scope == static_cast<uint64_t>(spv::ScopeWorkgroup) ||
                                 scope == static_cast<uint64_t>(spv::ScopeDevice));
                    }
                    bool divergent = false;
                    Id div_cond = 0;
                    for (const auto& [h, reg] : cfg.control_dep_region) {
                        if (!reg.count(bi)) continue;
                        Id c = cfg.branch_cond[h];
                        if (is_wg ? wg_div.count(c) : sg_div.count(c)) {
                            if (!is_wg && inst.op != spv::OpControlBarrier && inst.operands.size() >= 4 &&
                                inst.operands[1].value == static_cast<uint32_t>(spv::GroupOperationClusteredReduce)) {
                                continue;
                            }
                            verify::BVContext ctx;
                            verify::BVNodeId pc_base = 0;
                            SymbolicBuilder sb = make_bv_prover(h, ctx, pc_base);
                            verify::BVNodeId cnode = sb.build(c);
                            if (ctx.width(cnode) != 1) {
                                cnode = ctx.make_binary(verify::BVOp::Ne, cnode, ctx.make_const(0, ctx.width(cnode)), 1);
                            }
                            auto can_true = verify::solve_nonzero(
                                ctx, ctx.make_binary(verify::BVOp::And, pc_base, cnode, 1), 2000);
                            auto can_false = verify::solve_nonzero(
                                ctx, ctx.make_binary(verify::BVOp::And, pc_base,
                                                     ctx.make_unary(verify::BVOp::Not, cnode, 1), 1), 2000);
                            if (can_true.status == verify::SolveOutcome::Sat &&
                                can_false.status == verify::SolveOutcome::Sat) {
                                divergent = true;
                                div_cond = c;
                                break;
                            }
                        }
                    }
                    std::string detail = divergent
                        ? ("divergent control flow on condition %" + std::to_string(div_cond))
                        : "uniform control flow";
                    report.barriers.push_back(BarrierAuditEntry{fn.name, b.label, inst.op, is_wg, !divergent, detail});
                    if (divergent) {
                        report.barriers_uniform = false;
                        report.hazards.push_back(ShaderHazard{
                            is_wg ? HazardKind::DivergentBarrier : HazardKind::DivergentSubgroupOp,
                            fn.name, b.label, inst.result, inst.op, detail});
                    }
                }

                // Property 3: Workgroup `shared` Memory Array Bounds Safety
                if ((inst.op == spv::OpAccessChain || inst.op == spv::OpInBoundsAccessChain) &&
                    inst.operands.size() >= 2) {
                    Id base = inst.operands[0].value;
                    auto scit = mc.var_storage.find(base);
                    auto pit = mc.var_pointee.find(base);
                    if (scit != mc.var_storage.end() && scit->second == spv::StorageClassWorkgroup &&
                        pit != mc.var_pointee.end()) {
                        Id cur_type = pit->second;
                        for (size_t idx_pos = 1; idx_pos < inst.operands.size(); ++idx_pos) {
                            const TypeInfo* ti = mod.type_info(cur_type);
                            if (!ti) break;
                            if (ti->op == spv::OpTypeArray && ti->width > 0) {
                                Id idx_id = inst.operands[idx_pos].value;
                                uint32_t arr_len = ti->width;
                                verify::BVContext ctx;
                                verify::BVNodeId pc = 0;
                                SymbolicBuilder sb = make_bv_prover(bi, ctx, pc);
                                verify::BVNodeId idx_bv = sb.build(idx_id);
                                if (ctx.width(idx_bv) < 32) idx_bv = ctx.make_zext(idx_bv, 32 - ctx.width(idx_bv));
                                for (verify::BVNodeId ec : sb.extra_constraints) {
                                    pc = ctx.make_binary(verify::BVOp::And, pc, ec, 1);
                                }
                                verify::BVNodeId oob_cond = ctx.make_binary(
                                    verify::BVOp::And, pc,
                                    ctx.make_binary(verify::BVOp::Uge, idx_bv, ctx.make_const(arr_len, 32), 1), 1);
                                auto res = verify::solve_nonzero(ctx, oob_cond, 4000);

                                std::vector<verify::BVNodeId> c_nodes;
                                std::vector<uint32_t> c_vars;
                                ctx.compute_cone(idx_bv, c_nodes, c_vars);
                                bool only_constrained_builtins = (arr_len > 1);
                                for (uint32_t vi : c_vars) {
                                    if (vi >= 5) { // 0..4 are lid_x, lid_y, lid_z, sg_local_id, sg_id
                                        only_constrained_builtins = false;
                                        break;
                                    }
                                }

                                bool proven_ok = (res.status == verify::SolveOutcome::Unsat);
                                bool static_oob = (res.status == verify::SolveOutcome::Sat && only_constrained_builtins);
                                std::string detail = proven_ok
                                    ? "proven in-bounds (< " + std::to_string(arr_len) + ") by BVSolver"
                                    : (static_oob ? "STATIC OOB (>= " + std::to_string(arr_len) + ")"
                                                  : "dynamic index into shared[" + std::to_string(arr_len) + "]");
                                report.shared_accesses.push_back(SharedMemoryAccessAudit{
                                    fn.name, b.label, inst.result, base, idx_id, arr_len, proven_ok, static_oob, detail});
                                if (static_oob) {
                                    report.shared_bounds_ok = false;
                                    report.hazards.push_back(ShaderHazard{
                                        HazardKind::SharedMemoryOutOfBounds, fn.name, b.label, inst.result, inst.op,
                                        "shared memory index %" + std::to_string(idx_id) +
                                            " can exceed array length " + std::to_string(arr_len)});
                                }
                            }
                            cur_type = ti->elem;
                        }
                    }
                }

                // Property 4: Division / Reciprocal / Sqrt Hazard Analysis
                bool is_div = (inst.op == spv::OpFDiv || inst.op == spv::OpUDiv || inst.op == spv::OpSDiv);
                bool is_sqrt = (inst.op == spv::OpExtInst && inst.operands.size() >= 3 &&
                                (inst.operands[1].value == GLSLstd450Sqrt ||
                                 inst.operands[1].value == GLSLstd450InverseSqrt));
                if ((is_div && inst.operands.size() >= 2) || is_sqrt) {
                    uint32_t ext_op = is_sqrt ? inst.operands[1].value : 0;
                    Id opd = is_sqrt ? inst.operands[2].value : inst.operands[1].value;
                    bool safe = false;
                    std::string reason = "runtime-guarded / input-dependent";

                    if (is_div && mc.zero_consts.count(opd)) {
                        report.div_sqrt_ok = false;
                        reason = "constant zero divisor";
                        report.hazards.push_back(ShaderHazard{
                            HazardKind::DefiniteDivisionByZero, fn.name, b.label, inst.result, inst.op,
                            "division by constant zero (%" + std::to_string(opd) + ")"});
                    } else if (is_sqrt && mc.float_consts.count(opd) &&
                               (mc.float_consts[opd] < 0.0 ||
                                (ext_op == GLSLstd450InverseSqrt && mc.float_consts[opd] == 0.0))) {
                        report.div_sqrt_ok = false;
                        reason = "constant negative/zero sqrt operand";
                        report.hazards.push_back(ShaderHazard{
                            ext_op == GLSLstd450InverseSqrt && mc.float_consts[opd] == 0.0
                                ? HazardKind::DefiniteDivisionByZero
                                : HazardKind::DefiniteNegativeSqrt,
                            fn.name, b.label, inst.result, inst.op,
                            "sqrt/inversesqrt on invalid constant (%" + std::to_string(opd) + ")"});
                    } else if (mc.int_consts.count(opd) && mc.int_consts[opd] != 0) {
                        safe = true;
                        reason = "non-zero integer constant";
                    } else if (mc.float_consts.count(opd) && mc.float_consts[opd] != 0.0) {
                        safe = true;
                        reason = "non-zero float constant";
                    } else if (select_operands.count(inst.result)) {
                        safe = true;
                        reason = "guarded by OpSelect ternary";
                    } else {
                        auto dit = def_inst.find(opd);
                        if (dit != def_inst.end() && dit->second) {
                            const Inst& dinst = *dit->second;
                            if (dinst.op == spv::OpExtInst && dinst.operands.size() >= 3) {
                                uint32_t deop = dinst.operands[1].value;
                                if (deop == GLSLstd450FMax || deop == GLSLstd450UMax || deop == GLSLstd450SMax ||
                                    deop == GLSLstd450FClamp || deop == GLSLstd450UClamp || deop == GLSLstd450SClamp) {
                                    safe = true;
                                    reason = "clamped via max/clamp";
                                } else if (deop == GLSLstd450Exp || deop == GLSLstd450Exp2 ||
                                           deop == GLSLstd450Length || deop == GLSLstd450Sqrt) {
                                    safe = true;
                                    reason = "positive intrinsic result";
                                }
                            } else if (dinst.op == spv::OpFAdd || dinst.op == spv::OpIAdd ||
                                       dinst.op == spv::OpFMul || dinst.op == spv::OpDot ||
                                       dinst.op == spv::OpSelect) {
                                safe = true;
                                reason = "composite / epsilon-shifted expression";
                            }
                        }
                        if (!safe) {
                            for (const auto& [h, reg] : cfg.control_dep_region) {
                                if (reg.count(bi)) {
                                    safe = true;
                                    reason = "guarded by enclosing branch";
                                    break;
                                }
                            }
                        }
                    }

                    report.divisions.push_back(DivisionAuditEntry{
                        fn.name, b.label, inst.result, inst.op, ext_op, opd, safe, reason});
                }
            }
        }
    }

    return report;
}

ComputeShaderAuditReport audit_compute_shader_binary(std::span<const uint32_t> words) {
    Module mod;
    std::string err;
    if (!parse_binary(words, mod, &err)) {
        ComputeShaderAuditReport rep;
        rep.parse_ok = false;
        rep.hazards.push_back(ShaderHazard{
            HazardKind::StructuralOrDominance, "", 0, 0, spv::OpNop, "parse_binary failed: " + err});
        return rep;
    }
    return audit_compute_shader(mod);
}

} // namespace brass::spirv
