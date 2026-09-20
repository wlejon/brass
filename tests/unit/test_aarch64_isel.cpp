#include "test_framework.hpp"
#include <brass/codegen/lir.hpp>
#include <brass/mir/builder.hpp>
#include <brass/mir/module.hpp>
#include <brass/mir/verifier.hpp>
#include <brass/target/aarch64/aarch64_isel.hpp>

using namespace brass;
using namespace brass::codegen;
using namespace brass::aarch64;
using brass::x64::Scale;

// 1. Integer Arithmetic, Logic, Neg, Not
TEST_CASE("AArch64 ISEL - Integer Arithmetic, Logic, Neg, Not") {
    Module mod;
    Function* fn = mod.create_function("test_arith", Type::i64(), {Type::i64(), Type::i64()});

    Builder b(mod);
    b.set_function(fn);

    BasicBlock* entry = b.append_block("entry");
    Value* arg0 = b.add_block_param(entry, Type::i64());
    Value* arg1 = b.add_block_param(entry, Type::i64());

    Value* v_add = b.build_add(arg0, arg1);
    Value* v_sub = b.build_sub(v_add, arg0);
    Value* v_mul = b.build_mul(v_sub, arg1);
    Value* v_and = b.build_and(v_mul, arg0);
    Value* v_or  = b.build_or(v_and, arg1);
    Value* v_xor = b.build_xor(v_or, arg0);
    Value* v_neg = b.build_neg(v_xor);
    Value* v_not = b.build_not(v_neg);
    b.build_ret(v_not);

    fn->rebuild_cfg_predecessors();
    CHECK(verify_function(*fn));

    AArch64ISel isel(Target::aarch64_linux(), CallingConvention::aapcs64());
    auto lir = isel.lower(*fn);

    CHECK(lir != nullptr);
    CHECK_EQ(lir->blocks.size(), size_t(1));
    const auto& bb = lir->blocks[0];

    bool has_add = false, has_sub = false, has_imul = false;
    bool has_and = false, has_or = false, has_xor = false;
    bool has_neg = false, has_not = false, has_ret = false;

    for (const auto& inst : bb->instructions) {
        if (inst->opcode == LirOpcode::Add) has_add = true;
        if (inst->opcode == LirOpcode::Sub) has_sub = true;
        if (inst->opcode == LirOpcode::Imul) has_imul = true;
        if (inst->opcode == LirOpcode::And) has_and = true;
        if (inst->opcode == LirOpcode::Or) has_or = true;
        if (inst->opcode == LirOpcode::Xor) has_xor = true;
        if (inst->opcode == LirOpcode::Neg) has_neg = true;
        if (inst->opcode == LirOpcode::Not) has_not = true;
        if (inst->opcode == LirOpcode::Ret) has_ret = true;
    }

    CHECK(has_add);
    CHECK(has_sub);
    CHECK(has_imul);
    CHECK(has_and);
    CHECK(has_or);
    CHECK(has_xor);
    CHECK(has_neg);
    CHECK(has_not);
    CHECK(has_ret);
}

// 2. Division and Modulo Pure 3-Register (No Fixed Constraints)
TEST_CASE("AArch64 ISEL - Division and Modulo Pure 3-Register (No Fixed Constraints)") {
    Module mod;
    Function* fn = mod.create_function("test_div_mod", Type::i64(), {Type::i64(), Type::i64()});

    Builder b(mod);
    b.set_function(fn);

    BasicBlock* entry = b.append_block("entry");
    Value* a = b.add_block_param(entry, Type::i64());
    Value* c = b.add_block_param(entry, Type::i64());

    Value* v_sdiv = b.build_sdiv(a, c);
    Value* v_smod = b.build_smod(v_sdiv, c);
    Value* v_udiv = b.build_udiv(v_smod, c);
    Value* v_umod = b.build_umod(v_udiv, c);
    b.build_ret(v_umod);

    fn->rebuild_cfg_predecessors();
    CHECK(verify_function(*fn));

    AArch64ISel isel(Target::aarch64_linux(), CallingConvention::aapcs64());
    auto lir = isel.lower(*fn);
    CHECK(lir != nullptr);

    bool found_idiv = false, found_div = false;
    bool found_mul_for_mod = false, found_sub_for_mod = false;

    for (const auto& inst : lir->blocks[0]->instructions) {
        if (inst->opcode == LirOpcode::Idiv) {
            found_idiv = true;
            // Pure 3-register: check NO fixed constraints on defs or uses
            for (const auto& constraint : inst->def_constraints) {
                CHECK(!constraint.has_fixed_preg);
            }
            for (const auto& constraint : inst->use_constraints) {
                CHECK(!constraint.has_fixed_preg);
            }
        }
        if (inst->opcode == LirOpcode::Div) {
            found_div = true;
            // Pure 3-register: check NO fixed constraints on defs or uses
            for (const auto& constraint : inst->def_constraints) {
                CHECK(!constraint.has_fixed_preg);
            }
            for (const auto& constraint : inst->use_constraints) {
                CHECK(!constraint.has_fixed_preg);
            }
        }
        if (inst->opcode == LirOpcode::Imul) {
            found_mul_for_mod = true;
        }
        if (inst->opcode == LirOpcode::Sub) {
            found_sub_for_mod = true;
        }
    }

    CHECK(found_idiv);
    CHECK(found_div);
    CHECK(found_mul_for_mod);
    CHECK(found_sub_for_mod);
}

// 3. Shifts Without Fixed RCX Constraint
TEST_CASE("AArch64 ISEL - Shifts Without Fixed RCX Constraint") {
    Module mod;
    Function* fn = mod.create_function("test_shift", Type::i64(), {Type::i64(), Type::i64()});

    Builder b(mod);
    b.set_function(fn);

    BasicBlock* entry = b.append_block("entry");
    Value* val = b.add_block_param(entry, Type::i64());
    Value* amt = b.add_block_param(entry, Type::i64());

    Value* v_shl = b.build_shl(val, amt);
    Value* v_shr = b.build_lshr(v_shl, amt);
    Value* v_sar = b.build_ashr(v_shr, amt);
    b.build_ret(v_sar);

    fn->rebuild_cfg_predecessors();
    CHECK(verify_function(*fn));

    AArch64ISel isel(Target::aarch64_linux(), CallingConvention::aapcs64());
    auto lir = isel.lower(*fn);
    CHECK(lir != nullptr);

    bool found_shl = false, found_shr = false, found_sar = false;
    for (const auto& inst : lir->blocks[0]->instructions) {
        if (inst->opcode == LirOpcode::Shl) {
            found_shl = true;
            for (const auto& constraint : inst->use_constraints) {
                CHECK(!constraint.has_fixed_preg);
            }
        }
        if (inst->opcode == LirOpcode::Shr) {
            found_shr = true;
            for (const auto& constraint : inst->use_constraints) {
                CHECK(!constraint.has_fixed_preg);
            }
        }
        if (inst->opcode == LirOpcode::Sar) {
            found_sar = true;
            for (const auto& constraint : inst->use_constraints) {
                CHECK(!constraint.has_fixed_preg);
            }
        }
    }

    CHECK(found_shl);
    CHECK(found_shr);
    CHECK(found_sar);
}

// 4. AAPCS64 Calling Convention Parameter Passing (GPR X0..X7, FPR V0..V7, Stack >8)
TEST_CASE("AArch64 ISEL - AAPCS64 Calling Convention Parameter Passing") {
    Module mod;
    std::vector<Type> param_types;
    for (int i = 0; i < 10; ++i) param_types.push_back(Type::i64());
    for (int i = 0; i < 10; ++i) param_types.push_back(Type::f64());

    Function* fn = mod.create_function("test_params", Type::i64(), param_types);
    Builder b(mod);
    b.set_function(fn);

    BasicBlock* entry = b.append_block("entry");
    std::vector<Value*> params;
    for (size_t i = 0; i < param_types.size(); ++i) {
        params.push_back(b.add_block_param(entry, param_types[i]));
    }

    b.build_ret(params[0]);
    fn->rebuild_cfg_predecessors();
    CHECK(verify_function(*fn));

    AArch64ISel isel(Target::aarch64_linux(), CallingConvention::aapcs64());
    auto lir = isel.lower(*fn);
    CHECK(lir != nullptr);

    bool found_pcopy = false;
    for (const auto& inst : lir->blocks[0]->instructions) {
        if (inst->opcode == LirOpcode::ParallelCopy) {
            found_pcopy = true;
            int gpr_reg_moves = 0;
            int fpr_reg_moves = 0;
            int stack_loads = 0;

            for (const auto& use : inst->uses) {
                if (use.is_preg()) {
                    if (use.preg_val.is_gpr()) gpr_reg_moves++;
                    if (use.preg_val.is_xmm()) fpr_reg_moves++;
                } else if (use.is_mem()) {
                    if (use.mem_val.base_preg.is_valid() && use.mem_val.base_preg.as_aarch64_gpr() == GPR::FP) {
                        stack_loads++;
                    }
                }
            }

            CHECK_EQ(gpr_reg_moves, 8);
            CHECK_EQ(fpr_reg_moves, 8);
            CHECK_EQ(stack_loads, 4); // 2 stack GPRs + 2 stack FPRs
        }
    }
    CHECK(found_pcopy);
}

// 5. Function Calls with AAPCS64 Registers and Clobber Masks
TEST_CASE("AArch64 ISEL - Function Calls with AAPCS64 Registers and Clobber Masks") {
    Module mod;
    std::vector<Type> call_param_types;
    for (int i = 0; i < 9; ++i) call_param_types.push_back(Type::i64());
    for (int i = 0; i < 9; ++i) call_param_types.push_back(Type::f64());

    Function* fn = mod.create_function("caller", Type::i64(), {});
    Builder b(mod);
    b.set_function(fn);

    b.append_block("entry");
    std::vector<Value*> call_args;
    for (int i = 0; i < 9; ++i) {
        call_args.push_back(b.build_iconst_i64(i + 1));
    }
    for (int i = 0; i < 9; ++i) {
        call_args.push_back(b.build_fconst_f64(static_cast<double>(i + 1)));
    }

    Value* res = b.build_call("callee", Type::i64(), call_args);
    b.build_ret(res);

    fn->rebuild_cfg_predecessors();
    CHECK(verify_function(*fn));

    AArch64ISel isel(Target::aarch64_linux(), CallingConvention::aapcs64());
    auto lir = isel.lower(*fn);
    CHECK(lir != nullptr);

    LirInst* call_inst = nullptr;
    for (const auto& inst : lir->blocks[0]->instructions) {
        if (inst->opcode == LirOpcode::Call) {
            call_inst = inst.get();
            break;
        }
    }

    CHECK(call_inst != nullptr);
    if (call_inst) {
        // Clobbers
        CHECK_EQ(call_inst->clobbered_gprs, 0x0007FFFFu); // X0..X18
        CHECK_EQ(call_inst->clobbered_xmms, 0xFFFF00FFu); // V0..V7, V16..V31

        // Return register X0
        CHECK_EQ(call_inst->defs.size(), size_t(1));
        CHECK(call_inst->def_constraints[0].has_fixed_preg);
        CHECK_EQ(call_inst->def_constraints[0].fixed_preg.as_aarch64_gpr(), GPR::X0);

        // Uses: 8 GPRs (X0..X7) + 8 FPRs (V0..V7)
        int gpr_call_args = 0;
        int fpr_call_args = 0;
        for (const auto& constraint : call_inst->use_constraints) {
            if (constraint.has_fixed_preg) {
                if (constraint.fixed_preg.is_gpr()) gpr_call_args++;
                if (constraint.fixed_preg.is_xmm()) fpr_call_args++;
            }
        }
        CHECK_EQ(gpr_call_args, 8);
        CHECK_EQ(fpr_call_args, 8);
    }

    // Outgoing stack space allocated for 2 spilled arguments (1 i64 + 1 f64)
    CHECK(lir->frame.outgoing_arg_space >= 16);
}

// 6. Comparisons and Conditional Branching
TEST_CASE("AArch64 ISEL - Comparisons and Conditional Branching") {
    Module mod;
    Function* fn = mod.create_function("test_cmp_br", Type::i64(), {Type::i64(), Type::i64(), Type::f64(), Type::f64()});

    Builder b(mod);
    b.set_function(fn);

    BasicBlock* entry = b.append_block("entry");
    BasicBlock* b_then = b.create_block("then");
    BasicBlock* b_else = b.create_block("else");

    Value* a = b.add_block_param(entry, Type::i64());
    Value* c = b.add_block_param(entry, Type::i64());
    Value* f1 = b.add_block_param(entry, Type::f64());
    Value* f2 = b.add_block_param(entry, Type::f64());

    Value* cond_int = b.build_slt(a, c);
    Value* cond_fp = b.build_eq(f1, f2);
    Value* cond = b.build_and(cond_int, cond_fp);

    b.build_br_if(cond, b_then, {}, b_else, {});

    fn->append_block(b_then);
    b.position_at_end(b_then);
    b.build_ret(a);

    fn->append_block(b_else);
    b.position_at_end(b_else);
    b.build_ret(c);

    fn->rebuild_cfg_predecessors();
    CHECK(verify_function(*fn));

    AArch64ISel isel(Target::aarch64_linux(), CallingConvention::aapcs64());
    auto lir = isel.lower(*fn);
    CHECK(lir != nullptr);

    bool found_cmp = false, found_ucomisd = false, found_jcc = false, found_jmp = false;
    for (const auto& bb : lir->blocks) {
        for (const auto& inst : bb->instructions) {
            if (inst->opcode == LirOpcode::Cmp) found_cmp = true;
            if (inst->opcode == LirOpcode::Ucomisd) found_ucomisd = true;
            if (inst->opcode == LirOpcode::Jcc) found_jcc = true;
            if (inst->opcode == LirOpcode::Jmp) found_jmp = true;
        }
    }

    CHECK(found_cmp);
    CHECK(found_ucomisd);
    CHECK(found_jcc);
    CHECK(found_jmp);
}

// 7. Memory Addressing Modes (Base, Indexed, Offset)
TEST_CASE("AArch64 ISEL - Memory Addressing Modes (Base, Indexed, Offset)") {
    Module mod;
    Function* fn = mod.create_function("test_mem", Type::i64(), {Type::ptr(), Type::i64()});

    Builder b(mod);
    b.set_function(fn);

    BasicBlock* entry = b.append_block("entry");
    Value* base = b.add_block_param(entry, Type::ptr());
    Value* idx = b.add_block_param(entry, Type::i64());

    // Simple load/store with offset
    Value* l1 = b.build_load(Type::i64(), base, 16);
    b.build_store(Type::i64(), base, 24, l1);

    // Indexed load/store with scale and offset
    Value* l2 = b.build_load_indexed(Type::i64(), base, idx, 8, 32);
    b.build_store_indexed(Type::i64(), base, idx, 8, 40, l2);

    b.build_ret(l2);

    fn->rebuild_cfg_predecessors();
    CHECK(verify_function(*fn));

    AArch64ISel isel(Target::aarch64_linux(), CallingConvention::aapcs64());
    auto lir = isel.lower(*fn);
    CHECK(lir != nullptr);

    bool found_load_disp = false, found_store_disp = false;
    bool found_load_idx = false, found_store_idx = false;

    for (const auto& inst : lir->blocks[0]->instructions) {
        if (inst->opcode == LirOpcode::Mov) {
            if (!inst->uses.empty() && inst->uses[0].is_mem()) {
                const auto& m = inst->uses[0].mem_val;
                if (m.disp == 16 && !m.has_index()) found_load_disp = true;
                if (m.disp == 32 && m.has_index() && m.scale == Scale::Eight) found_load_idx = true;
            }
            if (!inst->defs.empty() && inst->defs[0].is_mem()) {
                const auto& m = inst->defs[0].mem_val;
                if (m.disp == 24 && !m.has_index()) found_store_disp = true;
                if (m.disp == 40 && m.has_index() && m.scale == Scale::Eight) found_store_idx = true;
            }
        }
    }

    CHECK(found_load_disp);
    CHECK(found_store_disp);
    CHECK(found_load_idx);
    CHECK(found_store_idx);
}

// 8. Select Lowering to Cmovcc
TEST_CASE("AArch64 ISEL - Select Lowering to Cmovcc") {
    Module mod;
    Function* fn = mod.create_function("test_select", Type::i64(), {Type::i64(), Type::i64(), Type::i64()});

    Builder b(mod);
    b.set_function(fn);

    BasicBlock* entry = b.append_block("entry");
    Value* c = b.add_block_param(entry, Type::i64());
    Value* t = b.add_block_param(entry, Type::i64());
    Value* f = b.add_block_param(entry, Type::i64());

    Value* zero = b.build_iconst_i64(0);
    Value* cond = b.build_ne(c, zero);
    Value* sel = b.build_select(cond, t, f);
    b.build_ret(sel);

    fn->rebuild_cfg_predecessors();
    CHECK(verify_function(*fn));

    AArch64ISel isel(Target::aarch64_linux(), CallingConvention::aapcs64());
    auto lir = isel.lower(*fn);
    CHECK(lir != nullptr);

    bool found_cmov = false;
    for (const auto& inst : lir->blocks[0]->instructions) {
        if (inst->opcode == LirOpcode::Cmovcc) {
            found_cmov = true;
            CHECK_EQ(inst->condition, brass::x64::Condition::NE);
        }
    }
    CHECK(found_cmov);
}

// 9. Overflow Checks
TEST_CASE("AArch64 ISEL - Overflow Checks") {
    Module mod;
    Function* fn = mod.create_function("test_overflow", Type::i32(), {Type::i64(), Type::i64()});

    Builder b(mod);
    b.set_function(fn);

    BasicBlock* entry = b.append_block("entry");
    Value* x = b.add_block_param(entry, Type::i64());
    Value* y = b.add_block_param(entry, Type::i64());

    Value* s_add = b.build_sadd_overflow(x, y);
    Value* s_sub = b.build_ssub_overflow(x, y);
    Value* s_mul = b.build_smul_overflow(x, y);
    Value* u_add = b.build_uadd_overflow(x, y);
    Value* u_sub = b.build_usub_overflow(x, y);

    Value* sum1 = b.build_or(s_add, s_sub);
    Value* sum2 = b.build_or(sum1, s_mul);
    Value* sum3 = b.build_or(sum2, u_add);
    Value* sum4 = b.build_or(sum3, u_sub);
    b.build_ret(sum4);

    fn->rebuild_cfg_predecessors();
    CHECK(verify_function(*fn));

    AArch64ISel isel(Target::aarch64_linux(), CallingConvention::aapcs64());
    auto lir = isel.lower(*fn);
    CHECK(lir != nullptr);

    bool found_o = false;
    bool found_b = false;
    for (const auto& inst : lir->blocks[0]->instructions) {
        if (inst->opcode == LirOpcode::Setcc) {
            if (inst->condition == brass::x64::Condition::O) found_o = true;
            if (inst->condition == brass::x64::Condition::B) found_b = true;
        }
    }
    CHECK(found_o);
    CHECK(found_b);
}

// 10. Exception Handling and Coroutines
TEST_CASE("AArch64 ISEL - Exception Handling and Coroutines") {
    Module mod;
    Function* fn = mod.create_function("test_eh_coro", Type::i64(), {Type::i64()});

    Builder b(mod);
    b.set_function(fn);

    BasicBlock* entry = b.append_block("entry");
    Value* val = b.add_block_param(entry, Type::i64());

    Value* coro = b.build_coro_create("coro_func", {val});
    Value* resumed = b.build_coro_resume(coro, val);
    Value* suspended = b.build_coro_suspend(resumed, 1);
    b.build_coro_destroy(coro);

    b.build_ret(suspended);

    fn->rebuild_cfg_predecessors();
    CHECK(verify_function(*fn));

    AArch64ISel isel(Target::aarch64_linux(), CallingConvention::aapcs64());
    auto lir = isel.lower(*fn);
    CHECK(lir != nullptr);

    int call_count = 0;
    for (const auto& inst : lir->blocks[0]->instructions) {
        if (inst->opcode == LirOpcode::Call) {
            call_count++;
            // Each call should use X0 for first arg or return
            CHECK_EQ(inst->clobbered_gprs, 0x0007FFFFu);
        }
    }
    CHECK(call_count >= 3); // coro_create, coro_resume, coro_destroy
}

// 11. 128-bit Vector SIMD Operations
TEST_CASE("AArch64 ISEL - 128-bit Vector SIMD Operations") {
    Module mod;
    Function* fn = mod.create_function("test_simd", Type::f32x4(), {Type::f32x4(), Type::f32x4(), Type::ptr()});

    Builder b(mod);
    b.set_function(fn);

    BasicBlock* entry = b.append_block("entry");
    Value* v0 = b.add_block_param(entry, Type::f32x4());
    Value* v1 = b.add_block_param(entry, Type::f32x4());
    Value* ptr = b.add_block_param(entry, Type::ptr());

    Value* v_add = b.build_vadd(v0, v1);
    Value* v_sub = b.build_vsub(v_add, v1);
    Value* v_mul = b.build_vmul(v_sub, v0);
    b.build_vstore(Type::f32x4(), ptr, v_mul);
    Value* v_ld = b.build_vload(Type::f32x4(), ptr);

    b.build_ret(v_ld);

    fn->rebuild_cfg_predecessors();
    CHECK(verify_function(*fn));

    AArch64ISel isel(Target::aarch64_linux(), CallingConvention::aapcs64());
    auto lir = isel.lower(*fn);
    CHECK(lir != nullptr);

    bool found_addps = false, found_subps = false, found_mulps = false;
    bool found_load = false, found_store = false;

    for (const auto& inst : lir->blocks[0]->instructions) {
        if (inst->opcode == LirOpcode::Addps) found_addps = true;
        if (inst->opcode == LirOpcode::Subps) found_subps = true;
        if (inst->opcode == LirOpcode::Mulps) found_mulps = true;
        if (inst->opcode == LirOpcode::Movaps || inst->opcode == LirOpcode::Movups) {
            if (!inst->uses.empty() && inst->uses[0].is_mem()) found_load = true;
            if (!inst->defs.empty() && inst->defs[0].is_mem()) found_store = true;
        }
    }

    CHECK(found_addps);
    CHECK(found_subps);
    CHECK(found_mulps);
    CHECK(found_load);
    CHECK(found_store);
}

// 12. Stack Alloca Lowering
TEST_CASE("AArch64 ISEL - Stack Alloca Lowering") {
    Module mod;
    Function* fn = mod.create_function("test_alloca", Type::ptr(), {});

    Builder b(mod);
    b.set_function(fn);

    BasicBlock* entry = b.append_block("entry");
    Value* slot = b.build_alloca(32, 16);
    b.build_ret(slot);

    fn->rebuild_cfg_predecessors();
    CHECK(verify_function(*fn));

    AArch64ISel isel(Target::aarch64_linux(), CallingConvention::aapcs64());
    auto lir = isel.lower(*fn);
    REQUIRE(lir != nullptr);
    CHECK(lir->frame.local_frame_bytes >= 32);

    bool found_lea = false;
    for (const auto& inst : lir->blocks[0]->instructions) {
        if (inst->opcode == LirOpcode::Lea) {
            found_lea = true;
            REQUIRE(!inst->uses.empty());
            CHECK(inst->uses[0].is_local_slot());
            CHECK_EQ(inst->uses[0].size, 8);
        }
    }
    CHECK(found_lea);
}

