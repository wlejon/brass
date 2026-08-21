#include "test_framework.hpp"
#include <brass/mir/module.hpp>
#include <brass/mir/builder.hpp>
#include <brass/mir/verifier.hpp>
#include <brass/target/x64/x64_isel.hpp>
#include <brass/codegen/lir.hpp>

using namespace brass;
using namespace brass::codegen;
using namespace brass::x64;

TEST_CASE("ISEL - Integer Arithmetic and Logic Lowering") {
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

    X64ISel isel(Target::x64_windows(), CallingConvention::win64());
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

TEST_CASE("ISEL - Division and Modulo Fixed Register Constraints") {
    Module mod;
    Function* fn = mod.create_function("test_div", Type::i64(), {Type::i64(), Type::i64()});

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

    X64ISel isel(Target::x64_windows(), CallingConvention::win64());
    auto lir = isel.lower(*fn);
    CHECK(lir != nullptr);

    bool found_idiv = false, found_div = false, found_cqo = false;
    for (const auto& inst : lir->blocks[0]->instructions) {
        if (inst->opcode == LirOpcode::Cqo) {
            found_cqo = true;
            CHECK_EQ(inst->defs[0].preg_val.as_gpr(), GPR::RDX);
            CHECK_EQ(inst->uses[0].preg_val.as_gpr(), GPR::RAX);
        }
        if (inst->opcode == LirOpcode::Idiv) {
            found_idiv = true;
            CHECK_EQ(inst->defs[0].preg_val.as_gpr(), GPR::RAX);
            CHECK_EQ(inst->defs[1].preg_val.as_gpr(), GPR::RDX);
            CHECK_EQ(inst->uses[0].preg_val.as_gpr(), GPR::RAX);
            CHECK_EQ(inst->uses[1].preg_val.as_gpr(), GPR::RDX);
        }
        if (inst->opcode == LirOpcode::Div) {
            found_div = true;
            CHECK_EQ(inst->defs[0].preg_val.as_gpr(), GPR::RAX);
            CHECK_EQ(inst->defs[1].preg_val.as_gpr(), GPR::RDX);
        }
    }

    CHECK(found_cqo);
    CHECK(found_idiv);
    CHECK(found_div);
}

TEST_CASE("ISEL - Variable Shifts with Fixed RCX Constraint") {
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

    X64ISel isel(Target::x64_windows(), CallingConvention::win64());
    auto lir = isel.lower(*fn);
    CHECK(lir != nullptr);

    bool found_shl = false, found_shr = false, found_sar = false;
    for (const auto& inst : lir->blocks[0]->instructions) {
        if (inst->opcode == LirOpcode::Shl) {
            found_shl = true;
            CHECK_EQ(inst->uses.back().preg_val.as_gpr(), GPR::RCX);
        }
        if (inst->opcode == LirOpcode::Shr) {
            found_shr = true;
            CHECK_EQ(inst->uses.back().preg_val.as_gpr(), GPR::RCX);
        }
        if (inst->opcode == LirOpcode::Sar) {
            found_sar = true;
            CHECK_EQ(inst->uses.back().preg_val.as_gpr(), GPR::RCX);
        }
    }

    CHECK(found_shl);
    CHECK(found_shr);
    CHECK(found_sar);
}

TEST_CASE("ISEL - Bit Counting (clz, ctz, popcnt)") {
    Module mod;
    Function* fn = mod.create_function("test_bitops", Type::i64(), {Type::i64()});

    Builder b(mod);
    b.set_function(fn);

    BasicBlock* entry = b.append_block("entry");
    Value* x = b.add_block_param(entry, Type::i64());
    Value* v_clz = b.build_clz(x);
    Value* v_ctz = b.build_ctz(v_clz);
    Value* v_pop = b.build_popcnt(v_ctz);
    b.build_ret(v_pop);

    fn->rebuild_cfg_predecessors();
    CHECK(verify_function(*fn));

    X64ISel isel(Target::x64_windows(), CallingConvention::win64());
    auto lir = isel.lower(*fn);
    CHECK(lir != nullptr);

    bool has_lzcnt = false, has_tzcnt = false, has_popcnt = false;
    for (const auto& inst : lir->blocks[0]->instructions) {
        if (inst->opcode == LirOpcode::Lzcnt) has_lzcnt = true;
        if (inst->opcode == LirOpcode::Tzcnt) has_tzcnt = true;
        if (inst->opcode == LirOpcode::Popcnt) has_popcnt = true;
    }

    CHECK(has_lzcnt);
    CHECK(has_tzcnt);
    CHECK(has_popcnt);
}

TEST_CASE("ISEL - Memory Addressing Modes (Base, Indexed, Offset)") {
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

    X64ISel isel(Target::x64_windows(), CallingConvention::win64());
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

TEST_CASE("ISEL - Floating Point Arithmetic and Conversions") {
    Module mod;
    Function* fn = mod.create_function("test_fp", Type::f64(), {Type::f64(), Type::i64()});

    Builder b(mod);
    b.set_function(fn);

    BasicBlock* entry = b.append_block("entry");
    Value* f = b.add_block_param(entry, Type::f64());
    Value* i = b.add_block_param(entry, Type::i64());

    Value* f_const = b.build_fconst_f64(3.14159);
    Value* f_add = b.build_add(f, f_const);
    Value* f_sub = b.build_sub(f_add, f);
    Value* f_mul = b.build_mul(f_sub, f_const);
    Value* f_div = b.build_sdiv(f_mul, f_const); // floating point division
    Value* i_from_f = b.build_fptosi_i64(f_div);
    Value* f_from_i = b.build_sitofp_f64_i64(i_from_f);
    Value* f_final = b.build_add(f_from_i, f_div);
    b.build_ret(f_final);

    fn->rebuild_cfg_predecessors();
    CHECK(verify_function(*fn));

    X64ISel isel(Target::x64_windows(), CallingConvention::win64());
    auto lir = isel.lower(*fn);
    CHECK(lir != nullptr);

    bool has_addsd = false, has_subsd = false, has_mulsd = false, has_divsd = false;
    bool has_cvttsd2si = false, has_cvtsi2sd = false;

    for (const auto& inst : lir->blocks[0]->instructions) {
        if (inst->opcode == LirOpcode::Addsd) has_addsd = true;
        if (inst->opcode == LirOpcode::Subsd) has_subsd = true;
        if (inst->opcode == LirOpcode::Mulsd) has_mulsd = true;
        if (inst->opcode == LirOpcode::Divsd) has_divsd = true;
        if (inst->opcode == LirOpcode::Cvttsd2si) has_cvttsd2si = true;
        if (inst->opcode == LirOpcode::Cvtsi2sd) has_cvtsi2sd = true;
    }

    CHECK(has_addsd);
    CHECK(has_subsd);
    CHECK(has_mulsd);
    CHECK(has_divsd);
    CHECK(has_cvttsd2si);
    CHECK(has_cvtsi2sd);
}

TEST_CASE("ISEL - Function Calls under Win64 and SysV") {
    Module mod;
    Function* fn = mod.create_function("caller", Type::i64(), {Type::i64(), Type::i64(), Type::i64(), Type::i64(), Type::i64()});

    Builder b(mod);
    b.set_function(fn);

    BasicBlock* entry = b.append_block("entry");
    Value* p0 = b.add_block_param(entry, Type::i64());
    Value* p1 = b.add_block_param(entry, Type::i64());
    Value* p2 = b.add_block_param(entry, Type::i64());
    Value* p3 = b.add_block_param(entry, Type::i64());
    Value* p4 = b.add_block_param(entry, Type::i64());

    Value* call_res = b.build_call("target_callee", Type::i64(), {p0, p1, p2, p3, p4});
    b.build_ret(call_res);

    fn->rebuild_cfg_predecessors();
    CHECK(verify_function(*fn));

    // Win64
    {
        X64ISel isel_win(Target::x64_windows(), CallingConvention::win64());
        auto lir_win = isel_win.lower(*fn);
        CHECK(lir_win != nullptr);
        CHECK(lir_win->frame.outgoing_arg_space >= size_t(32 + 8)); // 5 args = 32 shadow + 8 bytes 5th arg
    }

    // SysV
    {
        X64ISel isel_sysv(Target::x64_linux(), CallingConvention::sysv64());
        auto lir_sysv = isel_sysv.lower(*fn);
        CHECK(lir_sysv != nullptr);
        CHECK_EQ(lir_sysv->frame.outgoing_arg_space, size_t(0)); // 5 args fit in 6 GPRs
    }
}
