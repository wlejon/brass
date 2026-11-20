#include "test_framework.hpp"
#include <brass/brass.hpp>
#include <brass/verify/smt_equiv.hpp>
#include <brass/mir/builder.hpp>
#include <brass/target/aarch64/aarch64_isel.hpp>
#include <brass/target/target.hpp>
#include <brass/target/calling_conv.hpp>

using namespace brass;
using namespace brass::verify;

TEST_CASE("ARM64 ISel Equiv - Integer ALU, Shifts, Select, and Stack Args") {
    Module mod("arm64_isel_alu");
    Builder b(mod);

    Function* f32 = mod.create_function("alu32", Type::i32(), {Type::i32(), Type::i32(), Type::i32()});
    b.set_function(f32);
    BasicBlock* e32 = b.append_block("entry");
    Value* a0 = b.add_block_param(e32, Type::i32());
    Value* a1 = b.add_block_param(e32, Type::i32());
    Value* a2 = b.add_block_param(e32, Type::i32());
    b.position_at_end(e32);
    Value* sh = b.build_and(a2, b.build_iconst_i32(15));
    Value* sum = b.build_add(b.build_mul(a0, a1), b.build_shl(a0, sh));
    Value* diff = b.build_sub(sum, b.build_ashr(a1, b.build_iconst_i32(2)));
    Value* cond = b.build_slt(a0, a1);
    Value* sel = b.build_select(cond, diff, b.build_xor(a0, a1));
    b.build_ret(sel);
    f32->rebuild_cfg_predecessors();

    CHECK_EQ(verify_arm64_isel(*f32).status, EquivStatus::ProvenEquivalent);

    Function* f64 = mod.create_function("alu64", Type::i64(), {Type::i64(), Type::i64(), Type::i64()});
    b.set_function(f64);
    BasicBlock* e64 = b.append_block("entry");
    Value* b0 = b.add_block_param(e64, Type::i64());
    Value* b1 = b.add_block_param(e64, Type::i64());
    Value* b2 = b.add_block_param(e64, Type::i64());
    b.position_at_end(e64);
    Value* mul6 = b.build_mul(b0, b.build_iconst_i64(7));
    Value* sum6 = b.build_add(mul6, b.build_lshr(b1, b.build_iconst_i64(3)));
    Value* cmp6 = b.build_sgt(b0, b1);
    b.build_ret(b.build_select(cmp6, sum6, b.build_sub(b0, b2)));
    f64->rebuild_cfg_predecessors();

    CHECK_EQ(verify_arm64_isel(*f64).status, EquivStatus::ProvenEquivalent);

    // > 8 scalar arguments to exercise Apple AAPCS64 stack-passed parameter lowering (-1 - caller_offset)
    std::vector<Type> ten_i32(10, Type::i32());
    Function* stack_fn = mod.create_function("stack_args10", Type::i32(), ten_i32);
    b.set_function(stack_fn);
    BasicBlock* se = b.append_block("entry");
    Value* pacc = b.add_block_param(se, Type::i32());
    std::vector<Value*> ps{pacc};
    for (int i = 1; i < 10; ++i) ps.push_back(b.add_block_param(se, Type::i32()));
    b.position_at_end(se);
    Value* tot = pacc;
    for (int i = 1; i < 10; ++i) tot = b.build_add(tot, ps[static_cast<size_t>(i)]);
    b.build_ret(tot);
    stack_fn->rebuild_cfg_predecessors();

    CHECK_EQ(verify_arm64_isel(*stack_fn).status, EquivStatus::ProvenEquivalent);
}

TEST_CASE("ARM64 ISel Equiv - Signed/Unsigned Div/Mod (Variable & Magic/Shift Folds)") {
    Module mod("arm64_isel_div");
    Builder b(mod);

    auto make_bin = [&](const char* name, Type ty, Opcode op) -> Function* {
        Function* fn = mod.create_function(name, ty, {ty, ty});
        b.set_function(fn);
        BasicBlock* bb = b.append_block("entry");
        Value* x = b.add_block_param(bb, ty);
        Value* y = b.add_block_param(bb, ty);
        b.position_at_end(bb);
        Value* r = nullptr;
        switch (op) {
            case Opcode::sdiv: r = b.build_sdiv(x, y); break;
            case Opcode::udiv: r = b.build_udiv(x, y); break;
            case Opcode::smod: r = b.build_smod(x, y); break;
            case Opcode::umod: r = b.build_umod(x, y); break;
            default: break;
        }
        b.build_ret(r);
        fn->rebuild_cfg_predecessors();
        return fn;
    };

    CHECK_EQ(verify_arm64_isel(*make_bin("sdiv32", Type::i32(), Opcode::sdiv)).status, EquivStatus::ProvenEquivalent);
    CHECK_EQ(verify_arm64_isel(*make_bin("udiv32", Type::i32(), Opcode::udiv)).status, EquivStatus::ProvenEquivalent);
    CHECK_EQ(verify_arm64_isel(*make_bin("smod32", Type::i32(), Opcode::smod)).status, EquivStatus::ProvenEquivalent);
    CHECK_EQ(verify_arm64_isel(*make_bin("umod32", Type::i32(), Opcode::umod)).status, EquivStatus::ProvenEquivalent);
    CHECK_EQ(verify_arm64_isel(*make_bin("sdiv64", Type::i64(), Opcode::sdiv)).status, EquivStatus::ProvenEquivalent);
    CHECK_EQ(verify_arm64_isel(*make_bin("udiv64", Type::i64(), Opcode::udiv)).status, EquivStatus::ProvenEquivalent);
    CHECK_EQ(verify_arm64_isel(*make_bin("smod64", Type::i64(), Opcode::smod)).status, EquivStatus::ProvenEquivalent);
    CHECK_EQ(verify_arm64_isel(*make_bin("umod64", Type::i64(), Opcode::umod)).status, EquivStatus::ProvenEquivalent);

    Function* fold_div = mod.create_function("fold_div", Type::i64(), {Type::i64()});
    b.set_function(fold_div);
    BasicBlock* fb = b.append_block("entry");
    Value* fx = b.add_block_param(fb, Type::i64());
    b.position_at_end(fb);
    Value* fq8 = b.build_udiv(fx, b.build_iconst_i64(8));
    Value* fr8 = b.build_umod(fx, b.build_iconst_i64(8));
    Value* fsq4 = b.build_sdiv(fx, b.build_iconst_i64(4));
    b.build_ret(b.build_add(b.build_add(fq8, fr8), fsq4));
    fold_div->rebuild_cfg_predecessors();

    CHECK_EQ(verify_arm64_isel(*fold_div).status, EquivStatus::ProvenEquivalent);

    Function* magic = mod.create_function("magic_sdiv_smod", Type::i64(), {Type::i64()});
    b.set_function(magic);
    BasicBlock* mb = b.append_block("entry");
    Value* raw_mx = b.add_block_param(mb, Type::i64());
    b.position_at_end(mb);
    Value* mx = b.build_ashr(b.build_shl(raw_mx, b.build_iconst_i64(58)), b.build_iconst_i64(58));
    Value* q7 = b.build_sdiv(mx, b.build_iconst_i64(7));
    Value* r7 = b.build_smod(mx, b.build_iconst_i64(7));
    b.build_ret(b.build_add(q7, r7));
    magic->rebuild_cfg_predecessors();

    EquivOptions magic_opts;
    magic_opts.max_sat_conflicts = 50000;
    CHECK_EQ(verify_arm64_isel(*magic, magic_opts).status, EquivStatus::ProvenEquivalent);
}

TEST_CASE("ARM64 ISel Equiv - Diamond CFG, Bounded Loop, and Switch") {
    Module mod("arm64_isel_cfg");
    Builder b(mod);

    Function* diamond = mod.create_function("diamond", Type::i64(), {Type::i64(), Type::i64()});
    b.set_function(diamond);
    BasicBlock* entry = b.append_block("entry");
    BasicBlock* left  = b.append_block("left");
    BasicBlock* right = b.append_block("right");
    BasicBlock* join  = b.append_block("join");
    Value* x = b.add_block_param(entry, Type::i64());
    Value* y = b.add_block_param(entry, Type::i64());
    Value* jv = b.add_block_param(join, Type::i64());

    b.position_at_end(entry);
    b.build_br_if(b.build_slt(x, y), left, {}, right, {});
    b.position_at_end(left);
    b.build_br(join, {b.build_sub(y, x)});
    b.position_at_end(right);
    b.build_br(join, {b.build_add(x, b.build_iconst_i64(10))});
    b.position_at_end(join);
    b.build_ret(jv);
    diamond->rebuild_cfg_predecessors();

    CHECK_EQ(verify_arm64_isel(*diamond).status, EquivStatus::ProvenEquivalent);

    Function* loop = mod.create_function("sum4", Type::i32(), {Type::i32()});
    b.set_function(loop);
    BasicBlock* le = b.append_block("entry");
    BasicBlock* lh = b.append_block("head");
    BasicBlock* lb = b.append_block("body");
    BasicBlock* ld = b.append_block("done");
    Value* base = b.add_block_param(le, Type::i32());
    Value* li = b.add_block_param(lh, Type::i32());
    Value* lacc = b.add_block_param(lh, Type::i32());

    b.position_at_end(le);
    b.build_br(lh, {b.build_iconst_i32(0), base});
    b.position_at_end(lh);
    b.build_br_if(b.build_slt(li, b.build_iconst_i32(4)), lb, {}, ld, {});
    b.position_at_end(lb);
    b.build_br(lh, {b.build_add(li, b.build_iconst_i32(1)), b.build_add(lacc, li)});
    b.position_at_end(ld);
    b.build_ret(lacc);
    loop->rebuild_cfg_predecessors();

    EquivOptions lopts;
    lopts.loop_unroll_bound = 4;
    CHECK_EQ(verify_arm64_isel(*loop, lopts).status, EquivStatus::ProvenEquivalent);

    Function* sw = mod.create_function("sw3", Type::i32(), {Type::i32(), Type::i32()});
    b.set_function(sw);
    BasicBlock* se = b.append_block("entry");
    BasicBlock* c0 = b.append_block("c0");
    BasicBlock* c1 = b.append_block("c1");
    BasicBlock* cd = b.append_block("cdef");
    Value* sv = b.add_block_param(se, Type::i32());
    Value* sa = b.add_block_param(se, Type::i32());
    b.position_at_end(se);
    b.build_switch(sv, cd, {{0, c0}, {1, c1}});
    b.position_at_end(c0);
    b.build_ret(b.build_add(sa, b.build_iconst_i32(10)));
    b.position_at_end(c1);
    b.build_ret(b.build_add(sa, b.build_iconst_i32(20)));
    b.position_at_end(cd);
    b.build_ret(sa);
    sw->rebuild_cfg_predecessors();

    CHECK_EQ(verify_arm64_isel(*sw).status, EquivStatus::ProvenEquivalent);
}

TEST_CASE("ARM64 ISel Equiv - Memory, Float Arithmetic, NEON v128, and Negative Mutations") {
    Module mod("arm64_isel_mem_fp_vec");
    Builder b(mod);

    Function* mem_fn = mod.create_function("mem_swap_add", Type::i64(), {Type::ptr(), Type::i64()});
    b.set_function(mem_fn);
    BasicBlock* me = b.append_block("entry");
    Value* ptr = b.add_block_param(me, Type::ptr());
    Value* inc = b.add_block_param(me, Type::i64());
    b.position_at_end(me);
    Value* local = b.build_alloca(16, 8);
    Value* v0 = b.build_load(Type::i64(), ptr, 0);
    Value* v1 = b.build_load_indexed(Type::i64(), ptr, b.build_iconst_i64(1), 8, 0);
    Value* sum = b.build_add(v0, inc);
    b.build_store(Type::i64(), local, 0, sum);
    b.build_store(Type::i64(), ptr, 8, b.build_add(v1, inc));
    Value* reloaded = b.build_load(Type::i64(), local, 0);
    b.build_ret(reloaded);
    mem_fn->rebuild_cfg_predecessors();

    EquivOptions mopts;
    mopts.check_memory = true;
    CHECK_EQ(verify_arm64_isel(*mem_fn, mopts).status, EquivStatus::ProvenEquivalent);

    Function* fp32 = mod.create_function("fp32_ops", Type::f32(), {Type::f32(), Type::f32()});
    b.set_function(fp32);
    BasicBlock* f32e = b.append_block("entry");
    Value* x32 = b.add_block_param(f32e, Type::f32());
    Value* y32 = b.add_block_param(f32e, Type::f32());
    b.position_at_end(f32e);
    Value* fma32 = b.build_fma_f32(x32, y32, b.build_fconst_f32(1.5f));
    Value* q32 = b.build_sdiv(b.build_floor_f32(fma32), b.build_fconst_f32(2.0f));
    b.build_ret(q32);
    fp32->rebuild_cfg_predecessors();

    CHECK_EQ(verify_arm64_isel(*fp32).status, EquivStatus::ProvenEquivalent);

    // 128-bit NEON vector arithmetic, FMA, rounding, conversion, compare, select
    Function* vec_f32 = mod.create_function(
        "neon_f32x4", Type::f32x4(), {Type::f32x4(), Type::f32x4(), Type::f32x4()});
    b.set_function(vec_f32);
    BasicBlock* vfe = b.append_block("entry");
    Value* vx = b.add_block_param(vfe, Type::f32x4());
    Value* vy = b.add_block_param(vfe, Type::f32x4());
    Value* vz = b.add_block_param(vfe, Type::f32x4());
    b.position_at_end(vfe);
    Value* vfm = b.build_vfma(vx, vy, b.build_vround(vz));
    Value* vcmp = b.build_vcmp(Opcode::vcmp_lt, vx, vy);
    Value* vsel = b.build_vselect(vcmp, b.build_vfloor(vfm), b.build_vtrunc(b.build_vadd(vx, vy)));
    b.build_ret(vsel);
    vec_f32->rebuild_cfg_predecessors();

    CHECK_EQ(verify_arm64_isel(*vec_f32).status, EquivStatus::ProvenEquivalent);

    Function* vec_i32 = mod.create_function("neon_i32x4", Type::i32x4(), {Type::f32x4(), Type::i32x4()});
    b.set_function(vec_i32);
    BasicBlock* vie = b.append_block("entry");
    Value* vf = b.add_block_param(vie, Type::f32x4());
    Value* vi = b.add_block_param(vie, Type::i32x4());
    b.position_at_end(vie);
    Value* cvt = b.build_vconvert(Type::i32x4(), vf);
    Value* sh1 = b.build_vshl(cvt, b.build_vbroadcast(Type::i32x4(), b.build_iconst_i32(2)));
    Value* sh2 = b.build_vashr(vi, b.build_vbroadcast(Type::i32x4(), b.build_iconst_i32(1)));
    b.build_ret(b.build_vadd(sh1, sh2));
    vec_i32->rebuild_cfg_predecessors();

    CHECK_EQ(verify_arm64_isel(*vec_i32).status, EquivStatus::ProvenEquivalent);

    Function* vec_mem = mod.create_function("neon_vload_vstore", Type::i32(), {Type::ptr(), Type::ptr()});
    b.set_function(vec_mem);
    BasicBlock* vme = b.append_block("entry");
    Value* p_in = b.add_block_param(vme, Type::ptr());
    Value* p_out = b.add_block_param(vme, Type::ptr());
    b.position_at_end(vme);
    Value* ld0 = b.build_vload(Type::i32x4(), p_in, 0);
    Value* ld1 = b.build_vload(Type::i32x4(), p_in, 16);
    Value* sumv = b.build_vadd(ld0, ld1);
    b.build_vstore(Type::i32x4(), p_out, 0, sumv);
    Value* negs = b.build_vcmp(Opcode::vcmp_lt, sumv, b.build_vzero(Type::i32x4()));
    b.build_ret(b.build_vany(negs));
    vec_mem->rebuild_cfg_predecessors();

    CHECK_EQ(verify_arm64_isel(*vec_mem, mopts).status, EquivStatus::ProvenEquivalent);

    // Negative check 1: mutate lowered Add -> Sub in scalar LIR -> Counterexample
    Function* mut_fn = mod.create_function("add_two", Type::i64(), {Type::i64(), Type::i64()});
    b.set_function(mut_fn);
    BasicBlock* ae = b.append_block("entry");
    Value* u = b.add_block_param(ae, Type::i64());
    Value* v = b.add_block_param(ae, Type::i64());
    b.position_at_end(ae);
    b.build_ret(b.build_add(u, v));
    mut_fn->rebuild_cfg_predecessors();

    aarch64::AArch64ISel isel(Target::aarch64_macos(), CallingConvention::apple_aapcs64());
    auto lir = isel.lower(*mut_fn);
    REQUIRE(lir != nullptr);
    bool mutated = false;
    for (auto& blk : lir->blocks) {
        for (auto& inst : blk->instructions) {
            if (inst->opcode == codegen::LirOpcode::Add || inst->opcode == codegen::LirOpcode::Adds) {
                inst->opcode = codegen::LirOpcode::Sub;
                mutated = true;
                break;
            }
        }
        if (mutated) break;
    }
    REQUIRE(mutated);
    CHECK_EQ(verify_arm64_isel_equivalence(*mut_fn, *lir).status, EquivStatus::Counterexample);

    // Negative check 2: mutate lowered NEON Addps -> Subps -> Counterexample
    Function* mut_vec = mod.create_function("vadd_two", Type::f32x4(), {Type::f32x4(), Type::f32x4()});
    b.set_function(mut_vec);
    BasicBlock* mve = b.append_block("entry");
    Value* va = b.add_block_param(mve, Type::f32x4());
    Value* vb = b.add_block_param(mve, Type::f32x4());
    b.position_at_end(mve);
    b.build_ret(b.build_vadd(va, vb));
    mut_vec->rebuild_cfg_predecessors();

    auto vec_lir = isel.lower(*mut_vec);
    REQUIRE(vec_lir != nullptr);
    bool vec_mutated = false;
    for (auto& blk : vec_lir->blocks) {
        for (auto& inst : blk->instructions) {
            if (inst->opcode == codegen::LirOpcode::Addps || inst->opcode == codegen::LirOpcode::Vaddps) {
                inst->opcode = codegen::LirOpcode::Subps;
                vec_mutated = true;
                break;
            }
        }
        if (vec_mutated) break;
    }
    REQUIRE(vec_mutated);
    CHECK_EQ(verify_arm64_isel_equivalence(*mut_vec, *vec_lir).status, EquivStatus::Counterexample);

    // Negative check 3: invert lowered Jcc condition -> Counterexample
    Function* mut_cfg = mod.create_function("diamond_mut", Type::i64(), {Type::i64(), Type::i64()});
    b.set_function(mut_cfg);
    BasicBlock* mde = b.append_block("entry");
    BasicBlock* mdt = b.append_block("then");
    BasicBlock* mdf = b.append_block("else");
    BasicBlock* mdm = b.append_block("merge");
    Value* da = b.add_block_param(mde, Type::i64());
    Value* db = b.add_block_param(mde, Type::i64());
    Value* dm_v = b.add_block_param(mdm, Type::i64());
    b.position_at_end(mde);
    b.build_br_if(b.build_slt(da, db), mdt, mdf);
    b.position_at_end(mdt);
    b.build_br(mdm, {b.build_add(da, db)});
    b.position_at_end(mdf);
    b.build_br(mdm, {b.build_sub(da, db)});
    b.position_at_end(mdm);
    b.build_ret(dm_v);
    mut_cfg->rebuild_cfg_predecessors();

    auto cfg_lir = isel.lower(*mut_cfg);
    REQUIRE(cfg_lir != nullptr);
    bool br_mutated = false;
    for (auto& blk : cfg_lir->blocks) {
        for (auto& inst : blk->instructions) {
            if (inst->opcode == codegen::LirOpcode::Jcc) {
                inst->condition = x64::Condition::GE;
                br_mutated = true;
                break;
            }
        }
        if (br_mutated) break;
    }
    REQUIRE(br_mutated);
    CHECK_EQ(verify_arm64_isel_equivalence(*mut_cfg, *cfg_lir).status, EquivStatus::Counterexample);
}

#ifdef BRASS_STANDALONE_TEST
BRASS_TEST_MAIN()
#endif
