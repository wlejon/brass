#include "test_framework.hpp"
#include <brass/brass.hpp>
#include <brass/verify/smt_equiv.hpp>
#include <brass/mir/builder.hpp>
#include <brass/target/x64/x64_isel.hpp>
#include <brass/target/target.hpp>
#include <brass/target/calling_conv.hpp>

using namespace brass;
using namespace brass::verify;

TEST_CASE("X64 ISel Equiv - Integer ALU, Shifts, and Select (i32 and i64)") {
    Module mod("x64_isel_alu");
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

    CHECK_EQ(verify_x64_isel(*f32, false).status, EquivStatus::ProvenEquivalent);
    CHECK_EQ(verify_x64_isel(*f32, true).status, EquivStatus::ProvenEquivalent);

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

    CHECK_EQ(verify_x64_isel(*f64, false).status, EquivStatus::ProvenEquivalent);
}

TEST_CASE("X64 ISel Equiv - Signed/Unsigned Div/Mod (Variable & Magic/Shift Folds)") {
    Module mod("x64_isel_div");
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

    CHECK_EQ(verify_x64_isel(*make_bin("sdiv32", Type::i32(), Opcode::sdiv)).status, EquivStatus::ProvenEquivalent);
    CHECK_EQ(verify_x64_isel(*make_bin("udiv32", Type::i32(), Opcode::udiv)).status, EquivStatus::ProvenEquivalent);
    CHECK_EQ(verify_x64_isel(*make_bin("smod32", Type::i32(), Opcode::smod)).status, EquivStatus::ProvenEquivalent);
    CHECK_EQ(verify_x64_isel(*make_bin("umod32", Type::i32(), Opcode::umod)).status, EquivStatus::ProvenEquivalent);
    CHECK_EQ(verify_x64_isel(*make_bin("sdiv64", Type::i64(), Opcode::sdiv)).status, EquivStatus::ProvenEquivalent);
    CHECK_EQ(verify_x64_isel(*make_bin("udiv64", Type::i64(), Opcode::udiv)).status, EquivStatus::ProvenEquivalent);

    // Constant power-of-two div/mod folds and magic-mul (Smulh)
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

    CHECK_EQ(verify_x64_isel(*fold_div).status, EquivStatus::ProvenEquivalent);

    Function* magic = mod.create_function("magic_sdiv_smod", Type::i64(), {Type::i64()});
    b.set_function(magic);
    BasicBlock* mb = b.append_block("entry");
    Value* raw_mx = b.add_block_param(mb, Type::i64());
    b.position_at_end(mb);
    // Keep 64-bit signed range [-32, 31] non-constant so X64ISel emits full Smulh magic
    // while keeping 64-bit SAT bit-blasting fast.
    Value* mx = b.build_ashr(b.build_shl(raw_mx, b.build_iconst_i64(58)), b.build_iconst_i64(58));
    Value* q7 = b.build_sdiv(mx, b.build_iconst_i64(7));
    Value* r7 = b.build_smod(mx, b.build_iconst_i64(7));
    b.build_ret(b.build_add(q7, r7));
    magic->rebuild_cfg_predecessors();

    EquivOptions magic_opts;
    magic_opts.max_sat_conflicts = 50000;
    CHECK_EQ(verify_x64_isel(*magic, false, magic_opts).status, EquivStatus::ProvenEquivalent);

    // 32-bit signed & unsigned non-power-of-two magic division/modulo (unbounded i32)
    for (int32_t d : {3, 7, 10, 255}) {
        Function* s32 = mod.create_function("magic_s32_" + std::to_string(d), Type::i32(), {Type::i32()});
        b.set_function(s32);
        BasicBlock* sb = b.append_block("entry");
        Value* sx = b.add_block_param(sb, Type::i32());
        b.position_at_end(sb);
        Value* dc = b.build_iconst_i32(d);
        b.build_ret(b.build_add(b.build_sdiv(sx, dc), b.build_smod(sx, dc)));
        s32->rebuild_cfg_predecessors();
        CHECK_EQ(verify_x64_isel(*s32, false, magic_opts).status, EquivStatus::ProvenEquivalent);

        Function* u32 = mod.create_function("magic_u32_" + std::to_string(d), Type::i32(), {Type::i32()});
        b.set_function(u32);
        BasicBlock* ub = b.append_block("entry");
        Value* ux = b.add_block_param(ub, Type::i32());
        b.position_at_end(ub);
        Value* udc = b.build_iconst_i32(d);
        b.build_ret(b.build_add(b.build_udiv(ux, udc), b.build_umod(ux, udc)));
        u32->rebuild_cfg_predecessors();
        CHECK_EQ(verify_x64_isel(*u32, false, magic_opts).status, EquivStatus::ProvenEquivalent);
    }

    // 64-bit unsigned non-power-of-two magic division/modulo (Umulh)
    for (int64_t d : {int64_t{3}, int64_t{7}}) {
        Function* u64 = mod.create_function("magic_u64_" + std::to_string(d), Type::i64(), {Type::i64()});
        b.set_function(u64);
        BasicBlock* ub = b.append_block("entry");
        Value* raw_ux = b.add_block_param(ub, Type::i64());
        b.position_at_end(ub);
        Value* ux = b.build_and(raw_ux, b.build_iconst_i64(63));
        Value* udc = b.build_iconst_i64(d);
        b.build_ret(b.build_add(b.build_udiv(ux, udc), b.build_umod(ux, udc)));
        u64->rebuild_cfg_predecessors();
        CHECK_EQ(verify_x64_isel(*u64, false, magic_opts).status, EquivStatus::ProvenEquivalent);
    }
}

TEST_CASE("X64 ISel Equiv - Diamond CFG and Bounded Loop") {
    Module mod("x64_isel_cfg");
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

    CHECK_EQ(verify_x64_isel(*diamond).status, EquivStatus::ProvenEquivalent);

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
    CHECK_EQ(verify_x64_isel(*loop, false, lopts).status, EquivStatus::ProvenEquivalent);
}

TEST_CASE("X64 ISel Equiv - Memory Load/Store, Alloca, Float Arithmetic, and Negative Mutation") {
    Module mod("x64_isel_mem_fp");
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
    CHECK_EQ(verify_x64_isel(*mem_fn, false, mopts).status, EquivStatus::ProvenEquivalent);

    Function* fp32 = mod.create_function("fp32_ops", Type::f32(), {Type::f32(), Type::f32()});
    b.set_function(fp32);
    BasicBlock* f32e = b.append_block("entry");
    Value* x32 = b.add_block_param(f32e, Type::f32());
    Value* y32 = b.add_block_param(f32e, Type::f32());
    b.position_at_end(f32e);
    Value* p32 = b.build_mul(x32, y32);
    Value* q32 = b.build_sdiv(b.build_add(p32, b.build_fconst_f32(1.5f)), b.build_fconst_f32(2.0f));
    b.build_ret(q32);
    fp32->rebuild_cfg_predecessors();

    CHECK_EQ(verify_x64_isel(*fp32, false).status, EquivStatus::ProvenEquivalent);
    CHECK_EQ(verify_x64_isel(*fp32, true).status, EquivStatus::ProvenEquivalent);

    Function* fp64 = mod.create_function("fp64_ops", Type::f64(), {Type::f64(), Type::f64()});
    b.set_function(fp64);
    BasicBlock* f64e = b.append_block("entry");
    Value* x64 = b.add_block_param(f64e, Type::f64());
    Value* y64 = b.add_block_param(f64e, Type::f64());
    b.position_at_end(f64e);
    b.build_ret(b.build_sub(b.build_mul(x64, y64), b.build_fconst_f64(3.25)));
    fp64->rebuild_cfg_predecessors();

    CHECK_EQ(verify_x64_isel(*fp64, false).status, EquivStatus::ProvenEquivalent);
    CHECK_EQ(verify_x64_isel(*fp64, true).status, EquivStatus::ProvenEquivalent);

    // Negative check: mutate lowered Add/Lea -> Sub in LIR and ensure Counterexample is detected
    Function* mut_fn = mod.create_function("add_one", Type::i64(), {Type::i64(), Type::i64()});
    b.set_function(mut_fn);
    BasicBlock* ae = b.append_block("entry");
    Value* u = b.add_block_param(ae, Type::i64());
    Value* v = b.add_block_param(ae, Type::i64());
    b.position_at_end(ae);
    b.build_ret(b.build_add(u, v));
    mut_fn->rebuild_cfg_predecessors();

    x64::X64ISel isel(Target::x64_linux(), CallingConvention::sysv64());
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
            if (inst->opcode == codegen::LirOpcode::Lea && !inst->uses.empty() && inst->uses[0].is_mem()) {
                auto base_op = inst->uses[0].mem_val.base_vreg.is_valid()
                    ? codegen::LirOperand::vreg(inst->uses[0].mem_val.base_vreg, 8)
                    : codegen::LirOperand::preg(inst->uses[0].mem_val.base_preg, 8);
                inst->opcode = codegen::LirOpcode::Mov;
                inst->uses.clear();
                inst->use_constraints.clear();
                inst->add_use(base_op);
                mutated = true;
                break;
            }
        }
        if (mutated) break;
    }
    REQUIRE(mutated);
    CHECK_EQ(verify_x64_isel_equivalence(*mut_fn, *lir).status, EquivStatus::Counterexample);
}

#ifdef BRASS_STANDALONE_TEST
BRASS_TEST_MAIN()
#endif
