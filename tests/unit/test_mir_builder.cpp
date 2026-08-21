#include "test_framework.hpp"
#include <brass/mir/module.hpp>
#include <brass/mir/builder.hpp>
#include <brass/mir/verifier.hpp>

using namespace brass;

TEST_CASE("Builder simple add function") {
    Module mod("math_module");
    Function* fn = mod.create_function("add2", Type::i32(), {Type::i32(), Type::i32()});

    Builder b(mod);
    b.set_function(fn);

    BasicBlock* entry = b.append_block("entry");
    Value* a = b.add_block_param(entry, Type::i32());
    Value* c = b.add_block_param(entry, Type::i32());

    Value* sum = b.build_add(a, c);
    b.build_ret(sum);

    fn->rebuild_cfg_predecessors();

    DiagnosticReporter diag;
    bool ok = verify_module(mod, &diag);
    if (!ok) {
        std::cout << diag.format_all();
    }
    CHECK(ok);
}

TEST_CASE("Builder fibonacci function with branches and calls") {
    Module mod("fib_module");
    Function* fn = mod.create_function("fibonacci", Type::i32(), {Type::i32()});

    Builder b(mod);
    b.set_function(fn);

    BasicBlock* bb0 = b.append_block("bb0");
    BasicBlock* bb_base = b.create_block("bb_base");
    BasicBlock* bb_rec = b.create_block("bb_rec");

    fn->append_block(bb_base);
    fn->append_block(bb_rec);

    // Entry block bb0
    Value* n = b.add_block_param(bb0, Type::i32());
    Value* c2 = b.build_iconst_i32(2);
    Value* cond = b.build_slt(n, c2);
    b.build_br_if(cond, bb_base, bb_rec);

    // bb_base: ret n
    b.position_at_end(bb_base);
    b.build_ret(n);

    // bb_rec: recursive calls
    b.position_at_end(bb_rec);
    Value* c1 = b.build_iconst_i32(1);
    Value* n1 = b.build_sub(n, c1);
    Value* r1 = b.build_call("fibonacci", Type::i32(), {n1});
    Value* n2 = b.build_sub(n, c2);
    Value* r2 = b.build_call("fibonacci", Type::i32(), {n2});
    Value* sum = b.build_add(r1, r2);
    b.build_ret(sum);

    fn->rebuild_cfg_predecessors();

    DiagnosticReporter diag;
    bool ok = verify_module(mod, &diag);
    if (!ok) {
        std::cout << diag.format_all();
    }
    CHECK(ok);
}

TEST_CASE("Builder loop with block parameters (sum 1 to N)") {
    Module mod("loop_module");
    Function* fn = mod.create_function("sum_to_n", Type::i64(), {Type::i32()});

    Builder b(mod);
    b.set_function(fn);

    BasicBlock* entry = b.append_block("entry");
    BasicBlock* header = b.create_block("loop_header");
    BasicBlock* body = b.create_block("loop_body");
    BasicBlock* exit = b.create_block("loop_exit");

    fn->append_block(header);
    fn->append_block(body);
    fn->append_block(exit);

    // entry: n parameter
    b.position_at_end(entry);
    Value* n = b.add_block_param(entry, Type::i32());
    Value* init_i = b.build_iconst_i32(1);
    Value* init_sum = b.build_iconst_i64(0);
    b.build_br(header, {init_i, init_sum});

    // header: loop phi arguments (i: i32, sum: i64)
    b.position_at_end(header);
    Value* cur_i = b.add_block_param(header, Type::i32());
    Value* cur_sum = b.add_block_param(header, Type::i64());
    Value* cmp = b.build_sle(cur_i, n);
    b.build_br_if(cmp, body, {}, exit, {cur_sum});

    // body: add to sum, increment i, loop back
    b.position_at_end(body);
    Value* i_ext = b.build_sext_i64(cur_i);
    Value* next_sum = b.build_add(cur_sum, i_ext);
    Value* one = b.build_iconst_i32(1);
    Value* next_i = b.build_add(cur_i, one);
    b.build_br(header, {next_i, next_sum});

    // exit: takes final sum
    b.position_at_end(exit);
    Value* final_sum = b.add_block_param(exit, Type::i64());
    b.build_ret(final_sum);

    fn->rebuild_cfg_predecessors();

    DiagnosticReporter diag;
    bool ok = verify_module(mod, &diag);
    if (!ok) {
        std::cout << diag.format_all();
    }
    CHECK(ok);
}

TEST_CASE("Builder memory operations and speculation primitives") {
    Module mod("memory_and_spec_module");
    Function* fn = mod.create_function("test_mem_spec", Type::void_type(), {Type::ptr(), Type::gcref()});

    Builder b(mod);
    b.set_function(fn);

    BasicBlock* entry = b.append_block("entry");
    Value* raw_ptr = b.add_block_param(entry, Type::ptr());
    Value* gc_ref = b.add_block_param(entry, Type::gcref());

    // Memory load & store
    Value* loaded32 = b.build_load(Type::i32(), raw_ptr, 16);
    b.build_store(Type::i32(), raw_ptr, 20, loaded32);

    // Indexed load & store
    Value* idx = b.build_iconst_i64(2);
    Value* loaded_elem = b.build_load_indexed(Type::i64(), gc_ref, idx, 8, 32);
    b.build_store_indexed(Type::i64(), gc_ref, idx, 8, 32, loaded_elem);

    // Speculation guard and safepoint
    Value* cond = b.build_iconst_i32(1);
    b.build_guard(cond, "deopt_stub_exit_1", {raw_ptr, gc_ref, loaded32});
    b.build_safepoint();

    // Patchable const & patchable call
    Value* patch_const = b.build_patchable_const_i32("ic_slot_0", 123);
    b.build_patchable_call("call_site_1", "default_target_stub", Type::void_type(), {patch_const});

    b.build_ret_void();

    fn->rebuild_cfg_predecessors();

    DiagnosticReporter diag;
    bool ok = verify_module(mod, &diag);
    if (!ok) {
        std::cout << diag.format_all();
    }
    CHECK(ok);
}

TEST_CASE("Builder all arithmetic, logic, conversions, bitcasts and comparisons") {
    Module mod("comprehensive_ops_mod");
    Function* fn = mod.create_function("all_ops", Type::void_type(), {Type::i32(), Type::i64(), Type::f64(), Type::ptr()});

    Builder b(mod);
    b.set_function(fn);

    BasicBlock* entry = b.append_block("entry");
    Value* i32_val = b.add_block_param(entry, Type::i32());
    Value* i64_val = b.add_block_param(entry, Type::i64());
    Value* f64_val = b.add_block_param(entry, Type::f64());
    Value* ptr_val = b.add_block_param(entry, Type::ptr());

    // Conversions
    Value* sext = b.build_sext_i64(i32_val);
    Value* zext = b.build_zext_i64(i32_val);
    Value* trunc = b.build_trunc_i32(i64_val);
    Value* fptosi32 = b.build_fptosi_i32(f64_val);
    Value* fptosi64 = b.build_fptosi_i64(f64_val);
    Value* sitofp32 = b.build_sitofp_f64_i32(i32_val);
    Value* sitofp64 = b.build_sitofp_f64_i64(i64_val);
    Value* bitcast_i_f = b.build_bitcast_i64_f64(f64_val);
    Value* bitcast_f_i = b.build_bitcast_f64_i64(i64_val);

    CHECK_EQ(sext->type(), Type::i64());
    CHECK_EQ(zext->type(), Type::i64());
    CHECK_EQ(trunc->type(), Type::i32());
    CHECK_EQ(fptosi32->type(), Type::i32());
    CHECK_EQ(fptosi64->type(), Type::i64());
    CHECK_EQ(sitofp32->type(), Type::f64());
    CHECK_EQ(sitofp64->type(), Type::f64());
    CHECK_EQ(bitcast_i_f->type(), Type::i64());
    CHECK_EQ(bitcast_f_i->type(), Type::f64());

    // Arithmetic & Logic
    Value* sdiv = b.build_sdiv(i32_val, trunc);
    Value* udiv = b.build_udiv(i32_val, trunc);
    Value* smod = b.build_smod(i32_val, trunc);
    Value* umod = b.build_umod(i32_val, trunc);
    Value* neg = b.build_neg(i32_val);
    Value* band = b.build_and(i32_val, trunc);
    Value* bor = b.build_or(i32_val, trunc);
    Value* bxor = b.build_xor(i32_val, trunc);
    Value* shl = b.build_shl(i32_val, trunc);
    Value* lshr = b.build_lshr(i32_val, trunc);
    Value* ashr = b.build_ashr(i32_val, trunc);
    Value* bnot = b.build_not(i32_val);
    Value* clz = b.build_clz(i32_val);
    Value* ctz = b.build_ctz(i32_val);
    Value* popcnt = b.build_popcnt(i32_val);

    CHECK_EQ(sdiv->type(), Type::i32());
    CHECK_EQ(udiv->type(), Type::i32());
    CHECK_EQ(smod->type(), Type::i32());
    CHECK_EQ(umod->type(), Type::i32());
    CHECK_EQ(neg->type(), Type::i32());
    CHECK_EQ(band->type(), Type::i32());
    CHECK_EQ(bor->type(), Type::i32());
    CHECK_EQ(bxor->type(), Type::i32());
    CHECK_EQ(shl->type(), Type::i32());
    CHECK_EQ(lshr->type(), Type::i32());
    CHECK_EQ(ashr->type(), Type::i32());
    CHECK_EQ(bnot->type(), Type::i32());
    CHECK_EQ(clz->type(), Type::i32());
    CHECK_EQ(ctz->type(), Type::i32());
    CHECK_EQ(popcnt->type(), Type::i32());

    // Comparisons
    Value* eq = b.build_eq(i32_val, trunc);
    Value* ne = b.build_ne(i32_val, trunc);
    Value* slt = b.build_slt(i32_val, trunc);
    Value* ult = b.build_ult(i32_val, trunc);
    Value* sle = b.build_sle(i32_val, trunc);
    Value* ule = b.build_ule(i32_val, trunc);
    Value* sgt = b.build_sgt(i32_val, trunc);
    Value* ugt = b.build_ugt(i32_val, trunc);
    Value* sge = b.build_sge(i32_val, trunc);
    Value* uge = b.build_uge(i32_val, trunc);

    CHECK_EQ(eq->type(), Type::i32());
    CHECK_EQ(ne->type(), Type::i32());
    CHECK_EQ(slt->type(), Type::i32());
    CHECK_EQ(ult->type(), Type::i32());
    CHECK_EQ(sle->type(), Type::i32());
    CHECK_EQ(ule->type(), Type::i32());
    CHECK_EQ(sgt->type(), Type::i32());
    CHECK_EQ(ugt->type(), Type::i32());
    CHECK_EQ(sge->type(), Type::i32());
    CHECK_EQ(uge->type(), Type::i32());

    // Indirect call
    b.build_call_indirect(ptr_val, Type::void_type(), {i32_val});

    b.build_ret_void();

    fn->rebuild_cfg_predecessors();

    DiagnosticReporter diag;
    bool ok = verify_module(mod, &diag);
    if (!ok) {
        std::cout << diag.format_all();
    }
    CHECK(ok);
}

TEST_CASE("BasicBlock instruction list manipulation") {
    Module mod("block_manip_mod");
    Function* fn = mod.create_function("manip_fn", Type::void_type());
    Builder b(mod);
    b.set_function(fn);

    BasicBlock* bb = b.append_block("entry");
    Instruction* i1 = mod.arena().make<Instruction>(Opcode::iconst_i32, Type::i32());
    Instruction* i2 = mod.arena().make<Instruction>(Opcode::iconst_i64, Type::i64());
    Instruction* i3 = mod.arena().make<Instruction>(Opcode::fconst_f64, Type::f64());

    bb->append_instruction(i1);
    bb->append_instruction(i3);
    CHECK_EQ(bb->instruction_count(), size_t{2});
    CHECK_EQ(bb->head(), i1);
    CHECK_EQ(bb->tail(), i3);

    // Insert i2 before i3
    bb->insert_before(i2, i3);
    CHECK_EQ(bb->instruction_count(), size_t{3});
    CHECK_EQ(i1->next(), i2);
    CHECK_EQ(i2->prev(), i1);
    CHECK_EQ(i2->next(), i3);
    CHECK_EQ(i3->prev(), i2);

    // Remove i2
    bb->remove_instruction(i2);
    CHECK_EQ(bb->instruction_count(), size_t{2});
    CHECK_EQ(i1->next(), i3);
    CHECK_EQ(i3->prev(), i1);
    CHECK(i2->parent() == nullptr);
}
