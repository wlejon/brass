#include "fuzz_generator.hpp"
#include <brass/mir/builder.hpp>
#include <brass/mir/verifier.hpp>
#include <random>
#include <vector>
#include <string>

namespace brass::test {

void generate_fuzz_expr(Module& mod, std::string_view fn_name, uint64_t seed) {
    std::mt19937_64 rng(seed);
    Function* fn = mod.create_function(fn_name, Type::i64(), {Type::i64(), Type::i64()});

    Builder b(mod);
    b.set_function(fn);

    BasicBlock* entry = b.append_block("bb0");
    b.position_at_end(entry);
    Value* x = b.add_block_param(entry, Type::i64());
    Value* y = b.add_block_param(entry, Type::i64());

    std::vector<Value*> pool = {x, y};
    size_t num_insts = 15 + (seed % 20);

    for (size_t i = 0; i < num_insts; ++i) {
        uint32_t op_choice = static_cast<uint32_t>(rng() % 11);
        Value* opA = pool[rng() % pool.size()];
        Value* opB = pool[rng() % pool.size()];

        Value* res = nullptr;
        switch (op_choice) {
            case 0:
                res = b.build_add(opA, opB);
                break;
            case 1:
                res = b.build_sub(opA, opB);
                break;
            case 2:
                res = b.build_and(opA, opB);
                break;
            case 3:
                res = b.build_or(opA, opB);
                break;
            case 4:
                res = b.build_xor(opA, opB);
                break;
            case 5:
                res = b.build_not(opA);
                break;
            case 6: {
                Value* shift_c = b.build_iconst_i64(static_cast<int64_t>(rng() % 32));
                res = b.build_shl(opA, shift_c);
                break;
            }
            case 7: {
                Value* shift_c = b.build_iconst_i64(static_cast<int64_t>(rng() % 32));
                res = b.build_lshr(opA, shift_c);
                break;
            }
            case 8: {
                Value* shift_c = b.build_iconst_i64(static_cast<int64_t>(rng() % 32));
                res = b.build_ashr(opA, shift_c);
                break;
            }
            case 9:
                res = b.build_popcnt(opA);
                break;
            case 10:
                res = b.build_clz(opA);
                break;
        }
        if (res) pool.push_back(res);
    }

    b.build_ret(pool.back());
    fn->rebuild_cfg_predecessors();
}

void generate_fuzz_loops(Module& mod, std::string_view fn_name, uint64_t seed) {
    std::mt19937_64 rng(seed);
    Function* fn = mod.create_function(fn_name, Type::i64(), {Type::i64(), Type::i64()});

    Builder b(mod);
    b.set_function(fn);

    BasicBlock* entry = b.append_block("bb0");
    BasicBlock* loop_hdr = b.create_block("bb_loop_hdr");
    BasicBlock* loop_body = b.create_block("bb_loop_body");
    BasicBlock* loop_branch = b.create_block("bb_loop_branch");
    BasicBlock* loop_next = b.create_block("bb_loop_next");
    BasicBlock* exit_bb = b.create_block("bb_exit");

    fn->append_block(loop_hdr);
    fn->append_block(loop_body);
    fn->append_block(loop_branch);
    fn->append_block(loop_next);
    fn->append_block(exit_bb);

    b.position_at_end(entry);
    Value* limit = b.add_block_param(entry, Type::i64());
    Value* init_val = b.add_block_param(entry, Type::i64());

    Value* zero = b.build_iconst_i64(0);
    Value* one = b.build_iconst_i64(1);
    b.build_br(loop_hdr, {zero, init_val, zero});

    b.position_at_end(loop_hdr);
    Value* i = b.add_block_param(loop_hdr, Type::i64());
    Value* acc1 = b.add_block_param(loop_hdr, Type::i64());
    Value* acc2 = b.add_block_param(loop_hdr, Type::i64());

    Value* cond = b.build_slt(i, limit);
    b.build_br_if(cond, loop_body, {}, exit_bb, {acc1});

    b.position_at_end(loop_body);
    Value* mask1 = b.build_and(i, one);
    Value* is_even = b.build_eq(mask1, zero);
    b.build_br_if(is_even, loop_branch, {}, loop_next, {acc1, acc2});

    b.position_at_end(loop_branch);
    Value* mod_acc1 = b.build_add(acc1, i);
    Value* mod_acc2 = b.build_xor(acc2, b.build_iconst_i64(static_cast<int64_t>(rng() % 255)));
    b.build_br(loop_next, {mod_acc1, mod_acc2});

    b.position_at_end(loop_next);
    Value* cur_acc1 = b.add_block_param(loop_next, Type::i64());
    Value* cur_acc2 = b.add_block_param(loop_next, Type::i64());
    Value* step_acc = b.build_add(cur_acc1, cur_acc2);
    Value* next_i = b.build_add(i, one);
    b.build_br(loop_hdr, {next_i, step_acc, cur_acc2});

    b.position_at_end(exit_bb);
    Value* final_res = b.add_block_param(exit_bb, Type::i64());
    b.build_ret(final_res);

    fn->rebuild_cfg_predecessors();
}

void generate_fuzz_gc(Module& mod, std::string_view fn_name, uint64_t seed) {
    std::mt19937_64 rng(seed);
    mod.add_external_symbol("brass_gc_alloc");
    mod.add_external_symbol("brass_gc_safepoint");

    std::string helper_name = std::string(fn_name) + "_alloc_helper";
    Function* fn_helper = mod.create_function(helper_name, Type::gcref(), {Type::i64()});
    {
        Builder b(mod);
        b.set_function(fn_helper);
        BasicBlock* entry = b.append_block("bb0");
        b.position_at_end(entry);
        Value* v = b.add_block_param(entry, Type::i64());

        Value* sz16 = b.build_iconst_i64(16);
        Value* mask0 = b.build_iconst_i64(0);
        Value* tag1 = b.build_iconst_i32(1);
        Value* obj = b.build_call("brass_gc_alloc", Type::gcref(), {sz16, mask0, tag1});
        b.build_store(Type::i64(), obj, 0, v);
        b.build_safepoint();
        b.build_ret(obj);
        fn_helper->rebuild_cfg_predecessors();
    }

    Function* fn_main = mod.create_function(fn_name, Type::i64(), {Type::i64(), Type::i64()});
    {
        Builder b(mod);
        b.set_function(fn_main);
        BasicBlock* entry = b.append_block("bb0");
        b.position_at_end(entry);
        Value* a = b.add_block_param(entry, Type::i64());
        Value* c = b.add_block_param(entry, Type::i64());

        // Allocate obj1
        Value* obj1 = b.build_call(helper_name, Type::gcref(), {a});

        // Allocate obj2 while obj1 is live
        Value* obj2 = b.build_call(helper_name, Type::gcref(), {c});

        // Safepoint while both are live
        b.build_safepoint();

        // Allocate obj3 while obj1 and obj2 are live
        Value* sum_ac = b.build_add(a, c);
        Value* obj3 = b.build_call(helper_name, Type::gcref(), {sum_ac});

        b.build_safepoint();

        Value* v1 = b.build_load(Type::i64(), obj1, 0);
        Value* v2 = b.build_load(Type::i64(), obj2, 0);
        Value* v3 = b.build_load(Type::i64(), obj3, 0);

        Value* total = b.build_add(b.build_add(v1, v2), v3);
        b.build_ret(total);
        fn_main->rebuild_cfg_predecessors();
    }
}

void generate_fuzz_speculation(Module& mod, std::string_view fn_name, uint64_t seed) {
    (void)seed;
    std::string twin_name = std::string(fn_name) + "_twin";
    Function* twin = mod.create_function(twin_name, Type::i64(), {Type::i32(), Type::ptr()});
    {
        Builder tb(mod);
        tb.set_function(twin);
        BasicBlock* tb0 = tb.append_block("bb0");
        BasicBlock* tb_res0 = tb.append_block("bb_res_0");
        BasicBlock* tb_res1 = tb.append_block("bb_res_1");

        twin->add_resume_point(0, tb_res0);
        twin->add_resume_point(1, tb_res1);

        tb.position_at_end(tb0);
        tb.add_block_param(tb0, Type::i32());
        tb.add_block_param(tb0, Type::ptr());
        tb.build_ret(tb.build_iconst_i64(0));

        // Resume 0: load state [a, b], return a + b
        tb.position_at_end(tb_res0);
        Value* ptr_param0 = twin->entry_block()->param(1);
        Value* x0 = tb.build_load(Type::i64(), ptr_param0, 0);
        Value* y0 = tb.build_load(Type::i64(), ptr_param0, 8);
        Value* sum0 = tb.build_add(x0, y0);
        tb.build_ret(sum0);

        // Resume 1: load state [a, b], return (a + 10) * b
        tb.position_at_end(tb_res1);
        Value* ptr_param1 = twin->entry_block()->param(1);
        Value* x1 = tb.build_load(Type::i64(), ptr_param1, 0);
        Value* y1 = tb.build_load(Type::i64(), ptr_param1, 8);
        Value* add_x = tb.build_add(x1, tb.build_iconst_i64(10));
        Value* mul1 = tb.build_mul(add_x, y1);
        tb.build_ret(mul1);

        twin->rebuild_cfg_predecessors();
    }

    Function* spec = mod.create_function(fn_name, Type::i64(), {Type::i64(), Type::i64(), Type::i32()});
    {
        Builder sb(mod);
        sb.set_function(spec);
        BasicBlock* sb_entry = sb.append_block("bb0");
        sb.position_at_end(sb_entry);
        Value* sa = sb.add_block_param(sb_entry, Type::i64());
        Value* sb_param = sb.add_block_param(sb_entry, Type::i64());
        Value* stype = sb.add_block_param(sb_entry, Type::i32());

        // Guard 1: stype != 0 -> if false, deopt to twin at resume 0
        Instruction* g0 = sb.build_guard(stype, twin_name, {sa, sb_param});
        g0->set_resume_id(0);

        // Guard 2: check secondary condition
        Value* c10 = sb.build_iconst_i64(1000);
        Value* cond2 = sb.build_slt(sa, c10);
        Instruction* g1 = sb.build_guard(cond2, twin_name, {sa, sb_param});
        g1->set_resume_id(1);

        Value* smul = sb.build_mul(sa, sb_param);
        sb.build_ret(smul);
        spec->rebuild_cfg_predecessors();
    }
}

void generate_fuzz_memory(Module& mod, std::string_view fn_name, uint64_t seed) {
    (void)seed;
    Function* fn = mod.create_function(fn_name, Type::i64(), {Type::ptr(), Type::i64()});

    Builder b(mod);
    b.set_function(fn);

    BasicBlock* entry = b.append_block("bb0");
    BasicBlock* loop_hdr = b.create_block("bb_loop_hdr");
    BasicBlock* loop_body = b.create_block("bb_loop_body");
    BasicBlock* exit_bb = b.create_block("bb_exit");

    fn->append_block(loop_hdr);
    fn->append_block(loop_body);
    fn->append_block(exit_bb);

    b.position_at_end(entry);
    Value* buf_ptr = b.add_block_param(entry, Type::ptr());
    Value* count = b.add_block_param(entry, Type::i64());
    Value* zero = b.build_iconst_i64(0);
    b.build_br(loop_hdr, {zero, zero});

    b.position_at_end(loop_hdr);
    Value* idx = b.add_block_param(loop_hdr, Type::i64());
    Value* acc = b.add_block_param(loop_hdr, Type::i64());
    Value* cond = b.build_slt(idx, count);
    b.build_br_if(cond, loop_body, {}, exit_bb, {acc});

    b.position_at_end(loop_body);
    // Load indexed: buf_ptr[idx * 8 + 0]
    Value* val = b.build_load_indexed(Type::i64(), buf_ptr, idx, 8, 0);
    Value* mod_val = b.build_add(val, b.build_iconst_i64(7));
    b.build_store_indexed(Type::i64(), buf_ptr, idx, 8, 0, mod_val);
    Value* new_acc = b.build_add(acc, mod_val);
    Value* one = b.build_iconst_i64(1);
    Value* next_idx = b.build_add(idx, one);
    b.build_br(loop_hdr, {next_idx, new_acc});

    b.position_at_end(exit_bb);
    Value* final_res = b.add_block_param(exit_bb, Type::i64());
    b.build_ret(final_res);

    fn->rebuild_cfg_predecessors();
}

void generate_fuzz_patching(Module& mod, std::string_view fn_name, uint64_t seed) {
    (void)seed;
    std::string stub_add_name = std::string(fn_name) + "_stub_add";
    Function* s_add = mod.create_function(stub_add_name, Type::i32(), {Type::i32()});
    {
        Builder ba(mod);
        ba.set_function(s_add);
        BasicBlock* bba = ba.append_block("bb0");
        ba.position_at_end(bba);
        Value* p = ba.add_block_param(bba, Type::i32());
        ba.build_ret(ba.build_add(p, ba.build_iconst_i32(100)));
        s_add->rebuild_cfg_predecessors();
    }

    std::string stub_sub_name = std::string(fn_name) + "_stub_sub";
    Function* s_sub = mod.create_function(stub_sub_name, Type::i32(), {Type::i32()});
    {
        Builder bs(mod);
        bs.set_function(s_sub);
        BasicBlock* bbs = bs.append_block("bb0");
        bs.position_at_end(bbs);
        Value* p = bs.add_block_param(bbs, Type::i32());
        bs.build_ret(bs.build_sub(p, bs.build_iconst_i32(50)));
        s_sub->rebuild_cfg_predecessors();
    }

    Function* caller = mod.create_function(fn_name, Type::i32(), {Type::i32()});
    {
        Builder bc(mod);
        bc.set_function(caller);
        BasicBlock* bbc = bc.append_block("bb0");
        bc.position_at_end(bbc);
        Value* x = bc.add_block_param(bbc, Type::i32());
        std::string bias_name = std::string(fn_name) + "_bias";
        std::string call_site_name = std::string(fn_name) + "_call_site";
        Value* bias = bc.build_patchable_const_i32(bias_name, 5);
        Value* biased_x = bc.build_add(x, bias);
        Value* res = bc.build_patchable_call(call_site_name, stub_add_name, Type::i32(), {biased_x});
        bc.build_ret(res);
        caller->rebuild_cfg_predecessors();
    }
}

} // namespace brass::test
