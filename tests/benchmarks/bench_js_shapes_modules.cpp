#include "bench_js_shapes_modules.hpp"
#include <brass/brass.hpp>

namespace brass::bench {

std::unique_ptr<Module> build_nanbox_tag_test_module() {
    auto mod = std::make_unique<Module>("bench_nanbox");
    Builder b(*mod);

    Function* fn = mod->create_function("nanbox_tag_test", Type::f64(), {Type::ptr(), Type::i64()});
    b.set_function(fn);

    BasicBlock* entry = b.append_block("entry");
    Value* data_ptr = b.add_block_param(entry, Type::ptr());
    Value* len = b.add_block_param(entry, Type::i64());

    BasicBlock* loop_body = b.create_block("loop_body");
    BasicBlock* bb_double = b.create_block("bb_double");
    BasicBlock* bb_tag_check = b.create_block("bb_tag_check");
    BasicBlock* bb_int = b.create_block("bb_int");
    BasicBlock* bb_fallback = b.create_block("bb_fallback");
    BasicBlock* exit_bb = b.create_block("exit");

    Value* zero_i = b.build_iconst_i64(0);
    Value* zero_f = b.build_fconst_f64(0.0);
    Value* one_i = b.build_iconst_i64(1);
    Value* one_f = b.build_fconst_f64(1.0);
    Value* tag_thresh = b.build_iconst_i64(static_cast<int64_t>(0xFFF8000000000000ULL));
    Value* shift_32 = b.build_iconst_i64(32);
    Value* int_tag = b.build_iconst_i64(0xFFF90000LL);

    Value* init_cond = b.build_slt(zero_i, len);
    b.build_br_if(init_cond, loop_body, {zero_i, zero_f}, exit_bb, {zero_f});

    fn->append_block(loop_body);
    b.position_at_end(loop_body);
    Value* i = b.add_block_param(loop_body, Type::i64());
    Value* acc = b.add_block_param(loop_body, Type::f64());
    Value* v = b.build_load_indexed(Type::i64(), data_ptr, i, 8, 0);
    Value* is_double = b.build_ult(v, tag_thresh);
    b.build_br_if(is_double, bb_double, {}, bb_tag_check, {});

    fn->append_block(bb_double);
    b.position_at_end(bb_double);
    Value* d = b.build_bitcast_f64_i64(v);
    Value* acc_d = b.build_add(acc, d);
    Value* next_i_d = b.build_add(i, one_i);
    Value* cond_d = b.build_slt(next_i_d, len);
    b.build_br_if(cond_d, loop_body, {next_i_d, acc_d}, exit_bb, {acc_d});

    fn->append_block(bb_tag_check);
    b.position_at_end(bb_tag_check);
    Value* tag = b.build_lshr(v, shift_32);
    Value* is_int = b.build_eq(tag, int_tag);
    b.build_br_if(is_int, bb_int, {}, bb_fallback, {});

    fn->append_block(bb_int);
    b.position_at_end(bb_int);
    Value* iv = b.build_trunc_i32(v);
    Value* div = b.build_sitofp_f64_i32(iv);
    Value* acc_int = b.build_add(acc, div);
    Value* next_i_int = b.build_add(i, one_i);
    Value* cond_int = b.build_slt(next_i_int, len);
    b.build_br_if(cond_int, loop_body, {next_i_int, acc_int}, exit_bb, {acc_int});

    fn->append_block(bb_fallback);
    b.position_at_end(bb_fallback);
    Value* acc_fb = b.build_add(acc, one_f);
    Value* next_i_fb = b.build_add(i, one_i);
    Value* cond_fb = b.build_slt(next_i_fb, len);
    b.build_br_if(cond_fb, loop_body, {next_i_fb, acc_fb}, exit_bb, {acc_fb});

    fn->append_block(exit_bb);
    b.position_at_end(exit_bb);
    Value* final_res = b.add_block_param(exit_bb, Type::f64());
    b.build_ret(final_res);

    fn->rebuild_cfg_predecessors();
    return mod;
}

std::unique_ptr<Module> build_shape_guard_module() {
    auto mod = std::make_unique<Module>("bench_shape_guard");
    Builder b(*mod);

    Function* fn = mod->create_function("shape_guard", Type::i64(), {Type::ptr(), Type::i64()});
    b.set_function(fn);

    BasicBlock* entry = b.append_block("entry");
    Value* objs_ptr = b.add_block_param(entry, Type::ptr());
    Value* len = b.add_block_param(entry, Type::i64());

    BasicBlock* loop_body = b.create_block("loop_body");
    BasicBlock* bb_hit_a = b.create_block("bb_hit_a");
    BasicBlock* loop_latch = b.create_block("loop_latch");
    BasicBlock* bb_slow_path = b.create_block("bb_slow_path");
    BasicBlock* bb_hit_b = b.create_block("bb_hit_b");
    BasicBlock* bb_hit_c = b.create_block("bb_hit_c");
    BasicBlock* exit_bb = b.create_block("exit");

    Value* zero = b.build_iconst_i64(0);
    Value* one = b.build_iconst_i64(1);
    Value* shape_a = b.build_iconst_i64(0xAA01);
    Value* shape_b = b.build_iconst_i64(0xBB02);

    Value* init_cond = b.build_slt(zero, len);
    b.build_br_if(init_cond, loop_body, {zero, zero}, exit_bb, {zero});

    // 1. Loop body (fast-path entry)
    fn->append_block(loop_body);
    b.position_at_end(loop_body);
    Value* i = b.add_block_param(loop_body, Type::i64());
    Value* sum = b.add_block_param(loop_body, Type::i64());

    Value* obj = b.build_load_indexed(Type::ptr(), objs_ptr, i, 8, 0);
    Value* shape = b.build_load(Type::i64(), obj, 0);
    Value* not_a = b.build_ne(shape, shape_a);
    b.build_br_if(not_a, bb_slow_path, {}, bb_hit_a, {});

    // 2. Fast-path Shape A hit (contiguous fallthrough)
    fn->append_block(bb_hit_a);
    b.position_at_end(bb_hit_a);
    Value* val_a = b.build_load(Type::i64(), obj, 8);
    b.build_br(loop_latch, {val_a});

    // 3. Fast-path Loop latch (contiguous fallthrough from bb_hit_a)
    fn->append_block(loop_latch);
    b.position_at_end(loop_latch);
    Value* val_to_add = b.add_block_param(loop_latch, Type::i64());
    Value* next_sum = b.build_add(sum, val_to_add);
    Value* next_i = b.build_add(i, one);
    Value* cond = b.build_slt(next_i, len);
    b.build_br_if(cond, loop_body, {next_i, next_sum}, exit_bb, {next_sum});

    // 4. Out-of-line slow path (Shape B / Shape C fallback)
    fn->append_block(bb_slow_path);
    b.position_at_end(bb_slow_path);
    Value* not_b = b.build_ne(shape, shape_b);
    b.build_br_if(not_b, bb_hit_c, {}, bb_hit_b, {});

    fn->append_block(bb_hit_b);
    b.position_at_end(bb_hit_b);
    Value* val_b = b.build_load(Type::i64(), obj, 16);
    b.build_br(loop_latch, {val_b});

    fn->append_block(bb_hit_c);
    b.position_at_end(bb_hit_c);
    Value* val_c = b.build_load(Type::i64(), obj, 24);
    Value* val_c2 = b.build_add(val_c, val_c);
    b.build_br(loop_latch, {val_c2});

    fn->append_block(exit_bb);
    b.position_at_end(exit_bb);
    Value* final_res = b.add_block_param(exit_bb, Type::i64());
    b.build_ret(final_res);

    fn->rebuild_cfg_predecessors();
    return mod;
}

std::unique_ptr<Module> build_patchable_ic_module() {
    auto mod = std::make_unique<Module>("bench_patchable_ic");
    Builder b(*mod);

    // target_stub_1(x) -> (x + 3) & 0x7FFFFFFF
    Function* fn1 = mod->create_function("target_stub_1", Type::i64(), {Type::i64()});
    b.set_function(fn1);
    BasicBlock* e1 = b.append_block("entry");
    b.position_at_end(e1);
    Value* x1 = b.add_block_param(e1, Type::i64());
    Value* res1 = b.build_and(b.build_add(x1, b.build_iconst_i64(3)), b.build_iconst_i64(0x7FFFFFFF));
    b.build_ret(res1);
    fn1->rebuild_cfg_predecessors();

    // target_stub_2(x) -> ((x ^ 0x5555) + 1) & 0x7FFFFFFF
    Function* fn2 = mod->create_function("target_stub_2", Type::i64(), {Type::i64()});
    b.set_function(fn2);
    BasicBlock* e2 = b.append_block("entry");
    b.position_at_end(e2);
    Value* x2 = b.add_block_param(e2, Type::i64());
    Value* res2 = b.build_and(b.build_add(b.build_xor(x2, b.build_iconst_i64(0x5555)), b.build_iconst_i64(1)), b.build_iconst_i64(0x7FFFFFFF));
    b.build_ret(res2);
    fn2->rebuild_cfg_predecessors();

    // patchable_ic_runner
    Function* fn = mod->create_function("patchable_ic_runner", Type::i64(), {Type::i64(), Type::i64()});
    b.set_function(fn);

    BasicBlock* entry = b.append_block("entry");
    Value* x = b.add_block_param(entry, Type::i64());
    Value* iters = b.add_block_param(entry, Type::i64());

    BasicBlock* loop_hdr = b.create_block("loop_hdr");
    BasicBlock* loop_body = b.create_block("loop_body");
    BasicBlock* exit_bb = b.create_block("exit");

    Value* zero = b.build_iconst_i64(0);
    Value* one = b.build_iconst_i64(1);

    b.build_br(loop_hdr, {zero, x});

    fn->append_block(loop_hdr);
    b.position_at_end(loop_hdr);
    Value* i = b.add_block_param(loop_hdr, Type::i64());
    Value* acc = b.add_block_param(loop_hdr, Type::i64());
    Value* cond = b.build_slt(i, iters);
    b.build_br_if(cond, loop_body, {}, exit_bb, {acc});

    fn->append_block(loop_body);
    b.position_at_end(loop_body);
    Value* sub_res = b.build_patchable_call("ic_site_bench", "target_stub_1", Type::i64(), {acc});
    Value* next_acc = b.build_add(acc, sub_res);
    Value* next_i = b.build_add(i, one);
    b.build_br(loop_hdr, {next_i, next_acc});

    fn->append_block(exit_bb);
    b.position_at_end(exit_bb);
    Value* final_res = b.add_block_param(exit_bb, Type::i64());
    b.build_ret(final_res);

    fn->rebuild_cfg_predecessors();
    return mod;
}

std::unique_ptr<Module> build_gc_alloc_loop_module() {
    auto mod = std::make_unique<Module>("bench_gc_alloc_loop");
    mod->add_external_symbol("brass_gc_alloc");
    mod->add_external_symbol("brass_gc_safepoint");
    Builder b(*mod);

    Function* fn = mod->create_function("gc_alloc_runner", Type::i64(), {Type::i64()});
    b.set_function(fn);

    BasicBlock* entry = b.append_block("entry");
    Value* num_nodes = b.add_block_param(entry, Type::i64());

    BasicBlock* loop_alloc_hdr = b.create_block("loop_alloc_hdr");
    BasicBlock* loop_alloc_body = b.create_block("loop_alloc_body");
    BasicBlock* loop_trav_hdr = b.create_block("loop_trav_hdr");
    BasicBlock* loop_trav_body = b.create_block("loop_trav_body");
    BasicBlock* exit_bb = b.create_block("exit");

    Value* zero_i = b.build_iconst_i64(0);
    Value* one_i = b.build_iconst_i64(1);
    Value* sz16 = b.build_iconst_i64(16);
    Value* mask2 = b.build_iconst_i64(2); // bit 1 indicates gcref field at offset 8
    Value* tag1 = b.build_iconst_i32(1);

    // Initial null gcref
    Value* null_gc = b.build_iconst_i64(0);

    b.build_br(loop_alloc_hdr, {zero_i, null_gc});

    fn->append_block(loop_alloc_hdr);
    b.position_at_end(loop_alloc_hdr);
    Value* i = b.add_block_param(loop_alloc_hdr, Type::i64());
    Value* head = b.add_block_param(loop_alloc_hdr, Type::gcref());
    Value* cond_alloc = b.build_slt(i, num_nodes);
    b.build_br_if(cond_alloc, loop_alloc_body, {}, loop_trav_hdr, {head, zero_i});

    fn->append_block(loop_alloc_body);
    b.position_at_end(loop_alloc_body);
    Value* node = b.build_call("brass_gc_alloc", Type::gcref(), {sz16, mask2, tag1});
    Value* val = b.build_add(i, one_i);
    b.build_store(Type::i64(), node, 0, val);
    b.build_store(Type::gcref(), node, 8, head);
    Value* next_i = b.build_add(i, one_i);
    b.build_br(loop_alloc_hdr, {next_i, node});

    fn->append_block(loop_trav_hdr);
    b.position_at_end(loop_trav_hdr);
    Value* cur = b.add_block_param(loop_trav_hdr, Type::gcref());
    Value* sum = b.add_block_param(loop_trav_hdr, Type::i64());
    Value* has_node = b.build_ne(cur, zero_i);
    b.build_br_if(has_node, loop_trav_body, {}, exit_bb, {sum});

    fn->append_block(loop_trav_body);
    b.position_at_end(loop_trav_body);
    Value* node_val = b.build_load(Type::i64(), cur, 0);
    Value* next_node = b.build_load(Type::gcref(), cur, 8);
    Value* next_sum = b.build_add(sum, node_val);
    b.build_br(loop_trav_hdr, {next_node, next_sum});

    fn->append_block(exit_bb);
    b.position_at_end(exit_bb);
    Value* final_res = b.add_block_param(exit_bb, Type::i64());
    b.build_ret(final_res);

    fn->rebuild_cfg_predecessors();
    return mod;
}

} // namespace brass::bench
