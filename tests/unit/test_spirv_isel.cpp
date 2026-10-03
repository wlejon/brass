// SpirvISel (MIR -> SPIR-V): every scalar ALU op, comparisons and their
// bool/i32 forms, the signed-division fixup, loads/stores (scalar, indexed,
// vector), every vector opcode on every vector type, and the control-flow
// shapes the structurizer handles -- if, if-else, loops, nested loops,
// early exits, shared joins, loop-carried values -- each run through
// spirv::verify and spirv-val. Diagnostics: irreducible CFGs, multi-exit
// loops, unsupported opcodes/terminators/calls, shared pointers that escape.

#include "spirv_test_support.hpp"

using namespace brass;
using namespace spvtest;

namespace {

// ---------------------------------------------------------------------------
// ALU
// ---------------------------------------------------------------------------

TEST_CASE("SPIR-V ISel - scalar ALU ops on i32 i64 f32 f64") {
    Kernel k({Type::ptr(), Type::i32(), Type::i32(), Type::i64(), Type::i64(), Type::f32(), Type::f32(), Type::f64(),
              Type::f64()});
    Builder& b = k.b;
    Value* out = k.p(0);
    int32_t slot = 0;
    auto put = [&](Value* v) {
        b.build_store(v->type(), out, slot, v);
        slot += 8;
    };
    for (auto [x, y] : {std::pair{k.p(1), k.p(2)}, std::pair{k.p(3), k.p(4)}}) {
        put(b.build_add(x, y));
        put(b.build_sub(x, y));
        put(b.build_mul(x, y));
        put(b.build_sdiv(x, y));
        put(b.build_udiv(x, y));
        put(b.build_smod(x, y));
        put(b.build_umod(x, y));
        put(b.build_neg(x));
        put(b.build_and(x, y));
        put(b.build_or(x, y));
        put(b.build_xor(x, y));
        put(b.build_not(x));
        put(b.build_shl(x, y));
        put(b.build_lshr(x, y));
        put(b.build_ashr(x, y));
    }
    for (auto [x, y] : {std::pair{k.p(5), k.p(6)}, std::pair{k.p(7), k.p(8)}}) {
        put(b.build_add(x, y));
        put(b.build_sub(x, y));
        put(b.build_mul(x, y));
        put(b.build_sdiv(x, y));
        put(b.build_smod(x, y));
        put(b.build_neg(x));
        put(b.build_fma(x, y, x));
        put(b.build_sqrt(x));
        put(b.build_floor(x));
        put(b.build_ceil(x));
        put(b.build_round(x));
        put(b.build_fabs(x));
        put(b.build_fmin(x, y));
        put(b.build_fmax(x, y));
    }
    put(b.build_sext_i64(k.p(1)));
    put(b.build_zext_i64(k.p(1)));
    put(b.build_trunc_i32(k.p(3)));
    put(b.build_sitofp_f32_i32(k.p(1)));
    put(b.build_sitofp_f32_i64(k.p(3)));
    put(b.build_sitofp_f64_i32(k.p(1)));
    put(b.build_sitofp_f64_i64(k.p(3)));
    put(b.build_fptosi_i32_f32(k.p(5)));
    put(b.build_fptosi_i64_f32(k.p(5)));
    put(b.build_fptosi_i32(k.p(7)));
    put(b.build_fptosi_i64(k.p(7)));
    put(b.build_fptrunc_f32_f64(k.p(7)));
    put(b.build_fpext_f64_f32(k.p(5)));
    put(b.build_bitcast_i64_f64(k.p(7)));
    put(b.build_bitcast_f64_i64(k.p(3)));
    put(b.build_add(k.p(1), b.build_iconst_i32(-7)));
    put(b.build_mul(k.p(5), b.build_fconst_f32(0.5f)));
    put(b.build_mul(k.p(7), b.build_fconst_f64(0.25)));
    b.build_ret_void();

    compile_checked(*k.fn);
    std::string d = dump_of(*k.fn);
    for (const char* op : {"OpIAdd", "OpISub", "OpIMul", "OpSDiv", "OpUDiv", "OpSRem", "OpUMod", "OpSNegate",
                           "OpBitwiseAnd", "OpBitwiseOr", "OpBitwiseXor", "OpNot", "OpShiftLeftLogical",
                           "OpShiftRightLogical", "OpShiftRightArithmetic", "OpFAdd", "OpFSub", "OpFMul", "OpFDiv",
                           "OpFRem", "OpFNegate", "OpSConvert", "OpUConvert", "OpConvertSToF", "OpConvertFToS",
                           "OpFConvert", "OpBitcast", "OpExtInst"}) {
        if (!has(d, op)) std::cerr << "missing " << op << "\n";
        CHECK(has(d, op));
    }
    CHECK(has(d, "OpCapability Float64"));
    CHECK(has(d, "OpCapability Int64"));
}

TEST_CASE("SPIR-V ISel - comparisons give bools, i32 only when used as a value") {
    Kernel k({Type::ptr(), Type::i32(), Type::i64(), Type::f32(), Type::f64()});
    Builder& b = k.b;
    using Cmp = Value* (Builder::*)(Value*, Value*);
    const Cmp cmps[] = {&Builder::build_eq, &Builder::build_ne, &Builder::build_slt, &Builder::build_ult,
                        &Builder::build_sle, &Builder::build_ule, &Builder::build_sgt, &Builder::build_ugt,
                        &Builder::build_sge, &Builder::build_uge};
    int32_t slot = 0;
    for (size_t p = 1; p <= 4; ++p) {
        for (Cmp c : cmps) {
            Value* r = (b.*c)(k.p(p), k.p(p));
            b.build_store(Type::i32(), k.p(0), slot, r); // a value use: materialized
            slot += 4;
        }
    }
    // Only a select condition: no 0/1 materialization.
    Value* lt = b.build_slt(k.p(1), b.build_iconst_i32(3));
    b.build_store(Type::f32(), k.p(0), slot, b.build_select(lt, k.p(3), b.build_fconst_f32(1.0f)));
    b.build_ret_void();

    compile_checked(*k.fn);
    std::string d = dump_of(*k.fn);
    for (const char* op : {"OpIEqual", "OpINotEqual", "OpSLessThan", "OpULessThan", "OpSLessThanEqual",
                           "OpULessThanEqual", "OpSGreaterThan", "OpUGreaterThan", "OpSGreaterThanEqual",
                           "OpUGreaterThanEqual", "OpFOrdEqual", "OpFUnordNotEqual", "OpFOrdLessThan",
                           "OpFOrdLessThanEqual", "OpFOrdGreaterThan", "OpFOrdGreaterThanEqual"}) {
        if (!has(d, op)) std::cerr << "missing " << op << "\n";
        CHECK(has(d, op));
    }
    // 40 materialized comparisons + the float select = 41 OpSelect.
    CHECK_EQ(count_of(d, "OpSelect"), 41u);
    CHECK(!has(d, "OpFOrdNotEqual"));
}

TEST_CASE("SPIR-V ISel - signed division wraps MIN / -1, constant divisors stay plain") {
    Kernel k({Type::ptr(), Type::i32(), Type::i32()});
    Builder& b = k.b;
    b.build_store(Type::i32(), k.p(0), 0, b.build_sdiv(k.p(1), k.p(2)));
    b.build_store(Type::i32(), k.p(0), 4, b.build_smod(k.p(1), k.p(2)));
    b.build_store(Type::i32(), k.p(0), 8, b.build_sdiv(k.p(1), b.build_iconst_i32(3)));
    b.build_ret_void();
    compile_checked(*k.fn);
    std::string d = dump_of(*k.fn);
    CHECK_EQ(count_of(d, "OpSDiv"), 2u);
    CHECK_EQ(count_of(d, "OpSRem"), 1u);
    CHECK_EQ(count_of(d, "OpSelect"), 4u); // safe divisor + wrapped result, for each variable divisor
}

// ---------------------------------------------------------------------------
// Memory
// ---------------------------------------------------------------------------

TEST_CASE("SPIR-V ISel - loads and stores through device addresses") {
    Kernel k({Type::ptr(), Type::ptr(), Type::i32(), Type::i64()});
    Builder& b = k.b;
    Value* in = k.p(0);
    Value* out = k.p(1);
    int32_t off = 0;
    for (Type t : {Type::i32(), Type::i64(), Type::f32(), Type::f64(), Type::ptr()}) {
        Value* v = b.build_load(t, in, off);
        b.build_store(t, out, off, v);
        Value* vi = b.build_load_indexed(t, in, k.p(2), static_cast<uint8_t>(t.size_in_bytes()), 64);
        b.build_store_indexed(t, out, k.p(3), static_cast<uint8_t>(t.size_in_bytes()), 128, vi);
        Value* vc = b.build_load_indexed(t, in, b.build_iconst_i64(5), static_cast<uint8_t>(t.size_in_bytes()), 0);
        b.build_store(t, out, off + 256, vc);
        off += 8;
    }
    // One pointer read at two element types.
    b.build_store(Type::f32(), out, 512, b.build_load(Type::f32(), in, 0));
    b.build_store(Type::i32(), out, 516, b.build_load(Type::i32(), in, 0));
    b.build_ret_void();

    compile_checked(*k.fn);
    std::string d = dump_of(*k.fn);
    CHECK(has(d, "OpConvertUToPtr"));
    CHECK(has(d, "OpMemoryModel PhysicalStorageBuffer64 GLSL450"));
    CHECK(has(d, "OpCapability PhysicalStorageBufferAddresses"));
    CHECK(!has(d, "OpDecorate %") || !has(d, "DescriptorSet")); // no descriptors: everything is a push constant
}

TEST_CASE("SPIR-V ISel - vector loads and stores of all eight vector types") {
    Kernel k({Type::ptr(), Type::ptr()});
    Builder& b = k.b;
    int32_t off = 0;
    for (Type t : {Type::f32x4(), Type::f64x2(), Type::i32x4(), Type::i64x2(), Type::f32x8(), Type::f64x4(),
                   Type::i32x8(), Type::i64x4()}) {
        Value* v = b.build_vload(t, k.p(0), off);
        b.build_vstore(t, k.p(1), off, v);
        off += 32;
    }
    b.build_ret_void();
    compile_checked(*k.fn);
    std::string d = dump_of(*k.fn);
    // 4 128-bit types + 4 256-bit types (two parts each): 12 loads and 12 stores, all Aligned 16.
    CHECK_EQ(count_of(d, "OpLoad"), 12u + 2u);  // + the two push-constant loads
    CHECK_EQ(count_of(d, " 2 16"), 24u);         // MemoryAccess Aligned(2) 16
}

// ---------------------------------------------------------------------------
// Vectors
// ---------------------------------------------------------------------------

TEST_CASE("SPIR-V ISel - every vector opcode on every vector type") {
    for (Type t : {Type::f32x4(), Type::f64x2(), Type::i32x4(), Type::i64x2(), Type::f32x8(), Type::f64x4(),
                   Type::i32x8(), Type::i64x4()}) {
        Kernel k({Type::ptr(), Type::ptr()});
        Builder& b = k.b;
        bool fl = t.element_type().is_float();
        Value* a = b.build_vload(t, k.p(0), 0);
        Value* c = b.build_vload(t, k.p(0), 32);
        Value* r = b.build_vadd(a, c);
        r = b.build_vsub(r, c);
        r = b.build_vmul(r, a);
        r = b.build_vdiv(r, c);
        r = b.build_vmin(r, a);
        r = b.build_vmax(r, c);
        r = b.build_vfma(r, a, c);
        r = b.build_vneg(r);
        if (fl) r = b.build_vsqrt(r);
        r = b.build_vand(r, a);
        r = b.build_vor(r, c);
        r = b.build_vxor(r, a);
        r = b.build_vnot(r);
        Value* lane = b.build_vextract_lane(r, t.vector_lanes() - 1);
        r = b.build_vinsert_lane(r, lane, 0);
        r = b.build_vshuffle(r, a, 0x1B);
        r = b.build_vadd(r, b.build_vbroadcast(t, lane));
        r = b.build_vadd(r, b.build_vzero(t));
        b.build_vstore(t, k.p(1), 0, r);
        b.build_ret_void();
        compile_checked(*k.fn);
        std::string d = dump_of(*k.fn);
        CHECK(has(d, "OpVectorShuffle"));
        CHECK(has(d, "OpCompositeExtract"));
        CHECK(has(d, "OpCompositeInsert"));
        CHECK(has(d, "OpCompositeConstruct"));
        CHECK(has(d, "OpConstantNull"));
    }
}

// ---------------------------------------------------------------------------
// Control flow
// ---------------------------------------------------------------------------

// if (i < n) out[i] = 1
TEST_CASE("SPIR-V ISel - if-then") {
    Kernel k({Type::ptr(), Type::i32()});
    codegen::KernelBuilder kb(k.b);
    Value* i = kb.global_tid_x();
    kb.if_then(k.b.build_slt(i, k.p(1)), [&] { kb.store_i32_indexed(k.p(0), i, kb.const_i32(1)); });
    k.b.build_ret_void();
    compile_checked(*k.fn);
    std::string d = dump_of(*k.fn);
    CHECK_EQ(count_of(d, "OpSelectionMerge"), 1u);
    CHECK_EQ(count_of(d, "OpLoopMerge"), 0u);
}

// out = c ? a + 1 : a * 2, joined through a block parameter (OpPhi)
TEST_CASE("SPIR-V ISel - if-else with a joining block parameter") {
    Kernel k({Type::ptr(), Type::i32(), Type::i32()});
    Builder& b = k.b;
    BasicBlock* t = b.append_block("then");
    BasicBlock* e = b.append_block("else");
    BasicBlock* j = b.append_block("join");
    Value* x = b.add_block_param(j, Type::i32());
    b.position_at_end(k.entry);
    b.build_br_if(b.build_ne(k.p(1), b.build_iconst_i32(0)), t, e);
    b.position_at_end(t);
    b.build_br(j, {b.build_add(k.p(2), b.build_iconst_i32(1))});
    b.position_at_end(e);
    b.build_br(j, {b.build_mul(k.p(2), b.build_iconst_i32(2))});
    b.position_at_end(j);
    b.build_store(Type::i32(), k.p(0), 0, x);
    b.build_ret_void();
    compile_checked(*k.fn);
    std::string d = dump_of(*k.fn);
    CHECK_EQ(count_of(d, "OpPhi"), 1u);
    CHECK_EQ(count_of(d, "OpSelectionMerge"), 1u);
}

// for (i = tid; i < n; i += ntid) acc += in[i]; out[tid] = acc
TEST_CASE("SPIR-V ISel - loop with a carried value") {
    Kernel k({Type::ptr(), Type::ptr(), Type::i32()});
    codegen::KernelBuilder kb(k.b);
    Value* tid = kb.tid_x();
    Value* acc = kb.for_range_reduce(tid, k.p(2), kb.ntid_x(), kb.const_f32(0.0f), [&](Value* i, Value* a) {
        return k.b.build_add(a, kb.load_f32_indexed(k.p(0), i));
    });
    kb.store_f32_indexed(k.p(1), tid, acc);
    k.b.build_ret_void();
    compile_checked(*k.fn);
    std::string d = dump_of(*k.fn);
    CHECK_EQ(count_of(d, "OpLoopMerge"), 1u);
    CHECK(has(d, "OpPhi"));
}

TEST_CASE("SPIR-V ISel - nested and unrolled loops with ifs inside") {
    Kernel k({Type::ptr(), Type::i32(), Type::i32()});
    codegen::KernelBuilder kb(k.b);
    Value* total = kb.for_range_reduce(kb.const_i32(0), k.p(1), kb.const_i32(1), kb.const_f32(0.0f), [&](Value* i, Value* a) {
        Value* inner = kb.for_range_reduce(kb.const_i32(0), k.p(2), kb.const_i32(1), a, [&](Value* j, Value* acc) {
            Value* v = kb.load_f32_indexed(k.p(0), k.b.build_add(i, j));
            kb.if_then(k.b.build_slt(j, i), [&] { kb.store_f32_indexed(k.p(0), j, v); });
            return k.b.build_add(acc, v);
        }, 4);
        return inner;
    });
    kb.store_f32(k.p(0), total);
    k.b.build_ret_void();
    compile_checked(*k.fn);
    std::string d = dump_of(*k.fn);
    CHECK_EQ(count_of(d, "OpLoopMerge"), 3u); // outer + unrolled main + remainder
}

// for (...) { if (in[i] < 0) return; if (in[i] == 0) break; out[i] = 1 }
TEST_CASE("SPIR-V ISel - early return and break inside a loop") {
    Kernel k({Type::ptr(), Type::i32()});
    Builder& b = k.b;
    BasicBlock* head = b.append_block("head");
    BasicBlock* body = b.append_block("body");
    BasicBlock* ret_bb = b.append_block("early_ret");
    BasicBlock* check = b.append_block("check");
    BasicBlock* store = b.append_block("store");
    BasicBlock* exit = b.append_block("exit");
    Value* i = b.add_block_param(head, Type::i32());
    b.position_at_end(k.entry);
    b.build_br(head, {b.build_iconst_i32(0)});
    b.position_at_end(head);
    b.build_br_if(b.build_slt(i, k.p(1)), body, exit);
    b.position_at_end(body);
    Value* v = b.build_load_indexed(Type::i32(), k.p(0), i, 4, 0);
    b.build_br_if(b.build_slt(v, b.build_iconst_i32(0)), ret_bb, check);
    b.position_at_end(ret_bb);
    b.build_ret_void();
    b.position_at_end(check);
    b.build_br_if(b.build_eq(v, b.build_iconst_i32(0)), exit, store);
    b.position_at_end(store);
    b.build_store_indexed(Type::i32(), k.p(0), i, 4, 0, b.build_iconst_i32(1));
    b.build_br(head, {b.build_add(i, b.build_iconst_i32(1))});
    b.position_at_end(exit);
    b.build_ret_void();
    compile_checked(*k.fn);
}

// Two nested ifs whose arms jump straight to the outer join: the inner
// selection gets a fresh forwarding merge block.
TEST_CASE("SPIR-V ISel - nested ifs sharing one join block") {
    Kernel k({Type::ptr(), Type::i32(), Type::i32()});
    Builder& b = k.b;
    BasicBlock* inner = b.append_block("inner");
    BasicBlock* deep = b.append_block("deep");
    BasicBlock* join = b.append_block("join");
    Value* x = b.add_block_param(join, Type::i32());
    b.position_at_end(k.entry);
    b.build_br_if(b.build_sgt(k.p(1), b.build_iconst_i32(0)), inner, {}, join, {b.build_iconst_i32(1)});
    b.position_at_end(inner);
    b.build_br_if(b.build_sgt(k.p(2), b.build_iconst_i32(0)), deep, {}, join, {b.build_iconst_i32(2)});
    b.position_at_end(deep);
    b.build_br(join, {b.build_iconst_i32(3)});
    b.position_at_end(join);
    b.build_store(Type::i32(), k.p(0), 0, x);
    b.build_ret_void();
    compile_checked(*k.fn);
    std::string d = dump_of(*k.fn);
    CHECK_EQ(count_of(d, "OpSelectionMerge"), 2u);
    CHECK(has(d, "merge_bb")); // the forwarding merge block
}

// Both arms return: no reconvergence, one arm becomes the merge.
TEST_CASE("SPIR-V ISel - if-else where both arms return") {
    Kernel k({Type::ptr(), Type::i32()});
    Builder& b = k.b;
    BasicBlock* t = b.append_block("t");
    BasicBlock* e = b.append_block("e");
    b.position_at_end(k.entry);
    b.build_br_if(b.build_ne(k.p(1), b.build_iconst_i32(0)), t, e);
    b.position_at_end(t);
    b.build_store(Type::i32(), k.p(0), 0, b.build_iconst_i32(1));
    b.build_ret_void();
    b.position_at_end(e);
    b.build_store(Type::i32(), k.p(0), 0, b.build_iconst_i32(2));
    b.build_ret_void();
    compile_checked(*k.fn);
    CHECK_EQ(count_of(dump_of(*k.fn), "OpSelectionMerge"), 1u);
}

// do { out[i] = i; i++ } while (i < n): the latch's br_if has a continue and
// a break arm, so its selection merge is an unreachable block; and a loop
// left only through a block that returns.
TEST_CASE("SPIR-V ISel - do-while latch and a loop left by return") {
    {
        Kernel k({Type::ptr(), Type::i32()});
        Builder& b = k.b;
        BasicBlock* body = b.append_block("body");
        BasicBlock* exit = b.append_block("exit");
        Value* i = b.add_block_param(body, Type::i32());
        b.position_at_end(k.entry);
        b.build_br(body, {b.build_iconst_i32(0)});
        b.position_at_end(body);
        b.build_store_indexed(Type::i32(), k.p(0), i, 4, 0, i);
        Value* next = b.build_add(i, b.build_iconst_i32(1));
        b.build_br_if(b.build_slt(next, k.p(1)), body, {next}, exit, {});
        b.position_at_end(exit);
        b.build_ret_void();
        compile_checked(*k.fn);
        CHECK(has(dump_of(*k.fn), "OpUnreachable"));
    }
    {
        Kernel k({Type::ptr(), Type::i32()});
        Builder& b = k.b;
        BasicBlock* body = b.append_block("body");
        BasicBlock* done = b.append_block("done");
        Value* i = b.add_block_param(body, Type::i32());
        b.position_at_end(k.entry);
        b.build_br(body, {b.build_iconst_i32(0)});
        b.position_at_end(body);
        Value* next = b.build_add(i, b.build_iconst_i32(1));
        b.build_br_if(b.build_sge(next, k.p(1)), done, {}, body, {next});
        b.position_at_end(done);
        b.build_store(Type::i32(), k.p(0), 0, next);
        b.build_ret_void();
        compile_checked(*k.fn);
    }
}

// The entry block is the loop head (it has a back edge), and a br_if sends
// both edges to one block with different arguments.
TEST_CASE("SPIR-V ISel - loop at the entry block and a br_if with one target") {
    Kernel k({Type::ptr(), Type::i32()});
    Builder& b = k.b;
    BasicBlock* body = b.append_block("body");
    BasicBlock* same = b.append_block("same");
    BasicBlock* exit = b.append_block("exit");
    Value* s = b.add_block_param(same, Type::i32());
    b.position_at_end(k.entry);
    b.build_br_if(b.build_sgt(k.p(1), b.build_iconst_i32(0)), body, exit);
    b.position_at_end(body);
    b.build_br_if(b.build_slt(k.p(1), b.build_iconst_i32(5)), same, {b.build_iconst_i32(10)}, same, {b.build_iconst_i32(20)});
    b.position_at_end(same);
    b.build_store(Type::i32(), k.p(0), 0, s);
    b.build_br(k.entry, {k.p(0), b.build_sub(k.p(1), b.build_iconst_i32(1))});
    b.position_at_end(exit);
    b.build_ret_void();
    compile_checked(*k.fn);
}

// Loop-carried vector values (two parts each) through the phis.
TEST_CASE("SPIR-V ISel - loop-carried 256-bit vectors") {
    Kernel k({Type::ptr(), Type::i32()});
    codegen::KernelBuilder kb(k.b);
    Builder& b = k.b;
    auto accs = kb.for_range_reduce_n(kb.const_i32(0), k.p(1), kb.const_i32(1), {b.build_vzero(Type::f32x8()), b.build_vzero(Type::i64x4())},
        [&](Value* i, const std::vector<Value*>& a) {
            Value* off = b.build_mul(b.build_sext_i64(i), b.build_iconst_i64(32));
            Value* p = b.build_add(k.p(0), off);
            return std::vector<Value*>{b.build_vadd(a[0], b.build_vload(Type::f32x8(), p, 0)),
                                       b.build_vadd(a[1], b.build_vload(Type::i64x4(), p, 0))};
        });
    b.build_vstore(Type::f32x8(), k.p(0), 0, accs[0]);
    b.build_vstore(Type::i64x4(), k.p(0), 32, accs[1]);
    b.build_ret_void();
    compile_checked(*k.fn);
}

// Device finding (RADV): without NoContraction Mesa folds fpext(fptrunc(x))
// to x and fpext(sitofp_f32(i)) to sitofp_f64(i), dropping the f32 rounding.
// Every float conversion carries the decoration; integer ones do not.
TEST_CASE("SPIR-V ISel - float conversions are NoContraction") {
    Kernel k({Type::ptr(), Type::f64(), Type::i32(), Type::i64()});
    Builder& b = k.b;
    Value* t = b.build_fpext_f64_f32(b.build_fptrunc_f32_f64(k.p(1)));
    Value* s = b.build_fpext_f64_f32(b.build_sitofp_f32_i32(k.p(2)));
    Value* i = b.build_sitofp_f64_i64(b.build_fptosi_i64(k.p(1)));
    Value* u = b.build_call("ptx_u32_to_f32", Type::f32(), {k.p(2)});
    b.build_store(Type::f64(), k.p(0), 0, b.build_add(b.build_add(t, s), i));
    b.build_store(Type::f32(), k.p(0), 8, u);
    b.build_store(Type::i64(), k.p(0), 16, b.build_sext_i64(k.p(2)));
    b.build_ret_void();
    compile_checked(*k.fn);
    std::string d = dump_of(*k.fn);
    // fptrunc, fpext x2, sitofp x2, fptosi, ptx_u32_to_f32 (the dump prints
    // the decoration by number: NoContraction = 42)
    const std::string no_contraction = " " + std::to_string(static_cast<int>(spv::DecorationNoContraction)) + "\n";
    size_t n = 0;
    std::istringstream lines(d);
    for (std::string line; std::getline(lines, line);)
        if (line.rfind("OpDecorate %", 0) == 0 && has(line + "\n", no_contraction)) ++n;
    CHECK_EQ(n, 7u);
}

// ---------------------------------------------------------------------------
// Diagnostics
// ---------------------------------------------------------------------------

TEST_CASE("SPIR-V ISel - an irreducible CFG is diagnosed with the kernel and blocks") {
    Kernel k({Type::ptr(), Type::i32()}, "tangle");
    Builder& b = k.b;
    BasicBlock* a = b.append_block("a");
    BasicBlock* c = b.append_block("c");
    BasicBlock* exit = b.append_block("exit");
    b.position_at_end(k.entry);
    b.build_br_if(k.p(1), a, c);
    b.position_at_end(a);
    b.build_br_if(k.p(1), c, exit);
    b.position_at_end(c);
    b.build_br(a);
    b.position_at_end(exit);
    b.build_ret_void();
    require_diagnostic(*k.fn, {"kernel 'tangle'", "irreducible", "bb"});
}

TEST_CASE("SPIR-V ISel - a loop with two exit blocks is diagnosed") {
    Kernel k({Type::ptr(), Type::i32()}, "two_exits");
    Builder& b = k.b;
    BasicBlock* head = b.append_block("head");
    BasicBlock* body = b.append_block("body");
    BasicBlock* exit1 = b.append_block("exit1");
    BasicBlock* exit2 = b.append_block("exit2");
    BasicBlock* join = b.append_block("join");
    b.position_at_end(k.entry);
    b.build_br(head);
    b.position_at_end(head);
    b.build_br_if(k.p(1), body, exit1);
    b.position_at_end(body);
    b.build_br_if(k.p(1), head, exit2);
    b.position_at_end(exit1);
    b.build_store(Type::i32(), k.p(0), 0, k.p(1));
    b.build_br(join);
    b.position_at_end(exit2);
    b.build_br(join);
    b.position_at_end(join);
    b.build_ret_void();
    require_diagnostic(*k.fn, {"kernel 'two_exits'", "more than one block", "exit1", "exit2"});
}

TEST_CASE("SPIR-V ISel - unsupported opcodes, terminators, calls and non-void kernels") {
    {
        Kernel k({Type::ptr(), Type::i32()}, "bits");
        k.b.build_store(Type::i32(), k.p(0), 0, k.b.build_clz(k.p(1)));
        k.b.build_ret_void();
        require_diagnostic(*k.fn, {"kernel 'bits'", "unsupported opcode", "clz", "bb"});
    }
    {
        Kernel k({Type::ptr(), Type::i32()}, "sw");
        BasicBlock* d = k.b.append_block("d");
        k.b.position_at_end(k.entry);
        k.b.build_switch(k.p(1), d, std::initializer_list<SwitchCase>{});
        k.b.position_at_end(d);
        k.b.build_ret_void();
        require_diagnostic(*k.fn, {"kernel 'sw'", "switch"});
    }
    {
        Kernel k({Type::ptr()}, "caller");
        k.b.build_call("helper", Type::void_type(), {k.p(0)});
        k.b.build_ret_void();
        require_diagnostic(*k.fn, {"kernel 'caller'", "@helper", "cannot call functions"});
    }
    {
        Module mod("m");
        Function* fn = mod.create_function("valued", Type::i32(), {});
        Builder b(mod);
        b.set_function(fn);
        b.position_at_end(b.append_block("entry"));
        b.build_ret(b.build_iconst_i32(0));
        require_diagnostic(*fn, {"'valued'", "non-void"});
    }
}

TEST_CASE("SPIR-V ISel - a shared pointer flowing through a block parameter is diagnosed") {
    Kernel k({Type::ptr(), Type::i32()}, "escape");
    codegen::KernelBuilder kb(k.b);
    Builder& b = k.b;
    Value* smem = kb.shared_alloc_f32(64);
    BasicBlock* next = b.append_block("next");
    Value* p = b.add_block_param(next, Type::ptr());
    b.position_at_end(k.entry);
    b.build_br(next, {smem});
    b.position_at_end(next);
    kb.shared_store_f32(p, kb.const_f32(1.0f));
    b.build_ret_void();
    require_diagnostic(*k.fn, {"kernel 'escape'", "shared-memory pointer"});
}

} // namespace
