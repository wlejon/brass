// SPIR-V intrinsics: every entry of the SpirvISel table has a signature row
// below (the coverage test fails for names it does not know) and lowers to
// a module spirv-val accepts; every PTX intrinsic is either in the SPIR-V
// table or on the unsupported list, and each unsupported one throws with its
// reason. spirv-dis checks the opcode each family lowers to, and the
// KernelBuilder warp/block reductions validate as whole kernels.

#include "spirv_test_support.hpp"

#include <brass/target/ptx/ptx_isel.hpp>

#include <map>

using namespace brass;
using brass::codegen::KernelBuilder;
using brass::spirv::SpirvISel;
using namespace spvtest;

namespace {

enum class Sig {
    void_i32, void_i64, f32_f32, f32f32_f32, f64_f64, i32_f32, i64_f32, f32_i32, f32_f64, f64_f32,
    shfl_f32, shfl_i32, shfl_sync_f32, bar, bar_id, atom_f32, atom_i32, atom_i64, atom_shared_f32, atom_shared_i32,
    i32i32_i64, i32i32_i32, i32x3_i32, shared_alloc,
    shared_ld, shared_ldx, shared_st, shared_stx, // element type from the name's suffix
    load_narrow, store_narrow,
};

const std::map<std::string, Sig>& signatures() {
    static const std::map<std::string, Sig> kSigs = {
        {"ptx_tid_x", Sig::void_i32}, {"ptx_tid_y", Sig::void_i32}, {"ptx_tid_z", Sig::void_i32},
        {"ptx_ctaid_x", Sig::void_i32}, {"ptx_ctaid_y", Sig::void_i32}, {"ptx_ctaid_z", Sig::void_i32},
        {"ptx_ntid_x", Sig::void_i32}, {"ptx_ntid_y", Sig::void_i32}, {"ptx_ntid_z", Sig::void_i32},
        {"ptx_nctaid_x", Sig::void_i32}, {"ptx_nctaid_y", Sig::void_i32}, {"ptx_nctaid_z", Sig::void_i32},
        {"ptx_laneid", Sig::void_i32}, {"ptx_lane_id", Sig::void_i32},
        {"ptx_global_tid_x", Sig::void_i32}, {"ptx_global_id_x", Sig::void_i32},
        {"ptx_clock", Sig::void_i32}, {"ptx_clock64", Sig::void_i64},

        {"rsqrtf", Sig::f32_f32}, {"rsqrt", Sig::f32_f32}, {"ptx_rsqrt", Sig::f32_f32},
        {"sqrtf", Sig::f32_f32}, {"sqrt", Sig::f32_f32}, {"ptx_sqrt", Sig::f32_f32},
        {"sinf", Sig::f32_f32}, {"sin", Sig::f32_f32}, {"ptx_sin", Sig::f32_f32},
        {"cosf", Sig::f32_f32}, {"cos", Sig::f32_f32}, {"ptx_cos", Sig::f32_f32},
        {"ex2f", Sig::f32_f32}, {"ex2", Sig::f32_f32}, {"ptx_ex2", Sig::f32_f32},
        {"lg2f", Sig::f32_f32}, {"ptx_lg2", Sig::f32_f32},
        {"ptx_rcp", Sig::f32_f32}, {"ptx_rcp_approx", Sig::f32_f32},
        {"expf", Sig::f32_f32}, {"exp", Sig::f32_f32}, {"ptx_exp", Sig::f32_f32},
        {"logf", Sig::f32_f32}, {"log", Sig::f32_f32}, {"ptx_log", Sig::f32_f32},
        {"ptx_sqrt_rn", Sig::f64_f64},
        {"fabsf", Sig::f32_f32}, {"fabs", Sig::f64_f64}, {"ptx_fabs", Sig::f32_f32},
        {"fminf", Sig::f32f32_f32}, {"fmin", Sig::f32f32_f32}, {"ptx_fmin", Sig::f32f32_f32},
        {"fmaxf", Sig::f32f32_f32}, {"fmax", Sig::f32f32_f32}, {"ptx_fmax", Sig::f32f32_f32},
        {"ptx_div_approx", Sig::f32f32_f32},

        {"i32_to_f32", Sig::i32_f32}, {"ptx_i32_to_f32", Sig::i32_f32}, {"ptx_u32_to_f32", Sig::i32_f32},
        {"ptx_u64_to_f32", Sig::i64_f32}, {"ptx_i64_to_f32", Sig::i64_f32},
        {"ptx_f32_to_i32", Sig::f32_i32}, {"ptx_f32_to_u32", Sig::f32_i32},
        {"ptx_f16_to_f32", Sig::i32_f32}, {"ptx_f32_to_f16", Sig::f32_i32},
        {"ptx_f32_to_f64", Sig::f32_f64}, {"ptx_f64_to_f32", Sig::f64_f32},

        {"bar.sync", Sig::bar}, {"ptx_sync", Sig::bar}, {"ptx_bar_sync", Sig::bar_id},

        {"ptx_shfl_down_sync_f32", Sig::shfl_sync_f32}, {"shfl_down_sync_f32", Sig::shfl_sync_f32},
        {"ptx_shfl_down_f32", Sig::shfl_f32}, {"ptx_shfl_up_f32", Sig::shfl_f32},
        {"ptx_shfl_bfly_f32", Sig::shfl_f32}, {"ptx_shfl_xor_f32", Sig::shfl_f32}, {"ptx_shfl_idx_f32", Sig::shfl_f32},
        {"ptx_shfl_down_i32", Sig::shfl_i32}, {"ptx_shfl_up_i32", Sig::shfl_i32},
        {"ptx_shfl_bfly_i32", Sig::shfl_i32}, {"ptx_shfl_xor_i32", Sig::shfl_i32}, {"ptx_shfl_idx_i32", Sig::shfl_i32},

        {"ptx_atom_add_f32", Sig::atom_f32}, {"ptx_atom_add_i32", Sig::atom_i32}, {"ptx_atom_add_u32", Sig::atom_i32},
        {"ptx_atom_add_i64", Sig::atom_i64}, {"ptx_atom_min_i32", Sig::atom_i32}, {"ptx_atom_max_i32", Sig::atom_i32},
        {"ptx_atom_exch_i32", Sig::atom_i32},
        {"ptx_atom_shared_add_f32", Sig::atom_shared_f32}, {"ptx_atom_shared_add_i32", Sig::atom_shared_i32},

        {"ptx_mul_wide_u32", Sig::i32i32_i64}, {"ptx_mul_wide_s32", Sig::i32i32_i64},
        {"ptx_mul_hi_u32", Sig::i32i32_i32}, {"ptx_mad_lo_u32", Sig::i32x3_i32},

        {"ptx_shared_alloc_f32", Sig::shared_alloc}, {"ptx_shared_alloc_i32", Sig::shared_alloc},
        {"ptx_shared_alloc_f64", Sig::shared_alloc}, {"ptx_shared_alloc_i64", Sig::shared_alloc},
        {"ptx_shared_load_f32", Sig::shared_ld}, {"ptx_shared_load_i32", Sig::shared_ld},
        {"ptx_shared_load_f64", Sig::shared_ld}, {"ptx_shared_load_i64", Sig::shared_ld},
        {"ptx_shared_load_f32_indexed", Sig::shared_ldx}, {"ptx_shared_load_i32_indexed", Sig::shared_ldx},
        {"ptx_shared_load_f64_indexed", Sig::shared_ldx}, {"ptx_shared_load_i64_indexed", Sig::shared_ldx},
        {"ptx_shared_store_f32", Sig::shared_st}, {"ptx_shared_store_i32", Sig::shared_st},
        {"ptx_shared_store_f64", Sig::shared_st}, {"ptx_shared_store_i64", Sig::shared_st},
        {"ptx_shared_store_f32_indexed", Sig::shared_stx}, {"ptx_shared_store_i32_indexed", Sig::shared_stx},
        {"ptx_shared_store_f64_indexed", Sig::shared_stx}, {"ptx_shared_store_i64_indexed", Sig::shared_stx},

        {"ptx_load_u8", Sig::load_narrow}, {"ptx_load_s8", Sig::load_narrow},
        {"ptx_load_u16", Sig::load_narrow}, {"ptx_load_s16", Sig::load_narrow},
        {"ptx_store_u8", Sig::store_narrow}, {"ptx_store_u16", Sig::store_narrow},
    };
    return kSigs;
}

// Element type named by a shared intrinsic ("..._f32", "..._i64_indexed", ...).
Type shared_type(const std::string& name) {
    if (has(name, "_f64")) return Type::f64();
    if (has(name, "_i64")) return Type::i64();
    if (has(name, "_i32")) return Type::i32();
    return Type::f32();
}

// A kernel (i32 i, f32 x, f64 d, i64 l, ptr p) that calls `name` per its
// signature and stores any result through p. Shared arrays match the
// accessed element type (Workgroup arrays are typed on SPIR-V).
Function* build(Module& mod, const std::string& name, Sig sig) {
    Function* f = mod.create_function("intrin", Type::void_type(),
                                      {Type::i32(), Type::f32(), Type::f64(), Type::i64(), Type::ptr()});
    Builder b(mod);
    b.set_function(f);
    BasicBlock* e = b.append_block("entry");
    b.position_at_end(e);
    Value* i = b.add_block_param(e, Type::i32());
    Value* x = b.add_block_param(e, Type::f32());
    Value* d = b.add_block_param(e, Type::f64());
    Value* l = b.add_block_param(e, Type::i64());
    Value* p = b.add_block_param(e, Type::ptr());
    Value* c8 = b.build_iconst_i32(8);
    auto alloc = [&](Type t) {
        const char* n = t == Type::f64() ? "ptx_shared_alloc_f64" : t == Type::i64() ? "ptx_shared_alloc_i64"
                      : t == Type::i32() ? "ptx_shared_alloc_i32" : "ptx_shared_alloc_f32";
        return b.build_call(n, Type::ptr(), {b.build_iconst_i32(64)});
    };
    Type st = shared_type(name);
    Value* sval = st == Type::f64() ? d : st == Type::i64() ? l : st == Type::i32() ? i : x;

    Value* r = nullptr;
    switch (sig) {
        case Sig::void_i32: r = b.build_call(name, Type::i32()); break;
        case Sig::void_i64: r = b.build_call(name, Type::i64()); break;
        case Sig::f32_f32: r = b.build_call(name, Type::f32(), {x}); break;
        case Sig::f32f32_f32: r = b.build_call(name, Type::f32(), {x, x}); break;
        case Sig::f64_f64: r = b.build_call(name, Type::f64(), {d}); break;
        case Sig::i32_f32: r = b.build_call(name, Type::f32(), {i}); break;
        case Sig::i64_f32: r = b.build_call(name, Type::f32(), {l}); break;
        case Sig::f32_i32: r = b.build_call(name, Type::i32(), {x}); break;
        case Sig::f32_f64: r = b.build_call(name, Type::f64(), {x}); break;
        case Sig::f64_f32: r = b.build_call(name, Type::f32(), {d}); break;
        case Sig::shfl_f32: r = b.build_call(name, Type::f32(), {x, i}); break;  // register delta
        case Sig::shfl_i32: r = b.build_call(name, Type::i32(), {i, b.build_iconst_i32(4)}); break;
        case Sig::shfl_sync_f32: r = b.build_call(name, Type::f32(), {i, x, i}); break;
        case Sig::bar: b.build_call(name, Type::void_type()); break;
        case Sig::bar_id: b.build_call(name, Type::void_type(), {b.build_iconst_i32(1)}); break;
        case Sig::atom_f32: r = b.build_call(name, Type::f32(), {p, x}); break;
        case Sig::atom_i32: r = b.build_call(name, Type::i32(), {p, i}); break;
        case Sig::atom_i64: r = b.build_call(name, Type::i64(), {p, l}); break;
        case Sig::atom_shared_f32: r = b.build_call(name, Type::f32(), {alloc(Type::f32()), x}); break;
        case Sig::atom_shared_i32: r = b.build_call(name, Type::i32(), {alloc(Type::i32()), i}); break;
        case Sig::i32i32_i64: r = b.build_call(name, Type::i64(), {i, i}); break;
        case Sig::i32i32_i32: r = b.build_call(name, Type::i32(), {i, i}); break;
        case Sig::i32x3_i32: r = b.build_call(name, Type::i32(), {i, i, i}); break;
        case Sig::shared_alloc: {
            Value* s = b.build_call(name, Type::ptr(), {b.build_iconst_i32(16)});
            // round trip one element through the array, at a pointer offset with add
            Value* s8 = b.build_add(s, b.build_iconst_i64(static_cast<int64_t>(st.size_in_bytes())));
            std::string t = has(name, "f64") ? "f64" : has(name, "i64") ? "i64" : has(name, "i32") ? "i32" : "f32";
            b.build_call("ptx_shared_store_" + t, Type::void_type(), {s8, sval});
            r = b.build_call("ptx_shared_load_" + t, st, {s8});
            break;
        }
        case Sig::shared_ld: r = b.build_call(name, st, {alloc(st), b.build_iconst_i32(static_cast<int32_t>(st.size_in_bytes()))}); break;
        case Sig::shared_ldx: r = b.build_call(name, st, {alloc(st), i}); break;            // register index
        case Sig::shared_st: b.build_call(name, Type::void_type(), {alloc(st), sval, l}); break; // register byte offset
        case Sig::shared_stx: b.build_call(name, Type::void_type(), {alloc(st), c8, sval}); break;
        case Sig::load_narrow: r = b.build_call(name, Type::i32(), {p, b.build_iconst_i32(3)}); break;
        case Sig::store_narrow: b.build_call(name, Type::void_type(), {p, i, i}); break;
    }
    if (r) b.build_store(r->type(), p, 0, r);
    b.build_ret_void();
    return f;
}

std::vector<uint32_t> intrinsic_words(const std::string& name, Sig sig, std::string* dis = nullptr) {
    Module mod("intrin");
    Function* f = build(mod, name, sig);
    target::SpirvKernel k = compile_checked(*f);
    if (dis) *dis = disassemble(k.words);
    return k.words;
}

TEST_CASE("SPIR-V intrinsics - every table entry has a signature and validates") {
    const auto& sigs = signatures();
    for (const auto& [name, sig] : sigs) {
        if (!SpirvISel::is_intrinsic(name)) std::cerr << "SPIR-V intrinsic table is missing " << name << "\n";
        CHECK(SpirvISel::is_intrinsic(name));
    }
    for (std::string_view name : SpirvISel::intrinsic_names()) {
        auto it = sigs.find(std::string(name));
        if (it == sigs.end()) {
            std::cerr << "test_spirv_intrinsics.cpp has no signature for " << name << "\n";
            CHECK(false);
            continue;
        }
        intrinsic_words(it->first, it->second); // compile_checked REQUIREs ISel + verify + spirv-val
    }
}

TEST_CASE("SPIR-V intrinsics - every PTX intrinsic is lowered or diagnosed with a reason") {
    for (std::string_view name : ptx::PtxISel::intrinsic_names()) {
        bool lowered = SpirvISel::is_intrinsic(name);
        const char* reason = SpirvISel::unsupported_reason(name);
        if (lowered == (reason != nullptr)) std::cerr << name << ": lowered=" << lowered << " reason=" << (reason ? reason : "-") << "\n";
        CHECK(lowered != (reason != nullptr));
    }
    for (std::string_view name : SpirvISel::unsupported_names()) {
        CHECK(ptx::PtxISel::is_intrinsic(name)); // the list names real PTX intrinsics
        Module mod("u");
        Function* f = mod.create_function("unsupported_k", Type::void_type(), {Type::ptr()});
        Builder b(mod);
        b.set_function(f);
        BasicBlock* e = b.append_block("entry");
        b.position_at_end(e);
        Value* p = b.add_block_param(e, Type::ptr());
        Type rt = name == "ptx_globaltimer" ? Type::i64() : Type::i32();
        if (name == "ptx_bar_sync_count") {
            b.build_call(std::string(name), Type::void_type(), {b.build_iconst_i32(1), b.build_iconst_i32(64)});
        } else {
            b.build_store(rt, p, 0, b.build_call(std::string(name), rt));
        }
        b.build_ret_void();
        std::string reason = SpirvISel::unsupported_reason(name);
        require_diagnostic(*f, {"kernel 'unsupported_k'", std::string(name).c_str(), "no SPIR-V lowering", reason.c_str()});
    }
}

TEST_CASE("SPIR-V intrinsics - families lower to the expected SPIR-V (spirv-dis)") {
    if (!spirv_dis_available()) return;
    const auto& sigs = signatures();
    struct Expect { const char* name; std::vector<const char*> ops; };
    const Expect expects[] = {
        {"ptx_tid_x", {"BuiltIn LocalInvocationId", "OpCompositeExtract"}},
        {"ptx_ctaid_y", {"BuiltIn WorkgroupId"}},
        {"ptx_nctaid_z", {"BuiltIn NumWorkgroups"}},
        {"ptx_ntid_x", {"SpecId 0", "BuiltIn WorkgroupSize", "OpSpecConstantComposite"}},
        {"ptx_global_tid_x", {"BuiltIn GlobalInvocationId"}},
        {"ptx_laneid", {"BuiltIn SubgroupLocalInvocationId", "OpCapability GroupNonUniform"}},
        {"ptx_clock64", {"OpReadClockKHR", "OpCapability ShaderClockKHR", "SPV_KHR_shader_clock"}},
        {"ptx_rsqrt", {"InverseSqrt"}},
        {"ptx_ex2", {"Exp2"}},
        {"ptx_lg2", {"Log2"}},
        {"ptx_exp", {"Exp "}},
        {"ptx_rcp", {"OpFDiv"}},
        {"ptx_sqrt_rn", {"Sqrt", "OpTypeFloat 64"}},
        {"fminf", {"NMin"}},
        {"fmaxf", {"NMax"}},
        {"ptx_u32_to_f32", {"OpConvertUToF"}},
        {"ptx_f32_to_i32", {"OpConvertFToS"}},
        {"ptx_f16_to_f32", {"UnpackHalf2x16"}},
        {"ptx_f32_to_f16", {"PackHalf2x16"}},
        {"ptx_sync", {"OpControlBarrier"}},
        {"ptx_shfl_down_f32", {"OpGroupNonUniformShuffle ", "OpCapability GroupNonUniformShuffle"}},
        {"ptx_shfl_idx_i32", {"OpGroupNonUniformShuffle "}},
        {"ptx_atom_add_f32", {"OpAtomicFAddEXT", "SPV_EXT_shader_atomic_float_add", "AtomicFloat32AddEXT"}},
        {"ptx_atom_add_i64", {"OpAtomicIAdd", "Int64Atomics"}},
        {"ptx_atom_min_i32", {"OpAtomicSMin"}},
        {"ptx_atom_exch_i32", {"OpAtomicExchange"}},
        {"ptx_atom_shared_add_i32", {"OpAtomicIAdd", "Workgroup"}},
        {"ptx_mul_hi_u32", {"OpUMulExtended"}},
        {"ptx_shared_load_f32_indexed", {"OpAccessChain", "Workgroup"}},
        {"ptx_load_s8", {"OpCapability StorageBuffer8BitAccess", "OpSConvert", "OpTypeInt 8 0"}},
        {"ptx_store_u16", {"OpCapability StorageBuffer16BitAccess", "OpTypeInt 16 0"}},
    };
    for (const Expect& ex : expects) {
        std::string dis;
        intrinsic_words(ex.name, sigs.at(ex.name), &dis);
        for (const char* op : ex.ops) {
            if (!has(dis, op)) std::cerr << ex.name << ": spirv-dis output lacks '" << op << "'\n" << dis;
            CHECK(has(dis, op));
        }
    }
}

TEST_CASE("SPIR-V intrinsics - shared accesses must match the array element size") {
    Module mod("bad");
    Function* f = mod.create_function("mixed", Type::void_type(), {Type::ptr()});
    Builder b(mod);
    b.set_function(f);
    BasicBlock* e = b.append_block("entry");
    b.position_at_end(e);
    Value* p = b.add_block_param(e, Type::ptr());
    Value* s = b.build_call("ptx_shared_alloc_f32", Type::ptr(), {b.build_iconst_i32(64)});
    b.build_store(Type::f64(), p, 0, b.build_call("ptx_shared_load_f64", Type::f64(), {s}));
    b.build_ret_void();
    require_diagnostic(*f, {"kernel 'mixed'", "8-byte shared access", "f32"});

    // A non-constant element count.
    Module mod2("bad2");
    Function* g = mod2.create_function("dyn", Type::void_type(), {Type::i32()});
    Builder b2(mod2);
    b2.set_function(g);
    BasicBlock* e2 = b2.append_block("entry");
    b2.position_at_end(e2);
    Value* n = b2.add_block_param(e2, Type::i32());
    b2.build_call("ptx_shared_alloc_f32", Type::ptr(), {n});
    b2.build_ret_void();
    require_diagnostic(*g, {"kernel 'dyn'", "compile-time constant element count"});
}

TEST_CASE("SPIR-V intrinsics - KernelBuilder warp and block reductions validate") {
    Module mod("red");
    Function* f = mod.create_function("block_sum", Type::void_type(), {Type::ptr(), Type::ptr(), Type::i32()});
    KernelBuilder kb(mod, f);
    Builder& b = kb.builder();
    BasicBlock* e = b.append_block("entry");
    b.position_at_end(e);
    Value* in = b.add_block_param(e, Type::ptr());
    Value* out = b.add_block_param(e, Type::ptr());
    Value* n = b.add_block_param(e, Type::i32());
    Value* scratch = kb.shared_alloc_f32(32);
    Value* tid = kb.tid_x();
    Value* v = kb.for_range_reduce(tid, n, kb.ntid_x(), kb.const_f32(0.0f), [&](Value* i, Value* acc) {
        return b.build_add(acc, kb.load_f32_indexed(in, i));
    });
    Value* wmax = kb.warp_reduce_max_f32(v);
    Value* total = kb.block_reduce_sum_f32(v, scratch);
    kb.if_then(b.build_eq(tid, kb.const_i32(0)), [&] {
        kb.store_f32(out, total);
        kb.store_f32(out, wmax, 4);
        kb.atom_add_f32(out, total);
    });
    b.build_ret_void();
    target::SpirvKernel k = compile_checked(*f);
    CHECK_EQ(k.shared_bytes, 128u);
    std::string d = dump_of(*f);
    CHECK_EQ(count_of(d, "OpControlBarrier"), 2u);
    CHECK_EQ(count_of(d, "OpGroupNonUniformShuffle "), 16u); // 3 x 5-step butterflies + 1 broadcast
}

} // namespace
