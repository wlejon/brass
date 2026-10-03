// SpirvTarget facade: the push-constant parameter layout, workgroup size as
// spec constants or a fixed LocalSize, multi-kernel modules, SPIR-V 1.6,
// the disassembly of a full kernel, and all ten fused ML kernels (built for
// PTX with KernelBuilder) lowering unchanged to modules spirv-val accepts.

#include "spirv_test_support.hpp"

#include <brass/codegen/ml_fusion.hpp>

#include <algorithm>

using namespace brass;
using brass::codegen::KernelBuilder;
using brass::codegen::MlFusionCompiler;
using namespace spvtest;

namespace {

bool has_cap(const target::SpirvKernel& k, const char* cap) {
    return std::find(k.capabilities.begin(), k.capabilities.end(), cap) != k.capabilities.end();
}

// y[i] = a * x[i] + y[i] for i < n (grid-stride)
Function* saxpy(Module& mod, const char* name = "saxpy") {
    Function* f = mod.create_function(name, Type::void_type(), {Type::ptr(), Type::ptr(), Type::f32(), Type::i32()});
    KernelBuilder kb(mod, f);
    Builder& b = kb.builder();
    BasicBlock* e = b.append_block("entry");
    b.position_at_end(e);
    Value* x = b.add_block_param(e, Type::ptr());
    Value* y = b.add_block_param(e, Type::ptr());
    Value* a = b.add_block_param(e, Type::f32());
    Value* n = b.add_block_param(e, Type::i32());
    Value* stride = b.build_mul(kb.ntid_x(), kb.nctaid_x());
    kb.for_range(kb.global_tid_x(), n, stride, [&](Value* i) {
        kb.store_f32_indexed(y, i, kb.fma(a, kb.load_f32_indexed(x, i), kb.load_f32_indexed(y, i)));
    });
    b.build_ret_void();
    return f;
}

TEST_CASE("SPIR-V Target - parameters are one push-constant block in declaration order") {
    Kernel k({Type::ptr(), Type::i32(), Type::f32(), Type::i64(), Type::f64(), Type::i32(), Type::ptr()}, "layout");
    Builder& b = k.b;
    b.build_store(Type::i32(), k.p(6), 0, b.build_add(k.p(1), k.p(5)));
    b.build_store(Type::f32(), k.p(0), 0, k.p(2));
    b.build_store(Type::i64(), k.p(0), 8, k.p(3));
    b.build_store(Type::f64(), k.p(0), 16, k.p(4));
    b.build_ret_void();
    target::SpirvKernel kern = compile_checked(*k.fn);
    CHECK(kern.entry == "layout");
    REQUIRE_EQ(kern.params.size(), 7u);
    const uint32_t offsets[] = {0, 8, 12, 16, 24, 32, 40};
    const uint32_t sizes[] = {8, 4, 4, 8, 8, 4, 8};
    for (size_t i = 0; i < 7; ++i) {
        CHECK_EQ(kern.params[i].offset, offsets[i]);
        CHECK_EQ(kern.params[i].size, sizes[i]);
    }
    CHECK_EQ(kern.push_constant_bytes, 48u);
    CHECK_EQ(kern.shared_bytes, 0u);
    CHECK(has_cap(kern, "Shader"));
    CHECK(has_cap(kern, "PhysicalStorageBufferAddresses"));
    CHECK(has_cap(kern, "Int64"));
    CHECK(has_cap(kern, "Float64"));

    std::string dis = disassemble(kern.words);
    if (!dis.empty()) {
        CHECK(has(dis, "OpDecorate %") && has(dis, " Block"));
        CHECK(has(dis, "Offset 40"));
        CHECK(has(dis, "PushConstant"));
        CHECK(!has(dis, "DescriptorSet"));
    }
}

TEST_CASE("SPIR-V Target - workgroup size as spec constants or a fixed LocalSize") {
    Module mod("ws");
    Function* f = saxpy(mod);
    target::SpirvOptions spec;
    spec.local_size_x = 128;
    target::SpirvKernel ks = compile_checked(*f, spec);
    CHECK_EQ(ks.local_size[0], 128u);

    target::SpirvOptions fixed;
    fixed.local_size_x = 64;
    fixed.local_size_spec_constants = false;
    target::SpirvKernel kf = compile_checked(*f, fixed);

    std::string ds = disassemble(ks.words);
    std::string df = disassemble(kf.words);
    if (!ds.empty()) {
        CHECK(has(ds, "SpecId 0"));
        CHECK(has(ds, "OpSpecConstant %uint 128") || has(ds, "OpSpecConstant") );
        CHECK(has(ds, "BuiltIn WorkgroupSize"));
        CHECK(has(ds, "LocalSize 128 1 1"));
        CHECK(!has(df, "SpecId"));
        CHECK(has(df, "LocalSize 64 1 1"));
    }
}

TEST_CASE("SPIR-V Target - a module with several kernels and SPIR-V 1.6") {
    Module mod("multi");
    saxpy(mod, "saxpy_a");
    saxpy(mod, "saxpy_b");
    std::vector<uint32_t> words = target::SpirvTarget::emit_module(mod);
    require_valid(words);
    std::string dis = disassemble(words);
    if (!dis.empty()) {
        CHECK_EQ(count_of(dis, "OpEntryPoint GLCompute"), 2u);
        CHECK_EQ(count_of(dis, "BuiltIn WorkgroupSize"), 1u); // one per module, shared by the entry points
    }

    target::SpirvOptions v16;
    v16.spirv_version = 0x00010600;
    std::vector<uint32_t> w16 = target::SpirvTarget::emit_function(*mod.functions()[0], v16);
    CHECK_EQ(w16[1], 0x00010600u);
    require_valid(w16);

    target::SpirvOptions old;
    old.spirv_version = 0x00010300;
    bool threw = false;
    try {
        target::SpirvTarget::emit_function(*mod.functions()[0], old);
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    CHECK(threw);
}

TEST_CASE("SPIR-V Target - disassembly of a kernel shows the structured loop") {
    Module mod("dis");
    Function* f = saxpy(mod);
    target::SpirvKernel k = compile_checked(*f);
    std::string text = dump_of(*f);
    CHECK(has(text, "OpEntryPoint GLCompute"));
    CHECK(has(text, "OpLoopMerge"));
    std::string dis = disassemble(k.words);
    if (dis.empty()) return;
    for (const char* op : {"OpCapability PhysicalStorageBufferAddresses", "OpMemoryModel PhysicalStorageBuffer64 GLSL450",
                           "OpEntryPoint GLCompute", "OpLoopMerge", "OpSelectionMerge", "OpBranchConditional", "OpPhi",
                           "OpConvertUToPtr", "Aligned 4", "OpExtInst", "Fma", "OpReturn", "BuiltIn GlobalInvocationId"}) {
        if (!has(dis, op)) std::cerr << "spirv-dis lacks '" << op << "'\n" << dis;
        CHECK(has(dis, op));
    }
}

TEST_CASE("SPIR-V Target - the ten fused ML kernels lower and validate unchanged") {
    MlFusionCompiler c;
    std::vector<std::unique_ptr<Module>> mods;
    auto fresh = [&](const char* n) -> Module& {
        mods.push_back(std::make_unique<Module>(n));
        return *mods.back();
    };
    std::vector<Function*> kernels = {
        c.build_ptx_swiglu(fresh("a")),
        c.build_ptx_swiglu_packed(fresh("b")),
        c.build_ptx_adaln_modulate(fresh("c"), false),
        c.build_ptx_adaln_modulate(fresh("d"), true),
        c.build_ptx_residual_rms_norm(fresh("e")),
        c.build_ptx_layernorm_modulate(fresh("f")),
        c.build_ptx_residual_layernorm(fresh("g")),
        c.build_ptx_gemv_swiglu(fresh("h")),
        c.build_ptx_gemv_residual(fresh("i")),
        c.build_ptx_gemv_q8_0(fresh("j")),
        c.build_ptx_gemv_q4_k(fresh("k")),
    };
    for (Function* f : kernels) {
        target::SpirvKernel k = compile_checked(*f);
        CHECK(k.entry == std::string(f->name()));
        CHECK_EQ(k.params.size(), f->param_types().size());
        // KernelBuilder's shapes need no forwarding merge blocks.
        std::string d = dump_of(*f);
        if (has(d, "\"merge_")) std::cerr << f->name() << " needed a forwarding merge block\n";
        CHECK(!has(d, "\"merge_"));
    }
}

} // namespace
