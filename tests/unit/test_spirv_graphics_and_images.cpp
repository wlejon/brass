// Unit tests for:
//   - Issue #110: Descriptor Sets, 2D/3D Storage Images (OpImageRead/OpImageWrite),
//                 and Sampled Textures (OpTypeSampledImage, OpImageSampleExplicitLod)
//   - Issue #111: Graphics Rendering Stages (Vertex & Fragment Execution Models,
//                 BuiltIn Position/FragCoord/VertexIndex/InstanceIndex, Location I/O)
//   - Issue #113: SPIR-V Range-Proven RelaxedPrecision / Float16 Optimizer Pass
//                 (spirv::promote_relaxed_precision)

#include "spirv_test_support.hpp"

#include <brass/target/spirv/spirv_binary_reader.hpp>
#include <brass/target/spirv/spirv_relaxed_precision.hpp>

#include <algorithm>
#include <sstream>

using namespace brass;
using brass::codegen::KernelBuilder;
using namespace spvtest;

namespace {

bool has_diag_msg(const std::vector<spirv::Diagnostic>& diags, const std::string& needle) {
    for (const spirv::Diagnostic& d : diags) {
        if (d.message.find(needle) != std::string::npos) return true;
    }
    return false;
}

// ---------------------------------------------------------------------------
// Issue #110: Descriptor Sets, 2D/3D Storage Images & Sampled Textures
// ---------------------------------------------------------------------------

TEST_CASE("SPIR-V Descriptor Images & Samplers - 2D/3D storage images and sampled textures") {
    Module mod("img_mod");
    // Compute kernel combining:
    //   - (set 0, binding 0): 2D sampled texture
    //   - (set 0, binding 1): 2D storage image (read + write)
    //   - (set 1, binding 0): 3D sampled texture
    //   - (set 1, binding 1): 3D storage image (read + write)
    Function* fn = mod.create_function("volume_composite", Type::void_type(), {Type::f32()});
    KernelBuilder kb(mod, fn);
    Builder& b = kb.builder();
    BasicBlock* entry = b.append_block("entry");
    b.position_at_end(entry);
    Value* gain = b.add_block_param(entry, Type::f32());

    Value* ix = kb.tid_x();
    Value* iy = kb.tid_y();
    Value* iz = kb.tid_z();
    Value* fx = kb.i32_to_f32(ix);
    Value* fy = kb.i32_to_f32(iy);
    Value* fz = kb.i32_to_f32(iz);

    Value* inv_dim = kb.const_f32(1.0f / 64.0f);
    Value* u = kb.mul(fx, inv_dim);
    Value* v = kb.mul(fy, inv_dim);
    Value* w = kb.mul(fz, inv_dim);

    using target::SpirvImageDim;
    const uint32_t r32f = spv::ImageFormatR32f;
    KernelBuilder::ImageRef tex2 {0, 0, SpirvImageDim::D2};
    KernelBuilder::ImageRef img2 {0, 1, SpirvImageDim::D2};
    img2.format = r32f;
    KernelBuilder::ImageRef tex3 {1, 0, SpirvImageDim::D3};
    KernelBuilder::ImageRef img3 {1, 1, SpirvImageDim::D3};
    img3.format = r32f;
    auto red = [&](Value* texel) { return b.build_vextract_lane(texel, 0); };

    Value* tex2d = red(kb.texture_sample_lod(tex2, {u, v}, kb.const_f32(0.0f)));
    Value* prev2d = red(kb.image_load(img2, {ix, iy}));
    Value* out2d = kb.fma(tex2d, gain, prev2d);
    kb.image_store(img2, {ix, iy}, b.build_vbroadcast(Type::f32x4(), out2d));

    Value* tex3d = red(kb.texture_sample_lod(tex3, {u, v, w}, kb.const_f32(0.0f)));
    Value* prev3d = red(kb.image_load(img3, {ix, iy, iz}));
    Value* out3d = kb.add(prev3d, tex3d);
    kb.image_store(img3, {ix, iy, iz}, b.build_vbroadcast(Type::f32x4(), out3d));
    b.build_ret_void();

    target::SpirvOptions opts;
    opts.local_size_x = 8;
    opts.local_size_y = 8;
    opts.local_size_z = 4;
    target::SpirvKernel k = compile_checked(*fn, opts);

    CHECK(k.stage == target::SpirvStage::Compute);
    REQUIRE_EQ(k.descriptors.size(), 4u);
    CHECK_EQ(k.descriptors[0].set, 0u);
    CHECK_EQ(k.descriptors[0].binding, 0u);
    CHECK(k.descriptors[0].kind == target::SpirvDescriptorKind::CombinedImageSampler);
    CHECK(k.descriptors[0].dim == SpirvImageDim::D2);

    CHECK_EQ(k.descriptors[1].set, 0u);
    CHECK_EQ(k.descriptors[1].binding, 1u);
    CHECK(k.descriptors[1].kind == target::SpirvDescriptorKind::StorageImage);
    CHECK_EQ(k.descriptors[1].format, r32f);

    CHECK_EQ(k.descriptors[2].set, 1u);
    CHECK_EQ(k.descriptors[2].binding, 0u);
    CHECK(k.descriptors[2].kind == target::SpirvDescriptorKind::CombinedImageSampler);
    CHECK(k.descriptors[2].dim == SpirvImageDim::D3);

    CHECK_EQ(k.descriptors[3].set, 1u);
    CHECK_EQ(k.descriptors[3].binding, 1u);
    CHECK(k.descriptors[3].kind == target::SpirvDescriptorKind::StorageImage);
    CHECK(k.descriptors[3].dim == SpirvImageDim::D3);

    std::string dump_txt = dump_of(*fn, opts);
    CHECK(has(dump_txt, "OpTypeImage"));
    CHECK(has(dump_txt, "OpTypeSampledImage"));
    CHECK(has(dump_txt, "OpImageSampleExplicitLod"));
    CHECK(has(dump_txt, "OpImageRead"));
    CHECK(has(dump_txt, "OpImageWrite"));

    // Verify binary reader round-trip and static compute shader prover.
    spirv::Module parsed;
    std::string err;
    REQUIRE(spirv::parse_binary(k.words, parsed, &err));
    CHECK(spirv::verify(parsed).empty());
    CHECK(spirv::write(parsed) == k.words);

    spirv::ComputeShaderAuditReport audit = spirv::audit_compute_shader_binary(k.words);
    CHECK(audit.ok());
}

TEST_CASE("SPIR-V Descriptor Images & Samplers - vec4 RGBA storage images and sampled textures") {
    Module mod("img_vec4");
    Function* fn = mod.create_function("copy_rgba", Type::void_type(), {});
    KernelBuilder kb(mod, fn);
    Builder& b = kb.builder();
    BasicBlock* entry = b.append_block("entry");
    b.position_at_end(entry);

    Value* ix = kb.tid_x();
    Value* iy = kb.tid_y();
    Value* iz = kb.tid_z();
    Value* u = kb.i32_to_f32(ix);
    Value* v = kb.i32_to_f32(iy);
    Value* w = kb.i32_to_f32(iz);

    using target::SpirvImageDim;
    KernelBuilder::ImageRef tex2 {0, 0, SpirvImageDim::D2};
    KernelBuilder::ImageRef img2 {0, 1, SpirvImageDim::D2};
    img2.format = spv::ImageFormatRgba16f;
    KernelBuilder::ImageRef tex3 {0, 2, SpirvImageDim::D3};
    KernelBuilder::ImageRef img3 {0, 3, SpirvImageDim::D3};
    img3.format = spv::ImageFormatRgba32f;

    Value* s2 = kb.texture_sample_lod(tex2, {u, v}, kb.const_f32(0.0f));
    Value* ld2 = kb.image_load(img2, {ix, iy});
    kb.image_store(img2, {ix, iy}, kb.vadd(s2, ld2));

    Value* s3 = kb.texture_sample_lod(tex3, {u, v, w}, kb.const_f32(0.0f));
    Value* ld3 = kb.image_load(img3, {ix, iy, iz});
    kb.image_store(img3, {ix, iy, iz}, kb.vmul(s3, ld3));
    b.build_ret_void();

    target::SpirvKernel k = compile_checked(*fn);
    REQUIRE_EQ(k.descriptors.size(), 4u);
    CHECK_EQ(k.descriptors[1].format, static_cast<uint32_t>(spv::ImageFormatRgba16f));
    CHECK(spirv::audit_compute_shader_binary(k.words).ok());
}

// Arrays, cubes, depth comparison through a separate sampler, fetches, sizes,
// gathers, gradients and offsets, in a fragment stage: each validates.
TEST_CASE("SPIR-V Textures - arrays, cubes, shadows, separate samplers, fetch, size, gather") {
    Module mod("tex_kinds");
    Function* fn = mod.create_function("fs_kinds", Type::void_type(), {Type::f32()});
    KernelBuilder kb(mod, fn);
    Builder& b = kb.builder();
    BasicBlock* entry = b.append_block("entry");
    b.position_at_end(entry);
    Value* k = b.add_block_param(entry, Type::f32());
    using target::SpirvImageDim;

    KernelBuilder::ImageRef arr {0, 0, SpirvImageDim::D2};
    arr.arrayed = true;
    KernelBuilder::ImageRef cube {0, 1, SpirvImageDim::Cube};
    KernelBuilder::ImageRef shadow {0, 2, SpirvImageDim::D2};
    shadow.depth = true;
    shadow.sampler_set = 0;
    shadow.sampler_binding = 3;
    KernelBuilder::ImageRef plain {0, 4, SpirvImageDim::D2};

    Value* u = kb.frag_coord_x();
    Value* v = kb.frag_coord_y();
    Value* a = kb.texture_sample(arr, {u, v, kb.const_f32(2.0f)});
    Value* c = kb.texture_sample_lod(cube, {u, v, k}, kb.const_f32(1.0f));
    Value* s = kb.texture_compare(shadow, {u, v}, k, kb.const_f32(0.0f));
    Value* s2 = kb.texture_compare(shadow, {u, v}, k);
    Value* f = kb.texture_fetch(plain, {kb.const_i32(3), kb.const_i32(4)}, kb.const_i32(0));
    Value* g = kb.texture_gather(plain, {u, v}, kb.const_i32(1));
    Value* gr = kb.texture_sample_grad(plain, {u, v}, {k, k}, {k, k}, {kb.const_i32(1), kb.const_i32(-1)});
    Value* o = kb.texture_sample(plain, {u, v}, {kb.const_i32(-2), kb.const_i32(0)});
    Value* size = kb.texture_size(cube, kb.const_i32(0));
    Value* levels = kb.texture_levels(plain);
    Value* sum = kb.vadd(kb.vadd(kb.vadd(a, c), kb.vadd(f, g)), kb.vadd(gr, o));
    Value* extra = kb.add(kb.add(s, s2), kb.i32_to_f32(b.build_add(b.build_vextract_lane(size, 1), levels)));
    kb.set_frag_color_vec4(0, kb.vadd(sum, b.build_vbroadcast(Type::f32x4(), extra)));
    b.build_ret_void();

    target::SpirvOptions opts;
    opts.stage = target::SpirvStage::Fragment;
    target::SpirvKernel kern = compile_checked(*fn, opts);
    REQUIRE_EQ(kern.descriptors.size(), 5u);
    CHECK(kern.descriptors[0].arrayed);
    CHECK(kern.descriptors[1].dim == SpirvImageDim::Cube);
    CHECK(kern.descriptors[2].kind == target::SpirvDescriptorKind::SampledImage);
    CHECK(kern.descriptors[2].depth);
    CHECK(kern.descriptors[3].kind == target::SpirvDescriptorKind::Sampler);
    CHECK_EQ(kern.descriptors[3].binding, 3u);
    std::string dis = dump_of(*fn, opts);
    for (const char* op : {"OpImageSampleImplicitLod", "OpImageSampleDrefExplicitLod", "OpImageSampleDrefImplicitLod",
                           "OpImageFetch", "OpImageGather", "OpImageQuerySizeLod", "OpImageQueryLevels",
                           "OpSampledImage", "OpTypeSampler"}) {
        if (!has(dis, op)) std::cerr << "missing " << op << "\n";
        CHECK(has(dis, op));
    }
}

TEST_CASE("SPIR-V Descriptor Images & Samplers - conflicting descriptor binding kind is rejected") {
    Module mod("conflict");
    Function* fn = mod.create_function("bad_desc", Type::void_type(), {});
    KernelBuilder kb(mod, fn);
    Builder& b = kb.builder();
    BasicBlock* entry = b.append_block("entry");
    b.position_at_end(entry);

    Value* ix = kb.tid_x();
    KernelBuilder::ImageRef img2 {0, 0, target::SpirvImageDim::D2};
    img2.format = spv::ImageFormatR32f;
    KernelBuilder::ImageRef img3 = img2;
    img3.dim = target::SpirvImageDim::D3;
    Value* v = kb.image_load(img2, {ix, ix});
    // Same (set=0, binding=0) used as a 3D storage image:
    kb.image_store(img3, {ix, ix, ix}, v);
    b.build_ret_void();

    require_diagnostic(*fn, {"used as two different descriptors"});
}

// ---------------------------------------------------------------------------
// Issue #111: Graphics Rendering Stages (Vertex & Fragment Execution Models)
// ---------------------------------------------------------------------------

TEST_CASE("SPIR-V Graphics Stages - Vertex shader with VertexIndex, Location I/O, and Position") {
    Module mod("vs_mod");
    Function* fn = mod.create_function("vs_main", Type::void_type(), {Type::f32()});
    KernelBuilder kb(mod, fn);
    Builder& b = kb.builder();
    BasicBlock* entry = b.append_block("entry");
    b.position_at_end(entry);
    Value* scale = b.add_block_param(entry, Type::f32());

    Value* vidx = kb.vertex_index();
    Value* iidx = kb.instance_index();
    Value* in_x = kb.stage_input(0, 3, 0);
    Value* in_y = kb.stage_input(0, 3, 1);
    Value* in_z = kb.stage_input(0, 3, 2);
    Value* uv_u = kb.stage_input(1, 2, 0);
    Value* uv_v = kb.stage_input(1, 2, 1);
    Value* id = kb.stage_input(2, 1, 0, true);

    Value* inst_off = kb.mul(kb.i32_to_f32(b.build_add(vidx, iidx)), kb.const_f32(0.01f));
    Value* px = kb.fma(in_x, scale, inst_off);
    Value* py = kb.mul(in_y, scale);

    kb.set_vertex_position(px, py, in_z, kb.const_f32(1.0f));
    kb.stage_output(0, {uv_u, uv_v});
    kb.stage_output(1, {px, py, in_z, kb.const_f32(1.0f)});
    kb.stage_output(2, {id});
    b.build_ret_void();

    target::SpirvOptions opts;
    opts.stage = target::SpirvStage::Vertex;
    target::SpirvKernel k = compile_checked(*fn, opts);

    CHECK(k.stage == target::SpirvStage::Vertex);
    CHECK_EQ(k.local_size[0], 0u);
    CHECK(!k.local_size_spec_constants);

    std::string dump_txt = dump_of(*fn, opts);
    CHECK(has(dump_txt, "OpEntryPoint Vertex"));
    CHECK(!has(dump_txt, "LocalSize"));

    spirv::Module parsed;
    std::string err;
    REQUIRE(spirv::parse_binary(k.words, parsed, &err));
    CHECK(spirv::verify(parsed).empty());
    CHECK(spirv::write(parsed) == k.words);
}

TEST_CASE("SPIR-V Graphics Stages - Fragment shader with OriginUpperLeft, FragCoord, sampler2D, and FragColor") {
    Module mod("fs_mod");
    Function* fn = mod.create_function("fs_main", Type::void_type(), {Type::f32()});
    KernelBuilder kb(mod, fn);
    Builder& b = kb.builder();
    BasicBlock* entry = b.append_block("entry");
    b.position_at_end(entry);
    Value* exposure = b.add_block_param(entry, Type::f32());

    Value* fcx = kb.frag_coord_x();
    Value* fcy = kb.frag_coord_y();
    Value* u = kb.stage_input(0, 2, 0);
    Value* v = kb.stage_input(0, 2, 1);
    Value* id = kb.stage_input(1, 1, 0, true);
    Value* w = kb.stage_input(2, 1, 0, false, target::kSpirvStageFlat);

    KernelBuilder::ImageRef tex {0, 0, target::SpirvImageDim::D2};
    Value* tex_r = b.build_vextract_lane(kb.texture_sample_lod(tex, {u, v}, kb.const_f32(0.0f)), 0);
    Value* vig = kb.mul(kb.add(fcx, fcy), kb.const_f32(0.0005f));
    // A uniform block of 64 bytes (its word at byte 20, dynamically indexed
    // by the id too) and a storage buffer read and written.
    Value* gain = kb.buffer_load(1, 0, 64, b.build_iconst_i32(20), Type::f32());
    Value* row = kb.buffer_load(1, 0, 64, b.build_add(b.build_mul(id, b.build_iconst_i32(16)), b.build_iconst_i32(4)),
                                Type::f32());
    Value* count = kb.buffer_load(1, 1, 0, b.build_mul(id, b.build_iconst_i32(4)), Type::i32());
    kb.buffer_store(1, 1, b.build_iconst_i32(0), b.build_add(count, b.build_iconst_i32(1)));
    Value* r = kb.mul(kb.mul(tex_r, exposure), kb.add(gain, row));
    Value* g = kb.clamp_f32(u, kb.const_f32(0.0f), kb.const_f32(1.0f));
    Value* bl = kb.clamp_f32(b.build_sub(v, vig), kb.const_f32(0.0f), kb.const_f32(1.0f));

    kb.set_frag_color(0, r, g, bl, w);
    kb.stage_output(1, {r});
    kb.set_frag_depth(kb.mul(fcy, kb.const_f32(0.001f)));
    b.build_ret_void();

    target::SpirvOptions opts;
    opts.stage = target::SpirvStage::Fragment;
    target::SpirvKernel k = compile_checked(*fn, opts);

    CHECK(k.stage == target::SpirvStage::Fragment);
    REQUIRE_EQ(k.descriptors.size(), 3u);
    CHECK(k.descriptors[0].kind == target::SpirvDescriptorKind::CombinedImageSampler);
    CHECK(k.descriptors[1].kind == target::SpirvDescriptorKind::UniformBuffer);
    CHECK_EQ(k.descriptors[1].bytes, 64u);
    CHECK(k.descriptors[2].kind == target::SpirvDescriptorKind::StorageBuffer);

    std::string dump_txt = dump_of(*fn, opts);
    CHECK(has(dump_txt, "OpEntryPoint Fragment"));
    CHECK(!has(dump_txt, "LocalSize"));
    // A line "<op> %id <value>" (decorations and modes print as numbers).
    auto has_line = [&](const std::string& op, const std::string& value) {
        std::istringstream lines(dump_txt);
        std::string line;
        while (std::getline(lines, line)) {
            if (line.rfind(op + " %", 0) == 0 && line.size() > value.size() &&
                line.compare(line.size() - value.size() - 1, std::string::npos, " " + value) == 0)
                return true;
        }
        return false;
    };
    CHECK(has_line("OpExecutionMode", "12"));   // DepthReplacing
    CHECK(has_line("OpDecorate", "14"));        // Flat
    CHECK(has(dump_txt, "OpTypeRuntimeArray"));

    spirv::Module parsed;
    std::string err;
    REQUIRE(spirv::parse_binary(k.words, parsed, &err));
    CHECK(spirv::verify(parsed).empty());
}

TEST_CASE("SPIR-V verify - validates graphics execution modes and descriptor/stage decorations") {
    // 1. Fragment entry point without OriginUpperLeft is flagged by spirv::verify.
    {
        spirv::Module m;
        spirv::Id void_t = m.t_void();
        spirv::Function& f = m.add_function("fs", void_t, m.t_function(void_t, {}));
        f.blocks.push_back(spirv::Block{m.new_id(), {spirv::Inst(spv::OpReturn)}});
        m.add_entry_point(spv::ExecutionModelFragment, f.id, "fs", {});
        CHECK(has_diag_msg(spirv::verify(m), "OriginUpperLeft"));
    }
    // 2. Vertex entry point with LocalSize is flagged by spirv::verify.
    {
        spirv::Module m;
        spirv::Id void_t = m.t_void();
        spirv::Function& f = m.add_function("vs", void_t, m.t_function(void_t, {}));
        f.blocks.push_back(spirv::Block{m.new_id(), {spirv::Inst(spv::OpReturn)}});
        m.add_entry_point(spv::ExecutionModelVertex, f.id, "vs", {});
        m.add_execution_mode(f.id, spv::ExecutionModeLocalSize, {64, 1, 1});
        CHECK(has_diag_msg(spirv::verify(m), "must not declare LocalSize"));
    }
    // 3. UniformConstant variable without DescriptorSet/Binding is flagged.
    {
        spirv::Module m;
        spirv::Id void_t = m.t_void();
        spirv::Id img_t = m.t_image(m.t_float(32), spv::Dim2D, 0, 0, 0, 2, spv::ImageFormatR32f);
        spirv::Id var = m.variable(m.t_pointer(spv::StorageClassUniformConstant, img_t),
                                   spv::StorageClassUniformConstant);
        spirv::Function& f = m.add_function("cs", void_t, m.t_function(void_t, {}));
        f.blocks.push_back(spirv::Block{m.new_id(), {spirv::Inst(spv::OpReturn)}});
        m.add_entry_point(spv::ExecutionModelGLCompute, f.id, "cs", {var});
        m.add_execution_mode(f.id, spv::ExecutionModeLocalSize, {1, 1, 1});
        CHECK(has_diag_msg(spirv::verify(m), "missing DescriptorSet or Binding"));
    }
    // 4. Stage Input variable without Location or BuiltIn is flagged.
    {
        spirv::Module m;
        spirv::Id void_t = m.t_void();
        spirv::Id v4 = m.t_vector(m.t_float(32), 4);
        spirv::Id var = m.variable(m.t_pointer(spv::StorageClassInput, v4), spv::StorageClassInput);
        spirv::Function& f = m.add_function("vs", void_t, m.t_function(void_t, {}));
        f.blocks.push_back(spirv::Block{m.new_id(), {spirv::Inst(spv::OpReturn)}});
        m.add_entry_point(spv::ExecutionModelVertex, f.id, "vs", {var});
        CHECK(has_diag_msg(spirv::verify(m), "missing BuiltIn or Location"));
    }
}

// ---------------------------------------------------------------------------
// Issue #113: SPIR-V Range-Proven RelaxedPrecision / Float16 Optimizer Pass
// ---------------------------------------------------------------------------

TEST_CASE("SPIR-V RelaxedPrecision Prover - promotes bounded trig/clamp/mix/color chains and keeps world coords FP32") {
    Module mod("rp_mod");
    Function* fn = mod.create_function(
        "terrain_shade",
        Type::void_type(),
        {Type::ptr(), Type::f32(), Type::f32(), Type::f32()});
    KernelBuilder kb(mod, fn);
    Builder& b = kb.builder();
    BasicBlock* entry = b.append_block("entry");
    b.position_at_end(entry);

    Value* out_ptr = b.add_block_param(entry, Type::ptr());
    Value* world_origin = b.add_block_param(entry, Type::f32());
    Value* world_step = b.add_block_param(entry, Type::f32());
    Value* raw_roughness = b.add_block_param(entry, Type::f32());

    Value* tid = kb.i32_to_f32(kb.global_tid_x());

    // 1. Unbounded world-space coordinate math (must stay in full FP32):
    Value* world_offset = kb.mul(tid, world_step);
    Value* world_pos = kb.add(world_origin, world_offset);
    Value* large_scaled = kb.mul(world_pos, kb.const_f32(4096.0f));

    // 2. Bounded shading math in [-1, 1] / [0, 1] (must be promoted to RelaxedPrecision):
    Value* s = kb.sin_approx(world_pos);
    Value* c = kb.cos_approx(world_pos);
    Value* wave = kb.fma(s, kb.const_f32(0.5f), kb.const_f32(0.5f));         // [0, 1]
    Value* rough = kb.clamp_f32(raw_roughness, kb.const_f32(0.0f), kb.const_f32(1.0f)); // [0, 1]
    Value* mask = kb.smoothstep_f32(kb.const_f32(0.1f), kb.const_f32(0.9f), wave);      // [0, 1]
    Value* horizon = kb.tanh_approx(c);                                       // [-1, 1]
    Value* horizon_01 = kb.fma(horizon, kb.const_f32(0.5f), kb.const_f32(0.5f)); // [0, 1]
    Value* shaded = kb.mix_f32(rough, horizon_01, mask);                      // [0, 1]
    Value* energy = kb.sqrt_approx(shaded);                                   // [0, 1]

    // Store both the unbounded world position and the bounded shaded energy.
    kb.store_f32_indexed(out_ptr, kb.global_tid_x(), kb.add(large_scaled, energy));
    b.build_ret_void();

    target::SpirvKernel k = compile_checked(*fn);
    std::vector<uint32_t> words = k.words;

    spirv::SpirvRelaxedPrecisionReport rep = spirv::promote_relaxed_precision(words);
    REQUIRE(rep.ok());
    CHECK(rep.promoted_count >= 9u); // sin, cos, wave, rough, mask, horizon, horizon_01, shaded, energy
    CHECK(rep.kept_fp32_count >= 4u); // push-constant loads + tid + world_offset + world_pos + large_scaled + final sum

    // Verify that the unbounded world-space FMul (x * 4096.0f) and FAdd were NOT promoted,
    // while the bounded Sin, Cos, Tanh, FClamp, SmoothStep, FMix, Sqrt, Fma WERE promoted.
    uint32_t promoted_ext = 0;
    uint32_t kept_fp32_fadd_or_fmul = 0;
    for (const auto& e : rep.entries) {
        if (e.promoted) {
            CHECK(e.interval.fits_within(1024.0));
            if (e.op == spv::OpExtInst) ++promoted_ext;
        }
        if (e.op == spv::OpFAdd || e.op == spv::OpFMul) {
            if (!e.promoted) ++kept_fp32_fadd_or_fmul;
        }
    }
    CHECK(promoted_ext >= 9u);
    CHECK(kept_fp32_fadd_or_fmul >= 3u);

    // Verify the modified binary passes spirv::verify, audit_compute_shader_binary, and spirv-val.
    CHECK(spirv::verify_spirv_module(words));
    spirv::ComputeShaderAuditReport audit = spirv::audit_compute_shader_binary(words);
    CHECK(audit.ok());
    require_valid(words);

    std::string dis = disassemble(words);
    if (!dis.empty()) {
        CHECK(has(dis, "RelaxedPrecision"));
    }

    // Running a second time is idempotent: no duplicate OpDecorate RelaxedPrecision added.
    size_t words_after_first = words.size();
    spirv::SpirvRelaxedPrecisionReport rep2 = spirv::promote_relaxed_precision(words);
    REQUIRE(rep2.ok());
    CHECK_EQ(rep2.promoted_count, 0u);
    CHECK_EQ(rep2.already_relaxed_count, rep.promoted_count);
    CHECK_EQ(words.size(), words_after_first);
}

} // namespace

#ifdef BRASS_STANDALONE_TEST
BRASS_TEST_MAIN()
#endif
