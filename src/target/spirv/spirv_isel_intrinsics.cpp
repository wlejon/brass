// SpirvISel: builtin calls. A `call` whose callee is in the table below is
// lowered inline; a PTX intrinsic with no SPIR-V meaning throws with the
// reason from the unsupported list; any other callee throws (a compute
// entry point calls nothing).
//
// Adding an intrinsic: a rule in SpirvISel::Intrinsics
// (spirv_isel_intrinsics.hpp), one row per name below, a row in the
// signature table of tests/unit/test_spirv_intrinsics.cpp (its coverage
// check fails for names it does not know) and a line in the mapping table of
// docs/spirv_backend_design.md.

#include "spirv_isel_intrinsics.hpp"

#include <spirv/unified1/GLSL.std.450.h>

#include <algorithm>
#include <string>

namespace brass::spirv {

// ---------------------------------------------------------------------------
// Builtins
// ---------------------------------------------------------------------------

Id SpirvISel::Intrinsics::raw_subgroup_lane(SpirvISel& isel) {
    isel.m_.add_capability(spv::CapabilityGroupNonUniform);
    return isel.builtin_scalar(spv::BuiltInSubgroupLocalInvocationId);
}

// PTX %laneid is 0..31. The subgroup lane modulo 32 keeps that meaning on a
// 64-wide subgroup too (and the shuffles below work on 32-lane segments), so
// warp-level code is correct for subgroup sizes 32 and 64.
Id SpirvISel::Intrinsics::lane_id(SpirvISel& isel) {
    Id raw = raw_subgroup_lane(isel);
    return isel.prologue_value("laneid", [&] {
        return isel.op(spv::OpBitwiseAnd, isel.t_u32(), {raw, isel.m_.c_u32(31)});
    });
}

void SpirvISel::Intrinsics::laneid(SpirvISel& isel, const brass::Instruction& inst) {
    isel.define(inst.result(), lane_id(isel));
}

void SpirvISel::Intrinsics::global_tid_x(SpirvISel& isel, const brass::Instruction& inst) {
    isel.define(inst.result(), isel.builtin_component(spv::BuiltInGlobalInvocationId, 0));
}

// %clock / %clock64: the subgroup-scope shader clock (SPV_KHR_shader_clock,
// VK_KHR_shader_clock), read at every use like the PTX registers.
template <bool Wide>
void SpirvISel::Intrinsics::clock(SpirvISel& isel, const brass::Instruction& inst) {
    isel.m_.add_capability(spv::CapabilityShaderClockKHR);
    isel.m_.add_extension("SPV_KHR_shader_clock");
    Id t = isel.op(spv::OpReadClockKHR, isel.t_u64(), {isel.m_.c_u32(spv::ScopeSubgroup)});
    isel.define(inst.result(), Wide ? t : isel.op(spv::OpUConvert, isel.t_u32(), {t}));
}

// ---------------------------------------------------------------------------
// Math
// ---------------------------------------------------------------------------

template <uint32_t Glsl>
void SpirvISel::Intrinsics::glsl_unary(SpirvISel& isel, const brass::Instruction& inst) {
    const Value* a = inst.operand(0);
    if (!a || !a->type().is_float()) isel.malformed(inst, "argument 0 must be f32 or f64");
    isel.define(inst.result(), isel.ext(Glsl, isel.scalar_type(a->type()), {isel.id_of(a, "argument 0")}));
}

template <uint32_t Glsl>
void SpirvISel::Intrinsics::glsl_binary(SpirvISel& isel, const brass::Instruction& inst) {
    const Value* a = inst.operand(0);
    if (!a || !a->type().is_float()) isel.malformed(inst, "argument 0 must be f32 or f64");
    isel.define(inst.result(), isel.ext(Glsl, isel.scalar_type(a->type()),
                                        {isel.id_of(a, "argument 0"), isel.id_of(inst.operand(1), "argument 1")}));
}

void SpirvISel::Intrinsics::rcp(SpirvISel& isel, const brass::Instruction& inst) {
    isel.define(inst.result(), isel.op(spv::OpFDiv, isel.t_f32(), {isel.m_.c_f32(1.0f), isel.id_of(inst.operand(0), "argument 0")}));
}

void SpirvISel::Intrinsics::fdiv(SpirvISel& isel, const brass::Instruction& inst) {
    isel.define(inst.result(), isel.op(spv::OpFDiv, isel.t_f32(),
                                       {isel.id_of(inst.operand(0), "argument 0"), isel.id_of(inst.operand(1), "argument 1")}));
}

// ---------------------------------------------------------------------------
// Conversions
// ---------------------------------------------------------------------------

template <spv::Op O, uint32_t Bits, bool Float>
void SpirvISel::Intrinsics::convert(SpirvISel& isel, const brass::Instruction& inst) {
    Id t = Float ? isel.m_.t_float(Bits) : isel.m_.t_int(Bits);
    isel.define(inst.result(), isel.op(O, t, {isel.id_of(inst.operand(0), "argument 0")}));
}

// The low 16 bits of the i32 as an IEEE half (UnpackHalf2x16, component 0).
void SpirvISel::Intrinsics::f16_to_f32(SpirvISel& isel, const brass::Instruction& inst) {
    Id v2 = isel.ext(GLSLstd450UnpackHalf2x16, isel.m_.t_vector(isel.t_f32(), 2), {isel.id_of(inst.operand(0), "argument 0")});
    Id r = isel.m_.new_id();
    isel.emit(Inst(spv::OpCompositeExtract, isel.t_f32(), r).id(v2).lit(0));
    isel.define(inst.result(), r);
}

// The half in the low 16 bits, zeros above (PackHalf2x16 of (x, 0)).
void SpirvISel::Intrinsics::f32_to_f16(SpirvISel& isel, const brass::Instruction& inst) {
    Id v2 = isel.op(spv::OpCompositeConstruct, isel.m_.t_vector(isel.t_f32(), 2),
                    {isel.id_of(inst.operand(0), "argument 0"), isel.m_.c_f32(0.0f)});
    isel.define(inst.result(), isel.ext(GLSLstd450PackHalf2x16, isel.t_u32(), {v2}));
}

// ---------------------------------------------------------------------------
// The table
// ---------------------------------------------------------------------------

const SpirvISel::Intrinsics::Table& SpirvISel::Intrinsics::table() {
    static const Table kIntrinsics = {
        // Indices                                                       -> i32
        { "ptx_tid_x",        &builtin<spv::BuiltInLocalInvocationId, 0> },
        { "ptx_tid_y",        &builtin<spv::BuiltInLocalInvocationId, 1> },
        { "ptx_tid_z",        &builtin<spv::BuiltInLocalInvocationId, 2> },
        { "ptx_ctaid_x",      &builtin<spv::BuiltInWorkgroupId, 0> },
        { "ptx_ctaid_y",      &builtin<spv::BuiltInWorkgroupId, 1> },
        { "ptx_ctaid_z",      &builtin<spv::BuiltInWorkgroupId, 2> },
        { "ptx_nctaid_x",     &builtin<spv::BuiltInNumWorkgroups, 0> },
        { "ptx_nctaid_y",     &builtin<spv::BuiltInNumWorkgroups, 1> },
        { "ptx_nctaid_z",     &builtin<spv::BuiltInNumWorkgroups, 2> },
        { "ptx_ntid_x",       &ntid<0> },
        { "ptx_ntid_y",       &ntid<1> },
        { "ptx_ntid_z",       &ntid<2> },
        { "ptx_laneid",       &laneid },
        { "ptx_lane_id",      &laneid },
        { "ptx_global_tid_x", &global_tid_x },
        { "ptx_global_id_x",  &global_tid_x },
        { "ptx_clock",        &clock<false> },
        { "ptx_clock64",      &clock<true> },                         // -> i64

        // Math                                                   (f32) -> f32
        { "rsqrtf",         &glsl_unary<GLSLstd450InverseSqrt> },
        { "rsqrt",          &glsl_unary<GLSLstd450InverseSqrt> },
        { "ptx_rsqrt",      &glsl_unary<GLSLstd450InverseSqrt> },
        { "sqrtf",          &glsl_unary<GLSLstd450Sqrt> },
        { "sqrt",           &glsl_unary<GLSLstd450Sqrt> },
        { "ptx_sqrt",       &glsl_unary<GLSLstd450Sqrt> },
        { "sinf",           &glsl_unary<GLSLstd450Sin> },
        { "sin",            &glsl_unary<GLSLstd450Sin> },
        { "ptx_sin",        &glsl_unary<GLSLstd450Sin> },
        { "cosf",           &glsl_unary<GLSLstd450Cos> },
        { "cos",            &glsl_unary<GLSLstd450Cos> },
        { "ptx_cos",        &glsl_unary<GLSLstd450Cos> },
        { "ex2f",           &glsl_unary<GLSLstd450Exp2> },
        { "ex2",            &glsl_unary<GLSLstd450Exp2> },
        { "ptx_ex2",        &glsl_unary<GLSLstd450Exp2> },
        { "lg2f",           &glsl_unary<GLSLstd450Log2> },
        { "ptx_lg2",        &glsl_unary<GLSLstd450Log2> },
        { "expf",           &glsl_unary<GLSLstd450Exp> },
        { "exp",            &glsl_unary<GLSLstd450Exp> },
        { "ptx_exp",        &glsl_unary<GLSLstd450Exp> },
        { "logf",           &glsl_unary<GLSLstd450Log> },
        { "log",            &glsl_unary<GLSLstd450Log> },
        { "ptx_log",        &glsl_unary<GLSLstd450Log> },
        { "ptx_rcp",        &rcp },
        { "ptx_rcp_approx", &rcp },
        { "ptx_div_approx", &fdiv },                                   // (f32, f32) -> f32
        // (f32|f64) -> same
        { "ptx_sqrt_rn",    &glsl_unary<GLSLstd450Sqrt> },
        { "fabsf",          &glsl_unary<GLSLstd450FAbs> },
        { "fabs",           &glsl_unary<GLSLstd450FAbs> },
        { "ptx_fabs",       &glsl_unary<GLSLstd450FAbs> },
        { "fminf",          &glsl_binary<GLSLstd450NMin> },
        { "fmin",           &glsl_binary<GLSLstd450NMin> },
        { "ptx_fmin",       &glsl_binary<GLSLstd450NMin> },
        { "fmaxf",          &glsl_binary<GLSLstd450NMax> },
        { "fmax",           &glsl_binary<GLSLstd450NMax> },
        { "ptx_fmax",       &glsl_binary<GLSLstd450NMax> },

        // Conversions
        { "i32_to_f32",     &convert<spv::OpConvertSToF, 32, true> },
        { "ptx_i32_to_f32", &convert<spv::OpConvertSToF, 32, true> },
        { "ptx_u32_to_f32", &convert<spv::OpConvertUToF, 32, true> },
        { "ptx_u64_to_f32", &convert<spv::OpConvertUToF, 32, true> },
        { "ptx_i64_to_f32", &convert<spv::OpConvertSToF, 32, true> },
        { "ptx_f32_to_i32", &convert<spv::OpConvertFToS, 32, false> },
        { "ptx_f32_to_u32", &convert<spv::OpConvertFToU, 32, false> },
        { "ptx_f32_to_f64", &convert<spv::OpFConvert, 64, true> },
        { "ptx_f64_to_f32", &convert<spv::OpFConvert, 32, true> },
        { "ptx_f16_to_f32", &f16_to_f32 },
        { "ptx_f32_to_f16", &f32_to_f16 },

        // Barriers                                                     -> void
        { "bar.sync",       &bar_sync },
        { "ptx_sync",       &bar_sync },
        { "ptx_bar_sync",   &bar_sync_id },

        // Shuffles (32-lane segments of the subgroup)
        { "ptx_shfl_down_sync_f32", &shfl_down_sync_f32 },
        { "shfl_down_sync_f32",     &shfl_down_sync_f32 },
        { "ptx_shfl_down_f32",  &shfl<Shfl::down> },
        { "ptx_shfl_up_f32",    &shfl<Shfl::up> },
        { "ptx_shfl_bfly_f32",  &shfl<Shfl::bfly> },
        { "ptx_shfl_xor_f32",   &shfl<Shfl::bfly> },
        { "ptx_shfl_idx_f32",   &shfl<Shfl::idx> },
        { "ptx_shfl_down_i32",  &shfl<Shfl::down> },
        { "ptx_shfl_up_i32",    &shfl<Shfl::up> },
        { "ptx_shfl_bfly_i32",  &shfl<Shfl::bfly> },
        { "ptx_shfl_xor_i32",   &shfl<Shfl::bfly> },
        { "ptx_shfl_idx_i32",   &shfl<Shfl::idx> },

        // Atomics (ptr, value) -> old value
        { "ptx_atom_add_f32",        &atom<Atom::add, false> },
        { "ptx_atom_add_i32",        &atom<Atom::add, false> },
        { "ptx_atom_add_u32",        &atom<Atom::add, false> },
        { "ptx_atom_add_i64",        &atom<Atom::add, false> },
        { "ptx_atom_min_i32",        &atom<Atom::min, false> },
        { "ptx_atom_max_i32",        &atom<Atom::max, false> },
        { "ptx_atom_exch_i32",       &atom<Atom::exch, false> },
        { "ptx_atom_shared_add_f32", &atom<Atom::add, true> },
        { "ptx_atom_shared_add_i32", &atom<Atom::add, true> },

        // Wide / fused integer multiplies
        { "ptx_mul_wide_u32", &mul_wide<false> },
        { "ptx_mul_wide_s32", &mul_wide<true> },
        { "ptx_mul_hi_u32",   &mul_hi_u32 },
        { "ptx_mad_lo_u32",   &mad_lo_u32 },

        // Shared memory
        { "ptx_shared_alloc_f32", &shared_alloc<TypeKind::F32> },
        { "ptx_shared_alloc_i32", &shared_alloc<TypeKind::I32> },
        { "ptx_shared_alloc_f64", &shared_alloc<TypeKind::F64> },
        { "ptx_shared_alloc_i64", &shared_alloc<TypeKind::I64> },
        { "ptx_shared_load_f32",  &shared_load<TypeKind::F32, false> },
        { "ptx_shared_load_i32",  &shared_load<TypeKind::I32, false> },
        { "ptx_shared_load_f64",  &shared_load<TypeKind::F64, false> },
        { "ptx_shared_load_i64",  &shared_load<TypeKind::I64, false> },
        { "ptx_shared_load_f32_indexed", &shared_load<TypeKind::F32, true> },
        { "ptx_shared_load_i32_indexed", &shared_load<TypeKind::I32, true> },
        { "ptx_shared_load_f64_indexed", &shared_load<TypeKind::F64, true> },
        { "ptx_shared_load_i64_indexed", &shared_load<TypeKind::I64, true> },
        { "ptx_shared_store_f32", &shared_store<TypeKind::F32, false> },
        { "ptx_shared_store_i32", &shared_store<TypeKind::I32, false> },
        { "ptx_shared_store_f64", &shared_store<TypeKind::F64, false> },
        { "ptx_shared_store_i64", &shared_store<TypeKind::I64, false> },
        { "ptx_shared_store_f32_indexed", &shared_store<TypeKind::F32, true> },
        { "ptx_shared_store_i32_indexed", &shared_store<TypeKind::I32, true> },
        { "ptx_shared_store_f64_indexed", &shared_store<TypeKind::F64, true> },
        { "ptx_shared_store_i64_indexed", &shared_store<TypeKind::I64, true> },

        // Narrow global access (8/16-bit storage)
        { "ptx_load_u8",   &load_narrow<8, false> },
        { "ptx_load_s8",   &load_narrow<8, true> },
        { "ptx_load_u16",  &load_narrow<16, false> },
        { "ptx_load_s16",  &load_narrow<16, true> },
        { "ptx_store_u8",  &store_narrow<8> },
        { "ptx_store_u16", &store_narrow<16> },
    };
    return kIntrinsics;
}

// PTX intrinsics with no honest SPIR-V lowering.
const std::unordered_map<std::string_view, const char*>& SpirvISel::Intrinsics::unsupported() {
    static const std::unordered_map<std::string_view, const char*> kUnsupported = {
        { "ptx_warpid",  "%warpid is the warp's hardware scheduler slot, which Vulkan does not expose "
                         "(KernelBuilder::warp_id(), tid.x >> 5, is the warp index within the block)" },
        { "ptx_warp_id", "%warpid is the warp's hardware scheduler slot, which Vulkan does not expose "
                         "(KernelBuilder::warp_id(), tid.x >> 5, is the warp index within the block)" },
        { "ptx_nwarpid", "%nwarpid (warp slots per SM) has no SPIR-V equivalent" },
        { "ptx_smid",    "%smid (the SM a thread runs on) has no core SPIR-V equivalent" },
        { "ptx_nsmid",   "%nsmid (the number of SMs) has no core SPIR-V equivalent" },
        { "ptx_globaltimer", "%globaltimer is a nanosecond clock; SPIR-V's device-scope OpReadClockKHR counts "
                             "ticks of an unspecified frequency" },
        { "ptx_bar_sync_count", "bar.sync with a thread count synchronizes a subset of the block; "
                                "OpControlBarrier always waits for the whole workgroup" },
    };
    return kUnsupported;
}

SpirvISel::IntrinsicLowering SpirvISel::find_intrinsic(std::string_view callee) noexcept {
    const auto& table = Intrinsics::table();
    auto it = table.find(callee);
    return it == table.end() ? nullptr : it->second;
}

const char* SpirvISel::unsupported_reason(std::string_view callee) noexcept {
    const auto& u = Intrinsics::unsupported();
    auto it = u.find(callee);
    return it == u.end() ? nullptr : it->second;
}

std::vector<std::string_view> SpirvISel::intrinsic_names() {
    std::vector<std::string_view> names;
    for (const auto& e : Intrinsics::table()) names.push_back(e.first);
    std::sort(names.begin(), names.end());
    return names;
}

std::vector<std::string_view> SpirvISel::unsupported_names() {
    std::vector<std::string_view> names;
    for (const auto& e : Intrinsics::unsupported()) names.push_back(e.first);
    std::sort(names.begin(), names.end());
    return names;
}

void SpirvISel::lower_call(const brass::Instruction& inst) {
    std::string_view callee = inst.symbol();
    if (IntrinsicLowering rule = find_intrinsic(callee)) {
        rule(*this, inst);
        return;
    }
    if (const char* reason = unsupported_reason(callee)) {
        fail("intrinsic '" + std::string(callee) + "' has no SPIR-V lowering: " + reason);
    }
    fail("call to '@" + std::string(callee) +
         "': a SPIR-V compute kernel cannot call functions; only the GPU intrinsics are lowered");
}

} // namespace brass::spirv
