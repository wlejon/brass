#pragma once

// Private to the SPIR-V ISel: the lowering rules behind the intrinsic table.
// The names are the PTX backend's (docs/ptx_kernel_authoring.md), so one
// KernelBuilder kernel lowers to both targets. Each rule is
// `void(SpirvISel&, const brass::Instruction&)`; it reads its arguments with
// `isel.id_of` / `isel.const_int`, emits through `isel.op` / `isel.ext` and
// defines the call's result with `isel.define`.
//
//   spirv_isel_intrinsics.cpp      the table, the unsupported list, builtins,
//                                  math, conversions, `call`
//   spirv_isel_intrinsics_mem.cpp  barriers, shuffles, atomics, wide
//                                  multiplies, shared memory, narrow access

#include <brass/target/spirv/spirv_isel.hpp>

#include <string_view>
#include <unordered_map>

namespace brass::spirv {

struct SpirvISel::Intrinsics {
    using Table = std::unordered_map<std::string_view, IntrinsicLowering>;
    static const Table& table();
    static const std::unordered_map<std::string_view, const char*>& unsupported();

    // ---- builtins / math / conversions (spirv_isel_intrinsics.cpp) --------
    template <spv::BuiltIn B, uint32_t C>
    static void builtin(SpirvISel& isel, const brass::Instruction& inst) {
        isel.define(inst.result(), isel.builtin_component(B, C));
    }
    template <uint32_t C>
    static void ntid(SpirvISel& isel, const brass::Instruction& inst) {
        isel.define(inst.result(), isel.workgroup_size(C));
    }
    static Id lane_id(SpirvISel& isel);         // SubgroupLocalInvocationId & 31
    static Id raw_subgroup_lane(SpirvISel& isel);
    static void laneid(SpirvISel& isel, const brass::Instruction& inst);
    static void global_tid_x(SpirvISel& isel, const brass::Instruction& inst);
    template <bool Wide>
    static void clock(SpirvISel& isel, const brass::Instruction& inst);
    template <uint32_t Glsl>
    static void glsl_unary(SpirvISel& isel, const brass::Instruction& inst);  // type from argument 0
    template <uint32_t Glsl>
    static void glsl_binary(SpirvISel& isel, const brass::Instruction& inst); // type from argument 0
    static void rcp(SpirvISel& isel, const brass::Instruction& inst);
    static void fdiv(SpirvISel& isel, const brass::Instruction& inst);
    template <spv::Op O, uint32_t Bits, bool Float>
    static void convert(SpirvISel& isel, const brass::Instruction& inst);
    static void f16_to_f32(SpirvISel& isel, const brass::Instruction& inst);
    static void f32_to_f16(SpirvISel& isel, const brass::Instruction& inst);

    // ---- barriers, shuffles, atomics, multiplies, memory (spirv_isel_intrinsics_mem.cpp)
    static void barrier(SpirvISel& isel);
    static void bar_sync(SpirvISel& isel, const brass::Instruction& inst);
    static void bar_sync_id(SpirvISel& isel, const brass::Instruction& inst);
    enum class Shfl { down, up, bfly, idx };
    static Id shuffle(SpirvISel& isel, const brass::Instruction& inst, Shfl mode, const Value* value, const Value* delta);
    template <Shfl M>
    static void shfl(SpirvISel& isel, const brass::Instruction& inst);
    static void shfl_down_sync_f32(SpirvISel& isel, const brass::Instruction& inst);
    enum class Atom { add, min, max, exch };
    template <Atom A, bool Shared>
    static void atom(SpirvISel& isel, const brass::Instruction& inst);
    template <bool Signed>
    static void mul_wide(SpirvISel& isel, const brass::Instruction& inst);
    static void mul_hi_u32(SpirvISel& isel, const brass::Instruction& inst);
    static void mad_lo_u32(SpirvISel& isel, const brass::Instruction& inst);
    template <TypeKind K>
    static void shared_alloc(SpirvISel& isel, const brass::Instruction& inst);
    template <TypeKind K, bool Indexed>
    static void shared_load(SpirvISel& isel, const brass::Instruction& inst);
    template <TypeKind K, bool Indexed>
    static void shared_store(SpirvISel& isel, const brass::Instruction& inst);
    template <uint32_t Bits, bool Signed>
    static void load_narrow(SpirvISel& isel, const brass::Instruction& inst);
    template <uint32_t Bits>
    static void store_narrow(SpirvISel& isel, const brass::Instruction& inst);
    static Id byte_address(SpirvISel& isel, const Value* ptr, const Value* offset);
};

} // namespace brass::spirv
