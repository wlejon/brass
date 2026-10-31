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
    template <uint32_t Glsl>
    static void glsl_ternary(SpirvISel& isel, const brass::Instruction& inst); // type from argument 0
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
    enum class Atom { add, min, umin, max, umax, exch, and_, or_, xor_, cas };
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
    enum class Group { add, min, max };
    template <Group G>
    static void subgroup_reduce(SpirvISel& isel, const brass::Instruction& inst);
    static void subgroup_broadcast_first(SpirvISel& isel, const brass::Instruction& inst);
    template <bool All>
    static void subgroup_vote(SpirvISel& isel, const brass::Instruction& inst);
    static void subgroup_size(SpirvISel& isel, const brass::Instruction& inst);
    static void subgroup_local_id(SpirvISel& isel, const brass::Instruction& inst);

    // ---- descriptor images, samplers, graphics stages (spirv_isel_graphics.cpp)
    static void parse_set_binding(SpirvISel& isel, const brass::Instruction& inst, uint32_t& out_set, uint32_t& out_binding);
    // A texture or image call's image (image_call reads its leading operands).
    struct ImageCall {
        uint32_t set = 0, binding = 0;
        target::SpirvImageDim dim = target::SpirvImageDim::D2;
        bool arrayed = false, depth = false;
        target::SpirvImageScalar scalar = target::SpirvImageScalar::F32;
        uint32_t format = 0;
        Id texel_scalar = 0, texel_type = 0, image_type = 0;
        Id image = 0;     // the OpTypeImage value
        Id sampled = 0;   // the OpTypeSampledImage value (none for a storage image)
        uint32_t coords() const;
    };
    static ImageCall image_call(SpirvISel& isel, const brass::Instruction& inst, bool storage, size_t own_operands);
    static Id image_coord(SpirvISel& isel, const brass::Instruction& inst, size_t first, uint32_t n, bool integer);
    static Id image_const_offset(SpirvISel& isel, const brass::Instruction& inst, size_t first, uint32_t n);
    static constexpr int kOffsetBit = 4;
    static constexpr int kDrefBit = 8;
    template <int Op>
    static void tex_sample(SpirvISel& isel, const brass::Instruction& inst);
    static void tex_fetch(SpirvISel& isel, const brass::Instruction& inst);
    static void tex_gather(SpirvISel& isel, const brass::Instruction& inst);
    template <bool Storage>
    static void tex_size(SpirvISel& isel, const brass::Instruction& inst);
    static void tex_levels(SpirvISel& isel, const brass::Instruction& inst);
    static void img_load(SpirvISel& isel, const brass::Instruction& inst);
    static void img_store(SpirvISel& isel, const brass::Instruction& inst);

    template <spv::BuiltIn B>
    static void builtin_scalar_in(SpirvISel& isel, const brass::Instruction& inst) {
        isel.define(inst.result(), isel.builtin_scalar(B));
    }
    template <uint32_t C>
    static void frag_coord_comp(SpirvISel& isel, const brass::Instruction& inst) {
        isel.define(inst.result(), isel.builtin_fvec4_component(spv::BuiltInFragCoord, C));
    }
    static void frag_coord_vec4(SpirvISel& isel, const brass::Instruction& inst) {
        isel.define(inst.result(), isel.builtin_fvec4(spv::BuiltInFragCoord));
    }
    struct StageVar {
        uint32_t location = 0, lanes = 1, interp = 0;
        bool integer = false;
    };
    static StageVar stage_var(SpirvISel& isel, const brass::Instruction& inst);
    static void stage_in(SpirvISel& isel, const brass::Instruction& inst);
    static void stage_out(SpirvISel& isel, const brass::Instruction& inst);
    static void set_frag_depth(SpirvISel& isel, const brass::Instruction& inst);
    static Id buffer_var(SpirvISel& isel, const brass::Instruction& inst, bool store, bool& storage, bool& vec4_rows);
    static Id buffer_word(SpirvISel& isel, const brass::Instruction& inst, bool store);   // a pointer to the word
    static void buffer_length(SpirvISel& isel, const brass::Instruction& inst);
    static void buffer_load(SpirvISel& isel, const brass::Instruction& inst);
    static void buffer_store(SpirvISel& isel, const brass::Instruction& inst);
    template <Atom A>
    static void buffer_atomic(SpirvISel& isel, const brass::Instruction& inst);
    static void set_vertex_position(SpirvISel& isel, const brass::Instruction& inst);
    static void require_fragment(SpirvISel& isel, const brass::Instruction& inst);
    template <spv::Op O>
    static void derivative(SpirvISel& isel, const brass::Instruction& inst);  // f32 -> f32
    static void front_facing(SpirvISel& isel, const brass::Instruction& inst); // -> i32 0/1
    static void demote_to_helper(SpirvISel& isel, const brass::Instruction& inst);
};

} // namespace brass::spirv
