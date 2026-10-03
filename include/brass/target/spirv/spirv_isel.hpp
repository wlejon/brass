#pragma once

// SpirvISel: MIR -> SPIR-V compute entry point, the SPIR-V counterpart of
// ptx::PtxISel. One `lower()` call adds one function plus its OpEntryPoint
// to a spirv::Module.
//
//   - control flow: the structurizer (src/target/spirv/spirv_structurize.*)
//     turns the MIR CFG into SPIR-V structured control flow; MIR block
//     parameters become OpPhi;
//   - memory: MIR `ptr` is a 64-bit buffer device address (u64). Loads and
//     stores convert `ptr + offset` with OpConvertUToPtr to a
//     PhysicalStorageBuffer pointer of the accessed type, so MIR's untyped
//     byte-offset arithmetic stays integer arithmetic and one pointer may be
//     read at any element type;
//   - kernel parameters: one push-constant block, in declaration order;
//   - GPU intrinsics: the PTX names (ptx_tid_x, ptx_shfl_down_f32, ...)
//     where SPIR-V has a clean equivalent; the rest are listed by
//     `unsupported_reason()` and throw with that reason.
//
// See docs/spirv_backend_design.md.

#include <brass/mir/function.hpp>
#include <brass/mir/instruction.hpp>
#include <brass/target/spirv/spirv_ir.hpp>
#include <brass/target/spirv_target.hpp>

#include <cstdint>
#include <initializer_list>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace brass::spirv {

struct StructuredCfg;

struct KernelLayout {
    std::vector<target::SpirvParam> params;
    uint32_t push_constant_bytes = 0;
    uint32_t shared_bytes = 0;
};

class SpirvISel {
public:
    // Sets the module's addressing/memory model (PhysicalStorageBuffer64,
    // GLSL450) and the capabilities every kernel needs.
    SpirvISel(Module& m, const target::SpirvOptions& opts);

    // Throws std::runtime_error naming the kernel (and block, where there is
    // one) for non-void kernels, unsupported opcodes/types/intrinsics and
    // control flow the structurizer cannot express.
    KernelLayout lower(const brass::Function& fn);

    using IntrinsicLowering = void (*)(SpirvISel&, const brass::Instruction&);
    static IntrinsicLowering find_intrinsic(std::string_view callee) noexcept;
    static bool is_intrinsic(std::string_view callee) noexcept { return find_intrinsic(callee) != nullptr; }
    static std::vector<std::string_view> intrinsic_names();   // sorted
    // PTX intrinsics with no SPIR-V lowering: the reason, or nullptr.
    static const char* unsupported_reason(std::string_view callee) noexcept;
    static std::vector<std::string_view> unsupported_names(); // sorted

private:
    struct Intrinsics;
    friend struct Intrinsics;

    // A pointer into a Workgroup array (the result of ptx_shared_alloc_*,
    // possibly offset with `add`): Workgroup memory has no address, so it is
    // tracked by provenance instead of as a u64.
    struct SharedRef {
        Id var = 0;
        Type elem;            // f32, i32, f64 or i64
        int64_t const_off = 0; // byte offset, constant part
        Id dyn_off = 0;        // byte offset, dynamic part (u32), 0 for none
    };

    // ---- state -----------------------------------------------------------------
    Module& m_;
    target::SpirvOptions opts_;
    const brass::Function* mir_ = nullptr;
    const StructuredCfg* cfg_ = nullptr;
    size_t fn_index_ = 0;
    size_t cur_ = 0;                                          // current block index
    const brass::Instruction* origin_ = nullptr;
    std::unordered_map<const Value*, std::vector<Id>> ids_;   // value -> one id per 16-byte part (1 for scalars)
    std::unordered_map<const Value*, Id> bools_;              // comparison result -> bool id
    std::unordered_map<const Value*, SharedRef> shared_;
    std::unordered_map<const Value*, uint32_t> value_uses_;   // uses other than as a br_if/select condition
    std::unordered_map<std::string, Id> prologue_cache_;      // per-kernel values computed in the prologue
    std::vector<Id> prologue_args_;                           // kernel parameter values when the entry has phis
    std::vector<size_t> node_block_;                          // cfg node -> block index (kNoNode: none)
    std::vector<std::vector<Id>> node_phis_;                  // cfg node -> its phi ids, in parameter/part order
    uint32_t shared_bytes_ = 0;

    // ---- driver (spirv_isel.cpp) --------------------------------------------------
    void analyze_uses();
    void lower_params(KernelLayout& layout);
    void lower_node(size_t node);
    void fill_phis();
    Id label_of(size_t node) const;
    void lower_instruction(const brass::Instruction& inst);
    void lower_terminator(size_t node, const brass::Instruction& term);
    [[noreturn]] void fail(const std::string& what) const;   // names the kernel and current MIR block
    [[noreturn]] void malformed(const brass::Instruction& inst, const char* what) const;

    // ---- emission helpers -----------------------------------------------------
    Function& fn();
    Block& block();
    Inst& emit(Inst inst);
    Id op(spv::Op o, Id type, std::initializer_list<Id> ids);
    Id ext(uint32_t glsl_inst, Id type, std::initializer_list<Id> ids); // GLSL.std.450
    void use_global(Id var);
    // A value computed once per kernel in the prologue (builtin reads); `make`
    // emits into the prologue block and returns the id.
    template <typename F>
    Id prologue_value(const std::string& key, F make);
    Id builtin_component(spv::BuiltIn b, uint32_t component); // uvec3 builtin, one component
    Id builtin_scalar(spv::BuiltIn b);                        // u32 builtin
    Id workgroup_size(uint32_t component);

    // ---- types and values -------------------------------------------------------
    Id t_u32() { return m_.t_int(32); }
    Id t_u64() { return m_.t_int(64); }
    Id t_f32() { return m_.t_float(32); }
    Id t_bool() { return m_.t_bool(); }
    Id scalar_type(Type t);                 // MIR scalar -> SPIR-V type (ptr is u64)
    Id part_type(Type t);                   // MIR type -> type of one part (a 16-byte vector for vectors)
    uint32_t part_count(Type t) const;      // 1 for scalars, size/16 for vectors
    Id psb_pointer(Id pointee);             // PhysicalStorageBuffer pointer type

    const std::vector<Id>& ids_of(const Value* v, const char* what);
    Id id_of(const Value* v, const char* what) { return ids_of(v, what).front(); }
    void define(const Value* v, Id id) { ids_[v] = {id}; }
    void define(const Value* v, std::vector<Id> ids) { ids_[v] = std::move(ids); }
    Id cond_of(const Value* v);             // bool for a br_if/select condition
    bool has_value_uses(const Value* v) const;
    bool const_int(const Value* v, int64_t* out) const;
    bool const_float(const Value* v, double* out) const;
    Id as_u32(const Value* v, const char* what); // i32 as is, i64 truncated
    Id as_u64(const Value* v, const char* what); // i64/ptr as is, i32 sign-extended

    // ---- alu (spirv_isel_alu.cpp) ---------------------------------------------------
    void lower_binary(const brass::Instruction& inst, spv::Op int_op, spv::Op float_op);
    void lower_int_div_rem(const brass::Instruction& inst, bool is_rem, bool is_unsigned);
    void lower_unary_glsl(const brass::Instruction& inst, uint32_t glsl_inst);
    void lower_comparison(const brass::Instruction& inst, spv::Op int_op, spv::Op float_op);
    void lower_select(const brass::Instruction& inst);
    void lower_convert(const brass::Instruction& inst, spv::Op o, Type dst);
    void lower_add_ptr(const brass::Instruction& inst); // add on a shared pointer

    // ---- memory (spirv_isel_mem.cpp) ------------------------------------------------
    Id address(Id base_u64, int64_t offset);              // base + offset (u64)
    Id indexed_address(const brass::Instruction& inst, const Value* base, const Value* index);
    Id load_global(Id addr, Id type, uint32_t align);
    void store_global(Id addr, Id value, Id type, uint32_t align);
    void lower_load(const brass::Instruction& inst);
    void lower_store(const brass::Instruction& inst);
    void lower_vload(const brass::Instruction& inst);
    void lower_vstore(const brass::Instruction& inst);
    void lower_load_indexed(const brass::Instruction& inst);
    void lower_store_indexed(const brass::Instruction& inst);
    // Shared (Workgroup) memory through SharedRef provenance.
    const SharedRef& shared_ref(const Value* v, const char* what) const;
    Id shared_element_pointer(const SharedRef& ref, const Value* extra, bool extra_is_index, Type access);
    Id shared_load(const SharedRef& ref, const Value* extra, bool extra_is_index, Type access);
    void shared_store(const SharedRef& ref, const Value* extra, bool extra_is_index, const Value* value);

    // ---- vectors (spirv_isel_vec.cpp) ------------------------------------------------
    void lower_vector_binary(const brass::Instruction& inst, spv::Op int_op, spv::Op float_op);
    void lower_vector_minmax(const brass::Instruction& inst, bool is_max);
    void lower_vector_fma(const brass::Instruction& inst);
    void lower_vector_unary(const brass::Instruction& inst);    // vneg, vsqrt
    void lower_vector_bitwise(const brass::Instruction& inst, spv::Op o); // vand/vor/vxor/vnot (OpNot)
    void lower_vbroadcast(const brass::Instruction& inst);
    void lower_vzero(const brass::Instruction& inst);
    void lower_vextract_lane(const brass::Instruction& inst);
    void lower_vinsert_lane(const brass::Instruction& inst);
    void lower_vshuffle(const brass::Instruction& inst);

    // ---- calls (spirv_isel_intrinsics*.cpp) ----------------------------------------------
    void lower_call(const brass::Instruction& inst);
};

template <typename F>
Id SpirvISel::prologue_value(const std::string& key, F make) {
    auto it = prologue_cache_.find(key);
    if (it != prologue_cache_.end()) return it->second;
    size_t saved = cur_;
    cur_ = node_block_[0]; // the prologue is the first node
    Id id = make();
    cur_ = saved;
    prologue_cache_[key] = id;
    return id;
}

} // namespace brass::spirv
