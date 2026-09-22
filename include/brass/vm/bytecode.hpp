#pragma once

#include <brass/mir/types.hpp>
#include <brass/debug/source_loc.hpp>
#include <cstdint>
#include <cstring>
#include <string>
#include <string_view>
#include <vector>
#include <unordered_map>
#include <memory>
#include <iosfwd>

namespace brass {

enum class BytecodeOp : uint8_t {
    // Nop / Trap
    nop = 0,
    unreachable,

    // Constants
    iconst32,
    iconst64,
    fconst32,
    fconst64,
    load_const,
    patchable_const32,
    patchable_const64,

    // Register moves
    mov,
    mov_imm,

    // Conversions & Bitcasts
    sext64,
    zext64,
    trunc32,
    trunc8,
    fptosi32,
    fptosi64,
    fptosi32_f32,
    fptosi64_f32,
    sitofp_f64,
    sitofp_f32,
    sitofp_f64_i64,
    sitofp_f32_i64,
    fptrunc_f32,
    fpext_f64,
    bitcast_i64_f64,
    bitcast_f64_i64,

    // Integer Arithmetic
    add_i32,
    add_i64,
    sub_i32,
    sub_i64,
    mul_i32,
    mul_i64,
    sdiv_i32,
    sdiv_i64,
    udiv_i32,
    udiv_i64,
    smod_i32,
    smod_i64,
    umod_i32,
    umod_i64,
    neg_i32,
    neg_i64,

    // Floating-point Arithmetic
    add_f32,
    add_f64,
    sub_f32,
    sub_f64,
    mul_f32,
    mul_f64,
    fdiv_f32,
    fdiv_f64,
    neg_f32,
    neg_f64,
    fma_f32,
    fma_f64,
    sqrt_f32,
    sqrt_f64,
    fabs_f32,
    fabs_f64,
    floor_f32,
    floor_f64,
    ceil_f32,
    ceil_f64,
    round_f32,
    round_f64,
    fmin_f32,
    fmin_f64,
    fmax_f32,
    fmax_f64,

    // Overflow-checked Arithmetic
    sadd_overflow_i32,
    sadd_overflow_i64,
    ssub_overflow_i32,
    ssub_overflow_i64,
    smul_overflow_i32,
    smul_overflow_i64,
    uadd_overflow_i32,
    uadd_overflow_i64,
    usub_overflow_i32,
    usub_overflow_i64,
    umul_overflow_i32,
    umul_overflow_i64,

    // Bitwise & Bit-manipulation
    and_i32,
    and_i64,
    or_i32,
    or_i64,
    xor_i32,
    xor_i64,
    shl_i32,
    shl_i64,
    lshr_i32,
    lshr_i64,
    ashr_i32,
    ashr_i64,
    not_i32,
    not_i64,
    clz_i32,
    clz_i64,
    ctz_i32,
    ctz_i64,
    popcnt_i32,
    popcnt_i64,

    // Comparisons
    eq_i32,
    eq_i64,
    eq_f32,
    eq_f64,
    ne_i32,
    ne_i64,
    ne_f32,
    ne_f64,
    slt_i32,
    slt_i64,
    ult_i32,
    ult_i64,
    lt_f32,
    lt_f64,
    sle_i32,
    sle_i64,
    ule_i32,
    ule_i64,
    le_f32,
    le_f64,
    sgt_i32,
    sgt_i64,
    ugt_i32,
    ugt_i64,
    gt_f32,
    gt_f64,
    sge_i32,
    sge_i64,
    uge_i32,
    uge_i64,
    ge_f32,
    ge_f64,
    select,

    // Memory
    load8,
    load16,
    load32,
    load64,
    store8,
    store16,
    store32,
    store64,
    alloca_,
    load_indexed,
    store_indexed,

    // Control Flow
    jump,
    jump_if,
    jump_if_not,
    ret,
    ret_void,
    switch_,

    // Calls
    call,
    call_indirect,
    patchable_call,
    func_addr,

    // Runtime & GC
    safepoint,
    write_barrier,
    guard,
    resume_point,
    osr_entry,

    // Exceptions & Coroutines
    throw_,
    invoke,
    landing_pad,
    resume,
    coro_create,
    coro_suspend,
    coro_resume,
    coro_destroy,

    // Vector operations
    vadd,
    vsub,
    vmul,
    vdiv,
    vfma,
    vneg,
    vmin,
    vmax,
    vsqrt,
    vand,
    vor,
    vxor,
    vnot,
    vload,
    vstore,
    vbroadcast,
    vextract_lane,
    vinsert_lane,
    vshuffle,
    vzero
};

std::string_view bytecode_op_name(BytecodeOp op) noexcept;
bool is_bytecode_jump(BytecodeOp op) noexcept;
bool is_bytecode_call(BytecodeOp op) noexcept;
bool is_bytecode_terminator(BytecodeOp op) noexcept;

// Compact 32-bit instruction encoding:
// Format ABC: [op: 8, dst: 8, src1: 8, src2: 8]
// Format AD:  [op: 8, dst: 8, imm16: 16]

inline constexpr uint32_t encode_abc(BytecodeOp op, uint8_t dst, uint8_t src1, uint8_t src2) noexcept {
    return static_cast<uint32_t>(op) |
           (static_cast<uint32_t>(dst) << 8) |
           (static_cast<uint32_t>(src1) << 16) |
           (static_cast<uint32_t>(src2) << 24);
}

inline constexpr uint32_t encode_ad(BytecodeOp op, uint8_t dst, uint16_t imm16) noexcept {
    return static_cast<uint32_t>(op) |
           (static_cast<uint32_t>(dst) << 8) |
           (static_cast<uint32_t>(imm16) << 16);
}

inline constexpr uint32_t encode_ad(BytecodeOp op, uint8_t dst, int16_t simm16) noexcept {
    return encode_ad(op, dst, static_cast<uint16_t>(simm16));
}

inline constexpr BytecodeOp decode_op(uint32_t inst) noexcept {
    return static_cast<BytecodeOp>(inst & 0xFF);
}

inline constexpr uint8_t decode_dst(uint32_t inst) noexcept {
    return static_cast<uint8_t>((inst >> 8) & 0xFF);
}

inline constexpr uint8_t decode_src1(uint32_t inst) noexcept {
    return static_cast<uint8_t>((inst >> 16) & 0xFF);
}

inline constexpr uint8_t decode_src2(uint32_t inst) noexcept {
    return static_cast<uint8_t>((inst >> 24) & 0xFF);
}

inline constexpr uint16_t decode_u16(uint32_t inst) noexcept {
    return static_cast<uint16_t>((inst >> 16) & 0xFFFF);
}

inline constexpr int16_t decode_s16(uint32_t inst) noexcept {
    return static_cast<int16_t>((inst >> 16) & 0xFFFF);
}

struct BytecodeInstruction {
    uint32_t raw = 0;

    constexpr BytecodeInstruction() noexcept = default;
    constexpr explicit BytecodeInstruction(uint32_t r) noexcept : raw(r) {}
    constexpr BytecodeInstruction(BytecodeOp op, uint8_t dst, uint8_t src1, uint8_t src2) noexcept
        : raw(encode_abc(op, dst, src1, src2)) {}
    constexpr BytecodeInstruction(BytecodeOp op, uint8_t dst, uint16_t imm16) noexcept
        : raw(encode_ad(op, dst, imm16)) {}
    constexpr BytecodeInstruction(BytecodeOp op, uint8_t dst, int16_t imm16) noexcept
        : raw(encode_ad(op, dst, imm16)) {}

    constexpr BytecodeOp op() const noexcept { return decode_op(raw); }
    constexpr uint8_t dst() const noexcept { return decode_dst(raw); }
    constexpr uint8_t src1() const noexcept { return decode_src1(raw); }
    constexpr uint8_t src2() const noexcept { return decode_src2(raw); }
    constexpr uint16_t u16() const noexcept { return decode_u16(raw); }
    constexpr int16_t s16() const noexcept { return decode_s16(raw); }
};

struct ExceptionEntry {
    uint32_t start_pc = 0;
    uint32_t end_pc = 0;
    uint32_t handler_pc = 0;
};

struct ResumePointEntry {
    uint32_t resume_id = 0;
    uint32_t target_pc = 0;
    std::vector<uint8_t> param_regs;
};

struct OsrEntry {
    uint32_t pc = 0;
    uint32_t loop_header_pc = 0;
    std::vector<uint8_t> live_regs;
};

struct LineInfoEntry {
    uint32_t pc = 0;
    DebugLoc loc;
};

struct CallSiteInfo {
    std::string callee;
    std::string extra_symbol;
    uint32_t site_id = 0;
    uint8_t dst_reg = 0;
    uint8_t callee_reg = 0;
    std::vector<uint8_t> arg_regs;
};

struct SwitchTable {
    int32_t default_offset = 0;
    std::vector<std::pair<int64_t, int32_t>> cases; // (value, relative PC offset)
};

struct GuardInfo {
    uint32_t resume_id = 0;
    std::string exit_stub;
    std::vector<uint8_t> state_regs;
};

class BytecodeModule;

class BytecodeFunction {
public:
    std::string name;
    std::vector<uint32_t> code;
    std::vector<uint64_t> constants;
    uint32_t num_registers = 0;
    uint32_t num_params = 0;
    Type return_type = Type::void_type();
    std::vector<Type> param_types;
    std::vector<ExceptionEntry> exception_table;
    std::vector<ResumePointEntry> resume_points;
    std::vector<OsrEntry> osr_entries;
    std::vector<LineInfoEntry> line_info_table;

    // Helper metadata
    std::vector<CallSiteInfo> call_sites;
    std::vector<SwitchTable> switch_tables;
    std::vector<GuardInfo> guards;
    std::vector<std::string> string_pool;
    BytecodeModule* parent = nullptr;

    BytecodeFunction() = default;
    BytecodeFunction(std::string_view n, Type ret_ty, std::vector<Type> p_types)
        : name(n),
          num_params(static_cast<uint32_t>(p_types.size())),
          return_type(ret_ty),
          param_types(std::move(p_types)) {}

    uint32_t add_constant(uint64_t val);
    uint32_t add_constant_f64(double val);
    uint32_t add_constant_f32(float val);
    uint32_t add_string_constant(std::string_view str);

    uint32_t add_call_site(CallSiteInfo info);
    uint32_t add_switch_table(SwitchTable table);
    uint32_t add_guard(GuardInfo guard);

    void emit(uint32_t inst) { code.push_back(inst); }
    void emit(BytecodeOp op, uint8_t dst, uint8_t src1, uint8_t src2) {
        code.push_back(encode_abc(op, dst, src1, src2));
    }
    void emit(BytecodeOp op, uint8_t dst, uint16_t imm16) {
        code.push_back(encode_ad(op, dst, imm16));
    }
    void emit(BytecodeOp op, uint8_t dst, int16_t imm16) {
        code.push_back(encode_ad(op, dst, imm16));
    }

    size_t current_pc() const noexcept { return code.size(); }
    void set_line_info(uint32_t pc, DebugLoc loc);
    DebugLoc get_line_info(uint32_t pc) const;
};

class BytecodeModule {
public:
    explicit BytecodeModule(std::string_view name = "") : name_(name) {}

    std::string_view name() const noexcept { return name_; }
    void set_name(std::string_view name) { name_ = name; }

    void add_function(std::unique_ptr<BytecodeFunction> fn);
    BytecodeFunction* get_function(std::string_view name) const noexcept;
    BytecodeFunction* get_function(size_t index) const noexcept;
    size_t function_count() const noexcept { return functions_.size(); }
    const std::vector<std::unique_ptr<BytecodeFunction>>& functions() const noexcept { return functions_; }

    void add_symbol(std::string_view name, uint32_t func_index);
    uint32_t find_symbol(std::string_view name) const;
    bool has_symbol(std::string_view name) const;
    const std::unordered_map<std::string, uint32_t>& symbol_table() const noexcept { return symbol_table_; }

    std::vector<uint64_t>& module_constants() noexcept { return module_constants_; }
    const std::vector<uint64_t>& module_constants() const noexcept { return module_constants_; }

private:
    std::string name_;
    std::vector<std::unique_ptr<BytecodeFunction>> functions_;
    std::unordered_map<std::string, uint32_t> symbol_table_;
    std::vector<uint64_t> module_constants_;
};

// Bytecode Disassembler
std::string disassemble(const BytecodeFunction& fn);
std::string disassemble(const BytecodeModule& mod);
void dump(const BytecodeFunction& fn, std::ostream& os);
void dump(const BytecodeModule& mod, std::ostream& os);

} // namespace brass
