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

class BasicBlock;

enum class BytecodeOp : uint8_t {
    // Nop / Trap
    nop = 0,
    unreachable,

    // Constants
    iconst32,   // AI32 a <- zext(imm32): the i32 register form
    load_const, // AI32 a <- constants[uimm32]
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
    trunc16,  // AB a <- zext(b & 0xffff)
    sext8,    // AB a <- zext32(sext(b & 0xff)): an i8 read as a signed i32
    sext16,   // AB a <- zext32(sext(b & 0xffff))
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
    pinned_tls_read,
    pinned_tls_write,
    read_sp,

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
    vzero,

    // Additions of the 64-bit encoding (see the format table below).
    vmov,        // ABC   a <- b, scalar and 256-bit vector state
    index_addr,  // ABCD  a <- b + c * d8
    br_eq_i32,   // ABI24 if a == b goto pc + imm24 (and the rest: fused compare-branch)
    br_ne_i32,
    br_slt_i32,
    br_sle_i32,
    br_ult_i32,
    br_ule_i32,
    br_eq_i64,
    br_ne_i64,
    br_slt_i64,
    br_sle_i64,
    br_ult_i64,
    br_ule_i64,
    add_imm_i32, // ABI24 a <- b + imm24 (32-bit wrap)
    add_imm_i64, // ABI24 a <- b + imm24

    op_count_
};

std::string_view bytecode_op_name(BytecodeOp op) noexcept;
bool is_bytecode_jump(BytecodeOp op) noexcept;
bool is_bytecode_call(BytecodeOp op) noexcept;
bool is_bytecode_terminator(BytecodeOp op) noexcept;
bool is_bytecode_cond_branch(BytecodeOp op) noexcept;

// Register numbers. The instruction word carries 16-bit register fields, so
// a function can use up to kMaxBytecodeRegisters registers after
// allocation (the compiler reports an error past that). kNoReg marks an
// absent register ("no result", "no input").
using BcReg = uint16_t;
inline constexpr BcReg kNoReg = 0xFFFF;
inline constexpr uint32_t kMaxBytecodeRegisters = 0xFFFF;

// 64-bit instruction word. Every format keeps the opcode in bits 0-7 and
// the first register in bits 8-23:
//   ABC   [op:8][a:16][b:16 @24][c:16 @40][d:8 @56]
//   AI32  [op:8][a:16][imm32 @32]            imm32 signed or unsigned per op
//   ABI24 [op:8][a:16][b:16 @24][imm24 @40]  imm24 signed
// Jump offsets are relative to the jumping instruction: AI32 for jump /
// jump_if / jump_if_not, ABI24 for the fused compare-branches. Memory
// offsets are ABI24 (a = value, b = base).
using BytecodeWord = uint64_t;

inline constexpr int32_t kImm24Min = -(1 << 23);
inline constexpr int32_t kImm24Max = (1 << 23) - 1;
inline constexpr bool fits_imm24(int64_t v) noexcept { return v >= kImm24Min && v <= kImm24Max; }

inline constexpr BytecodeWord encode_abc(BytecodeOp op, uint32_t a, uint32_t b, uint32_t c, uint32_t d = 0) noexcept {
    return static_cast<uint64_t>(op) |
           (static_cast<uint64_t>(a & 0xFFFF) << 8) |
           (static_cast<uint64_t>(b & 0xFFFF) << 24) |
           (static_cast<uint64_t>(c & 0xFFFF) << 40) |
           (static_cast<uint64_t>(d & 0xFF) << 56);
}

inline constexpr BytecodeWord encode_ai(BytecodeOp op, uint32_t a, int32_t imm) noexcept {
    return static_cast<uint64_t>(op) |
           (static_cast<uint64_t>(a & 0xFFFF) << 8) |
           (static_cast<uint64_t>(static_cast<uint32_t>(imm)) << 32);
}

inline constexpr BytecodeWord encode_abi(BytecodeOp op, uint32_t a, uint32_t b, int32_t imm24) noexcept {
    return static_cast<uint64_t>(op) |
           (static_cast<uint64_t>(a & 0xFFFF) << 8) |
           (static_cast<uint64_t>(b & 0xFFFF) << 24) |
           (static_cast<uint64_t>(static_cast<uint32_t>(imm24) & 0xFFFFFFu) << 40);
}

inline constexpr BytecodeOp decode_op(BytecodeWord inst) noexcept { return static_cast<BytecodeOp>(inst & 0xFF); }
inline constexpr uint32_t decode_a(BytecodeWord inst) noexcept { return static_cast<uint32_t>((inst >> 8) & 0xFFFF); }
inline constexpr uint32_t decode_b(BytecodeWord inst) noexcept { return static_cast<uint32_t>((inst >> 24) & 0xFFFF); }
inline constexpr uint32_t decode_c(BytecodeWord inst) noexcept { return static_cast<uint32_t>((inst >> 40) & 0xFFFF); }
inline constexpr uint32_t decode_d(BytecodeWord inst) noexcept { return static_cast<uint32_t>(inst >> 56); }
inline constexpr int32_t decode_imm32(BytecodeWord inst) noexcept { return static_cast<int32_t>(static_cast<uint32_t>(inst >> 32)); }
inline constexpr uint32_t decode_uimm32(BytecodeWord inst) noexcept { return static_cast<uint32_t>(inst >> 32); }
inline constexpr int32_t decode_imm24(BytecodeWord inst) noexcept { return static_cast<int32_t>(static_cast<int64_t>(inst) >> 40); }

// Operand-role names for the ABC fields.
inline constexpr uint32_t decode_dst(BytecodeWord inst) noexcept { return decode_a(inst); }
inline constexpr uint32_t decode_src1(BytecodeWord inst) noexcept { return decode_b(inst); }
inline constexpr uint32_t decode_src2(BytecodeWord inst) noexcept { return decode_c(inst); }

// Target pc of a jump or fused compare-branch at `pc`.
int64_t bytecode_branch_target(BytecodeWord inst, size_t pc) noexcept;

// Code words an instruction occupies: alloca_ (size), coro_suspend (resume
// id) and vshuffle (mask) carry a second, raw data word.
inline constexpr size_t bytecode_inst_words(BytecodeOp op) noexcept {
    return (op == BytecodeOp::alloca_ || op == BytecodeOp::coro_suspend || op == BytecodeOp::vshuffle) ? 2 : 1;
}

struct ExceptionEntry {
    uint32_t start_pc = 0;
    uint32_t end_pc = 0;
    uint32_t handler_pc = 0;
};

struct ResumePointEntry {
    uint32_t resume_id = 0;
    uint32_t target_pc = 0;
    std::vector<BcReg> param_regs;
};

struct OsrEntry {
    uint32_t pc = 0;
    uint32_t loop_header_pc = 0;
    std::vector<BcReg> live_regs;
};

struct LineInfoEntry {
    uint32_t pc = 0;
    DebugLoc loc;
};

struct CallSiteInfo {
    std::string callee;
    std::string extra_symbol;
    uint32_t site_id = 0;
    BcReg dst_reg = kNoReg;
    BcReg callee_reg = kNoReg;
    bool patchable = false; // patchable_call: the callee is looked up in patch_call()'s table
    std::vector<BcReg> arg_regs;
};

struct SwitchTable {
    int32_t default_offset = 0;                     // absolute target pc
    std::vector<std::pair<int64_t, int32_t>> cases; // (value, absolute target pc), sorted by value
    bool is_i32 = false;                            // compare the low 32 bits, sign-extended
};

struct GuardInfo {
    uint32_t resume_id = 0;
    std::string exit_stub;
    std::vector<BcReg> state_regs;
};

// A patchable constant: the value patched under `symbol`, else the MIR default.
struct PatchConstSite {
    std::string symbol;
    int64_t default_value = 0;
};

class BytecodeModule;

class BytecodeFunction {
public:
    std::string name;
    std::vector<BytecodeWord> code;
    std::vector<uint64_t> constants;
    uint32_t num_registers = 0;
    uint32_t num_params = 0;
    // SSA values before register allocation, for diagnostics.
    uint32_t num_ssa_values = 0;
    Type return_type = Type::void_type();
    std::vector<Type> param_types;
    // Every value sharing a register has the same type, so this is exact.
    std::vector<Type> register_types;
    std::unordered_map<uint32_t, BcReg> ssa_to_reg;
    std::unordered_map<uint32_t, const BasicBlock*> pc_block_map;
    std::vector<ExceptionEntry> exception_table;
    std::vector<ResumePointEntry> resume_points;
    std::vector<OsrEntry> osr_entries;
    std::vector<LineInfoEntry> line_info_table;

    // Helper metadata
    std::vector<CallSiteInfo> call_sites;
    std::vector<SwitchTable> switch_tables;
    std::vector<GuardInfo> guards;
    std::vector<PatchConstSite> patch_consts;
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
    uint32_t add_patch_const(PatchConstSite site);

    void emit(BytecodeWord inst) { code.push_back(inst); }

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
