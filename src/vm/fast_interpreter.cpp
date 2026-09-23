// The FastInterpreter dispatch loop: direct threading (computed goto) on
// GCC/Clang, a switch elsewhere. Instructions are 64-bit words (see
// bytecode.hpp); each handler decodes its fields straight from the word.
// Calls, exceptions, coroutines and vectors leave the loop through the
// helpers in the other fast_interpreter_*.cpp files.

#include "fast_interpreter_impl.hpp"
#include <brass/runtime/osr_coordinator.hpp>
#include <brass/runtime/code_installer.hpp>

#if defined(__GNUC__) || defined(__clang__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wpedantic"
#endif

namespace brass {

void FastInterpreter::throw_instruction_limit() const {
    throw InterpreterException("Maximum instruction execution count exceeded (" + std::to_string(max_instructions_) + ")");
}

RuntimeValue FastInterpreter::execute_frame(FastFrame& frame) {
    const BytecodeFunction& fn = *frame.bfn;
    const BytecodeWord* const code_base = fn.code.data();
    if (code_base == nullptr || fn.code.empty()) {
        return RuntimeValue::from_void();
    }
    if (!frame.info) frame.info = &fn_info(fn, frame.mir_fn);
    if (!frame.mir_fn) frame.mir_fn = frame.info->mir_fn;

    const BytecodeWord* pc = code_base + frame.pc;
    uint64_t* const registers = frame.registers;
    BytecodeWord inst = *pc;

    // Loop bookkeeping happens at backward branches only: the instruction
    // budget is charged by the loop's length, and the function's backedge
    // counter (or the OSR coordinator, when enabled) is told.
    const uint64_t insn_limit = max_instructions_ > 0 ? max_instructions_ : ~uint64_t{0};
    const bool osr_on = frame.mir_fn != nullptr && dispatch_table().osr().is_enabled();
    runtime::TieringFeedback* const feedback = &frame.info->tiering(dispatch_table_);

#define RA registers[decode_a(inst)]
#define RB registers[decode_b(inst)]
#define RC registers[decode_c(inst)]

#define BACKEDGE(off)                                                                         \
    do {                                                                                      \
        total_instructions_executed_ += static_cast<uint64_t>(-static_cast<int64_t>(off)) + 1; \
        if (BRASS_UNLIKELY(total_instructions_executed_ > insn_limit)) throw_instruction_limit(); \
        if (BRASS_UNLIKELY(osr_on)) {                                                         \
            RuntimeValue osr_res_;                                                            \
            const uint32_t target_pc_ = static_cast<uint32_t>((pc - code_base) + (off));      \
            if (handle_osr_backedge(frame, target_pc_, osr_res_)) return osr_res_;            \
        } else {                                                                              \
            feedback->count_backedge_fast();                                                  \
        }                                                                                     \
    } while (0)

#if defined(__GNUC__) || defined(__clang__)
#define BRASS_DIRECT_THREADED 1
    static void* dispatch_table[256];
    static bool table_inited = false;
    if (BRASS_UNLIKELY(!table_inited)) {
        for (size_t i = 0; i < 256; ++i) dispatch_table[i] = &&do_invalid_op;
#define TENTRY(op) dispatch_table[static_cast<size_t>(BytecodeOp::op)] = &&do_##op
        TENTRY(nop); TENTRY(unreachable); TENTRY(iconst32); TENTRY(load_const);
        TENTRY(patchable_const32); TENTRY(patchable_const64);
        TENTRY(mov); TENTRY(mov_imm); TENTRY(sext64); TENTRY(zext64);
        TENTRY(trunc32); TENTRY(trunc8); TENTRY(fptosi32); TENTRY(fptosi64);
        TENTRY(fptosi32_f32); TENTRY(fptosi64_f32); TENTRY(sitofp_f64);
        TENTRY(sitofp_f32); TENTRY(sitofp_f64_i64); TENTRY(sitofp_f32_i64);
        TENTRY(fptrunc_f32); TENTRY(fpext_f64); TENTRY(bitcast_i64_f64); TENTRY(bitcast_f64_i64);
        TENTRY(add_i32); TENTRY(add_i64); TENTRY(sub_i32); TENTRY(sub_i64);
        TENTRY(mul_i32); TENTRY(mul_i64); TENTRY(sdiv_i32); TENTRY(sdiv_i64);
        TENTRY(udiv_i32); TENTRY(udiv_i64); TENTRY(smod_i32); TENTRY(smod_i64);
        TENTRY(umod_i32); TENTRY(umod_i64); TENTRY(neg_i32); TENTRY(neg_i64);
        TENTRY(add_f32); TENTRY(add_f64); TENTRY(sub_f32); TENTRY(sub_f64);
        TENTRY(mul_f32); TENTRY(mul_f64); TENTRY(fdiv_f32); TENTRY(fdiv_f64);
        TENTRY(neg_f32); TENTRY(neg_f64); TENTRY(fma_f32); TENTRY(fma_f64);
        TENTRY(sqrt_f32); TENTRY(sqrt_f64); TENTRY(fabs_f32); TENTRY(fabs_f64);
        TENTRY(floor_f32); TENTRY(floor_f64); TENTRY(ceil_f32); TENTRY(ceil_f64);
        TENTRY(round_f32); TENTRY(round_f64); TENTRY(fmin_f32); TENTRY(fmin_f64);
        TENTRY(fmax_f32); TENTRY(fmax_f64);
        TENTRY(sadd_overflow_i32); TENTRY(sadd_overflow_i64);
        TENTRY(ssub_overflow_i32); TENTRY(ssub_overflow_i64);
        TENTRY(smul_overflow_i32); TENTRY(smul_overflow_i64);
        TENTRY(uadd_overflow_i32); TENTRY(uadd_overflow_i64);
        TENTRY(usub_overflow_i32); TENTRY(usub_overflow_i64);
        TENTRY(umul_overflow_i32); TENTRY(umul_overflow_i64);
        TENTRY(and_i32); TENTRY(and_i64); TENTRY(or_i32); TENTRY(or_i64);
        TENTRY(xor_i32); TENTRY(xor_i64); TENTRY(shl_i32); TENTRY(shl_i64);
        TENTRY(lshr_i32); TENTRY(lshr_i64); TENTRY(ashr_i32); TENTRY(ashr_i64);
        TENTRY(not_i32); TENTRY(not_i64); TENTRY(clz_i32); TENTRY(clz_i64);
        TENTRY(ctz_i32); TENTRY(ctz_i64); TENTRY(popcnt_i32); TENTRY(popcnt_i64);
        TENTRY(eq_i32); TENTRY(eq_i64); TENTRY(eq_f32); TENTRY(eq_f64);
        TENTRY(ne_i32); TENTRY(ne_i64); TENTRY(ne_f32); TENTRY(ne_f64);
        TENTRY(slt_i32); TENTRY(slt_i64); TENTRY(ult_i32); TENTRY(ult_i64);
        TENTRY(lt_f32); TENTRY(lt_f64); TENTRY(sle_i32); TENTRY(sle_i64);
        TENTRY(ule_i32); TENTRY(ule_i64); TENTRY(le_f32); TENTRY(le_f64);
        TENTRY(sgt_i32); TENTRY(sgt_i64); TENTRY(ugt_i32); TENTRY(ugt_i64);
        TENTRY(gt_f32); TENTRY(gt_f64); TENTRY(sge_i32); TENTRY(sge_i64);
        TENTRY(uge_i32); TENTRY(uge_i64); TENTRY(ge_f32); TENTRY(ge_f64);
        TENTRY(select); TENTRY(load8); TENTRY(load16); TENTRY(load32); TENTRY(load64);
        TENTRY(store8); TENTRY(store16); TENTRY(store32); TENTRY(store64); TENTRY(alloca_);
        TENTRY(jump); TENTRY(jump_if); TENTRY(jump_if_not);
        TENTRY(ret); TENTRY(ret_void); TENTRY(switch_);
        TENTRY(call); TENTRY(call_indirect); TENTRY(patchable_call); TENTRY(func_addr);
        TENTRY(safepoint); TENTRY(write_barrier); TENTRY(guard); TENTRY(resume_point); TENTRY(osr_entry);
        TENTRY(pinned_tls_read); TENTRY(pinned_tls_write); TENTRY(read_sp);
        TENTRY(throw_); TENTRY(invoke); TENTRY(landing_pad); TENTRY(resume);
        TENTRY(coro_create); TENTRY(coro_suspend); TENTRY(coro_resume); TENTRY(coro_destroy);
        TENTRY(vadd); TENTRY(vsub); TENTRY(vmul); TENTRY(vdiv); TENTRY(vfma); TENTRY(vneg);
        TENTRY(vmin); TENTRY(vmax); TENTRY(vsqrt); TENTRY(vand); TENTRY(vor); TENTRY(vxor);
        TENTRY(vnot); TENTRY(vload); TENTRY(vstore); TENTRY(vbroadcast);
        TENTRY(vextract_lane); TENTRY(vinsert_lane); TENTRY(vshuffle); TENTRY(vzero);
        TENTRY(vmov); TENTRY(index_addr);
        TENTRY(br_eq_i32); TENTRY(br_ne_i32); TENTRY(br_slt_i32); TENTRY(br_sle_i32);
        TENTRY(br_ult_i32); TENTRY(br_ule_i32); TENTRY(br_eq_i64); TENTRY(br_ne_i64);
        TENTRY(br_slt_i64); TENTRY(br_sle_i64); TENTRY(br_ult_i64); TENTRY(br_ule_i64);
        TENTRY(add_imm_i32); TENTRY(add_imm_i64);
#undef TENTRY
        table_inited = true;
    }

#define OP_CASE(name) do_##name:
#define DISPATCH() do { inst = *pc; goto *dispatch_table[inst & 0xFF]; } while (0)
    DISPATCH();
#else
#define BRASS_DIRECT_THREADED 0
#define OP_CASE(name) case BytecodeOp::name:
#define DISPATCH() goto loop_start

loop_start:
    inst = *pc;
    switch (decode_op(inst))
#endif
    {
#define NEXT() do { ++pc; DISPATCH(); } while (0)
#define JUMP_BY(off) do { const int32_t o_ = (off); if (o_ <= 0) BACKEDGE(o_); pc += o_; DISPATCH(); } while (0)
#define UN(name, expr) OP_CASE(name) { const uint64_t x = RB; RA = (expr); NEXT(); }
#define BIN(name, T, expr) OP_CASE(name) { const T a = static_cast<T>(RB); const T b = static_cast<T>(RC); RA = (expr); NEXT(); }
#define BIN_F32(name, expr) OP_CASE(name) { const float a = get_f32(RB); const float b = get_f32(RC); RA = put_f32(expr); NEXT(); }
#define BIN_F64(name, expr) OP_CASE(name) { const double a = get_f64(RB); const double b = get_f64(RC); RA = put_f64(expr); NEXT(); }
#define CMP(name, T, op) OP_CASE(name) { RA = (static_cast<T>(RB) op static_cast<T>(RC)) ? 1 : 0; NEXT(); }
#define CMP_F32(name, op) OP_CASE(name) { RA = (get_f32(RB) op get_f32(RC)) ? 1 : 0; NEXT(); }
#define CMP_F64(name, op) OP_CASE(name) { RA = (get_f64(RB) op get_f64(RC)) ? 1 : 0; NEXT(); }
#define BRANCH(name, T, op) OP_CASE(name) { if (static_cast<T>(RA) op static_cast<T>(RB)) JUMP_BY(decode_imm24(inst)); NEXT(); }
#define LOAD(name, T) OP_CASE(name) { T v_; std::memcpy(&v_, reinterpret_cast<const void*>(RB + static_cast<int64_t>(decode_imm24(inst))), sizeof(T)); RA = static_cast<uint64_t>(v_); NEXT(); }
#define STORE(name, T) OP_CASE(name) { const T v_ = static_cast<T>(RA); std::memcpy(reinterpret_cast<void*>(RB + static_cast<int64_t>(decode_imm24(inst))), &v_, sizeof(T)); NEXT(); }

        OP_CASE(nop) { NEXT(); }
        OP_CASE(unreachable) { throw InterpreterException("Execution reached unreachable instruction"); }

        OP_CASE(iconst32) { RA = decode_uimm32(inst); NEXT(); }
        OP_CASE(load_const) { RA = fn.constants[decode_uimm32(inst)]; NEXT(); }
        OP_CASE(patchable_const32) {
            const PatchConstSite& site = fn.patch_consts[decode_uimm32(inst)];
            RA = static_cast<uint32_t>(get_patched_const(site.symbol, site.default_value));
            NEXT();
        }
        OP_CASE(patchable_const64) {
            const PatchConstSite& site = fn.patch_consts[decode_uimm32(inst)];
            RA = static_cast<uint64_t>(get_patched_const(site.symbol, site.default_value));
            NEXT();
        }

        OP_CASE(mov) { RA = RB; NEXT(); }
        OP_CASE(vmov) {
            RA = RB;
            if (frame.vector_regs) {
                std::memcpy(frame.vector_regs + static_cast<size_t>(decode_a(inst)) * kFastVecBytes,
                            frame.vector_regs + static_cast<size_t>(decode_b(inst)) * kFastVecBytes, kFastVecBytes);
            }
            NEXT();
        }
        OP_CASE(mov_imm) { RA = static_cast<uint64_t>(static_cast<int64_t>(decode_imm32(inst))); NEXT(); }

        UN(sext64, static_cast<uint64_t>(static_cast<int64_t>(static_cast<int32_t>(x))))
        UN(zext64, static_cast<uint32_t>(x))
        UN(trunc32, static_cast<uint32_t>(x))
        UN(trunc8, static_cast<uint8_t>(x))
        UN(fptosi32, static_cast<uint32_t>(static_cast<int32_t>(get_f64(x))))
        UN(fptosi64, static_cast<uint64_t>(static_cast<int64_t>(get_f64(x))))
        UN(fptosi32_f32, static_cast<uint32_t>(static_cast<int32_t>(get_f32(x))))
        UN(fptosi64_f32, static_cast<uint64_t>(static_cast<int64_t>(get_f32(x))))
        UN(sitofp_f64, put_f64(static_cast<double>(static_cast<int32_t>(x))))
        UN(sitofp_f32, put_f32(static_cast<float>(static_cast<int32_t>(x))))
        UN(sitofp_f64_i64, put_f64(static_cast<double>(static_cast<int64_t>(x))))
        UN(sitofp_f32_i64, put_f32(static_cast<float>(static_cast<int64_t>(x))))
        UN(fptrunc_f32, put_f32(static_cast<float>(get_f64(x))))
        UN(fpext_f64, put_f64(static_cast<double>(get_f32(x))))
        UN(bitcast_i64_f64, x)
        UN(bitcast_f64_i64, x)

        BIN(add_i32, uint32_t, static_cast<uint32_t>(a + b))
        BIN(add_i64, uint64_t, a + b)
        BIN(sub_i32, uint32_t, static_cast<uint32_t>(a - b))
        BIN(sub_i64, uint64_t, a - b)
        BIN(mul_i32, uint32_t, static_cast<uint32_t>(a * b))
        BIN(mul_i64, uint64_t, a * b)
        OP_CASE(add_imm_i32) { RA = static_cast<uint32_t>(static_cast<uint32_t>(RB) + static_cast<uint32_t>(decode_imm24(inst))); NEXT(); }
        OP_CASE(add_imm_i64) { RA = RB + static_cast<uint64_t>(static_cast<int64_t>(decode_imm24(inst))); NEXT(); }
        OP_CASE(index_addr) { RA = RB + RC * decode_d(inst); NEXT(); }

        OP_CASE(sdiv_i32) {
            const int32_t b = static_cast<int32_t>(RC);
            if (BRASS_UNLIKELY(b == 0)) throw InterpreterException("Division by zero");
            const int32_t a = static_cast<int32_t>(RB);
            RA = (BRASS_UNLIKELY(a == std::numeric_limits<int32_t>::min() && b == -1)) ? static_cast<uint32_t>(a) : static_cast<uint32_t>(a / b);
            NEXT();
        }
        OP_CASE(sdiv_i64) {
            const int64_t b = static_cast<int64_t>(RC);
            if (BRASS_UNLIKELY(b == 0)) throw InterpreterException("Division by zero");
            const int64_t a = static_cast<int64_t>(RB);
            RA = (BRASS_UNLIKELY(a == std::numeric_limits<int64_t>::min() && b == -1)) ? static_cast<uint64_t>(a) : static_cast<uint64_t>(a / b);
            NEXT();
        }
        OP_CASE(udiv_i32) {
            const uint32_t b = static_cast<uint32_t>(RC);
            if (BRASS_UNLIKELY(b == 0)) throw InterpreterException("Division by zero");
            RA = static_cast<uint32_t>(RB) / b;
            NEXT();
        }
        OP_CASE(udiv_i64) {
            const uint64_t b = RC;
            if (BRASS_UNLIKELY(b == 0)) throw InterpreterException("Division by zero");
            RA = RB / b;
            NEXT();
        }
        OP_CASE(smod_i32) {
            const int32_t b = static_cast<int32_t>(RC);
            if (BRASS_UNLIKELY(b == 0)) throw InterpreterException("Modulo by zero");
            const int32_t a = static_cast<int32_t>(RB);
            RA = (BRASS_UNLIKELY(a == std::numeric_limits<int32_t>::min() && b == -1)) ? 0 : static_cast<uint32_t>(a % b);
            NEXT();
        }
        OP_CASE(smod_i64) {
            const int64_t b = static_cast<int64_t>(RC);
            if (BRASS_UNLIKELY(b == 0)) throw InterpreterException("Modulo by zero");
            const int64_t a = static_cast<int64_t>(RB);
            RA = (BRASS_UNLIKELY(a == std::numeric_limits<int64_t>::min() && b == -1)) ? 0 : static_cast<uint64_t>(a % b);
            NEXT();
        }
        OP_CASE(umod_i32) {
            const uint32_t b = static_cast<uint32_t>(RC);
            if (BRASS_UNLIKELY(b == 0)) throw InterpreterException("Modulo by zero");
            RA = static_cast<uint32_t>(RB) % b;
            NEXT();
        }
        OP_CASE(umod_i64) {
            const uint64_t b = RC;
            if (BRASS_UNLIKELY(b == 0)) throw InterpreterException("Modulo by zero");
            RA = RB % b;
            NEXT();
        }

        UN(neg_i32, static_cast<uint32_t>(0u - static_cast<uint32_t>(x)))
        UN(neg_i64, 0ull - x)

        BIN_F32(add_f32, a + b)
        BIN_F64(add_f64, a + b)
        BIN_F32(sub_f32, a - b)
        BIN_F64(sub_f64, a - b)
        BIN_F32(mul_f32, a * b)
        BIN_F64(mul_f64, a * b)
        BIN_F32(fdiv_f32, a / b)
        BIN_F64(fdiv_f64, a / b)
        UN(neg_f32, put_f32(-get_f32(x)))
        UN(neg_f64, put_f64(-get_f64(x)))
        // dst already holds the addend.
        OP_CASE(fma_f32) { RA = put_f32(std::fma(get_f32(RB), get_f32(RC), get_f32(RA))); NEXT(); }
        OP_CASE(fma_f64) { RA = put_f64(std::fma(get_f64(RB), get_f64(RC), get_f64(RA))); NEXT(); }
        UN(sqrt_f32, put_f32(std::sqrt(get_f32(x))))
        UN(sqrt_f64, put_f64(std::sqrt(get_f64(x))))
        UN(fabs_f32, put_f32(std::fabs(get_f32(x))))
        UN(fabs_f64, put_f64(std::fabs(get_f64(x))))
        UN(floor_f32, put_f32(std::floor(get_f32(x))))
        UN(floor_f64, put_f64(std::floor(get_f64(x))))
        UN(ceil_f32, put_f32(std::ceil(get_f32(x))))
        UN(ceil_f64, put_f64(std::ceil(get_f64(x))))
        UN(round_f32, put_f32(std::round(get_f32(x))))
        UN(round_f64, put_f64(std::round(get_f64(x))))
        BIN_F32(fmin_f32, std::fmin(a, b))
        BIN_F64(fmin_f64, std::fmin(a, b))
        BIN_F32(fmax_f32, std::fmax(a, b))
        BIN_F64(fmax_f64, std::fmax(a, b))

        BIN(sadd_overflow_i32, int32_t, (static_cast<int64_t>(a) + b < INT32_MIN || static_cast<int64_t>(a) + b > INT32_MAX) ? 1 : 0)
        BIN(ssub_overflow_i32, int32_t, (static_cast<int64_t>(a) - b < INT32_MIN || static_cast<int64_t>(a) - b > INT32_MAX) ? 1 : 0)
        BIN(smul_overflow_i32, int32_t, (static_cast<int64_t>(a) * b < INT32_MIN || static_cast<int64_t>(a) * b > INT32_MAX) ? 1 : 0)
        BIN(sadd_overflow_i64, int64_t, ((b > 0 && a > INT64_MAX - b) || (b < 0 && a < INT64_MIN - b)) ? 1 : 0)
        BIN(ssub_overflow_i64, int64_t, ((b < 0 && a > INT64_MAX + b) || (b > 0 && a < INT64_MIN + b)) ? 1 : 0)
        OP_CASE(smul_overflow_i64) {
            int64_t res = 0;
            RA = mul_overflows(static_cast<int64_t>(RB), static_cast<int64_t>(RC), res) ? 1 : 0;
            NEXT();
        }
        BIN(uadd_overflow_i32, uint32_t, static_cast<uint32_t>(a + b) < a ? 1 : 0)
        BIN(uadd_overflow_i64, uint64_t, a + b < a ? 1 : 0)
        BIN(usub_overflow_i32, uint32_t, a < b ? 1 : 0)
        BIN(usub_overflow_i64, uint64_t, a < b ? 1 : 0)
        BIN(umul_overflow_i32, uint64_t, (static_cast<uint32_t>(a) * static_cast<uint64_t>(static_cast<uint32_t>(b))) > UINT32_MAX ? 1 : 0)
        OP_CASE(umul_overflow_i64) {
            uint64_t res = 0;
            RA = mul_overflows(RB, RC, res) ? 1 : 0;
            NEXT();
        }

        BIN(and_i32, uint64_t, static_cast<uint32_t>(a & b))
        BIN(and_i64, uint64_t, a & b)
        BIN(or_i32, uint64_t, static_cast<uint32_t>(a | b))
        BIN(or_i64, uint64_t, a | b)
        BIN(xor_i32, uint64_t, static_cast<uint32_t>(a ^ b))
        BIN(xor_i64, uint64_t, a ^ b)
        BIN(shl_i32, uint32_t, static_cast<uint32_t>(a << (b & 31)))
        BIN(shl_i64, uint64_t, a << (b & 63))
        BIN(lshr_i32, uint32_t, a >> (b & 31))
        BIN(lshr_i64, uint64_t, a >> (b & 63))
        BIN(ashr_i32, uint32_t, static_cast<uint32_t>(static_cast<int32_t>(a) >> (b & 31)))
        BIN(ashr_i64, uint64_t, static_cast<uint64_t>(static_cast<int64_t>(a) >> (b & 63)))
        UN(not_i32, static_cast<uint32_t>(~static_cast<uint32_t>(x)))
        UN(not_i64, ~x)
        UN(clz_i32, static_cast<uint64_t>(std::countl_zero(static_cast<uint32_t>(x))))
        UN(clz_i64, static_cast<uint64_t>(std::countl_zero(x)))
        UN(ctz_i32, static_cast<uint64_t>(std::countr_zero(static_cast<uint32_t>(x))))
        UN(ctz_i64, static_cast<uint64_t>(std::countr_zero(x)))
        UN(popcnt_i32, static_cast<uint64_t>(std::popcount(static_cast<uint32_t>(x))))
        UN(popcnt_i64, static_cast<uint64_t>(std::popcount(x)))

        CMP(eq_i32, uint32_t, ==) CMP(eq_i64, uint64_t, ==) CMP_F32(eq_f32, ==) CMP_F64(eq_f64, ==)
        CMP(ne_i32, uint32_t, !=) CMP(ne_i64, uint64_t, !=) CMP_F32(ne_f32, !=) CMP_F64(ne_f64, !=)
        CMP(slt_i32, int32_t, <) CMP(slt_i64, int64_t, <) CMP(ult_i32, uint32_t, <) CMP(ult_i64, uint64_t, <)
        CMP_F32(lt_f32, <) CMP_F64(lt_f64, <)
        CMP(sle_i32, int32_t, <=) CMP(sle_i64, int64_t, <=) CMP(ule_i32, uint32_t, <=) CMP(ule_i64, uint64_t, <=)
        CMP_F32(le_f32, <=) CMP_F64(le_f64, <=)
        CMP(sgt_i32, int32_t, >) CMP(sgt_i64, int64_t, >) CMP(ugt_i32, uint32_t, >) CMP(ugt_i64, uint64_t, >)
        CMP_F32(gt_f32, >) CMP_F64(gt_f64, >)
        CMP(sge_i32, int32_t, >=) CMP(sge_i64, int64_t, >=) CMP(uge_i32, uint32_t, >=) CMP(uge_i64, uint64_t, >=)
        CMP_F32(ge_f32, >=) CMP_F64(ge_f64, >=)

        // dst already holds the false value; d = 1 moves vector state too.
        OP_CASE(select) {
            if (RB != 0) {
                RA = RC;
                if (decode_d(inst) != 0 && frame.vector_regs) {
                    std::memcpy(frame.vector_regs + static_cast<size_t>(decode_a(inst)) * kFastVecBytes,
                                frame.vector_regs + static_cast<size_t>(decode_c(inst)) * kFastVecBytes, kFastVecBytes);
                }
            }
            NEXT();
        }

        LOAD(load8, uint8_t)
        LOAD(load16, uint16_t)
        LOAD(load32, uint32_t)
        LOAD(load64, uint64_t)
        STORE(store8, uint8_t)
        STORE(store16, uint16_t)
        STORE(store32, uint32_t)
        STORE(store64, uint64_t)

        OP_CASE(alloca_) {
            // The size is the next code word; the alignment is d.
            const size_t size = static_cast<size_t>(pc[1]);
            RA = reinterpret_cast<uintptr_t>(alloca_arena_->allocate(size, decode_d(inst)));
            pc += 2;
            DISPATCH();
        }

        OP_CASE(jump) { JUMP_BY(decode_imm32(inst)); }
        OP_CASE(jump_if) { if (RA != 0) JUMP_BY(decode_imm32(inst)); NEXT(); }
        OP_CASE(jump_if_not) { if (RA == 0) JUMP_BY(decode_imm32(inst)); NEXT(); }

        BRANCH(br_eq_i32, uint32_t, ==) BRANCH(br_ne_i32, uint32_t, !=)
        BRANCH(br_slt_i32, int32_t, <) BRANCH(br_sle_i32, int32_t, <=)
        BRANCH(br_ult_i32, uint32_t, <) BRANCH(br_ule_i32, uint32_t, <=)
        BRANCH(br_eq_i64, uint64_t, ==) BRANCH(br_ne_i64, uint64_t, !=)
        BRANCH(br_slt_i64, int64_t, <) BRANCH(br_sle_i64, int64_t, <=)
        BRANCH(br_ult_i64, uint64_t, <) BRANCH(br_ule_i64, uint64_t, <=)

        OP_CASE(ret) {
            const uint32_t r = decode_a(inst);
            return marshal_return_value(fn, registers[r], frame.vector_regs, r);
        }
        OP_CASE(ret_void) { return RuntimeValue::from_void(); }

        OP_CASE(switch_) {
            const SwitchTable& table = fn.switch_tables[decode_uimm32(inst)];
            const int64_t val = table.is_i32 ? static_cast<int64_t>(static_cast<int32_t>(RA)) : static_cast<int64_t>(RA);
            int32_t target_pc = table.default_offset;
            // Cases are sorted by value; the first of equal values wins.
            auto it = std::lower_bound(table.cases.begin(), table.cases.end(), val,
                                       [](const std::pair<int64_t, int32_t>& c, int64_t v) { return c.first < v; });
            if (it != table.cases.end() && it->first == val) target_pc = it->second;
            JUMP_BY(target_pc - static_cast<int32_t>(pc - code_base));
        }

        OP_CASE(call)
        OP_CASE(patchable_call) {
            execute_call(frame, decode_uimm32(inst));
            NEXT();
        }
        OP_CASE(call_indirect) {
            execute_call_indirect(frame, decode_uimm32(inst));
            NEXT();
        }

        OP_CASE(func_addr) {
            const std::string& sym = fn.string_pool[decode_uimm32(inst)];
            const Function* target_fn = module_ ? module_->get_function(sym) : nullptr;
            uintptr_t fn_ptr = reinterpret_cast<uintptr_t>(target_fn);
            if (target_fn) {
                register_function_pointer(fn_ptr, target_fn);
            } else if (const BytecodeFunction* bfn = bytecode_module_ ? bytecode_module_->get_function(sym) : nullptr) {
                fn_ptr = reinterpret_cast<uintptr_t>(bfn);
                register_function_pointer(fn_ptr, bfn);
            } else if (void* ext_sym = find_external_symbol(sym)) {
                fn_ptr = reinterpret_cast<uintptr_t>(ext_sym);
            }
            RA = fn_ptr;
            NEXT();
        }

        OP_CASE(safepoint) { handle_safepoint(frame); NEXT(); }
        OP_CASE(write_barrier) { handle_write_barrier(frame, decode_b(inst), decode_c(inst)); NEXT(); }

        OP_CASE(guard) {
            if (BRASS_LIKELY(RA != 0)) NEXT();
            const GuardInfo& g = fn.guards[decode_uimm32(inst)];
            last_deopt_.deoptimized = true;
            last_deopt_.exit_stub = g.exit_stub;
            last_deopt_.resume_id = g.resume_id;
            last_deopt_.state_map.clear();
            last_deopt_.state_map.reserve(g.state_regs.size());
            // Own scope: the resume path below leaves by computed goto,
            // which would skip df's destructor.
            {
                runtime::DeoptFrame df;
                df.resume_id = g.resume_id;
                df.exit_symbol = g.exit_stub;
                df.reason = runtime::DeoptReason::Generic;
                for (BcReg sreg : g.state_regs) {
                    RuntimeValue rv = fast_reg_value(frame, sreg);
                    last_deopt_.state_map.push_back(rv);
                    df.push_value(registers[sreg], runtime::DeoptValue::from_runtime_value(rv).kind);
                }
                runtime::set_thread_deopt_frame(&df);
            }
            if (deopt_handler_) {
                return deopt_handler_(*this, last_deopt_);
            }
            if (module_ && !g.exit_stub.empty()) {
                if (const Function* stub_fn = module_->get_function(g.exit_stub)) {
                    return run(*stub_fn, last_deopt_.state_map);
                }
            }
            for (const auto& rp : fn.resume_points) {
                if (rp.resume_id != g.resume_id) continue;
                for (size_t i = 0; i < last_deopt_.state_map.size() && i < rp.param_regs.size(); ++i) {
                    fast_set_reg(frame, rp.param_regs[i], last_deopt_.state_map[i]);
                }
                JUMP_BY(static_cast<int32_t>(rp.target_pc) - static_cast<int32_t>(pc - code_base));
            }
            throw DeoptException(last_deopt_);
        }

        OP_CASE(resume_point) { NEXT(); }
        OP_CASE(osr_entry) { NEXT(); }

        OP_CASE(pinned_tls_write) { tls_block_ = RB; NEXT(); }
        OP_CASE(pinned_tls_read) {
            if (tls_block_ == 0) {
                void* sym = find_external_symbol("bronze_tls_enter");
                if (!sym) sym = find_external_symbol("bronze_tls_block_addr");
                if (sym) tls_block_ = reinterpret_cast<uint64_t>(reinterpret_cast<void* (*)()>(sym)());
            }
            RA = tls_block_;
            NEXT();
        }
        OP_CASE(read_sp) {
            char marker = 0;
            RA = reinterpret_cast<uint64_t>(&marker);
            NEXT();
        }

        OP_CASE(throw_) {
            handle_throw(frame, decode_a(inst), pc, code_base);
            // A throw caught in this frame can loop without a backward
            // branch, so it is charged too.
            if (BRASS_UNLIKELY(++total_instructions_executed_ > insn_limit)) throw_instruction_limit();
            DISPATCH();
        }
        OP_CASE(invoke) {
            handle_invoke(frame, decode_uimm32(inst), pc, code_base);
            DISPATCH();
        }
        OP_CASE(landing_pad) { RA = current_exception_.raw_bits(); NEXT(); }
        OP_CASE(resume) { handle_resume(frame, decode_a(inst)); NEXT(); }

        OP_CASE(coro_create) {
            const CallSiteInfo& cs = fn.call_sites[decode_uimm32(inst)];
            // Own scope: DISPATCH() leaves by computed goto, which would skip
            // the vector's destructor and leak it.
            {
                std::vector<RuntimeValue> args;
                args.reserve(cs.arg_regs.size());
                for (BcReg ar : cs.arg_regs) args.push_back(fast_reg_value(frame, ar));
                RA = coro_create(cs.callee, args);
            }
            NEXT();
        }
        OP_CASE(coro_suspend) {
            // The resume id is the next code word.
            frame.pc = static_cast<uint32_t>(pc - code_base);
            coro_suspend(frame, decode_a(inst), decode_b(inst), static_cast<uint32_t>(pc[1]));
            pc += 2;
            DISPATCH();
        }
        OP_CASE(coro_resume) {
            const uint32_t input_reg = decode_c(inst);
            const uint64_t input_val = input_reg != kNoReg ? registers[input_reg] : 0;
            RA = coro_resume(static_cast<uintptr_t>(RB), input_val);
            NEXT();
        }
        OP_CASE(coro_destroy) { coro_destroy(static_cast<uintptr_t>(RB)); NEXT(); }

        OP_CASE(vadd) OP_CASE(vsub) OP_CASE(vmul) OP_CASE(vdiv) OP_CASE(vfma) OP_CASE(vneg)
        OP_CASE(vmin) OP_CASE(vmax) OP_CASE(vsqrt) OP_CASE(vand) OP_CASE(vor) OP_CASE(vxor)
        OP_CASE(vnot) OP_CASE(vload) OP_CASE(vstore) OP_CASE(vbroadcast)
        OP_CASE(vextract_lane) OP_CASE(vinsert_lane) OP_CASE(vzero) {
            execute_vector_op(frame, inst, pc);
            NEXT();
        }
        OP_CASE(vshuffle) {
            execute_vector_op(frame, inst, pc);
            pc += 2;
            DISPATCH();
        }

#if BRASS_DIRECT_THREADED
    do_invalid_op:
#else
        default:
#endif
            throw InterpreterException("Invalid opcode " + std::to_string(static_cast<int>(decode_op(inst))) +
                                       " in bytecode function " + fn.name);
    }
    // Unreachable: every handler dispatches onward, returns, or throws.

#undef NEXT
#undef JUMP_BY
#undef UN
#undef BIN
#undef BIN_F32
#undef BIN_F64
#undef CMP
#undef CMP_F32
#undef CMP_F64
#undef BRANCH
#undef LOAD
#undef STORE
#undef OP_CASE
#undef DISPATCH
#undef BACKEDGE
#undef RA
#undef RB
#undef RC
}

} // namespace brass

#if defined(__GNUC__) || defined(__clang__)
#pragma GCC diagnostic pop
#endif
