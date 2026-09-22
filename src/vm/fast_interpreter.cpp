#include "fast_interpreter_impl.hpp"

#if defined(__GNUC__) || defined(__clang__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wpedantic"
#endif

namespace brass {

FastInterpreter::FastInterpreter(size_t gc_semispace_size)
    : gc_(gc_semispace_size) {
    gc_.set_root_provider([this](std::vector<uintptr_t*>& roots) {
        this->collect_all_roots(roots);
    });
    register_builtin_host_functions();
}

FastInterpreter::~FastInterpreter() = default;

FastInterpreter::FastInterpreter(FastInterpreter&&) noexcept = default;
FastInterpreter& FastInterpreter::operator=(FastInterpreter&&) noexcept = default;

static thread_local FastInterpreter* s_current_fast_interp = nullptr;

FastInterpreter* FastInterpreter::current() noexcept {
    return s_current_fast_interp;
}

void FastInterpreter::set_current(FastInterpreter* interp) noexcept {
    s_current_fast_interp = interp;
}

void FastInterpreter::clear_compile_cache() noexcept {
    compiled_functions_.clear();
}

const BytecodeFunction* FastInterpreter::get_or_compile(const Function& fn) {
    auto it = compiled_functions_.find(&fn);
    if (it != compiled_functions_.end()) {
        return it->second.get();
    }
    auto bfn = compiler_.compile(fn);
    const BytecodeFunction* ptr = bfn.get();
    compiled_functions_[&fn] = std::move(bfn);
    return ptr;
}

void FastInterpreter::set_generational_gc(GenerationalGC* gc) noexcept {
    gen_gc_ = gc;
    if (gen_gc_) {
        gen_gc_->set_root_provider([this](std::vector<uintptr_t*>& roots) {
            this->collect_all_roots(roots);
        });
    }
}

uintptr_t FastInterpreter::allocate_gc(size_t size, uint64_t pointer_mask, uint32_t type_tag) {
    std::vector<uintptr_t*> roots;
    collect_all_roots(roots);
    if (gen_gc_) {
        return gen_gc_->allocate(size, pointer_mask, type_tag, roots);
    }
    return gc_.allocate(size, pointer_mask, type_tag, roots);
}

void FastInterpreter::collect_all_roots(std::vector<uintptr_t*>& roots) {
    // 1. Walk active frames on the call stack
    for (FastFrame* f = current_frame_; f != nullptr; f = f->caller) {
        for (uint32_t i = 0; i < f->num_registers; ++i) {
            if (f->registers[i] == 0) continue;
            bool is_gc = false;
            if (f->bfn && i < f->bfn->register_types.size()) {
                is_gc = f->bfn->register_types[i].is_gcref();
            }
            if (!is_gc) {
                uintptr_t val = static_cast<uintptr_t>(f->registers[i]);
                is_gc = gc_.is_valid_object(val) || (gen_gc_ && gen_gc_->is_valid_object(val));
            }
            if (is_gc) {
                roots.push_back(reinterpret_cast<uintptr_t*>(&f->registers[i]));
            }
        }
    }

    // 2. Scan suspended coroutines
    for (auto& [handle, coro] : active_coros_) {
        if (coro && !coro->is_done) {
            for (size_t i = 0; i < coro->registers.size(); ++i) {
                if (coro->registers[i] == 0) continue;
                bool is_gc = false;
                if (coro->bfn && i < coro->bfn->register_types.size()) {
                    is_gc = coro->bfn->register_types[i].is_gcref();
                }
                if (!is_gc) {
                    uintptr_t val = static_cast<uintptr_t>(coro->registers[i]);
                    is_gc = gc_.is_valid_object(val) || (gen_gc_ && gen_gc_->is_valid_object(val));
                }
                if (is_gc) {
                    roots.push_back(reinterpret_cast<uintptr_t*>(&coro->registers[i]));
                }
            }
        }
    }

    // 3. Scan current_exception_ if gcref
    if (current_exception_.is_gcref() && !current_exception_.is_null()) {
        roots.push_back(reinterpret_cast<uintptr_t*>(&current_exception_.raw_bits_ref()));
    }

    // 4. Scan last_deopt_ state map if gcref
    for (auto& val : last_deopt_.state_map) {
        if (val.is_gcref() && !val.is_null()) {
            roots.push_back(reinterpret_cast<uintptr_t*>(&val.raw_bits_ref()));
        }
    }
}

RuntimeValue FastInterpreter::resume(const Function& fn, uint32_t resume_id, const std::vector<RuntimeValue>& state_values) {
    if (fn.parent()) {
        module_ = fn.parent();
    }
    const BytecodeFunction* bfn = get_or_compile(fn);
    return resume(*bfn, resume_id, state_values);
}

RuntimeValue FastInterpreter::resume(const BytecodeFunction& fn, uint32_t resume_id, const std::vector<RuntimeValue>& state_values) {
    const ResumePointEntry* target_entry = nullptr;
    for (const auto& rp : fn.resume_points) {
        if (rp.resume_id == resume_id) {
            target_entry = &rp;
            break;
        }
    }
    if (!target_entry) {
        throw InterpreterException("Resume target ID " + std::to_string(resume_id) + " not found in function " + fn.name);
    }

    uint32_t num_regs = std::max<uint32_t>(fn.num_registers, 1);
    uint64_t* registers = static_cast<uint64_t*>(BRASS_ALLOCA(num_regs * sizeof(uint64_t)));
    std::memset(registers, 0, num_regs * sizeof(uint64_t));

    FastFrame frame;
    frame.bfn = &fn;
    frame.mir_fn = module_ ? module_->get_function(fn.name) : nullptr;
    frame.registers = registers;
    frame.num_registers = num_regs;
    frame.pc = target_entry->target_pc;

    if (!target_entry->param_regs.empty()) {
        for (size_t i = 0; i < state_values.size() && i < target_entry->param_regs.size(); ++i) {
            uint8_t reg = target_entry->param_regs[i];
            if (reg < num_regs) {
                registers[reg] = state_values[i].raw_bits();
                if (state_values[i].is_vector()) {
                    uint8_t* vregs = frame.ensure_vector_regs();
                    std::memcpy(vregs + reg * 32, state_values[i].vec_bytes(), 32);
                }
            }
        }
    } else {
        for (size_t i = 0; i < state_values.size() && i < num_regs; ++i) {
            registers[i] = state_values[i].raw_bits();
            if (state_values[i].is_vector()) {
                uint8_t* vregs = frame.ensure_vector_regs();
                std::memcpy(vregs + i * 32, state_values[i].vec_bytes(), 32);
            }
        }
    }

    FrameGuard guard(*this, frame);
    return execute_frame(frame);
}

RuntimeValue FastInterpreter::execute_frame(FastFrame& frame) {
    const BytecodeFunction& fn = *frame.bfn;
    const uint32_t* const code_base = fn.code.data();
    if (code_base == nullptr || fn.code.empty()) {
        return RuntimeValue::from_void();
    }

    const uint32_t* pc = code_base + frame.pc;
    uint64_t* const registers = frame.registers;
    uint32_t inst = *pc;

#if defined(__GNUC__) || defined(__clang__)
#define BRASS_DIRECT_THREADED 1
    static void* dispatch_table[256];
    static bool table_inited = false;
    if (BRASS_UNLIKELY(!table_inited)) {
        for (size_t i = 0; i < 256; ++i) dispatch_table[i] = &&do_unreachable;
#define TENTRY(op) dispatch_table[static_cast<size_t>(BytecodeOp::op)] = &&do_##op
        TENTRY(nop); TENTRY(unreachable); TENTRY(iconst32); TENTRY(iconst64);
        TENTRY(fconst32); TENTRY(fconst64); TENTRY(load_const);
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
        TENTRY(store8); TENTRY(store16); TENTRY(store32); TENTRY(store64);
        TENTRY(alloca_); TENTRY(load_indexed); TENTRY(store_indexed);
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
#undef TENTRY
        table_inited = true;
    }

#define OP_CASE(name) do_##name:
#define DISPATCH() do { \
    ++total_instructions_executed_; \
    if (BRASS_UNLIKELY(max_instructions_ > 0 && total_instructions_executed_ > max_instructions_)) { \
        throw InterpreterException("Maximum instruction execution count exceeded (" + std::to_string(max_instructions_) + ")"); \
    } \
    inst = *pc; \
    goto *dispatch_table[inst & 0xFF]; \
} while (0)

    goto *dispatch_table[inst & 0xFF];

#else
#define BRASS_DIRECT_THREADED 0
#define OP_CASE(name) case BytecodeOp::name:
#define DISPATCH() do { \
    ++total_instructions_executed_; \
    if (BRASS_UNLIKELY(max_instructions_ > 0 && total_instructions_executed_ > max_instructions_)) { \
        throw InterpreterException("Maximum instruction execution count exceeded (" + std::to_string(max_instructions_) + ")"); \
    } \
    goto loop_start; \
} while (0)

loop_start:
    inst = *pc;
    switch (decode_op(inst))
#endif
    {
        OP_CASE(nop) { pc++; DISPATCH(); }
        OP_CASE(unreachable) { throw InterpreterException("Execution reached unreachable instruction"); }

        OP_CASE(iconst32) { registers[decode_dst(inst)] = static_cast<uint64_t>(decode_u16(inst)); pc++; DISPATCH(); }
        OP_CASE(iconst64) { registers[decode_dst(inst)] = static_cast<uint64_t>(decode_u16(inst)); pc++; DISPATCH(); }
        OP_CASE(fconst32) { registers[decode_dst(inst)] = static_cast<uint64_t>(decode_u16(inst)); pc++; DISPATCH(); }
        OP_CASE(fconst64) { registers[decode_dst(inst)] = static_cast<uint64_t>(decode_u16(inst)); pc++; DISPATCH(); }
        OP_CASE(load_const) { registers[decode_dst(inst)] = fn.constants[decode_u16(inst)]; pc++; DISPATCH(); }

        OP_CASE(patchable_const32) {
            uint16_t s_idx = decode_u16(inst);
            const std::string& sym = fn.string_pool[s_idx];
            int64_t v = get_patched_const(sym, 0);
            registers[decode_dst(inst)] = static_cast<uint64_t>(static_cast<uint32_t>(static_cast<int32_t>(v)));
            pc++;
            DISPATCH();
        }

        OP_CASE(patchable_const64) {
            uint16_t s_idx = decode_u16(inst);
            const std::string& sym = fn.string_pool[s_idx];
            int64_t v = get_patched_const(sym, 0);
            registers[decode_dst(inst)] = static_cast<uint64_t>(v);
            pc++;
            DISPATCH();
        }

        OP_CASE(mov) { registers[decode_dst(inst)] = registers[decode_src1(inst)]; pc++; DISPATCH(); }
        OP_CASE(mov_imm) { registers[decode_dst(inst)] = static_cast<uint64_t>(static_cast<int64_t>(decode_s16(inst))); pc++; DISPATCH(); }
        OP_CASE(sext64) { registers[decode_dst(inst)] = static_cast<uint64_t>(static_cast<int64_t>(static_cast<int32_t>(registers[decode_src1(inst)]))); pc++; DISPATCH(); }
        OP_CASE(zext64) { registers[decode_dst(inst)] = static_cast<uint64_t>(static_cast<uint32_t>(registers[decode_src1(inst)])); pc++; DISPATCH(); }
        OP_CASE(trunc32) { registers[decode_dst(inst)] = static_cast<uint64_t>(static_cast<uint32_t>(registers[decode_src1(inst)])); pc++; DISPATCH(); }
        OP_CASE(trunc8) { registers[decode_dst(inst)] = static_cast<uint64_t>(static_cast<uint8_t>(registers[decode_src1(inst)])); pc++; DISPATCH(); }
        OP_CASE(fptosi32) { registers[decode_dst(inst)] = static_cast<uint64_t>(static_cast<uint32_t>(static_cast<int32_t>(get_f64(registers[decode_src1(inst)])))); pc++; DISPATCH(); }
        OP_CASE(fptosi64) { registers[decode_dst(inst)] = static_cast<uint64_t>(static_cast<int64_t>(get_f64(registers[decode_src1(inst)]))); pc++; DISPATCH(); }
        OP_CASE(fptosi32_f32) { registers[decode_dst(inst)] = static_cast<uint64_t>(static_cast<uint32_t>(static_cast<int32_t>(get_f32(registers[decode_src1(inst)])))); pc++; DISPATCH(); }
        OP_CASE(fptosi64_f32) { registers[decode_dst(inst)] = static_cast<uint64_t>(static_cast<int64_t>(get_f32(registers[decode_src1(inst)]))); pc++; DISPATCH(); }
        OP_CASE(sitofp_f64) { registers[decode_dst(inst)] = put_f64(static_cast<double>(static_cast<int32_t>(registers[decode_src1(inst)]))); pc++; DISPATCH(); }
        OP_CASE(sitofp_f32) { registers[decode_dst(inst)] = put_f32(static_cast<float>(static_cast<int32_t>(registers[decode_src1(inst)]))); pc++; DISPATCH(); }
        OP_CASE(sitofp_f64_i64) { registers[decode_dst(inst)] = put_f64(static_cast<double>(static_cast<int64_t>(registers[decode_src1(inst)]))); pc++; DISPATCH(); }
        OP_CASE(sitofp_f32_i64) { registers[decode_dst(inst)] = put_f32(static_cast<float>(static_cast<int64_t>(registers[decode_src1(inst)]))); pc++; DISPATCH(); }
        OP_CASE(fptrunc_f32) { registers[decode_dst(inst)] = put_f32(static_cast<float>(get_f64(registers[decode_src1(inst)]))); pc++; DISPATCH(); }
        OP_CASE(fpext_f64) { registers[decode_dst(inst)] = put_f64(static_cast<double>(get_f32(registers[decode_src1(inst)]))); pc++; DISPATCH(); }
        OP_CASE(bitcast_i64_f64) { registers[decode_dst(inst)] = registers[decode_src1(inst)]; pc++; DISPATCH(); }
        OP_CASE(bitcast_f64_i64) { registers[decode_dst(inst)] = registers[decode_src1(inst)]; pc++; DISPATCH(); }

        OP_CASE(add_i32) { registers[decode_dst(inst)] = static_cast<uint32_t>(static_cast<uint32_t>(registers[decode_src1(inst)]) + static_cast<uint32_t>(registers[decode_src2(inst)])); pc++; DISPATCH(); }
        OP_CASE(add_i64) { registers[decode_dst(inst)] = registers[decode_src1(inst)] + registers[decode_src2(inst)]; pc++; DISPATCH(); }
        OP_CASE(sub_i32) { registers[decode_dst(inst)] = static_cast<uint32_t>(static_cast<uint32_t>(registers[decode_src1(inst)]) - static_cast<uint32_t>(registers[decode_src2(inst)])); pc++; DISPATCH(); }
        OP_CASE(sub_i64) { registers[decode_dst(inst)] = registers[decode_src1(inst)] - registers[decode_src2(inst)]; pc++; DISPATCH(); }
        OP_CASE(mul_i32) { registers[decode_dst(inst)] = static_cast<uint32_t>(static_cast<uint32_t>(registers[decode_src1(inst)]) * static_cast<uint32_t>(registers[decode_src2(inst)])); pc++; DISPATCH(); }
        OP_CASE(mul_i64) { registers[decode_dst(inst)] = registers[decode_src1(inst)] * registers[decode_src2(inst)]; pc++; DISPATCH(); }

        OP_CASE(sdiv_i32) {
            int32_t b = static_cast<int32_t>(registers[decode_src2(inst)]);
            if (BRASS_UNLIKELY(b == 0)) throw InterpreterException("Division by zero");
            int32_t a = static_cast<int32_t>(registers[decode_src1(inst)]);
            registers[decode_dst(inst)] = (BRASS_UNLIKELY(a == std::numeric_limits<int32_t>::min() && b == -1)) ? static_cast<uint32_t>(a) : static_cast<uint32_t>(a / b);
            pc++;
            DISPATCH();
        }

        OP_CASE(sdiv_i64) {
            int64_t b = static_cast<int64_t>(registers[decode_src2(inst)]);
            if (BRASS_UNLIKELY(b == 0)) throw InterpreterException("Division by zero");
            int64_t a = static_cast<int64_t>(registers[decode_src1(inst)]);
            registers[decode_dst(inst)] = (BRASS_UNLIKELY(a == std::numeric_limits<int64_t>::min() && b == -1)) ? static_cast<uint64_t>(a) : static_cast<uint64_t>(a / b);
            pc++;
            DISPATCH();
        }

        OP_CASE(udiv_i32) {
            uint32_t b = static_cast<uint32_t>(registers[decode_src2(inst)]);
            if (BRASS_UNLIKELY(b == 0)) throw InterpreterException("Division by zero");
            registers[decode_dst(inst)] = static_cast<uint32_t>(registers[decode_src1(inst)]) / b;
            pc++;
            DISPATCH();
        }

        OP_CASE(udiv_i64) {
            uint64_t b = registers[decode_src2(inst)];
            if (BRASS_UNLIKELY(b == 0)) throw InterpreterException("Division by zero");
            registers[decode_dst(inst)] = registers[decode_src1(inst)] / b;
            pc++;
            DISPATCH();
        }

        OP_CASE(smod_i32) {
            int32_t b = static_cast<int32_t>(registers[decode_src2(inst)]);
            if (BRASS_UNLIKELY(b == 0)) throw InterpreterException("Modulo by zero");
            int32_t a = static_cast<int32_t>(registers[decode_src1(inst)]);
            registers[decode_dst(inst)] = (BRASS_UNLIKELY(a == std::numeric_limits<int32_t>::min() && b == -1)) ? 0 : static_cast<uint32_t>(a % b);
            pc++;
            DISPATCH();
        }

        OP_CASE(smod_i64) {
            int64_t b = static_cast<int64_t>(registers[decode_src2(inst)]);
            if (BRASS_UNLIKELY(b == 0)) throw InterpreterException("Modulo by zero");
            int64_t a = static_cast<int64_t>(registers[decode_src1(inst)]);
            registers[decode_dst(inst)] = (BRASS_UNLIKELY(a == std::numeric_limits<int64_t>::min() && b == -1)) ? 0 : static_cast<uint64_t>(a % b);
            pc++;
            DISPATCH();
        }

        OP_CASE(umod_i32) {
            uint32_t b = static_cast<uint32_t>(registers[decode_src2(inst)]);
            if (BRASS_UNLIKELY(b == 0)) throw InterpreterException("Modulo by zero");
            registers[decode_dst(inst)] = static_cast<uint32_t>(registers[decode_src1(inst)]) % b;
            pc++;
            DISPATCH();
        }

        OP_CASE(umod_i64) {
            uint64_t b = registers[decode_src2(inst)];
            if (BRASS_UNLIKELY(b == 0)) throw InterpreterException("Modulo by zero");
            registers[decode_dst(inst)] = registers[decode_src1(inst)] % b;
            pc++;
            DISPATCH();
        }

        OP_CASE(neg_i32) { registers[decode_dst(inst)] = static_cast<uint32_t>(-static_cast<int32_t>(registers[decode_src1(inst)])); pc++; DISPATCH(); }
        OP_CASE(neg_i64) { registers[decode_dst(inst)] = static_cast<uint64_t>(-static_cast<int64_t>(registers[decode_src1(inst)])); pc++; DISPATCH(); }

        OP_CASE(add_f32) { registers[decode_dst(inst)] = put_f32(get_f32(registers[decode_src1(inst)]) + get_f32(registers[decode_src2(inst)])); pc++; DISPATCH(); }
        OP_CASE(add_f64) { registers[decode_dst(inst)] = put_f64(get_f64(registers[decode_src1(inst)]) + get_f64(registers[decode_src2(inst)])); pc++; DISPATCH(); }
        OP_CASE(sub_f32) { registers[decode_dst(inst)] = put_f32(get_f32(registers[decode_src1(inst)]) - get_f32(registers[decode_src2(inst)])); pc++; DISPATCH(); }
        OP_CASE(sub_f64) { registers[decode_dst(inst)] = put_f64(get_f64(registers[decode_src1(inst)]) - get_f64(registers[decode_src2(inst)])); pc++; DISPATCH(); }
        OP_CASE(mul_f32) { registers[decode_dst(inst)] = put_f32(get_f32(registers[decode_src1(inst)]) * get_f32(registers[decode_src2(inst)])); pc++; DISPATCH(); }
        OP_CASE(mul_f64) { registers[decode_dst(inst)] = put_f64(get_f64(registers[decode_src1(inst)]) * get_f64(registers[decode_src2(inst)])); pc++; DISPATCH(); }
        OP_CASE(fdiv_f32) { registers[decode_dst(inst)] = put_f32(get_f32(registers[decode_src1(inst)]) / get_f32(registers[decode_src2(inst)])); pc++; DISPATCH(); }
        OP_CASE(fdiv_f64) { registers[decode_dst(inst)] = put_f64(get_f64(registers[decode_src1(inst)]) / get_f64(registers[decode_src2(inst)])); pc++; DISPATCH(); }
        OP_CASE(neg_f32) { registers[decode_dst(inst)] = put_f32(-get_f32(registers[decode_src1(inst)])); pc++; DISPATCH(); }
        OP_CASE(neg_f64) { registers[decode_dst(inst)] = put_f64(-get_f64(registers[decode_src1(inst)])); pc++; DISPATCH(); }
        OP_CASE(fma_f32) { registers[decode_dst(inst)] = put_f32(std::fma(get_f32(registers[decode_src1(inst)]), get_f32(registers[decode_src2(inst)]), get_f32(registers[decode_dst(inst)]))); pc++; DISPATCH(); }
        OP_CASE(fma_f64) { registers[decode_dst(inst)] = put_f64(std::fma(get_f64(registers[decode_src1(inst)]), get_f64(registers[decode_src2(inst)]), get_f64(registers[decode_dst(inst)]))); pc++; DISPATCH(); }
        OP_CASE(sqrt_f32) { registers[decode_dst(inst)] = put_f32(std::sqrt(get_f32(registers[decode_src1(inst)]))); pc++; DISPATCH(); }
        OP_CASE(sqrt_f64) { registers[decode_dst(inst)] = put_f64(std::sqrt(get_f64(registers[decode_src1(inst)]))); pc++; DISPATCH(); }
        OP_CASE(fabs_f32) { registers[decode_dst(inst)] = put_f32(std::fabs(get_f32(registers[decode_src1(inst)]))); pc++; DISPATCH(); }
        OP_CASE(fabs_f64) { registers[decode_dst(inst)] = put_f64(std::fabs(get_f64(registers[decode_src1(inst)]))); pc++; DISPATCH(); }
        OP_CASE(floor_f32) { registers[decode_dst(inst)] = put_f32(std::floor(get_f32(registers[decode_src1(inst)]))); pc++; DISPATCH(); }
        OP_CASE(floor_f64) { registers[decode_dst(inst)] = put_f64(std::floor(get_f64(registers[decode_src1(inst)]))); pc++; DISPATCH(); }
        OP_CASE(ceil_f32) { registers[decode_dst(inst)] = put_f32(std::ceil(get_f32(registers[decode_src1(inst)]))); pc++; DISPATCH(); }
        OP_CASE(ceil_f64) { registers[decode_dst(inst)] = put_f64(std::ceil(get_f64(registers[decode_src1(inst)]))); pc++; DISPATCH(); }
        OP_CASE(round_f32) { registers[decode_dst(inst)] = put_f32(std::round(get_f32(registers[decode_src1(inst)]))); pc++; DISPATCH(); }
        OP_CASE(round_f64) { registers[decode_dst(inst)] = put_f64(std::round(get_f64(registers[decode_src1(inst)]))); pc++; DISPATCH(); }
        OP_CASE(fmin_f32) { registers[decode_dst(inst)] = put_f32(std::fmin(get_f32(registers[decode_src1(inst)]), get_f32(registers[decode_src2(inst)]))); pc++; DISPATCH(); }
        OP_CASE(fmin_f64) { registers[decode_dst(inst)] = put_f64(std::fmin(get_f64(registers[decode_src1(inst)]), get_f64(registers[decode_src2(inst)]))); pc++; DISPATCH(); }
        OP_CASE(fmax_f32) { registers[decode_dst(inst)] = put_f32(std::fmax(get_f32(registers[decode_src1(inst)]), get_f32(registers[decode_src2(inst)]))); pc++; DISPATCH(); }
        OP_CASE(fmax_f64) { registers[decode_dst(inst)] = put_f64(std::fmax(get_f64(registers[decode_src1(inst)]), get_f64(registers[decode_src2(inst)]))); pc++; DISPATCH(); }

        OP_CASE(sadd_overflow_i32) {
            int32_t a = static_cast<int32_t>(registers[decode_src1(inst)]);
            int32_t b = static_cast<int32_t>(registers[decode_src2(inst)]);
            int64_t sum = static_cast<int64_t>(a) + static_cast<int64_t>(b);
            registers[decode_dst(inst)] = (sum < INT32_MIN || sum > INT32_MAX) ? 1 : 0;
            pc++;
            DISPATCH();
        }

        OP_CASE(sadd_overflow_i64) {
            int64_t a = static_cast<int64_t>(registers[decode_src1(inst)]);
            int64_t b = static_cast<int64_t>(registers[decode_src2(inst)]);
            registers[decode_dst(inst)] = ((b > 0 && a > INT64_MAX - b) || (b < 0 && a < INT64_MIN - b)) ? 1 : 0;
            pc++;
            DISPATCH();
        }

        OP_CASE(ssub_overflow_i32) {
            int32_t a = static_cast<int32_t>(registers[decode_src1(inst)]);
            int32_t b = static_cast<int32_t>(registers[decode_src2(inst)]);
            int64_t diff = static_cast<int64_t>(a) - static_cast<int64_t>(b);
            registers[decode_dst(inst)] = (diff < INT32_MIN || diff > INT32_MAX) ? 1 : 0;
            pc++;
            DISPATCH();
        }

        OP_CASE(ssub_overflow_i64) {
            int64_t a = static_cast<int64_t>(registers[decode_src1(inst)]);
            int64_t b = static_cast<int64_t>(registers[decode_src2(inst)]);
            registers[decode_dst(inst)] = ((b < 0 && a > INT64_MAX + b) || (b > 0 && a < INT64_MIN + b)) ? 1 : 0;
            pc++;
            DISPATCH();
        }

        OP_CASE(smul_overflow_i32) {
            int32_t a = static_cast<int32_t>(registers[decode_src1(inst)]);
            int32_t b = static_cast<int32_t>(registers[decode_src2(inst)]);
            int64_t prod = static_cast<int64_t>(a) * static_cast<int64_t>(b);
            registers[decode_dst(inst)] = (prod < INT32_MIN || prod > INT32_MAX) ? 1 : 0;
            pc++;
            DISPATCH();
        }

        OP_CASE(smul_overflow_i64) {
            int64_t a = static_cast<int64_t>(registers[decode_src1(inst)]);
            int64_t b = static_cast<int64_t>(registers[decode_src2(inst)]);
#if defined(__GNUC__) || defined(__clang__)
            int64_t res = 0;
            registers[decode_dst(inst)] = __builtin_mul_overflow(a, b, &res) ? 1 : 0;
#else
            if (a == 0 || b == 0) registers[decode_dst(inst)] = 0;
            else if (a == -1 && b == INT64_MIN) registers[decode_dst(inst)] = 1;
            else if (b == -1 && a == INT64_MIN) registers[decode_dst(inst)] = 1;
            else if (a > 0 && b > 0 && a > INT64_MAX / b) registers[decode_dst(inst)] = 1;
            else if (a > 0 && b < 0 && b < INT64_MIN / a) registers[decode_dst(inst)] = 1;
            else if (a < 0 && b > 0 && a < INT64_MIN / b) registers[decode_dst(inst)] = 1;
            else if (a < 0 && b < 0 && a < INT64_MAX / b) registers[decode_dst(inst)] = 1;
            else registers[decode_dst(inst)] = 0;
#endif
            pc++;
            DISPATCH();
        }

        OP_CASE(uadd_overflow_i32) { registers[decode_dst(inst)] = (static_cast<uint32_t>(registers[decode_src1(inst)]) + static_cast<uint32_t>(registers[decode_src2(inst)]) < static_cast<uint32_t>(registers[decode_src1(inst)])) ? 1 : 0; pc++; DISPATCH(); }
        OP_CASE(uadd_overflow_i64) { registers[decode_dst(inst)] = (registers[decode_src1(inst)] + registers[decode_src2(inst)] < registers[decode_src1(inst)]) ? 1 : 0; pc++; DISPATCH(); }
        OP_CASE(usub_overflow_i32) { registers[decode_dst(inst)] = (static_cast<uint32_t>(registers[decode_src1(inst)]) < static_cast<uint32_t>(registers[decode_src2(inst)])) ? 1 : 0; pc++; DISPATCH(); }
        OP_CASE(usub_overflow_i64) { registers[decode_dst(inst)] = (registers[decode_src1(inst)] < registers[decode_src2(inst)]) ? 1 : 0; pc++; DISPATCH(); }
        OP_CASE(umul_overflow_i32) { registers[decode_dst(inst)] = (static_cast<uint64_t>(static_cast<uint32_t>(registers[decode_src1(inst)])) * static_cast<uint64_t>(static_cast<uint32_t>(registers[decode_src2(inst)])) > UINT32_MAX) ? 1 : 0; pc++; DISPATCH(); }

        OP_CASE(umul_overflow_i64) {
            uint64_t a = registers[decode_src1(inst)];
            uint64_t b = registers[decode_src2(inst)];
#if defined(__GNUC__) || defined(__clang__)
            uint64_t res = 0;
            registers[decode_dst(inst)] = __builtin_mul_overflow(a, b, &res) ? 1 : 0;
#else
            if (a == 0 || b == 0) registers[decode_dst(inst)] = 0;
            else registers[decode_dst(inst)] = ((a * b) / a != b) ? 1 : 0;
#endif
            pc++;
            DISPATCH();
        }

        OP_CASE(and_i32) { registers[decode_dst(inst)] = static_cast<uint32_t>(registers[decode_src1(inst)] & registers[decode_src2(inst)]); pc++; DISPATCH(); }
        OP_CASE(and_i64) { registers[decode_dst(inst)] = registers[decode_src1(inst)] & registers[decode_src2(inst)]; pc++; DISPATCH(); }
        OP_CASE(or_i32) { registers[decode_dst(inst)] = static_cast<uint32_t>(registers[decode_src1(inst)] | registers[decode_src2(inst)]); pc++; DISPATCH(); }
        OP_CASE(or_i64) { registers[decode_dst(inst)] = registers[decode_src1(inst)] | registers[decode_src2(inst)]; pc++; DISPATCH(); }
        OP_CASE(xor_i32) { registers[decode_dst(inst)] = static_cast<uint32_t>(registers[decode_src1(inst)] ^ registers[decode_src2(inst)]); pc++; DISPATCH(); }
        OP_CASE(xor_i64) { registers[decode_dst(inst)] = registers[decode_src1(inst)] ^ registers[decode_src2(inst)]; pc++; DISPATCH(); }
        OP_CASE(shl_i32) { registers[decode_dst(inst)] = static_cast<uint32_t>(static_cast<uint32_t>(registers[decode_src1(inst)]) << (registers[decode_src2(inst)] & 31)); pc++; DISPATCH(); }
        OP_CASE(shl_i64) { registers[decode_dst(inst)] = registers[decode_src1(inst)] << (registers[decode_src2(inst)] & 63); pc++; DISPATCH(); }
        OP_CASE(lshr_i32) { registers[decode_dst(inst)] = static_cast<uint32_t>(static_cast<uint32_t>(registers[decode_src1(inst)]) >> (registers[decode_src2(inst)] & 31)); pc++; DISPATCH(); }
        OP_CASE(lshr_i64) { registers[decode_dst(inst)] = registers[decode_src1(inst)] >> (registers[decode_src2(inst)] & 63); pc++; DISPATCH(); }
        OP_CASE(ashr_i32) { registers[decode_dst(inst)] = static_cast<uint32_t>(static_cast<int32_t>(registers[decode_src1(inst)]) >> (registers[decode_src2(inst)] & 31)); pc++; DISPATCH(); }
        OP_CASE(ashr_i64) { registers[decode_dst(inst)] = static_cast<uint64_t>(static_cast<int64_t>(registers[decode_src1(inst)]) >> (registers[decode_src2(inst)] & 63)); pc++; DISPATCH(); }
        OP_CASE(not_i32) { registers[decode_dst(inst)] = static_cast<uint32_t>(~static_cast<uint32_t>(registers[decode_src1(inst)])); pc++; DISPATCH(); }
        OP_CASE(not_i64) { registers[decode_dst(inst)] = ~registers[decode_src1(inst)]; pc++; DISPATCH(); }

        OP_CASE(clz_i32) {
            uint32_t val = static_cast<uint32_t>(registers[decode_src1(inst)]);
            registers[decode_dst(inst)] = (val == 0) ? 32 : static_cast<uint32_t>(std::countl_zero(val));
            pc++;
            DISPATCH();
        }

        OP_CASE(clz_i64) {
            uint64_t val = registers[decode_src1(inst)];
            registers[decode_dst(inst)] = (val == 0) ? 64 : static_cast<uint64_t>(std::countl_zero(val));
            pc++;
            DISPATCH();
        }

        OP_CASE(ctz_i32) {
            uint32_t val = static_cast<uint32_t>(registers[decode_src1(inst)]);
            registers[decode_dst(inst)] = (val == 0) ? 32 : static_cast<uint32_t>(std::countr_zero(val));
            pc++;
            DISPATCH();
        }

        OP_CASE(ctz_i64) {
            uint64_t val = registers[decode_src1(inst)];
            registers[decode_dst(inst)] = (val == 0) ? 64 : static_cast<uint64_t>(std::countr_zero(val));
            pc++;
            DISPATCH();
        }

        OP_CASE(popcnt_i32) { registers[decode_dst(inst)] = static_cast<uint32_t>(std::popcount(static_cast<uint32_t>(registers[decode_src1(inst)]))); pc++; DISPATCH(); }
        OP_CASE(popcnt_i64) { registers[decode_dst(inst)] = static_cast<uint64_t>(std::popcount(registers[decode_src1(inst)])); pc++; DISPATCH(); }

        OP_CASE(eq_i32) { registers[decode_dst(inst)] = (static_cast<uint32_t>(registers[decode_src1(inst)]) == static_cast<uint32_t>(registers[decode_src2(inst)])) ? 1 : 0; pc++; DISPATCH(); }
        OP_CASE(eq_i64) { registers[decode_dst(inst)] = (registers[decode_src1(inst)] == registers[decode_src2(inst)]) ? 1 : 0; pc++; DISPATCH(); }
        OP_CASE(eq_f32) { registers[decode_dst(inst)] = (get_f32(registers[decode_src1(inst)]) == get_f32(registers[decode_src2(inst)])) ? 1 : 0; pc++; DISPATCH(); }
        OP_CASE(eq_f64) { registers[decode_dst(inst)] = (get_f64(registers[decode_src1(inst)]) == get_f64(registers[decode_src2(inst)])) ? 1 : 0; pc++; DISPATCH(); }
        OP_CASE(ne_i32) { registers[decode_dst(inst)] = (static_cast<uint32_t>(registers[decode_src1(inst)]) != static_cast<uint32_t>(registers[decode_src2(inst)])) ? 1 : 0; pc++; DISPATCH(); }
        OP_CASE(ne_i64) { registers[decode_dst(inst)] = (registers[decode_src1(inst)] != registers[decode_src2(inst)]) ? 1 : 0; pc++; DISPATCH(); }
        OP_CASE(ne_f32) { registers[decode_dst(inst)] = (get_f32(registers[decode_src1(inst)]) != get_f32(registers[decode_src2(inst)])) ? 1 : 0; pc++; DISPATCH(); }
        OP_CASE(ne_f64) { registers[decode_dst(inst)] = (get_f64(registers[decode_src1(inst)]) != get_f64(registers[decode_src2(inst)])) ? 1 : 0; pc++; DISPATCH(); }
        OP_CASE(slt_i32) { registers[decode_dst(inst)] = (static_cast<int32_t>(registers[decode_src1(inst)]) < static_cast<int32_t>(registers[decode_src2(inst)])) ? 1 : 0; pc++; DISPATCH(); }
        OP_CASE(slt_i64) { registers[decode_dst(inst)] = (static_cast<int64_t>(registers[decode_src1(inst)]) < static_cast<int64_t>(registers[decode_src2(inst)])) ? 1 : 0; pc++; DISPATCH(); }
        OP_CASE(ult_i32) { registers[decode_dst(inst)] = (static_cast<uint32_t>(registers[decode_src1(inst)]) < static_cast<uint32_t>(registers[decode_src2(inst)])) ? 1 : 0; pc++; DISPATCH(); }
        OP_CASE(ult_i64) { registers[decode_dst(inst)] = (registers[decode_src1(inst)] < registers[decode_src2(inst)]) ? 1 : 0; pc++; DISPATCH(); }
        OP_CASE(lt_f32) { registers[decode_dst(inst)] = (get_f32(registers[decode_src1(inst)]) < get_f32(registers[decode_src2(inst)])) ? 1 : 0; pc++; DISPATCH(); }
        OP_CASE(lt_f64) { registers[decode_dst(inst)] = (get_f64(registers[decode_src1(inst)]) < get_f64(registers[decode_src2(inst)])) ? 1 : 0; pc++; DISPATCH(); }
        OP_CASE(sle_i32) { registers[decode_dst(inst)] = (static_cast<int32_t>(registers[decode_src1(inst)]) <= static_cast<int32_t>(registers[decode_src2(inst)])) ? 1 : 0; pc++; DISPATCH(); }
        OP_CASE(sle_i64) { registers[decode_dst(inst)] = (static_cast<int64_t>(registers[decode_src1(inst)]) <= static_cast<int64_t>(registers[decode_src2(inst)])) ? 1 : 0; pc++; DISPATCH(); }
        OP_CASE(ule_i32) { registers[decode_dst(inst)] = (static_cast<uint32_t>(registers[decode_src1(inst)]) <= static_cast<uint32_t>(registers[decode_src2(inst)])) ? 1 : 0; pc++; DISPATCH(); }
        OP_CASE(ule_i64) { registers[decode_dst(inst)] = (registers[decode_src1(inst)] <= registers[decode_src2(inst)]) ? 1 : 0; pc++; DISPATCH(); }
        OP_CASE(le_f32) { registers[decode_dst(inst)] = (get_f32(registers[decode_src1(inst)]) <= get_f32(registers[decode_src2(inst)])) ? 1 : 0; pc++; DISPATCH(); }
        OP_CASE(le_f64) { registers[decode_dst(inst)] = (get_f64(registers[decode_src1(inst)]) <= get_f64(registers[decode_src2(inst)])) ? 1 : 0; pc++; DISPATCH(); }
        OP_CASE(sgt_i32) { registers[decode_dst(inst)] = (static_cast<int32_t>(registers[decode_src1(inst)]) > static_cast<int32_t>(registers[decode_src2(inst)])) ? 1 : 0; pc++; DISPATCH(); }
        OP_CASE(sgt_i64) { registers[decode_dst(inst)] = (static_cast<int64_t>(registers[decode_src1(inst)]) > static_cast<int64_t>(registers[decode_src2(inst)])) ? 1 : 0; pc++; DISPATCH(); }
        OP_CASE(ugt_i32) { registers[decode_dst(inst)] = (static_cast<uint32_t>(registers[decode_src1(inst)]) > static_cast<uint32_t>(registers[decode_src2(inst)])) ? 1 : 0; pc++; DISPATCH(); }
        OP_CASE(ugt_i64) { registers[decode_dst(inst)] = (registers[decode_src1(inst)] > registers[decode_src2(inst)]) ? 1 : 0; pc++; DISPATCH(); }
        OP_CASE(gt_f32) { registers[decode_dst(inst)] = (get_f32(registers[decode_src1(inst)]) > get_f32(registers[decode_src2(inst)])) ? 1 : 0; pc++; DISPATCH(); }
        OP_CASE(gt_f64) { registers[decode_dst(inst)] = (get_f64(registers[decode_src1(inst)]) > get_f64(registers[decode_src2(inst)])) ? 1 : 0; pc++; DISPATCH(); }
        OP_CASE(sge_i32) { registers[decode_dst(inst)] = (static_cast<int32_t>(registers[decode_src1(inst)]) >= static_cast<int32_t>(registers[decode_src2(inst)])) ? 1 : 0; pc++; DISPATCH(); }
        OP_CASE(sge_i64) { registers[decode_dst(inst)] = (static_cast<int64_t>(registers[decode_src1(inst)]) >= static_cast<int64_t>(registers[decode_src2(inst)])) ? 1 : 0; pc++; DISPATCH(); }
        OP_CASE(uge_i32) { registers[decode_dst(inst)] = (static_cast<uint32_t>(registers[decode_src1(inst)]) >= static_cast<uint32_t>(registers[decode_src2(inst)])) ? 1 : 0; pc++; DISPATCH(); }
        OP_CASE(uge_i64) { registers[decode_dst(inst)] = (registers[decode_src1(inst)] >= registers[decode_src2(inst)]) ? 1 : 0; pc++; DISPATCH(); }
        OP_CASE(ge_f32) { registers[decode_dst(inst)] = (get_f32(registers[decode_src1(inst)]) >= get_f32(registers[decode_src2(inst)])) ? 1 : 0; pc++; DISPATCH(); }
        OP_CASE(ge_f64) { registers[decode_dst(inst)] = (get_f64(registers[decode_src1(inst)]) >= get_f64(registers[decode_src2(inst)])) ? 1 : 0; pc++; DISPATCH(); }

        OP_CASE(select) {
            if (registers[decode_src1(inst)] != 0) {
                registers[decode_dst(inst)] = registers[decode_src2(inst)];
            }
            pc++;
            DISPATCH();
        }

        OP_CASE(load8) {
            uintptr_t addr = static_cast<uintptr_t>(registers[decode_src1(inst)]) + decode_src2(inst);
            registers[decode_dst(inst)] = *reinterpret_cast<const uint8_t*>(addr);
            pc++;
            DISPATCH();
        }

        OP_CASE(load16) {
            uintptr_t addr = static_cast<uintptr_t>(registers[decode_src1(inst)]) + decode_src2(inst);
            uint16_t val = 0;
            std::memcpy(&val, reinterpret_cast<const void*>(addr), 2);
            registers[decode_dst(inst)] = static_cast<uint64_t>(val);
            pc++;
            DISPATCH();
        }

        OP_CASE(load32) {
            uintptr_t addr = static_cast<uintptr_t>(registers[decode_src1(inst)]) + decode_src2(inst);
            uint32_t val = 0;
            std::memcpy(&val, reinterpret_cast<const void*>(addr), 4);
            registers[decode_dst(inst)] = static_cast<uint64_t>(val);
            pc++;
            DISPATCH();
        }

        OP_CASE(load64) {
            uintptr_t addr = static_cast<uintptr_t>(registers[decode_src1(inst)]) + decode_src2(inst);
            uint64_t val = 0;
            std::memcpy(&val, reinterpret_cast<const void*>(addr), 8);
            registers[decode_dst(inst)] = val;
            pc++;
            DISPATCH();
        }

        OP_CASE(store8) {
            uintptr_t addr = static_cast<uintptr_t>(registers[decode_src1(inst)]) + decode_src2(inst);
            *reinterpret_cast<uint8_t*>(addr) = static_cast<uint8_t>(registers[decode_dst(inst)]);
            pc++;
            DISPATCH();
        }

        OP_CASE(store16) {
            uintptr_t addr = static_cast<uintptr_t>(registers[decode_src1(inst)]) + decode_src2(inst);
            uint16_t val = static_cast<uint16_t>(registers[decode_dst(inst)]);
            std::memcpy(reinterpret_cast<void*>(addr), &val, 2);
            pc++;
            DISPATCH();
        }

        OP_CASE(store32) {
            uintptr_t addr = static_cast<uintptr_t>(registers[decode_src1(inst)]) + decode_src2(inst);
            uint32_t val = static_cast<uint32_t>(registers[decode_dst(inst)]);
            std::memcpy(reinterpret_cast<void*>(addr), &val, 4);
            pc++;
            DISPATCH();
        }

        OP_CASE(store64) {
            uintptr_t addr = static_cast<uintptr_t>(registers[decode_src1(inst)]) + decode_src2(inst);
            uint64_t val = registers[decode_dst(inst)];
            std::memcpy(reinterpret_cast<void*>(addr), &val, 8);
            pc++;
            DISPATCH();
        }

        OP_CASE(alloca_) {
            uint16_t sz = decode_u16(inst);
            void* mem = frame.allocate_alloca(sz, 16);
            registers[decode_dst(inst)] = reinterpret_cast<uintptr_t>(mem);
            pc++;
            DISPATCH();
        }

        OP_CASE(load_indexed) {
            uintptr_t addr = static_cast<uintptr_t>(registers[decode_src1(inst)]) + registers[decode_src2(inst)];
            uint64_t val = 0;
            std::memcpy(&val, reinterpret_cast<const void*>(addr), 8);
            registers[decode_dst(inst)] = val;
            pc++;
            DISPATCH();
        }

        OP_CASE(store_indexed) {
            uintptr_t addr = static_cast<uintptr_t>(registers[decode_src1(inst)]) + registers[decode_src2(inst)];
            uint64_t val = registers[decode_dst(inst)];
            std::memcpy(reinterpret_cast<void*>(addr), &val, 8);
            pc++;
            DISPATCH();
        }

        OP_CASE(jump) {
            int16_t off = decode_s16(inst);
            if (off < 0) {
                uint32_t target_pc = static_cast<uint32_t>(pc - code_base + off);
                RuntimeValue osr_res;
                if (handle_osr_backedge(frame, target_pc, osr_res)) {
                    return osr_res;
                }
            }
            pc += off;
            DISPATCH();
        }

        OP_CASE(jump_if) {
            if (registers[decode_dst(inst)] != 0) {
                int16_t off = decode_s16(inst);
                if (off < 0) {
                    uint32_t target_pc = static_cast<uint32_t>(pc - code_base + off);
                    RuntimeValue osr_res;
                    if (handle_osr_backedge(frame, target_pc, osr_res)) {
                        return osr_res;
                    }
                }
                pc += off;
            } else {
                pc++;
            }
            DISPATCH();
        }

        OP_CASE(jump_if_not) {
            if (registers[decode_dst(inst)] == 0) {
                int16_t off = decode_s16(inst);
                if (off < 0) {
                    uint32_t target_pc = static_cast<uint32_t>(pc - code_base + off);
                    RuntimeValue osr_res;
                    if (handle_osr_backedge(frame, target_pc, osr_res)) {
                        return osr_res;
                    }
                }
                pc += off;
            } else {
                pc++;
            }
            DISPATCH();
        }

        OP_CASE(ret) {
            uint8_t ret_reg = decode_dst(inst);
            return marshal_return_value(fn, registers[ret_reg], frame.vector_regs, ret_reg);
        }

        OP_CASE(ret_void) {
            return RuntimeValue::from_void();
        }

        OP_CASE(switch_) {
            uint8_t cond_reg = decode_dst(inst);
            uint16_t table_idx = decode_u16(inst);
            const auto& table = fn.switch_tables[table_idx];
            int64_t val = static_cast<int64_t>(registers[cond_reg]);
            uint32_t target_pc = static_cast<uint32_t>(table.default_offset);
            for (const auto& c : table.cases) {
                if (c.first == val) {
                    target_pc = static_cast<uint32_t>(c.second);
                    break;
                }
            }
            pc = code_base + target_pc;
            DISPATCH();
        }

        OP_CASE(call)
        OP_CASE(patchable_call) {
            uint16_t cs_idx = decode_u16(inst);
            const auto& cs = fn.call_sites[cs_idx];
            execute_call(frame, cs, decode_op(inst));
            pc++;
            DISPATCH();
        }

        OP_CASE(call_indirect) {
            uint16_t cs_idx = decode_u16(inst);
            const auto& cs = fn.call_sites[cs_idx];
            execute_call_indirect(frame, cs);
            pc++;
            DISPATCH();
        }

        OP_CASE(func_addr) {
            uint8_t dst = decode_dst(inst);
            uint16_t s_idx = decode_u16(inst);
            const std::string& sym = fn.string_pool[s_idx];
            const Function* target_fn = module_ ? module_->get_function(sym) : nullptr;
            uintptr_t fn_ptr = reinterpret_cast<uintptr_t>(target_fn);
            if (target_fn) {
                register_function_pointer(fn_ptr, target_fn);
            } else {
                const BytecodeFunction* bfn = bytecode_module_ ? bytecode_module_->get_function(sym) : nullptr;
                if (bfn) {
                    fn_ptr = reinterpret_cast<uintptr_t>(bfn);
                    register_function_pointer(fn_ptr, bfn);
                } else {
                    void* ext_sym = find_external_symbol(sym);
                    if (ext_sym) {
                        fn_ptr = reinterpret_cast<uintptr_t>(ext_sym);
                    }
                }
            }
            registers[dst] = fn_ptr;
            pc++;
            DISPATCH();
        }

        OP_CASE(safepoint) {
            handle_safepoint(frame);
            pc++;
            DISPATCH();
        }

        OP_CASE(write_barrier) {
            handle_write_barrier(frame, decode_src1(inst), decode_src2(inst));
            pc++;
            DISPATCH();
        }

        OP_CASE(guard) {
            uint8_t cond_reg = decode_dst(inst);
            if (BRASS_UNLIKELY(registers[cond_reg] == 0)) {
                uint16_t g_idx = decode_u16(inst);
                const auto& g = fn.guards[g_idx];
                last_deopt_.deoptimized = true;
                last_deopt_.exit_stub = g.exit_stub;
                last_deopt_.resume_id = g.resume_id;
                last_deopt_.state_map.clear();
                last_deopt_.state_map.reserve(g.state_regs.size());

                runtime::DeoptFrame df;
                df.resume_id = g.resume_id;
                df.exit_symbol = g.exit_stub;
                df.reason = runtime::DeoptReason::Generic;

                for (uint8_t sreg : g.state_regs) {
                    Type t = (sreg < fn.register_types.size()) ? fn.register_types[sreg] : Type::i64();
                    RuntimeValue rv = RuntimeValue::from_bits(t, registers[sreg]);
                    last_deopt_.state_map.push_back(rv);
                    df.push_value(registers[sreg], t.is_gcref() ? runtime::DeoptValueKind::GcRef : runtime::DeoptValueKind::Int64);
                }
                runtime::set_thread_deopt_frame(&df);

                if (deopt_handler_) {
                    return deopt_handler_(*this, last_deopt_);
                }

                if (module_ && !g.exit_stub.empty()) {
                    const Function* stub_fn = module_->get_function(g.exit_stub);
                    if (stub_fn) {
                        return run(*stub_fn, last_deopt_.state_map);
                    }
                }

                for (const auto& rp : fn.resume_points) {
                    if (rp.resume_id == g.resume_id) {
                        if (!rp.param_regs.empty()) {
                            for (size_t i = 0; i < last_deopt_.state_map.size() && i < rp.param_regs.size(); ++i) {
                                registers[rp.param_regs[i]] = last_deopt_.state_map[i].raw_bits();
                            }
                        }
                        pc = code_base + rp.target_pc;
                        DISPATCH();
                    }
                }

                throw DeoptException(last_deopt_);
            }
            pc++;
            DISPATCH();
        }

        OP_CASE(resume_point) {
            pc++;
            DISPATCH();
        }

        OP_CASE(osr_entry) {
            pc++;
            DISPATCH();
        }

        OP_CASE(pinned_tls_write) {
            uint8_t src = decode_src1(inst);
            tls_block_ = registers[src];
            pc++;
            DISPATCH();
        }

        OP_CASE(pinned_tls_read) {
            uint8_t dst = decode_dst(inst);
            if (tls_block_ == 0) {
                void* sym = find_external_symbol("bronze_tls_enter");
                if (sym) {
                    auto fn_ptr = reinterpret_cast<void*(*)()>(sym);
                    tls_block_ = reinterpret_cast<uint64_t>(fn_ptr());
                } else {
                    sym = find_external_symbol("bronze_tls_block_addr");
                    if (sym) {
                        auto fn_ptr = reinterpret_cast<void*(*)()>(sym);
                        tls_block_ = reinterpret_cast<uint64_t>(fn_ptr());
                    }
                }
            }
            registers[dst] = tls_block_;
            pc++;
            DISPATCH();
        }

        OP_CASE(read_sp) {
            uint8_t dst = decode_dst(inst);
            char marker = 0;
            registers[dst] = reinterpret_cast<uint64_t>(&marker);
            pc++;
            DISPATCH();
        }

        OP_CASE(throw_) {
            uint8_t reg = decode_dst(inst);
            handle_throw(frame, reg, pc, code_base);
            DISPATCH();
        }

        OP_CASE(invoke) {
            uint16_t cs_idx = decode_u16(inst);
            const auto& cs = fn.call_sites[cs_idx];
            handle_invoke(frame, cs, pc, code_base);
            DISPATCH();
        }

        OP_CASE(landing_pad) {
            registers[decode_dst(inst)] = current_exception_.raw_bits();
            pc++;
            DISPATCH();
        }

        OP_CASE(resume) {
            uint8_t reg = decode_dst(inst);
            handle_resume(frame, reg);
            pc++;
            DISPATCH();
        }

        OP_CASE(coro_create) {
            uint8_t dst = decode_dst(inst);
            uint16_t cs_idx = decode_u16(inst);
            const auto& cs = fn.call_sites[cs_idx];
            std::vector<RuntimeValue> args;
            args.reserve(cs.arg_regs.size());
            for (uint8_t ar : cs.arg_regs) {
                args.push_back(RuntimeValue::from_bits(Type::i64(), registers[ar]));
            }
            uintptr_t handle = coro_create(cs.callee, args);
            registers[dst] = handle;
            pc++;
            DISPATCH();
        }

        OP_CASE(coro_suspend) {
            uint8_t dst_reg = decode_dst(inst);
            uint8_t yield_reg = decode_src1(inst);
            uint32_t resume_id = decode_src2(inst);
            frame.pc = static_cast<uint32_t>(pc - code_base);
            coro_suspend(frame, dst_reg, yield_reg, resume_id);
            pc++;
            DISPATCH();
        }

        OP_CASE(coro_resume) {
            uint8_t dst = decode_dst(inst);
            uint8_t coro_reg = decode_src1(inst);
            uint8_t input_reg = decode_src2(inst);
            uintptr_t handle = registers[coro_reg];
            uint64_t input_val = (input_reg != 255 && input_reg < frame.num_registers) ? registers[input_reg] : 0;
            uint64_t res = coro_resume(handle, input_val);
            registers[dst] = res;
            pc++;
            DISPATCH();
        }

        OP_CASE(coro_destroy) {
            uint8_t coro_reg = decode_src1(inst);
            uintptr_t handle = registers[coro_reg];
            coro_destroy(handle);
            pc++;
            DISPATCH();
        }

        OP_CASE(vadd)
        OP_CASE(vsub)
        OP_CASE(vmul)
        OP_CASE(vdiv)
        OP_CASE(vfma)
        OP_CASE(vneg)
        OP_CASE(vmin)
        OP_CASE(vmax)
        OP_CASE(vsqrt)
        OP_CASE(vand)
        OP_CASE(vor)
        OP_CASE(vxor)
        OP_CASE(vnot)
        OP_CASE(vload)
        OP_CASE(vstore)
        OP_CASE(vbroadcast)
        OP_CASE(vextract_lane)
        OP_CASE(vinsert_lane)
        OP_CASE(vshuffle)
        OP_CASE(vzero) {
            execute_vector_op(frame, inst);
            pc++;
            DISPATCH();
        }

#if !BRASS_DIRECT_THREADED
        default:
            throw InterpreterException("Unknown opcode in fast interpreter: " + std::to_string(static_cast<int>(decode_op(inst))));
#endif
    }

    return RuntimeValue::from_void();
}

} // namespace brass

#if defined(__GNUC__) || defined(__clang__)
#pragma GCC diagnostic pop
#endif
