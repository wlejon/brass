#include "fast_interpreter_impl.hpp"
#include <brass/runtime/parallel_runtime.hpp>
#include <brass/runtime/osr_coordinator.hpp>
#include <cmath>

namespace brass {

namespace {
thread_local void* s_fast_interp_red_target = nullptr;
thread_local runtime::ReductionKind s_fast_interp_red_kind = runtime::ReductionKind::None;
} // namespace

void FastInterpreter::register_builtin_host_functions() {
    register_external_function("brass_parallel_alloc_context", [](FastInterpreter&, const std::vector<RuntimeValue>& args) -> RuntimeValue {
        uint64_t bytes = args.empty() ? 64ULL : args[0].as_u64();
        void* ptr = brass_parallel_alloc_context(bytes);
        return RuntimeValue::from_ptr(reinterpret_cast<uintptr_t>(ptr));
    });

    register_external_function("brass_parallel_free_context", [](FastInterpreter&, const std::vector<RuntimeValue>& args) -> RuntimeValue {
        if (!args.empty()) {
            brass_parallel_free_context(reinterpret_cast<void*>(args[0].as_ptr()));
        }
        return RuntimeValue::from_void();
    });

    register_external_function("brass_parallel_reduce_i64", [](FastInterpreter&, const std::vector<RuntimeValue>& args) -> RuntimeValue {
        if (!args.empty()) {
            int64_t val = args[0].as_i64();
            if (s_fast_interp_red_target) {
                int64_t* ptr = reinterpret_cast<int64_t*>(s_fast_interp_red_target);
                *ptr = runtime::combine_reduction_i64(s_fast_interp_red_kind, *ptr, val);
            } else {
                brass_parallel_reduce_i64(val);
            }
        }
        return RuntimeValue::from_void();
    });

    register_external_function("brass_parallel_reduce_f64", [](FastInterpreter&, const std::vector<RuntimeValue>& args) -> RuntimeValue {
        if (!args.empty()) {
            double val = args[0].as_f64();
            if (s_fast_interp_red_target) {
                double* ptr = reinterpret_cast<double*>(s_fast_interp_red_target);
                *ptr = runtime::combine_reduction_f64(s_fast_interp_red_kind, *ptr, val);
            } else {
                brass_parallel_reduce_f64(val);
            }
        }
        return RuntimeValue::from_void();
    });

    register_external_function("brass_parallel_for", [](FastInterpreter& interp, const std::vector<RuntimeValue>& args) -> RuntimeValue {
        if (args.size() < 6) return RuntimeValue::from_void();
        uint64_t trip_count = args[0].as_u64();
        uint64_t grain_size = args[1].as_u64();
        uintptr_t kernel_ptr = args[2].as_ptr();
        uintptr_t ctx_ptr = args[3].as_ptr();
        auto red_kind = static_cast<runtime::ReductionKind>(args[4].as_u64());
        void* red_target = reinterpret_cast<void*>(args[5].as_ptr());

        const Function* target_fn = interp.find_function_by_pointer(kernel_ptr);
        if (target_fn) {
            void* prev_target = s_fast_interp_red_target;
            runtime::ReductionKind prev_kind = s_fast_interp_red_kind;
            s_fast_interp_red_target = red_target;
            s_fast_interp_red_kind = red_kind;

            std::vector<RuntimeValue> kargs = {
                RuntimeValue::from_u64(0),
                RuntimeValue::from_u64(trip_count),
                RuntimeValue::from_ptr(ctx_ptr)
            };
            interp.run(*target_fn, kargs);

            s_fast_interp_red_target = prev_target;
            s_fast_interp_red_kind = prev_kind;
        } else {
            auto kernel_fn = reinterpret_cast<runtime::ParallelKernelFn>(kernel_ptr);
            brass_parallel_for(trip_count, grain_size, kernel_fn, reinterpret_cast<void*>(ctx_ptr), red_kind, red_target);
        }
        return RuntimeValue::from_void();
    });

    register_external_function("brass_gc_alloc", [](FastInterpreter& interp, const std::vector<RuntimeValue>& args) -> RuntimeValue {
        if (args.empty()) {
            throw InterpreterException("brass_gc_alloc requires at least 1 argument (size)");
        }
        size_t size = static_cast<size_t>(args[0].is_i32() ? args[0].as_u32() : args[0].as_u64());
        uint64_t pointer_mask = (args.size() > 1) ? args[1].as_u64() : 0ULL;
        uint32_t type_tag = (args.size() > 2) ? args[2].as_u32() : 0U;
        uintptr_t addr = interp.allocate_gc(size, pointer_mask, type_tag);
        return RuntimeValue::from_gcref(addr);
    });

    register_external_function("brass_gc_collect", [](FastInterpreter& interp, const std::vector<RuntimeValue>&) -> RuntimeValue {
        interp.gc().collect();
        return RuntimeValue::from_void();
    });

    register_external_function("sqrt", [](FastInterpreter&, const std::vector<RuntimeValue>& args) -> RuntimeValue {
        if (args.empty()) return RuntimeValue::from_f64(0.0);
        return RuntimeValue::from_f64(std::sqrt(args[0].as_f64()));
    });

    register_external_function("fabs", [](FastInterpreter&, const std::vector<RuntimeValue>& args) -> RuntimeValue {
        if (args.empty()) return RuntimeValue::from_f64(0.0);
        return RuntimeValue::from_f64(std::fabs(args[0].as_f64()));
    });

    register_external_function("floor", [](FastInterpreter&, const std::vector<RuntimeValue>& args) -> RuntimeValue {
        if (args.empty()) return RuntimeValue::from_f64(0.0);
        return RuntimeValue::from_f64(std::floor(args[0].as_f64()));
    });

    register_external_function("ceil", [](FastInterpreter&, const std::vector<RuntimeValue>& args) -> RuntimeValue {
        if (args.empty()) return RuntimeValue::from_f64(0.0);
        return RuntimeValue::from_f64(std::ceil(args[0].as_f64()));
    });
}

void FastInterpreter::handle_write_barrier(FastFrame& frame, uint32_t obj_reg, uint32_t val_reg) {
    if (gen_gc_) {
        gen_gc_->write_barrier(frame.registers[obj_reg], frame.registers[val_reg]);
    }
}

void FastInterpreter::handle_safepoint(FastFrame& /*frame*/) {
    if (gc_.stress_mode()) {
        gc_.collect();
    }
}

// Vector instructions run through the reference interpreter's value
// operations, typed by the registers' types, so every vector type (f32x4
// through i64x4) has the oracle's lane semantics.
void FastInterpreter::execute_vector_op(FastFrame& frame, BytecodeWord inst, const BytecodeWord* pc) {
    const BytecodeOp op = decode_op(inst);
    const uint32_t a = decode_a(inst);
    const uint32_t b = decode_b(inst);
    const uint32_t c = decode_c(inst);
    const auto& types = frame.bfn->register_types;
    auto type_of = [&](uint32_t r) {
        if (r >= types.size()) {
            throw InterpreterException("vector operand register " + std::to_string(r) + " out of range in " + frame.bfn->name);
        }
        return types[r];
    };
    auto val = [&](uint32_t r) { return fast_reg_value(frame, r); };

    RuntimeValue out;
    switch (op) {
        case BytecodeOp::vzero: out = val_vzero(type_of(a)); break;
        case BytecodeOp::vbroadcast: out = val_vbroadcast(type_of(a), val(b)); break;
        case BytecodeOp::vload:
            out = gc_.read_memory(static_cast<uintptr_t>(frame.registers[b]), decode_imm24(inst), type_of(a));
            break;
        case BytecodeOp::vstore:
            gc_.write_memory(static_cast<uintptr_t>(frame.registers[b]), decode_imm24(inst), type_of(a), val(a));
            return;
        case BytecodeOp::vadd: out = val_vadd(val(b), val(c)); break;
        case BytecodeOp::vsub: out = val_vsub(val(b), val(c)); break;
        case BytecodeOp::vmul: out = val_vmul(val(b), val(c)); break;
        case BytecodeOp::vdiv: out = val_vdiv(val(b), val(c)); break;
        case BytecodeOp::vmin: out = val_vmin(val(b), val(c)); break;
        case BytecodeOp::vmax: out = val_vmax(val(b), val(c)); break;
        case BytecodeOp::vand: out = val_vand(val(b), val(c)); break;
        case BytecodeOp::vor: out = val_vor(val(b), val(c)); break;
        case BytecodeOp::vxor: out = val_vxor(val(b), val(c)); break;
        // The destination already holds the addend.
        case BytecodeOp::vfma: out = val_vfma(val(b), val(c), val(a)); break;
        case BytecodeOp::vneg: out = val_vneg(val(b)); break;
        case BytecodeOp::vsqrt: out = val_vsqrt(val(b)); break;
        case BytecodeOp::vnot: out = val_vnot(val(b)); break;
        case BytecodeOp::vextract_lane: out = val_vextract_lane(val(b), decode_d(inst)); break;
        case BytecodeOp::vinsert_lane: out = val_vinsert_lane(val(b), val(c), decode_d(inst)); break;
        // The shuffle mask is the next code word.
        case BytecodeOp::vshuffle: out = val_vshuffle(val(b), val(c), static_cast<uint32_t>(pc[1])); break;
        default:
            throw InterpreterException("not a vector opcode: " + std::string(bytecode_op_name(op)));
    }
    fast_set_reg(frame, a, out);
}

bool FastInterpreter::handle_osr_backedge(FastFrame& frame, uint32_t target_pc, RuntimeValue& out_res) {
    runtime::TieringFeedback& fb = frame.info->tiering(dispatch_table_);
    auto& coordinator = runtime::OsrCoordinator::instance();
    const BytecodeFunction* bfn = frame.bfn;
    // Below the threshold only the count matters; try_osr_migration counts
    // the backedge itself once it is called.
    if (!frame.mir_fn || !bfn || fb.backedge_count() + 1 < coordinator.threshold()) {
        fb.record_backedge();
        return false;
    }
    auto it = bfn->pc_block_map.find(target_pc);
    if (it == bfn->pc_block_map.end() || !it->second) {
        fb.record_backedge();
        return false;
    }
    BasicBlock* loop_header = const_cast<BasicBlock*>(it->second);
    return coordinator.try_osr_migration(*this, *frame.mir_fn, loop_header, frame, out_res);
}

} // namespace brass
