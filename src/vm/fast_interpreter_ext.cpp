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

void FastInterpreter::handle_write_barrier(FastFrame& frame, uint8_t obj_reg, uint8_t val_reg) {
    if (gen_gc_) {
        gen_gc_->write_barrier(frame.registers[obj_reg], frame.registers[val_reg]);
    }
}

void FastInterpreter::handle_safepoint(FastFrame& /*frame*/) {
    if (gc_.stress_mode()) {
        gc_.collect();
    }
}

void FastInterpreter::execute_vector_op(FastFrame& frame, uint32_t inst) {
    BytecodeOp op = decode_op(inst);
    uint8_t dst = decode_dst(inst);
    uint8_t src1 = decode_src1(inst);
    uint8_t src2 = decode_src2(inst);
    uint8_t* vregs = frame.ensure_vector_regs();

    uint8_t* dst_bytes = vregs + dst * 32;
    const uint8_t* s1_bytes = vregs + src1 * 32;
    const uint8_t* s2_bytes = vregs + src2 * 32;

    switch (op) {
        case BytecodeOp::vzero: {
            std::memset(dst_bytes, 0, 32);
            break;
        }

        case BytecodeOp::vbroadcast: {
            uint64_t val = frame.registers[src1];
            // Broadcast 32-bit float / int or 64-bit float / int
            // Set all 8 32-bit lanes and 4 64-bit lanes
            uint32_t w = static_cast<uint32_t>(val);
            uint32_t* dst_u32 = reinterpret_cast<uint32_t*>(dst_bytes);
            for (size_t i = 0; i < 8; ++i) dst_u32[i] = w;
            break;
        }

        case BytecodeOp::vload: {
            uintptr_t addr = static_cast<uintptr_t>(frame.registers[src1]);
            std::memcpy(dst_bytes, reinterpret_cast<const void*>(addr), 32);
            break;
        }

        case BytecodeOp::vstore: {
            uintptr_t addr = static_cast<uintptr_t>(frame.registers[src1]);
            std::memcpy(reinterpret_cast<void*>(addr), dst_bytes, 32);
            break;
        }

        case BytecodeOp::vadd: {
            // Default to 8 x float
            const float* a = reinterpret_cast<const float*>(s1_bytes);
            const float* b = reinterpret_cast<const float*>(s2_bytes);
            float* out = reinterpret_cast<float*>(dst_bytes);
            for (size_t i = 0; i < 8; ++i) out[i] = a[i] + b[i];
            break;
        }

        case BytecodeOp::vsub: {
            const float* a = reinterpret_cast<const float*>(s1_bytes);
            const float* b = reinterpret_cast<const float*>(s2_bytes);
            float* out = reinterpret_cast<float*>(dst_bytes);
            for (size_t i = 0; i < 8; ++i) out[i] = a[i] - b[i];
            break;
        }

        case BytecodeOp::vmul: {
            const float* a = reinterpret_cast<const float*>(s1_bytes);
            const float* b = reinterpret_cast<const float*>(s2_bytes);
            float* out = reinterpret_cast<float*>(dst_bytes);
            for (size_t i = 0; i < 8; ++i) out[i] = a[i] * b[i];
            break;
        }

        case BytecodeOp::vdiv: {
            const float* a = reinterpret_cast<const float*>(s1_bytes);
            const float* b = reinterpret_cast<const float*>(s2_bytes);
            float* out = reinterpret_cast<float*>(dst_bytes);
            for (size_t i = 0; i < 8; ++i) out[i] = a[i] / b[i];
            break;
        }

        case BytecodeOp::vfma: {
            const float* a = reinterpret_cast<const float*>(s1_bytes);
            const float* b = reinterpret_cast<const float*>(s2_bytes);
            float* out = reinterpret_cast<float*>(dst_bytes);
            for (size_t i = 0; i < 8; ++i) out[i] = std::fma(a[i], b[i], out[i]);
            break;
        }

        case BytecodeOp::vneg: {
            const float* a = reinterpret_cast<const float*>(s1_bytes);
            float* out = reinterpret_cast<float*>(dst_bytes);
            for (size_t i = 0; i < 8; ++i) out[i] = -a[i];
            break;
        }

        case BytecodeOp::vmin: {
            const float* a = reinterpret_cast<const float*>(s1_bytes);
            const float* b = reinterpret_cast<const float*>(s2_bytes);
            float* out = reinterpret_cast<float*>(dst_bytes);
            for (size_t i = 0; i < 8; ++i) out[i] = std::fmin(a[i], b[i]);
            break;
        }

        case BytecodeOp::vmax: {
            const float* a = reinterpret_cast<const float*>(s1_bytes);
            const float* b = reinterpret_cast<const float*>(s2_bytes);
            float* out = reinterpret_cast<float*>(dst_bytes);
            for (size_t i = 0; i < 8; ++i) out[i] = std::fmax(a[i], b[i]);
            break;
        }

        case BytecodeOp::vsqrt: {
            const float* a = reinterpret_cast<const float*>(s1_bytes);
            float* out = reinterpret_cast<float*>(dst_bytes);
            for (size_t i = 0; i < 8; ++i) out[i] = std::sqrt(a[i]);
            break;
        }

        case BytecodeOp::vand: {
            const uint64_t* a = reinterpret_cast<const uint64_t*>(s1_bytes);
            const uint64_t* b = reinterpret_cast<const uint64_t*>(s2_bytes);
            uint64_t* out = reinterpret_cast<uint64_t*>(dst_bytes);
            for (size_t i = 0; i < 4; ++i) out[i] = a[i] & b[i];
            break;
        }

        case BytecodeOp::vor: {
            const uint64_t* a = reinterpret_cast<const uint64_t*>(s1_bytes);
            const uint64_t* b = reinterpret_cast<const uint64_t*>(s2_bytes);
            uint64_t* out = reinterpret_cast<uint64_t*>(dst_bytes);
            for (size_t i = 0; i < 4; ++i) out[i] = a[i] | b[i];
            break;
        }

        case BytecodeOp::vxor: {
            const uint64_t* a = reinterpret_cast<const uint64_t*>(s1_bytes);
            const uint64_t* b = reinterpret_cast<const uint64_t*>(s2_bytes);
            uint64_t* out = reinterpret_cast<uint64_t*>(dst_bytes);
            for (size_t i = 0; i < 4; ++i) out[i] = a[i] ^ b[i];
            break;
        }

        case BytecodeOp::vnot: {
            const uint64_t* a = reinterpret_cast<const uint64_t*>(s1_bytes);
            uint64_t* out = reinterpret_cast<uint64_t*>(dst_bytes);
            for (size_t i = 0; i < 4; ++i) out[i] = ~a[i];
            break;
        }

        case BytecodeOp::vextract_lane: {
            uint8_t lane = src2;
            const uint32_t* in = reinterpret_cast<const uint32_t*>(s1_bytes);
            frame.registers[dst] = (lane < 8) ? in[lane] : 0;
            break;
        }

        case BytecodeOp::vinsert_lane: {
            uint8_t lane = decode_src1(inst);
            uint32_t val = static_cast<uint32_t>(frame.registers[src2]);
            uint32_t* out = reinterpret_cast<uint32_t*>(dst_bytes);
            if (lane < 8) out[lane] = val;
            break;
        }

        case BytecodeOp::vshuffle: {
            // Copy s1 to dst
            std::memcpy(dst_bytes, s1_bytes, 32);
            break;
        }

        default:
            break;
    }
}

bool FastInterpreter::handle_osr_backedge(FastFrame& frame, uint32_t target_pc, RuntimeValue& out_res) {
    if (!frame.mir_fn) return false;
    const Function& fn = *frame.mir_fn;

    runtime::TieringFeedback& fb = runtime::TieringRegistry::instance().get_or_create(fn.name());
    fb.record_backedge();

    if (!runtime::OsrCoordinator::instance().is_enabled()) {
        return false;
    }

    const BytecodeFunction* bfn = frame.bfn;
    if (!bfn) return false;

    auto it = bfn->pc_block_map.find(target_pc);
    if (it == bfn->pc_block_map.end() || !it->second) {
        return false;
    }

    BasicBlock* loop_header = const_cast<BasicBlock*>(it->second);
    return runtime::OsrCoordinator::instance().try_osr_migration(*this, fn, loop_header, frame, out_res);
}

} // namespace brass
