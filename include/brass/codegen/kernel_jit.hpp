#pragma once

#include <brass/target/target.hpp>
#include <brass/mir/module.hpp>
#include <brass/mir/function.hpp>
#include <brass/mir/block.hpp>
#include <brass/mir/builder.hpp>
#include <brass/mir/types.hpp>
#include <brass/mir/loop_parallel.hpp>
#include <brass/codegen/jit_exec.hpp>

#include <functional>
#include <string>
#include <string_view>
#include <vector>
#include <memory>
#include <unordered_map>
#include <cstdint>
#include <cstddef>

namespace brass::codegen {

struct KernelOptions {
    bool enable_optimizations = true;
    bool enable_avx2 = true;
    bool enable_fma = true;
    bool enable_fp_reassociation = true;
    bool enable_vectorize = true;
    bool enable_slp = true;
    bool enable_unroll = true;
    size_t unroll_factor = 4;
    bool enable_parallel = false;
    uint64_t parallel_threshold = 1000;
    uint32_t parallel_workers = 0; // 0 = hardware concurrency
    ParallelLoopStats* parallel_stats = nullptr;
};

struct KernelDependenceInfo {
    DependenceKind kind = DependenceKind::None;
    DependenceDir dir = DependenceDir::Equal;
    int64_t distance = 0;
    bool has_distance = false;
    bool is_loop_carried = false;
};

struct KernelLoopAnalysis {
    bool is_parallelizable = false;
    bool is_doall = false;
    bool is_reduction = false;
    bool has_const_trip_count = false;
    uint64_t const_trip_count = 0;
    std::string rejection_reason;
    std::vector<KernelDependenceInfo> dependences;
};

class KernelFunction {
public:
    KernelFunction() = default;
    KernelFunction(
        std::string name,
        void* entry_point,
        size_t code_size,
        std::shared_ptr<JitExecutionEngine> engine
    ) : name_(std::move(name)),
        entry_point_(entry_point),
        code_size_(code_size),
        engine_(std::move(engine)) {}

    bool is_valid() const noexcept { return entry_point_ != nullptr; }
    explicit operator bool() const noexcept { return is_valid(); }

    std::string_view name() const noexcept { return name_; }
    void* entry_point() const noexcept { return entry_point_; }
    size_t code_size() const noexcept { return code_size_; }

    template <typename FuncPtr>
    FuncPtr get_function_pointer() const noexcept {
        return reinterpret_cast<FuncPtr>(entry_point_);
    }

    template <typename FuncPtr>
    FuncPtr as() const noexcept {
        return get_function_pointer<FuncPtr>();
    }

    std::shared_ptr<JitExecutionEngine> engine() const noexcept { return engine_; }

private:
    std::string name_;
    void* entry_point_ = nullptr;
    size_t code_size_ = 0;
    std::shared_ptr<JitExecutionEngine> engine_;
};

class KernelBuilder {
public:
    explicit KernelBuilder(Builder& b) noexcept : b_(b) {}
    KernelBuilder(Module& mod, Function* fn = nullptr);

    Builder& builder() noexcept { return b_; }
    const Builder& builder() const noexcept { return b_; }

    void position_at_end(BasicBlock* bb) noexcept { b_.position_at_end(bb); }

    // Raw typed pointer loads & stores (no GC overhead)
    Value* load_f32(Value* ptr, int32_t offset = 0) { return b_.build_load(Type::f32(), ptr, offset); }
    Value* load_f64(Value* ptr, int32_t offset = 0) { return b_.build_load(Type::f64(), ptr, offset); }
    Value* load_i32(Value* ptr, int32_t offset = 0) { return b_.build_load(Type::i32(), ptr, offset); }
    Value* load_i64(Value* ptr, int32_t offset = 0) { return b_.build_load(Type::i64(), ptr, offset); }

    Value* load_f32_indexed(Value* ptr, Value* index, uint8_t scale = 4, int32_t offset = 0) {
        return b_.build_load_indexed(Type::f32(), ptr, index, scale, offset);
    }
    Value* load_f64_indexed(Value* ptr, Value* index, uint8_t scale = 8, int32_t offset = 0) {
        return b_.build_load_indexed(Type::f64(), ptr, index, scale, offset);
    }
    Value* load_i32_indexed(Value* ptr, Value* index, uint8_t scale = 4, int32_t offset = 0) {
        return b_.build_load_indexed(Type::i32(), ptr, index, scale, offset);
    }
    Value* load_i64_indexed(Value* ptr, Value* index, uint8_t scale = 8, int32_t offset = 0) {
        return b_.build_load_indexed(Type::i64(), ptr, index, scale, offset);
    }

    Instruction* store_f32(Value* ptr, Value* val, int32_t offset = 0) {
        return b_.build_store(Type::f32(), ptr, offset, val);
    }
    Instruction* store_f64(Value* ptr, Value* val, int32_t offset = 0) {
        return b_.build_store(Type::f64(), ptr, offset, val);
    }
    Instruction* store_i32(Value* ptr, Value* val, int32_t offset = 0) {
        return b_.build_store(Type::i32(), ptr, offset, val);
    }
    Instruction* store_i64(Value* ptr, Value* val, int32_t offset = 0) {
        return b_.build_store(Type::i64(), ptr, offset, val);
    }

    Instruction* store_f32_indexed(Value* ptr, Value* index, Value* val, uint8_t scale = 4, int32_t offset = 0) {
        return b_.build_store_indexed(Type::f32(), ptr, index, scale, offset, val);
    }
    Instruction* store_f64_indexed(Value* ptr, Value* index, Value* val, uint8_t scale = 8, int32_t offset = 0) {
        return b_.build_store_indexed(Type::f64(), ptr, index, scale, offset, val);
    }
    Instruction* store_i32_indexed(Value* ptr, Value* index, Value* val, uint8_t scale = 4, int32_t offset = 0) {
        return b_.build_store_indexed(Type::i32(), ptr, index, scale, offset, val);
    }
    Instruction* store_i64_indexed(Value* ptr, Value* index, Value* val, uint8_t scale = 8, int32_t offset = 0) {
        return b_.build_store_indexed(Type::i64(), ptr, index, scale, offset, val);
    }

    // Vector operations
    Value* vload(Type type, Value* ptr, int32_t offset = 0) { return b_.build_vload(type, ptr, offset); }
    Instruction* vstore(Type type, Value* ptr, int32_t offset, Value* val) { return b_.build_vstore(type, ptr, offset, val); }
    Instruction* vstore(Type type, Value* ptr, Value* val) { return b_.build_vstore(type, ptr, 0, val); }

    Value* vload_f32x4(Value* ptr, int32_t offset = 0) { return b_.build_vload(Type::f32x4(), ptr, offset); }
    Value* vload_f64x2(Value* ptr, int32_t offset = 0) { return b_.build_vload(Type::f64x2(), ptr, offset); }
    Value* vload_f32x8(Value* ptr, int32_t offset = 0) { return b_.build_vload(Type::f32x8(), ptr, offset); }
    Value* vload_f64x4(Value* ptr, int32_t offset = 0) { return b_.build_vload(Type::f64x4(), ptr, offset); }

    Instruction* vstore_f32x4(Value* ptr, Value* val, int32_t offset = 0) { return b_.build_vstore(Type::f32x4(), ptr, offset, val); }
    Instruction* vstore_f64x2(Value* ptr, Value* val, int32_t offset = 0) { return b_.build_vstore(Type::f64x2(), ptr, offset, val); }
    Instruction* vstore_f32x8(Value* ptr, Value* val, int32_t offset = 0) { return b_.build_vstore(Type::f32x8(), ptr, offset, val); }
    Instruction* vstore_f64x4(Value* ptr, Value* val, int32_t offset = 0) { return b_.build_vstore(Type::f64x4(), ptr, offset, val); }

    // Arithmetic & FMA
    Value* add(Value* lhs, Value* rhs) { return b_.build_add(lhs, rhs); }
    Value* sub(Value* lhs, Value* rhs) { return b_.build_sub(lhs, rhs); }
    Value* mul(Value* lhs, Value* rhs) { return b_.build_mul(lhs, rhs); }
    Value* div(Value* lhs, Value* rhs) { return b_.build_sdiv(lhs, rhs); }
    Value* fma(Value* a, Value* b, Value* c) { return b_.build_fma(a, b, c); }
    Value* vfma(Value* a, Value* b, Value* c) { return b_.build_vfma(a, b, c); }

    Value* vadd(Value* lhs, Value* rhs) { return b_.build_vadd(lhs, rhs); }
    Value* vsub(Value* lhs, Value* rhs) { return b_.build_vsub(lhs, rhs); }
    Value* vmul(Value* lhs, Value* rhs) { return b_.build_vmul(lhs, rhs); }
    Value* vdiv(Value* lhs, Value* rhs) { return b_.build_vdiv(lhs, rhs); }
    Value* vmin(Value* lhs, Value* rhs) { return b_.build_vmin(lhs, rhs); }
    Value* vmax(Value* lhs, Value* rhs) { return b_.build_vmax(lhs, rhs); }
    Value* vbroadcast(Type vec_type, Value* scalar_val) { return b_.build_vbroadcast(vec_type, scalar_val); }
    Value* vzero(Type vec_type) { return b_.build_vzero(vec_type); }

    // Activation helpers
    Value* relu(Value* val);
    Value* relu_bias(Value* val, Value* bias);

    // ------------------------------------------------------------------
    // GPU (PTX) kernel helpers. Each is a few lines of MIR around a
    // `ptx_*` builtin call (lowered by PtxISel, see
    // docs/ptx_backend_design.md "Stage 4 implementation notes"); nothing
    // here knows PTX syntax. Implemented in src/codegen/kernel_builder_ptx.cpp.
    // ------------------------------------------------------------------

    // Constants
    Value* const_i32(int32_t v) { return b_.build_iconst_i32(v); }
    Value* const_i64(int64_t v) { return b_.build_iconst_i64(v); }
    Value* const_f32(float v) { return b_.build_fconst_f32(v); }

    // Thread / block / grid indices (all i32)
    Value* tid_x();     Value* tid_y();     Value* tid_z();
    Value* ctaid_x();   Value* ctaid_y();   Value* ctaid_z();
    Value* ntid_x();    Value* ntid_y();    Value* ntid_z();
    Value* nctaid_x();  Value* nctaid_y();  Value* nctaid_z();
    Value* lane_id();
    Value* warp_id();        // tid_x >> 5 (the warp index within the block, not %warpid)
    Value* global_tid_x();   // ctaid_x * ntid_x + tid_x

    // Barriers
    Instruction* sync();                 // bar.sync 0
    Instruction* bar_sync(uint32_t id);  // bar.sync <id>

    // Warp shuffles: 32-bit values (f32 or i32); delta/lane may be a constant.
    Value* shfl_down_f32(Value* v, uint32_t delta);
    Value* shfl_down_f32(Value* v, Value* delta);
    Value* shfl_up_f32(Value* v, uint32_t delta);
    Value* shfl_bfly_f32(Value* v, uint32_t lane_mask);
    Value* shfl_idx_f32(Value* v, uint32_t src_lane);
    Value* shfl_idx_f32(Value* v, Value* src_lane);
    Value* shfl_down_i32(Value* v, uint32_t delta);
    Value* shfl_bfly_i32(Value* v, uint32_t lane_mask);
    Value* shfl_idx_i32(Value* v, uint32_t src_lane);

    // Reductions. warp_reduce_* leave the full-warp result in lane 0 (the
    // other lanes hold partial sums). block_reduce_sum_f32 leaves the block
    // total in *every* thread; `shared_scratch` must be a shared_alloc_f32 of
    // at least 32 elements, the block size must be a multiple of 32 (up to
    // 1024) and every thread of the block must reach the call. It contains
    // two bar.sync and may be called repeatedly with the same scratch. The
    // builder is left positioned in a new block that the call created.
    Value* warp_reduce_sum_f32(Value* v);
    Value* warp_reduce_max_f32(Value* v);
    Value* block_reduce_sum_f32(Value* v, Value* shared_scratch);

    // Shared memory (per-kernel .shared arrays; count is a compile-time constant)
    Value* shared_alloc_f32(uint32_t count);
    Value* shared_alloc_i32(uint32_t count);
    Value* shared_load_f32(Value* smem, int32_t byte_offset = 0);
    Value* shared_load_f32_indexed(Value* smem, Value* index);
    Instruction* shared_store_f32(Value* smem, Value* val, int32_t byte_offset = 0);
    Instruction* shared_store_f32_indexed(Value* smem, Value* index, Value* val);
    Value* shared_load_i32(Value* smem, int32_t byte_offset = 0);
    Value* shared_load_i32_indexed(Value* smem, Value* index);
    Instruction* shared_store_i32(Value* smem, Value* val, int32_t byte_offset = 0);
    Instruction* shared_store_i32_indexed(Value* smem, Value* index, Value* val);

    // Fast math (f32)
    Value* rcp_approx(Value* x);
    Value* div_approx(Value* a, Value* b);
    Value* ex2_approx(Value* x);
    Value* lg2_approx(Value* x);
    Value* exp_fast(Value* x);      // ex2(x * log2 e)
    Value* log_fast(Value* x);      // lg2(x) * ln 2
    Value* rsqrt_approx(Value* x);
    Value* sqrt_approx(Value* x);
    Value* fabs(Value* x);          // f32 or f64
    Value* fmin(Value* a, Value* b);
    Value* fmax(Value* a, Value* b);

    // Conversions
    Value* f16_to_f32(Value* bits_i32);
    Value* f32_to_f16(Value* x);     // i32 holding the f16 bits
    Value* u32_to_f32(Value* x);
    Value* i32_to_f32(Value* x);
    Value* f32_to_u32(Value* x);
    Value* f32_to_i32(Value* x);

    // Narrow global loads/stores (result/value in i32, zero- or sign-extended)
    Value* load_u8(Value* ptr, int32_t byte_offset = 0);
    Value* load_s8(Value* ptr, int32_t byte_offset = 0);
    Value* load_u16(Value* ptr, int32_t byte_offset = 0);
    Value* load_s16(Value* ptr, int32_t byte_offset = 0);
    Instruction* store_u8(Value* ptr, Value* val, int32_t byte_offset = 0);
    Instruction* store_u16(Value* ptr, Value* val, int32_t byte_offset = 0);

    // Atomics (global memory; return the previous value)
    Value* atom_add_f32(Value* ptr, Value* val);
    Value* atom_add_i32(Value* ptr, Value* val);

    // Structured control flow. Both leave the builder positioned in the
    // join/exit block they create. `body` receives the induction variable
    // (same type as `start`); the loop runs while i < end (signed).
    void if_then(Value* cond, const std::function<void()>& body);
    void for_range(Value* start, Value* end, Value* step, const std::function<void(Value*)>& body);

private:
    std::unique_ptr<Builder> owned_builder_;
    Builder& b_;
};

class KernelJit {
public:
    explicit KernelJit(const KernelOptions& options = KernelOptions(), Target target = Target::host());
    ~KernelJit() = default;

    KernelJit(const KernelJit&) = delete;
    KernelJit& operator=(const KernelJit&) = delete;
    KernelJit(KernelJit&&) noexcept = default;
    KernelJit& operator=(KernelJit&&) noexcept = default;

    const KernelOptions& options() const noexcept { return options_; }
    KernelOptions& options() noexcept { return options_; }
    void set_options(const KernelOptions& options) { options_ = options; }

    const Target& target() const noexcept { return target_; }

    void set_parallel_workers(uint32_t workers);
    uint32_t get_parallel_workers() const;

    void set_parallel_threshold(uint64_t threshold) noexcept { options_.parallel_threshold = threshold; }
    uint64_t get_parallel_threshold() const noexcept { return options_.parallel_threshold; }

    void set_enable_parallel(bool enable) noexcept { options_.enable_parallel = enable; }
    bool is_parallel_enabled() const noexcept { return options_.enable_parallel; }

    void register_external_symbol(std::string_view name, void* address);

    // Polyhedral loop dependence analysis and parallelization
    std::vector<KernelLoopAnalysis> analyze_loops(Function& fn, const ParallelLoopOptions& options = {}) const;
    bool auto_parallelize(Function& fn, const ParallelLoopOptions& options = {}) const;

    // Kernel compilation
    KernelFunction compile(Function& fn);
    KernelFunction compile(Module& mod, std::string_view entry_name);

private:
    KernelOptions options_;
    Target target_;
    std::unordered_map<std::string, void*> external_symbols_;

    void setup_default_symbols(codegen::JitExecutionEngine& engine) const;
    void run_kernel_optimizations(Module& mod) const;
};

} // namespace brass::codegen
