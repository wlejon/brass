#pragma once

#include <brass/target/target.hpp>
#include <brass/mir/module.hpp>
#include <brass/mir/function.hpp>
#include <brass/mir/block.hpp>
#include <brass/mir/builder.hpp>
#include <brass/mir/types.hpp>
#include <brass/mir/loop_parallel.hpp>
#include <brass/codegen/jit_exec.hpp>

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
