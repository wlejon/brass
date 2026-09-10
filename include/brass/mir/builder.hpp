#pragma once

#include <brass/mir/module.hpp>
#include <brass/mir/function.hpp>
#include <brass/mir/block.hpp>
#include <brass/mir/instruction.hpp>
#include <brass/core/span.hpp>
#include <string_view>
#include <vector>
#include <initializer_list>
#include <cstdint>

namespace brass {

class Builder {
public:
    Builder() noexcept = default;
    explicit Builder(Module& module) noexcept;
    explicit Builder(Function& fn) noexcept;

    void set_module(Module* m) noexcept { module_ = m; }
    void set_function(Function* fn) noexcept;
    void position_at_end(BasicBlock* bb) noexcept;
    void position_before(Instruction* inst) noexcept;
    void position_after(Instruction* inst) noexcept;

    Module* current_module() const noexcept { return module_; }
    Function* current_function() const noexcept { return function_; }
    BasicBlock* current_block() const noexcept { return block_; }

    void set_current_loc(DebugLoc loc) noexcept { current_loc_ = loc; }
    void set_current_loc(uint32_t file_id, uint32_t line, uint32_t col = 0) noexcept {
        current_loc_ = DebugLoc(file_id, line, col);
    }
    DebugLoc current_loc() const noexcept { return current_loc_; }
    void clear_current_loc() noexcept { current_loc_ = DebugLoc(); }

    BasicBlock* create_block();
    BasicBlock* create_block(std::string_view name);
    BasicBlock* append_block();
    BasicBlock* append_block(std::string_view name);
    Value* add_block_param(BasicBlock* block, Type type);
    Value* add_param(Type type);

    // Constants
    Value* build_iconst_i32(int32_t val);
    Value* build_iconst_i64(int64_t val);
    Value* build_fconst_f64(double val);
    Value* build_patchable_const_i32(std::string_view symbol, int32_t initial_val);
    Value* build_patchable_const_i64(std::string_view symbol, int64_t initial_val);

    // Conversions
    Value* build_sext_i64(Value* val);
    Value* build_zext_i64(Value* val);
    Value* build_trunc_i32(Value* val);
    Value* build_fptosi_i32(Value* val);
    Value* build_fptosi_i64(Value* val);
    Value* build_sitofp_f64_i32(Value* val);
    Value* build_sitofp_f64_i64(Value* val);
    Value* build_bitcast_i64_f64(Value* val);
    Value* build_bitcast_f64_i64(Value* val);

    // Arithmetic / Logic
    Value* build_add(Value* lhs, Value* rhs);
    Value* build_sub(Value* lhs, Value* rhs);
    Value* build_mul(Value* lhs, Value* rhs);
    Value* build_sdiv(Value* lhs, Value* rhs);
    Value* build_udiv(Value* lhs, Value* rhs);
    Value* build_smod(Value* lhs, Value* rhs);
    Value* build_umod(Value* lhs, Value* rhs);
    Value* build_neg(Value* val);
    Value* build_and(Value* lhs, Value* rhs);
    Value* build_or(Value* lhs, Value* rhs);
    Value* build_xor(Value* lhs, Value* rhs);
    Value* build_shl(Value* lhs, Value* rhs);
    Value* build_lshr(Value* lhs, Value* rhs);
    Value* build_ashr(Value* lhs, Value* rhs);
    Value* build_not(Value* val);
    Value* build_clz(Value* val);
    Value* build_ctz(Value* val);
    Value* build_popcnt(Value* val);

    // Comparisons
    Value* build_eq(Value* lhs, Value* rhs);
    Value* build_ne(Value* lhs, Value* rhs);
    Value* build_slt(Value* lhs, Value* rhs);
    Value* build_ult(Value* lhs, Value* rhs);
    Value* build_sle(Value* lhs, Value* rhs);
    Value* build_ule(Value* lhs, Value* rhs);
    Value* build_sgt(Value* lhs, Value* rhs);
    Value* build_ugt(Value* lhs, Value* rhs);
    Value* build_sge(Value* lhs, Value* rhs);
    Value* build_uge(Value* lhs, Value* rhs);

    // Overflow-checked Arithmetic
    Value* build_sadd_overflow(Value* lhs, Value* rhs);
    Value* build_ssub_overflow(Value* lhs, Value* rhs);
    Value* build_smul_overflow(Value* lhs, Value* rhs);
    Value* build_uadd_overflow(Value* lhs, Value* rhs);
    Value* build_usub_overflow(Value* lhs, Value* rhs);
    Value* build_umul_overflow(Value* lhs, Value* rhs);

    // Selection
    Value* build_select(Value* cond, Value* true_val, Value* false_val);

    // Memory
    Value* build_load(Type type, Value* base);
    Value* build_load(Type type, Value* base, int32_t offset);
    Instruction* build_store(Type type, Value* base, int32_t offset, Value* val);
    Instruction* build_store(Type type, Value* base, Value* val);
    Value* build_load_indexed(Type type, Value* base, Value* index, uint8_t scale);
    Value* build_load_indexed(Type type, Value* base, Value* index, uint8_t scale, int32_t offset);
    Instruction* build_store_indexed(Type type, Value* base, Value* index, uint8_t scale, int32_t offset, Value* val);
    Instruction* build_store_indexed(Type type, Value* base, Value* index, uint8_t scale, Value* val);
    Instruction* build_write_barrier(Value* obj, Value* val);

    // Vector Arithmetic & Logic
    Value* build_vadd(Value* lhs, Value* rhs);
    Value* build_vsub(Value* lhs, Value* rhs);
    Value* build_vmul(Value* lhs, Value* rhs);
    Value* build_vdiv(Value* lhs, Value* rhs);
    Value* build_vneg(Value* val);
    Value* build_vmin(Value* lhs, Value* rhs);
    Value* build_vmax(Value* lhs, Value* rhs);
    Value* build_vsqrt(Value* val);
    Value* build_vand(Value* lhs, Value* rhs);
    Value* build_vor(Value* lhs, Value* rhs);
    Value* build_vxor(Value* lhs, Value* rhs);
    Value* build_vnot(Value* val);

    // Vector Memory
    Value* build_vload(Type type, Value* base);
    Value* build_vload(Type type, Value* base, int32_t offset);
    Instruction* build_vstore(Type type, Value* base, Value* val);
    Instruction* build_vstore(Type type, Value* base, int32_t offset, Value* val);

    // Vector Construction & Swizzle
    Value* build_vbroadcast(Type vec_type, Value* scalar_val);
    Value* build_vextract_lane(Value* vec_val, uint32_t lane);
    Value* build_vinsert_lane(Value* vec_val, Value* scalar_val, uint32_t lane);
    Value* build_vshuffle(Value* v1, Value* v2, uint32_t mask);
    Value* build_vzero(Type vec_type);

    // Calls & Safepoints
    Value* build_call(std::string_view callee, Type return_type, Span<Value* const> args);
    Value* build_call(std::string_view callee, Type return_type, std::initializer_list<Value*> args);
    Value* build_call(std::string_view callee, Type return_type);
    Value* build_call_indirect(Value* callee_ptr, Type return_type, Span<Value* const> args);
    Value* build_call_indirect(Value* callee_ptr, Type return_type, std::initializer_list<Value*> args);
    Value* build_call_indirect(Value* callee_ptr, Type return_type);
    Value* build_patchable_call(std::string_view patch_symbol, std::string_view callee, Type return_type, Span<Value* const> args);
    Value* build_patchable_call(std::string_view patch_symbol, std::string_view callee, Type return_type, std::initializer_list<Value*> args);
    Value* build_patchable_call(std::string_view patch_symbol, std::string_view callee, Type return_type);
    Instruction* build_safepoint();

    // Speculation
    Instruction* build_guard(Value* cond, std::string_view exit_label, Span<Value* const> state_values);
    Instruction* build_guard(Value* cond, std::string_view exit_label, std::initializer_list<Value*> state_values);
    Instruction* build_guard(Value* cond, std::string_view exit_label);
    Instruction* build_resume_point(uint32_t resume_id);

    // Terminators
    Instruction* build_br(BasicBlock* target);
    Instruction* build_br(BasicBlock* target, Span<Value* const> args);
    Instruction* build_br(BasicBlock* target, std::initializer_list<Value*> args);
    Instruction* build_br_if(Value* cond, BasicBlock* true_target, BasicBlock* false_target);
    Instruction* build_br_if(Value* cond, BasicBlock* true_target, Span<Value* const> true_args, BasicBlock* false_target, Span<Value* const> false_args);
    Instruction* build_br_if(Value* cond, BasicBlock* true_target, std::initializer_list<Value*> true_args, BasicBlock* false_target, std::initializer_list<Value*> false_args);
    Instruction* build_switch(Value* val, BasicBlock* default_target, Span<const SwitchCase> cases);
    Instruction* build_switch(Value* val, BasicBlock* default_target, std::initializer_list<SwitchCase> cases);
    Instruction* build_switch(Value* val, BasicBlock* default_target, Span<Value* const> default_args, Span<const SwitchCase> cases);
    Instruction* build_switch(Value* val, BasicBlock* default_target, std::initializer_list<Value*> default_args, std::initializer_list<SwitchCase> cases);
    Instruction* build_ret(Value* val);
    Instruction* build_ret_void();
    Instruction* build_unreachable();

    // Exceptions & Unwinding
    Instruction* build_throw(Value* val);
    Instruction* build_resume(Value* val = nullptr);
    Value* build_landing_pad(Type type = Type::i64());
    Instruction* build_invoke(std::string_view callee, Type return_type, Span<Value* const> args, BasicBlock* normal_target, BasicBlock* unwind_target);
    Instruction* build_invoke(std::string_view callee, Type return_type, std::initializer_list<Value*> args, BasicBlock* normal_target, BasicBlock* unwind_target);
    Instruction* build_invoke(std::string_view callee, Type return_type, Span<Value* const> args, BasicBlock* normal_target, Span<Value* const> normal_args, BasicBlock* unwind_target, Span<Value* const> unwind_args = {});
    Instruction* build_invoke(std::string_view callee, Type return_type, std::initializer_list<Value*> args, BasicBlock* normal_target, std::initializer_list<Value*> normal_args, BasicBlock* unwind_target, std::initializer_list<Value*> unwind_args = {});

    // Coroutines
    Value* build_coro_create(std::string_view callee, Span<Value* const> args = {});
    Value* build_coro_create(std::string_view callee, std::initializer_list<Value*> args);
    Value* build_coro_suspend(Value* yield_val, uint32_t state_id = 0, Type return_type = Type::i64());
    Value* build_coro_resume(Value* coro_val, Value* input_val = nullptr, Type return_type = Type::i64());
    Instruction* build_coro_destroy(Value* coro_val);

    Instruction* insert(Instruction* inst);
    Value* create_value(Type type);

private:
    Arena& get_arena();
    StringPool& get_string_pool();

    Module* module_ = nullptr;
    Function* function_ = nullptr;
    BasicBlock* block_ = nullptr;
    Instruction* insert_before_ = nullptr;
    DebugLoc current_loc_;
};

} // namespace brass
