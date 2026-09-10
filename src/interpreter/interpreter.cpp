#include <brass/interpreter/interpreter.hpp>
#include <brass/gc/generational_gc.hpp>
#include <brass/gc/runtime_gc.hpp>
#include "interpreter_coro.hpp"
#include <brass/runtime/osr_coordinator.hpp>
#include <brass/runtime/code_installer.hpp>
#include <brass/runtime/tiering.hpp>
#include <iostream>
#include <cmath>

namespace brass {

Interpreter::Interpreter(size_t gc_semispace_size)
    : gc_(gc_semispace_size) {
    gc_.set_root_provider([this](std::vector<uintptr_t*>& roots) {
        this->collect_all_roots(roots);
    });
    register_builtin_host_functions();
}

RuntimeValue Interpreter::run(const Function& fn, const std::vector<RuntimeValue>& args) {
    if (fn.parent()) {
        module_ = fn.parent();
        runtime::TieringRegistry::instance().set_active_module(fn.parent());
        runtime::FunctionDispatchTable::instance().get_or_create(fn.name(), &fn);
    }
    return execute_function(fn, args);
}

RuntimeValue Interpreter::resume(const Function& fn, uint32_t resume_id, const std::vector<RuntimeValue>& state_values) {
    if (fn.parent()) {
        module_ = fn.parent();
    }
    BasicBlock* target_bb = fn.get_resume_target(resume_id);
    if (!target_bb) {
        throw InterpreterException("Resume target ID " + std::to_string(resume_id) + " not found in function " + std::string(fn.name()));
    }
    return execute_function_from_block(fn, target_bb, state_values);
}

RuntimeValue Interpreter::resume_with_frame(const Function& fn, uint32_t resume_id, const std::vector<RuntimeValue>& state_values, InterpreterFrame& frame) {
    if (fn.parent()) {
        module_ = fn.parent();
    }
    BasicBlock* target_bb = fn.get_resume_target(resume_id);
    if (!target_bb) {
        throw InterpreterException("Resume target ID " + std::to_string(resume_id) + " not found in function " + std::string(fn.name()));
    }

    const Instruction* guard_inst = nullptr;
    for (const auto* bb : fn.blocks()) {
        if (!bb) continue;
        for (const auto* inst : *bb) {
            if (inst && inst->opcode() == Opcode::guard && inst->resume_id() == resume_id) {
                guard_inst = inst;
                break;
            }
        }
        if (guard_inst) break;
    }

    if (guard_inst) {
        for (size_t i = 0; i < guard_inst->state_map().size() && i < state_values.size(); ++i) {
            const Value* sv = guard_inst->state_map()[i];
            if (sv) {
                frame.set_value(sv, state_values[i]);
            }
        }
    }
    for (size_t i = 0; i < target_bb->param_count() && i < state_values.size(); ++i) {
        frame.set_value(target_bb->param(i), state_values[i]);
    }

    return execute_function_from_block(fn, target_bb, {}, &frame);
}

RuntimeValue Interpreter::execute_function(const Function& fn, const std::vector<RuntimeValue>& args) {
    if (fn.parent() && !runtime::TieringRegistry::instance().active_module()) {
        runtime::TieringRegistry::instance().set_active_module(fn.parent());
    }

    auto* handle = runtime::FunctionDispatchTable::instance().find(fn.name());
    if (handle) {
        void* native_code = handle->native_entry();
        if (native_code != nullptr) {
            return handle->call_native(args);
        }
    }

    auto& feedback = runtime::TieringRegistry::instance().get_or_create(fn.name());
    feedback.record_invocation();

    if (handle) {
        void* native_code = handle->native_entry();
        if (native_code != nullptr) {
            return handle->call_native(args);
        }
    }

    BasicBlock* entry = fn.entry_block();
    if (!entry) {
        return RuntimeValue::from_void();
    }
    return execute_function_from_block(fn, entry, args);
}

RuntimeValue Interpreter::execute_function_from_block(const Function& fn, BasicBlock* start_block, const std::vector<RuntimeValue>& block_args, InterpreterFrame* existing_frame) {
    if (call_depth_ + 1 > max_call_depth_) {
        throw InterpreterException("Maximum interpreter call depth exceeded (" + std::to_string(max_call_depth_) + ")");
    }

    InterpreterFrame local_frame(&fn, current_frame_);
    InterpreterFrame& frame = existing_frame ? *existing_frame : local_frame;
    InterpreterFrame* prev_frame = current_frame_;
    current_frame_ = &frame;
    call_depth_++;

    struct StackGuard {
        InterpreterFrame*& cur;
        InterpreterFrame* prev;
        size_t& depth;
        ~StackGuard() {
            cur = prev;
            depth--;
        }
    } guard{current_frame_, prev_frame, call_depth_};

    if (!existing_frame) {
        // Bind block arguments to start block parameters
        for (size_t i = 0; i < start_block->param_count() && i < block_args.size(); ++i) {
            frame.set_value(start_block->param(i), block_args[i]);
        }

        // Also bind to entry block parameters if starting at an interior resume block
        if (start_block != fn.entry_block() && fn.entry_block()) {
            const auto* entry = fn.entry_block();
            for (size_t i = 0; i < entry->param_count() && i < block_args.size(); ++i) {
                frame.set_value(entry->param(i), block_args[i]);
            }
        }
    }

    BasicBlock* cur_bb = start_block;

    while (cur_bb != nullptr) {
        bool transitioned = false;

        for (Instruction* inst = cur_bb->head(); inst != nullptr; inst = inst->next()) {
            if (max_instructions_ > 0 && ++total_instructions_executed_ > max_instructions_) {
                throw InterpreterException("Maximum instruction execution count exceeded (" + std::to_string(max_instructions_) + ")");
            } else {
                total_instructions_executed_++;
            }

            switch (inst->opcode()) {
                case Opcode::iconst_i32: {
                    frame.set_value(inst->result(), RuntimeValue::from_i32(inst->imm_i32()));
                    break;
                }
                case Opcode::iconst_i64: {
                    frame.set_value(inst->result(), RuntimeValue::from_i64(inst->imm_i64()));
                    break;
                }
                case Opcode::fconst_f64: {
                    frame.set_value(inst->result(), RuntimeValue::from_f64(inst->imm_f64()));
                    break;
                }
                case Opcode::patchable_const_i32: {
                    int64_t v = get_patched_const(inst->symbol(), inst->imm_i32());
                    frame.set_value(inst->result(), RuntimeValue::from_i32(static_cast<int32_t>(v)));
                    break;
                }
                case Opcode::patchable_const_i64: {
                    int64_t v = get_patched_const(inst->symbol(), inst->imm_i64());
                    frame.set_value(inst->result(), RuntimeValue::from_i64(v));
                    break;
                }

                case Opcode::sext_i64: {
                    frame.set_value(inst->result(), val_sext_i64(frame.get_value(inst->operand(0))));
                    break;
                }
                case Opcode::zext_i64: {
                    frame.set_value(inst->result(), val_zext_i64(frame.get_value(inst->operand(0))));
                    break;
                }
                case Opcode::trunc_i32: {
                    frame.set_value(inst->result(), val_trunc_i32(frame.get_value(inst->operand(0))));
                    break;
                }
                case Opcode::fptosi_i32: {
                    frame.set_value(inst->result(), val_fptosi_i32(frame.get_value(inst->operand(0))));
                    break;
                }
                case Opcode::fptosi_i64: {
                    frame.set_value(inst->result(), val_fptosi_i64(frame.get_value(inst->operand(0))));
                    break;
                }
                case Opcode::sitofp_f64_i32: {
                    frame.set_value(inst->result(), val_sitofp_f64_i32(frame.get_value(inst->operand(0))));
                    break;
                }
                case Opcode::sitofp_f64_i64: {
                    frame.set_value(inst->result(), val_sitofp_f64_i64(frame.get_value(inst->operand(0))));
                    break;
                }
                case Opcode::bitcast_i64_f64: {
                    frame.set_value(inst->result(), val_bitcast_i64_f64(frame.get_value(inst->operand(0))));
                    break;
                }
                case Opcode::bitcast_f64_i64: {
                    frame.set_value(inst->result(), val_bitcast_f64_i64(frame.get_value(inst->operand(0))));
                    break;
                }

                case Opcode::add: {
                    frame.set_value(inst->result(), val_add(frame.get_value(inst->operand(0)), frame.get_value(inst->operand(1))));
                    break;
                }
                case Opcode::sub: {
                    frame.set_value(inst->result(), val_sub(frame.get_value(inst->operand(0)), frame.get_value(inst->operand(1))));
                    break;
                }
                case Opcode::mul: {
                    frame.set_value(inst->result(), val_mul(frame.get_value(inst->operand(0)), frame.get_value(inst->operand(1))));
                    break;
                }
                case Opcode::fma_f32: {
                    frame.set_value(inst->result(), val_fma_f32(frame.get_value(inst->operand(0)), frame.get_value(inst->operand(1)), frame.get_value(inst->operand(2))));
                    break;
                }
                case Opcode::fma_f64: {
                    frame.set_value(inst->result(), val_fma_f64(frame.get_value(inst->operand(0)), frame.get_value(inst->operand(1)), frame.get_value(inst->operand(2))));
                    break;
                }
                case Opcode::sdiv: {
                    frame.set_value(inst->result(), val_sdiv(frame.get_value(inst->operand(0)), frame.get_value(inst->operand(1))));
                    break;
                }
                case Opcode::udiv: {
                    frame.set_value(inst->result(), val_udiv(frame.get_value(inst->operand(0)), frame.get_value(inst->operand(1))));
                    break;
                }
                case Opcode::smod: {
                    frame.set_value(inst->result(), val_smod(frame.get_value(inst->operand(0)), frame.get_value(inst->operand(1))));
                    break;
                }
                case Opcode::umod: {
                    frame.set_value(inst->result(), val_umod(frame.get_value(inst->operand(0)), frame.get_value(inst->operand(1))));
                    break;
                }
                case Opcode::sadd_overflow: {
                    frame.set_value(inst->result(), val_sadd_overflow(frame.get_value(inst->operand(0)), frame.get_value(inst->operand(1))));
                    break;
                }
                case Opcode::ssub_overflow: {
                    frame.set_value(inst->result(), val_ssub_overflow(frame.get_value(inst->operand(0)), frame.get_value(inst->operand(1))));
                    break;
                }
                case Opcode::smul_overflow: {
                    frame.set_value(inst->result(), val_smul_overflow(frame.get_value(inst->operand(0)), frame.get_value(inst->operand(1))));
                    break;
                }
                case Opcode::uadd_overflow: {
                    frame.set_value(inst->result(), val_uadd_overflow(frame.get_value(inst->operand(0)), frame.get_value(inst->operand(1))));
                    break;
                }
                case Opcode::usub_overflow: {
                    frame.set_value(inst->result(), val_usub_overflow(frame.get_value(inst->operand(0)), frame.get_value(inst->operand(1))));
                    break;
                }
                case Opcode::umul_overflow: {
                    frame.set_value(inst->result(), val_umul_overflow(frame.get_value(inst->operand(0)), frame.get_value(inst->operand(1))));
                    break;
                }
                case Opcode::neg: {
                    frame.set_value(inst->result(), val_neg(frame.get_value(inst->operand(0))));
                    break;
                }
                case Opcode::and_: {
                    frame.set_value(inst->result(), val_and(frame.get_value(inst->operand(0)), frame.get_value(inst->operand(1))));
                    break;
                }
                case Opcode::or_: {
                    frame.set_value(inst->result(), val_or(frame.get_value(inst->operand(0)), frame.get_value(inst->operand(1))));
                    break;
                }
                case Opcode::xor_: {
                    frame.set_value(inst->result(), val_xor(frame.get_value(inst->operand(0)), frame.get_value(inst->operand(1))));
                    break;
                }
                case Opcode::shl: {
                    frame.set_value(inst->result(), val_shl(frame.get_value(inst->operand(0)), frame.get_value(inst->operand(1))));
                    break;
                }
                case Opcode::lshr: {
                    frame.set_value(inst->result(), val_lshr(frame.get_value(inst->operand(0)), frame.get_value(inst->operand(1))));
                    break;
                }
                case Opcode::ashr: {
                    frame.set_value(inst->result(), val_ashr(frame.get_value(inst->operand(0)), frame.get_value(inst->operand(1))));
                    break;
                }
                case Opcode::not_: {
                    frame.set_value(inst->result(), val_not(frame.get_value(inst->operand(0))));
                    break;
                }
                case Opcode::clz: {
                    frame.set_value(inst->result(), val_clz(frame.get_value(inst->operand(0))));
                    break;
                }
                case Opcode::ctz: {
                    frame.set_value(inst->result(), val_ctz(frame.get_value(inst->operand(0))));
                    break;
                }
                case Opcode::popcnt: {
                    frame.set_value(inst->result(), val_popcnt(frame.get_value(inst->operand(0))));
                    break;
                }

                case Opcode::eq: {
                    frame.set_value(inst->result(), val_eq(frame.get_value(inst->operand(0)), frame.get_value(inst->operand(1))));
                    break;
                }
                case Opcode::ne: {
                    frame.set_value(inst->result(), val_ne(frame.get_value(inst->operand(0)), frame.get_value(inst->operand(1))));
                    break;
                }
                case Opcode::slt: {
                    frame.set_value(inst->result(), val_slt(frame.get_value(inst->operand(0)), frame.get_value(inst->operand(1))));
                    break;
                }
                case Opcode::ult: {
                    frame.set_value(inst->result(), val_ult(frame.get_value(inst->operand(0)), frame.get_value(inst->operand(1))));
                    break;
                }
                case Opcode::sle: {
                    frame.set_value(inst->result(), val_sle(frame.get_value(inst->operand(0)), frame.get_value(inst->operand(1))));
                    break;
                }
                case Opcode::ule: {
                    frame.set_value(inst->result(), val_ule(frame.get_value(inst->operand(0)), frame.get_value(inst->operand(1))));
                    break;
                }
                case Opcode::sgt: {
                    frame.set_value(inst->result(), val_sgt(frame.get_value(inst->operand(0)), frame.get_value(inst->operand(1))));
                    break;
                }
                case Opcode::ugt: {
                    frame.set_value(inst->result(), val_ugt(frame.get_value(inst->operand(0)), frame.get_value(inst->operand(1))));
                    break;
                }
                case Opcode::sge: {
                    frame.set_value(inst->result(), val_sge(frame.get_value(inst->operand(0)), frame.get_value(inst->operand(1))));
                    break;
                }
                case Opcode::uge: {
                    frame.set_value(inst->result(), val_uge(frame.get_value(inst->operand(0)), frame.get_value(inst->operand(1))));
                    break;
                }

                case Opcode::select: {
                    RuntimeValue cond_v = frame.get_value(inst->operand(0));
                    int32_t cond = cond_v.is_i32() ? cond_v.as_i32() : (cond_v.as_i64() != 0 ? 1 : 0);
                    if (cond != 0) {
                        frame.set_value(inst->result(), frame.get_value(inst->operand(1)));
                    } else {
                        frame.set_value(inst->result(), frame.get_value(inst->operand(2)));
                    }
                    break;
                }

                case Opcode::load: {
                    RuntimeValue base = frame.get_value(inst->operand(0));
                    RuntimeValue res = gc_.read_memory(base.raw_bits(), inst->offset(), inst->type());
                    frame.set_value(inst->result(), res);
                    break;
                }
                case Opcode::store: {
                    RuntimeValue base = frame.get_value(inst->operand(0));
                    RuntimeValue val = frame.get_value(inst->operand(1));
                    gc_.write_memory(base.raw_bits(), inst->offset(), inst->memory_type(), val);
                    break;
                }
                case Opcode::load_indexed: {
                    RuntimeValue base = frame.get_value(inst->operand(0));
                    RuntimeValue idx = frame.get_value(inst->operand(1));
                    int64_t idx_val = idx.is_i32() ? idx.as_i32() : idx.as_i64();
                    int32_t effective_offset = static_cast<int32_t>(idx_val * inst->scale()) + inst->offset();
                    RuntimeValue res = gc_.read_memory(base.raw_bits(), effective_offset, inst->type());
                    frame.set_value(inst->result(), res);
                    break;
                }
                case Opcode::store_indexed: {
                    RuntimeValue base = frame.get_value(inst->operand(0));
                    RuntimeValue idx = frame.get_value(inst->operand(1));
                    RuntimeValue val = frame.get_value(inst->operand(2));
                    int64_t idx_val = idx.is_i32() ? idx.as_i32() : idx.as_i64();
                    int32_t effective_offset = static_cast<int32_t>(idx_val * inst->scale()) + inst->offset();
                    gc_.write_memory(base.raw_bits(), effective_offset, inst->memory_type(), val);
                    break;
                }
                case Opcode::write_barrier: {
                    RuntimeValue obj = frame.get_value(inst->operand(0));
                    RuntimeValue val = frame.get_value(inst->operand(1));
                    uintptr_t obj_addr = obj.raw_bits();
                    uintptr_t val_addr = val.raw_bits();
                    GenerationalGC* gen = gen_gc_;
                    if (!gen) {
                        gen = brass_get_active_generational_gc();
                    }
                    if (gen) {
                        if (gen->is_old(obj_addr) && gen->is_young(val_addr)) {
                            gen->card_table().mark_card(obj_addr);
                        }
                    }
                    break;
                }

                case Opcode::vadd: {
                    frame.set_value(inst->result(), val_vadd(frame.get_value(inst->operand(0)), frame.get_value(inst->operand(1))));
                    break;
                }
                case Opcode::vsub: {
                    frame.set_value(inst->result(), val_vsub(frame.get_value(inst->operand(0)), frame.get_value(inst->operand(1))));
                    break;
                }
                case Opcode::vmul: {
                    frame.set_value(inst->result(), val_vmul(frame.get_value(inst->operand(0)), frame.get_value(inst->operand(1))));
                    break;
                }
                case Opcode::vdiv: {
                    frame.set_value(inst->result(), val_vdiv(frame.get_value(inst->operand(0)), frame.get_value(inst->operand(1))));
                    break;
                }
                case Opcode::vfma: {
                    frame.set_value(inst->result(), val_vfma(frame.get_value(inst->operand(0)), frame.get_value(inst->operand(1)), frame.get_value(inst->operand(2))));
                    break;
                }
                case Opcode::vneg: {
                    frame.set_value(inst->result(), val_vneg(frame.get_value(inst->operand(0))));
                    break;
                }
                case Opcode::vmin: {
                    frame.set_value(inst->result(), val_vmin(frame.get_value(inst->operand(0)), frame.get_value(inst->operand(1))));
                    break;
                }
                case Opcode::vmax: {
                    frame.set_value(inst->result(), val_vmax(frame.get_value(inst->operand(0)), frame.get_value(inst->operand(1))));
                    break;
                }
                case Opcode::vsqrt: {
                    frame.set_value(inst->result(), val_vsqrt(frame.get_value(inst->operand(0))));
                    break;
                }
                case Opcode::vand: {
                    frame.set_value(inst->result(), val_vand(frame.get_value(inst->operand(0)), frame.get_value(inst->operand(1))));
                    break;
                }
                case Opcode::vor: {
                    frame.set_value(inst->result(), val_vor(frame.get_value(inst->operand(0)), frame.get_value(inst->operand(1))));
                    break;
                }
                case Opcode::vxor: {
                    frame.set_value(inst->result(), val_vxor(frame.get_value(inst->operand(0)), frame.get_value(inst->operand(1))));
                    break;
                }
                case Opcode::vnot: {
                    frame.set_value(inst->result(), val_vnot(frame.get_value(inst->operand(0))));
                    break;
                }
                case Opcode::vload: {
                    RuntimeValue base = frame.get_value(inst->operand(0));
                    RuntimeValue res = gc_.read_memory(base.raw_bits(), inst->offset(), inst->type());
                    frame.set_value(inst->result(), res);
                    break;
                }
                case Opcode::vstore: {
                    RuntimeValue base = frame.get_value(inst->operand(0));
                    RuntimeValue val = frame.get_value(inst->operand(1));
                    gc_.write_memory(base.raw_bits(), inst->offset(), inst->memory_type(), val);
                    break;
                }
                case Opcode::vbroadcast: {
                    frame.set_value(inst->result(), val_vbroadcast(inst->type(), frame.get_value(inst->operand(0))));
                    break;
                }
                case Opcode::vextract_lane: {
                    frame.set_value(inst->result(), val_vextract_lane(frame.get_value(inst->operand(0)), inst->lane()));
                    break;
                }
                case Opcode::vinsert_lane: {
                    frame.set_value(inst->result(), val_vinsert_lane(frame.get_value(inst->operand(0)), frame.get_value(inst->operand(1)), inst->lane()));
                    break;
                }
                case Opcode::vshuffle: {
                    frame.set_value(inst->result(), val_vshuffle(frame.get_value(inst->operand(0)), frame.get_value(inst->operand(1)), inst->shuffle_mask()));
                    break;
                }
                case Opcode::vzero: {
                    frame.set_value(inst->result(), val_vzero(inst->type()));
                    break;
                }

                case Opcode::call: {
                    std::string_view callee = inst->symbol();
                    std::vector<RuntimeValue> call_args;
                    call_args.reserve(inst->operand_count());
                    for (size_t i = 0; i < inst->operand_count(); ++i) {
                        call_args.push_back(frame.get_value(inst->operand(i)));
                    }

                    RuntimeValue call_res;
                    const Function* target_fn = module_ ? module_->get_function(callee) : nullptr;
                    if (target_fn) {
                        runtime::FunctionDispatchTable::instance().get_or_create(callee, target_fn);
                        call_res = execute_function(*target_fn, call_args);
                    } else {
                        auto it = external_functions_.find(std::string(callee));
                        if (it != external_functions_.end()) {
                            call_res = it->second(*this, call_args);
                        } else {
                            throw InterpreterException("Call to undefined function: " + std::string(callee));
                        }
                    }

                    if (inst->result()) {
                        frame.set_value(inst->result(), call_res);
                    }
                    break;
                }

                case Opcode::call_indirect: {
                    RuntimeValue callee_val = frame.get_value(inst->operand(0));
                    uintptr_t callee_ptr = callee_val.as_ptr();

                    std::vector<RuntimeValue> call_args;
                    call_args.reserve(inst->operand_count() > 0 ? inst->operand_count() - 1 : 0);
                    for (size_t i = 1; i < inst->operand_count(); ++i) {
                        call_args.push_back(frame.get_value(inst->operand(i)));
                    }

                    RuntimeValue call_res;
                    auto fn_it = function_pointers_.find(callee_ptr);
                    if (fn_it != function_pointers_.end()) {
                        call_res = execute_function(*fn_it->second, call_args);
                    } else {
                        auto host_it = host_function_pointers_.find(callee_ptr);
                        if (host_it != host_function_pointers_.end()) {
                            call_res = host_it->second(*this, call_args);
                        } else {
                            throw InterpreterException("Call indirect to unregistered target pointer: " + std::to_string(callee_ptr));
                        }
                    }

                    if (inst->result()) {
                        frame.set_value(inst->result(), call_res);
                    }
                    break;
                }

                case Opcode::patchable_call: {
                    std::string_view default_callee = inst->extra_symbol().empty() ? inst->symbol() : inst->extra_symbol();
                    std::string_view callee = get_patched_call(inst->symbol(), default_callee);

                    std::vector<RuntimeValue> call_args;
                    call_args.reserve(inst->operand_count());
                    for (size_t i = 0; i < inst->operand_count(); ++i) {
                        call_args.push_back(frame.get_value(inst->operand(i)));
                    }

                    RuntimeValue call_res;
                    const Function* target_fn = module_ ? module_->get_function(callee) : nullptr;
                    if (target_fn) {
                        call_res = execute_function(*target_fn, call_args);
                    } else {
                        auto it = external_functions_.find(std::string(callee));
                        if (it != external_functions_.end()) {
                            call_res = it->second(*this, call_args);
                        } else {
                            throw InterpreterException("Patchable call to undefined function: " + std::string(callee));
                        }
                    }

                    if (inst->result()) {
                        frame.set_value(inst->result(), call_res);
                    }
                    break;
                }

                case Opcode::safepoint: {
                    if (gc_.stress_mode()) {
                        gc_.collect();
                    }
                    break;
                }

                case Opcode::guard: {
                    RuntimeValue cond = frame.get_value(inst->operand(0));
                    if (cond.as_i32() == 0) {
                        std::vector<RuntimeValue> captured;
                        captured.reserve(inst->state_map().size());
                        for (Value* v : inst->state_map()) {
                            captured.push_back(frame.get_value(v));
                        }

                        last_deopt_.deoptimized = true;
                        last_deopt_.exit_stub = std::string(inst->symbol());
                        last_deopt_.state_map = std::move(captured);
                        last_deopt_.resume_id = inst->resume_id();

                        if (deopt_handler_) {
                            return deopt_handler_(*this, last_deopt_);
                        }

                        if (module_ && !inst->symbol().empty()) {
                            const Function* stub_fn = module_->get_function(inst->symbol());
                            if (stub_fn) {
                                return execute_function(*stub_fn, last_deopt_.state_map);
                            }
                        }

                        BasicBlock* resume_target = fn.get_resume_target(inst->resume_id());
                        if (resume_target) {
                            for (size_t i = 0; i < resume_target->param_count() && i < last_deopt_.state_map.size(); ++i) {
                                frame.set_value(resume_target->param(i), last_deopt_.state_map[i]);
                            }
                            cur_bb = resume_target;
                            transitioned = true;
                            break;
                        }

                        throw DeoptException(last_deopt_);
                    }
                    break;
                }

                case Opcode::resume_point:
                case Opcode::osr_entry: {
                    // Metadata marker: no-op during forward execution
                    break;
                }

                case Opcode::br: {
                    std::vector<RuntimeValue> target_args;
                    target_args.reserve(inst->branch_target().args.size());
                    for (Value* a : inst->branch_target().args) {
                        target_args.push_back(frame.get_value(a));
                    }

                    BasicBlock* next_bb = inst->branch_target().block;
                    if (!next_bb) {
                        throw InterpreterException("Branch to null basic block");
                    }

                    for (size_t i = 0; i < next_bb->param_count() && i < target_args.size(); ++i) {
                        frame.set_value(next_bb->param(i), target_args[i]);
                    }

                    if (runtime::OsrCoordinator::instance().is_enabled() &&
                        runtime::OsrCoordinator::instance().is_loop_backedge(fn, cur_bb, next_bb)) {
                        RuntimeValue osr_res;
                        if (runtime::OsrCoordinator::instance().try_osr_migration(*this, fn, next_bb, frame, osr_res)) {
                            return osr_res;
                        }
                    }

                    cur_bb = next_bb;
                    transitioned = true;
                    break;
                }

                case Opcode::br_if: {
                    RuntimeValue cond = frame.get_value(inst->operand(0));
                    const BranchTarget& target = (cond.as_i32() != 0) ? inst->true_target() : inst->false_target();

                    std::vector<RuntimeValue> target_args;
                    target_args.reserve(target.args.size());
                    for (Value* a : target.args) {
                        target_args.push_back(frame.get_value(a));
                    }

                    BasicBlock* next_bb = target.block;
                    if (!next_bb) {
                        throw InterpreterException("Conditional branch to null basic block");
                    }

                    for (size_t i = 0; i < next_bb->param_count() && i < target_args.size(); ++i) {
                        frame.set_value(next_bb->param(i), target_args[i]);
                    }

                    if (runtime::OsrCoordinator::instance().is_enabled() &&
                        runtime::OsrCoordinator::instance().is_loop_backedge(fn, cur_bb, next_bb)) {
                        RuntimeValue osr_res;
                        if (runtime::OsrCoordinator::instance().try_osr_migration(*this, fn, next_bb, frame, osr_res)) {
                            return osr_res;
                        }
                    }

                    cur_bb = next_bb;
                    transitioned = true;
                    break;
                }

                case Opcode::switch_: {
                    RuntimeValue cond = frame.get_value(inst->operand(0));
                    int64_t cond_val = cond.is_i32() ? static_cast<int64_t>(cond.as_i32()) : cond.as_i64();
                    const BranchTarget* selected_target = &inst->default_target();
                    for (const auto& sc : inst->switch_cases()) {
                        if (sc.value == cond_val) {
                            selected_target = &sc.target;
                            break;
                        }
                    }

                    std::vector<RuntimeValue> target_args;
                    target_args.reserve(selected_target->args.size());
                    for (Value* a : selected_target->args) {
                        target_args.push_back(frame.get_value(a));
                    }

                    BasicBlock* next_bb = selected_target->block;
                    if (!next_bb) {
                        throw InterpreterException("Switch target to null basic block");
                    }

                    for (size_t i = 0; i < next_bb->param_count() && i < target_args.size(); ++i) {
                        frame.set_value(next_bb->param(i), target_args[i]);
                    }

                    if (runtime::OsrCoordinator::instance().is_enabled() &&
                        runtime::OsrCoordinator::instance().is_loop_backedge(fn, cur_bb, next_bb)) {
                        RuntimeValue osr_res;
                        if (runtime::OsrCoordinator::instance().try_osr_migration(*this, fn, next_bb, frame, osr_res)) {
                            return osr_res;
                        }
                    }

                    cur_bb = next_bb;
                    transitioned = true;
                    break;
                }

                case Opcode::ret: {
                    if (inst->operand_count() > 0 && inst->operand(0) != nullptr) {
                        return frame.get_value(inst->operand(0));
                    }
                    return RuntimeValue::from_void();
                }

                case Opcode::unreachable: {
                    throw InterpreterException("Execution reached unreachable instruction");
                }

                case Opcode::throw_: {
                    RuntimeValue val = frame.get_value(inst->operand(0));
                    current_exception_ = val;
                    throw InterpreterThrownException(val);
                }

                case Opcode::resume: {
                    RuntimeValue val = (inst->operand_count() > 0 && inst->operand(0))
                        ? frame.get_value(inst->operand(0))
                        : current_exception_;
                    current_exception_ = val;
                    throw InterpreterThrownException(val);
                }

                case Opcode::landing_pad: {
                    frame.set_value(inst->result(), current_exception_);
                    break;
                }

                case Opcode::coro_create: {
                    RuntimeValue res = interp_coro_create(*inst, frame, module_);
                    if (inst->result()) {
                        frame.set_value(inst->result(), res);
                    }
                    break;
                }

                case Opcode::coro_suspend: {
                    interp_coro_suspend(*inst, frame, cur_bb);
                    break;
                }

                case Opcode::coro_resume: {
                    RuntimeValue res = interp_coro_resume(
                        *inst, frame, module_,
                        [this](const Function& f, const std::vector<RuntimeValue>& a) {
                            return execute_function(f, a);
                        },
                        [this](const Function& f, BasicBlock* bb, const std::vector<RuntimeValue>& a) {
                            return execute_function_from_block(f, bb, a);
                        }
                    );
                    if (inst->result()) {
                        frame.set_value(inst->result(), res);
                    }
                    break;
                }

                case Opcode::coro_destroy: {
                    interp_coro_destroy(*inst, frame);
                    break;
                }

                case Opcode::invoke: {
                    std::string_view callee = inst->symbol();
                    std::vector<RuntimeValue> call_args;
                    call_args.reserve(inst->operand_count());
                    for (size_t i = 0; i < inst->operand_count(); ++i) {
                        call_args.push_back(frame.get_value(inst->operand(i)));
                    }

                    RuntimeValue call_res;
                    bool threw = false;
                    try {
                        const Function* target_fn = module_ ? module_->get_function(callee) : nullptr;
                        if (target_fn) {
                            call_res = execute_function(*target_fn, call_args);
                        } else {
                            auto it = external_functions_.find(std::string(callee));
                            if (it != external_functions_.end()) {
                                call_res = it->second(*this, call_args);
                            } else {
                                throw InterpreterException("Invoke to undefined function: " + std::string(callee));
                            }
                        }
                    } catch (const InterpreterThrownException& ex) {
                        threw = true;
                        current_exception_ = ex.value();
                    }

                    if (!threw) {
                        if (inst->result() && !inst->type().is_void()) {
                            frame.set_value(inst->result(), call_res);
                        }

                        const BranchTarget& target = inst->normal_target();
                        std::vector<RuntimeValue> target_args;
                        target_args.reserve(target.args.size());
                        for (Value* a : target.args) {
                            target_args.push_back(frame.get_value(a));
                        }

                        BasicBlock* next_bb = target.block;
                        if (!next_bb) {
                            throw InterpreterException("Invoke normal branch to null basic block");
                        }

                        for (size_t i = 0; i < next_bb->param_count() && i < target_args.size(); ++i) {
                            frame.set_value(next_bb->param(i), target_args[i]);
                        }

                        cur_bb = next_bb;
                        transitioned = true;
                    } else {
                        const BranchTarget& target = inst->unwind_target();
                        std::vector<RuntimeValue> target_args;
                        target_args.reserve(target.args.size());
                        for (Value* a : target.args) {
                            target_args.push_back(frame.get_value(a));
                        }

                        BasicBlock* next_bb = target.block;
                        if (!next_bb) {
                            throw InterpreterException("Invoke unwind branch to null basic block");
                        }

                        for (size_t i = 0; i < next_bb->param_count() && i < target_args.size(); ++i) {
                            frame.set_value(next_bb->param(i), target_args[i]);
                        }

                        cur_bb = next_bb;
                        transitioned = true;
                    }
                    break;
                }
            }

            if (transitioned) {
                break;
            }
        }

        if (!transitioned) {
            // Block ended without a terminator (malformed MIR or fallen off end)
            break;
        }
    }

    return RuntimeValue::from_void();
}

} // namespace brass
