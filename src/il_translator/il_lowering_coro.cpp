#include "il_lowering_coro.hpp"
#include "il_lowering.hpp"
#include <brass/runtime/coroutine.hpp>
#include <brass/mir/module.hpp>

namespace brass::il {

bool is_coro_il_op(BronzeOp op) {
    switch (op) {
        case BronzeOp::CreateAsyncMachine:
        case BronzeOp::AsyncStart:
        case BronzeOp::AsyncAwait:
        case BronzeOp::IterOpen:
        case BronzeOp::IterStep:
        case BronzeOp::IterValue:
        case BronzeOp::IterClose:
        case BronzeOp::IterRest:
        case BronzeOp::IterDelegate:
        case BronzeOp::AsyncIterOpen:
        case BronzeOp::AsyncIterNext:
        case BronzeOp::AsyncIterClose:
        case BronzeOp::Yield:
            return true;
        default:
            return false;
    }
}

bool lower_coro_instruction(
    IlLowering* lowering,
    const BronzeInstruction& inst_ast,
    Builder& b,
    Function* fn,
    std::unordered_map<uint32_t, Value*>& val_map,
    Value*& res_val
) {
    auto get_opd = [&](size_t idx) -> Value* {
        if (idx < inst_ast.operands.size()) {
            uint32_t oid = inst_ast.operands[idx];
            if (lowering) {
                return lowering->get_val_by_id(oid, b, val_map);
            }
            if (val_map.count(oid)) return val_map[oid];
        }
        return nullptr;
    };

    switch (inst_ast.op) {
        case BronzeOp::CreateAsyncMachine: {
            uint32_t slot_count = inst_ast.param_count ? inst_ast.param_count : 16;
            Value* slot_count_val = b.build_iconst_i32(static_cast<int32_t>(slot_count));
            uint64_t ptr_mask = (1ULL << 5); // slot 0 holds env
            Value* mask_val = b.build_iconst_i64(static_cast<int64_t>(ptr_mask));
            Value* env_val = get_opd(0);
            if (!env_val) {
                env_val = b.build_iconst_i64(0);
            }

            std::string callee = inst_ast.callee_name;
            const char* name_ptr = fn->parent()->string_pool().intern(callee).data();
            Value* fn_name_val = b.build_iconst_i64(static_cast<int64_t>(reinterpret_cast<uintptr_t>(name_ptr)));

            res_val = b.build_call("bronze_create_async_machine", Type::i64(), {
                fn_name_val,
                slot_count_val,
                mask_val,
                env_val
            });
            return true;
        }

        case BronzeOp::AsyncStart: {
            Value* mach = get_opd(0);
            Value* arg = get_opd(1);
            if (!arg) arg = b.build_iconst_i64(0);
            res_val = b.build_call("bronze_async_start", Type::i64(), {mach, arg});
            return true;
        }

        case BronzeOp::AsyncAwait: {
            Value* mach = get_opd(0);
            Value* val = get_opd(1);
            if (!val) val = b.build_iconst_i64(0);
            res_val = b.build_call("bronze_async_await", Type::i64(), {mach, val});
            return true;
        }

        case BronzeOp::IterOpen: {
            Value* gen = get_opd(0);
            res_val = b.build_call("bronze_iter_open", Type::i64(), {gen});
            return true;
        }

        case BronzeOp::IterStep: {
            Value* iter = get_opd(0);
            res_val = b.build_and(b.build_call("bronze_iter_step", Type::i32(), {iter}), b.build_iconst_i32(1));
            return true;
        }

        case BronzeOp::IterValue: {
            Value* iter = get_opd(0);
            res_val = b.build_call("bronze_iter_value", Type::i64(), {iter});
            return true;
        }

        case BronzeOp::IterClose: {
            Value* iter = get_opd(0);
            Value* suppress = b.build_iconst_i32(inst_ast.imm_i64 != 0 ? 1 : 0);
            b.build_call("bronze_iter_close", Type::void_type(), {iter, suppress});
            return true;
        }

        case BronzeOp::IterRest: {
            Value* iter = get_opd(0);
            res_val = b.build_call("bronze_iter_rest", Type::i64(), {iter});
            return true;
        }

        case BronzeOp::IterDelegate: {
            Value* iter = get_opd(0);
            Value* mode = get_opd(1);
            Value* sent = get_opd(2);
            res_val = b.build_call("bronze_iter_delegate", Type::i64(), {iter, mode, sent});
            return true;
        }

        case BronzeOp::AsyncIterOpen: {
            Value* iter = get_opd(0);
            res_val = b.build_call("bronze_async_iter_open", Type::i64(), {iter});
            return true;
        }

        case BronzeOp::AsyncIterNext: {
            Value* iter = get_opd(0);
            res_val = b.build_call("bronze_async_iter_next", Type::i64(), {iter});
            return true;
        }

        case BronzeOp::AsyncIterClose: {
            Value* iter = get_opd(0);
            Value* suppress = b.build_iconst_i32(inst_ast.imm_i64 != 0 ? 1 : 0);
            b.build_call("bronze_async_iter_close", Type::void_type(), {iter, suppress});
            return true;
        }

        case BronzeOp::Yield: {
            Value* yield_val = get_opd(0);
            res_val = b.build_coro_suspend(yield_val, 0, Type::i64());
            return true;
        }

        default:
            return false;
    }
}

} // namespace brass::il
