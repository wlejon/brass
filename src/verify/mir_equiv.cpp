#include <brass/verify/smt_equiv.hpp>
#include "bv_solver.hpp"
#include <brass/mir/module.hpp>
#include <brass/mir/function.hpp>
#include <brass/mir/block.hpp>
#include <brass/mir/instruction.hpp>
#include <brass/mir/opcodes.hpp>
#include <brass/interpreter/narrow_int.hpp>
#include <algorithm>
#include <sstream>
#include <unordered_map>
#include <unordered_set>

namespace brass::verify {

namespace {

uint32_t type_bv_width(Type t) noexcept {
    switch (t.kind()) {
        case TypeKind::I8: return 8;
        case TypeKind::I16: return 16;
        case TypeKind::I32: case TypeKind::F32: return 32;
        case TypeKind::I64: case TypeKind::F64: case TypeKind::Ptr:
        case TypeKind::GCRef: case TypeKind::Tagged: return 64;
        default: return 0;
    }
}

struct SymStore {
    BVNodeId guard = 0, addr = 0;
    uint32_t num_bytes = 0;
    BVNodeId val = 0;
    bool is_local_alloca = false;
};

struct SymGuardExit {
    BVNodeId fail_cond = 0;
    uint32_t resume_id = 0;
    std::string exit_symbol;
    std::vector<Type> state_types;
    std::vector<BVNodeId> state_vals;
};

struct InitMemEnv {
    struct ByteEntry { BVNodeId addr, val; };
    std::vector<ByteEntry> bytes;
    struct CallEntry { std::string sym; std::vector<BVNodeId> args; BVNodeId ret_val; };
    std::vector<CallEntry> calls;

    BVNodeId read_init_byte(BVContext& ctx, BVNodeId byte_addr) {
        for (const auto& e : bytes) if (e.addr == byte_addr) return e.val;
        BVNodeId v = ctx.make_var("mem_b" + std::to_string(bytes.size()), 8, false);
        for (const auto& e : bytes) v = ctx.make_ite(ctx.make_binary(BVOp::Eq, byte_addr, e.addr, 1), e.val, v);
        bytes.push_back({byte_addr, v});
        return v;
    }

    BVNodeId eval_uninterpreted_call(BVContext& ctx, std::string_view sym,
                                     const std::vector<BVNodeId>& args, Type ret_type) {
        uint32_t w = type_bv_width(ret_type);
        if (w == 0) return 0;
        for (const auto& c : calls) {
            if (c.sym == sym && c.args == args && ctx.width(c.ret_val) == w) return c.ret_val;
        }
        BVNodeId v = ctx.make_var("call_" + std::string(sym) + "_" + std::to_string(calls.size()),
                                  w, ret_type.is_float());
        for (const auto& c : calls) {
            if (c.sym == sym && c.args.size() == args.size() && ctx.width(c.ret_val) == w) {
                BVNodeId all_eq = ctx.make_const(1, 1);
                for (size_t i = 0; i < args.size(); ++i) {
                    if (ctx.width(args[i]) != ctx.width(c.args[i])) {
                        all_eq = ctx.make_const(0, 1);
                        break;
                    }
                    all_eq = ctx.make_binary(BVOp::And, all_eq,
                                             ctx.make_binary(BVOp::Eq, args[i], c.args[i], 1), 1);
                }
                v = ctx.make_ite(all_eq, c.ret_val, v);
            }
        }
        calls.push_back({std::string(sym), args, v});
        return v;
    }
};

struct SymExecOutput {
    bool supported = true;
    bool has_type_error = false;
    std::string error_msg;
    BVNodeId completed = 0;
    BVNodeId guard_failed = 0;
    BVNodeId trapped = 0;
    BVNodeId returned = 0;
    BVNodeId ret_val = 0;
    std::vector<SymStore> stores;
    std::vector<SymGuardExit> guard_exits;
};

BVNodeId read_byte_at(BVContext& ctx, InitMemEnv& init_mem,
                      const std::vector<SymStore>& stores, BVNodeId byte_addr) {
    BVNodeId val = init_mem.read_init_byte(ctx, byte_addr);
    for (const auto& st : stores) {
        for (uint32_t b = 0; b < st.num_bytes; ++b) {
            BVNodeId st_b_addr = (b == 0)
                ? st.addr
                : ctx.make_binary(BVOp::Add, st.addr, ctx.make_const(b, 64), 64);
            BVNodeId hit = ctx.make_binary(BVOp::And, st.guard,
                                           ctx.make_binary(BVOp::Eq, byte_addr, st_b_addr, 1), 1);
            BVNodeId st_byte = (st.num_bytes == 1) ? st.val : ctx.make_extract(st.val, b * 8 + 7, b * 8);
            val = ctx.make_ite(hit, st_byte, val);
        }
    }
    return val;
}

BVNodeId read_memory_bv(BVContext& ctx, InitMemEnv& init_mem,
                        const std::vector<SymStore>& stores, BVNodeId addr, uint32_t num_bytes) {
    BVNodeId cur = 0;
    for (uint32_t b = 0; b < num_bytes; ++b) {
        BVNodeId b_addr = (b == 0) ? addr : ctx.make_binary(BVOp::Add, addr, ctx.make_const(b, 64), 64);
        BVNodeId byte_val = read_byte_at(ctx, init_mem, stores, b_addr);
        cur = (b == 0) ? byte_val : ctx.make_concat(byte_val, cur);
    }
    return cur;
}

void compute_rpo(const BasicBlock* bb,
                 std::unordered_set<const BasicBlock*>& visited,
                 std::vector<const BasicBlock*>& postorder) {
    if (!bb || !visited.insert(bb).second) return;
    if (const Instruction* term = bb->terminator()) {
        switch (term->opcode()) {
            case Opcode::br:
                compute_rpo(term->branch_target().block, visited, postorder);
                break;
            case Opcode::br_if:
                compute_rpo(term->false_target().block, visited, postorder);
                compute_rpo(term->true_target().block, visited, postorder);
                break;
            case Opcode::switch_:
                compute_rpo(term->default_target().block, visited, postorder);
                for (const auto& sc : term->switch_cases()) compute_rpo(sc.target.block, visited, postorder);
                break;
            default: break;
        }
    }
    postorder.push_back(bb);
}

BVNodeId coerce_width(BVContext& ctx, BVNodeId node, uint32_t target_w, bool sign_ext = false) {
    uint32_t w = ctx.width(node);
    if (w == target_w) return node;
    if (w > target_w) return ctx.make_extract(node, target_w - 1, 0);
    return sign_ext ? ctx.make_sext(node, target_w - w) : ctx.make_zext(node, target_w - w);
}

SymExecOutput symbolically_execute_mir(
    BVContext& ctx,
    const Function& fn,
    const std::vector<BVNodeId>& param_vars,
    InitMemEnv& init_mem,
    const EquivOptions& opts) {

    SymExecOutput out;
    out.completed = ctx.make_const(0, 1);
    out.guard_failed = ctx.make_const(0, 1);
    out.trapped = ctx.make_const(0, 1);
    out.returned = ctx.make_const(0, 1);
    uint32_t ret_w = type_bv_width(fn.return_type());
    out.ret_val = ret_w > 0 ? ctx.make_const(0, ret_w) : 0;

    if (!fn.entry_block()) {
        out.supported = false;
        out.error_msg = "Function has no entry block";
        return out;
    }
    if (!fn.return_type().is_void() && ret_w == 0) {
        out.supported = false;
        out.error_msg = "Unsupported return type in MIR equivalence checker";
        return out;
    }
    for (size_t i = 0; i < fn.param_count(); ++i) {
        if (type_bv_width(fn.param_type(i)) == 0) {
            out.supported = false;
            out.error_msg = "Unsupported parameter type in MIR equivalence checker";
            return out;
        }
    }

    std::unordered_set<const BasicBlock*> visited;
    std::vector<const BasicBlock*> rpo;
    compute_rpo(fn.entry_block(), visited, rpo);
    std::reverse(rpo.begin(), rpo.end());

    std::unordered_map<const BasicBlock*, size_t> rpo_pos;
    for (size_t i = 0; i < rpo.size(); ++i) rpo_pos[rpo[i]] = i;

    struct IncomingEdge {
        BVNodeId cond = 0;
        std::unordered_map<const Value*, BVNodeId> vals;
    };

    const uint32_t max_level = opts.loop_unroll_bound;
    std::vector<std::vector<IncomingEdge>> incoming((max_level + 1) * rpo.size());

    {
        IncomingEdge seed;
        seed.cond = ctx.make_const(1, 1);
        const BasicBlock* entry = fn.entry_block();
        if (entry->param_count() != param_vars.size()) {
            out.has_type_error = true;
            out.error_msg = "Entry block parameter count mismatch";
            return out;
        }
        for (size_t i = 0; i < entry->param_count(); ++i) {
            if (entry->param(i)->type() != fn.param_type(i)) {
                out.has_type_error = true;
                out.error_msg = "Entry block parameter type mismatch";
                return out;
            }
            seed.vals[entry->param(i)] = param_vars[i];
        }
        incoming[0].push_back(std::move(seed));
    }

    uint64_t next_alloca_addr = 0x7FFF000000000000ULL;
    std::unordered_set<uint64_t> alloca_bases;
    BVNodeId tls_reg_val = ctx.make_const(0x7FFE000000000000ULL, 64);
    BVNodeId sp_reg_val = ctx.make_const(0x7FFD000000000000ULL, 64);

    auto route_branch = [&](size_t src_bi, uint32_t level, BVNodeId edge_cond,
                            const BranchTarget& target,
                            const std::unordered_map<const Value*, BVNodeId>& cur_vals) {
        if (ctx.nodes()[edge_cond].op == BVOp::Const && ctx.nodes()[edge_cond].imm == 0) return;
        auto it = rpo_pos.find(target.block);
        if (it == rpo_pos.end()) return;
        size_t dst_bi = it->second;
        const BasicBlock* dst = target.block;

        std::unordered_map<const Value*, BVNodeId> next_vals = cur_vals;
        if (target.args.size() != dst->param_count()) {
            out.has_type_error = true;
            out.error_msg = "Branch argument count mismatch";
        }
        size_t n_args = std::min(target.args.size(), dst->param_count());
        for (size_t i = 0; i < n_args; ++i) {
            const Value* arg_v = target.args[i];
            const Value* param_v = dst->param(i);
            if (!arg_v || !param_v || arg_v->type() != param_v->type()) {
                out.has_type_error = true;
                out.error_msg = "Branch argument type mismatch";
            }
            auto vit = cur_vals.find(arg_v);
            if (vit != cur_vals.end()) next_vals[param_v] = vit->second;
        }

        uint32_t dst_level = (dst_bi <= src_bi) ? (level + 1) : level;
        if (dst_level > max_level) return;
        incoming[dst_level * rpo.size() + dst_bi].push_back({edge_cond, std::move(next_vals)});
    };

    for (uint32_t level = 0; level <= max_level; ++level) {
        for (size_t bi = 0; bi < rpo.size(); ++bi) {
            auto& edges = incoming[level * rpo.size() + bi];
            if (edges.empty()) continue;

            BVNodeId cur_cond = edges[0].cond;
            std::unordered_map<const Value*, BVNodeId> vals = edges[0].vals;
            for (size_t ei = 1; ei < edges.size(); ++ei) {
                BVNodeId ec = edges[ei].cond;
                cur_cond = ctx.make_binary(BVOp::Or, cur_cond, ec, 1);
                for (const auto& [val_ptr, node_id] : edges[ei].vals) {
                    auto vit = vals.find(val_ptr);
                    if (vit == vals.end()) {
                        vals[val_ptr] = node_id;
                    } else if (vit->second != node_id) {
                        uint32_t w = ctx.width(vit->second);
                        vit->second = ctx.make_ite(ec, coerce_width(ctx, node_id, w, false), vit->second);
                    }
                }
            }
            if (ctx.nodes()[cur_cond].op == BVOp::Const && ctx.nodes()[cur_cond].imm == 0) continue;

            for (const Instruction* inst = rpo[bi]->head(); inst != nullptr; inst = inst->next()) {
                Opcode op = inst->opcode();
                uint32_t res_w = type_bv_width(inst->type());

                auto get_op = [&](size_t idx) -> BVNodeId {
                    const Value* v = inst->operand(idx);
                    if (!v) {
                        out.has_type_error = true;
                        out.error_msg = "Null instruction operand";
                        return ctx.make_const(0, 64);
                    }
                    auto it = vals.find(v);
                    if (it == vals.end()) {
                        out.has_type_error = true;
                        out.error_msg = "Use of undefined/undominated SSA value";
                        uint32_t w = type_bv_width(v->type());
                        return ctx.make_const(0, w ? w : 64);
                    }
                    return it->second;
                };
                auto set_res = [&](BVNodeId id) {
                    if (inst->result()) vals[inst->result()] = id;
                };

                switch (op) {
                    case Opcode::iconst_i32:
                        if (inst->type() != Type::i32()) out.has_type_error = true;
                        set_res(ctx.make_const(static_cast<uint32_t>(inst->imm_i32()), 32));
                        break;
                    case Opcode::iconst_i64:
                        if (inst->type() != Type::i64()) out.has_type_error = true;
                        set_res(ctx.make_const(static_cast<uint64_t>(inst->imm_i64()), 64));
                        break;
                    case Opcode::fconst_f64: {
                        if (inst->type() == Type::f32()) {
                            float f = static_cast<float>(inst->imm_f64());
                            uint32_t bits = 0;
                            std::memcpy(&bits, &f, 4);
                            set_res(ctx.make_const(bits, 32));
                        } else {
                            if (inst->type() != Type::f64()) out.has_type_error = true;
                            uint64_t bits = 0;
                            double d = inst->imm_f64();
                            std::memcpy(&bits, &d, 8);
                            set_res(ctx.make_const(bits, 64));
                        }
                        break;
                    }
                    case Opcode::patchable_const_i32: set_res(ctx.make_const(static_cast<uint32_t>(inst->imm_i32()), 32)); break;
                    case Opcode::patchable_const_i64: set_res(ctx.make_const(static_cast<uint64_t>(inst->imm_i64()), 64)); break;
                    case Opcode::sext_i64:
                        if (!inst->operand(0) || inst->operand(0)->type() != Type::i32() || inst->type() != Type::i64())
                            out.has_type_error = true;
                        set_res(ctx.make_sext(coerce_width(ctx, get_op(0), 32, true), 32));
                        break;
                    case Opcode::zext_i64: {
                        BVNodeId a = get_op(0);
                        if (!inst->operand(0) ||
                            (inst->operand(0)->type() != Type::i32() && inst->operand(0)->type() != Type::i8()) ||
                            inst->type() != Type::i64())
                            out.has_type_error = true;
                        uint32_t w = ctx.width(a);
                        set_res(w < 64 ? ctx.make_zext(a, 64 - w) : coerce_width(ctx, a, 64, false));
                        break;
                    }
                    case Opcode::trunc_i32:
                        if (!inst->operand(0) || inst->operand(0)->type() != Type::i64() || inst->type() != Type::i32())
                            out.has_type_error = true;
                        set_res(coerce_width(ctx, get_op(0), 32, false));
                        break;
                    case Opcode::trunc_i8:
                        if (!inst->operand(0) || !inst->operand(0)->type().is_integer() || inst->type() != Type::i8())
                            out.has_type_error = true;
                        set_res(coerce_width(ctx, get_op(0), 8, false));
                        break;
                    case Opcode::fptosi_i32:
                    case Opcode::fptosi_i32_f32: set_res(ctx.make_unary(BVOp::FpToSi, get_op(0), 32)); break;
                    case Opcode::fptosi_i64:
                    case Opcode::fptosi_i64_f32: set_res(ctx.make_unary(BVOp::FpToSi, get_op(0), 64)); break;
                    case Opcode::sitofp_f64_i32:
                    case Opcode::sitofp_f64_i64: set_res(ctx.make_unary(BVOp::SiToFp, get_op(0), 64)); break;
                    case Opcode::sitofp_f32_i32:
                    case Opcode::sitofp_f32_i64: set_res(ctx.make_unary(BVOp::SiToFp, get_op(0), 32)); break;
                    case Opcode::fptrunc_f32_f64: set_res(ctx.make_unary(BVOp::FpTrunc, get_op(0), 32)); break;
                    case Opcode::fpext_f64_f32:   set_res(ctx.make_unary(BVOp::FpExt, get_op(0), 64)); break;
                    case Opcode::bitcast_i64_f64:
                    case Opcode::bitcast_f64_i64:
                    case Opcode::bitcast_i64_tagged:
                    case Opcode::bitcast_tagged_i64: set_res(coerce_width(ctx, get_op(0), 64, false)); break;
                    case Opcode::bitcast_i32_f32:
                    case Opcode::bitcast_f32_i32:    set_res(coerce_width(ctx, get_op(0), 32, false)); break;

                    case Opcode::add:
                    case Opcode::sub:
                    case Opcode::mul: {
                        BVNodeId a = get_op(0), b = get_op(1);
                        Type t0 = inst->operand(0) ? inst->operand(0)->type() : Type::void_type();
                        Type t1 = inst->operand(1) ? inst->operand(1)->type() : Type::void_type();
                        if (t0.is_pointer_or_gcref() && t1.is_integer() && (op == Opcode::add || op == Opcode::sub)) {
                            b = coerce_width(ctx, b, 64, true);
                            set_res(ctx.make_binary(op == Opcode::add ? BVOp::Add : BVOp::Sub, a, b, 64));
                        } else if (op == Opcode::sub && t0.is_pointer_or_gcref() && t0 == t1) {
                            set_res(ctx.make_binary(BVOp::Sub, a, b, 64));
                        } else {
                            if (t0 != t1 || inst->type() != t0 || res_w == 0) out.has_type_error = true;
                            uint32_t w = res_w ? res_w : 64;
                            a = coerce_width(ctx, a, w, false);
                            b = coerce_width(ctx, b, w, false);
                            BVOp bop = t0.is_float()
                                ? (op == Opcode::add ? BVOp::FAdd : op == Opcode::sub ? BVOp::FSub : BVOp::FMul)
                                : (op == Opcode::add ? BVOp::Add  : op == Opcode::sub ? BVOp::Sub  : BVOp::Mul);
                            set_res(ctx.make_binary(bop, a, b, w));
                        }
                        break;
                    }

                    case Opcode::sdiv:
                    case Opcode::udiv:
                    case Opcode::smod:
                    case Opcode::umod: {
                        BVNodeId a = get_op(0), b = get_op(1);
                        Type t0 = inst->operand(0) ? inst->operand(0)->type() : Type::void_type();
                        Type t1 = inst->operand(1) ? inst->operand(1)->type() : Type::void_type();
                        if (t0 != t1 || inst->type() != t0 || res_w == 0) out.has_type_error = true;
                        uint32_t w = res_w ? res_w : 64;
                        a = coerce_width(ctx, a, w, false);
                        b = coerce_width(ctx, b, w, false);
                        if (t0.is_float()) {
                            set_res(ctx.make_binary(op == Opcode::sdiv ? BVOp::FDiv : BVOp::FSub, a, b, w));
                        } else {
                            BVNodeId is_zero = ctx.make_binary(BVOp::Eq, b, ctx.make_const(0, w), 1);
                            BVNodeId trap_now = ctx.make_binary(BVOp::And, cur_cond, is_zero, 1);
                            out.trapped = ctx.make_binary(BVOp::Or, out.trapped, trap_now, 1);
                            out.completed = ctx.make_binary(BVOp::Or, out.completed, trap_now, 1);
                            cur_cond = ctx.make_binary(BVOp::And, cur_cond, ctx.make_unary(BVOp::Not, is_zero, 1), 1);
                            BVOp iop = (op == Opcode::sdiv) ? BVOp::SDiv : (op == Opcode::udiv) ? BVOp::UDiv
                                     : (op == Opcode::smod) ? BVOp::SRem : BVOp::URem;
                            set_res(ctx.make_binary(iop, a, b, w));
                        }
                        break;
                    }

                    case Opcode::neg: {
                        BVNodeId a = get_op(0);
                        uint32_t w = res_w ? res_w : ctx.width(a);
                        a = coerce_width(ctx, a, w, false);
                        set_res(ctx.make_unary(inst->type().is_float() ? BVOp::FNeg : BVOp::Neg, a, w));
                        break;
                    }

                    case Opcode::fabs_f32: case Opcode::fabs_f64:
                    case Opcode::sqrt_f32: case Opcode::sqrt_f64:
                    case Opcode::floor_f32: case Opcode::floor_f64:
                    case Opcode::ceil_f32: case Opcode::ceil_f64:
                    case Opcode::round_f32: case Opcode::round_f64: {
                        uint32_t w = (op == Opcode::fabs_f32 || op == Opcode::sqrt_f32 || op == Opcode::floor_f32 ||
                                      op == Opcode::ceil_f32 || op == Opcode::round_f32) ? 32 : 64;
                        BVOp uop = (op == Opcode::fabs_f32 || op == Opcode::fabs_f64) ? BVOp::FAbs
                                 : (op == Opcode::sqrt_f32 || op == Opcode::sqrt_f64) ? BVOp::FSqrt
                                 : (op == Opcode::floor_f32 || op == Opcode::floor_f64) ? BVOp::FFloor
                                 : (op == Opcode::ceil_f32 || op == Opcode::ceil_f64) ? BVOp::FCeil : BVOp::FRound;
                        set_res(ctx.make_unary(uop, coerce_width(ctx, get_op(0), w, false), w));
                        break;
                    }
                    case Opcode::fmin_f32: case Opcode::fmin_f64:
                    case Opcode::fmax_f32: case Opcode::fmax_f64: {
                        uint32_t w = (op == Opcode::fmin_f32 || op == Opcode::fmax_f32) ? 32 : 64;
                        BVOp bop = (op == Opcode::fmin_f32 || op == Opcode::fmin_f64) ? BVOp::FMin : BVOp::FMax;
                        set_res(ctx.make_binary(bop, coerce_width(ctx, get_op(0), w, false),
                                                coerce_width(ctx, get_op(1), w, false), w));
                        break;
                    }
                    case Opcode::and_: case Opcode::or_: case Opcode::xor_:
                    case Opcode::shl:  case Opcode::lshr: case Opcode::ashr: {
                        BVNodeId a = get_op(0), b = get_op(1);
                        Type t0 = inst->operand(0) ? inst->operand(0)->type() : Type::void_type();
                        Type t1 = inst->operand(1) ? inst->operand(1)->type() : Type::void_type();
                        if (!t0.is_integer() || !t1.is_integer() || inst->type() != t0 || res_w == 0)
                            out.has_type_error = true;
                        if ((op == Opcode::and_ || op == Opcode::or_ || op == Opcode::xor_) && t0 != t1)
                            out.has_type_error = true;
                        uint32_t w = res_w ? res_w : 64;
                        BVOp bv_op = (op == Opcode::and_) ? BVOp::And : (op == Opcode::or_)  ? BVOp::Or
                                   : (op == Opcode::xor_) ? BVOp::Xor : (op == Opcode::shl)  ? BVOp::Shl
                                   : (op == Opcode::lshr) ? BVOp::LShr : BVOp::AShr;
                        set_res(ctx.make_binary(bv_op, coerce_width(ctx, a, w, false),
                                                coerce_width(ctx, b, w, false), w));
                        break;
                    }
                    case Opcode::not_: case Opcode::clz: case Opcode::ctz: case Opcode::popcnt: {
                        BVNodeId a = get_op(0);
                        uint32_t w = res_w ? res_w : ctx.width(a);
                        BVOp bv_op = (op == Opcode::not_) ? BVOp::Not : (op == Opcode::clz) ? BVOp::Clz
                                   : (op == Opcode::ctz)  ? BVOp::Ctz : BVOp::Popcnt;
                        set_res(ctx.make_unary(bv_op, coerce_width(ctx, a, w, false), w));
                        break;
                    }
                    case Opcode::eq: case Opcode::ne:
                    case Opcode::slt: case Opcode::ult: case Opcode::sle: case Opcode::ule:
                    case Opcode::sgt: case Opcode::ugt: case Opcode::sge: case Opcode::uge: {
                        BVNodeId a = get_op(0), b = get_op(1);
                        Type t0 = inst->operand(0) ? inst->operand(0)->type() : Type::void_type();
                        Type t1 = inst->operand(1) ? inst->operand(1)->type() : Type::void_type();
                        if (t0 != t1 || inst->type() != Type::i32()) out.has_type_error = true;
                        uint32_t w = type_bv_width(t0);
                        if (w == 0) w = 64;
                        a = coerce_width(ctx, a, w, false);
                        b = coerce_width(ctx, b, w, false);
                        BVOp bop = t0.is_float()
                            ? ((op == Opcode::eq) ? BVOp::FEq : (op == Opcode::ne) ? BVOp::FNe
                               : (op == Opcode::slt || op == Opcode::ult) ? BVOp::FLt
                               : (op == Opcode::sle || op == Opcode::ule) ? BVOp::FLe
                               : (op == Opcode::sgt || op == Opcode::ugt) ? BVOp::FGt : BVOp::FGe)
                            : ((op == Opcode::eq)  ? BVOp::Eq  : (op == Opcode::ne)  ? BVOp::Ne
                               : (op == Opcode::slt) ? BVOp::Slt : (op == Opcode::ult) ? BVOp::Ult
                               : (op == Opcode::sle) ? BVOp::Sle : (op == Opcode::ule) ? BVOp::Ule
                               : (op == Opcode::sgt) ? BVOp::Sgt : (op == Opcode::ugt) ? BVOp::Ugt
                               : (op == Opcode::sge) ? BVOp::Sge : BVOp::Uge);
                        set_res(ctx.make_zext(ctx.make_binary(bop, a, b, 1), 31));
                        break;
                    }
                    case Opcode::sadd_overflow: case Opcode::ssub_overflow: case Opcode::smul_overflow:
                    case Opcode::uadd_overflow: case Opcode::usub_overflow: case Opcode::umul_overflow: {
                        BVNodeId a = get_op(0), b = get_op(1);
                        Type t0 = inst->operand(0) ? inst->operand(0)->type() : Type::void_type();
                        Type t1 = inst->operand(1) ? inst->operand(1)->type() : Type::void_type();
                        if (t0 != t1 || !t0.is_integer() || inst->type() != Type::i32()) out.has_type_error = true;
                        uint32_t w = type_bv_width(t0);
                        if (w == 0) w = 32;
                        BVOp ovf_op = (op == Opcode::sadd_overflow) ? BVOp::SAddOvf
                                    : (op == Opcode::ssub_overflow) ? BVOp::SSubOvf
                                    : (op == Opcode::smul_overflow) ? BVOp::SMulOvf
                                    : (op == Opcode::uadd_overflow) ? BVOp::UAddOvf
                                    : (op == Opcode::usub_overflow) ? BVOp::USubOvf : BVOp::UMulOvf;
                        set_res(ctx.make_zext(ctx.make_binary(ovf_op, coerce_width(ctx, a, w, false),
                                                              coerce_width(ctx, b, w, false), 1), 31));
                        break;
                    }

                    case Opcode::select: {
                        BVNodeId cond_v = get_op(0), t_v = get_op(1), f_v = get_op(2);
                        if (!inst->operand(0) || inst->operand(0)->type() != Type::i32() ||
                            !inst->operand(1) || !inst->operand(2) ||
                            inst->operand(1)->type() != inst->operand(2)->type() ||
                            inst->type() != inst->operand(1)->type())
                            out.has_type_error = true;
                        uint32_t cw = ctx.width(cond_v);
                        BVNodeId cond_bit = ctx.make_binary(BVOp::Ne, cond_v, ctx.make_const(0, cw), 1);
                        uint32_t w = res_w ? res_w : ctx.width(t_v);
                        set_res(ctx.make_ite(cond_bit, coerce_width(ctx, t_v, w, false), coerce_width(ctx, f_v, w, false)));
                        break;
                    }

                    case Opcode::alloca_: {
                        uint64_t addr = next_alloca_addr;
                        uint64_t sz = inst->imm_i32() > 0 ? static_cast<uint64_t>(inst->imm_i32()) : 16ULL;
                        next_alloca_addr += (sz + 15ULL) & ~15ULL;
                        alloca_bases.insert(addr);
                        set_res(ctx.make_const(addr, 64));
                        break;
                    }
                    case Opcode::pinned_tls_read:  set_res(tls_reg_val); break;
                    case Opcode::pinned_tls_write: tls_reg_val = coerce_width(ctx, get_op(0), 64, false); break;
                    case Opcode::read_sp:          set_res(sp_reg_val); break;

                    case Opcode::load:
                    case Opcode::load_indexed: {
                        BVNodeId addr = coerce_width(ctx, get_op(0), 64, false);
                        if (op == Opcode::load_indexed) {
                            Type idx_t = inst->operand(1) ? inst->operand(1)->type() : Type::i64();
                            BVNodeId idx64 = coerce_width(ctx, get_op(1), 64, idx_t.is_integer());
                            if (inst->scale() > 1)
                                idx64 = ctx.make_binary(BVOp::Mul, idx64, ctx.make_const(inst->scale(), 64), 64);
                            addr = ctx.make_binary(BVOp::Add, addr, idx64, 64);
                        }
                        if (inst->offset() != 0)
                            addr = ctx.make_binary(BVOp::Add, addr,
                                ctx.make_const(static_cast<uint64_t>(static_cast<int64_t>(inst->offset())), 64), 64);
                        uint32_t num_bytes = static_cast<uint32_t>(inst->type().size_in_bytes());
                        if (num_bytes == 0 || num_bytes > 8) {
                            out.supported = false;
                            out.error_msg = "Unsupported load width";
                            return out;
                        }
                        set_res(read_memory_bv(ctx, init_mem, out.stores, addr, num_bytes));
                        break;
                    }

                    case Opcode::store:
                    case Opcode::store_indexed: {
                        BVNodeId base = coerce_width(ctx, get_op(0), 64, false);
                        BVNodeId addr = base;
                        BVNodeId val = (op == Opcode::store) ? get_op(1) : get_op(2);
                        if (op == Opcode::store_indexed) {
                            Type idx_t = inst->operand(1) ? inst->operand(1)->type() : Type::i64();
                            BVNodeId idx64 = coerce_width(ctx, get_op(1), 64, idx_t.is_integer());
                            if (inst->scale() > 1)
                                idx64 = ctx.make_binary(BVOp::Mul, idx64, ctx.make_const(inst->scale(), 64), 64);
                            addr = ctx.make_binary(BVOp::Add, addr, idx64, 64);
                        }
                        if (inst->offset() != 0)
                            addr = ctx.make_binary(BVOp::Add, addr,
                                ctx.make_const(static_cast<uint64_t>(static_cast<int64_t>(inst->offset())), 64), 64);
                        uint32_t num_bytes = static_cast<uint32_t>(inst->memory_type().size_in_bytes());
                        if (num_bytes == 0 || num_bytes > 8) {
                            out.supported = false;
                            out.error_msg = "Unsupported store width";
                            return out;
                        }
                        bool is_local = (ctx.nodes()[base].op == BVOp::Const &&
                                         alloca_bases.count(ctx.nodes()[base].imm) > 0);
                        out.stores.push_back({cur_cond, addr, num_bytes, coerce_width(ctx, val, num_bytes * 8, false), is_local});
                        break;
                    }

                    case Opcode::write_barrier:
                    case Opcode::safepoint:
                    case Opcode::keep_alive:
                    case Opcode::resume_point:
                        break;

                    case Opcode::func_addr:
                        set_res(init_mem.eval_uninterpreted_call(ctx, inst->symbol(), {}, Type::ptr()));
                        break;

                    case Opcode::call:
                    case Opcode::patchable_call: {
                        std::vector<BVNodeId> call_args;
                        for (size_t i = 0; i < inst->operand_count(); ++i) call_args.push_back(get_op(i));
                        if (inst->result() && !inst->type().is_void())
                            set_res(init_mem.eval_uninterpreted_call(ctx, inst->symbol(), call_args, inst->type()));
                        break;
                    }

                    case Opcode::guard: {
                        BVNodeId cond_v = get_op(0);
                        if (!inst->operand(0) || inst->operand(0)->type() != Type::i32()) out.has_type_error = true;
                        uint32_t cw = ctx.width(cond_v);
                        BVNodeId is_zero = ctx.make_binary(BVOp::Eq, cond_v, ctx.make_const(0, cw), 1);
                        BVNodeId fail_now = ctx.make_binary(BVOp::And, cur_cond, is_zero, 1);
                        if (!(ctx.nodes()[fail_now].op == BVOp::Const && ctx.nodes()[fail_now].imm == 0)) {
                            SymGuardExit ge;
                            ge.fail_cond = fail_now;
                            ge.resume_id = inst->resume_id();
                            ge.exit_symbol = std::string(inst->symbol());
                            for (const Value* sv : inst->state_map()) {
                                if (!sv) {
                                    out.has_type_error = true;
                                    out.error_msg = "Null state_map value in guard";
                                    ge.state_types.push_back(Type::i64());
                                    ge.state_vals.push_back(ctx.make_const(0, 64));
                                    continue;
                                }
                                auto vit = vals.find(sv);
                                if (vit == vals.end()) {
                                    out.has_type_error = true;
                                    out.error_msg = "Undefined state_map SSA value in guard";
                                    uint32_t sw = type_bv_width(sv->type());
                                    ge.state_types.push_back(sv->type());
                                    ge.state_vals.push_back(ctx.make_const(0, sw ? sw : 64));
                                } else {
                                    ge.state_types.push_back(sv->type());
                                    ge.state_vals.push_back(vit->second);
                                }
                            }
                            out.guard_exits.push_back(std::move(ge));
                        }
                        out.guard_failed = ctx.make_binary(BVOp::Or, out.guard_failed, fail_now, 1);
                        out.completed = ctx.make_binary(BVOp::Or, out.completed, fail_now, 1);
                        cur_cond = ctx.make_binary(BVOp::And, cur_cond, ctx.make_unary(BVOp::Not, is_zero, 1), 1);
                        break;
                    }

                    case Opcode::br:
                        route_branch(bi, level, cur_cond, inst->branch_target(), vals);
                        break;

                    case Opcode::br_if: {
                        BVNodeId cond_v = get_op(0);
                        if (!inst->operand(0) || inst->operand(0)->type() != Type::i32()) out.has_type_error = true;
                        uint32_t cw = ctx.width(cond_v);
                        BVNodeId c_true = ctx.make_binary(BVOp::Ne, cond_v, ctx.make_const(0, cw), 1);
                        BVNodeId c_false = ctx.make_binary(BVOp::Eq, cond_v, ctx.make_const(0, cw), 1);
                        route_branch(bi, level, ctx.make_binary(BVOp::And, cur_cond, c_true, 1), inst->true_target(), vals);
                        route_branch(bi, level, ctx.make_binary(BVOp::And, cur_cond, c_false, 1), inst->false_target(), vals);
                        break;
                    }

                    case Opcode::switch_: {
                        BVNodeId cond_v = get_op(0);
                        Type ct = inst->operand(0) ? inst->operand(0)->type() : Type::i32();
                        uint32_t cw = ctx.width(cond_v);
                        BVNodeId rem_cond = cur_cond;
                        for (const auto& sc : inst->switch_cases()) {
                            uint64_t case_bits = (cw < 64)
                                ? (static_cast<uint64_t>(sc.value) & ((1ULL << cw) - 1ULL))
                                : static_cast<uint64_t>(sc.value);
                            if (unsigned nb = narrow_int_bits(ct))
                                case_bits = static_cast<uint64_t>(narrow_zext(static_cast<uint64_t>(sc.value), nb));
                            BVNodeId hit = ctx.make_binary(BVOp::Eq, cond_v, ctx.make_const(case_bits, cw), 1);
                            route_branch(bi, level, ctx.make_binary(BVOp::And, rem_cond, hit, 1), sc.target, vals);
                            rem_cond = ctx.make_binary(BVOp::And, rem_cond, ctx.make_unary(BVOp::Not, hit, 1), 1);
                        }
                        route_branch(bi, level, rem_cond, inst->default_target(), vals);
                        break;
                    }

                    case Opcode::ret: {
                        bool first_ret = (ctx.nodes()[out.returned].op == BVOp::Const &&
                                          ctx.nodes()[out.returned].imm == 0);
                        out.returned = ctx.make_binary(BVOp::Or, out.returned, cur_cond, 1);
                        out.completed = ctx.make_binary(BVOp::Or, out.completed, cur_cond, 1);
                        if (!fn.return_type().is_void()) {
                            if (inst->operand_count() == 0 || !inst->operand(0) ||
                                inst->operand(0)->type() != fn.return_type())
                                out.has_type_error = true;
                            BVNodeId rv = coerce_width(ctx, get_op(0), ret_w, false);
                            out.ret_val = first_ret ? rv : ctx.make_ite(cur_cond, rv, out.ret_val);
                        }
                        break;
                    }

                    case Opcode::unreachable:
                        out.trapped = ctx.make_binary(BVOp::Or, out.trapped, cur_cond, 1);
                        out.completed = ctx.make_binary(BVOp::Or, out.completed, cur_cond, 1);
                        cur_cond = ctx.make_const(0, 1);
                        break;

                    default:
                        out.supported = false;
                        out.error_msg = "Unsupported opcode in MIR equivalence checker: " +
                                        std::string(opcode_name(op));
                        return out;
                }
            }
        }
    }

    return out;
}

RuntimeValue model_bits_to_runtime_value(Type t, uint64_t bits) {
    switch (t.kind()) {
        case TypeKind::I8:     return RuntimeValue::from_i32(static_cast<int32_t>(bits & 0xFFu));
        case TypeKind::I16:    return RuntimeValue::from_i32(static_cast<int32_t>(bits & 0xFFFFu));
        case TypeKind::I32:    return RuntimeValue::from_i32(static_cast<int32_t>(static_cast<uint32_t>(bits)));
        case TypeKind::I64:    return RuntimeValue::from_i64(static_cast<int64_t>(bits));
        case TypeKind::F32: {
            float f = 0.0f;
            uint32_t b32 = static_cast<uint32_t>(bits);
            std::memcpy(&f, &b32, 4);
            return RuntimeValue::from_f32(f);
        }
        case TypeKind::F64: {
            double d = 0.0;
            std::memcpy(&d, &bits, 8);
            return RuntimeValue::from_f64(d);
        }
        case TypeKind::Ptr:    return RuntimeValue::from_ptr(static_cast<uintptr_t>(bits));
        case TypeKind::GCRef:  return RuntimeValue::from_gcref(static_cast<uintptr_t>(bits));
        case TypeKind::Tagged: return RuntimeValue::from_tagged(bits);
        default:               return RuntimeValue::from_void();
    }
}

struct BuiltEquivQuery {
    BVContext ctx;
    std::vector<BVNodeId> param_vars;
    SymExecOutput out_before;
    SymExecOutput out_after;
    BVNodeId mismatch = 0;
    BVNodeId guard_diff = 0;
    BVNodeId deopt_diff = 0;
    BVNodeId trap_diff = 0;
    BVNodeId ret_diff = 0;
    BVNodeId mem_diff = 0;
    EquivStatus early_status = EquivStatus::ProvenEquivalent;
    std::string early_diag;
};

BuiltEquivQuery build_mir_equiv_query(
    const Function& before,
    const Function& after,
    const EquivOptions& opts) {

    BuiltEquivQuery q;
    if (before.param_count() != after.param_count() || before.return_type() != after.return_type()) {
        q.early_status = EquivStatus::Counterexample;
        q.early_diag = "Function signature mismatch between unoptimized and optimized MIR";
        return q;
    }
    for (size_t i = 0; i < before.param_count(); ++i) {
        if (before.param_type(i) != after.param_type(i)) {
            q.early_status = EquivStatus::Counterexample;
            q.early_diag = "Parameter " + std::to_string(i) + " type mismatch";
            return q;
        }
        uint32_t w = type_bv_width(before.param_type(i));
        if (w == 0) {
            q.early_status = EquivStatus::Unsupported;
            q.early_diag = "Unsupported parameter type";
            return q;
        }
        q.param_vars.push_back(q.ctx.make_var("arg" + std::to_string(i), w, before.param_type(i).is_float()));
    }

    InitMemEnv init_mem;
    q.out_before = symbolically_execute_mir(q.ctx, before, q.param_vars, init_mem, opts);
    if (!q.out_before.supported) {
        q.early_status = EquivStatus::Unsupported;
        q.early_diag = q.out_before.error_msg;
        return q;
    }
    q.out_after = symbolically_execute_mir(q.ctx, after, q.param_vars, init_mem, opts);
    if (!q.out_after.supported) {
        q.early_status = EquivStatus::Unsupported;
        q.early_diag = q.out_after.error_msg;
        return q;
    }
    if (!q.out_before.has_type_error && q.out_after.has_type_error) {
        q.early_diag = "Optimized MIR has type/SSA error: " + q.out_after.error_msg;
    }

    BVNodeId both_completed = q.ctx.make_binary(BVOp::And, q.out_before.completed, q.out_after.completed, 1);

    q.guard_diff = q.ctx.make_const(0, 1);
    q.deopt_diff = q.ctx.make_const(0, 1);
    if (opts.check_guards) {
        BVNodeId g_ne = q.ctx.make_binary(BVOp::Xor, q.out_before.guard_failed, q.out_after.guard_failed, 1);
        q.guard_diff = q.ctx.make_binary(BVOp::And, both_completed, g_ne, 1);

        // Verify deopt state_map(), resume_id(), and exit_symbol when both trigger a guard on the same input (#75)
        for (const auto& gb : q.out_before.guard_exits) {
            for (const auto& ga : q.out_after.guard_exits) {
                BVNodeId pair_cond = q.ctx.make_binary(
                    BVOp::And, both_completed,
                    q.ctx.make_binary(BVOp::And, gb.fail_cond, ga.fail_cond, 1), 1);
                if (q.ctx.nodes()[pair_cond].op == BVOp::Const && q.ctx.nodes()[pair_cond].imm == 0) continue;

                BVNodeId pair_bad = q.ctx.make_const(0, 1);
                if (gb.resume_id != ga.resume_id || gb.exit_symbol != ga.exit_symbol ||
                    gb.state_vals.size() != ga.state_vals.size()) {
                    pair_bad = pair_cond;
                } else {
                    for (size_t k = 0; k < gb.state_vals.size(); ++k) {
                        if (gb.state_types[k] != ga.state_types[k] ||
                            q.ctx.width(gb.state_vals[k]) != q.ctx.width(ga.state_vals[k])) {
                            pair_bad = pair_cond;
                            break;
                        }
                        BVNodeId v_ne = q.ctx.make_binary(BVOp::Ne, gb.state_vals[k], ga.state_vals[k], 1);
                        pair_bad = q.ctx.make_binary(BVOp::Or, pair_bad,
                                                     q.ctx.make_binary(BVOp::And, pair_cond, v_ne, 1), 1);
                    }
                }
                q.deopt_diff = q.ctx.make_binary(BVOp::Or, q.deopt_diff, pair_bad, 1);
            }
        }
    }

    BVNodeId no_guard_fail = q.ctx.make_binary(
        BVOp::And, both_completed,
        q.ctx.make_binary(BVOp::And,
                          q.ctx.make_unary(BVOp::Not, q.out_before.guard_failed, 1),
                          q.ctx.make_unary(BVOp::Not, q.out_after.guard_failed, 1), 1), 1);

    BVNodeId t_ne = q.ctx.make_binary(BVOp::Xor, q.out_before.trapped, q.out_after.trapped, 1);
    q.trap_diff = q.ctx.make_binary(BVOp::And, no_guard_fail, t_ne, 1);

    BVNodeId both_returned = q.ctx.make_binary(
        BVOp::And, no_guard_fail,
        q.ctx.make_binary(BVOp::And, q.out_before.returned, q.out_after.returned, 1), 1);

    q.ret_diff = q.ctx.make_const(0, 1);
    if (!before.return_type().is_void()) {
        BVNodeId r_ne = q.ctx.make_binary(BVOp::Ne, q.out_before.ret_val, q.out_after.ret_val, 1);
        q.ret_diff = q.ctx.make_binary(BVOp::And, both_returned, r_ne, 1);
    }

    q.mem_diff = q.ctx.make_const(0, 1);
    if (opts.check_memory) {
        std::vector<BVNodeId> check_byte_addrs;
        auto collect_addrs = [&](const std::vector<SymStore>& stores) {
            for (const auto& st : stores) {
                if (st.is_local_alloca) continue;
                for (uint32_t b = 0; b < st.num_bytes; ++b) {
                    BVNodeId ba = (b == 0) ? st.addr : q.ctx.make_binary(BVOp::Add, st.addr, q.ctx.make_const(b, 64), 64);
                    if (std::find(check_byte_addrs.begin(), check_byte_addrs.end(), ba) == check_byte_addrs.end())
                        check_byte_addrs.push_back(ba);
                }
            }
        };
        collect_addrs(q.out_before.stores);
        collect_addrs(q.out_after.stores);

        for (BVNodeId ba : check_byte_addrs) {
            BVNodeId mb = read_byte_at(q.ctx, init_mem, q.out_before.stores, ba);
            BVNodeId ma = read_byte_at(q.ctx, init_mem, q.out_after.stores, ba);
            BVNodeId m_ne = q.ctx.make_binary(BVOp::Ne, mb, ma, 1);
            q.mem_diff = q.ctx.make_binary(BVOp::Or, q.mem_diff,
                                           q.ctx.make_binary(BVOp::And, both_returned, m_ne, 1), 1);
        }
    }

    BVNodeId any_guard_or_deopt = q.ctx.make_binary(BVOp::Or, q.guard_diff, q.deopt_diff, 1);
    q.mismatch = q.ctx.make_binary(
        BVOp::Or,
        q.ctx.make_binary(BVOp::Or, any_guard_or_deopt, q.trap_diff, 1),
        q.ctx.make_binary(BVOp::Or, q.ret_diff, q.mem_diff, 1), 1);

    if (!q.out_before.has_type_error && q.out_after.has_type_error) {
        q.mismatch = q.ctx.make_const(1, 1);
    }

    return q;
}

} // namespace

EquivResult verify_mir_equivalence(
    const Function& before,
    const Function& after,
    const EquivOptions& opts) {

    EquivResult res;
    BuiltEquivQuery q = build_mir_equiv_query(before, after, opts);
    if (q.early_status != EquivStatus::ProvenEquivalent) {
        res.status = q.early_status;
        res.diagnostic = q.early_diag;
        return res;
    }

    SolveOutcome solved = solve_nonzero(q.ctx, q.mismatch, opts.max_sat_conflicts);
    if (solved.status == SolveOutcome::Unsat) {
        res.status = EquivStatus::ProvenEquivalent;
        return res;
    }
    if (solved.status == SolveOutcome::Unknown) {
        res.status = EquivStatus::Unknown;
        res.diagnostic = "Solver budget exhausted before proving equivalence";
        return res;
    }

    res.status = EquivStatus::Counterexample;
    for (size_t i = 0; i < before.param_count(); ++i) {
        uint64_t bits = (i < solved.model.size()) ? solved.model[i] : 0;
        res.counterexample.push_back(model_bits_to_runtime_value(before.param_type(i), bits));
    }

    std::ostringstream diag;
    diag << "MIR equivalence counterexample in @" << before.name() << "(";
    for (size_t i = 0; i < res.counterexample.size(); ++i) {
        if (i > 0) diag << ", ";
        diag << to_string(res.counterexample[i]);
    }
    diag << "): ";
    if (!q.early_diag.empty()) {
        diag << q.early_diag;
    } else if (q.ctx.eval_concrete(q.guard_diff, solved.model) != 0) {
        diag << "guard failure mismatch (before guard_failed="
             << q.ctx.eval_concrete(q.out_before.guard_failed, solved.model)
             << ", after guard_failed="
             << q.ctx.eval_concrete(q.out_after.guard_failed, solved.model) << ")";
    } else if (q.ctx.eval_concrete(q.deopt_diff, solved.model) != 0) {
        diag << "deopt state_map/resume mismatch on triggered guard";
    } else if (q.ctx.eval_concrete(q.trap_diff, solved.model) != 0) {
        diag << "trap mismatch (before trapped="
             << q.ctx.eval_concrete(q.out_before.trapped, solved.model)
             << ", after trapped="
             << q.ctx.eval_concrete(q.out_after.trapped, solved.model) << ")";
    } else if (q.ctx.eval_concrete(q.ret_diff, solved.model) != 0) {
        diag << "return value mismatch (before="
             << to_string(model_bits_to_runtime_value(before.return_type(),
                                                      q.ctx.eval_concrete(q.out_before.ret_val, solved.model)))
             << ", after="
             << to_string(model_bits_to_runtime_value(after.return_type(),
                                                      q.ctx.eval_concrete(q.out_after.ret_val, solved.model)))
             << ")";
    } else {
        diag << "observable memory store mismatch";
    }
    res.diagnostic = diag.str();
    return res;
}

EquivResult verify_mir_module_equivalence(
    const Module& before,
    const Module& after,
    const EquivOptions& opts) {

    EquivResult overall;
    overall.status = EquivStatus::ProvenEquivalent;
    for (const Function* fn_before : before.functions()) {
        if (!fn_before || fn_before->blocks().empty()) continue;
        const Function* fn_after = after.get_function(fn_before->name());
        if (!fn_after || fn_after->blocks().empty()) continue;
        EquivResult r = verify_mir_equivalence(*fn_before, *fn_after, opts);
        if (r.status == EquivStatus::Counterexample) return r;
        if (r.status == EquivStatus::Unknown && overall.status == EquivStatus::ProvenEquivalent) overall = r;
    }
    return overall;
}

std::string export_mir_equivalence_smt2(
    const Function& before,
    const Function& after,
    const EquivOptions& opts) {

    BuiltEquivQuery q = build_mir_equiv_query(before, after, opts);
    if (q.early_status != EquivStatus::ProvenEquivalent && q.mismatch == 0) {
        return "; Unable to build SMT-LIB2 equivalence query: " + q.early_diag + "\n";
    }
    return q.ctx.export_smt2(q.mismatch, "MIR equivalence query for @" + std::string(before.name()));
}

} // namespace brass::verify
