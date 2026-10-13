// Control flow lowering for the bytecode compiler: block-argument parallel
// moves, jumps with fall-through, br_if (plain and fused with the integer
// comparison feeding it), switch and invoke.

#include "bytecode_compiler_impl.hpp"

namespace brass::detail {

namespace {

// i8 / i16 comparisons are not fused: they compare widened operands
// (bytecode_compiler_narrow.cpp), which only the unfused lowering makes.
bool is_int_compare_type(Type t) {
    return t == Type::i32() || t == Type::i64() || t.is_pointer() || t.is_gcref();
}

bool is_float_compare_op(Opcode op) {
    switch (op) {
        case Opcode::eq: case Opcode::ne:
        case Opcode::slt: case Opcode::sle: case Opcode::sgt: case Opcode::sge:
            return true;
        default:
            return false;
    }
}

struct FusedBranch {
    BytecodeOp op;
    BcReg lhs;
    BcReg rhs;
};

// The fused branch taken when `cmp` holds (or, with `invert`, fails).
FusedBranch fused_branch(const FunctionCompilerContext& ctx, const Instruction& cmp, bool invert) {
    BcReg a = ctx.get_reg(cmp.operand(0));
    BcReg b = ctx.get_reg(cmp.operand(1));
    Type t = cmp.operand(0)->type();
    if (t.is_float()) {
        const bool f32 = t == Type::f32();
        auto fpick = [f32](BytecodeOp op32, BytecodeOp op64) { return f32 ? op32 : op64; };
        const BytecodeOp eq = fpick(BytecodeOp::br_eq_f32, BytecodeOp::br_eq_f64);
        const BytecodeOp ne = fpick(BytecodeOp::br_ne_f32, BytecodeOp::br_ne_f64);
        const BytecodeOp lt = fpick(BytecodeOp::br_lt_f32, BytecodeOp::br_lt_f64);
        const BytecodeOp le = fpick(BytecodeOp::br_le_f32, BytecodeOp::br_le_f64);
        const BytecodeOp not_lt = fpick(BytecodeOp::br_not_lt_f32, BytecodeOp::br_not_lt_f64);
        const BytecodeOp not_le = fpick(BytecodeOp::br_not_le_f32, BytecodeOp::br_not_le_f64);
        // For IEEE-754 floats, !(a < b) is NOT (b <= a) when either operand is NaN:
        // inverted ordered comparisons use the explicit unordered-or-negated opcodes.
        switch (cmp.opcode()) {
            case Opcode::eq: return {invert ? ne : eq, a, b};
            case Opcode::ne: return {invert ? eq : ne, a, b};
            case Opcode::slt: return {invert ? not_lt : lt, a, b};
            case Opcode::sle: return {invert ? not_le : le, a, b};
            case Opcode::sgt: return {invert ? not_lt : lt, b, a};
            case Opcode::sge: return {invert ? not_le : le, b, a};
            default: break;
        }
        ctx.fail("cannot fuse float " + std::string(opcode_name(cmp.opcode())) + " into a branch");
    }
    const bool w32 = t == Type::i32() || t == Type::i16() || t == Type::i8();
    auto pick = [w32](BytecodeOp op32, BytecodeOp op64) { return w32 ? op32 : op64; };
    const BytecodeOp eq = pick(BytecodeOp::br_eq_i32, BytecodeOp::br_eq_i64);
    const BytecodeOp ne = pick(BytecodeOp::br_ne_i32, BytecodeOp::br_ne_i64);
    const BytecodeOp slt = pick(BytecodeOp::br_slt_i32, BytecodeOp::br_slt_i64);
    const BytecodeOp sle = pick(BytecodeOp::br_sle_i32, BytecodeOp::br_sle_i64);
    const BytecodeOp ult = pick(BytecodeOp::br_ult_i32, BytecodeOp::br_ult_i64);
    const BytecodeOp ule = pick(BytecodeOp::br_ule_i32, BytecodeOp::br_ule_i64);
    // !(a < b) == (b <= a) and !(a <= b) == (b < a).
    switch (cmp.opcode()) {
        case Opcode::eq: return {invert ? ne : eq, a, b};
        case Opcode::ne: return {invert ? eq : ne, a, b};
        case Opcode::slt: return invert ? FusedBranch{sle, b, a} : FusedBranch{slt, a, b};
        case Opcode::sle: return invert ? FusedBranch{slt, b, a} : FusedBranch{sle, a, b};
        case Opcode::sgt: return invert ? FusedBranch{sle, a, b} : FusedBranch{slt, b, a};
        case Opcode::sge: return invert ? FusedBranch{slt, a, b} : FusedBranch{sle, b, a};
        case Opcode::ult: return invert ? FusedBranch{ule, b, a} : FusedBranch{ult, a, b};
        case Opcode::ule: return invert ? FusedBranch{ult, b, a} : FusedBranch{ule, a, b};
        case Opcode::ugt: return invert ? FusedBranch{ule, a, b} : FusedBranch{ult, b, a};
        case Opcode::uge: return invert ? FusedBranch{ult, a, b} : FusedBranch{ule, b, a};
        default: break;
    }
    ctx.fail("cannot fuse " + std::string(opcode_name(cmp.opcode())) + " into a branch");
}

} // namespace

bool FunctionCompilerContext::can_fuse_compare_branch(const Instruction& cmp) const {
    if (cmp.operand_count() != 2 || !cmp.operand(0) || !cmp.operand(1)) return false;
    const Type t = cmp.operand(0)->type();
    if (t.is_float()) return is_float_compare_op(cmp.opcode());
    switch (cmp.opcode()) {
        case Opcode::eq: case Opcode::ne:
        case Opcode::slt: case Opcode::sle: case Opcode::sgt: case Opcode::sge:
        case Opcode::ult: case Opcode::ule: case Opcode::ugt: case Opcode::uge:
            return is_int_compare_type(t);
        default:
            return false;
    }
}

void FunctionCompilerContext::emit_jump_to(const BasicBlock* target) {
    jump_fixups.push_back({out.current_pc(), target, false});
    emit_ai(BytecodeOp::jump, 0, 0);
}

void FunctionCompilerContext::emit_parallel_moves(const BranchTarget& target) {
    const BasicBlock* target_bb = target.block;
    if (!target_bb || target.args.empty()) return;
    if (target.args.size() != target_bb->param_count()) {
        fail("branch passes " + std::to_string(target.args.size()) + " arguments to a block with " +
             std::to_string(target_bb->param_count()) + " parameters");
    }

    struct Move {
        BcReg src;
        BcReg dst;
        bool done;
    };
    std::vector<Move> moves;
    moves.reserve(target.args.size());
    for (size_t i = 0; i < target.args.size(); ++i) {
        const Value* param_val = target_bb->param(i);
        if (!param_val) continue;
        BcReg src = get_reg(target.args[i]);
        BcReg dst = get_reg(param_val);
        if (src == dst || move_writer_of[dst] >= 0) continue;
        move_writer_of[dst] = static_cast<int32_t>(moves.size());
        ++move_use_count[src];
        moves.push_back({src, dst, false});
    }
    if (moves.empty()) return;

    auto move_op = [&](BcReg dst) {
        return reg_type(dst).is_vector() ? BytecodeOp::vmov : BytecodeOp::mov;
    };

    std::vector<uint32_t> ready;
    ready.reserve(moves.size());
    for (uint32_t i = 0; i < moves.size(); ++i) {
        if (move_use_count[moves[i].dst] == 0) ready.push_back(i);
    }

    size_t scan = 0;
    size_t remaining = moves.size();
    while (remaining > 0) {
        while (!ready.empty()) {
            uint32_t idx = ready.back();
            ready.pop_back();
            Move& m = moves[idx];
            if (m.done) continue;
            emit(move_op(m.dst), m.dst, m.src);
            m.done = true;
            move_writer_of[m.dst] = -1;
            --remaining;
            if (m.src != scratch_reg && --move_use_count[m.src] == 0) {
                int32_t w = move_writer_of[m.src];
                if (w >= 0 && !moves[static_cast<size_t>(w)].done) {
                    ready.push_back(static_cast<uint32_t>(w));
                }
            }
        }
        if (remaining == 0) break;
        // Only disjoint cycles remain: each remaining dst has use_count == 1.
        while (moves[scan].done) ++scan;
        Move& m = moves[scan];
        BcReg old_src = m.src;
        emit(move_op(m.dst), scratch_reg, old_src);
        m.src = scratch_reg;
        if (--move_use_count[old_src] == 0) {
            int32_t w = move_writer_of[old_src];
            if (w >= 0 && !moves[static_cast<size_t>(w)].done) {
                ready.push_back(static_cast<uint32_t>(w));
            }
        }
    }
}

uint32_t FunctionCompilerContext::emit_edge(const BranchTarget& target) {
    uint32_t pc = static_cast<uint32_t>(out.current_pc());
    emit_parallel_moves(target);
    emit_jump_to(target.block);
    return pc;
}

void FunctionCompilerContext::lower_terminator(const Instruction& inst) {
    switch (inst.opcode()) {
        case Opcode::ret:
            if (inst.operand_count() > 0 && inst.operand(0) != nullptr) {
                emit(BytecodeOp::ret, get_reg(inst.operand(0)));
            } else {
                emit(BytecodeOp::ret_void, 0);
            }
            break;
        case Opcode::unreachable:
            emit(BytecodeOp::unreachable, 0);
            break;
        case Opcode::br: {
            const auto& target = inst.branch_target();
            emit_parallel_moves(target);
            if (target.block != next_block) emit_jump_to(target.block);
            break;
        }
        case Opcode::br_if:
            lower_br_if(inst);
            break;
        case Opcode::switch_:
            lower_switch(inst);
            break;
        case Opcode::invoke:
            lower_invoke(inst);
            break;
        default:
            fail("not a terminator: " + std::string(opcode_name(inst.opcode())));
    }
}

void FunctionCompilerContext::lower_br_if(const Instruction& inst) {
    const BranchTarget& t = inst.true_target();
    const BranchTarget& f = inst.false_target();
    const Instruction* cmp = fused_compare;
    if (!cmp && !inst.operand(0)) fail("br_if without a condition");
    const BcReg cond = cmp ? kNoReg : get_reg(inst.operand(0));

    // Emits "if cond (or !cond with invert) goto <patched later>" and
    // returns its index.
    auto emit_branch = [&](bool invert) -> size_t {
        size_t idx = out.current_pc();
        if (cmp) {
            FusedBranch fb = fused_branch(*this, *cmp, invert);
            emit_abi(fb.op, fb.lhs, fb.rhs, 0);
        } else {
            emit_ai(invert ? BytecodeOp::jump_if_not : BytecodeOp::jump_if, cond, 0);
        }
        return idx;
    };
    auto branch_to_block = [&](bool invert, const BasicBlock* target) {
        size_t idx = emit_branch(invert);
        jump_fixups.push_back({idx, target, cmp != nullptr});
    };

    const bool t_args = !t.args.empty();
    const bool f_args = !f.args.empty();
    if (!t_args && !f_args) {
        if (f.block == next_block) {
            branch_to_block(false, t.block);
        } else if (t.block == next_block) {
            branch_to_block(true, f.block);
        } else {
            branch_to_block(false, t.block);
            emit_jump_to(f.block);
        }
    } else if (t_args && !f_args) {
        branch_to_block(true, f.block);
        emit_parallel_moves(t);
        if (t.block != next_block) emit_jump_to(t.block);
    } else if (!t_args && f_args) {
        branch_to_block(false, t.block);
        emit_parallel_moves(f);
        if (f.block != next_block) emit_jump_to(f.block);
    } else {
        size_t idx = emit_branch(false);
        emit_edge(f);
        int64_t rel = static_cast<int64_t>(out.current_pc()) - static_cast<int64_t>(idx);
        BytecodeWord w = out.code[idx];
        out.code[idx] = cmp ? encode_abi(decode_op(w), decode_a(w), decode_b(w), static_cast<int32_t>(rel))
                            : encode_ai(decode_op(w), decode_a(w), static_cast<int32_t>(rel));
        emit_parallel_moves(t);
        if (t.block != next_block) emit_jump_to(t.block);
    }
}

void FunctionCompilerContext::lower_switch(const Instruction& inst) {
    BcReg cond_reg = get_reg(inst.operand(0));
    Type ct = inst.operand(0)->type();
    // Narrow case values are signed: switch on the condition sign-extended.
    cond_reg = widen_narrow(inst.operand(0), true, scratch_reg);
    SwitchTable st;
    st.is_i32 = ct == Type::i32() || ct == Type::i16() || ct == Type::i8();
    size_t t_idx = out.switch_tables.size();
    for (const auto& sc : inst.switch_cases()) st.cases.push_back({sc.value, 0});
    out.switch_tables.push_back(std::move(st));
    emit_ai(BytecodeOp::switch_, cond_reg, static_cast<int32_t>(t_idx));

    // Edges with arguments get a move trampoline after the switch (which
    // never falls through).
    const auto& cases = inst.switch_cases();
    for (size_t i = 0; i < cases.size(); ++i) {
        SwitchFixup sf{t_idx, i, cases[i].target.block, false, -1};
        if (!cases[i].target.args.empty()) sf.trampoline_pc = emit_edge(cases[i].target);
        switch_fixups.push_back(sf);
    }
    SwitchFixup df{t_idx, 0, inst.default_target().block, true, -1};
    if (!inst.default_target().args.empty()) df.trampoline_pc = emit_edge(inst.default_target());
    switch_fixups.push_back(df);
}

void FunctionCompilerContext::lower_invoke(const Instruction& inst) {
    uint32_t start_pc = static_cast<uint32_t>(out.current_pc());
    BcReg dst = result_reg_or_none(inst);
    CallSiteInfo cs;
    cs.callee = std::string(inst.symbol());
    cs.dst_reg = dst;
    for (size_t i = 0; i < inst.operand_count(); ++i) cs.arg_regs.push_back(get_reg(inst.operand(i)));
    uint32_t cs_idx = out.add_call_site(std::move(cs));
    emit_ai(BytecodeOp::invoke, dst, static_cast<int32_t>(cs_idx));

    const BranchTarget& unwind = inst.unwind_target();
    const bool unwind_moves = unwind.block && !unwind.args.empty();
    emit_parallel_moves(inst.normal_target());
    if (inst.normal_target().block != next_block || unwind_moves) emit_jump_to(inst.normal_target().block);

    // Without an unwind block the exception simply propagates.
    if (!unwind.block) return;
    ExceptionEntry ee;
    ee.start_pc = start_pc;
    ee.end_pc = start_pc;
    out.exception_table.push_back(ee);
    ExceptionFixup ef{out.exception_table.size() - 1, unwind.block, -1};
    if (unwind_moves) ef.trampoline_pc = emit_edge(unwind);
    exception_fixups.push_back(ef);
}

} // namespace brass::detail
