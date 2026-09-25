// PtxISel core: driver, register assignment, kernel parameters and the opcode
// dispatch. The per-opcode lowering lives in ptx_isel_alu.cpp,
// ptx_isel_mem.cpp, ptx_isel_control.cpp and ptx_isel_intrinsics.cpp.

#include <brass/target/ptx/ptx_isel.hpp>
#include <brass/mir/block.hpp>
#include <brass/mir/instruction.hpp>
#include <brass/mir/opcodes.hpp>

#include <algorithm>
#include <stdexcept>
#include <string>

namespace brass::ptx {

namespace {

std::string block_label(const BasicBlock* bb) {
    return "$L_bb_" + std::to_string(bb->id());
}

// The condition operand of br_if/select is consumed as a predicate; every
// other operand position is a value use.
bool is_predicate_use(const brass::Instruction& inst, size_t operand_index) {
    return operand_index == 0 &&
           (inst.opcode() == brass::Opcode::br_if || inst.opcode() == brass::Opcode::select);
}

} // namespace

// ---------------------------------------------------------------------------
// Driver
// ---------------------------------------------------------------------------

Function PtxISel::lower(const brass::Function& mir_fn) {
    // PTX `ret` takes no operand and `.entry` kernels cannot return values;
    // results must be written through pointer parameters.
    if (!mir_fn.return_type().is_void()) {
        throw std::runtime_error("PtxTarget: kernel '" + std::string(mir_fn.name()) +
                                 "' has a non-void return type; .entry kernels cannot return values");
    }

    Function out(std::string(mir_fn.name()), /*entry=*/true);
    fn_ = &out;
    bb_ = nullptr;
    prologue_ = nullptr;
    origin_ = nullptr;
    regs_.clear();
    preds_.clear();
    blocks_.clear();
    value_uses_.clear();
    special_cache_.clear();

    analyze_uses(mir_fn);
    allocate_registers(mir_fn);

    // Create every block first so branch targets resolve to stable labels.
    for (const BasicBlock* bb : mir_fn.blocks()) {
        blocks_[bb] = out.add_block(block_label(bb));
    }

    lower_params(mir_fn);
    for (const BasicBlock* bb : mir_fn.blocks()) {
        lower_block(*bb);
    }

    fn_ = nullptr;
    bb_ = nullptr;
    return out;
}

// ---------------------------------------------------------------------------
// Analysis and register assignment
// ---------------------------------------------------------------------------

void PtxISel::analyze_uses(const brass::Function& mir_fn) {
    auto count = [&](const Value* v) { if (v) ++value_uses_[v]; };
    for (const BasicBlock* bb : mir_fn.blocks()) {
        for (const brass::Instruction* inst : *bb) {
            for (size_t i = 0; i < inst->operand_count(); ++i) {
                if (!is_predicate_use(*inst, i)) count(inst->operand(i));
            }
            for (const Value* v : inst->branch_target().args) count(v);
            for (const Value* v : inst->true_target().args) count(v);
            for (const Value* v : inst->false_target().args) count(v);
            for (const auto& c : inst->switch_cases()) {
                for (const Value* v : c.target.args) count(v);
            }
            for (const Value* v : inst->state_map()) count(v);
        }
    }
}

// Every MIR value gets a register run up front: one register for scalars, a
// contiguous run of `lanes` registers for vector types. Comparison results
// additionally get a predicate. Temporaries needed during lowering are
// allocated lazily and therefore number after these.
void PtxISel::allocate_registers(const brass::Function& mir_fn) {
    auto assign = [&](const Value* v) {
        if (!v || regs_.count(v)) return;
        brass::Type t = v->type();
        if (t.is_void()) return;
        size_t lanes = t.is_vector() ? t.vector_lanes() : 1;
        regs_[v] = fn_->new_regs(reg_class_for(t), lanes);
    };

    if (const BasicBlock* entry = mir_fn.entry_block()) {
        for (size_t i = 0; i < entry->param_count(); ++i) assign(entry->param(i));
    }
    for (const BasicBlock* bb : mir_fn.blocks()) {
        for (const Value* p : bb->params()) assign(p);
        for (const brass::Instruction* inst : *bb) {
            if (inst->result() && !inst->type().is_void()) assign(inst->result());
            if (brass::is_comparison(inst->opcode()) && inst->result()) {
                preds_[inst->result()] = fn_->new_pred();
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Kernel parameters
// ---------------------------------------------------------------------------

void PtxISel::lower_params(const brass::Function& mir_fn) {
    const auto& types = mir_fn.param_types();
    std::vector<Operand> param_operands;
    param_operands.reserve(types.size());
    for (size_t i = 0; i < types.size(); ++i) {
        param_operands.push_back(fn_->add_param(type_for(types[i]), "param_" + std::to_string(i)));
    }

    const BasicBlock* entry = mir_fn.entry_block();
    if (!entry || entry->param_count() == 0) return;

    bb_ = prologue_block();
    for (size_t i = 0; i < entry->param_count(); ++i) {
        const Value* p = entry->param(i);
        if (!p) {
            throw std::runtime_error("PtxISel: entry block of '" + fn_->name +
                                     "' has a null parameter at index " + std::to_string(i));
        }
        if (i >= param_operands.size()) {
            throw std::runtime_error("PtxISel: entry block of '" + fn_->name + "' has " +
                                     std::to_string(entry->param_count()) + " parameters but the function declares " +
                                     std::to_string(param_operands.size()));
        }
        emit(Inst::make(Opcode::ld, type_for(p->type())).space(StateSpace::param)
                 .dst(reg_of(p, "entry param")).src(param_operands[i]));
    }
    bb_ = nullptr;
}

// The prologue is a fall-through block ahead of the MIR entry block, so that
// an entry block with incoming edges is not re-entered through the loads or
// the special-register reads. It is created on first use, so a kernel with
// neither params nor special-register reads has no $L_params label.
Block* PtxISel::prologue_block() {
    if (!prologue_) {
        prologue_ = fn_->add_block("$L_params");
        std::rotate(fn_->blocks.begin(), fn_->blocks.end() - 1, fn_->blocks.end());
    }
    return prologue_;
}

// ---------------------------------------------------------------------------
// Blocks and dispatch
// ---------------------------------------------------------------------------

void PtxISel::lower_block(const BasicBlock& bb) {
    bb_ = blocks_.at(&bb);
    for (const brass::Instruction* inst : bb) {
        origin_ = inst;
        lower_instruction(*inst);
    }
    origin_ = nullptr;
    bb_ = nullptr;
}

void PtxISel::lower_instruction(const brass::Instruction& inst) {
    using brass::Opcode;
    switch (inst.opcode()) {
        // Constants
        case Opcode::iconst_i32:  lower_iconst(inst, Type::b32); break;
        case Opcode::iconst_i64:  lower_iconst(inst, Type::b64); break;
        case Opcode::fconst_f64:  lower_fconst(inst); break;

        // Arithmetic
        case Opcode::add:     lower_binary(inst, ptx::Opcode::add); break;
        case Opcode::sub:     lower_binary(inst, ptx::Opcode::sub); break;
        case Opcode::mul:     lower_mul(inst); break;
        case Opcode::fma_f32:
        case Opcode::fma_f64: lower_fma(inst); break;
        case Opcode::sdiv:    lower_div(inst, false); break;
        case Opcode::udiv:    lower_div(inst, true); break;
        case Opcode::smod:    lower_rem(inst, false); break;
        case Opcode::umod:    lower_rem(inst, true); break;
        case Opcode::neg:     lower_neg(inst); break;
        case Opcode::sqrt_f32:
        case Opcode::sqrt_f64: {
            Type t = type_for(inst.type());
            emit(Inst::make(ptx::Opcode::sqrt, t).rnd(Rounding::rn)
                     .dst(result_reg(inst))
                     .src(reg_of(inst.operand(0), "operand 0")));
            break;
        }
        case Opcode::floor_f32:
        case Opcode::floor_f64: {
            Type t = type_for(inst.type());
            lower_cvt(inst, t, t, Rounding::rmi);
            break;
        }
        case Opcode::ceil_f32:
        case Opcode::ceil_f64: {
            Type t = type_for(inst.type());
            lower_cvt(inst, t, t, Rounding::rpi);
            break;
        }
        case Opcode::round_f32:
        case Opcode::round_f64: {
            Type t = type_for(inst.type());
            lower_cvt(inst, t, t, Rounding::rni);
            break;
        }
        case Opcode::fabs_f32:
        case Opcode::fabs_f64: {
            Type t = type_for(inst.type());
            emit(Inst::make(ptx::Opcode::abs, t)
                     .dst(result_reg(inst))
                     .src(reg_of(inst.operand(0), "operand 0")));
            break;
        }
        case Opcode::fmin_f32:
        case Opcode::fmin_f64: lower_binary(inst, ptx::Opcode::min); break;
        case Opcode::fmax_f32:
        case Opcode::fmax_f64: lower_binary(inst, ptx::Opcode::max); break;

        // Bitwise and shifts
        case Opcode::and_: lower_bitwise(inst, ptx::Opcode::and_); break;
        case Opcode::or_:  lower_bitwise(inst, ptx::Opcode::or_); break;
        case Opcode::xor_: lower_bitwise(inst, ptx::Opcode::xor_); break;
        case Opcode::not_: lower_not(inst); break;
        case Opcode::shl:  lower_shift(inst, ptx::Opcode::shl, bit_type_for(inst.type())); break;
        case Opcode::lshr: lower_shift(inst, ptx::Opcode::shr, type_for(inst.type())); break;
        case Opcode::ashr: lower_shift(inst, ptx::Opcode::shr, signed_type_for(inst.type())); break;

        // Comparisons
        case Opcode::eq:  lower_comparison(inst, CmpOp::eq, false); break;
        case Opcode::ne:  lower_comparison(inst, CmpOp::ne, false); break;
        case Opcode::slt: lower_comparison(inst, CmpOp::lt, false); break;
        case Opcode::ult: lower_comparison(inst, CmpOp::lt, true); break;
        case Opcode::sle: lower_comparison(inst, CmpOp::le, false); break;
        case Opcode::ule: lower_comparison(inst, CmpOp::le, true); break;
        case Opcode::sgt: lower_comparison(inst, CmpOp::gt, false); break;
        case Opcode::ugt: lower_comparison(inst, CmpOp::gt, true); break;
        case Opcode::sge: lower_comparison(inst, CmpOp::ge, false); break;
        case Opcode::uge: lower_comparison(inst, CmpOp::ge, true); break;

        case Opcode::select: lower_select(inst); break;

        // Conversions
        case Opcode::sext_i64:       lower_cvt(inst, Type::s64, Type::s32); break;
        case Opcode::zext_i64:
            lower_cvt(inst, Type::u64, inst.operand(0)->type().is_i8() ? Type::u8 : Type::u32);
            break;
        case Opcode::trunc_i32:      lower_cvt(inst, Type::u32, Type::u64); break;
        case Opcode::trunc_i8:
            lower_cvt(inst, Type::u8, inst.operand(0)->type().is_i64() ? Type::u64 : Type::u32);
            break;
        case Opcode::sitofp_f64_i32:
        case Opcode::sitofp_f32_i32: lower_sitofp(inst, Type::s32); break;
        case Opcode::sitofp_f64_i64:
        case Opcode::sitofp_f32_i64: lower_sitofp(inst, Type::s64); break;
        case Opcode::fptosi_i32:
        case Opcode::fptosi_i32_f32: lower_fptosi(inst, Type::s32); break;
        case Opcode::fptosi_i64:
        case Opcode::fptosi_i64_f32: lower_fptosi(inst, Type::s64); break;
        case Opcode::fptrunc_f32_f64: lower_cvt(inst, Type::f32, Type::f64, Rounding::rn); break;
        case Opcode::fpext_f64_f32:   lower_cvt(inst, Type::f64, Type::f32, Rounding::rn); break;
        case Opcode::bitcast_i64_f64:
        case Opcode::bitcast_f64_i64: lower_bitcast(inst); break;

        // Memory
        case Opcode::load:          lower_load(inst); break;
        case Opcode::store:         lower_store(inst); break;
        case Opcode::vload:         lower_vload(inst); break;
        case Opcode::vstore:        lower_vstore(inst); break;
        case Opcode::load_indexed:  lower_load_indexed(inst); break;
        case Opcode::store_indexed: lower_store_indexed(inst); break;

        // Vectors (per-lane scalar instructions, ptx_isel_vec.cpp)
        case Opcode::vadd:  lower_vector_binary(inst, ptx::Opcode::add); break;
        case Opcode::vsub:  lower_vector_binary(inst, ptx::Opcode::sub); break;
        case Opcode::vmul:  lower_vector_binary(inst, ptx::Opcode::mul); break;
        case Opcode::vdiv:  lower_vector_binary(inst, ptx::Opcode::div); break;
        case Opcode::vmin:  lower_vector_binary(inst, ptx::Opcode::min); break;
        case Opcode::vmax:  lower_vector_binary(inst, ptx::Opcode::max); break;
        case Opcode::vfma:  lower_vector_fma(inst); break;
        case Opcode::vneg:  lower_vector_unary(inst, ptx::Opcode::neg); break;
        case Opcode::vsqrt: lower_vector_unary(inst, ptx::Opcode::sqrt); break;
        case Opcode::vand:  lower_vector_bitwise(inst, ptx::Opcode::and_); break;
        case Opcode::vor:   lower_vector_bitwise(inst, ptx::Opcode::or_); break;
        case Opcode::vxor:  lower_vector_bitwise(inst, ptx::Opcode::xor_); break;
        case Opcode::vnot:  lower_vector_not(inst); break;
        case Opcode::vbroadcast:    lower_vbroadcast(inst); break;
        case Opcode::vextract_lane: lower_vextract_lane(inst); break;
        case Opcode::vinsert_lane:  lower_vinsert_lane(inst); break;
        case Opcode::vshuffle:      lower_vshuffle(inst); break;
        case Opcode::vzero:         lower_vzero(inst); break;

        // Calls
        case Opcode::call: lower_call(inst); break;

        // No-ops in a GPU kernel
        case Opcode::safepoint:
        case Opcode::write_barrier:
        case Opcode::resume_point:
            break;

        // Terminators
        case Opcode::br:          lower_branch(inst); break;
        case Opcode::br_if:       lower_branch_if(inst); break;
        case Opcode::ret:         lower_return(inst); break;
        case Opcode::unreachable: lower_unreachable(inst); break;

        default:
            throw std::runtime_error("PtxISel: unsupported opcode in PTX lowering: " +
                                     std::string(brass::opcode_name(inst.opcode())));
    }
}

// ---------------------------------------------------------------------------
// Value access and helpers
// ---------------------------------------------------------------------------

const std::vector<Reg>& PtxISel::regs_of(const Value* v, const char* what) const {
    if (!v) throw std::runtime_error(std::string("PtxISel: malformed instruction (missing ") + what + ")");
    auto it = regs_.find(v);
    if (it == regs_.end()) {
        throw std::runtime_error(std::string("PtxISel: ") + what + " refers to a value (%" +
                                 std::to_string(v->id()) + ") with no register");
    }
    return it->second;
}

Reg PtxISel::reg_of(const Value* v, const char* what) const {
    return regs_of(v, what).front();
}

Reg PtxISel::result_reg(const brass::Instruction& inst) const {
    return reg_of(inst.result(), "result");
}

const std::vector<Reg>& PtxISel::result_regs(const brass::Instruction& inst) const {
    return regs_of(inst.result(), "result");
}

Reg PtxISel::pred_of(const Value* v) const {
    auto it = preds_.find(v);
    if (it == preds_.end()) throw std::runtime_error("PtxISel: value is not a comparison result");
    return it->second;
}

bool PtxISel::has_value_uses(const Value* v) const {
    auto it = value_uses_.find(v);
    return it != value_uses_.end() && it->second > 0;
}

Reg PtxISel::materialize_pred(const Value* cond) {
    if (!cond) throw std::runtime_error("PtxISel: malformed instruction (missing condition)");
    auto it = preds_.find(cond);
    if (it != preds_.end()) return it->second;

    Reg p = fn_->new_pred();
    Type t = type_for(cond->type());
    Inst setp = Inst::make(Opcode::setp, t).dst(p).src(reg_of(cond, "condition"));
    if (t == Type::f32)      setp.cmp(CmpOp::neu).src(Operand::imm_f32(0.0f));
    else if (t == Type::f64) setp.cmp(CmpOp::neu).src(Operand::imm_f64(0.0));
    else                     setp.cmp(CmpOp::ne).src(Operand::imm(0));
    emit(std::move(setp));
    return p;
}

Operand PtxISel::operand_of(const Value* v, Opcode op, size_t src_index, Type t, const char* what) {
    if (allows_immediate(op, src_index)) {
        if (is_float(t)) {
            double d = 0.0;
            if (const_float(v, &d)) {
                return t == Type::f32 ? Operand::imm_f32(static_cast<float>(d)) : Operand::imm_f64(d);
            }
        } else {
            int64_t i = 0;
            if (const_int(v, &i) && imm_fits(t, i)) return Operand::imm(i);
        }
    }
    return Operand::reg(reg_of(v, what));
}

Operand PtxISel::shift_amount(const Value* amt) {
    int64_t c = 0;
    if (const_int(amt, &c) && imm_fits(Type::u32, c)) return Operand::imm(c);
    Reg r = reg_of(amt, "shift amount");
    if (r.cls == RegClass::B32) return Operand::reg(r);
    Reg narrow = fn_->new_b32();
    emit(Inst::make(Opcode::cvt, Type::u32).from(Type::u64).dst(narrow).src(r));
    return Operand::reg(narrow);
}

Reg PtxISel::special_register(SpecialReg s) {
    Type t = reg_class_for(s) == RegClass::B64 ? Type::u64 : Type::u32;
    if (!is_invariant(s)) {
        Reg r = fn_->new_reg(reg_class_for(s));
        emit(Inst::make(Opcode::mov, t).dst(r).src(Operand::special(s)));
        return r;
    }
    auto key = static_cast<uint8_t>(s);
    auto it = special_cache_.find(key);
    if (it != special_cache_.end()) return it->second;
    Reg r = fn_->new_reg(reg_class_for(s));
    prologue_block()->append(Inst::make(Opcode::mov, t).dst(r).src(Operand::special(s)).origin(origin_));
    special_cache_[key] = r;
    return r;
}

Operand PtxISel::label_of(const BasicBlock* bb) const {
    if (!bb) throw std::runtime_error("PtxISel: malformed instruction (missing branch target)");
    auto it = blocks_.find(bb);
    if (it == blocks_.end()) throw std::runtime_error("PtxISel: branch to a block outside the function");
    return Operand::label(it->second->label);
}

Inst& PtxISel::emit(Inst inst) {
    if (!inst.mir_origin) inst.mir_origin = origin_;
    return bb_->append(std::move(inst));
}

void PtxISel::malformed(const brass::Instruction& inst, const char* what) const {
    throw std::runtime_error("PtxISel: malformed " + std::string(brass::opcode_name(inst.opcode())) +
                             " (" + what + ")");
}

} // namespace brass::ptx
