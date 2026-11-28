#pragma once

#include <brass/core/span.hpp>
#include <brass/mir/types.hpp>
#include <brass/mir/opcodes.hpp>
#include <brass/mir/proof_cert.hpp>
#include <brass/debug/source_loc.hpp>
#include <bit>
#include <cstdint>
#include <memory>
#include <string_view>
#include <utility>
#include <vector>

namespace brass {

class BasicBlock;
class Function;
class Instruction;

enum class ValueKind : uint8_t {
    InstructionResult,
    BlockParam
};

class Value {
public:
    Value() noexcept = default;
    Value(uint32_t id, Type type, ValueKind kind) noexcept
        : id_(id), type_(type), kind_(kind) {}

    uint32_t id() const noexcept { return id_; }
    void set_id(uint32_t id) noexcept { id_ = id; }

    Type type() const noexcept { return type_; }
    void set_type(Type type) noexcept { type_ = type; }

    ValueKind kind() const noexcept { return kind_; }
    bool is_block_param() const noexcept { return kind_ == ValueKind::BlockParam; }
    bool is_instruction() const noexcept { return kind_ == ValueKind::InstructionResult; }

    Instruction* defining_instruction() const noexcept { return def_inst_; }
    void set_defining_instruction(Instruction* inst) noexcept {
        def_inst_ = inst;
        kind_ = ValueKind::InstructionResult;
    }

    BasicBlock* defining_block() const noexcept { return def_block_; }
    uint32_t param_index() const noexcept { return param_index_; }
    void set_block_param(BasicBlock* block, uint32_t idx) noexcept {
        def_block_ = block;
        param_index_ = idx;
        kind_ = ValueKind::BlockParam;
    }

    bool is_noalias() const noexcept { return is_noalias_; }
    void set_noalias(bool noalias = true) noexcept { is_noalias_ = noalias; }

private:
    uint32_t id_ = 0;
    Type type_ = Type::void_type();
    ValueKind kind_ = ValueKind::InstructionResult;
    Instruction* def_inst_ = nullptr;
    BasicBlock* def_block_ = nullptr;
    uint32_t param_index_ = 0;
    bool is_noalias_ = false;
};

struct BranchTarget {
    BasicBlock* block = nullptr;
    std::vector<Value*> args;

    BranchTarget() = default;
    explicit BranchTarget(BasicBlock* b) : block(b) {}
    BranchTarget(BasicBlock* b, std::vector<Value*> a) : block(b), args(std::move(a)) {}
};

struct SwitchCase {
    int64_t value = 0;
    BranchTarget target;

    SwitchCase() = default;
    SwitchCase(int64_t v, BasicBlock* b) : value(v), target(b) {}
    SwitchCase(int64_t v, BasicBlock* b, std::vector<Value*> a) : value(v), target(b, std::move(a)) {}
    SwitchCase(int64_t v, BranchTarget t) : value(v), target(std::move(t)) {}
};

class Instruction {
public:
    explicit Instruction(Opcode op = Opcode::unreachable, Type type = Type::void_type()) noexcept
        : opcode_(op), type_(type) {}

    Opcode opcode() const noexcept { return opcode_; }
    void set_opcode(Opcode op) noexcept { opcode_ = op; }

    Type type() const noexcept { return type_; }
    void set_type(Type type) noexcept { type_ = type; }

    Value* result() const noexcept { return result_; }
    void set_result(Value* res) noexcept { result_ = res; }

    BasicBlock* parent() const noexcept { return parent_; }
    void set_parent(BasicBlock* bb) noexcept { parent_ = bb; }

    Instruction* prev() const noexcept { return prev_; }
    void set_prev(Instruction* p) noexcept { prev_ = p; }

    Instruction* next() const noexcept { return next_; }
    void set_next(Instruction* n) noexcept { next_ = n; }

    const std::vector<Value*>& operands() const noexcept { return operands_; }
    std::vector<Value*>& operands() noexcept { return operands_; }
    Value* operand(size_t idx) const noexcept {
        return (idx < operands_.size()) ? operands_[idx] : nullptr;
    }
    size_t operand_count() const noexcept { return operands_.size(); }

    void add_operand(Value* val) { operands_.push_back(val); }
    void set_operand(size_t idx, Value* val) {
        if (idx < operands_.size()) {
            operands_[idx] = val;
        }
    }

    int64_t imm_i64() const noexcept { return imm_bits_; }
    void set_imm_i64(int64_t val) noexcept { imm_bits_ = val; }

    int32_t imm_i32() const noexcept { return static_cast<int32_t>(imm_bits_); }
    void set_imm_i32(int32_t val) noexcept { imm_bits_ = static_cast<int64_t>(val); }

    double imm_f64() const noexcept { return std::bit_cast<double>(imm_bits_); }
    void set_imm_f64(double val) noexcept { imm_bits_ = std::bit_cast<int64_t>(val); }

    uint8_t scale() const noexcept { return scale_; }
    void set_scale(uint8_t scale) noexcept { scale_ = scale; }

    int32_t offset() const noexcept { return offset_; }
    void set_offset(int32_t offset) noexcept { offset_ = offset; }

    Type memory_type() const noexcept { return memory_type_; }
    void set_memory_type(Type t) noexcept { memory_type_ = t; }

    // The memory a load or store touches, as its front end names it: two
    // accesses of different non-zero classes never overlap (a table's
    // header fields and its array's elements). 0 is any memory.
    uint8_t alias_class() const noexcept { return alias_class_; }
    void set_alias_class(uint8_t c) noexcept { alias_class_ = c; }

    // A call its front end wants inlined (a call it has bet the callee of,
    // in a hot path): the inliner weighs the callee against its hinted
    // budget (InlinerOptions::hinted_callee_instruction_count), not the
    // general one.
    bool inline_hint() const noexcept { return inline_hint_; }
    void set_inline_hint(bool hint = true) noexcept { inline_hint_ = hint; }

    // An `alloca` whose contents never live across a guard (scratch, such as
    // a buffer a call stages its arguments in): a guard's state need not
    // carry it, and a resume twin makes one of its own (resume_twin.hpp).
    bool fresh_in_twin() const noexcept { return fresh_in_twin_; }
    void set_fresh_in_twin(bool fresh = true) noexcept { fresh_in_twin_ = fresh; }

    // A load (e.g. a runtime object's verified internal header field) that
    // may safely be hoisted past in-loop bounds checks whose bound came from an
    // already-invariant load on the same object, provided no in-loop test of
    // the object's own identity/kind remains before it.
    bool non_trapping_load() const noexcept { return non_trapping_load_; }
    void set_non_trapping_load(bool non_trapping = true) noexcept { non_trapping_load_ = non_trapping; }

    bool no_contraction() const noexcept { return no_contraction_; }
    void set_no_contraction(bool on = true) noexcept { no_contraction_ = on; }

    // The plain attributes a copy of `src` keeps: immediates, scale, offset,
    // memory type, alias class, inline hint, fresh-in-twin, non-trapping-load,
    // no-contraction (not symbols, which live in a module's pool, nor the debug
    // location or a barrier certificate).
    void copy_attributes_from(const Instruction& src) noexcept {
        imm_bits_ = src.imm_bits_;
        scale_ = src.scale_;
        offset_ = src.offset_;
        memory_type_ = src.memory_type_;
        alias_class_ = src.alias_class_;
        inline_hint_ = src.inline_hint_;
        fresh_in_twin_ = src.fresh_in_twin_;
        non_trapping_load_ = src.non_trapping_load_;
        no_contraction_ = src.no_contraction_;
    }

    bool has_extra() const noexcept { return extra_ != nullptr; }

    std::string_view symbol() const noexcept { return symbol_; }
    void set_symbol(std::string_view sym) noexcept { symbol_ = sym; }

    // A patchable_call's default target. On a `call` or `invoke`: the call's
    // entry cell, a word of the image that holds the address called (x64:
    // `call [rip + cell]`). Whatever the cell holds computes what the named
    // callee computes, which is all the passes read; a front end stores
    // another entry of the callee there (a resume twin's generic entry,
    // docs/speculation_and_deopt.md 7.3.7).
    std::string_view extra_symbol() const noexcept {
        return extra_ ? extra_->extra_symbol : std::string_view{};
    }
    void set_extra_symbol(std::string_view sym) {
        if (sym.empty() && !extra_) return;
        ensure_extra().extra_symbol = sym;
    }

    uint32_t resume_id() const noexcept { return static_cast<uint32_t>(imm_bits_); }
    // A guard's id is reported to its function (Function::
    // note_guard_resume_id), which hands out the next fresh one.
    void set_resume_id(uint32_t id) noexcept;

    uint32_t site_id() const noexcept { return static_cast<uint32_t>(imm_bits_); }
    void set_site_id(uint32_t id) noexcept { imm_bits_ = static_cast<int64_t>(id); }

    uint32_t lane() const noexcept { return static_cast<uint32_t>(imm_bits_); }
    void set_lane(uint32_t l) noexcept { imm_bits_ = static_cast<int64_t>(l); }

    uint32_t shuffle_mask() const noexcept { return static_cast<uint32_t>(imm_bits_); }
    void set_shuffle_mask(uint32_t m) noexcept { imm_bits_ = static_cast<int64_t>(m); }

    const BranchTarget& branch_target() const noexcept {
        return extra_ ? extra_->branch_target : empty_branch_target();
    }
    BranchTarget& branch_target() { return ensure_extra().branch_target; }
    void set_branch_target(BranchTarget target) { ensure_extra().branch_target = std::move(target); }

    const BranchTarget& default_target() const noexcept { return branch_target(); }
    BranchTarget& default_target() { return branch_target(); }
    void set_default_target(BranchTarget target) { set_branch_target(std::move(target)); }

    const std::vector<SwitchCase>& switch_cases() const noexcept {
        return extra_ ? extra_->switch_cases : empty_switch_cases();
    }
    std::vector<SwitchCase>& switch_cases() { return ensure_extra().switch_cases; }
    void add_switch_case(int64_t value, BranchTarget target) {
        ensure_extra().switch_cases.push_back(SwitchCase(value, std::move(target)));
    }
    void add_switch_case(int64_t value, BasicBlock* block, std::vector<Value*> args = {}) {
        ensure_extra().switch_cases.push_back(SwitchCase(value, block, std::move(args)));
    }

    const BranchTarget& true_target() const noexcept {
        return extra_ ? extra_->true_target : empty_branch_target();
    }
    BranchTarget& true_target() { return ensure_extra().true_target; }
    void set_true_target(BranchTarget target) { ensure_extra().true_target = std::move(target); }

    const BranchTarget& false_target() const noexcept {
        return extra_ ? extra_->false_target : empty_branch_target();
    }
    BranchTarget& false_target() { return ensure_extra().false_target; }
    void set_false_target(BranchTarget target) { ensure_extra().false_target = std::move(target); }

    const BranchTarget& normal_target() const noexcept { return true_target(); }
    BranchTarget& normal_target() { return true_target(); }
    void set_normal_target(BranchTarget target) { set_true_target(std::move(target)); }

    const BranchTarget& unwind_target() const noexcept { return false_target(); }
    BranchTarget& unwind_target() { return false_target(); }
    void set_unwind_target(BranchTarget target) { set_false_target(std::move(target)); }

    const std::vector<Value*>& state_map() const noexcept {
        return extra_ ? extra_->state_map : empty_state_map();
    }
    std::vector<Value*>& state_map() { return ensure_extra().state_map; }
    void add_state_value(Value* val) { ensure_extra().state_map.push_back(val); }

    bool is_terminator() const noexcept { return brass::is_terminator(opcode_); }
    bool is_branch() const noexcept { return brass::is_branch(opcode_); }
    bool is_call() const noexcept { return brass::is_call(opcode_); }
    bool has_side_effects() const noexcept { return brass::has_side_effects(opcode_); }
    bool produces_value() const noexcept { return result_ != nullptr && !type_.is_void(); }

    DebugLoc loc() const noexcept { return loc_; }
    void set_loc(DebugLoc loc) noexcept { loc_ = loc; }
    void set_loc(uint32_t file_id, uint32_t line, uint32_t col = 0, uint32_t inlined = 0) noexcept {
        loc_ = DebugLoc(file_id, line, col, inlined);
    }

    BarrierCertKind barrier_cert() const noexcept { return barrier_cert_; }
    void set_barrier_cert(BarrierCertKind cert) noexcept { barrier_cert_ = cert; }

private:
    struct ExtraData {
        BranchTarget branch_target;
        BranchTarget true_target;
        BranchTarget false_target;
        std::vector<SwitchCase> switch_cases;
        std::vector<Value*> state_map;
        std::string_view extra_symbol;
    };

    ExtraData& ensure_extra();
    static const BranchTarget& empty_branch_target() noexcept;
    static const std::vector<SwitchCase>& empty_switch_cases() noexcept;
    static const std::vector<Value*>& empty_state_map() noexcept;

    Value* result_ = nullptr;
    BasicBlock* parent_ = nullptr;
    Instruction* prev_ = nullptr;
    Instruction* next_ = nullptr;
    int64_t imm_bits_ = 0;

    std::string_view symbol_;
    std::vector<Value*> operands_;
    std::unique_ptr<ExtraData> extra_;
    DebugLoc loc_;

    int32_t offset_ = 0;
    Opcode opcode_ = Opcode::unreachable;
    Type type_ = Type::void_type();
    Type memory_type_ = Type::void_type();
    uint8_t scale_ = 1;
    uint8_t alias_class_ = 0;
    bool inline_hint_ = false;
    bool fresh_in_twin_ = false;
    bool non_trapping_load_ = false;
    bool no_contraction_ = false;
    BarrierCertKind barrier_cert_ = BarrierCertKind::None;
};

} // namespace brass
