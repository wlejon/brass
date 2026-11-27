#include <brass/mir/instruction.hpp>
#include <brass/mir/block.hpp>
#include <brass/mir/function.hpp>

namespace brass {

static_assert(sizeof(Instruction) <= 136, "mir::Instruction layout regressed");

Instruction::ExtraData& Instruction::ensure_extra() {
    if (!extra_) {
        extra_ = std::make_unique<ExtraData>();
    }
    return *extra_;
}

const BranchTarget& Instruction::empty_branch_target() noexcept {
    static const BranchTarget kEmpty{};
    return kEmpty;
}

const std::vector<SwitchCase>& Instruction::empty_switch_cases() noexcept {
    static const std::vector<SwitchCase> kEmpty{};
    return kEmpty;
}

const std::vector<Value*>& Instruction::empty_state_map() noexcept {
    static const std::vector<Value*> kEmpty{};
    return kEmpty;
}

void Instruction::set_resume_id(uint32_t id) noexcept {
    imm_bits_ = static_cast<int64_t>(id);
    if (opcode() == Opcode::guard && parent_ && parent_->parent()) parent_->parent()->note_guard_resume_id(id);
}

} // namespace brass
