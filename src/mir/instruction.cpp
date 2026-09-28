#include <brass/mir/instruction.hpp>
#include <brass/mir/block.hpp>
#include <brass/mir/function.hpp>

namespace brass {

void Instruction::set_resume_id(uint32_t id) noexcept {
    imm_i64_ = static_cast<int64_t>(id);
    if (opcode() == Opcode::guard && parent_ && parent_->parent()) parent_->parent()->note_guard_resume_id(id);
}

} // namespace brass
