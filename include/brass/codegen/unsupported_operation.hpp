#pragma once

#include <stdexcept>
#include <string>
#include <string_view>

// The error channel for code generation gaps. Instruction selection and the
// emitters throw this for an operation they have no lowering for, instead of
// quietly selecting a trap, emitting a ud2, or emitting nothing: a gap is a
// compile-time error that names the stage and the operation, never a program
// that traps (or silently computes garbage) at run time.
//
// MIR `unreachable` is not a gap: it lowers to a trap by definition.
namespace brass::codegen {

class UnsupportedOperation : public std::runtime_error {
public:
    UnsupportedOperation(std::string_view stage, std::string_view operation)
        : std::runtime_error(std::string(stage) + ": unsupported operation '" + std::string(operation) + "'"),
          stage_(stage), operation_(operation) {}

    // e.g. "x64 isel", "aarch64 emit"
    const std::string& stage() const noexcept { return stage_; }
    // The MIR or LIR opcode name.
    const std::string& operation() const noexcept { return operation_; }

private:
    std::string stage_;
    std::string operation_;
};

[[noreturn]] inline void throw_unsupported(std::string_view stage, std::string_view operation) {
    throw UnsupportedOperation(stage, operation);
}

} // namespace brass::codegen
