#pragma once

#include <brass/mir/instruction.hpp>
#include <brass/mir/builder.hpp>
#include <brass/mir/lexer.hpp>
#include <string>
#include <functional>

namespace brass {

struct ParserVecContext {
    std::function<Token()> peek;
    std::function<Token()> advance;
    std::function<bool(TokenKind, const std::string&)> expect;
    std::function<bool(TokenKind)> match;
    std::function<Value*()> parse_val;
    std::function<void(SourceLocation, const std::string&)> error;
};

bool decode_vector_opcode(
    std::string_view str,
    Opcode& op,
    Type& type_suffix,
    Type& mem_type
);

bool parse_vector_instruction(
    ParserVecContext& ctx,
    Opcode op,
    Type type_annotation,
    Type type_suffix,
    Type mem_type,
    Builder& b,
    Value*& res_val,
    Instruction*& res_inst
);

} // namespace brass
