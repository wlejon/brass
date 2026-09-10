#pragma once

#include <brass/mir/instruction.hpp>
#include <brass/mir/builder.hpp>
#include <brass/mir/lexer.hpp>
#include <string>
#include <functional>

namespace brass {

struct ParserCoroContext {
    std::function<Token()> peek;
    std::function<Token()> advance;
    std::function<bool(TokenKind, const std::string&)> expect;
    std::function<bool(TokenKind)> match;
    std::function<Value*()> parse_val;
    std::function<std::string_view()> parse_symbol_name;
    std::function<void(SourceLocation, const std::string&)> error;
};

bool decode_coro_opcode(
    std::string_view str,
    Opcode& op,
    Type& type_suffix
);

bool parse_coro_instruction(
    ParserCoroContext& ctx,
    Opcode op,
    Type type_annotation,
    Type type_suffix,
    Builder& b,
    Value*& res_val,
    Instruction*& res_inst
);

} // namespace brass
