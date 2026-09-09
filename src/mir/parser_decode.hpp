#pragma once

#include <brass/mir/types.hpp>
#include <brass/mir/opcodes.hpp>
#include <brass/mir/lexer.hpp>
#include <string_view>

namespace brass {

Type parse_type_from_string(std::string_view s);
bool is_identifier_or_keyword(TokenKind k) noexcept;
bool decode_opcode_string(std::string_view str, Opcode& op, Type& type_suffix, Type& mem_type);

} // namespace brass
