#pragma once

#include <brass/mir/types.hpp>
#include <brass/mir/opcodes.hpp>
#include <brass/mir/lexer.hpp>
#include <cstdint>
#include <string>
#include <string_view>

namespace brass {

Type parse_type_from_string(std::string_view s);
bool is_identifier_or_keyword(TokenKind k) noexcept;
bool decode_opcode_string(std::string_view str, Opcode& op, Type& type_suffix, Type& mem_type);

// Range check for the IntLiteral `tok` against a field that holds [lo, hi].
// Decimal literals must be written in range, as the printer writes them
// (`iconst.i32 4294967295` is rejected; the printer writes -1). A hex literal
// may also spell the bit pattern of a `hex_bits`-wide value (0 = none), so
// `iconst.i32 0xFFFFFFFF` is -1. On success stores the value in `out`;
// otherwise sets `err` to an "integer literal out of range" message naming
// `what`.
bool int_literal_in_range(const Token& tok, int64_t lo, int64_t hi, unsigned hex_bits,
                          std::string_view what, int64_t& out, std::string& err);

// The [lo, hi] range of a signed `bits`-wide integer.
constexpr int64_t signed_min_of(unsigned bits) noexcept {
    return bits >= 64 ? INT64_MIN : -(int64_t{1} << (bits - 1));
}
constexpr int64_t signed_max_of(unsigned bits) noexcept {
    return bits >= 64 ? INT64_MAX : (int64_t{1} << (bits - 1)) - 1;
}

} // namespace brass
