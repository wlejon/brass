#pragma once

#include <brass/core/diagnostics.hpp>
#include <cstdint>
#include <string_view>
#include <string>
#include <iosfwd>

namespace brass {

enum class TokenKind : uint16_t {
    Eof,
    Invalid,

    // Keywords
    Kw_func,
    Kw_extern,
    Kw_module,
    Kw_resume_table,
    Kw_entry,
    Kw_ret,
    Kw_br,
    Kw_br_if,
    Kw_guard,
    Kw_resume_point,
    Kw_safepoint,
    Kw_unreachable,

    // Types
    Kw_i32,
    Kw_i64,
    Kw_f32,
    Kw_f64,
    Kw_ptr,
    Kw_gcref,
    Kw_void,
    Kw_f32x4,
    Kw_f64x2,
    Kw_i32x4,
    Kw_i64x2,
    Kw_f32x8,
    Kw_f64x4,
    Kw_i32x8,
    Kw_i64x4,

    // Identifiers and Names
    Ident,          // foo, bb0, loop_header, add.i32, load.i32
    ValueIdent,     // %0, %v0, %arg0
    SymbolIdent,    // @fib, @exit_stub

    // Literals
    IntLiteral,     // 123, -456, 0x1a
    FloatLiteral,   // 3.14, -0.5, 1e-5, 0x1.5p+3, nan, inf
    StringLiteral,  // "hello"

    // Punctuation
    LParen,         // (
    RParen,         // )
    LBrace,         // {
    RBrace,         // }
    LBracket,       // [
    RBracket,       // ]
    Comma,          // ,
    Colon,          // :
    Arrow,          // ->
    Equal           // =
};

std::string_view token_kind_name(TokenKind kind) noexcept;
std::ostream& operator<<(std::ostream& os, TokenKind kind);

struct Token {
    TokenKind kind = TokenKind::Eof;
    std::string_view text;
    SourceLocation location;
    int64_t int_val = 0;
    double float_val = 0.0;

    bool is(TokenKind k) const noexcept { return kind == k; }
    bool is_not(TokenKind k) const noexcept { return kind != k; }
};

class Lexer {
public:
    explicit Lexer(std::string_view source, std::string_view filename = "<input>") noexcept;

    Token next_token();
    Token peek_token();
    bool is_block_header_ahead() const noexcept;

    std::string_view source() const noexcept { return source_; }
    std::string_view filename() const noexcept { return filename_; }

    SourceLocation current_location() const noexcept;

private:
    void skip_whitespace_and_comments();
    Token scan_identifier_or_keyword();
    Token scan_value_identifier();
    Token scan_symbol_identifier();
    Token scan_number_or_minus();
    Token scan_string_literal();

    char peek() const noexcept;
    char peek(size_t offset) const noexcept;
    char advance() noexcept;
    bool is_at_end() const noexcept;

    std::string_view source_;
    std::string_view filename_;
    size_t cursor_ = 0;
    uint32_t line_ = 1;
    uint32_t col_ = 1;

    Token peeked_token_;
    bool has_peeked_ = false;
};

} // namespace brass
