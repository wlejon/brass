#pragma once

#include "il_ast.hpp"
#include <string_view>
#include <string>
#include <vector>

namespace brass::il {

enum class TokenType {
    Eof,
    Module,
    Census,
    Func,
    Export,
    Handler,
    Arrow,          // ->
    Colon,          // :
    Equal,          // =
    Comma,          // ,
    LParen,         // (
    RParen,         // )
    LBrace,         // {
    RBrace,         // }
    LBracket,       // [
    RBracket,       // ]
    PercentValue,   // %0, %12
    BlockLabel,     // b0, b1
    AtFunction,     // @fn_name
    Identifier,     // identifier word
    NumberFloat,    // floating point literal
    NumberInt,      // integer literal
    StringLiteral,  // "..."
    KeywordOp,      // instruction opcode
    KeywordType     // type keyword
};

struct Token {
    TokenType type = TokenType::Eof;
    std::string_view text;
    uint32_t line = 1;
    uint32_t col = 1;
    uint32_t id_num = 0;       // For %N or bN
    double num_f64 = 0.0;
    int64_t num_i64 = 0;
    BronzeOp op = BronzeOp::Unknown;
    BronzeType btype = BronzeType::Unknown;
    BronzeType box_type = BronzeType::Unknown;
};

class IlLexer {
public:
    explicit IlLexer(std::string_view source);

    Token next_token();
    const Token& peek_token();
    std::string scan_to_eol();

    bool is_eof() const noexcept { return pos_ >= source_.size() && peeked_.type == TokenType::Eof; }

private:
    void skip_whitespace_and_comments();
    Token scan_token();
    Token scan_number(bool negative);

    std::string_view source_;
    size_t pos_ = 0;
    uint32_t line_ = 1;
    uint32_t col_ = 1;

    Token peeked_;
    bool has_peeked_ = false;
};

} // namespace brass::il
