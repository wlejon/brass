#include <brass/mir/lexer.hpp>
#include <cctype>
#include <cstdlib>
#include <cmath>
#include <string>
#include <ostream>

namespace brass {

std::string_view token_kind_name(TokenKind kind) noexcept {
    switch (kind) {
        case TokenKind::Eof: return "EOF";
        case TokenKind::Invalid: return "invalid";
        case TokenKind::Kw_func: return "func";
        case TokenKind::Kw_extern: return "extern";
        case TokenKind::Kw_module: return "module";
        case TokenKind::Kw_resume_table: return "resume_table";
        case TokenKind::Kw_entry: return "entry";
        case TokenKind::Kw_ret: return "ret";
        case TokenKind::Kw_br: return "br";
        case TokenKind::Kw_br_if: return "br_if";
        case TokenKind::Kw_guard: return "guard";
        case TokenKind::Kw_resume_point: return "resume_point";
        case TokenKind::Kw_safepoint: return "safepoint";
        case TokenKind::Kw_unreachable: return "unreachable";
        case TokenKind::Kw_i32: return "i32";
        case TokenKind::Kw_i64: return "i64";
        case TokenKind::Kw_f32: return "f32";
        case TokenKind::Kw_f64: return "f64";
        case TokenKind::Kw_ptr: return "ptr";
        case TokenKind::Kw_gcref: return "gcref";
        case TokenKind::Kw_void: return "void";
        case TokenKind::Kw_f32x4: return "f32x4";
        case TokenKind::Kw_f64x2: return "f64x2";
        case TokenKind::Kw_i32x4: return "i32x4";
        case TokenKind::Kw_i64x2: return "i64x2";
        case TokenKind::Ident: return "identifier";
        case TokenKind::ValueIdent: return "value identifier";
        case TokenKind::SymbolIdent: return "symbol identifier";
        case TokenKind::IntLiteral: return "integer literal";
        case TokenKind::FloatLiteral: return "float literal";
        case TokenKind::StringLiteral: return "string literal";
        case TokenKind::LParen: return "(";
        case TokenKind::RParen: return ")";
        case TokenKind::LBrace: return "{";
        case TokenKind::RBrace: return "}";
        case TokenKind::LBracket: return "[";
        case TokenKind::RBracket: return "]";
        case TokenKind::Comma: return ",";
        case TokenKind::Colon: return ":";
        case TokenKind::Arrow: return "->";
        case TokenKind::Equal: return "=";
    }
    return "unknown";
}

std::ostream& operator<<(std::ostream& os, TokenKind kind) {
    return os << token_kind_name(kind);
}

Lexer::Lexer(std::string_view source, std::string_view filename) noexcept
    : source_(source), filename_(filename), cursor_(0), line_(1), col_(1) {}

SourceLocation Lexer::current_location() const noexcept {
    return SourceLocation(std::string(filename_), line_, col_);
}

char Lexer::peek() const noexcept {
    if (cursor_ >= source_.size()) return '\0';
    return source_[cursor_];
}

char Lexer::peek(size_t offset) const noexcept {
    if (cursor_ + offset >= source_.size()) return '\0';
    return source_[cursor_ + offset];
}

char Lexer::advance() noexcept {
    if (cursor_ >= source_.size()) return '\0';
    char c = source_[cursor_++];
    if (c == '\n') {
        line_++;
        col_ = 1;
    } else {
        col_++;
    }
    return c;
}

bool Lexer::is_at_end() const noexcept {
    return cursor_ >= source_.size();
}

void Lexer::skip_whitespace_and_comments() {
    while (!is_at_end()) {
        char c = peek();
        if (c == ' ' || c == '\t' || c == '\r' || c == '\n') {
            advance();
        } else if (c == ';') {
            // Line comment with ';'
            while (!is_at_end() && peek() != '\n') {
                advance();
            }
        } else if (c == '/' && peek(1) == '/') {
            // Line comment with '//'
            advance();
            advance();
            while (!is_at_end() && peek() != '\n') {
                advance();
            }
        } else {
            break;
        }
    }
}

Token Lexer::peek_token() {
    if (!has_peeked_) {
        peeked_token_ = next_token();
        has_peeked_ = true;
    }
    return peeked_token_;
}

Token Lexer::next_token() {
    if (has_peeked_) {
        has_peeked_ = false;
        return peeked_token_;
    }

    skip_whitespace_and_comments();

    if (is_at_end()) {
        Token tok;
        tok.kind = TokenKind::Eof;
        tok.text = "";
        tok.location = current_location();
        return tok;
    }

    SourceLocation loc = current_location();
    char c = peek();

    // Check punctuation
    if (c == '(') { advance(); return Token{TokenKind::LParen, source_.substr(cursor_ - 1, 1), loc, 0, 0.0}; }
    if (c == ')') { advance(); return Token{TokenKind::RParen, source_.substr(cursor_ - 1, 1), loc, 0, 0.0}; }
    if (c == '{') { advance(); return Token{TokenKind::LBrace, source_.substr(cursor_ - 1, 1), loc, 0, 0.0}; }
    if (c == '}') { advance(); return Token{TokenKind::RBrace, source_.substr(cursor_ - 1, 1), loc, 0, 0.0}; }
    if (c == '[') { advance(); return Token{TokenKind::LBracket, source_.substr(cursor_ - 1, 1), loc, 0, 0.0}; }
    if (c == ']') { advance(); return Token{TokenKind::RBracket, source_.substr(cursor_ - 1, 1), loc, 0, 0.0}; }
    if (c == ',') { advance(); return Token{TokenKind::Comma, source_.substr(cursor_ - 1, 1), loc, 0, 0.0}; }
    if (c == ':') { advance(); return Token{TokenKind::Colon, source_.substr(cursor_ - 1, 1), loc, 0, 0.0}; }
    if (c == '=') { advance(); return Token{TokenKind::Equal, source_.substr(cursor_ - 1, 1), loc, 0, 0.0}; }

    if (c == '-' && peek(1) == '>') {
        advance();
        advance();
        return Token{TokenKind::Arrow, source_.substr(cursor_ - 2, 2), loc, 0, 0.0};
    }

    if (c == '-' && peek(1) == 'i' && peek(2) == 'n' && peek(3) == 'f') {
        size_t start = cursor_;
        advance(); advance(); advance(); advance();
        if (peek() == 'i' && peek(1) == 'n' && peek(2) == 'i' && peek(3) == 't' && peek(4) == 'y') {
            for (int i = 0; i < 5; ++i) advance();
        }
        std::string_view text = source_.substr(start, cursor_ - start);
        return Token{TokenKind::FloatLiteral, text, loc, 0, -HUGE_VAL};
    }

    if (c == '-' && peek(1) == 'n' && peek(2) == 'a' && peek(3) == 'n') {
        size_t start = cursor_;
        advance(); advance(); advance(); advance();
        std::string_view text = source_.substr(start, cursor_ - start);
        return Token{TokenKind::FloatLiteral, text, loc, 0, -std::nan("")};
    }

    if (c == '%') {
        return scan_value_identifier();
    }

    if (c == '@') {
        return scan_symbol_identifier();
    }

    if (c == '"') {
        return scan_string_literal();
    }

    if (std::isdigit(static_cast<unsigned char>(c)) || (c == '-' && (std::isdigit(static_cast<unsigned char>(peek(1))) || (peek(1) == '.' && std::isdigit(static_cast<unsigned char>(peek(2))))))) {
        return scan_number_or_minus();
    }

    if (std::isalpha(static_cast<unsigned char>(c)) || c == '_') {
        return scan_identifier_or_keyword();
    }

    // Unrecognized character
    advance();
    return Token{TokenKind::Invalid, source_.substr(cursor_ - 1, 1), loc, 0, 0.0};
}

Token Lexer::scan_value_identifier() {
    SourceLocation loc = current_location();
    size_t start = cursor_;
    advance(); // Consume '%'

    while (!is_at_end()) {
        char c = peek();
        if (std::isalnum(static_cast<unsigned char>(c)) || c == '_' || c == '.') {
            advance();
        } else {
            break;
        }
    }

    std::string_view text = source_.substr(start, cursor_ - start);
    return Token{TokenKind::ValueIdent, text, loc, 0, 0.0};
}

Token Lexer::scan_symbol_identifier() {
    SourceLocation loc = current_location();
    size_t start = cursor_;
    advance(); // Consume '@'

    while (!is_at_end()) {
        char c = peek();
        if (std::isalnum(static_cast<unsigned char>(c)) || c == '_' || c == '.' || c == '-') {
            advance();
        } else {
            break;
        }
    }

    std::string_view text = source_.substr(start, cursor_ - start);
    return Token{TokenKind::SymbolIdent, text, loc, 0, 0.0};
}

Token Lexer::scan_string_literal() {
    SourceLocation loc = current_location();
    size_t start = cursor_;
    advance(); // Consume opening quote '"'

    while (!is_at_end() && peek() != '"') {
        if (peek() == '\\' && !is_at_end()) {
            advance();
        }
        advance();
    }

    if (!is_at_end() && peek() == '"') {
        advance(); // Consume closing quote '"'
    }

    std::string_view text = source_.substr(start, cursor_ - start);
    return Token{TokenKind::StringLiteral, text, loc, 0, 0.0};
}

Token Lexer::scan_identifier_or_keyword() {
    SourceLocation loc = current_location();
    size_t start = cursor_;

    while (!is_at_end()) {
        char c = peek();
        if (std::isalnum(static_cast<unsigned char>(c)) || c == '_' || c == '.') {
            advance();
        } else {
            break;
        }
    }

    std::string_view text = source_.substr(start, cursor_ - start);

    // Check special float values
    if (text == "nan") {
        return Token{TokenKind::FloatLiteral, text, loc, 0, std::nan("")};
    }
    if (text == "inf" || text == "infinity") {
        return Token{TokenKind::FloatLiteral, text, loc, 0, HUGE_VAL};
    }

    // Check keywords
    if (text == "func") return Token{TokenKind::Kw_func, text, loc, 0, 0.0};
    if (text == "extern") return Token{TokenKind::Kw_extern, text, loc, 0, 0.0};
    if (text == "module") return Token{TokenKind::Kw_module, text, loc, 0, 0.0};
    if (text == "resume_table") return Token{TokenKind::Kw_resume_table, text, loc, 0, 0.0};
    if (text == "entry") return Token{TokenKind::Kw_entry, text, loc, 0, 0.0};
    if (text == "ret") return Token{TokenKind::Kw_ret, text, loc, 0, 0.0};
    if (text == "br") return Token{TokenKind::Kw_br, text, loc, 0, 0.0};
    if (text == "br_if") return Token{TokenKind::Kw_br_if, text, loc, 0, 0.0};
    if (text == "guard") return Token{TokenKind::Kw_guard, text, loc, 0, 0.0};
    if (text == "resume_point") return Token{TokenKind::Kw_resume_point, text, loc, 0, 0.0};
    if (text == "safepoint") return Token{TokenKind::Kw_safepoint, text, loc, 0, 0.0};
    if (text == "unreachable") return Token{TokenKind::Kw_unreachable, text, loc, 0, 0.0};

    // Types
    if (text == "i32") return Token{TokenKind::Kw_i32, text, loc, 0, 0.0};
    if (text == "i64") return Token{TokenKind::Kw_i64, text, loc, 0, 0.0};
    if (text == "f32") return Token{TokenKind::Kw_f32, text, loc, 0, 0.0};
    if (text == "f64") return Token{TokenKind::Kw_f64, text, loc, 0, 0.0};
    if (text == "ptr") return Token{TokenKind::Kw_ptr, text, loc, 0, 0.0};
    if (text == "gcref") return Token{TokenKind::Kw_gcref, text, loc, 0, 0.0};
    if (text == "void") return Token{TokenKind::Kw_void, text, loc, 0, 0.0};
    if (text == "f32x4") return Token{TokenKind::Kw_f32x4, text, loc, 0, 0.0};
    if (text == "f64x2") return Token{TokenKind::Kw_f64x2, text, loc, 0, 0.0};
    if (text == "i32x4") return Token{TokenKind::Kw_i32x4, text, loc, 0, 0.0};
    if (text == "i64x2") return Token{TokenKind::Kw_i64x2, text, loc, 0, 0.0};

    return Token{TokenKind::Ident, text, loc, 0, 0.0};
}

Token Lexer::scan_number_or_minus() {
    SourceLocation loc = current_location();
    size_t start = cursor_;

    if (peek() == '-') {
        advance();
    }

    bool is_hex = false;
    if (peek() == '0' && (peek(1) == 'x' || peek(1) == 'X')) {
        is_hex = true;
        advance();
        advance();
    }

    bool is_float = false;

    if (is_hex) {
        // Hexadecimal integer or hex float (e.g. 0x1.5p+3)
        while (!is_at_end()) {
            char c = peek();
            if (std::isxdigit(static_cast<unsigned char>(c))) {
                advance();
            } else if (c == '.') {
                is_float = true;
                advance();
            } else if (c == 'p' || c == 'P') {
                is_float = true;
                advance();
                if (peek() == '+' || peek() == '-') {
                    advance();
                }
            } else {
                break;
            }
        }
    } else {
        // Decimal integer or decimal float
        while (!is_at_end()) {
            char c = peek();
            if (std::isdigit(static_cast<unsigned char>(c))) {
                advance();
            } else if (c == '.') {
                // Ensure dot is part of number and not something else
                is_float = true;
                advance();
            } else if (c == 'e' || c == 'E') {
                is_float = true;
                advance();
                if (peek() == '+' || peek() == '-') {
                    advance();
                }
            } else {
                break;
            }
        }
    }

    std::string_view text = source_.substr(start, cursor_ - start);
    std::string str(text);

    if (is_float) {
        char* endptr = nullptr;
        double fval = std::strtod(str.c_str(), &endptr);
        return Token{TokenKind::FloatLiteral, text, loc, 0, fval};
    } else {
        char* endptr = nullptr;
        int base = is_hex ? 16 : 10;
        int64_t ival = std::strtoll(str.c_str(), &endptr, base);
        return Token{TokenKind::IntLiteral, text, loc, ival, static_cast<double>(ival)};
    }
}

bool Lexer::is_block_header_ahead() const noexcept {
    size_t pos = cursor_;
    // Skip identifier characters if cursor is not at end of token
    if (!has_peeked_) {
        while (pos < source_.size() && (std::isalnum(static_cast<unsigned char>(source_[pos])) || source_[pos] == '_' || source_[pos] == '.')) {
            pos++;
        }
    }

    // Skip whitespace and comments
    while (pos < source_.size()) {
        char c = source_[pos];
        if (c == ' ' || c == '\t' || c == '\r' || c == '\n') {
            pos++;
        } else if (c == ';' || (c == '/' && pos + 1 < source_.size() && source_[pos + 1] == '/')) {
            while (pos < source_.size() && source_[pos] != '\n') {
                pos++;
            }
        } else {
            break;
        }
    }

    if (pos < source_.size() && source_[pos] == ':') {
        return true;
    }

    if (pos < source_.size() && source_[pos] == '(') {
        int depth = 0;
        while (pos < source_.size()) {
            if (source_[pos] == '(') depth++;
            else if (source_[pos] == ')') {
                depth--;
                if (depth == 0) {
                    pos++;
                    break;
                }
            }
            pos++;
        }

        while (pos < source_.size()) {
            char c = source_[pos];
            if (c == ' ' || c == '\t' || c == '\r' || c == '\n') {
                pos++;
            } else if (c == ';' || (c == '/' && pos + 1 < source_.size() && source_[pos + 1] == '/')) {
                while (pos < source_.size() && source_[pos] != '\n') {
                    pos++;
                }
            } else {
                break;
            }
        }

        if (pos < source_.size() && source_[pos] == ':') {
            return true;
        }
    }

    return false;
}

} // namespace brass
