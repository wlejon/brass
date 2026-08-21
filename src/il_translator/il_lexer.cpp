#include "il_lexer.hpp"
#include <cctype>
#include <cstdlib>
#include <charconv>

namespace brass::il {

const char* bronze_type_name(BronzeType t) {
    switch (t) {
        case BronzeType::Void: return "void";
        case BronzeType::Bool: return "bool";
        case BronzeType::I32: return "i32";
        case BronzeType::F64: return "f64";
        case BronzeType::Str: return "str";
        case BronzeType::Dynamic: return "dynamic";
        case BronzeType::Unknown: return "?";
    }
    return "?";
}

const char* bronze_op_name(BronzeOp op) {
    switch (op) {
        case BronzeOp::ConstF64: return "const.f64";
        case BronzeOp::ConstI32: return "const.i32";
        case BronzeOp::ConstBool: return "const.bool";
        case BronzeOp::ConstUndefined: return "const.undefined";
        case BronzeOp::ConstNull: return "const.null";
        case BronzeOp::ConstBigInt: return "const.bigint";
        case BronzeOp::Add: return "add";
        case BronzeOp::Sub: return "sub";
        case BronzeOp::Neg: return "neg";
        case BronzeOp::Mul: return "mul";
        case BronzeOp::Div: return "div";
        case BronzeOp::Mod: return "mod";
        case BronzeOp::Pow: return "pow";
        case BronzeOp::BitAnd: return "and";
        case BronzeOp::BitOr: return "or";
        case BronzeOp::BitXor: return "xor";
        case BronzeOp::Shl: return "shl";
        case BronzeOp::Shr: return "shr";
        case BronzeOp::UShr: return "ushr";
        case BronzeOp::BitNot: return "bitnot";
        case BronzeOp::ToInt32: return "to.int32";
        case BronzeOp::ToNumeric: return "to.numeric";
        case BronzeOp::NumericStep: return "numeric.step";
        case BronzeOp::CmpLt: return "cmp.lt";
        case BronzeOp::CmpGt: return "cmp.gt";
        case BronzeOp::CmpLe: return "cmp.le";
        case BronzeOp::CmpGe: return "cmp.ge";
        case BronzeOp::CmpEq: return "cmp.eq";
        case BronzeOp::CmpNe: return "cmp.ne";
        case BronzeOp::StrictEq: return "strict.eq";
        case BronzeOp::LooseEq: return "loose.eq";
        case BronzeOp::RelLt: return "rel.lt";
        case BronzeOp::RelGt: return "rel.gt";
        case BronzeOp::RelLe: return "rel.le";
        case BronzeOp::RelGe: return "rel.ge";
        case BronzeOp::NumTruthy: return "num.truthy";
        case BronzeOp::TypeOf: return "typeof";
        case BronzeOp::ToStr: return "to.string";
        case BronzeOp::Box: return "box";
        case BronzeOp::Unbox: return "unbox";
        case BronzeOp::Call: return "call";
        case BronzeOp::CallDynamic: return "call.dynamic";
        case BronzeOp::NameResolve: return "name.resolve";
        case BronzeOp::EnvCreate: return "env.create";
        case BronzeOp::EnvGet: return "env.get";
        case BronzeOp::EnvSet: return "env.set";
        case BronzeOp::EnvGetTdz: return "env.get.tdz";
        case BronzeOp::EnvInitTdz: return "env.init.tdz";
        case BronzeOp::CreateFunc: return "create.func";
        case BronzeOp::CreateArray: return "create.array";
        case BronzeOp::PropSet: return "prop.set";
        case BronzeOp::ElemGet: return "elem.get";
        case BronzeOp::ElemSet: return "elem.set";
        case BronzeOp::Print: return "print";
        case BronzeOp::PrintErr: return "print.err";
        case BronzeOp::Ret: return "ret";
        case BronzeOp::Jump: return "jump";
        case BronzeOp::Branch: return "br";
        case BronzeOp::Unknown: return "?";
    }
    return "?";
}

IlLexer::IlLexer(std::string_view source)
    : source_(source) {}

const Token& IlLexer::peek_token() {
    if (!has_peeked_) {
        peeked_ = scan_token();
        has_peeked_ = true;
    }
    return peeked_;
}

Token IlLexer::next_token() {
    if (has_peeked_) {
        has_peeked_ = false;
        return peeked_;
    }
    return scan_token();
}

std::string IlLexer::scan_to_eol() {
    skip_whitespace_and_comments();
    size_t start = pos_;
    while (pos_ < source_.size() && source_[pos_] != '\r' && source_[pos_] != '\n') {
        pos_++; col_++;
    }
    has_peeked_ = false;
    return std::string(source_.substr(start, pos_ - start));
}

void IlLexer::skip_whitespace_and_comments() {
    while (pos_ < source_.size()) {
        char c = source_[pos_];
        if (c == ' ' || c == '\t' || c == '\r') {
            pos_++;
            col_++;
        } else if (c == '\n') {
            pos_++;
            line_++;
            col_ = 1;
        } else if (c == '/' && pos_ + 1 < source_.size() && source_[pos_ + 1] == '/') {
            // Line comment
            pos_ += 2;
            col_ += 2;
            while (pos_ < source_.size() && source_[pos_] != '\n') {
                pos_++;
                col_++;
            }
        } else {
            break;
        }
    }
}

Token IlLexer::scan_token() {
    skip_whitespace_and_comments();
    if (pos_ >= source_.size()) {
        Token tok;
        tok.type = TokenType::Eof;
        tok.line = line_;
        tok.col = col_;
        return tok;
    }

    uint32_t start_line = line_;
    uint32_t start_col = col_;
    size_t start_pos = pos_;
    char c = source_[pos_];

    // Single / double char punctuation
    if (c == '-' && pos_ + 1 < source_.size() && source_[pos_ + 1] == '>') {
        pos_ += 2;
        col_ += 2;
        Token tok;
        tok.type = TokenType::Arrow;
        tok.text = source_.substr(start_pos, 2);
        tok.line = start_line;
        tok.col = start_col;
        return tok;
    }

    if (c == ':') {
        pos_++; col_++;
        return Token{TokenType::Colon, source_.substr(start_pos, 1), start_line, start_col};
    }
    if (c == '=') {
        pos_++; col_++;
        return Token{TokenType::Equal, source_.substr(start_pos, 1), start_line, start_col};
    }
    if (c == ',') {
        pos_++; col_++;
        return Token{TokenType::Comma, source_.substr(start_pos, 1), start_line, start_col};
    }
    if (c == '(') {
        pos_++; col_++;
        return Token{TokenType::LParen, source_.substr(start_pos, 1), start_line, start_col};
    }
    if (c == ')') {
        pos_++; col_++;
        return Token{TokenType::RParen, source_.substr(start_pos, 1), start_line, start_col};
    }
    if (c == '{') {
        pos_++; col_++;
        return Token{TokenType::LBrace, source_.substr(start_pos, 1), start_line, start_col};
    }
    if (c == '}') {
        pos_++; col_++;
        return Token{TokenType::RBrace, source_.substr(start_pos, 1), start_line, start_col};
    }

    // Value reference %N
    if (c == '%') {
        pos_++; col_++;
        size_t num_start = pos_;
        while (pos_ < source_.size() && std::isdigit(static_cast<unsigned char>(source_[pos_]))) {
            pos_++; col_++;
        }
        std::string_view num_str = source_.substr(num_start, pos_ - num_start);
        uint32_t val_id = 0;
        std::from_chars(num_str.data(), num_str.data() + num_str.size(), val_id);
        Token tok{TokenType::PercentValue, source_.substr(start_pos, pos_ - start_pos), start_line, start_col};
        tok.id_num = val_id;
        return tok;
    }

    // Function reference @name
    if (c == '@') {
        pos_++; col_++;
        size_t id_start = pos_;
        while (pos_ < source_.size() && (std::isalnum(static_cast<unsigned char>(source_[pos_])) || source_[pos_] == '_' || source_[pos_] == '.' || source_[pos_] == '$')) {
            pos_++; col_++;
        }
        Token tok{TokenType::AtFunction, source_.substr(id_start, pos_ - id_start), start_line, start_col};
        return tok;
    }

    // String literal "..."
    if (c == '"') {
        pos_++; col_++;
        size_t str_start = pos_;
        while (pos_ < source_.size() && source_[pos_] != '"') {
            if (source_[pos_] == '\\' && pos_ + 1 < source_.size()) {
                pos_ += 2; col_ += 2;
            } else {
                if (source_[pos_] == '\n') { line_++; col_ = 1; }
                else { col_++; }
                pos_++;
            }
        }
        size_t str_len = pos_ - str_start;
        if (pos_ < source_.size() && source_[pos_] == '"') {
            pos_++; col_++;
        }
        Token tok{TokenType::StringLiteral, source_.substr(str_start, str_len), start_line, start_col};
        return tok;
    }

    // Number literal
    if (std::isdigit(static_cast<unsigned char>(c)) || (c == '-' && pos_ + 1 < source_.size() && std::isdigit(static_cast<unsigned char>(source_[pos_ + 1])))) {
        return scan_number(c == '-');
    }

    // Identifiers, keywords, block labels bN, opcodes with dots (const.f64, to.int32, cmp.lt)
    size_t word_start = pos_;
    while (pos_ < source_.size()) {
        char wc = source_[pos_];
        if (std::isalnum(static_cast<unsigned char>(wc)) || wc == '_' || wc == '.' || wc == '$') {
            pos_++; col_++;
        } else {
            break;
        }
    }
    std::string_view word = source_.substr(word_start, pos_ - word_start);

    // Block label: b<digits>
    if (word.size() >= 2 && word[0] == 'b' && std::isdigit(static_cast<unsigned char>(word[1]))) {
        bool all_digits = true;
        for (size_t i = 1; i < word.size(); ++i) {
            if (!std::isdigit(static_cast<unsigned char>(word[i]))) {
                all_digits = false;
                break;
            }
        }
        if (all_digits) {
            uint32_t block_id = 0;
            std::from_chars(word.data() + 1, word.data() + word.size(), block_id);
            Token tok{TokenType::BlockLabel, word, start_line, start_col};
            tok.id_num = block_id;
            return tok;
        }
    }

    // Check keywords
    if (word == "module") return Token{TokenType::Module, word, start_line, start_col};
    if (word == "func") return Token{TokenType::Func, word, start_line, start_col};
    if (word == "export") return Token{TokenType::Export, word, start_line, start_col};
    if (word == "handler") return Token{TokenType::Handler, word, start_line, start_col};

    // Types
    if (word == "void") {
        Token tok{TokenType::KeywordType, word, start_line, start_col};
        tok.btype = BronzeType::Void;
        return tok;
    }
    if (word == "bool") {
        Token tok{TokenType::KeywordType, word, start_line, start_col};
        tok.btype = BronzeType::Bool;
        return tok;
    }
    if (word == "i32") {
        Token tok{TokenType::KeywordType, word, start_line, start_col};
        tok.btype = BronzeType::I32;
        return tok;
    }
    if (word == "f64") {
        Token tok{TokenType::KeywordType, word, start_line, start_col};
        tok.btype = BronzeType::F64;
        return tok;
    }
    if (word == "str") {
        Token tok{TokenType::KeywordType, word, start_line, start_col};
        tok.btype = BronzeType::Str;
        return tok;
    }
    if (word == "dynamic") {
        Token tok{TokenType::KeywordType, word, start_line, start_col};
        tok.btype = BronzeType::Dynamic;
        return tok;
    }

    // Box types: box.f64, box.i32, box.bool, box.str, box.dynamic
    if (word.rfind("box.", 0) == 0) {
        std::string_view subtype = word.substr(4);
        Token tok{TokenType::KeywordOp, word, start_line, start_col};
        tok.op = BronzeOp::Box;
        if (subtype == "f64") tok.box_type = BronzeType::F64;
        else if (subtype == "i32") tok.box_type = BronzeType::I32;
        else if (subtype == "bool") tok.box_type = BronzeType::Bool;
        else if (subtype == "str") tok.box_type = BronzeType::Str;
        else tok.box_type = BronzeType::Dynamic;
        return tok;
    }

    // Unbox types: unbox.f64, unbox.i32, unbox.bool, unbox.str
    if (word.rfind("unbox.", 0) == 0) {
        std::string_view subtype = word.substr(6);
        Token tok{TokenType::KeywordOp, word, start_line, start_col};
        tok.op = BronzeOp::Unbox;
        if (subtype == "f64") tok.box_type = BronzeType::F64;
        else if (subtype == "i32") tok.box_type = BronzeType::I32;
        else if (subtype == "bool") tok.box_type = BronzeType::Bool;
        else if (subtype == "str") tok.box_type = BronzeType::Str;
        else tok.box_type = BronzeType::Dynamic;
        return tok;
    }

    // Opcode mapping
    auto match_op = [](std::string_view w, BronzeOp& op_out) -> bool {
        if (w == "const.f64") { op_out = BronzeOp::ConstF64; return true; }
        if (w == "const.i32") { op_out = BronzeOp::ConstI32; return true; }
        if (w == "const.bool") { op_out = BronzeOp::ConstBool; return true; }
        if (w == "const.undefined") { op_out = BronzeOp::ConstUndefined; return true; }
        if (w == "const.null") { op_out = BronzeOp::ConstNull; return true; }
        if (w == "const.bigint") { op_out = BronzeOp::ConstBigInt; return true; }
        if (w == "add") { op_out = BronzeOp::Add; return true; }
        if (w == "sub") { op_out = BronzeOp::Sub; return true; }
        if (w == "neg") { op_out = BronzeOp::Neg; return true; }
        if (w == "mul") { op_out = BronzeOp::Mul; return true; }
        if (w == "div") { op_out = BronzeOp::Div; return true; }
        if (w == "mod") { op_out = BronzeOp::Mod; return true; }
        if (w == "pow") { op_out = BronzeOp::Pow; return true; }
        if (w == "and") { op_out = BronzeOp::BitAnd; return true; }
        if (w == "or") { op_out = BronzeOp::BitOr; return true; }
        if (w == "xor") { op_out = BronzeOp::BitXor; return true; }
        if (w == "shl") { op_out = BronzeOp::Shl; return true; }
        if (w == "shr") { op_out = BronzeOp::Shr; return true; }
        if (w == "ushr") { op_out = BronzeOp::UShr; return true; }
        if (w == "bitnot") { op_out = BronzeOp::BitNot; return true; }
        if (w == "to.int32") { op_out = BronzeOp::ToInt32; return true; }
        if (w == "to.numeric") { op_out = BronzeOp::ToNumeric; return true; }
        if (w == "numeric.step") { op_out = BronzeOp::NumericStep; return true; }
        if (w == "cmp.lt") { op_out = BronzeOp::CmpLt; return true; }
        if (w == "cmp.gt") { op_out = BronzeOp::CmpGt; return true; }
        if (w == "cmp.le") { op_out = BronzeOp::CmpLe; return true; }
        if (w == "cmp.ge") { op_out = BronzeOp::CmpGe; return true; }
        if (w == "cmp.eq") { op_out = BronzeOp::CmpEq; return true; }
        if (w == "cmp.ne") { op_out = BronzeOp::CmpNe; return true; }
        if (w == "strict.eq") { op_out = BronzeOp::StrictEq; return true; }
        if (w == "loose.eq") { op_out = BronzeOp::LooseEq; return true; }
        if (w == "rel.lt") { op_out = BronzeOp::RelLt; return true; }
        if (w == "rel.gt") { op_out = BronzeOp::RelGt; return true; }
        if (w == "rel.le") { op_out = BronzeOp::RelLe; return true; }
        if (w == "rel.ge") { op_out = BronzeOp::RelGe; return true; }
        if (w == "num.truthy") { op_out = BronzeOp::NumTruthy; return true; }
        if (w == "typeof") { op_out = BronzeOp::TypeOf; return true; }
        if (w == "to.string") { op_out = BronzeOp::ToStr; return true; }
        if (w == "box") { op_out = BronzeOp::Box; return true; }
        if (w == "unbox") { op_out = BronzeOp::Unbox; return true; }
        if (w == "call") { op_out = BronzeOp::Call; return true; }
        if (w == "call.dynamic") { op_out = BronzeOp::CallDynamic; return true; }
        if (w == "name.resolve") { op_out = BronzeOp::NameResolve; return true; }
        if (w == "env.create") { op_out = BronzeOp::EnvCreate; return true; }
        if (w == "env.get") { op_out = BronzeOp::EnvGet; return true; }
        if (w == "env.set") { op_out = BronzeOp::EnvSet; return true; }
        if (w == "env.get.tdz") { op_out = BronzeOp::EnvGetTdz; return true; }
        if (w == "env.init.tdz") { op_out = BronzeOp::EnvInitTdz; return true; }
        if (w == "create.func") { op_out = BronzeOp::CreateFunc; return true; }
        if (w == "create.array") { op_out = BronzeOp::CreateArray; return true; }
        if (w == "prop.set") { op_out = BronzeOp::PropSet; return true; }
        if (w == "elem.get") { op_out = BronzeOp::ElemGet; return true; }
        if (w == "elem.set") { op_out = BronzeOp::ElemSet; return true; }
        if (w == "print") { op_out = BronzeOp::Print; return true; }
        if (w == "print.err") { op_out = BronzeOp::PrintErr; return true; }
        if (w == "ret") { op_out = BronzeOp::Ret; return true; }
        if (w == "jump") { op_out = BronzeOp::Jump; return true; }
        if (w == "br") { op_out = BronzeOp::Branch; return true; }
        return false;
    };

    BronzeOp op = BronzeOp::Unknown;
    if (match_op(word, op)) {
        Token tok{TokenType::KeywordOp, word, start_line, start_col};
        tok.op = op;
        return tok;
    }

    return Token{TokenType::Identifier, word, start_line, start_col};
}

Token IlLexer::scan_number(bool negative) {
    uint32_t start_line = line_;
    uint32_t start_col = col_;
    size_t start_pos = pos_;

    if (negative) {
        pos_++; col_++;
    }

    bool has_dot = false;
    bool has_exp = false;

    while (pos_ < source_.size()) {
        char c = source_[pos_];
        if (std::isdigit(static_cast<unsigned char>(c))) {
            pos_++; col_++;
        } else if (c == '.' && !has_dot && !has_exp && pos_ + 1 < source_.size() && std::isdigit(static_cast<unsigned char>(source_[pos_ + 1]))) {
            has_dot = true;
            pos_++; col_++;
        } else if ((c == 'e' || c == 'E') && !has_exp) {
            has_exp = true;
            pos_++; col_++;
            if (pos_ < source_.size() && (source_[pos_] == '+' || source_[pos_] == '-')) {
                pos_++; col_++;
            }
        } else {
            break;
        }
    }

    std::string_view num_str = source_.substr(start_pos, pos_ - start_pos);
    std::string temp(num_str);

    Token tok;
    tok.text = num_str;
    tok.line = start_line;
    tok.col = start_col;

    if (has_dot || has_exp) {
        tok.type = TokenType::NumberFloat;
        tok.num_f64 = std::strtod(temp.c_str(), nullptr);
        tok.num_i64 = static_cast<int64_t>(tok.num_f64);
    } else {
        tok.type = TokenType::NumberInt;
        tok.num_i64 = std::strtoll(temp.c_str(), nullptr, 10);
        tok.num_f64 = static_cast<double>(tok.num_i64);
    }
    return tok;
}

} // namespace brass::il
