#include "il_parser.hpp"
#include <sstream>

namespace brass::il {

IlParser::IlParser(std::string_view source, DiagnosticReporter* diag)
    : lexer_(source), diag_(diag) {}

void IlParser::error(std::string_view msg, const Token& tok) {
    has_error_ = true;
    if (diag_) {
        diag_->error(SourceLocation("", tok.line, tok.col), std::string(msg));
    }
}

bool IlParser::match(TokenType type, Token* out_tok) {
    const Token& tok = lexer_.peek_token();
    if (tok.type == type) {
        if (out_tok) *out_tok = tok;
        lexer_.next_token();
        return true;
    }
    return false;
}

bool IlParser::expect(TokenType type, std::string_view err_msg, Token* out_tok) {
    const Token& tok = lexer_.peek_token();
    if (tok.type == type) {
        if (out_tok) *out_tok = tok;
        lexer_.next_token();
        return true;
    }
    error(err_msg, tok);
    return false;
}

bool IlParser::parse_type(BronzeType& out_type) {
    Token tok;
    if (match(TokenType::KeywordType, &tok)) {
        out_type = tok.btype;
        return true;
    }
    const Token& peek = lexer_.peek_token();
    error("Expected type (void, bool, i32, f64, str, dynamic)", peek);
    return false;
}

bool IlParser::parse_block_target(BronzeBlockTarget& out_target) {
    Token tok;
    if (!expect(TokenType::BlockLabel, "Expected block label (e.g. b0, b1)", &tok)) {
        return false;
    }
    out_target.block_id = tok.id_num;
    out_target.args.clear();

    if (match(TokenType::LParen)) {
        if (!match(TokenType::RParen)) {
            while (true) {
                Token arg_tok;
                if (!expect(TokenType::PercentValue, "Expected value %N in block target arguments", &arg_tok)) {
                    return false;
                }
                out_target.args.push_back(arg_tok.id_num);
                if (match(TokenType::RParen)) break;
                if (!expect(TokenType::Comma, "Expected ',' between block target arguments")) {
                    return false;
                }
            }
        }
    }
    return true;
}

bool IlParser::parse_module(BronzeModuleAST& out_ast) {
    // Optional 'module <name>'
    if (match(TokenType::Module)) {
        out_ast.name = lexer_.scan_to_eol();
    } else {
        out_ast.name = "bronze_module";
    }

    while (!lexer_.is_eof() && lexer_.peek_token().type != TokenType::Eof) {
        if (lexer_.peek_token().type == TokenType::Census ||
            (lexer_.peek_token().type == TokenType::Identifier && lexer_.peek_token().text == "census")) {
            lexer_.next_token();
            while (!lexer_.is_eof() && lexer_.peek_token().type != TokenType::LBrace && lexer_.peek_token().type != TokenType::Eof) {
                lexer_.next_token();
            }
            if (match(TokenType::LBrace)) {
                int depth = 1;
                while (!lexer_.is_eof() && depth > 0 && lexer_.peek_token().type != TokenType::Eof) {
                    if (lexer_.peek_token().type == TokenType::LBrace) depth++;
                    else if (lexer_.peek_token().type == TokenType::RBrace) depth--;
                    lexer_.next_token();
                }
            }
            continue;
        }
        BronzeFunction fn;
        if (!parse_function(fn)) {
            return false;
        }
        out_ast.functions.push_back(std::move(fn));
    }
    return !has_error_;
}

bool IlParser::parse_function(BronzeFunction& out_fn) {
    if (!expect(TokenType::Func, "Expected 'func' keyword")) {
        return false;
    }

    Token name_tok = lexer_.peek_token();
    if (name_tok.type == TokenType::LParen || name_tok.type == TokenType::Eof) {
        error("Expected function name", name_tok);
        return false;
    }
    lexer_.next_token();
    out_fn.name = std::string(name_tok.text);
    while (lexer_.peek_token().type != TokenType::LParen &&
           lexer_.peek_token().type != TokenType::Eof &&
           lexer_.peek_token().line == name_tok.line) {
        Token part = lexer_.next_token();
        out_fn.name += part.text;
    }

    if (!expect(TokenType::LParen, "Expected '(' after function name")) {
        return false;
    }

    // Parameters: (%0: f64, %1: i32)
    if (!match(TokenType::RParen)) {
        while (true) {
            Token p_tok;
            if (!expect(TokenType::PercentValue, "Expected parameter %N", &p_tok)) {
                return false;
            }
            if (!expect(TokenType::Colon, "Expected ':' after parameter name")) {
                return false;
            }
            BronzeType p_type = BronzeType::Unknown;
            if (!parse_type(p_type)) return false;
            out_fn.params.push_back({p_tok.id_num, p_type});

            if (match(TokenType::RParen)) break;
            if (!expect(TokenType::Comma, "Expected ',' between parameters")) {
                return false;
            }
        }
    }

    if (!expect(TokenType::Arrow, "Expected '->' for function return type")) {
        return false;
    }

    if (!parse_type(out_fn.return_type)) {
        return false;
    }

    if (match(TokenType::Export)) {
        out_fn.is_exported = true;
    }

    if (!expect(TokenType::LBrace, "Expected '{' to start function body")) {
        return false;
    }

    while (!match(TokenType::RBrace)) {
        if (lexer_.peek_token().type == TokenType::Eof) {
            error("Unexpected EOF while parsing function body", lexer_.peek_token());
            return false;
        }
        BronzeBlock block;
        if (!parse_block(block)) {
            return false;
        }
        out_fn.blocks.push_back(std::move(block));
    }
    return true;
}

bool IlParser::parse_block(BronzeBlock& out_block) {
    Token label_tok;
    if (!expect(TokenType::BlockLabel, "Expected block label (e.g. b0, b1)", &label_tok)) {
        return false;
    }
    out_block.id = label_tok.id_num;

    // Block parameters: b1(%0: f64, %1: i32):
    if (match(TokenType::LParen)) {
        if (!match(TokenType::RParen)) {
            while (true) {
                Token p_tok;
                if (!expect(TokenType::PercentValue, "Expected block parameter %N", &p_tok)) {
                    return false;
                }
                if (!expect(TokenType::Colon, "Expected ':' after block parameter")) {
                    return false;
                }
                BronzeType p_type = BronzeType::Unknown;
                if (!parse_type(p_type)) return false;
                out_block.params.push_back({p_tok.id_num, p_type});

                if (match(TokenType::RParen)) break;
                if (!expect(TokenType::Comma, "Expected ',' between block parameters")) {
                    return false;
                }
            }
        }
    }

    // Optional handler: handler b<id>
    if (match(TokenType::Handler)) {
        Token handler_tok;
        if (!expect(TokenType::BlockLabel, "Expected block label after 'handler'", &handler_tok)) {
            return false;
        }
        out_block.handler_id = handler_tok.id_num;
    }

    // Optional block annotations: [fast 0], [slow 0], etc.
    while (match(TokenType::LBracket)) {
        while (!match(TokenType::RBracket) && !match(TokenType::Eof)) {
            lexer_.next_token();
        }
    }

    if (!expect(TokenType::Colon, "Expected ':' after block header")) {
        return false;
    }

    // Parse instructions until next block label or closing brace
    while (true) {
        const Token& peek = lexer_.peek_token();
        if (peek.type == TokenType::BlockLabel || peek.type == TokenType::RBrace || peek.type == TokenType::Eof) {
            break;
        }
        BronzeInstruction inst;
        if (!parse_instruction(inst)) {
            return false;
        }
        out_block.instructions.push_back(std::move(inst));
    }
    return true;
}

} // namespace brass::il

