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

    Token name_tok;
    if (!expect(TokenType::Identifier, "Expected function name", &name_tok)) {
        return false;
    }
    out_fn.name = std::string(name_tok.text);

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

bool IlParser::parse_instruction(BronzeInstruction& out_inst) {
    Token peek = lexer_.peek_token();

    // Assignment format: %N: <type> = <op> ...
    if (peek.type == TokenType::PercentValue) {
        Token dst_tok = lexer_.next_token();
        out_inst.result_id = dst_tok.id_num;

        if (!expect(TokenType::Colon, "Expected ':' after result value")) {
            return false;
        }
        if (!parse_type(out_inst.result_type)) {
            return false;
        }
        if (!expect(TokenType::Equal, "Expected '=' in instruction assignment")) {
            return false;
        }
    }

    Token op_tok = lexer_.next_token();
    if (op_tok.type != TokenType::KeywordOp) {
        error("Expected instruction opcode (e.g. add, const.f64, call, jump, br, ret)", op_tok);
        return false;
    }
    out_inst.op = op_tok.op;
    out_inst.box_type = op_tok.box_type;

    switch (out_inst.op) {
        case BronzeOp::ConstF64: {
            Token num_tok = lexer_.next_token();
            out_inst.imm_f64 = num_tok.num_f64;
            break;
        }
        case BronzeOp::ConstI32: {
            Token num_tok = lexer_.next_token();
            out_inst.imm_i64 = num_tok.num_i64;
            break;
        }
        case BronzeOp::ConstBool: {
            Token b_tok = lexer_.next_token();
            out_inst.imm_bool = (b_tok.text == "true" || b_tok.num_i64 != 0);
            break;
        }
        case BronzeOp::ConstUndefined:
        case BronzeOp::ConstNull:
            break;

        case BronzeOp::Call: {
            Token callee_tok;
            if (!expect(TokenType::AtFunction, "Expected @callee after 'call'", &callee_tok)) {
                return false;
            }
            out_inst.callee_name = std::string(callee_tok.text);

            if (!expect(TokenType::LParen, "Expected '(' for call argument list")) {
                return false;
            }
            if (!match(TokenType::RParen)) {
                while (true) {
                    Token arg_tok;
                    if (!expect(TokenType::PercentValue, "Expected %N argument", &arg_tok)) {
                        return false;
                    }
                    out_inst.operands.push_back(arg_tok.id_num);
                    if (match(TokenType::RParen)) break;
                    if (!expect(TokenType::Comma, "Expected ',' between call arguments")) {
                        return false;
                    }
                }
            }
            break;
        }

        case BronzeOp::Ret: {
            if (lexer_.peek_token().type == TokenType::PercentValue) {
                Token val_tok = lexer_.next_token();
                out_inst.operands.push_back(val_tok.id_num);
            }
            break;
        }

        case BronzeOp::Jump: {
            if (!parse_block_target(out_inst.target)) {
                return false;
            }
            break;
        }

        case BronzeOp::Branch: {
            Token cond_tok;
            if (!expect(TokenType::PercentValue, "Expected %cond value for branch", &cond_tok)) {
                return false;
            }
            out_inst.operands.push_back(cond_tok.id_num);
            if (!expect(TokenType::Comma, "Expected ',' after branch condition")) {
                return false;
            }
            if (!parse_block_target(out_inst.target)) {
                return false;
            }
            if (!expect(TokenType::Comma, "Expected ',' between branch targets")) {
                return false;
            }
            if (!parse_block_target(out_inst.else_target)) {
                return false;
            }
            break;
        }

        default: {
            // General opcode with comma-separated %N operands
            // e.g. add %0, %1 or unbox.f64 %0, raw
            while (lexer_.peek_token().type == TokenType::PercentValue) {
                Token opd_tok = lexer_.next_token();
                out_inst.operands.push_back(opd_tok.id_num);
                if (match(TokenType::Comma)) {
                    // Check if next token is trailing attribute like 'raw'
                    if (lexer_.peek_token().type == TokenType::Identifier) {
                        Token attr_tok = lexer_.next_token();
                        if (attr_tok.text == "raw") out_inst.raw_unbox = true;
                        break;
                    }
                } else {
                    break;
                }
            }
            break;
        }
    }
    return true;
}

} // namespace brass::il
