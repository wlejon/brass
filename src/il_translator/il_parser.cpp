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
        if (lexer_.peek_token().type == TokenType::Identifier && lexer_.peek_token().text == "census") {
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

    Token name_tok;
    if (!expect(TokenType::Identifier, "Expected function name", &name_tok)) {
        return false;
    }
    out_fn.name = std::string(name_tok.text);
    while (lexer_.peek_token().type != TokenType::LParen &&
           lexer_.peek_token().type != TokenType::Eof &&
           lexer_.peek_token().line == name_tok.line) {
        Token part = lexer_.next_token();
        out_fn.name += " ";
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
    out_inst.line = op_tok.line;
    out_inst.column = op_tok.col;

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
        case BronzeOp::ExcTake:
            break;

        case BronzeOp::Call: {
            Token callee_tok;
            if (!expect(TokenType::AtFunction, "Expected @callee after 'call'", &callee_tok)) {
                return false;
            }
            out_inst.callee_name = std::string(callee_tok.text);
            while (lexer_.peek_token().type != TokenType::LParen &&
                   lexer_.peek_token().type != TokenType::Eof &&
                   lexer_.peek_token().line == callee_tok.line) {
                Token part = lexer_.next_token();
                out_inst.callee_name += " ";
                out_inst.callee_name += part.text;
            }

            if (!expect(TokenType::LParen, "Expected '(' for call argument list")) {
                return false;
            }
            if (!match(TokenType::RParen)) {
                while (true) {
                    if (lexer_.peek_token().text.rfind("env+", 0) == 0) {
                        lexer_.next_token();
                    }
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

        case BronzeOp::NameResolve: {
            Token str_tok;
            if (!expect(TokenType::StringLiteral, "Expected string literal after name.resolve", &str_tok)) {
                return false;
            }
            out_inst.string_literal = std::string(str_tok.text);
            break;
        }

        case BronzeOp::CallDynamic: {
            Token callee_tok;
            if (!expect(TokenType::PercentValue, "Expected %callee in call.dynamic", &callee_tok)) {
                return false;
            }
            out_inst.operands.push_back(callee_tok.id_num);
            if (!expect(TokenType::Comma, "Expected ',' after %callee")) return false;

            Token this_tok;
            if (!expect(TokenType::PercentValue, "Expected %this in call.dynamic", &this_tok)) {
                return false;
            }
            out_inst.operands.push_back(this_tok.id_num);
            if (!expect(TokenType::Comma, "Expected ',' after %this")) return false;

            Token argc_tok = lexer_.next_token();
            out_inst.param_count = static_cast<uint32_t>(argc_tok.num_i64);

            for (uint32_t i = 0; i < out_inst.param_count; ++i) {
                if (!expect(TokenType::Comma, "Expected ',' before call.dynamic argument")) return false;
                Token arg_tok;
                if (!expect(TokenType::PercentValue, "Expected %arg in call.dynamic", &arg_tok)) return false;
                out_inst.operands.push_back(arg_tok.id_num);
            }
            break;
        }

        case BronzeOp::EnvCreate: {
            Token parent_tok;
            if (!expect(TokenType::PercentValue, "Expected %parent in env.create", &parent_tok)) return false;
            out_inst.operands.push_back(parent_tok.id_num);
            if (!expect(TokenType::Comma, "Expected ',' after %parent")) return false;
            Token size_tok = lexer_.next_token();
            out_inst.param_count = static_cast<uint32_t>(size_tok.num_i64);
            break;
        }

        case BronzeOp::EnvSet: {
            Token env_tok;
            if (!expect(TokenType::PercentValue, "Expected %env in env.set", &env_tok)) return false;
            out_inst.operands.push_back(env_tok.id_num);
            if (!expect(TokenType::Comma, "Expected ',' after %env")) return false;
            Token depth_tok = lexer_.next_token();
            out_inst.depth = static_cast<uint32_t>(depth_tok.num_i64);
            if (!expect(TokenType::Comma, "Expected ',' after depth")) return false;
            Token idx_tok = lexer_.next_token();
            out_inst.index = static_cast<uint32_t>(idx_tok.num_i64);
            if (!expect(TokenType::Comma, "Expected ',' after index")) return false;
            Token val_tok;
            if (!expect(TokenType::PercentValue, "Expected %val in env.set", &val_tok)) return false;
            out_inst.operands.push_back(val_tok.id_num);
            break;
        }

        case BronzeOp::EnvGet:
        case BronzeOp::EnvInitTdz: {
            Token env_tok;
            if (!expect(TokenType::PercentValue, "Expected %env in env op", &env_tok)) return false;
            out_inst.operands.push_back(env_tok.id_num);
            if (!expect(TokenType::Comma, "Expected ',' after %env")) return false;
            Token depth_tok = lexer_.next_token();
            out_inst.depth = static_cast<uint32_t>(depth_tok.num_i64);
            if (!expect(TokenType::Comma, "Expected ',' after depth")) return false;
            Token idx_tok = lexer_.next_token();
            out_inst.index = static_cast<uint32_t>(idx_tok.num_i64);
            break;
        }

        case BronzeOp::EnvGetTdz: {
            Token env_tok;
            if (!expect(TokenType::PercentValue, "Expected %env in env.get.tdz", &env_tok)) return false;
            out_inst.operands.push_back(env_tok.id_num);
            if (!expect(TokenType::Comma, "Expected ',' after %env")) return false;
            Token depth_tok = lexer_.next_token();
            out_inst.depth = static_cast<uint32_t>(depth_tok.num_i64);
            if (!expect(TokenType::Comma, "Expected ',' after depth")) return false;
            Token idx_tok = lexer_.next_token();
            out_inst.index = static_cast<uint32_t>(idx_tok.num_i64);
            if (!expect(TokenType::Comma, "Expected ',' after index")) return false;
            Token str_tok;
            if (!expect(TokenType::StringLiteral, "Expected identifier name in env.get.tdz", &str_tok)) return false;
            out_inst.string_literal = std::string(str_tok.text);
            break;
        }

        case BronzeOp::ModuleEnvSet: {
            Token env_tok;
            if (!expect(TokenType::PercentValue, "Expected %env in module.env.set", &env_tok)) return false;
            out_inst.operands.push_back(env_tok.id_num);
            break;
        }

        case BronzeOp::ModuleEnvGet: {
            break;
        }

        case BronzeOp::CreateFunc: {
            Token fn_tok;
            if (!expect(TokenType::AtFunction, "Expected @func after create.func", &fn_tok)) return false;
            out_inst.callee_name = std::string(fn_tok.text);
            while (lexer_.peek_token().type != TokenType::Comma &&
                   lexer_.peek_token().type != TokenType::Eof &&
                   lexer_.peek_token().line == fn_tok.line) {
                Token part = lexer_.next_token();
                out_inst.callee_name += " ";
                out_inst.callee_name += part.text;
            }
            if (!expect(TokenType::Comma, "Expected ',' after @func")) return false;
            Token argc_tok = lexer_.next_token();
            out_inst.param_count = static_cast<uint32_t>(argc_tok.num_i64);
            if (!expect(TokenType::Comma, "Expected ',' after param count")) return false;
            Token env_tok;
            if (!expect(TokenType::PercentValue, "Expected %env in create.func", &env_tok)) return false;
            out_inst.operands.push_back(env_tok.id_num);
            break;
        }

        case BronzeOp::CreateAsyncMachine: {
            Token fn_tok;
            if (!expect(TokenType::AtFunction, "Expected @func after create.async_machine", &fn_tok)) return false;
            out_inst.callee_name = std::string(fn_tok.text);
            while (lexer_.peek_token().type != TokenType::Comma &&
                   lexer_.peek_token().type != TokenType::Eof &&
                   lexer_.peek_token().line == fn_tok.line) {
                Token part = lexer_.next_token();
                out_inst.callee_name += " ";
                out_inst.callee_name += part.text;
            }
            if (match(TokenType::Comma)) {
                Token next_t = lexer_.next_token();
                if (next_t.type == TokenType::NumberInt) {
                    out_inst.param_count = static_cast<uint32_t>(next_t.num_i64);
                    if (match(TokenType::Comma)) {
                        Token env_tok;
                        if (!expect(TokenType::PercentValue, "Expected %env in create.async_machine", &env_tok)) return false;
                        out_inst.operands.push_back(env_tok.id_num);
                    }
                } else if (next_t.type == TokenType::PercentValue) {
                    out_inst.operands.push_back(next_t.id_num);
                }
            }
            break;
        }

        case BronzeOp::CreateArray: {
            Token size_tok = lexer_.next_token();
            out_inst.param_count = static_cast<uint32_t>(size_tok.num_i64);
            break;
        }

        case BronzeOp::CreateObject: {
            // no operands
            break;
        }

        case BronzeOp::PropGet: {
            Token obj_tok;
            if (!expect(TokenType::PercentValue, "Expected %obj in prop.get", &obj_tok)) return false;
            out_inst.operands.push_back(obj_tok.id_num);
            if (!expect(TokenType::Comma, "Expected ',' after %obj in prop.get")) return false;

            Token key_tok = lexer_.next_token();
            if (key_tok.type == TokenType::StringLiteral) {
                out_inst.string_literal = std::string(key_tok.text);
            } else {
                out_inst.index = static_cast<uint32_t>(key_tok.num_i64);
            }

            if (match(TokenType::Comma)) {
                Token slot_tok = lexer_.next_token();
                out_inst.depth = static_cast<uint32_t>(slot_tok.num_i64);
            }
            while (lexer_.peek_token().line == op_tok.line && lexer_.peek_token().type != TokenType::Eof &&
                   lexer_.peek_token().type != TokenType::BlockLabel && lexer_.peek_token().type != TokenType::RBrace) {
                lexer_.next_token();
            }
            break;
        }

        case BronzeOp::PropSet: {
            Token obj_tok;
            if (!expect(TokenType::PercentValue, "Expected %obj in prop.set", &obj_tok)) return false;
            out_inst.operands.push_back(obj_tok.id_num);
            if (!expect(TokenType::Comma, "Expected ',' after %obj in prop.set")) return false;

            Token key_tok = lexer_.next_token();
            if (key_tok.type == TokenType::StringLiteral) {
                out_inst.string_literal = std::string(key_tok.text);
            } else {
                out_inst.index = static_cast<uint32_t>(key_tok.num_i64);
            }
            if (!expect(TokenType::Comma, "Expected ',' after key in prop.set")) return false;

            Token val_tok;
            if (!expect(TokenType::PercentValue, "Expected %val in prop.set", &val_tok)) return false;
            out_inst.operands.push_back(val_tok.id_num);

            if (match(TokenType::Comma)) {
                Token slot_tok = lexer_.next_token();
                out_inst.depth = static_cast<uint32_t>(slot_tok.num_i64);
                if (match(TokenType::Comma)) {
                    Token imm_tok = lexer_.next_token();
                    out_inst.imm_i64 = imm_tok.num_i64;
                }
            }
            while (lexer_.peek_token().line == op_tok.line && lexer_.peek_token().type != TokenType::Eof &&
                   lexer_.peek_token().type != TokenType::BlockLabel && lexer_.peek_token().type != TokenType::RBrace) {
                lexer_.next_token();
            }
            break;
        }

        case BronzeOp::MethodDef: {
            Token obj_tok;
            if (!expect(TokenType::PercentValue, "Expected %obj in method.def", &obj_tok)) return false;
            out_inst.operands.push_back(obj_tok.id_num);
            if (!expect(TokenType::Comma, "Expected ',' after %obj in method.def")) return false;

            Token key_tok = lexer_.next_token();
            if (key_tok.type == TokenType::StringLiteral) {
                out_inst.string_literal = std::string(key_tok.text);
            } else {
                out_inst.index = static_cast<uint32_t>(key_tok.num_i64);
            }
            if (!expect(TokenType::Comma, "Expected ',' after key in method.def")) return false;

            Token closure_tok;
            if (!expect(TokenType::PercentValue, "Expected %closure in method.def", &closure_tok)) return false;
            out_inst.operands.push_back(closure_tok.id_num);
            while (lexer_.peek_token().line == op_tok.line && lexer_.peek_token().type != TokenType::Eof &&
                   lexer_.peek_token().type != TokenType::BlockLabel && lexer_.peek_token().type != TokenType::RBrace) {
                lexer_.next_token();
            }
            break;
        }

        case BronzeOp::ElemSet: {
            Token obj_tok;
            if (!expect(TokenType::PercentValue, "Expected %obj in elem.set", &obj_tok)) return false;
            out_inst.operands.push_back(obj_tok.id_num);
            if (!expect(TokenType::Comma, "Expected ',' after %obj in elem.set")) return false;

            Token idx_tok;
            if (!expect(TokenType::PercentValue, "Expected %idx in elem.set", &idx_tok)) return false;
            out_inst.operands.push_back(idx_tok.id_num);
            if (!expect(TokenType::Comma, "Expected ',' after %idx in elem.set")) return false;

            Token val_tok;
            if (!expect(TokenType::PercentValue, "Expected %val in elem.set", &val_tok)) return false;
            out_inst.operands.push_back(val_tok.id_num);

            if (match(TokenType::Comma)) {
                Token ic_tok = lexer_.next_token();
                out_inst.index = static_cast<uint32_t>(ic_tok.num_i64);
            }
            while (lexer_.peek_token().line == op_tok.line && lexer_.peek_token().type != TokenType::Eof &&
                   lexer_.peek_token().type != TokenType::BlockLabel && lexer_.peek_token().type != TokenType::RBrace) {
                lexer_.next_token();
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

        case BronzeOp::Box: {
            const Token& peek = lexer_.peek_token();
            if (peek.type == TokenType::PercentValue) {
                Token val_tok = lexer_.next_token();
                out_inst.operands.push_back(val_tok.id_num);
            } else if (peek.type == TokenType::Identifier || peek.type == TokenType::StringLiteral) {
                Token str_tok = lexer_.next_token();
                out_inst.string_literal = std::string(str_tok.text);
            }
            while (lexer_.peek_token().line == op_tok.line && lexer_.peek_token().type != TokenType::Eof &&
                   lexer_.peek_token().type != TokenType::BlockLabel && lexer_.peek_token().type != TokenType::RBrace) {
                lexer_.next_token();
            }
            break;
        }

        case BronzeOp::GlobalGet: {
            Token peek = lexer_.peek_token();
            if (peek.type == TokenType::StringLiteral) {
                Token str_tok = lexer_.next_token();
                out_inst.string_literal = std::string(str_tok.text);
            } else if (peek.type == TokenType::Identifier) {
                Token id_tok = lexer_.next_token();
                out_inst.string_literal = std::string(id_tok.text);
            } else if (peek.type == TokenType::NumberInt) {
                Token num_tok = lexer_.next_token();
                out_inst.index = static_cast<uint32_t>(num_tok.num_i64);
            }
            while (lexer_.peek_token().line == op_tok.line && lexer_.peek_token().type != TokenType::Eof &&
                   lexer_.peek_token().type != TokenType::BlockLabel && lexer_.peek_token().type != TokenType::RBrace) {
                lexer_.next_token();
            }
            break;
        }

        case BronzeOp::ConcatBegin: {
            if (lexer_.peek_token().type == TokenType::PercentValue) {
                out_inst.operands.push_back(lexer_.next_token().id_num);
            }
            if (match(TokenType::Comma) && lexer_.peek_token().type == TokenType::PercentValue) {
                out_inst.operands.push_back(lexer_.next_token().id_num);
            }
            if (match(TokenType::Comma) && (lexer_.peek_token().type == TokenType::NumberInt || lexer_.peek_token().type == TokenType::NumberFloat)) {
                out_inst.imm_i64 = lexer_.next_token().num_i64;
            }
            while (lexer_.peek_token().line == op_tok.line && lexer_.peek_token().type != TokenType::Eof &&
                   lexer_.peek_token().type != TokenType::BlockLabel && lexer_.peek_token().type != TokenType::RBrace) {
                lexer_.next_token();
            }
            break;
        }

        case BronzeOp::FuncRef: {
            if (lexer_.peek_token().type == TokenType::AtFunction) {
                out_inst.callee_name = std::string(lexer_.next_token().text);
                while (lexer_.peek_token().line == op_tok.line && lexer_.peek_token().type != TokenType::Eof &&
                       lexer_.peek_token().type != TokenType::BlockLabel && lexer_.peek_token().type != TokenType::RBrace &&
                       lexer_.peek_token().type != TokenType::Comma) {
                    Token part = lexer_.next_token();
                    out_inst.callee_name += " ";
                    out_inst.callee_name += part.text;
                }
            }
            while (lexer_.peek_token().line == op_tok.line && lexer_.peek_token().type != TokenType::Eof &&
                   lexer_.peek_token().type != TokenType::BlockLabel && lexer_.peek_token().type != TokenType::RBrace) {
                lexer_.next_token();
            }
            break;
        }

        case BronzeOp::Construct: {
            if (lexer_.peek_token().type == TokenType::PercentValue) {
                out_inst.operands.push_back(lexer_.next_token().id_num);
            }
            if (match(TokenType::Comma) && (lexer_.peek_token().type == TokenType::NumberInt || lexer_.peek_token().type == TokenType::NumberFloat)) {
                out_inst.param_count = static_cast<uint32_t>(lexer_.next_token().num_i64);
            }
            while (match(TokenType::Comma)) {
                if (lexer_.peek_token().type == TokenType::PercentValue) {
                    out_inst.operands.push_back(lexer_.next_token().id_num);
                } else {
                    break;
                }
            }
            while (lexer_.peek_token().line == op_tok.line && lexer_.peek_token().type != TokenType::Eof &&
                   lexer_.peek_token().type != TokenType::BlockLabel && lexer_.peek_token().type != TokenType::RBrace) {
                lexer_.next_token();
            }
            break;
        }

        case BronzeOp::MethodCall: {
            if (lexer_.peek_token().type == TokenType::PercentValue) {
                out_inst.operands.push_back(lexer_.next_token().id_num);
            }
            if (match(TokenType::Comma) && (lexer_.peek_token().type == TokenType::NumberInt || lexer_.peek_token().type == TokenType::NumberFloat)) {
                out_inst.index = static_cast<uint32_t>(lexer_.next_token().num_i64);
            }
            if (match(TokenType::Comma) && (lexer_.peek_token().type == TokenType::NumberInt || lexer_.peek_token().type == TokenType::NumberFloat)) {
                out_inst.depth = static_cast<uint32_t>(lexer_.next_token().num_i64);
            }
            while (match(TokenType::Comma)) {
                Token peek = lexer_.peek_token();
                if (peek.type == TokenType::Identifier && peek.text == "direct") {
                    lexer_.next_token();
                    if (lexer_.peek_token().type == TokenType::AtFunction) {
                        out_inst.callee_name = std::string(lexer_.next_token().text);
                    }
                } else if (peek.type == TokenType::Identifier && (peek.text == "mono" || peek.text == "fn-recv" || peek.text == "family")) {
                    lexer_.next_token();
                } else if (peek.type == TokenType::NumberInt || peek.type == TokenType::NumberFloat) {
                    out_inst.param_count = static_cast<uint32_t>(lexer_.next_token().num_i64);
                    break;
                } else {
                    break;
                }
            }
            while (match(TokenType::Comma)) {
                if (lexer_.peek_token().type == TokenType::PercentValue) {
                    out_inst.operands.push_back(lexer_.next_token().id_num);
                } else {
                    break;
                }
            }
            while (lexer_.peek_token().line == op_tok.line && lexer_.peek_token().type != TokenType::Eof &&
                   lexer_.peek_token().type != TokenType::BlockLabel && lexer_.peek_token().type != TokenType::RBrace) {
                lexer_.next_token();
            }
            break;
        }

        case BronzeOp::SuperCall: {
            if (lexer_.peek_token().type == TokenType::PercentValue) {
                out_inst.operands.push_back(lexer_.next_token().id_num);
            }
            if (match(TokenType::Comma) && lexer_.peek_token().type == TokenType::PercentValue) {
                out_inst.operands.push_back(lexer_.next_token().id_num);
            }
            if (match(TokenType::Comma) && (lexer_.peek_token().type == TokenType::NumberInt || lexer_.peek_token().type == TokenType::NumberFloat)) {
                out_inst.param_count = static_cast<uint32_t>(lexer_.next_token().num_i64);
            }
            while (match(TokenType::Comma)) {
                if (lexer_.peek_token().type == TokenType::PercentValue) {
                    out_inst.operands.push_back(lexer_.next_token().id_num);
                } else {
                    break;
                }
            }
            while (lexer_.peek_token().line == op_tok.line && lexer_.peek_token().type != TokenType::Eof &&
                   lexer_.peek_token().type != TokenType::BlockLabel && lexer_.peek_token().type != TokenType::RBrace) {
                lexer_.next_token();
            }
            break;
        }

        case BronzeOp::SuperGet: {
            if (lexer_.peek_token().type == TokenType::PercentValue) {
                out_inst.operands.push_back(lexer_.next_token().id_num);
            }
            if (match(TokenType::Comma) && (lexer_.peek_token().type == TokenType::NumberInt || lexer_.peek_token().type == TokenType::NumberFloat)) {
                out_inst.index = static_cast<uint32_t>(lexer_.next_token().num_i64);
            }
            if (match(TokenType::Comma) && lexer_.peek_token().type == TokenType::PercentValue) {
                out_inst.operands.push_back(lexer_.next_token().id_num);
            }
            while (lexer_.peek_token().line == op_tok.line && lexer_.peek_token().type != TokenType::Eof &&
                   lexer_.peek_token().type != TokenType::BlockLabel && lexer_.peek_token().type != TokenType::RBrace) {
                lexer_.next_token();
            }
            break;
        }

        case BronzeOp::PinGuard:
        case BronzeOp::CensusRecord: {
            while (lexer_.peek_token().line == op_tok.line && lexer_.peek_token().type != TokenType::Eof &&
                   lexer_.peek_token().type != TokenType::BlockLabel && lexer_.peek_token().type != TokenType::RBrace) {
                if (lexer_.peek_token().type == TokenType::PercentValue) {
                    out_inst.operands.push_back(lexer_.peek_token().id_num);
                }
                lexer_.next_token();
            }
            break;
        }

        case BronzeOp::ElemSetTyped: {
            while (lexer_.peek_token().type == TokenType::PercentValue) {
                out_inst.operands.push_back(lexer_.next_token().id_num);
                if (!match(TokenType::Comma)) break;
            }
            if (lexer_.peek_token().type == TokenType::NumberInt || lexer_.peek_token().type == TokenType::NumberFloat) {
                out_inst.imm_i64 = lexer_.next_token().num_i64;
            }
            while (lexer_.peek_token().line == op_tok.line && lexer_.peek_token().type != TokenType::Eof &&
                   lexer_.peek_token().type != TokenType::BlockLabel && lexer_.peek_token().type != TokenType::RBrace) {
                lexer_.next_token();
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
            while (lexer_.peek_token().line == op_tok.line && lexer_.peek_token().type != TokenType::Eof &&
                   lexer_.peek_token().type != TokenType::BlockLabel && lexer_.peek_token().type != TokenType::RBrace) {
                lexer_.next_token();
            }
            break;
        }
    }
    return true;
}

} // namespace brass::il
