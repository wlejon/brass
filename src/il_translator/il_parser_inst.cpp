#include "il_parser.hpp"
#include <brass/il_translator/il_translator.hpp>
#include <charconv>

namespace brass::il {

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
                out_inst.callee_name += part.text;
            }

            if (!expect(TokenType::LParen, "Expected '(' for call argument list")) {
                return false;
            }
            if (!match(TokenType::RParen)) {
                while (true) {
                    if (lexer_.peek_token().text.rfind("env+", 0) == 0) {
                        Token env_tok = lexer_.next_token();
                        std::string_view num_str = env_tok.text.substr(4);
                        uint32_t hops = 0;
                        std::from_chars(num_str.data(), num_str.data() + num_str.size(), hops);
                        out_inst.env_hops = hops;
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
            if (lexer_.peek_token().line == str_tok.line && match(TokenType::Comma)) {
                Token mode_tok = lexer_.next_token();
                if (mode_tok.text == "soft") {
                    out_inst.index = 1;
                }
            }
            break;
        }

        case BronzeOp::ImmutableAssign: {
            Token str_tok;
            if (!expect(TokenType::StringLiteral, "Expected string literal after immutable.assign", &str_tok)) {
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
            if (lexer_.peek_token().type == TokenType::PercentValue) {
                Token tok = lexer_.next_token();
                out_inst.operands.push_back(tok.id_num);
            } else {
                Token fn_tok;
                if (!expect(TokenType::AtFunction, "Expected @func after create.async_machine", &fn_tok)) return false;
                out_inst.callee_name = std::string(fn_tok.text);
                while (lexer_.peek_token().type != TokenType::Comma &&
                       lexer_.peek_token().type != TokenType::Eof &&
                       lexer_.peek_token().line == fn_tok.line) {
                    Token part = lexer_.next_token();
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
                // The site number, as `prop.get`/`prop.set` carry it: both
                // fields, so the textual and the in-memory AST paths agree.
                out_inst.ic_index = static_cast<uint32_t>(lexer_.next_token().num_i64);
                out_inst.depth = out_inst.ic_index;
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
            break;
        }

        case BronzeOp::MethodCallSpread: {
            if (lexer_.peek_token().type == TokenType::PercentValue) {
                out_inst.operands.push_back(lexer_.next_token().id_num);
            }
            if (match(TokenType::Comma) && (lexer_.peek_token().type == TokenType::NumberInt || lexer_.peek_token().type == TokenType::NumberFloat)) {
                out_inst.index = static_cast<uint32_t>(lexer_.next_token().num_i64);
            }
            if (match(TokenType::Comma) && (lexer_.peek_token().type == TokenType::NumberInt || lexer_.peek_token().type == TokenType::NumberFloat)) {
                // The site number, as `prop.get`/`prop.set` carry it: both
                // fields, so the textual and the in-memory AST paths agree.
                out_inst.ic_index = static_cast<uint32_t>(lexer_.next_token().num_i64);
                out_inst.depth = out_inst.ic_index;
            }
            while (match(TokenType::Comma)) {
                Token peek = lexer_.peek_token();
                if (peek.type == TokenType::Identifier && (peek.text == "mono" || peek.text == "direct" || peek.text == "fn-recv" || peek.text == "family")) {
                    lexer_.next_token();
                } else if (peek.type == TokenType::PercentValue) {
                    out_inst.operands.push_back(lexer_.next_token().id_num);
                    break;
                } else {
                    break;
                }
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
            break;
        }

        case BronzeOp::SuperCallSpread: {
            while (true) {
                if (lexer_.peek_token().type == TokenType::PercentValue) {
                    out_inst.operands.push_back(lexer_.next_token().id_num);
                } else {
                    break;
                }
                if (!match(TokenType::Comma)) break;
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
            break;
        }

        case BronzeOp::SuperSet: {
            if (lexer_.peek_token().type == TokenType::PercentValue) {
                out_inst.operands.push_back(lexer_.next_token().id_num);
            }
            if (match(TokenType::Comma) && (lexer_.peek_token().type == TokenType::NumberInt || lexer_.peek_token().type == TokenType::NumberFloat)) {
                out_inst.index = static_cast<uint32_t>(lexer_.next_token().num_i64);
            }
            if (match(TokenType::Comma) && lexer_.peek_token().type == TokenType::PercentValue) {
                out_inst.operands.push_back(lexer_.next_token().id_num);
            }
            if (match(TokenType::Comma) && lexer_.peek_token().type == TokenType::PercentValue) {
                out_inst.operands.push_back(lexer_.next_token().id_num);
            }
            if (match(TokenType::Comma) && (lexer_.peek_token().type == TokenType::NumberInt || lexer_.peek_token().type == TokenType::NumberFloat)) {
                out_inst.imm_i64 = lexer_.next_token().num_i64;
            }
            break;
        }

        default: {
            if (!parse_instruction_ops(out_inst, op_tok)) {
                return false;
            }
            break;
        }
    }
    while (lexer_.peek_token().line == op_tok.line && lexer_.peek_token().type != TokenType::Eof &&
           lexer_.peek_token().type != TokenType::BlockLabel && lexer_.peek_token().type != TokenType::RBrace) {
        lexer_.next_token();
    }
    return true;
}

} // namespace brass::il
