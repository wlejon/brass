#include "il_parser.hpp"
#include <charconv>
#include <cstdlib>
#include <limits>

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
        case BronzeOp::ConstF64: {
            Token num_tok = lexer_.next_token();
            if (num_tok.text == "nan" || num_tok.text == "-nan" || num_tok.text == "+nan") {
                out_inst.imm_f64 = std::numeric_limits<double>::quiet_NaN();
            } else if (num_tok.text == "inf" || num_tok.text == "+inf") {
                out_inst.imm_f64 = std::numeric_limits<double>::infinity();
            } else if (num_tok.text == "-inf") {
                out_inst.imm_f64 = -std::numeric_limits<double>::infinity();
            } else {
                out_inst.imm_f64 = num_tok.num_f64;
            }
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
        case BronzeOp::ConstBigInt: {
            Token num_tok = lexer_.next_token();
            std::string lit = std::string(num_tok.text);
            if (lexer_.peek_token().type == TokenType::Identifier && lexer_.peek_token().line == op_tok.line) {
                Token id_tok = lexer_.next_token();
                lit += id_tok.text;
            }
            if (!lit.empty() && lit.back() == 'n') {
                lit.pop_back();
            }
            out_inst.string_literal = lit;
            break;
        }
        case BronzeOp::ConstUndefined:
        case BronzeOp::ConstNull:
        case BronzeOp::ExcTake:
        case BronzeOp::GetNewTarget:
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
            break;
        }

        case BronzeOp::PropDelete: {
            Token obj_tok;
            if (!expect(TokenType::PercentValue, "Expected %obj in prop.delete", &obj_tok)) return false;
            out_inst.operands.push_back(obj_tok.id_num);
            if (!expect(TokenType::Comma, "Expected ',' after %obj in prop.delete")) return false;
            Token key_tok = lexer_.next_token();
            if (key_tok.type == TokenType::StringLiteral) {
                out_inst.string_literal = std::string(key_tok.text);
            } else {
                out_inst.index = static_cast<uint32_t>(key_tok.num_i64);
            }
            if (match(TokenType::Comma)) {
                Token imm_tok = lexer_.next_token();
                out_inst.imm_i64 = imm_tok.num_i64;
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
            break;
        }

        case BronzeOp::DefineOwnAttr: {
            Token obj_tok;
            if (!expect(TokenType::PercentValue, "Expected %obj in define.own.attr", &obj_tok)) return false;
            out_inst.operands.push_back(obj_tok.id_num);
            if (!expect(TokenType::Comma, "Expected ',' after %obj in define.own.attr")) return false;

            Token key_tok = lexer_.next_token();
            if (key_tok.type == TokenType::StringLiteral) {
                out_inst.string_literal = std::string(key_tok.text);
            } else {
                out_inst.index = static_cast<uint32_t>(key_tok.num_i64);
            }
            if (!expect(TokenType::Comma, "Expected ',' after key in define.own.attr")) return false;

            Token val_tok;
            if (!expect(TokenType::PercentValue, "Expected %val in define.own.attr", &val_tok)) return false;
            out_inst.operands.push_back(val_tok.id_num);
            if (!expect(TokenType::Comma, "Expected ',' after %val in define.own.attr")) return false;

            if (!expect(TokenType::LBrace, "Expected '{' in define.own.attr")) return false;
            uint32_t mask = 0;
            while (!match(TokenType::RBrace) && !lexer_.is_eof()) {
                if (lexer_.peek_token().type == TokenType::Comma) {
                    lexer_.next_token();
                    continue;
                }
                Token field_tok = lexer_.next_token();
                bool bool_val = false;
                if (match(TokenType::Equal)) {
                    Token val = lexer_.next_token();
                    bool_val = (val.text == "true" || val.num_i64 != 0);
                }
                if (field_tok.text == "value") mask |= 0x01u;
                else if (field_tok.text == "writable") mask |= 0x02u | (bool_val ? 0x04u : 0u);
                else if (field_tok.text == "enumerable") mask |= 0x08u | (bool_val ? 0x10u : 0u);
                else if (field_tok.text == "configurable") mask |= 0x20u | (bool_val ? 0x40u : 0u);
            }
            out_inst.imm_i64 = static_cast<int64_t>(mask);
            break;
        }

        case BronzeOp::AccessorDef: {
            Token tgt_tok, getter_tok, setter_tok;
            if (!expect(TokenType::PercentValue, "Expected %target in accessor.def", &tgt_tok)) return false;
            out_inst.operands.push_back(tgt_tok.id_num);
            if (!expect(TokenType::Comma, "Expected ',' after %target in accessor.def")) return false;
            Token key_tok = lexer_.next_token();
            if (key_tok.type == TokenType::StringLiteral) out_inst.string_literal = std::string(key_tok.text);
            else out_inst.index = static_cast<uint32_t>(key_tok.num_i64);
            if (!expect(TokenType::Comma, "Expected ',' after key in accessor.def")) return false;
            if (!expect(TokenType::PercentValue, "Expected %getter in accessor.def", &getter_tok)) return false;
            out_inst.operands.push_back(getter_tok.id_num);
            if (!expect(TokenType::Comma, "Expected ',' after %getter in accessor.def")) return false;
            if (!expect(TokenType::PercentValue, "Expected %setter in accessor.def", &setter_tok)) return false;
            out_inst.operands.push_back(setter_tok.id_num);
            if (match(TokenType::Comma)) {
                out_inst.imm_bool = (lexer_.next_token().text == "enumerable");
            }
            break;
        }

        case BronzeOp::AccessorDefComputed: {
            Token tgt_tok, key_tok, getter_tok, setter_tok;
            if (!expect(TokenType::PercentValue, "Expected %target in accessor.def.computed", &tgt_tok)) return false;
            out_inst.operands.push_back(tgt_tok.id_num);
            if (!expect(TokenType::Comma, "Expected ',' in accessor.def.computed")) return false;
            if (!expect(TokenType::PercentValue, "Expected %key in accessor.def.computed", &key_tok)) return false;
            out_inst.operands.push_back(key_tok.id_num);
            if (!expect(TokenType::Comma, "Expected ',' in accessor.def.computed")) return false;
            if (!expect(TokenType::PercentValue, "Expected %getter in accessor.def.computed", &getter_tok)) return false;
            out_inst.operands.push_back(getter_tok.id_num);
            if (!expect(TokenType::Comma, "Expected ',' in accessor.def.computed")) return false;
            if (!expect(TokenType::PercentValue, "Expected %setter in accessor.def.computed", &setter_tok)) return false;
            out_inst.operands.push_back(setter_tok.id_num);
            if (match(TokenType::Comma)) {
                out_inst.imm_bool = (lexer_.next_token().text == "enumerable");
            }
            break;
        }

        case BronzeOp::ModuleNamespace: {
            Token src_tok;
            if (!expect(TokenType::PercentValue, "Expected %src in module.namespace", &src_tok)) return false;
            out_inst.operands.push_back(src_tok.id_num);
            break;
        }

        case BronzeOp::DynamicImport: {
            Token src_tok;
            if (!expect(TokenType::PercentValue, "Expected %src in dynamic_import", &src_tok)) return false;
            out_inst.operands.push_back(src_tok.id_num);
            if (match(TokenType::Comma)) {
                Token key_tok = lexer_.next_token();
                if (key_tok.type == TokenType::StringLiteral) out_inst.string_literal = std::string(key_tok.text);
                else out_inst.index = static_cast<uint32_t>(key_tok.num_i64);
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
            break;
        }

        case BronzeOp::ElemDelete: {
            Token obj_tok;
            if (!expect(TokenType::PercentValue, "Expected %obj in elem.delete", &obj_tok)) return false;
            out_inst.operands.push_back(obj_tok.id_num);
            if (!expect(TokenType::Comma, "Expected ',' after %obj in elem.delete")) return false;
            Token idx_tok;
            if (!expect(TokenType::PercentValue, "Expected %idx in elem.delete", &idx_tok)) return false;
            out_inst.operands.push_back(idx_tok.id_num);
            if (match(TokenType::Comma)) {
                Token imm_tok = lexer_.next_token();
                out_inst.imm_i64 = imm_tok.num_i64;
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
                if (peek.type == TokenType::Identifier && !str_tok.text.empty() && str_tok.text[0] == 'k') {
                    char* endptr = nullptr;
                    unsigned long idx = std::strtoul(str_tok.text.data() + 1, &endptr, 10);
                    if (endptr != str_tok.text.data() + 1) {
                        out_inst.index = static_cast<uint32_t>(idx);
                    }
                }
            }
            while (lexer_.peek_token().line == op_tok.line && lexer_.peek_token().type != TokenType::Eof &&
                   lexer_.peek_token().type != TokenType::BlockLabel && lexer_.peek_token().type != TokenType::RBrace) {
                lexer_.next_token();
            }
            break;
        }

        case BronzeOp::GlobalGet:
        case BronzeOp::ImportMeta: {
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

        case BronzeOp::MethodCallSpread: {
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
                if (peek.type == TokenType::Identifier && (peek.text == "mono" || peek.text == "direct" || peek.text == "fn-recv" || peek.text == "family")) {
                    lexer_.next_token();
                } else if (peek.type == TokenType::PercentValue) {
                    out_inst.operands.push_back(lexer_.next_token().id_num);
                    break;
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

        case BronzeOp::PrivateNew:
            break;

        case BronzeOp::PrivateAdd:
        case BronzeOp::PrivateGet:
        case BronzeOp::PrivateHas:
        case BronzeOp::PrivateSet: {
            while (true) {
                if (lexer_.peek_token().type == TokenType::PercentValue && lexer_.peek_token().line == op_tok.line) {
                    out_inst.operands.push_back(lexer_.next_token().id_num);
                } else {
                    break;
                }
                if (!match(TokenType::Comma)) break;
            }
            if (lexer_.peek_token().type == TokenType::StringLiteral && lexer_.peek_token().line == op_tok.line) {
                out_inst.string_literal = std::string(lexer_.next_token().text);
            }
            break;
        }

        case BronzeOp::PrivateMisuse: {
            if (lexer_.peek_token().type == TokenType::StringLiteral && lexer_.peek_token().line == op_tok.line) {
                out_inst.string_literal = std::string(lexer_.next_token().text);
            }
            if (match(TokenType::Comma) && (lexer_.peek_token().type == TokenType::NumberInt || lexer_.peek_token().type == TokenType::NumberFloat)) {
                out_inst.imm_i64 = lexer_.next_token().num_i64;
            }
            break;
        }

        case BronzeOp::PinGuard: {
            Token reg_tok;
            if (expect(TokenType::PercentValue, "Expected %reg in pin.guard", &reg_tok)) {
                out_inst.operands.push_back(reg_tok.id_num);
            }
            if (match(TokenType::Comma)) {
                Token shape_tok = lexer_.next_token();
                if (shape_tok.text == "number") {
                    out_inst.imm_i64 = 0;
                } else if (shape_tok.text == "number-or-nullish") {
                    out_inst.imm_i64 = 1;
                } else if (shape_tok.text == "dense-array") {
                    out_inst.imm_i64 = 2;
                } else if (shape_tok.type == TokenType::NumberInt) {
                    out_inst.imm_i64 = shape_tok.num_i64;
                }
            }
            if (match(TokenType::Comma)) {
                Token name_tok = lexer_.next_token();
                out_inst.string_literal = std::string(name_tok.text);
            }
            break;
        }

        case BronzeOp::CensusRecord: {
            Token reg_tok;
            if (expect(TokenType::PercentValue, "Expected %reg in census.record", &reg_tok)) {
                out_inst.operands.push_back(reg_tok.id_num);
            }
            if (match(TokenType::Comma)) {
                Token site_tok = lexer_.next_token();
                if (site_tok.text == "env-slot") {
                    out_inst.imm_i64 = 0;
                } else if (site_tok.text == "field") {
                    out_inst.imm_i64 = 1;
                } else if (site_tok.text == "param") {
                    out_inst.imm_i64 = 2;
                } else if (site_tok.text == "return") {
                    out_inst.imm_i64 = 3;
                } else if (site_tok.text == "opaque-store") {
                    out_inst.imm_i64 = 4;
                } else if (site_tok.type == TokenType::NumberInt) {
                    out_inst.imm_i64 = site_tok.num_i64;
                }
            }
            if (match(TokenType::Comma)) {
                Token name_tok = lexer_.next_token();
                if (name_tok.type == TokenType::NumberInt) {
                    out_inst.index = static_cast<uint32_t>(name_tok.num_i64);
                } else {
                    out_inst.string_literal = std::string(name_tok.text);
                }
            }
            break;
        }

        case BronzeOp::MathUnary: {
            if (lexer_.peek_token().type == TokenType::PercentValue) out_inst.operands.push_back(lexer_.next_token().id_num);
            if (match(TokenType::Comma) && (lexer_.peek_token().type == TokenType::NumberInt || lexer_.peek_token().type == TokenType::NumberFloat))
                out_inst.imm_i64 = lexer_.next_token().num_i64;
            break;
        }

        case BronzeOp::NumericStep: {
            if (lexer_.peek_token().type == TokenType::PercentValue) out_inst.operands.push_back(lexer_.next_token().id_num);
            if (match(TokenType::Comma)) {
                Token peek = lexer_.peek_token();
                if (peek.type == TokenType::Identifier || peek.type == TokenType::NumberInt) {
                    lexer_.next_token();
                    out_inst.imm_i64 = (peek.text == "+1" || peek.num_i64 > 0) ? 1 : -1;
                }
            }
            break;
        }

        case BronzeOp::ElemSetTyped: {
            while (lexer_.peek_token().type == TokenType::PercentValue) {
                out_inst.operands.push_back(lexer_.next_token().id_num);
                if (!match(TokenType::Comma)) break;
            }
            if (lexer_.peek_token().type == TokenType::NumberInt || lexer_.peek_token().type == TokenType::NumberFloat)
                out_inst.imm_i64 = lexer_.next_token().num_i64;
            break;
        }

        case BronzeOp::ToStr: {
            if (lexer_.peek_token().type == TokenType::PercentValue && lexer_.peek_token().line == op_tok.line) {
                out_inst.operands.push_back(lexer_.next_token().id_num);
            }
            break;
        }

        case BronzeOp::TemplateCached: {
            if (lexer_.peek_token().type == TokenType::NumberInt || lexer_.peek_token().type == TokenType::NumberFloat) {
                out_inst.imm_i64 = lexer_.next_token().num_i64;
            }
            break;
        }

        case BronzeOp::TemplateObject: {
            while (lexer_.peek_token().type == TokenType::PercentValue && lexer_.peek_token().line == op_tok.line) {
                out_inst.operands.push_back(lexer_.next_token().id_num);
                if (!match(TokenType::Comma)) break;
            }
            if (lexer_.peek_token().type == TokenType::NumberInt || lexer_.peek_token().type == TokenType::NumberFloat) {
                out_inst.imm_i64 = lexer_.next_token().num_i64;
            }
            break;
        }

        default: {
            // General opcode with comma-separated %N operands
            // e.g. add %0, %1 or unbox.f64 %0, raw
            while (lexer_.peek_token().type == TokenType::PercentValue && lexer_.peek_token().line == op_tok.line) {
                Token opd_tok = lexer_.next_token();
                out_inst.operands.push_back(opd_tok.id_num);
                if (match(TokenType::Comma)) {
                    // Check if next token is trailing attribute like 'raw'
                    if (lexer_.peek_token().type == TokenType::Identifier && lexer_.peek_token().line == op_tok.line) {
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
    while (lexer_.peek_token().line == op_tok.line && lexer_.peek_token().type != TokenType::Eof &&
           lexer_.peek_token().type != TokenType::BlockLabel && lexer_.peek_token().type != TokenType::RBrace) {
        lexer_.next_token();
    }
    return true;
}

} // namespace brass::il
