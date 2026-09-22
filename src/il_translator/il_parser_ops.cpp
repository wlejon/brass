#include "il_parser.hpp"
#include <brass/il_translator/il_translator.hpp>
#include <cstdlib>
#include <limits>

namespace brass::il {

static bool parse_prop_mod(IlLexer& lexer, BronzeInstruction& out_inst, uint32_t op_line, const Token& peek) {
    if (peek.type == TokenType::Identifier && peek.text == "slot") {
        lexer.next_token();
        Token slot_tok = lexer.next_token();
        if (slot_tok.type == TokenType::NumberInt) {
            out_inst.static_slot = static_cast<uint32_t>(slot_tok.num_i64);
        }
        if (lexer.peek_token().line == op_line && lexer.peek_token().type == TokenType::AtFunction) {
            lexer.next_token();
        } else if (lexer.peek_token().line == op_line && lexer.peek_token().type == TokenType::Identifier && lexer.peek_token().text == "family") {
            lexer.next_token();
            while (lexer.peek_token().line == op_line && lexer.peek_token().type != TokenType::Comma &&
                   lexer.peek_token().type != TokenType::Eof && lexer.peek_token().type != TokenType::BlockLabel) {
                lexer.next_token();
            }
        }
        return true;
    }
    if (peek.type == TokenType::Identifier && peek.text == "mono") {
        lexer.next_token(); out_inst.is_mono = true; return true;
    }
    if (peek.type == TokenType::Identifier && peek.text == "fn-recv") {
        lexer.next_token(); out_inst.is_fn_recv = true; return true;
    }
    return false;
}

bool IlParser::parse_instruction_ops(BronzeInstruction& out_inst, const Token& op_tok) {
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

            while (match(TokenType::Comma)) {
                const Token& peek = lexer_.peek_token();
                if (peek.line != op_tok.line) break;
                if (peek.type == TokenType::NumberInt) {
                    out_inst.ic_index = static_cast<uint32_t>(lexer_.next_token().num_i64);
                    out_inst.depth = out_inst.ic_index;
                } else if (!parse_prop_mod(lexer_, out_inst, op_tok.line, peek)) {
                    break;
                }
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

            bool has_ic_index = false;
            while (match(TokenType::Comma)) {
                const Token& peek = lexer_.peek_token();
                if (peek.line != op_tok.line) break;
                if (peek.type == TokenType::NumberInt) {
                    Token num_tok = lexer_.next_token();
                    if (!has_ic_index) {
                        out_inst.ic_index = static_cast<uint32_t>(num_tok.num_i64);
                        out_inst.depth = out_inst.ic_index;
                        has_ic_index = true;
                    } else {
                        out_inst.imm_i64 = num_tok.num_i64;
                    }
                } else if (!parse_prop_mod(lexer_, out_inst, op_tok.line, peek)) {
                    break;
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

        case BronzeOp::PrivateNew:
            break;

        case BronzeOp::PrivateAdd:
        case BronzeOp::PrivateGet:
        case BronzeOp::PrivateHas:
        case BronzeOp::PrivateSet: {
            while (lexer_.peek_token().type == TokenType::PercentValue && lexer_.peek_token().line == op_tok.line) {
                Token opd_tok = lexer_.next_token();
                out_inst.operands.push_back(opd_tok.id_num);
                if (!match(TokenType::Comma)) break;
            }
            if (lexer_.peek_token().type == TokenType::NumberInt && lexer_.peek_token().line == op_tok.line) {
                out_inst.imm_i64 = lexer_.next_token().num_i64;
                out_inst.index = static_cast<uint32_t>(out_inst.imm_i64);
            }
            break;
        }

        case BronzeOp::PrivateMisuse: {
            if (lexer_.peek_token().type == TokenType::NumberInt && lexer_.peek_token().line == op_tok.line) {
                out_inst.imm_i64 = lexer_.next_token().num_i64;
                out_inst.index = static_cast<uint32_t>(out_inst.imm_i64);
            }
            if (match(TokenType::Comma) && lexer_.peek_token().type == TokenType::NumberInt && lexer_.peek_token().line == op_tok.line) {
                out_inst.param_count = static_cast<uint32_t>(lexer_.next_token().num_i64);
            }
            break;
        }

        case BronzeOp::PinGuard: {
            if (lexer_.peek_token().type == TokenType::PercentValue) {
                out_inst.operands.push_back(lexer_.next_token().id_num);
            }
            if (match(TokenType::Comma)) {
                Token shape_tok = lexer_.next_token();
                if (shape_tok.type == TokenType::NumberInt) {
                    out_inst.imm_i64 = shape_tok.num_i64;
                }
            }
            if (match(TokenType::Comma)) {
                Token name_tok = lexer_.next_token();
                if (name_tok.type == TokenType::StringLiteral) {
                    out_inst.string_literal = std::string(name_tok.text);
                }
            }
            break;
        }

        case BronzeOp::CensusRecord: {
            if (lexer_.peek_token().type == TokenType::NumberInt) {
                out_inst.imm_i64 = lexer_.next_token().num_i64;
                out_inst.index = static_cast<uint32_t>(out_inst.imm_i64);
            }
            if (match(TokenType::Comma) && lexer_.peek_token().type == TokenType::NumberInt) {
                out_inst.param_count = static_cast<uint32_t>(lexer_.next_token().num_i64);
            }
            if (match(TokenType::Comma) && lexer_.peek_token().type == TokenType::PercentValue) {
                out_inst.operands.push_back(lexer_.next_token().id_num);
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
    return true;
}

} // namespace brass::il
