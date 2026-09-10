#include "parser_coro.hpp"
#include "parser_decode.hpp"
#include <brass/mir/opcodes.hpp>
#include <vector>

namespace brass {

bool decode_coro_opcode(
    std::string_view str,
    Opcode& op,
    Type& type_suffix
) {
    if (str == "coro_create") {
        op = Opcode::coro_create;
        type_suffix = Type::gcref();
        return true;
    }
    if (str == "coro_destroy") {
        op = Opcode::coro_destroy;
        type_suffix = Type::void_type();
        return true;
    }
    if (str == "coro_suspend") {
        op = Opcode::coro_suspend;
        type_suffix = Type::i64();
        return true;
    }
    if (str.starts_with("coro_suspend.")) {
        op = Opcode::coro_suspend;
        type_suffix = parse_type_from_string(str.substr(13));
        return true;
    }
    if (str == "coro_resume") {
        op = Opcode::coro_resume;
        type_suffix = Type::i64();
        return true;
    }
    if (str.starts_with("coro_resume.")) {
        op = Opcode::coro_resume;
        type_suffix = parse_type_from_string(str.substr(12));
        return true;
    }
    return false;
}

bool parse_coro_instruction(
    ParserCoroContext& ctx,
    Opcode op,
    Type type_annotation,
    Type type_suffix,
    Builder& b,
    Value*& res_val,
    Instruction*& res_inst
) {
    switch (op) {
        case Opcode::coro_create: {
            std::string_view callee = ctx.parse_symbol_name();
            if (callee.empty()) return false;

            std::vector<Value*> args;
            if (ctx.match(TokenKind::LParen)) {
                while (!ctx.peek().is(TokenKind::RParen) && !ctx.peek().is(TokenKind::Eof)) {
                    Value* arg = ctx.parse_val();
                    if (!arg) return false;
                    args.push_back(arg);

                    if (!ctx.peek().is(TokenKind::RParen)) {
                        if (!ctx.expect(TokenKind::Comma, "',' between args")) return false;
                    }
                }
                if (!ctx.expect(TokenKind::RParen, "')'")) return false;
            }

            res_val = b.build_coro_create(callee, args);
            res_inst = res_val ? res_val->defining_instruction() : nullptr;
            return true;
        }

        case Opcode::coro_suspend: {
            Value* yield_val = ctx.parse_val();
            if (!yield_val) return false;

            uint32_t state_id = 0;
            if (ctx.match(TokenKind::Comma)) {
                Token num_tok = ctx.peek();
                if (num_tok.is(TokenKind::IntLiteral)) {
                    ctx.advance();
                    state_id = static_cast<uint32_t>(num_tok.int_val);
                } else {
                    ctx.error(num_tok.location, "Expected state ID integer after comma in coro_suspend");
                    return false;
                }
            }

            Type ret_type = !type_annotation.is_void() ? type_annotation :
                            (!type_suffix.is_void() ? type_suffix : Type::i64());
            res_val = b.build_coro_suspend(yield_val, state_id, ret_type);
            res_inst = res_val ? res_val->defining_instruction() : nullptr;
            return true;
        }

        case Opcode::coro_resume: {
            Value* coro_val = ctx.parse_val();
            if (!coro_val) return false;

            Value* input_val = nullptr;
            if (ctx.match(TokenKind::Comma)) {
                input_val = ctx.parse_val();
                if (!input_val) return false;
            }

            Type ret_type = !type_annotation.is_void() ? type_annotation :
                            (!type_suffix.is_void() ? type_suffix : Type::i64());
            res_val = b.build_coro_resume(coro_val, input_val, ret_type);
            res_inst = res_val ? res_val->defining_instruction() : nullptr;
            return true;
        }

        case Opcode::coro_destroy: {
            Value* coro_val = ctx.parse_val();
            if (!coro_val) return false;

            res_inst = b.build_coro_destroy(coro_val);
            res_val = nullptr;
            return true;
        }

        default:
            return false;
    }
}

} // namespace brass
