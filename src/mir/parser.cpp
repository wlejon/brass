// Textual MIR parser: one instruction (optional `%name[: type] =` result,
// opcode, operands) into the builder's current block. Module, function and
// block structure are parsed in parser_function.cpp.

#include "parser_impl.hpp"
#include "parser_vec.hpp"
#include "parser_coro.hpp"
#include "parser_decode.hpp"

#include <vector>

namespace brass::mir_parser {

    bool Parser::parse_instruction(Builder& b, Function* fn, ValueMap& value_map,
                                   const BlockLookup& get_or_create_block) {
        (void)fn;
        std::string result_name;
        Type type_annotation = Type::void_type();
        bool has_assignment = false;

        if (peek().is(TokenKind::ValueIdent)) {
            result_name = std::string(advance().text);
            if (match(TokenKind::Colon)) {
                type_annotation = parse_type();
            }
            if (!expect(TokenKind::Equal, "'='")) return false;
            has_assignment = true;
        }

        Token op_tok = advance();
        Opcode op = Opcode::unreachable;
        Type type_suffix = Type::void_type();
        Type mem_type = Type::void_type();

        if (!decode_opcode_string(op_tok.text, op, type_suffix, mem_type)) {
            error(op_tok.location, "Unrecognized opcode or instruction: '" + std::string(op_tok.text) + "'");
            return false;
        }

        auto parse_val = [&]() -> Value* {
            if (!peek().is(TokenKind::ValueIdent)) {
                error(peek().location, "Expected value identifier (e.g. '%0'), got '" + std::string(peek().text) + "'");
                return nullptr;
            }
            const Token name_tok = advance();
            std::string name = std::string(name_tok.text);
            auto it = value_map.find(name);
            if (it == value_map.end()) {
                // Possibly defined by a block later in the text: stand in a
                // placeholder and let parse_function_decl re-parse this block
                // once the rest is known (see parse_block_body).
                if (forward_refs_ok_) {
                    saw_forward_ref_ = true;
                    return &forward_placeholder_;
                }
                error(name_tok.location, "Use of undefined value: '" + name + "'");
                return nullptr;
            }
            return it->second;
        };

        auto parse_branch_target = [&](BranchTarget& target) -> bool {
            if (!is_identifier_or_keyword(peek().kind)) {
                error(peek().location, "Expected target basic block identifier, got '" + std::string(peek().text) + "'");
                return false;
            }
            std::string_view target_name = advance().text;
            target.block = get_or_create_block(target_name);

            if (match(TokenKind::LParen)) {
                while (!peek().is(TokenKind::RParen) && !peek().is(TokenKind::Eof)) {
                    Value* arg = parse_val();
                    if (!arg) return false;
                    target.args.push_back(arg);

                    if (!peek().is(TokenKind::RParen)) {
                        if (!expect(TokenKind::Comma, "','")) return false;
                    }
                }
                if (!expect(TokenKind::RParen, "')'")) return false;
            }
            return true;
        };

        Value* res_val = nullptr;
        Instruction* res_inst = nullptr;

        switch (op) {
            case Opcode::iconst_i32:
            case Opcode::iconst_i64: {
                const bool is32 = op == Opcode::iconst_i32;
                if (!peek().is(TokenKind::IntLiteral)) {
                    error(peek().location, std::string("Expected integer literal for ") + (is32 ? "iconst.i32" : "iconst.i64"));
                    return false;
                }
                const unsigned bits = is32 ? 32 : 64;
                int64_t val = 0;
                if (!take_int(signed_min_of(bits), signed_max_of(bits), bits, is32 ? "i32" : "i64", val)) return false;
                res_val = is32 ? b.build_iconst_i32(static_cast<int32_t>(val)) : b.build_iconst_i64(val);
                break;
            }
            case Opcode::fconst_f64: {
                if (!peek().is(TokenKind::FloatLiteral) && !peek().is(TokenKind::IntLiteral)) {
                    error(peek().location, "Expected float literal for fconst");
                    return false;
                }
                double val = 0.0;
                if (peek().is(TokenKind::FloatLiteral)) {
                    val = advance().float_val;
                } else {
                    Token tok = advance();
                    if (tok.int_overflow) {
                        error(tok.location, "Integer literal '" + std::string(tok.text) + "' out of range for fconst");
                        return false;
                    }
                    val = static_cast<double>(tok.int_magnitude);
                    if (tok.int_negative) val = -val;
                }
                if (type_suffix == Type::f32()) {
                    res_val = b.build_fconst_f32(static_cast<float>(val));
                } else {
                    res_val = b.build_fconst_f64(val);
                }
                break;
            }
            case Opcode::patchable_const_i32: {
                std::string_view sym = parse_symbol_name();
                if (has_error_) return false;
                if (!expect(TokenKind::Comma, "','")) return false;
                if (!peek().is(TokenKind::IntLiteral)) {
                    error(peek().location, "Expected integer literal for patchable_const.i32");
                    return false;
                }
                int64_t val = 0;
                if (!take_int(INT32_MIN, INT32_MAX, 32, "i32", val)) return false;
                res_val = b.build_patchable_const_i32(sym, static_cast<int32_t>(val));
                break;
            }
            case Opcode::patchable_const_i64: {
                std::string_view sym = parse_symbol_name();
                if (has_error_) return false;
                if (!expect(TokenKind::Comma, "','")) return false;
                if (!peek().is(TokenKind::IntLiteral)) {
                    error(peek().location, "Expected integer literal for patchable_const.i64");
                    return false;
                }
                int64_t val = 0;
                if (!take_int(INT64_MIN, INT64_MAX, 64, "i64", val)) return false;
                res_val = b.build_patchable_const_i64(sym, val);
                break;
            }

            case Opcode::sext_i64: { Value* v = parse_val(); if (!v) return false; res_val = b.build_sext_i64(v); break; }
            case Opcode::zext_i64: { Value* v = parse_val(); if (!v) return false; res_val = b.build_zext_i64(v); break; }
            case Opcode::trunc_i32: { Value* v = parse_val(); if (!v) return false; res_val = b.build_trunc_i32(v); break; }
            case Opcode::trunc_i8: { Value* v = parse_val(); if (!v) return false; res_val = b.build_trunc_i8(v); break; }
            case Opcode::fptosi_i32: { Value* v = parse_val(); if (!v) return false; res_val = b.build_fptosi_i32(v); break; }
            case Opcode::fptosi_i64: { Value* v = parse_val(); if (!v) return false; res_val = b.build_fptosi_i64(v); break; }
            case Opcode::fptosi_i32_f32: { Value* v = parse_val(); if (!v) return false; res_val = b.build_fptosi_i32_f32(v); break; }
            case Opcode::fptosi_i64_f32: { Value* v = parse_val(); if (!v) return false; res_val = b.build_fptosi_i64_f32(v); break; }
            // The bare `sitofp.f64` / `sitofp.f32` (docs/mir_reference.md)
            // take their source width from the operand: an i64 picks the
            // .i64 form. An explicit source suffix is kept as written.
            case Opcode::sitofp_f64_i32: {
                Value* v = parse_val(); if (!v) return false;
                res_val = (op_tok.text == "sitofp.f64" && v->type() == Type::i64()) ? b.build_sitofp_f64_i64(v)
                                                                                     : b.build_sitofp_f64_i32(v);
                break;
            }
            case Opcode::sitofp_f64_i64: { Value* v = parse_val(); if (!v) return false; res_val = b.build_sitofp_f64_i64(v); break; }
            case Opcode::sitofp_f32_i32: {
                Value* v = parse_val(); if (!v) return false;
                res_val = (op_tok.text == "sitofp.f32" && v->type() == Type::i64()) ? b.build_sitofp_f32_i64(v)
                                                                                     : b.build_sitofp_f32_i32(v);
                break;
            }
            case Opcode::sitofp_f32_i64: { Value* v = parse_val(); if (!v) return false; res_val = b.build_sitofp_f32_i64(v); break; }
            case Opcode::fptrunc_f32_f64: { Value* v = parse_val(); if (!v) return false; res_val = b.build_fptrunc_f32_f64(v); break; }
            case Opcode::fpext_f64_f32: { Value* v = parse_val(); if (!v) return false; res_val = b.build_fpext_f64_f32(v); break; }
            case Opcode::bitcast_i64_f64: { Value* v = parse_val(); if (!v) return false; res_val = b.build_bitcast_i64_f64(v); break; }
            case Opcode::bitcast_f64_i64: { Value* v = parse_val(); if (!v) return false; res_val = b.build_bitcast_f64_i64(v); break; }

            case Opcode::add:
            case Opcode::sub:
            case Opcode::mul:
            case Opcode::sdiv:
            case Opcode::udiv:
            case Opcode::smod:
            case Opcode::umod:
            case Opcode::and_:
            case Opcode::or_:
            case Opcode::xor_:
            case Opcode::shl:
            case Opcode::lshr:
            case Opcode::ashr:
            case Opcode::fmin_f32:
            case Opcode::fmin_f64:
            case Opcode::fmax_f32:
            case Opcode::fmax_f64: {
                Value* lhs = parse_val(); if (!lhs) return false;
                if (!expect(TokenKind::Comma, "','")) return false;
                Value* rhs = parse_val(); if (!rhs) return false;

                switch (op) {
                    case Opcode::add: res_val = b.build_add(lhs, rhs); break;
                    case Opcode::sub: res_val = b.build_sub(lhs, rhs); break;
                    case Opcode::mul: res_val = b.build_mul(lhs, rhs); break;
                    case Opcode::sdiv: res_val = b.build_sdiv(lhs, rhs); break;
                    case Opcode::udiv: res_val = b.build_udiv(lhs, rhs); break;
                    case Opcode::smod: res_val = b.build_smod(lhs, rhs); break;
                    case Opcode::umod: res_val = b.build_umod(lhs, rhs); break;
                    case Opcode::and_: res_val = b.build_and(lhs, rhs); break;
                    case Opcode::or_: res_val = b.build_or(lhs, rhs); break;
                    case Opcode::xor_: res_val = b.build_xor(lhs, rhs); break;
                    case Opcode::shl: res_val = b.build_shl(lhs, rhs); break;
                    case Opcode::lshr: res_val = b.build_lshr(lhs, rhs); break;
                    case Opcode::ashr: res_val = b.build_ashr(lhs, rhs); break;
                    case Opcode::fmin_f32: res_val = b.build_fmin_f32(lhs, rhs); break;
                    case Opcode::fmin_f64: res_val = b.build_fmin_f64(lhs, rhs); break;
                    case Opcode::fmax_f32: res_val = b.build_fmax_f32(lhs, rhs); break;
                    case Opcode::fmax_f64: res_val = b.build_fmax_f64(lhs, rhs); break;
                    default: break;
                }
                break;
            }

            case Opcode::fma_f32:
            case Opcode::fma_f64: {
                Value* a = parse_val(); if (!a) return false;
                if (!expect(TokenKind::Comma, "','")) return false;
                Value* b_val = parse_val(); if (!b_val) return false;
                if (!expect(TokenKind::Comma, "','")) return false;
                Value* c = parse_val(); if (!c) return false;
                if (op == Opcode::fma_f32) res_val = b.build_fma_f32(a, b_val, c);
                else res_val = b.build_fma_f64(a, b_val, c);
                break;
            }

            case Opcode::neg: { Value* v = parse_val(); if (!v) return false; res_val = b.build_neg(v); break; }
            case Opcode::sqrt_f32: { Value* v = parse_val(); if (!v) return false; res_val = b.build_sqrt_f32(v); break; }
            case Opcode::sqrt_f64: { Value* v = parse_val(); if (!v) return false; res_val = b.build_sqrt_f64(v); break; }
            case Opcode::floor_f32: { Value* v = parse_val(); if (!v) return false; res_val = b.build_floor_f32(v); break; }
            case Opcode::floor_f64: { Value* v = parse_val(); if (!v) return false; res_val = b.build_floor_f64(v); break; }
            case Opcode::ceil_f32: { Value* v = parse_val(); if (!v) return false; res_val = b.build_ceil_f32(v); break; }
            case Opcode::ceil_f64: { Value* v = parse_val(); if (!v) return false; res_val = b.build_ceil_f64(v); break; }
            case Opcode::round_f32: { Value* v = parse_val(); if (!v) return false; res_val = b.build_round_f32(v); break; }
            case Opcode::round_f64: { Value* v = parse_val(); if (!v) return false; res_val = b.build_round_f64(v); break; }
            case Opcode::fabs_f32: { Value* v = parse_val(); if (!v) return false; res_val = b.build_fabs_f32(v); break; }
            case Opcode::fabs_f64: { Value* v = parse_val(); if (!v) return false; res_val = b.build_fabs_f64(v); break; }
            case Opcode::not_: { Value* v = parse_val(); if (!v) return false; res_val = b.build_not(v); break; }
            case Opcode::clz: { Value* v = parse_val(); if (!v) return false; res_val = b.build_clz(v); break; }
            case Opcode::ctz: { Value* v = parse_val(); if (!v) return false; res_val = b.build_ctz(v); break; }
            case Opcode::popcnt: { Value* v = parse_val(); if (!v) return false; res_val = b.build_popcnt(v); break; }

            case Opcode::eq:
            case Opcode::ne:
            case Opcode::slt:
            case Opcode::ult:
            case Opcode::sle:
            case Opcode::ule:
            case Opcode::sgt:
            case Opcode::ugt:
            case Opcode::sge:
            case Opcode::uge: {
                Value* lhs = parse_val(); if (!lhs) return false;
                if (!expect(TokenKind::Comma, "','")) return false;
                Value* rhs = parse_val(); if (!rhs) return false;

                switch (op) {
                    case Opcode::eq: res_val = b.build_eq(lhs, rhs); break; case Opcode::ne: res_val = b.build_ne(lhs, rhs); break;
                    case Opcode::slt: res_val = b.build_slt(lhs, rhs); break; case Opcode::ult: res_val = b.build_ult(lhs, rhs); break;
                    case Opcode::sle: res_val = b.build_sle(lhs, rhs); break; case Opcode::ule: res_val = b.build_ule(lhs, rhs); break;
                    case Opcode::sgt: res_val = b.build_sgt(lhs, rhs); break; case Opcode::ugt: res_val = b.build_ugt(lhs, rhs); break;
                    case Opcode::sge: res_val = b.build_sge(lhs, rhs); break; case Opcode::uge: res_val = b.build_uge(lhs, rhs); break;
                    default: break;
                }
                break;
            }

            case Opcode::sadd_overflow:
            case Opcode::ssub_overflow:
            case Opcode::smul_overflow:
            case Opcode::uadd_overflow:
            case Opcode::usub_overflow:
            case Opcode::umul_overflow: {
                Value* lhs = parse_val(); if (!lhs) return false;
                if (!expect(TokenKind::Comma, "','")) return false;
                Value* rhs = parse_val(); if (!rhs) return false;

                switch (op) {
                    case Opcode::sadd_overflow: res_val = b.build_sadd_overflow(lhs, rhs); break;
                    case Opcode::ssub_overflow: res_val = b.build_ssub_overflow(lhs, rhs); break;
                    case Opcode::smul_overflow: res_val = b.build_smul_overflow(lhs, rhs); break;
                    case Opcode::uadd_overflow: res_val = b.build_uadd_overflow(lhs, rhs); break;
                    case Opcode::usub_overflow: res_val = b.build_usub_overflow(lhs, rhs); break;
                    case Opcode::umul_overflow: res_val = b.build_umul_overflow(lhs, rhs); break;
                    default: break;
                }
                break;
            }

            case Opcode::select: {
                Value* cond = parse_val(); if (!cond) return false;
                if (!expect(TokenKind::Comma, "','")) return false;
                Value* true_v = parse_val(); if (!true_v) return false;
                if (!expect(TokenKind::Comma, "','")) return false;
                Value* false_v = parse_val(); if (!false_v) return false;
                res_val = b.build_select(cond, true_v, false_v);
                break;
            }

            case Opcode::load: {
                Value* base = parse_val(); if (!base) return false;
                int32_t offset = 0;
                if (match(TokenKind::Comma)) {
                    if (!peek().is(TokenKind::IntLiteral)) {
                        error(peek().location, "Expected integer offset for load");
                        return false;
                    }
                    if (!take_offset(offset)) return false;
                }
                if (mem_type.is_void()) {
                    mem_type = Type::i32();
                }
                res_val = b.build_load(mem_type, base, offset);
                break;
            }

            case Opcode::store: {
                Value* base = parse_val(); if (!base) return false;
                if (!expect(TokenKind::Comma, "','")) return false;

                int32_t offset = 0;
                Value* val = nullptr;

                if (peek().is(TokenKind::IntLiteral)) {
                    if (!take_offset(offset)) return false;
                    if (!expect(TokenKind::Comma, "','")) return false;
                    val = parse_val();
                } else {
                    val = parse_val();
                }
                if (!val) return false;

                if (mem_type.is_void()) {
                    mem_type = val->type();
                }
                b.build_store(mem_type, base, offset, val);
                break;
            }

            case Opcode::load_indexed: {
                Value* base = parse_val(); if (!base) return false;
                if (!expect(TokenKind::Comma, "','")) return false;
                Value* idx = parse_val(); if (!idx) return false;
                if (!expect(TokenKind::Comma, "','")) return false;
                if (!peek().is(TokenKind::IntLiteral)) {
                    error(peek().location, "Expected integer scale (1, 2, 4, 8) for load_indexed");
                    return false;
                }
                uint8_t scale = 0;
                if (!take_unsigned(UINT8_MAX, "an index scale", scale)) return false;
                int32_t offset = 0;
                if (match(TokenKind::Comma)) {
                    if (!peek().is(TokenKind::IntLiteral)) {
                        error(peek().location, "Expected integer offset for load_indexed");
                        return false;
                    }
                    if (!take_offset(offset)) return false;
                }
                if (mem_type.is_void()) {
                    mem_type = Type::i64();
                }
                res_val = b.build_load_indexed(mem_type, base, idx, scale, offset);
                break;
            }

            case Opcode::store_indexed: {
                Value* base = parse_val(); if (!base) return false;
                if (!expect(TokenKind::Comma, "','")) return false;
                Value* idx = parse_val(); if (!idx) return false;
                if (!expect(TokenKind::Comma, "','")) return false;
                if (!peek().is(TokenKind::IntLiteral)) {
                    error(peek().location, "Expected integer scale (1, 2, 4, 8) for store_indexed");
                    return false;
                }
                uint8_t scale = 0;
                if (!take_unsigned(UINT8_MAX, "an index scale", scale)) return false;
                if (!expect(TokenKind::Comma, "','")) return false;

                int32_t offset = 0;
                Value* val = nullptr;
                if (peek().is(TokenKind::IntLiteral)) {
                    if (!take_offset(offset)) return false;
                    if (!expect(TokenKind::Comma, "','")) return false;
                    val = parse_val();
                } else {
                    val = parse_val();
                }
                if (!val) return false;

                if (mem_type.is_void()) {
                    mem_type = val->type();
                }
                b.build_store_indexed(mem_type, base, idx, scale, offset, val);
                break;
            }

            case Opcode::write_barrier: {
                Value* obj = parse_val(); if (!obj) return false;
                if (!expect(TokenKind::Comma, "','")) return false;
                Value* val = parse_val(); if (!val) return false;
                res_inst = b.build_write_barrier(obj, val);
                break;
            }

            case Opcode::call: {
                std::string_view callee = parse_symbol_name();
                if (has_error_) return false;
                if (!expect(TokenKind::LParen, "'('")) return false;

                std::vector<Value*> args;
                while (!peek().is(TokenKind::RParen) && !peek().is(TokenKind::Eof)) {
                    Value* arg = parse_val();
                    if (!arg) return false;
                    args.push_back(arg);

                    if (!peek().is(TokenKind::RParen)) {
                        if (!expect(TokenKind::Comma, "','")) return false;
                    }
                }
                if (!expect(TokenKind::RParen, "')'")) return false;

                Type call_ret_type = has_assignment ? (type_suffix.is_void() ? Type::i32() : type_suffix) : Type::void_type();
                res_val = b.build_call(callee, call_ret_type, args);
                break;
            }

            case Opcode::func_addr: {
                std::string_view sym = parse_symbol_name();
                if (has_error_) return false;
                res_val = b.build_func_addr(sym);
                break;
            }

            case Opcode::call_indirect: {
                Value* callee_ptr = parse_val(); if (!callee_ptr) return false;
                if (!expect(TokenKind::LParen, "'('")) return false;

                std::vector<Value*> args;
                while (!peek().is(TokenKind::RParen) && !peek().is(TokenKind::Eof)) {
                    Value* arg = parse_val();
                    if (!arg) return false;
                    args.push_back(arg);

                    if (!peek().is(TokenKind::RParen)) {
                        if (!expect(TokenKind::Comma, "','")) return false;
                    }
                }
                if (!expect(TokenKind::RParen, "')'")) return false;

                Type call_ret_type = has_assignment ? (type_suffix.is_void() ? Type::i32() : type_suffix) : Type::void_type();
                res_val = b.build_call_indirect(callee_ptr, call_ret_type, args);
                break;
            }

            case Opcode::patchable_call: {
                std::string_view site = parse_symbol_name();
                if (has_error_) return false;
                if (!expect(TokenKind::Comma, "','")) return false;
                std::string_view callee = parse_symbol_name();
                if (has_error_) return false;
                if (!expect(TokenKind::LParen, "'('")) return false;

                std::vector<Value*> args;
                while (!peek().is(TokenKind::RParen) && !peek().is(TokenKind::Eof)) {
                    Value* arg = parse_val();
                    if (!arg) return false;
                    args.push_back(arg);

                    if (!peek().is(TokenKind::RParen)) {
                        if (!expect(TokenKind::Comma, "','")) return false;
                    }
                }
                if (!expect(TokenKind::RParen, "')'")) return false;

                Type call_ret_type = has_assignment ? (type_suffix.is_void() ? Type::i32() : type_suffix) : Type::void_type();
                res_val = b.build_patchable_call(site, callee, call_ret_type, args);
                break;
            }

            case Opcode::safepoint: {
                b.build_safepoint();
                break;
            }

            case Opcode::pinned_tls_read: {
                res_val = b.build_pinned_tls_read();
                break;
            }

            case Opcode::read_sp: {
                res_val = b.build_read_sp();
                break;
            }

            case Opcode::pinned_tls_write: {
                Value* addr = parse_val();
                if (!addr) return false;
                res_inst = b.build_pinned_tls_write(addr);
                break;
            }

            case Opcode::alloca_: {  // `alloca SIZE, ALIGN`, as the printer writes it
                // Both are held as int32 (imm_i32 / offset) and printed signed.
                if (!peek().is(TokenKind::IntLiteral)) {
                    error(peek().location, "Expected 'alloca SIZE, ALIGN'");
                    return false;
                }
                uint32_t size = 0;
                if (!take_unsigned(INT32_MAX, "an alloca size", size)) return false;
                if (!match(TokenKind::Comma) || !peek().is(TokenKind::IntLiteral)) {
                    error(peek().location, "Expected 'alloca SIZE, ALIGN'");
                    return false;
                }
                uint32_t align = 0;
                if (!take_unsigned(INT32_MAX, "an alloca alignment", align)) return false;
                res_val = b.build_alloca(size, align);
                break;
            }

            case Opcode::guard: {
                Value* cond = parse_val(); if (!cond) return false;
                if (!expect(TokenKind::Comma, "','")) return false;
                std::string_view exit_stub = parse_symbol_name();
                if (has_error_) return false;

                std::vector<Value*> state_map;
                if (match(TokenKind::Comma)) {
                    if (!expect(TokenKind::LBracket, "'['")) return false;
                    while (!peek().is(TokenKind::RBracket) && !peek().is(TokenKind::Eof)) {
                        Value* sv = parse_val();
                        if (!sv) return false;
                        state_map.push_back(sv);

                        if (!peek().is(TokenKind::RBracket)) {
                            if (!expect(TokenKind::Comma, "','")) return false;
                        }
                    }
                    if (!expect(TokenKind::RBracket, "']'")) return false;
                }
                b.build_guard(cond, exit_stub, state_map);
                break;
            }

            case Opcode::resume_point: {
                if (!peek().is(TokenKind::IntLiteral)) {
                    error(peek().location, "Expected integer literal for resume_point ID");
                    return false;
                }
                uint32_t resume_id = 0;
                if (!take_unsigned(UINT32_MAX, "a resume point id", resume_id)) return false;
                b.build_resume_point(resume_id);
                break;
            }

            case Opcode::osr_entry: {  // `osr_entry HEADER_ID [ %live, ... ]`
                if (!peek().is(TokenKind::IntLiteral)) {
                    error(peek().location, "Expected integer loop header id for osr_entry");
                    return false;
                }
                uint32_t header_id = 0;
                if (!take_unsigned(UINT32_MAX, "a loop header id", header_id)) return false;
                std::vector<Value*> live_ins;
                if (match(TokenKind::LBracket)) {
                    while (!peek().is(TokenKind::RBracket) && !peek().is(TokenKind::Eof)) {
                        Value* v = parse_val();
                        if (!v) return false;
                        live_ins.push_back(v);
                        if (!peek().is(TokenKind::RBracket) && !expect(TokenKind::Comma, "','")) return false;
                    }
                    if (!expect(TokenKind::RBracket, "']'")) return false;
                }
                res_inst = b.build_osr_entry(header_id, Span<Value* const>(live_ins.data(), live_ins.size()));
                break;
            }

            case Opcode::br: {
                BranchTarget target;
                if (!parse_branch_target(target)) return false;
                b.build_br(target.block, target.args);
                break;
            }

            case Opcode::br_if: {
                Value* cond = parse_val(); if (!cond) return false;
                if (!expect(TokenKind::Comma, "','")) return false;
                BranchTarget true_t;
                if (!parse_branch_target(true_t)) return false;
                if (!expect(TokenKind::Comma, "','")) return false;
                BranchTarget false_t;
                if (!parse_branch_target(false_t)) return false;

                b.build_br_if(cond, true_t.block, true_t.args, false_t.block, false_t.args);
                break;
            }

            case Opcode::switch_: {
                Value* cond = parse_val(); if (!cond) return false;
                if (!expect(TokenKind::Comma, "','")) return false;

                if (peek().text != "default") {
                    error(peek().location, "Expected 'default:' for switch instruction");
                    return false;
                }
                advance();
                if (!expect(TokenKind::Colon, "':'")) return false;
                BranchTarget def_target;
                if (!parse_branch_target(def_target)) return false;

                if (!expect(TokenKind::Comma, "','")) return false;
                if (!expect(TokenKind::LBracket, "'['")) return false;

                std::vector<SwitchCase> cases;
                while (!peek().is(TokenKind::RBracket) && !peek().is(TokenKind::Eof)) {
                    if (!peek().is(TokenKind::IntLiteral)) {
                        error(peek().location, "Expected integer case value");
                        return false;
                    }
                    // A case value must fit the condition's type, as the
                    // verifier requires; a hex literal may spell its bits.
                    const Type cond_t = cond->type();
                    const unsigned bits = cond_t == Type::i8() ? 8 : cond_t == Type::i16() ? 16
                                        : cond_t == Type::i32() ? 32 : 64;
                    const std::string what = "switch." + std::string(cond_t.name()) + " case";
                    int64_t case_val = 0;
                    if (!take_int(signed_min_of(bits), signed_max_of(bits), bits, what, case_val)) return false;
                    if (!expect(TokenKind::Colon, "':'")) return false;
                    BranchTarget case_target;
                    if (!parse_branch_target(case_target)) return false;
                    cases.push_back(SwitchCase(case_val, std::move(case_target)));

                    if (!peek().is(TokenKind::RBracket)) {
                        if (!expect(TokenKind::Comma, "','")) return false;
                    }
                }
                if (!expect(TokenKind::RBracket, "']'")) return false;

                b.build_switch(cond, def_target.block, def_target.args, cases);
                break;
            }

            case Opcode::ret: {
                if (peek().is(TokenKind::ValueIdent)) {
                    Value* v = parse_val();
                    if (!v) return false;
                    b.build_ret(v);
                } else {
                    b.build_ret_void();
                }
                break;
            }

            case Opcode::unreachable: {
                b.build_unreachable();
                break;
            }

            case Opcode::throw_: {
                Value* v = parse_val();
                if (!v) return false;
                res_inst = b.build_throw(v);
                break;
            }

            case Opcode::resume: {
                if (peek().is(TokenKind::ValueIdent)) {
                    Value* v = parse_val();
                    if (!v) return false;
                    res_inst = b.build_resume(v);
                } else {
                    res_inst = b.build_resume();
                }
                break;
            }

            case Opcode::landing_pad: {
                Type lp_type = type_annotation.is_void() ? (type_suffix.is_void() ? Type::i64() : type_suffix) : type_annotation;
                res_val = b.build_landing_pad(lp_type);
                break;
            }

            case Opcode::invoke: {
                std::string_view callee = parse_symbol_name();
                if (has_error_) return false;
                if (!expect(TokenKind::LParen, "'('")) return false;

                std::vector<Value*> args;
                while (!peek().is(TokenKind::RParen) && !peek().is(TokenKind::Eof)) {
                    Value* arg = parse_val();
                    if (!arg) return false;
                    args.push_back(arg);

                    if (!peek().is(TokenKind::RParen)) {
                        if (!expect(TokenKind::Comma, "','")) return false;
                    }
                }
                if (!expect(TokenKind::RParen, "')'")) return false;
                if (!expect(TokenKind::Comma, "','")) return false;

                BranchTarget normal_t;
                if (!parse_branch_target(normal_t)) return false;
                if (!expect(TokenKind::Comma, "','")) return false;

                BranchTarget unwind_t;
                if (!parse_branch_target(unwind_t)) return false;

                Type invoke_ret_type = has_assignment ? (type_suffix.is_void() ? Type::i32() : type_suffix) : type_suffix;
                res_inst = b.build_invoke(callee, invoke_ret_type, args,
                                          normal_t.block, normal_t.args,
                                          unwind_t.block, unwind_t.args);
                if (res_inst && res_inst->result()) {
                    res_val = res_inst->result();
                }
                break;
            }

            default: {
                if (is_vector_op(op)) {
                    ParserVecContext ctx{
                        [this]() { return peek(); },
                        [this]() { return advance(); },
                        [this](TokenKind k, const std::string& desc) { return expect(k, desc); },
                        [this](TokenKind k) { return match(k); },
                        [&]() { return parse_val(); },
                        [this](SourceLocation loc, const std::string& msg) { error(loc, msg); }
                    };
                    if (!parse_vector_instruction(ctx, op, type_annotation, type_suffix, mem_type, b, res_val, res_inst)) {
                        return false;
                    }
                    break;
                }
                if (is_coro_op(op)) {
                    ParserCoroContext ctx{
                        [this]() { return peek(); },
                        [this]() { return advance(); },
                        [this](TokenKind k, const std::string& desc) { return expect(k, desc); },
                        [this](TokenKind k) { return match(k); },
                        [&]() { return parse_val(); },
                        [this]() { return parse_symbol_name(); },
                        [this](SourceLocation loc, const std::string& msg) { error(loc, msg); }
                    };
                    // The printer writes a void suspend / resume as the bare
                    // name with no result; with a result the bare name is i64.
                    if (!has_assignment && op_tok.text.find('.') == std::string_view::npos &&
                        (op == Opcode::coro_suspend || op == Opcode::coro_resume)) {
                        type_suffix = Type::void_type();
                    }
                    if (!parse_coro_instruction(ctx, op, type_annotation, type_suffix, b, res_val, res_inst)) {
                        return false;
                    }
                    break;
                }
                error(op_tok.location, "Unhandled opcode: '" + std::string(opcode_name(op)) + "'");
                return false;
            }
        }

        if (has_assignment && res_val) {
            auto prev = value_map.find(result_name);
            block_bindings_.emplace_back(result_name, prev == value_map.end() ? nullptr : prev->second);
            value_map[result_name] = res_val;
        }

        return true;
    }

} // namespace brass::mir_parser
