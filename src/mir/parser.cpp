#include <brass/mir/parser.hpp>
#include <brass/mir/lexer.hpp>
#include <brass/mir/builder.hpp>
#include <unordered_map>
#include <vector>
#include <string>
#include <sstream>
#include <functional>

namespace brass {

namespace {

Type parse_type_from_string(std::string_view s) {
    if (s == "i32") return Type::i32();
    if (s == "i64") return Type::i64();
    if (s == "f64") return Type::f64();
    if (s == "ptr") return Type::ptr();
    if (s == "gcref") return Type::gcref();
    if (s == "void") return Type::void_type();
    return Type::void_type();
}

bool is_identifier_or_keyword(TokenKind k) noexcept {
    return k == TokenKind::Ident || (k >= TokenKind::Kw_func && k <= TokenKind::Kw_void);
}

bool decode_opcode_string(std::string_view str, Opcode& op, Type& type_suffix, Type& mem_type) {
    type_suffix = Type::void_type();
    mem_type = Type::void_type();

    // Check exact matches first
    if (str == "func" || str == "extern" || str == "module" || str == "resume_table" || str == "entry") {
        return false;
    }

    if (str == "ret") { op = Opcode::ret; return true; }
    if (str == "br") { op = Opcode::br; return true; }
    if (str == "br_if") { op = Opcode::br_if; return true; }
    if (str == "guard") { op = Opcode::guard; return true; }
    if (str == "resume_point") { op = Opcode::resume_point; return true; }
    if (str == "safepoint") { op = Opcode::safepoint; return true; }
    if (str == "unreachable") { op = Opcode::unreachable; return true; }
    if (str == "call") { op = Opcode::call; return true; }

    // Special conversions
    if (str == "sitofp.f64.i32" || str == "sitofp_f64_i32") { op = Opcode::sitofp_f64_i32; type_suffix = Type::f64(); return true; }
    if (str == "sitofp.f64.i64" || str == "sitofp_f64_i64") { op = Opcode::sitofp_f64_i64; type_suffix = Type::f64(); return true; }
    if (str == "sitofp.f64") { op = Opcode::sitofp_f64_i32; type_suffix = Type::f64(); return true; }

    if (str == "bitcast.i64.f64" || str == "bitcast_i64_f64") { op = Opcode::bitcast_i64_f64; type_suffix = Type::i64(); return true; }
    if (str == "bitcast.f64.i64" || str == "bitcast_f64_i64") { op = Opcode::bitcast_f64_i64; type_suffix = Type::f64(); return true; }

    // Check dot separation: base.suffix or underscore separation
    std::string_view base = str;
    std::string_view suffix = "";
    size_t dot_pos = str.find('.');
    if (dot_pos != std::string_view::npos) {
        base = str.substr(0, dot_pos);
        suffix = str.substr(dot_pos + 1);
    } else {
        size_t underscore_pos = str.rfind('_');
        if (underscore_pos != std::string_view::npos) {
            std::string_view potential_suffix = str.substr(underscore_pos + 1);
            if (potential_suffix == "i32" || potential_suffix == "i64" || potential_suffix == "f64" ||
                potential_suffix == "ptr" || potential_suffix == "gcref" || potential_suffix == "void") {
                base = str.substr(0, underscore_pos);
                suffix = potential_suffix;
            }
        }
    }

    if (!suffix.empty()) {
        type_suffix = parse_type_from_string(suffix);
    }

    // Constants
    if (base == "iconst") {
        if (type_suffix == Type::i64()) { op = Opcode::iconst_i64; return true; }
        op = Opcode::iconst_i32;
        type_suffix = Type::i32();
        return true;
    }
    if (base == "fconst") { op = Opcode::fconst_f64; type_suffix = Type::f64(); return true; }
    if (base == "patchable_const") {
        if (type_suffix == Type::i64()) { op = Opcode::patchable_const_i64; return true; }
        op = Opcode::patchable_const_i32;
        type_suffix = Type::i32();
        return true;
    }

    // Conversions
    if (base == "sext") { op = Opcode::sext_i64; type_suffix = Type::i64(); return true; }
    if (base == "zext") { op = Opcode::zext_i64; type_suffix = Type::i64(); return true; }
    if (base == "trunc") { op = Opcode::trunc_i32; type_suffix = Type::i32(); return true; }
    if (base == "fptosi") {
        if (type_suffix == Type::i64()) { op = Opcode::fptosi_i64; return true; }
        op = Opcode::fptosi_i32;
        type_suffix = Type::i32();
        return true;
    }

    // Arithmetic / Bitwise
    if (base == "add") { op = Opcode::add; return true; }
    if (base == "sub") { op = Opcode::sub; return true; }
    if (base == "mul") { op = Opcode::mul; return true; }
    if (base == "sdiv") { op = Opcode::sdiv; return true; }
    if (base == "udiv") { op = Opcode::udiv; return true; }
    if (base == "smod") { op = Opcode::smod; return true; }
    if (base == "umod") { op = Opcode::umod; return true; }
    if (base == "neg") { op = Opcode::neg; return true; }
    if (base == "and") { op = Opcode::and_; return true; }
    if (base == "or") { op = Opcode::or_; return true; }
    if (base == "xor") { op = Opcode::xor_; return true; }
    if (base == "shl") { op = Opcode::shl; return true; }
    if (base == "lshr") { op = Opcode::lshr; return true; }
    if (base == "ashr") { op = Opcode::ashr; return true; }
    if (base == "not") { op = Opcode::not_; return true; }
    if (base == "clz") { op = Opcode::clz; return true; }
    if (base == "ctz") { op = Opcode::ctz; return true; }
    if (base == "popcnt") { op = Opcode::popcnt; return true; }

    // Comparisons
    if (base == "eq") { op = Opcode::eq; return true; }
    if (base == "ne") { op = Opcode::ne; return true; }
    if (base == "slt") { op = Opcode::slt; return true; }
    if (base == "ult") { op = Opcode::ult; return true; }
    if (base == "sle") { op = Opcode::sle; return true; }
    if (base == "ule") { op = Opcode::ule; return true; }
    if (base == "sgt") { op = Opcode::sgt; return true; }
    if (base == "ugt") { op = Opcode::ugt; return true; }
    if (base == "sge") { op = Opcode::sge; return true; }
    if (base == "uge") { op = Opcode::uge; return true; }

    // Selection
    if (base == "select") { op = Opcode::select; return true; }

    // Memory
    if (base == "load") {
        op = Opcode::load;
        mem_type = type_suffix;
        return true;
    }
    if (base == "store") {
        op = Opcode::store;
        mem_type = type_suffix;
        return true;
    }
    if (base == "load_indexed") {
        op = Opcode::load_indexed;
        mem_type = type_suffix;
        return true;
    }
    if (base == "store_indexed") {
        op = Opcode::store_indexed;
        mem_type = type_suffix;
        return true;
    }

    // Calls
    if (base == "call") { op = Opcode::call; return true; }
    if (base == "call_indirect") { op = Opcode::call_indirect; return true; }
    if (base == "patchable_call") { op = Opcode::patchable_call; return true; }

    return false;
}

class Parser {
public:
    Parser(std::string_view text, DiagnosticReporter* diag, std::string_view filename)
        : lexer_(text, filename), diag_(diag), filename_(filename) {}

    bool parse_module(Module& mod) {
        while (!peek().is(TokenKind::Eof)) {
            if (peek().is(TokenKind::Kw_module)) {
                if (!parse_module_decl(mod)) return false;
            } else if (peek().is(TokenKind::Kw_extern)) {
                if (!parse_extern_decl(mod)) return false;
            } else if (peek().is(TokenKind::Kw_func)) {
                if (!parse_function_decl(mod)) return false;
            } else {
                error(peek().location, "Unexpected top-level token: '" + std::string(peek().text) + "'");
                return false;
            }
        }
        return !has_error_;
    }

    Function* parse_single_function(Module& mod) {
        if (!peek().is(TokenKind::Kw_func)) {
            error(peek().location, "Expected 'func', got '" + std::string(peek().text) + "'");
            return nullptr;
        }
        return parse_function_decl(mod);
    }

private:
    Token peek() { return lexer_.peek_token(); }
    Token advance() { return lexer_.next_token(); }

    bool match(TokenKind k) {
        if (peek().is(k)) {
            advance();
            return true;
        }
        return false;
    }

    bool expect(TokenKind k, const std::string& desc = "") {
        if (peek().is(k)) {
            advance();
            return true;
        }
        std::string expected_str = desc.empty() ? std::string(token_kind_name(k)) : desc;
        error(peek().location, "Expected " + expected_str + ", got '" + std::string(peek().text) + "'");
        return false;
    }

    void error(SourceLocation loc, std::string msg) {
        has_error_ = true;
        if (diag_) {
            diag_->error(loc, std::move(msg));
        }
    }

    Type parse_type() {
        Token tok = peek();
        if (tok.is(TokenKind::Kw_i32)) { advance(); return Type::i32(); }
        if (tok.is(TokenKind::Kw_i64)) { advance(); return Type::i64(); }
        if (tok.is(TokenKind::Kw_f64)) { advance(); return Type::f64(); }
        if (tok.is(TokenKind::Kw_ptr)) { advance(); return Type::ptr(); }
        if (tok.is(TokenKind::Kw_gcref)) { advance(); return Type::gcref(); }
        if (tok.is(TokenKind::Kw_void)) { advance(); return Type::void_type(); }

        error(tok.location, "Expected type (i32, i64, f64, ptr, gcref, void), got '" + std::string(tok.text) + "'");
        return Type::void_type();
    }

    std::string_view parse_symbol_name() {
        if (!peek().is(TokenKind::SymbolIdent)) {
            error(peek().location, "Expected symbol identifier (starting with '@'), got '" + std::string(peek().text) + "'");
            return "";
        }
        std::string_view sym = advance().text;
        if (sym.starts_with('@')) {
            sym = sym.substr(1);
        }
        return sym;
    }

    bool parse_module_decl(Module& mod) {
        advance(); // consume 'module'
        std::string_view mod_name = parse_symbol_name();
        if (has_error_) return false;
        mod.set_name(mod_name);
        return true;
    }

    bool parse_extern_decl(Module& mod) {
        advance(); // consume 'extern'
        std::string_view sym_name = parse_symbol_name();
        if (has_error_) return false;

        // Optional signature: (types...) -> ret_type
        if (match(TokenKind::LParen)) {
            while (!peek().is(TokenKind::RParen) && !peek().is(TokenKind::Eof)) {
                parse_type();
                if (has_error_) return false;
                if (!peek().is(TokenKind::RParen)) {
                    if (!expect(TokenKind::Comma, "','")) return false;
                }
            }
            if (!expect(TokenKind::RParen, "')'")) return false;

            if (match(TokenKind::Arrow)) {
                parse_type();
                if (has_error_) return false;
            }
        }

        mod.add_external_symbol(sym_name);
        return true;
    }

    Function* parse_function_decl(Module& mod) {
        advance(); // consume 'func'
        std::string_view fn_name = parse_symbol_name();
        if (has_error_) return nullptr;

        if (!expect(TokenKind::LParen, "'('")) return nullptr;

        std::vector<Type> param_types;
        std::vector<std::string> param_names;

        while (!peek().is(TokenKind::RParen) && !peek().is(TokenKind::Eof)) {
            if (!peek().is(TokenKind::ValueIdent)) {
                error(peek().location, "Expected parameter value identifier (e.g. '%0'), got '" + std::string(peek().text) + "'");
                return nullptr;
            }
            std::string p_name = std::string(advance().text);
            if (!expect(TokenKind::Colon, "':'")) return nullptr;
            Type p_type = parse_type();
            if (has_error_) return nullptr;

            param_names.push_back(std::move(p_name));
            param_types.push_back(p_type);

            if (!peek().is(TokenKind::RParen)) {
                if (!expect(TokenKind::Comma, "','")) return nullptr;
            }
        }

        if (!expect(TokenKind::RParen, "')'")) return nullptr;
        if (!expect(TokenKind::Arrow, "'->'")) return nullptr;

        Type ret_type = parse_type();
        if (has_error_) return nullptr;

        if (!expect(TokenKind::LBrace, "'{'")) return nullptr;

        Function* fn = mod.create_function(fn_name, ret_type, param_types);
        Builder b(mod);
        b.set_function(fn);

        std::unordered_map<std::string, Value*> value_map;
        std::unordered_map<std::string, BasicBlock*> block_map;

        auto get_or_create_block = [&](std::string_view name) -> BasicBlock* {
            std::string n(name);
            auto it = block_map.find(n);
            if (it != block_map.end()) {
                return it->second;
            }
            BasicBlock* bb = b.create_block(name);
            block_map[n] = bb;
            return bb;
        };

        bool is_entry = true;

        while (!peek().is(TokenKind::RBrace) && !peek().is(TokenKind::Eof)) {
            if (peek().is(TokenKind::Kw_resume_table)) {
                advance(); // consume 'resume_table'
                if (!expect(TokenKind::LBrace, "'{'")) return nullptr;

                while (!peek().is(TokenKind::RBrace) && !peek().is(TokenKind::Eof)) {
                    if (!expect(TokenKind::Kw_entry, "'entry'")) return nullptr;
                    if (!peek().is(TokenKind::IntLiteral)) {
                        error(peek().location, "Expected resume point integer ID, got '" + std::string(peek().text) + "'");
                        return nullptr;
                    }
                    uint32_t resume_id = static_cast<uint32_t>(advance().int_val);
                    if (!expect(TokenKind::Arrow, "'->'")) return nullptr;

                    if (!is_identifier_or_keyword(peek().kind)) {
                        error(peek().location, "Expected target block identifier, got '" + std::string(peek().text) + "'");
                        return nullptr;
                    }
                    std::string_view target_name = advance().text;
                    BasicBlock* target_bb = get_or_create_block(target_name);
                    fn->add_resume_point(resume_id, target_bb);
                }
                if (!expect(TokenKind::RBrace, "'}'")) return nullptr;
                continue;
            }

            // Parse block header: BlockName [ ( params ) ] :
            if (!is_identifier_or_keyword(peek().kind)) {
                error(peek().location, "Expected basic block label (identifier), got '" + std::string(peek().text) + "'");
                return nullptr;
            }

            std::string_view block_name = advance().text;
            std::string b_name_str(block_name);

            BasicBlock* bb = nullptr;
            auto it = block_map.find(b_name_str);
            if (it != block_map.end()) {
                bb = it->second;
                fn->append_block(bb);
            } else {
                bb = b.create_block(block_name);
                block_map[b_name_str] = bb;
                fn->append_block(bb);
            }
            b.position_at_end(bb);

            // Block parameters
            if (match(TokenKind::LParen)) {
                size_t p_idx = 0;
                while (!peek().is(TokenKind::RParen) && !peek().is(TokenKind::Eof)) {
                    if (!peek().is(TokenKind::ValueIdent)) {
                        error(peek().location, "Expected block parameter value identifier (e.g. '%0'), got '" + std::string(peek().text) + "'");
                        return nullptr;
                    }
                    std::string p_name = std::string(advance().text);
                    if (!expect(TokenKind::Colon, "':'")) return nullptr;
                    Type p_t = parse_type();
                    if (has_error_) return nullptr;

                    if (is_entry) {
                        // Entry block param mapped from function param or created
                        Value* val = b.add_block_param(bb, p_t);
                        value_map[p_name] = val;
                    } else {
                        Value* val = b.add_block_param(bb, p_t);
                        value_map[p_name] = val;
                    }
                    p_idx++;

                    if (!peek().is(TokenKind::RParen)) {
                        if (!expect(TokenKind::Comma, "','")) return nullptr;
                    }
                }
                if (!expect(TokenKind::RParen, "')'")) return nullptr;
            } else if (is_entry) {
                // If entry block has no explicit parameter list in label, create params from function signature
                for (size_t i = 0; i < param_types.size(); ++i) {
                    Value* val = b.add_block_param(bb, param_types[i]);
                    if (i < param_names.size()) {
                        value_map[param_names[i]] = val;
                    }
                }
            }

            if (!expect(TokenKind::Colon, "':'")) return nullptr;
            is_entry = false;

            // Parse instructions in this block
            while (!peek().is(TokenKind::RBrace) && !peek().is(TokenKind::Kw_resume_table) && !peek().is(TokenKind::Eof)) {
                // Check if next token is start of another block
                if (is_identifier_or_keyword(peek().kind) && lexer_.is_block_header_ahead()) {
                    break;
                }

                if (!parse_instruction(b, fn, value_map, get_or_create_block)) {
                    return nullptr;
                }
            }
        }

        if (!expect(TokenKind::RBrace, "'}'")) return nullptr;

        fn->rebuild_cfg_predecessors();
        return fn;
    }

    bool parse_instruction(Builder& b, Function* fn,
                           std::unordered_map<std::string, Value*>& value_map,
                           const std::function<BasicBlock*(std::string_view)>& get_or_create_block) {
        (void)fn;
        std::string result_name;
        bool has_assignment = false;

        if (peek().is(TokenKind::ValueIdent)) {
            result_name = std::string(advance().text);
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
            std::string name = std::string(advance().text);
            auto it = value_map.find(name);
            if (it == value_map.end()) {
                error(peek().location, "Use of undefined value: '" + name + "'");
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

        switch (op) {
            case Opcode::iconst_i32: {
                if (!peek().is(TokenKind::IntLiteral)) {
                    error(peek().location, "Expected integer literal for iconst.i32");
                    return false;
                }
                int32_t val = static_cast<int32_t>(advance().int_val);
                res_val = b.build_iconst_i32(val);
                break;
            }
            case Opcode::iconst_i64: {
                if (!peek().is(TokenKind::IntLiteral)) {
                    error(peek().location, "Expected integer literal for iconst.i64");
                    return false;
                }
                int64_t val = advance().int_val;
                res_val = b.build_iconst_i64(val);
                break;
            }
            case Opcode::fconst_f64: {
                if (!peek().is(TokenKind::FloatLiteral) && !peek().is(TokenKind::IntLiteral)) {
                    error(peek().location, "Expected float literal for fconst.f64");
                    return false;
                }
                Token tok = advance();
                double val = tok.is(TokenKind::FloatLiteral) ? tok.float_val : static_cast<double>(tok.int_val);
                res_val = b.build_fconst_f64(val);
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
                int32_t val = static_cast<int32_t>(advance().int_val);
                res_val = b.build_patchable_const_i32(sym, val);
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
                int64_t val = advance().int_val;
                res_val = b.build_patchable_const_i64(sym, val);
                break;
            }

            case Opcode::sext_i64: { Value* v = parse_val(); if (!v) return false; res_val = b.build_sext_i64(v); break; }
            case Opcode::zext_i64: { Value* v = parse_val(); if (!v) return false; res_val = b.build_zext_i64(v); break; }
            case Opcode::trunc_i32: { Value* v = parse_val(); if (!v) return false; res_val = b.build_trunc_i32(v); break; }
            case Opcode::fptosi_i32: { Value* v = parse_val(); if (!v) return false; res_val = b.build_fptosi_i32(v); break; }
            case Opcode::fptosi_i64: { Value* v = parse_val(); if (!v) return false; res_val = b.build_fptosi_i64(v); break; }
            case Opcode::sitofp_f64_i32: { Value* v = parse_val(); if (!v) return false; res_val = b.build_sitofp_f64_i32(v); break; }
            case Opcode::sitofp_f64_i64: { Value* v = parse_val(); if (!v) return false; res_val = b.build_sitofp_f64_i64(v); break; }
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
            case Opcode::ashr: {
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
                    default: break;
                }
                break;
            }

            case Opcode::neg: { Value* v = parse_val(); if (!v) return false; res_val = b.build_neg(v); break; }
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
                    case Opcode::eq: res_val = b.build_eq(lhs, rhs); break;
                    case Opcode::ne: res_val = b.build_ne(lhs, rhs); break;
                    case Opcode::slt: res_val = b.build_slt(lhs, rhs); break;
                    case Opcode::ult: res_val = b.build_ult(lhs, rhs); break;
                    case Opcode::sle: res_val = b.build_sle(lhs, rhs); break;
                    case Opcode::ule: res_val = b.build_ule(lhs, rhs); break;
                    case Opcode::sgt: res_val = b.build_sgt(lhs, rhs); break;
                    case Opcode::ugt: res_val = b.build_ugt(lhs, rhs); break;
                    case Opcode::sge: res_val = b.build_sge(lhs, rhs); break;
                    case Opcode::uge: res_val = b.build_uge(lhs, rhs); break;
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
                    offset = static_cast<int32_t>(advance().int_val);
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
                    offset = static_cast<int32_t>(advance().int_val);
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
                uint8_t scale = static_cast<uint8_t>(advance().int_val);
                int32_t offset = 0;
                if (match(TokenKind::Comma)) {
                    if (!peek().is(TokenKind::IntLiteral)) {
                        error(peek().location, "Expected integer offset for load_indexed");
                        return false;
                    }
                    offset = static_cast<int32_t>(advance().int_val);
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
                uint8_t scale = static_cast<uint8_t>(advance().int_val);
                if (!expect(TokenKind::Comma, "','")) return false;

                int32_t offset = 0;
                Value* val = nullptr;
                if (peek().is(TokenKind::IntLiteral)) {
                    offset = static_cast<int32_t>(advance().int_val);
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
                uint32_t resume_id = static_cast<uint32_t>(advance().int_val);
                b.build_resume_point(resume_id);
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
        }

        if (has_assignment && res_val) {
            value_map[result_name] = res_val;
        }

        return true;
    }

    Lexer lexer_;
    DiagnosticReporter* diag_ = nullptr;
    std::string_view filename_;
    bool has_error_ = false;
};

} // namespace

std::unique_ptr<Module> parse_module(std::string_view text, DiagnosticReporter* diag, std::string_view filename) {
    auto mod = std::make_unique<Module>();
    Parser parser(text, diag, filename);
    if (!parser.parse_module(*mod)) {
        return nullptr;
    }
    return mod;
}

bool parse_module_into(std::string_view text, Module& mod, DiagnosticReporter* diag, std::string_view filename) {
    Parser parser(text, diag, filename);
    return parser.parse_module(mod);
}

Function* parse_function(std::string_view text, Module& mod, DiagnosticReporter* diag, std::string_view filename) {
    Parser parser(text, diag, filename);
    return parser.parse_single_function(mod);
}

} // namespace brass
