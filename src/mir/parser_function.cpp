// Textual MIR parser: top level (module / extern / func declarations), types,
// symbols and the block structure of a function body. Individual instructions
// are parsed in parser.cpp.

#include "parser_impl.hpp"
#include "parser_decode.hpp"

#include <brass/mir/runtime_symbols.hpp>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

namespace brass {

namespace mir_parser {

namespace {

// The top-level directives spelled as plain identifiers. Attribute and role
// lists end at one (the lexer does not see line ends).
bool is_top_level_directive(const Token& tok) {
    return tok.is(TokenKind::Ident) && (tok.text == "attributes" || tok.text == "string");
}

} // namespace

bool Parser::parse_module(Module& mod) {
    while (!peek().is(TokenKind::Eof)) {
        if (peek().is(TokenKind::Kw_module)) {
            if (!parse_module_decl(mod)) return false;
        } else if (peek().is(TokenKind::Kw_extern)) {
            if (!parse_extern_decl(mod)) return false;
        } else if (peek().is(TokenKind::Kw_func)) {
            if (!parse_function_decl(mod)) return false;
        } else if (peek().is(TokenKind::Ident) && peek().text == "attributes") {
            if (!parse_module_attributes(mod)) return false;
        } else if (peek().is(TokenKind::Ident) && peek().text == "string") {
            if (!parse_string_decl(mod)) return false;
        } else {
            error(peek().location, "Unexpected top-level token: '" + std::string(peek().text) + "'");
            return false;
        }
    }
    return !has_error_;
}

Function* Parser::parse_single_function(Module& mod) {
    if (!peek().is(TokenKind::Kw_func)) {
        error(peek().location, "Expected 'func', got '" + std::string(peek().text) + "'");
        return nullptr;
    }
    return parse_function_decl(mod);
}

Type Parser::parse_type() {
    Token tok = peek();
    if (tok.is(TokenKind::Kw_i32)) { advance(); return Type::i32(); }
    if (tok.is(TokenKind::Kw_i64)) { advance(); return Type::i64(); }
    if (tok.is(TokenKind::Kw_f32)) { advance(); return Type::f32(); }
    if (tok.is(TokenKind::Kw_f64)) { advance(); return Type::f64(); }
    if (tok.is(TokenKind::Kw_ptr)) { advance(); return Type::ptr(); }
    if (tok.is(TokenKind::Kw_gcref)) { advance(); return Type::gcref(); }
    if (tok.is(TokenKind::Kw_void)) { advance(); return Type::void_type(); }
    if (tok.is(TokenKind::Kw_f32x4)) { advance(); return Type::f32x4(); }
    if (tok.is(TokenKind::Kw_f64x2)) { advance(); return Type::f64x2(); }
    if (tok.is(TokenKind::Kw_i32x4)) { advance(); return Type::i32x4(); }
    if (tok.is(TokenKind::Kw_i64x2)) { advance(); return Type::i64x2(); }
    if (tok.is(TokenKind::Kw_f32x8)) { advance(); return Type::f32x8(); }
    if (tok.is(TokenKind::Kw_f64x4)) { advance(); return Type::f64x4(); }
    if (tok.is(TokenKind::Kw_i32x8)) { advance(); return Type::i32x8(); }
    if (tok.is(TokenKind::Kw_i64x4)) { advance(); return Type::i64x4(); }

    error(tok.location, "Expected type (i32, i64, f32, f64, ptr, gcref, void, f32x4, f64x2, i32x4, i64x2, f32x8, f64x4, i32x8, i64x4), got '" + std::string(tok.text) + "'");
    return Type::void_type();
}

std::string_view Parser::parse_symbol_name() {
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

bool Parser::parse_module_decl(Module& mod) {
    advance(); // consume 'module'
    std::string_view mod_name = parse_symbol_name();
    if (has_error_) return false;
    mod.set_name(mod_name);
    return true;
}

bool Parser::parse_extern_decl(Module& mod) {
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
    // Runtime-symbol roles (runtime_symbols.hpp), any number of them.
    while (peek().is(TokenKind::Ident) && !is_top_level_directive(peek())) {
        const std::optional<SymbolRole> role = parse_symbol_role(peek().text);
        if (!role) {
            error(peek().location, "Unknown runtime symbol role '" + std::string(peek().text) + "'");
            return false;
        }
        advance();
        mod.add_symbol_role(sym_name, *role);
    }
    return true;
}

// `attributes fp_reassociation pinned_tls_register loop_optimizations`:
// module attributes.
bool Parser::parse_module_attributes(Module& mod) {
    advance(); // consume 'attributes'
    while (peek().is(TokenKind::Ident) && !is_top_level_directive(peek())) {
        const std::string_view attr = peek().text;
        if (attr == "fp_reassociation") {
            mod.set_allow_fp_reassociation(true);
        } else if (attr == "pinned_tls_register") {
            mod.set_pinned_tls_register(true);
        } else if (attr == "loop_optimizations") {
            mod.set_has_loop_optimizations(true);
        } else {
            error(peek().location, "Unknown module attribute '" + std::string(attr) + "'");
            return false;
        }
        advance();
    }
    return true;
}

namespace {

int hex_digit(char c) {
    if (c >= '0' && c <= '9') return c - '0';
    if (c >= 'a' && c <= 'f') return c - 'a' + 10;
    if (c >= 'A' && c <= 'F') return c - 'A' + 10;
    return -1;
}

// Reverses the printer's quoting: `\\`, `\"`, `\xHH`. Anything else after a
// backslash, or a missing closing quote, is an error.
std::optional<std::string> unquote_mir_string(std::string_view lit) {
    if (lit.size() < 2 || lit.front() != '"' || lit.back() != '"') return std::nullopt;
    std::string out;
    for (size_t i = 1; i + 1 < lit.size(); ++i) {
        const char c = lit[i];
        if (c != '\\') {
            out += c;
            continue;
        }
        if (i + 2 >= lit.size()) return std::nullopt;
        const char e = lit[++i];
        if (e == '\\' || e == '"') {
            out += e;
        } else if (e == 'x') {
            if (i + 3 >= lit.size()) return std::nullopt;
            const int hi = hex_digit(lit[i + 1]), lo = hex_digit(lit[i + 2]);
            if (hi < 0 || lo < 0) return std::nullopt;
            out += static_cast<char>(hi * 16 + lo);
            i += 2;
        } else {
            return std::nullopt;
        }
    }
    return out;
}

} // namespace

// `string @sym "text"`: module-owned string data (Module::define_string_symbol).
bool Parser::parse_string_decl(Module& mod) {
    advance(); // consume 'string'
    std::string_view sym_name = parse_symbol_name();
    if (has_error_) return false;
    if (!peek().is(TokenKind::StringLiteral)) {
        error(peek().location, "Expected string literal, got '" + std::string(peek().text) + "'");
        return false;
    }
    const Token lit = advance();
    const std::optional<std::string> text = unquote_mir_string(lit.text);
    if (!text) {
        error(lit.location, "Malformed string literal " + std::string(lit.text));
        return false;
    }
    try {
        mod.define_string_symbol(sym_name, *text);
    } catch (const std::logic_error& e) {
        error(lit.location, e.what());
        return false;
    }
    return true;
}

// `resume_table { entry ID -> block ... }`
bool Parser::parse_resume_table(Function* fn, const BlockLookup& get_or_create_block) {
    advance(); // consume 'resume_table'
    if (!expect(TokenKind::LBrace, "'{'")) return false;

    while (!peek().is(TokenKind::RBrace) && !peek().is(TokenKind::Eof)) {
        if (!expect(TokenKind::Kw_entry, "'entry'")) return false;
        if (!peek().is(TokenKind::IntLiteral)) {
            error(peek().location, "Expected resume point integer ID, got '" + std::string(peek().text) + "'");
            return false;
        }
        uint32_t resume_id = static_cast<uint32_t>(advance().int_val);
        if (!expect(TokenKind::Arrow, "'->'")) return false;

        if (!is_identifier_or_keyword(peek().kind)) {
            error(peek().location, "Expected target block identifier, got '" + std::string(peek().text) + "'");
            return false;
        }
        fn->add_resume_point(resume_id, get_or_create_block(advance().text));
    }
    return expect(TokenKind::RBrace, "'}'");
}

Function* Parser::parse_function_decl(Module& mod) {
    advance(); // consume 'func'
    std::string_view fn_name = parse_symbol_name();
    if (has_error_) return nullptr;

    if (!expect(TokenKind::LParen, "'('")) return nullptr;

    std::vector<Type> param_types;
    std::vector<std::string> param_names;
    std::vector<bool> param_noalias;

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
        // Parameter attributes.
        bool noalias = false;
        while (peek().is(TokenKind::Ident)) {
            if (peek().text != "noalias") {
                error(peek().location, "Unknown parameter attribute '" + std::string(peek().text) + "'");
                return nullptr;
            }
            advance();
            noalias = true;
        }
        param_noalias.push_back(noalias);

        if (!peek().is(TokenKind::RParen)) {
            if (!expect(TokenKind::Comma, "','")) return nullptr;
        }
    }

    if (!expect(TokenKind::RParen, "')'")) return nullptr;
    if (!expect(TokenKind::Arrow, "'->'")) return nullptr;

    Type ret_type = parse_type();
    if (has_error_) return nullptr;

    // Function attributes.
    bool fp_reassociation = false;
    while (peek().is(TokenKind::Ident)) {
        if (peek().text != "fp_reassociation") {
            error(peek().location, "Unknown function attribute '" + std::string(peek().text) + "'");
            return nullptr;
        }
        advance();
        fp_reassociation = true;
    }

    if (!expect(TokenKind::LBrace, "'{'")) return nullptr;

    Function* fn = mod.create_function(fn_name, ret_type, param_types);
    fn->set_allow_fp_reassociation(fp_reassociation);
    for (size_t i = 0; i < param_noalias.size(); ++i) {
        if (param_noalias[i]) fn->set_param_noalias(i);
    }
    Builder b(mod);
    b.set_function(fn);

    ValueMap value_map;
    std::unordered_map<std::string, BasicBlock*> block_map;

    BlockLookup get_or_create_block = [&](std::string_view name) -> BasicBlock* {
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
            if (!parse_resume_table(fn, get_or_create_block)) return nullptr;
            continue;
        }

        // Parse block header: BlockName [ ( params ) ] :
        if (!is_identifier_or_keyword(peek().kind)) {
            error(peek().location, "Expected basic block label (identifier), got '" + std::string(peek().text) + "'");
            return nullptr;
        }

        // A block referenced by an earlier branch already exists; it takes
        // its layout position here, where it is defined.
        BasicBlock* bb = get_or_create_block(advance().text);
        fn->append_block(bb);
        b.position_at_end(bb);

        // Block parameters
        if (match(TokenKind::LParen)) {
            while (!peek().is(TokenKind::RParen) && !peek().is(TokenKind::Eof)) {
                if (!peek().is(TokenKind::ValueIdent)) {
                    error(peek().location, "Expected block parameter value identifier (e.g. '%0'), got '" + std::string(peek().text) + "'");
                    return nullptr;
                }
                std::string p_name = std::string(advance().text);
                if (!expect(TokenKind::Colon, "':'")) return nullptr;
                Type p_t = parse_type();
                if (has_error_) return nullptr;

                value_map[p_name] = b.add_block_param(bb, p_t);

                if (!peek().is(TokenKind::RParen)) {
                    if (!expect(TokenKind::Comma, "','")) return nullptr;
                }
            }
            if (!expect(TokenKind::RParen, "')'")) return nullptr;
        } else if (is_entry) {
            // An entry block without an explicit parameter list takes its
            // parameters from the function signature.
            for (size_t i = 0; i < param_types.size(); ++i) {
                value_map[param_names[i]] = b.add_block_param(bb, param_types[i]);
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

} // namespace mir_parser

std::unique_ptr<Module> parse_module(std::string_view text, DiagnosticReporter* diag, std::string_view filename) {
    auto mod = std::make_unique<Module>();
    mir_parser::Parser parser(text, diag, filename);
    if (!parser.parse_module(*mod)) {
        return nullptr;
    }
    return mod;
}

bool parse_module_into(std::string_view text, Module& mod, DiagnosticReporter* diag, std::string_view filename) {
    mir_parser::Parser parser(text, diag, filename);
    return parser.parse_module(mod);
}

Function* parse_function(std::string_view text, Module& mod, DiagnosticReporter* diag, std::string_view filename) {
    mir_parser::Parser parser(text, diag, filename);
    return parser.parse_single_function(mod);
}

} // namespace brass
