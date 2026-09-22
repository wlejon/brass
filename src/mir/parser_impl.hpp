#pragma once

// The textual MIR parser shared by parser_function.cpp (module, extern and
// function/block structure) and parser.cpp (one instruction at a time).
// Internal to the MIR library; the public entry points are in
// <brass/mir/parser.hpp>.

#include <brass/mir/parser.hpp>
#include <brass/mir/lexer.hpp>
#include <brass/mir/builder.hpp>

#include <functional>
#include <string>
#include <string_view>
#include <unordered_map>

namespace brass::mir_parser {

class Parser {
public:
    Parser(std::string_view text, DiagnosticReporter* diag, std::string_view filename)
        : lexer_(text, filename), diag_(diag), filename_(filename) {}

    // parser_function.cpp
    bool parse_module(Module& mod);
    Function* parse_single_function(Module& mod);

private:
    using ValueMap = std::unordered_map<std::string, Value*>;
    using BlockLookup = std::function<BasicBlock*(std::string_view)>;

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

    // parser_function.cpp
    Type parse_type();
    std::string_view parse_symbol_name();
    bool parse_module_decl(Module& mod);
    bool parse_extern_decl(Module& mod);
    Function* parse_function_decl(Module& mod);
    bool parse_resume_table(Function* fn, const BlockLookup& get_or_create_block);

    // parser.cpp: one instruction into the builder's current block.
    bool parse_instruction(Builder& b, Function* fn, ValueMap& value_map, const BlockLookup& get_or_create_block);

    Lexer lexer_;
    DiagnosticReporter* diag_ = nullptr;
    std::string_view filename_;
    bool has_error_ = false;
};

} // namespace brass::mir_parser
