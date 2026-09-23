#pragma once

// The textual MIR parser shared by parser_function.cpp (module, extern and
// function/block structure) and parser.cpp (one instruction at a time).
// Internal to the MIR library; the public entry points are in
// <brass/mir/parser.hpp>.

#include <brass/mir/parser.hpp>
#include <brass/mir/lexer.hpp>
#include <brass/mir/builder.hpp>

#include "parser_decode.hpp"

#include <deque>
#include <functional>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

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

    // Consumes the IntLiteral at the cursor (callers check the kind) into
    // `out` when it fits [lo, hi]; see int_literal_in_range.
    bool take_int(int64_t lo, int64_t hi, unsigned hex_bits, std::string_view what, int64_t& out) {
        Token tok = advance();
        std::string err;
        if (!int_literal_in_range(tok, lo, hi, hex_bits, what, out, err)) {
            error(tok.location, std::move(err));
            return false;
        }
        return true;
    }

    // A signed 32-bit memory offset.
    bool take_offset(int32_t& out) {
        int64_t v = 0;
        if (!take_int(INT32_MIN, INT32_MAX, 0, "a 32-bit offset", v)) return false;
        out = static_cast<int32_t>(v);
        return true;
    }

    // An unsigned immediate held in [0, hi].
    template <typename T>
    bool take_unsigned(int64_t hi, std::string_view what, T& out) {
        int64_t v = 0;
        if (!take_int(0, hi, 0, what, v)) return false;
        out = static_cast<T>(v);
        return true;
    }

    // parser_function.cpp
    Type parse_type();
    std::string_view parse_symbol_name();
    bool parse_module_decl(Module& mod);
    bool parse_extern_decl(Module& mod);
    bool parse_module_attributes(Module& mod);
    bool parse_string_decl(Module& mod);
    Function* parse_function_decl(Module& mod);
    bool parse_resume_table(Function* fn, const BlockLookup& get_or_create_block);

    // parser.cpp: one instruction into the builder's current block.
    bool parse_instruction(Builder& b, Function* fn, ValueMap& value_map, const BlockLookup& get_or_create_block);

    // parser_function.cpp: the instructions of `bb` up to the next block
    // header. Deferred: the body used a name no block parsed so far defines;
    // its instructions and bindings were rolled back so the caller can parse
    // it again (from a saved lexer) once more blocks are known.
    enum class BlockParse { Done, Deferred, Failed };
    BlockParse parse_block_body(Builder& b, Function* fn, BasicBlock* bb, ValueMap& value_map,
                                const BlockLookup& get_or_create_block);

    Lexer lexer_;
    // Forward references inside a function body: while forward_refs_ok_,
    // parse_val resolves an unknown name to forward_placeholder_ and sets
    // saw_forward_ref_ instead of reporting it; forward_ref_name_ is the
    // first such name in the body, which the body waits on before it is
    // parsed again. block_bindings_ holds the names the current block body
    // bound, with the value each one had before, so a deferred body can be
    // rolled back.
    bool forward_refs_ok_ = false;
    bool saw_forward_ref_ = false;
    std::string forward_ref_name_;
    Value forward_placeholder_{0, Type::i64(), ValueKind::InstructionResult};
    std::vector<std::pair<std::string, Value*>> block_bindings_;
    // Unescaped `@"..."` names; a deque so views handed out stay valid.
    std::deque<std::string> quoted_symbols_;
    DiagnosticReporter* diag_ = nullptr;
    std::string_view filename_;
    bool has_error_ = false;
};

} // namespace brass::mir_parser
