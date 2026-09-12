#pragma once

#include "il_ast.hpp"
#include "il_lexer.hpp"
#include <brass/core/diagnostics.hpp>
#include <string_view>
#include <memory>

namespace brass::il {

struct TranslatorOptions;

class IlParser {
public:
    IlParser(std::string_view source, DiagnosticReporter* diag = nullptr, const TranslatorOptions* options = nullptr);

    bool parse_module(BronzeModuleAST& out_ast);

private:
    bool parse_function(BronzeFunction& out_fn);
    bool parse_block(BronzeBlock& out_block);
    bool parse_instruction(BronzeInstruction& out_inst);
    bool parse_block_target(BronzeBlockTarget& out_target);
    bool parse_type(BronzeType& out_type);

    void error(std::string_view msg, const Token& tok);
    bool match(TokenType type, Token* out_tok = nullptr);
    bool expect(TokenType type, std::string_view err_msg, Token* out_tok = nullptr);

    IlLexer lexer_;
    DiagnosticReporter* diag_ = nullptr;
    const TranslatorOptions* options_ = nullptr;
    bool has_error_ = false;
};

} // namespace brass::il
