#include <brass/il_translator/il_translator.hpp>
#include "il_parser.hpp"
#include "il_lowering.hpp"

namespace brass::il {

TranslationResult translate_bronze_il(
    std::string_view il_text,
    const TranslatorOptions& options,
    DiagnosticReporter* diag
) {
    TranslationResult result;
    DiagnosticReporter default_diag;
    DiagnosticReporter* active_diag = diag ? diag : &default_diag;

    IlParser parser(il_text, active_diag);
    BronzeModuleAST ast;
    if (!parser.parse_module(ast)) {
        result.success = false;
        result.error_message = active_diag->has_errors() ? active_diag->format_all() : "Failed to parse Bronze IL";
        return result;
    }

    IlLowering lowering(options, active_diag);
    auto mod = lowering.lower_module(ast);
    if (!mod) {
        result.success = false;
        result.error_message = active_diag->has_errors() ? active_diag->format_all() : "Failed to lower Bronze IL AST to MIR";
        return result;
    }

    result.success = true;
    result.module = std::move(mod);
    return result;
}

} // namespace brass::il
