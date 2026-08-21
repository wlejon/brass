#pragma once

#include "il_ast.hpp"
#include <brass/il_translator/il_translator.hpp>
#include <brass/mir/builder.hpp>
#include <brass/core/diagnostics.hpp>
#include <memory>
#include <unordered_map>

namespace brass::il {

Type lower_type(BronzeType t);

class IlLowering {
public:
    IlLowering(const TranslatorOptions& options, DiagnosticReporter* diag = nullptr);

    std::unique_ptr<Module> lower_module(const BronzeModuleAST& ast);

private:
    bool lower_function(const BronzeFunction& fn_ast, Module& mod, const std::string& fn_name);
    bool lower_instruction(const BronzeInstruction& inst_ast, Builder& b, Function* fn,
                           std::unordered_map<uint32_t, Value*>& val_map,
                           const std::unordered_map<uint32_t, BasicBlock*>& block_map);

    Value* ensure_type(Value* val, Type target_type, Builder& b);

    TranslatorOptions options_;
    DiagnosticReporter* diag_ = nullptr;
    bool has_error_ = false;
};

} // namespace brass::il
