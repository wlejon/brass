#pragma once

#include "il_ast.hpp"
#include <brass/il_translator/il_translator.hpp>
#include <brass/il_translator/il_property.hpp>
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
                           const std::unordered_map<uint32_t, BasicBlock*>& block_map,
                           uint32_t handler_id = UINT32_MAX,
                           uint32_t block_id = 0,
                           uint32_t* cont_counter = nullptr);

    Value* ensure_type(Value* val, Type target_type, Builder& b);

    TranslatorOptions options_;
    DiagnosticReporter* diag_ = nullptr;
    PropertyLoweringHelper prop_lowering_;
    bool has_error_ = false;
};

} // namespace brass::il
