#pragma once

#include "il_ast.hpp"
#include <brass/il_translator/il_translator.hpp>
#include <brass/il_translator/il_property.hpp>
#include <brass/mir/builder.hpp>
#include <brass/core/diagnostics.hpp>
#include <memory>
#include <unordered_map>
#include <unordered_set>

namespace brass::il {

Type lower_type(BronzeType t);

class IlLowering {
public:
    IlLowering(const TranslatorOptions& options, DiagnosticReporter* diag = nullptr);

    std::unique_ptr<Module> lower_module(const BronzeModuleAST& ast);
    Value* ensure_type(Value* val, Type target_type, Builder& b);
    std::string resolve_callee(const std::string& callee_name) const;
    Value* get_key_id(Builder& b, uint32_t key_idx);

    Value* get_val_by_id(uint32_t id, Builder& b, const std::unordered_map<uint32_t, Value*>& val_map);
    void set_inst_result(uint32_t result_id, Value* res_val, Builder& b, std::unordered_map<uint32_t, Value*>& val_map);
    Value* current_fn_frame_ptr() const { return current_fn_frame_ptr_; }

private:
    bool lower_function(const BronzeFunction& fn_ast, Module& mod, const std::string& fn_name);
    bool emit_wrapper(const BronzeFunction& fn_ast, Module& mod, const std::string& fn_name, uint32_t declared_param_count, bool is_closure = false);
    bool lower_instruction(const BronzeInstruction& inst_ast, Builder& b, Function* fn,
                           std::unordered_map<uint32_t, Value*>& val_map,
                           const std::unordered_map<uint32_t, BasicBlock*>& block_map,
                           uint32_t handler_id = UINT32_MAX,
                           uint32_t block_id = 0,
                           uint32_t* cont_counter = nullptr);

    TranslatorOptions options_;
    DiagnosticReporter* diag_ = nullptr;
    PropertyLoweringHelper prop_lowering_;
    uint32_t current_file_id_ = 0;
    bool has_error_ = false;
    const BronzeModuleAST* current_ast_ = nullptr;
    size_t current_fn_idx_ = 0;
    std::unordered_map<size_t, std::unordered_map<std::string, std::string>> caller_to_callee_map_;
    std::unordered_set<uint32_t> module_env_regs_;
    std::unordered_map<uint32_t, uint32_t> current_fn_slot_of_;
    Value* current_fn_frame_ptr_ = nullptr;
};

} // namespace brass::il
