#pragma once

#include <brass/mir/module.hpp>
#include <brass/mir/function.hpp>
#include <brass/core/diagnostics.hpp>
#include <memory>
#include <string_view>

namespace brass {

std::unique_ptr<Module> parse_module(std::string_view text, DiagnosticReporter* diag = nullptr, std::string_view filename = "<input>");
bool parse_module_into(std::string_view text, Module& mod, DiagnosticReporter* diag = nullptr, std::string_view filename = "<input>");

Function* parse_function(std::string_view text, Module& mod, DiagnosticReporter* diag = nullptr, std::string_view filename = "<input>");

} // namespace brass
