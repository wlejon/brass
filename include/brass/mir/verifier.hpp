#pragma once

#include <brass/mir/module.hpp>
#include <brass/mir/function.hpp>
#include <brass/core/diagnostics.hpp>

namespace brass {

class Verifier {
public:
    explicit Verifier(DiagnosticReporter* diag = nullptr) noexcept
        : diag_(diag) {}

    bool verify_module(const Module& mod);
    bool verify_function(const Function& fn);

private:
    void report_error(std::string msg);
    void report_warning(std::string msg);

    DiagnosticReporter* diag_ = nullptr;
    bool has_error_ = false;
};

bool verify_module(const Module& mod, DiagnosticReporter* diag = nullptr);
bool verify_function(const Function& fn, DiagnosticReporter* diag = nullptr);

} // namespace brass
