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

// The proof certificates alone: each write barrier the optimizer elided and
// each guard it eliminated, checked against the reason it recorded, without
// the structural checks. What a host's release build runs on everything the
// optimizer removed (the full verifier checks the translator, and costs a
// quarter of a large app's load). A lazy body is checked when it is built.
bool verify_certificates(const Function& fn, DiagnosticReporter* diag = nullptr);
bool verify_module_certificates(const Module& mod, DiagnosticReporter* diag = nullptr);

} // namespace brass
