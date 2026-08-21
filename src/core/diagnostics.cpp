#include <brass/core/diagnostics.hpp>
#include <sstream>

namespace brass {

std::string SourceLocation::to_string() const {
    if (!is_valid()) {
        return "<unknown>";
    }
    std::ostringstream oss;
    if (!file.empty()) {
        oss << file << ":";
    }
    if (line > 0) {
        oss << line;
        if (column > 0) {
            oss << ":" << column;
        }
    }
    return oss.str();
}

std::string_view severity_name(DiagnosticSeverity s) noexcept {
    switch (s) {
        case DiagnosticSeverity::Note: return "note";
        case DiagnosticSeverity::Warning: return "warning";
        case DiagnosticSeverity::Error: return "error";
        case DiagnosticSeverity::Fatal: return "fatal error";
    }
    return "error";
}

std::string Diagnostic::to_string() const {
    std::ostringstream oss;
    if (location.is_valid()) {
        oss << location.to_string() << ": ";
    }
    oss << severity_name(severity) << ": " << message;
    return oss.str();
}

void DiagnosticReporter::report(DiagnosticSeverity severity, SourceLocation loc, std::string message) {
    if (severity == DiagnosticSeverity::Error || severity == DiagnosticSeverity::Fatal) {
        error_count_++;
    } else if (severity == DiagnosticSeverity::Warning) {
        warning_count_++;
    }
    diagnostics_.push_back(Diagnostic{severity, std::move(loc), std::move(message)});
}

void DiagnosticReporter::report(DiagnosticSeverity severity, std::string message) {
    report(severity, SourceLocation{}, std::move(message));
}

void DiagnosticReporter::error(std::string message) {
    report(DiagnosticSeverity::Error, std::move(message));
}

void DiagnosticReporter::error(SourceLocation loc, std::string message) {
    report(DiagnosticSeverity::Error, std::move(loc), std::move(message));
}

void DiagnosticReporter::warning(std::string message) {
    report(DiagnosticSeverity::Warning, std::move(message));
}

void DiagnosticReporter::warning(SourceLocation loc, std::string message) {
    report(DiagnosticSeverity::Warning, std::move(loc), std::move(message));
}

void DiagnosticReporter::note(std::string message) {
    report(DiagnosticSeverity::Note, std::move(message));
}

void DiagnosticReporter::note(SourceLocation loc, std::string message) {
    report(DiagnosticSeverity::Note, std::move(loc), std::move(message));
}

std::string DiagnosticReporter::format_all() const {
    std::ostringstream oss;
    for (const auto& diag : diagnostics_) {
        oss << diag.to_string() << "\n";
    }
    return oss.str();
}

void DiagnosticReporter::clear() {
    diagnostics_.clear();
    error_count_ = 0;
    warning_count_ = 0;
}

} // namespace brass
