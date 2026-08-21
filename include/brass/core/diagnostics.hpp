#pragma once

#include <string>
#include <string_view>
#include <vector>
#include <cstdint>

namespace brass {

struct SourceLocation {
    std::string file;
    uint32_t line = 0;
    uint32_t column = 0;

    constexpr SourceLocation() noexcept = default;
    SourceLocation(std::string file_name, uint32_t l, uint32_t c)
        : file(std::move(file_name)), line(l), column(c) {}

    bool is_valid() const noexcept {
        return line > 0 || !file.empty();
    }

    std::string to_string() const;
};

enum class DiagnosticSeverity {
    Note,
    Warning,
    Error,
    Fatal
};

std::string_view severity_name(DiagnosticSeverity s) noexcept;

struct Diagnostic {
    DiagnosticSeverity severity = DiagnosticSeverity::Error;
    SourceLocation location;
    std::string message;

    std::string to_string() const;
};

class DiagnosticReporter {
public:
    DiagnosticReporter() = default;

    void report(DiagnosticSeverity severity, SourceLocation loc, std::string message);
    void report(DiagnosticSeverity severity, std::string message);

    void error(std::string message);
    void error(SourceLocation loc, std::string message);

    void warning(std::string message);
    void warning(SourceLocation loc, std::string message);

    void note(std::string message);
    void note(SourceLocation loc, std::string message);

    bool has_errors() const noexcept { return error_count_ > 0; }
    bool has_warnings() const noexcept { return warning_count_ > 0; }
    size_t error_count() const noexcept { return error_count_; }
    size_t warning_count() const noexcept { return warning_count_; }

    const std::vector<Diagnostic>& diagnostics() const noexcept { return diagnostics_; }
    std::string format_all() const;
    void clear();

private:
    std::vector<Diagnostic> diagnostics_;
    size_t error_count_ = 0;
    size_t warning_count_ = 0;
};

} // namespace brass
