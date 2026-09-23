#pragma once

#include <brass/mir/module.hpp>
#include <iosfwd>

namespace brass::pgo {
class ProfileData;
}

// brass-opt's analysis reports: printed views of a module that change
// nothing.
namespace brass::opt {

void report_branch_probabilities(const Module& mod, const pgo::ProfileData& profile, std::ostream& os);
void report_escape_analysis(const Module& mod, std::ostream& os);
void report_partial_escape(const Module& mod, std::ostream& os);
void report_alias_analysis(const Module& mod, std::ostream& os);

// Checks that printing, parsing and printing again is byte-identical;
// reports on `os` / `err` and returns the exit code.
int check_roundtrip(const Module& mod, std::ostream& os, std::ostream& err);

} // namespace brass::opt
