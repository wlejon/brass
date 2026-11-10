#include <brass/mir/verifier.hpp>
#include <brass/core/parallel_for.hpp>
#include <string>
#include <unordered_set>
#include <vector>

// verify_module over many threads: each function verified on its own (the
// verifier only reads, and a function's checks read nothing another
// function's checks write), its diagnostics collected apart and reported in
// module order, with the module-level ones where the serial walk reports
// them, so the report is the serial verifier's.
namespace brass {

namespace {

// A thread per this many blocks; below it, threads cost more than they save.
constexpr size_t kBlocksPerThread = 4000;

} // namespace

bool verify_module(const Module& mod, DiagnosticReporter* diag, size_t threads) {
    const std::vector<Function*>& fns = mod.functions();
    std::vector<size_t> weights(fns.size(), 1);
    size_t total = 0;
    for (size_t i = 0; i < fns.size(); ++i) {
        if (fns[i]) weights[i] += fns[i]->block_count();
        total += weights[i];
    }
    const size_t n = threads_for_work(threads, fns.size(), total, kBlocksPerThread);
    if (n <= 1) return verify_module(mod, diag);

    // The module-level checks, in the serial walk's order.
    std::vector<std::string> before(fns.size());
    std::unordered_set<std::string_view> names;
    for (size_t i = 0; i < fns.size(); ++i) {
        if (!fns[i]) {
            before[i] = "Module contains a null function pointer.";
        } else if (!names.insert(fns[i]->name()).second) {
            before[i] = "Duplicate function name in module: '" + std::string(fns[i]->name()) + "'";
        }
    }

    std::vector<DiagnosticReporter> reports(fns.size());
    std::vector<uint8_t> ok(fns.size(), 1);
    parallel_for_each(heaviest_first(weights), n, [&](size_t i) {
        // A lazy body is not there yet; its producer verifies it when it
        // builds it (Module::set_body_provider).
        if (!fns[i] || fns[i]->is_lazy()) return;
        Verifier v(&reports[i]);
        ok[i] = v.verify_function(*fns[i]) ? 1 : 0;
    });

    bool valid = true;
    for (size_t i = 0; i < fns.size(); ++i) {
        if (!before[i].empty()) {
            valid = false;
            if (diag) diag->error(before[i]);
        }
        if (!ok[i]) valid = false;
        if (!diag) continue;
        for (const Diagnostic& d : reports[i].diagnostics()) diag->report(d.severity, d.location, d.message);
    }
    return valid;
}

} // namespace brass
