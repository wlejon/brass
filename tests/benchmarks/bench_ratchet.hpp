#pragma once

// Performance ratchet: one notion of pass. Every benchmark reports one
// number (its key's median over repetitions); the ratchet holds the golden
// value measured on a quiet machine, and a run fails only when the number
// regresses past the golden by more than that key's margin. There are no
// aspiration targets: a design claim (the stack-map GC model beating a
// shadow stack, compile speed) is held by its golden like everything else,
// and the goldens sit well inside those claims.
//
// Goldens are per machine: bench/hosts/<host>.json. Every key compares brass
// with native code the host compiler built (or, for compile_speed, is a raw
// time), so a number is only meaningful against the same CPU and toolchain.
// A machine with no file of its own reports its numbers and passes.
//
// Included from bench_utils.hpp after BenchmarkResult; do not include
// directly.

#include <algorithm>
#include <cctype>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <map>
#include <sstream>
#include <string>
#include <vector>

namespace brass::bench {

struct RatchetPolicy {
    bool higher_is_better = false; // speedups; everything else is a time ratio
    double margin = 0.10;          // allowed regression, as a fraction
    const char* unit = "x";
};

enum class RatchetVerdict { Pass, Fail, New };

struct RatchetCheck {
    RatchetVerdict verdict = RatchetVerdict::New;
    double golden = 0.0;
    double bound = 0.0; // worst value that still passes
};

class RatchetManager {
public:
    static RatchetManager*& active() {
        static RatchetManager* s_active = nullptr;
        return s_active;
    }

    // Kept for call sites that take a ratchet by default; goldens come only
    // from bench/hosts/<host>.json, so a missing file means every key is new.
    static RatchetManager defaults() { return RatchetManager{}; }

    // Per-key direction and margin. The default margin is 10%; a key whose
    // quiet-machine run-to-run spread is wider gets a wider margin so a quiet
    // run is reliably green, while staying tight enough that a real 15%
    // regression is red. Spreads are from repeated quiet runs on the Windows
    // box (README.md, Performance Tracking).
    static RatchetPolicy policy(const std::string& key) {
        RatchetPolicy p;
        // Quiet pinned runs (Windows, 7950X3D) put every key's median within
        // +4% of the five-run median (the golden) except the ones below,
        // which reached +4..+9% (cheney_gc, the 64x64 i64 matmuls, the fast
        // interpreter's loop-heavy programs); 12% keeps them green and still
        // fails a 15% regression.
        if (key == "gc_model_speedup") {
            p.higher_is_better = true;
        } else if (key == "compile_speed") {
            p.unit = " ms";
        } else if (key == "cheney_gc" || key == "matmul_i64_64_naive" || key == "matmul_i64_64_preopt") {
            p.margin = 0.12;
        } else if (key.rfind("interp_", 0) == 0) {
            // The fast interpreter's dispatch loop is library code the linker
            // places after the benchmark objects, so an unrelated edit to the
            // benchmarks moves its alignment: interp_matmul_32x32 went +18%
            // (and interp_collatz -10%) across such a relink. Until the
            // interpreter's hot loop has a fixed placement, these keys only
            // catch regressions past 25%.
            p.margin = 0.25;
        }
        return p;
    }

    bool has_ratio(const std::string& key) const {
        return ratios_.find(key) != ratios_.end();
    }

    double get_ratio(const std::string& key, double default_val = 0.0) const {
        auto it = ratios_.find(key);
        return it != ratios_.end() ? it->second : default_val;
    }

    void set_ratio(const std::string& key, double val) { ratios_[key] = val; }

    const std::map<std::string, double>& ratios() const { return ratios_; }

    // The key's policy on this host: policy(key), with the margin widened
    // when this host's goldens file says so ("margin.<key>", written by
    // --update-ratchet for a key whose recording runs disagreed).
    RatchetPolicy policy_for(const std::string& key) const {
        RatchetPolicy p = policy(key);
        auto it = margins_.find(key);
        if (it != margins_.end() && it->second > p.margin) p.margin = it->second;
        return p;
    }

    // --update-ratchet: `vals` are one key's numbers from the recording runs
    // and `golden` their median. Where the runs spread past half the key's
    // margin, this host cannot hold the key to it: some keys are bimodal
    // from process to process on some machines (fib on halo, Strix Halo;
    // interp_prime_sieve_100k on the Windows box and the M2 Pro; 20-35%
    // apart, not ASLR on halo), so the margin becomes the observed worst-run
    // excursion plus the key's margin. Otherwise any widening is dropped.
    // Returns the recorded margin override, or 0 when there is none.
    double record_margin(const std::string& key, const std::vector<double>& vals, double golden) {
        margins_.erase(key);
        if (vals.size() < 2 || golden <= 0.0) return 0.0;
        const RatchetPolicy p = policy(key);
        double worst = golden;
        for (double v : vals) worst = p.higher_is_better ? std::min(worst, v) : std::max(worst, v);
        if (worst <= 0.0) return 0.0;
        const double spread = p.higher_is_better ? golden / worst - 1.0 : worst / golden - 1.0;
        if (spread <= 0.5 * p.margin) return 0.0;
        const double m = std::ceil((spread + p.margin) * 100.0) / 100.0;
        margins_[key] = m;
        return m;
    }

    RatchetCheck check(const BenchmarkResult& res) const {
        RatchetCheck c;
        if (!has_ratio(res.key)) return c;
        const RatchetPolicy p = policy_for(res.key);
        c.golden = get_ratio(res.key);
        if (p.higher_is_better) {
            c.bound = c.golden / (1.0 + p.margin);
            c.verdict = res.ratio >= c.bound ? RatchetVerdict::Pass : RatchetVerdict::Fail;
        } else {
            c.bound = c.golden * (1.0 + p.margin);
            c.verdict = res.ratio <= c.bound ? RatchetVerdict::Pass : RatchetVerdict::Fail;
        }
        return c;
    }

    bool check_ratchet(const std::vector<BenchmarkResult>& results, std::vector<std::string>& out_failures) const {
        bool all_passed = true;
        for (const auto& res : results) {
            if (res.key.empty()) continue;
            const RatchetCheck c = check(res);
            if (c.verdict != RatchetVerdict::Fail) continue;
            all_passed = false;
            const RatchetPolicy p = policy_for(res.key);
            std::ostringstream oss;
            oss << std::fixed << std::setprecision(3) << res.key << " (" << res.name << "): measured "
                << res.ratio << p.unit << (p.higher_is_better ? " below " : " above ") << c.bound << p.unit
                << " (golden " << c.golden << p.unit << ", margin " << static_cast<int>(p.margin * 100.0 + 0.5)
                << "%)";
            out_failures.push_back(oss.str());
        }
        return all_passed;
    }

    // A host name as a file name: lower case, the first DNS label only
    // ("Jonnys-MacBook-Pro.local" -> "jonnys-macbook-pro"), anything outside
    // [a-z0-9_-] replaced by '-'.
    static std::string normalize_host(const std::string& raw) {
        std::string out;
        for (char ch : raw) {
            if (ch == '.') break;
            const unsigned char c = static_cast<unsigned char>(std::tolower(static_cast<unsigned char>(ch)));
            out += (std::isalnum(c) || c == '-' || c == '_') ? static_cast<char>(c) : '-';
        }
        return out.empty() ? "unknown-host" : out;
    }

    // The goldens for `host`: <bench dir>/hosts/<host>.json, the bench dir
    // being the source tree's bench/ (BRASS_BENCH_DIR) or, failing that, the
    // first bench/ found from the working directory up.
    static std::string host_ratchet_path(const std::string& host) {
        std::string dir = "bench";
#if defined(BRASS_BENCH_DIR)
        dir = BRASS_BENCH_DIR;
#else
        for (const char* cand : {"bench", "../bench", "../../bench"}) {
            std::error_code ec;
            if (std::filesystem::is_directory(std::string(cand) + "/hosts", ec)) { dir = cand; break; }
        }
#endif
        return dir + "/hosts/" + host + ".json";
    }

    // Flat {"key": number, ...} reader.
    bool load(const std::string& filepath) {
        std::ifstream file(filepath);
        if (!file.is_open()) return false;
        std::string content((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());

        size_t pos = 0;
        bool any = false;
        while (pos < content.size()) {
            size_t quote_start = content.find('"', pos);
            if (quote_start == std::string::npos) break;
            size_t quote_end = content.find('"', quote_start + 1);
            if (quote_end == std::string::npos) break;
            std::string key = content.substr(quote_start + 1, quote_end - quote_start - 1);

            size_t colon_pos = content.find(':', quote_end);
            if (colon_pos == std::string::npos) break;
            size_t val_start = colon_pos + 1;
            while (val_start < content.size() && std::isspace(static_cast<unsigned char>(content[val_start]))) {
                val_start++;
            }
            size_t val_end = val_start;
            while (val_end < content.size() &&
                   (std::isdigit(static_cast<unsigned char>(content[val_end])) || content[val_end] == '.' ||
                    content[val_end] == '-' || content[val_end] == '+' || content[val_end] == 'e' ||
                    content[val_end] == 'E')) {
                val_end++;
            }
            if (val_end > val_start) {
                try {
                    const double v = std::stod(content.substr(val_start, val_end - val_start));
                    if (key.rfind(kMarginPrefix, 0) == 0) {
                        margins_[key.substr(sizeof(kMarginPrefix) - 1)] = v;
                    } else {
                        ratios_[key] = v;
                    }
                    any = true;
                } catch (...) {}
            }
            pos = std::max(val_end, quote_end + 1);
        }
        return any;
    }

    // The median of a key's numbers over several suite runs (--runs).
    static double median(std::vector<double> vals) {
        if (vals.empty()) return 0.0;
        std::sort(vals.begin(), vals.end());
        const size_t n = vals.size();
        return n % 2 ? vals[n / 2] : 0.5 * (vals[n / 2 - 1] + vals[n / 2]);
    }

    bool save(const std::string& filepath) const {
        // Reload what is on disk so keys this process does not own survive.
        RatchetManager on_disk;
        on_disk.load(filepath);
        std::map<std::string, double> rows = on_disk.ratios_;
        for (const auto& [key, val] : ratios_) rows[key] = val;
        // Margins are this object's alone: it loaded the file's, and
        // record_margin adds or drops one per recorded key.
        for (const auto& [key, val] : margins_) rows[kMarginPrefix + key] = val;
        std::error_code ec;
        const std::filesystem::path parent = std::filesystem::path(filepath).parent_path();
        if (!parent.empty()) std::filesystem::create_directories(parent, ec);
        std::ofstream file(filepath);
        if (!file.is_open()) return false;
        file << "{\n";
        size_t idx = 0;
        for (const auto& [key, val] : rows) {
            file << "  \"" << key << "\": " << std::defaultfloat << std::setprecision(4) << val;
            if (++idx < rows.size()) file << ",";
            file << "\n";
        }
        file << "}\n";
        return true;
    }

private:
    static constexpr char kMarginPrefix[] = "margin.";

    std::map<std::string, double> ratios_;
    std::map<std::string, double> margins_;  // per-host widened margins, by key
};

} // namespace brass::bench
