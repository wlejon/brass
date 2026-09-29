#pragma once

// Performance ratchet: one notion of pass. Every benchmark reports one
// number (its key's median over repetitions); the ratchet holds the golden
// value measured on a quiet machine, and a run fails only when the number
// regresses past the golden by more than that key's margin. There are no
// aspiration targets: a design claim (the stack-map GC model beating a
// shadow stack, compile speed) is held by its golden like everything else,
// and the goldens sit well inside those claims.
//
// Included from bench_utils.hpp after BenchmarkResult; do not include
// directly.

#include <algorithm>
#include <cctype>
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
    // from bench/ratchet*.json, so a missing file means every key is new.
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

    RatchetCheck check(const BenchmarkResult& res) const {
        RatchetCheck c;
        if (!has_ratio(res.key)) return c;
        const RatchetPolicy p = policy(res.key);
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
            const RatchetPolicy p = policy(res.key);
            std::ostringstream oss;
            oss << std::fixed << std::setprecision(3) << res.key << " (" << res.name << "): measured "
                << res.ratio << p.unit << (p.higher_is_better ? " below " : " above ") << c.bound << p.unit
                << " (golden " << c.golden << p.unit << ", margin " << static_cast<int>(p.margin * 100.0 + 0.5)
                << "%)";
            out_failures.push_back(oss.str());
        }
        return all_passed;
    }

    static std::string find_ratchet_file(const std::string& explicit_path = "") {
        if (!explicit_path.empty()) {
            std::ifstream f(explicit_path);
            if (f.good()) return explicit_path;
        }
#if defined(BRASS_BENCH_RATCHET_PATH)
        {
            std::ifstream f(BRASS_BENCH_RATCHET_PATH);
            if (f.good()) return BRASS_BENCH_RATCHET_PATH;
        }
#endif
        const std::vector<std::string> candidates = {
            "bench/ratchet.json",
            "../bench/ratchet.json",
            "../../bench/ratchet.json"
        };
        for (const auto& path : candidates) {
            std::ifstream f(path);
            if (f.good()) return path;
        }
        return "bench/ratchet.json";
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
                    ratios_[key] = std::stod(content.substr(val_start, val_end - val_start));
                    any = true;
                } catch (...) {}
            }
            pos = std::max(val_end, quote_end + 1);
        }
        return any;
    }

    // Per-platform goldens. The ratios compare brass against native code the
    // host compiler built, so a baseline pinned under MSVC need not hold
    // against GCC or Clang: bench/ratchet.<platform>.json, when present,
    // overrides the keys it names.
    static const char* platform_name() {
#if defined(_WIN32)
        return "windows";
#elif defined(__APPLE__)
        return "macos";
#else
        return "linux";
#endif
    }

    static std::string platform_overlay_path(const std::string& base_path) {
        const std::string ext = ".json";
        std::string stem = base_path;
        if (stem.size() >= ext.size() && stem.compare(stem.size() - ext.size(), ext.size(), ext) == 0) {
            stem.resize(stem.size() - ext.size());
        }
        return stem + "." + platform_name() + ext;
    }

    bool load_overlay(const std::string& filepath) {
        RatchetManager overlay;
        if (!overlay.load(filepath)) return false;
        for (const auto& [key, val] : overlay.ratios_) {
            ratios_[key] = val;
            overlay_keys_.push_back(key);
        }
        return true;
    }

    bool has_overlay() const { return !overlay_keys_.empty(); }

    // --update-ratchet. With a platform overlay loaded, every measured key is
    // written to the overlay and the base file is left alone, so re-baselining
    // one platform never rewrites another's goldens. Without one, measured
    // keys go to the base file. Keys a run did not measure are kept.
    void update_from_results(const std::vector<BenchmarkResult>& results) {
        for (const auto& res : results) {
            if (res.key.empty() || res.ratio <= 0.0) continue;
            ratios_[res.key] = res.ratio;
            if (has_overlay() &&
                std::find(overlay_keys_.begin(), overlay_keys_.end(), res.key) == overlay_keys_.end()) {
                overlay_keys_.push_back(res.key);
            }
        }
    }

    bool save(const std::string& filepath) const { return save_filtered(filepath, false); }
    bool save_overlay(const std::string& filepath) const { return save_filtered(filepath, true); }

private:
    bool save_filtered(const std::string& filepath, bool overlay_only) const {
        // Reload what is on disk so keys this process does not own survive.
        RatchetManager on_disk;
        on_disk.load(filepath);
        std::map<std::string, double> rows = on_disk.ratios_;
        for (const auto& [key, val] : ratios_) {
            const bool in_overlay = std::find(overlay_keys_.begin(), overlay_keys_.end(), key) != overlay_keys_.end();
            if (in_overlay == overlay_only) rows[key] = val;
        }
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

    std::map<std::string, double> ratios_;
    std::vector<std::string> overlay_keys_;
};

} // namespace brass::bench
