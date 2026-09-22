#pragma once

#include <iostream>
#include <string>
#include <vector>
#include <functional>
#include <chrono>
#include <set>
#include <filesystem>
#include <system_error>
#if defined(_WIN32)
#include <process.h>
#else
#include <unistd.h>
#endif
#include <sstream>
#include <cmath>
#include <cstdlib>
#include <utility>
#include <type_traits>

namespace brass::test {

// Per-process scratch directory for files a test writes (objects, DLLs,
// PTX, generated sources). Tests run as separate, possibly concurrent
// processes, so fixed names under the shared temp dir would race.
// Removed after a fully passing run; kept after a failure for inspection.
struct ScratchDir {
    std::filesystem::path path;
    bool created = false;
};

inline ScratchDir& scratch_state() {
    static ScratchDir state;
    return state;
}

inline std::filesystem::path scratch_dir() {
    ScratchDir& s = scratch_state();
    if (!s.created) {
        std::error_code ec;
        std::filesystem::path base = std::filesystem::temp_directory_path(ec);
        if (ec) base = std::filesystem::current_path();
#if defined(_WIN32)
        auto pid = _getpid();
#else
        auto pid = getpid();
#endif
        s.path = base / "brass_tests" / std::to_string(pid);
        std::filesystem::remove_all(s.path, ec);
        std::filesystem::create_directories(s.path, ec);
        s.created = true;
    }
    return s.path;
}

inline void remove_scratch_dir() {
    ScratchDir& s = scratch_state();
    if (s.created) {
        std::error_code ec;
        std::filesystem::remove_all(s.path, ec);
        s.created = false;
    }
}

struct TestFailure {
    std::string file;
    int line;
    std::string expr;
    std::string message;
};

struct TestCase {
    std::string name;
    std::string file;
    int line;
    std::function<void()> func;
    std::vector<TestFailure> failures;
    bool passed = true;
    double duration_ms = 0.0;
};

class TestRegistry {
public:
    static TestRegistry& instance() {
        static TestRegistry reg;
        return reg;
    }

    void register_test(const std::string& name, const char* file, int line, std::function<void()> func) {
        tests_.push_back({name, file, line, std::move(func), {}, true, 0.0});
    }

    TestCase* current_test() {
        return current_test_;
    }

    void set_current_test(TestCase* tc) {
        current_test_ = tc;
    }

    // --list        print every test name, one per line (ctest discovery)
    // --exact=NAME  run only the test with exactly this name
    // --filter=STR  run the tests whose name contains STR
    // Selecting nothing is an error, so a renamed or compiled-out test cannot
    // pass silently.
    int run_all(int argc, char** argv) {
        std::string filter;
        std::string exact;
        bool has_exact = false;
        bool list = false;
        for (int i = 1; i < argc; ++i) {
            std::string arg = argv[i];
            if (arg.rfind("--filter=", 0) == 0) {
                filter = arg.substr(9);
            } else if (arg.rfind("--exact=", 0) == 0) {
                exact = arg.substr(8);
                has_exact = true;
            } else if (arg == "--list") {
                list = true;
            } else {
                std::cerr << "unknown argument: " << arg << "\n";
                return 2;
            }
        }

        if (list) {
            return list_tests();
        }

        int total = 0;
        int passed = 0;
        int failed = 0;

        std::cout << "==================================================\n";
        std::cout << "Running brass test suite...\n";
        std::cout << "==================================================\n";

        for (auto& tc : tests_) {
            if (has_exact ? tc.name != exact
                          : (!filter.empty() && tc.name.find(filter) == std::string::npos)) {
                continue;
            }

            total++;
            current_test_ = &tc;
            std::cout << "  [RUN ] " << tc.name << "\n" << std::flush;
            auto start = std::chrono::high_resolution_clock::now();
            try {
                tc.func();
            } catch (const std::exception& ex) {
                tc.passed = false;
                tc.failures.push_back({tc.file, tc.line, "exception", ex.what()});
            } catch (...) {
                tc.passed = false;
                tc.failures.push_back({tc.file, tc.line, "exception", "Unknown exception thrown"});
            }
            auto end = std::chrono::high_resolution_clock::now();
            tc.duration_ms = std::chrono::duration<double, std::milli>(end - start).count();

            if (tc.passed && tc.failures.empty()) {
                passed++;
                std::cout << "  [PASS] " << tc.name << " (" << tc.duration_ms << " ms)\n" << std::flush;
            } else {
                failed++;
                std::cout << "  [FAIL] " << tc.name << " (" << tc.duration_ms << " ms)\n" << std::flush;
                for (const auto& fail : tc.failures) {
                    std::cout << "         " << fail.file << ":" << fail.line << ": "
                              << fail.expr;
                    if (!fail.message.empty()) {
                        std::cout << " (" << fail.message << ")";
                    }
                    std::cout << "\n";
                }
            }
        }

        std::cout << "==================================================\n";
        std::cout << "Test Summary: " << passed << "/" << total << " passed";
        if (failed > 0) {
            std::cout << ", " << failed << " FAILED";
        }
        std::cout << "\n==================================================\n";

        if (total == 0) {
            std::cerr << "no test matched the selection\n";
            return 1;
        }
        if (failed == 0) {
            remove_scratch_dir();
        } else if (scratch_state().created) {
            std::cout << "Scratch files kept in " << scratch_state().path.string() << "\n";
        }
        return (failed == 0) ? 0 : 1;
    }

private:
    // Names are ctest identities, so they must be unique and must survive a
    // round trip through a CMake list.
    int list_tests() const {
        std::set<std::string> seen;
        int rc = 0;
        for (const auto& tc : tests_) {
            if (!seen.insert(tc.name).second) {
                std::cerr << "duplicate test name: " << tc.name << " (" << tc.file << ":" << tc.line << ")\n";
                rc = 2;
            }
            if (tc.name.find_first_of(";[]\\\"\n") != std::string::npos) {
                std::cerr << "test name has a character ctest discovery cannot carry: " << tc.name << "\n";
                rc = 2;
            }
            std::cout << tc.name << "\n";
        }
        return rc;
    }

    std::vector<TestCase> tests_;
    TestCase* current_test_ = nullptr;
};

struct AutoTestRegister {
    AutoTestRegister(const char* name, const char* file, int line, std::function<void()> func) {
        TestRegistry::instance().register_test(name, file, line, std::move(func));
    }
};

class RequireException : public std::runtime_error {
public:
    RequireException(const std::string& msg) : std::runtime_error(msg) {}
};

inline void report_failure(const char* file, int line, const char* expr, const std::string& msg, bool is_require) {
    auto* cur = TestRegistry::instance().current_test();
    if (cur) {
        cur->passed = false;
        cur->failures.push_back({file, line, expr, msg});
    }
    if (is_require) {
        throw RequireException(std::string(file) + ":" + std::to_string(line) + " REQUIRE failed: " + expr + " " + msg);
    }
}

template <typename T>
concept StandardInteger = std::is_integral_v<T> &&
    !std::is_same_v<T, bool> &&
    !std::is_same_v<T, char> &&
    !std::is_same_v<T, wchar_t> &&
    !std::is_same_v<T, char8_t> &&
    !std::is_same_v<T, char16_t> &&
    !std::is_same_v<T, char32_t>;

template <typename T, typename U>
inline bool check_equal_impl(const T& a, const U& b) {
    if constexpr (StandardInteger<T> && StandardInteger<U>) {
        return std::cmp_equal(a, b);
    } else {
        return a == b;
    }
}

template <typename T, typename U>
inline bool check_not_equal_impl(const T& a, const U& b) {
    if constexpr (StandardInteger<T> && StandardInteger<U>) {
        return !std::cmp_equal(a, b);
    } else {
        return a != b;
    }
}

template <typename T, typename U>
inline bool check_equal(const T& a, const U& b, const char* file, int line, const char* expr_a, const char* expr_b, bool is_require) {
    if (!check_equal_impl(a, b)) {
        std::ostringstream oss;
        oss << expr_a << " (" << a << ") == " << expr_b << " (" << b << ")";
        report_failure(file, line, "CHECK_EQ failed", oss.str(), is_require);
        return false;
    }
    return true;
}

template <typename T, typename U>
inline bool check_not_equal(const T& a, const U& b, const char* file, int line, const char* expr_a, const char* expr_b, bool is_require) {
    if (!check_not_equal_impl(a, b)) {
        std::ostringstream oss;
        oss << expr_a << " (" << a << ") != " << expr_b << " (" << b << ")";
        report_failure(file, line, "CHECK_NE failed", oss.str(), is_require);
        return false;
    }
    return true;
}

} // namespace brass::test

#define BRASS_CONCAT_INNER(a, b) a##b
#define BRASS_CONCAT(a, b) BRASS_CONCAT_INNER(a, b)

#define TEST_CASE(name) \
    static void BRASS_CONCAT(_brass_test_func_, __LINE__)(); \
    static ::brass::test::AutoTestRegister BRASS_CONCAT(_brass_test_reg_, __LINE__)(name, __FILE__, __LINE__, BRASS_CONCAT(_brass_test_func_, __LINE__)); \
    static void BRASS_CONCAT(_brass_test_func_, __LINE__)()

#define CHECK(expr) \
    do { \
        if (!(expr)) { \
            ::brass::test::report_failure(__FILE__, __LINE__, #expr, "", false); \
        } \
    } while ((void)0, 0)

#define REQUIRE(expr) \
    do { \
        if (!(expr)) { \
            ::brass::test::report_failure(__FILE__, __LINE__, #expr, "", true); \
        } \
    } while ((void)0, 0)

#define CHECK_FALSE(expr) CHECK(!(expr))
#define REQUIRE_FALSE(expr) REQUIRE(!(expr))

#define CHECK_EQ(a, b) ::brass::test::check_equal((a), (b), __FILE__, __LINE__, #a, #b, false)
#define REQUIRE_EQ(a, b) ::brass::test::check_equal((a), (b), __FILE__, __LINE__, #a, #b, true)

#define CHECK_NE(a, b) ::brass::test::check_not_equal((a), (b), __FILE__, __LINE__, #a, #b, false)
#define REQUIRE_NE(a, b) ::brass::test::check_not_equal((a), (b), __FILE__, __LINE__, #a, #b, true)

#define BRASS_TEST_MAIN() \
    int main(int argc, char** argv) { \
        return ::brass::test::TestRegistry::instance().run_all(argc, argv); \
    }
