#pragma once

#include <iostream>
#include <string>
#include <vector>
#include <functional>
#include <chrono>
#include <sstream>
#include <cmath>
#include <cstdlib>

namespace brass::test {

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

    int run_all(int argc, char** argv) {
        std::string filter;
        for (int i = 1; i < argc; ++i) {
            std::string arg = argv[i];
            if (arg.rfind("--filter=", 0) == 0) {
                filter = arg.substr(9);
            }
        }

        int total = 0;
        int passed = 0;
        int failed = 0;

        std::cout << "==================================================\n";
        std::cout << "Running brass test suite...\n";
        std::cout << "==================================================\n";

        for (auto& tc : tests_) {
            if (!filter.empty() && tc.name.find(filter) == std::string::npos) {
                continue;
            }

            total++;
            current_test_ = &tc;
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
                std::cout << "  [PASS] " << tc.name << " (" << tc.duration_ms << " ms)\n";
            } else {
                failed++;
                std::cout << "  [FAIL] " << tc.name << " (" << tc.duration_ms << " ms)\n";
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

        return (failed == 0) ? 0 : 1;
    }

private:
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

template <typename T, typename U>
inline bool check_equal_impl(const T& a, const U& b) {
    return a == b;
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
    if (a == b) {
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
    } while (false)

#define REQUIRE(expr) \
    do { \
        if (!(expr)) { \
            ::brass::test::report_failure(__FILE__, __LINE__, #expr, "", true); \
        } \
    } while (false)

#define CHECK_EQ(a, b) ::brass::test::check_equal((a), (b), __FILE__, __LINE__, #a, #b, false)
#define REQUIRE_EQ(a, b) ::brass::test::check_equal((a), (b), __FILE__, __LINE__, #a, #b, true)

#define CHECK_NE(a, b) ::brass::test::check_not_equal((a), (b), __FILE__, __LINE__, #a, #b, false)
#define REQUIRE_NE(a, b) ::brass::test::check_not_equal((a), (b), __FILE__, __LINE__, #a, #b, true)

#define BRASS_TEST_MAIN() \
    int main(int argc, char** argv) { \
        return ::brass::test::TestRegistry::instance().run_all(argc, argv); \
    }
