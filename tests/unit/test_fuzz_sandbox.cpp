#include "test_framework.hpp"
#include <brass/fuzz/diff_fuzzer.hpp>
#include <brass/mir/builder.hpp>
#include <brass/mir/verifier.hpp>
#include <thread>
#include <chrono>

using namespace brass;
using namespace brass::fuzz;

TEST_CASE("DiffFuzzer_WatchdogTimeout") {
    DiffFuzzerOptions opts;
    opts.timeout_ms = 100;
    DiffFuzzer fuzzer(opts);

    bool ok = DiffFuzzer::run_with_watchdog([]() {
        std::this_thread::sleep_for(std::chrono::milliseconds(500));
    }, 100);

    CHECK_FALSE(ok);
}

TEST_CASE("DiffFuzzer_WatchdogSuccess") {
    bool ok = DiffFuzzer::run_with_watchdog([]() {
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }, 500);

    CHECK(ok);
}

TEST_CASE("DiffFuzzer_ProtectedCppException") {
    std::string fault;
    bool ok = DiffFuzzer::run_protected([]() {
        throw std::runtime_error("Simulated test error");
    }, fault);

    CHECK_FALSE(ok);
    CHECK(fault.find("Simulated test error") != std::string::npos);
}

TEST_CASE("DiffFuzzer_ProtectedFaultDivideByZero") {
    std::string fault;
    bool ok = DiffFuzzer::run_protected([]() {
        volatile int a = 42;
        volatile int b = 0;
        volatile int c = a / b;
        (void)c;
    }, fault);

    CHECK_FALSE(ok);
    CHECK(fault.find("Divide by Zero") != std::string::npos || fault.find("fault") != std::string::npos);
}

TEST_CASE("DiffFuzzer_JobObjectSandboxExecution") {
    std::string log;
#if defined(_WIN32)
    const char* cmd = "cmd.exe /c exit 0";
#else
    const char* cmd = "exit 0";
#endif
    ExecutionStatus status = DiffFuzzer::run_sandboxed_command(cmd, 1000, 64 * 1024 * 1024, log);
    CHECK(status == ExecutionStatus::Success);
}

TEST_CASE("DiffFuzzer_JobObjectSandboxTimeout") {
    std::string log;
#if defined(_WIN32)
    const char* cmd = "cmd.exe /c ping 127.0.0.1 -n 4 > nul";
#else
    const char* cmd = "sleep 3";
#endif
    ExecutionStatus status = DiffFuzzer::run_sandboxed_command(cmd, 150, 64 * 1024 * 1024, log);
    CHECK(status == ExecutionStatus::Timeout);
}
