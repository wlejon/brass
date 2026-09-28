#pragma once

#include <brass/vm/bytecode.hpp>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <string_view>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace brass {
class Function;
class Module;
}

namespace brass::runtime {

// Tier-0 bytecode of one program, built ahead of its first call.
//
// A program's functions reach Tier 0 one first call at a time: the body is
// built (Module::materialize) and compiled to bytecode on the calling thread,
// which for a large program's startup is a third of the time it takes to get
// going. The warmer takes that work off the calling thread when the order in
// which the functions will first run is known in advance — in practice, the
// order they first ran in last time (the host keeps it; record() and
// first_use_log() are how it learns it).
//
// warm() hands the list to a few threads of the warmer's own (not the compile
// pool, whose tier-up and OSR compiles must not queue behind it), which build
// each function's bytecode in list order. An interpreter asks take() before it
// compiles a function itself: built, it gets the shared bytecode; being built
// on a warmer thread, it waits for that rather than repeating it; neither, the
// function is claimed so that the warmer skips it, and the interpreter compiles
// it as it always did. A list naming functions the program no longer has (the
// host's list is from another build) only wastes the warmer's time.
//
// Every interpreter of the program reads the one BytecodeFunction a warmer
// built: bytecode is immutable once compiled, and an interpreter keeps its
// per-function state (FastFnInfo) beside it.
class BytecodeWarmer {
public:
    BytecodeWarmer();
    ~BytecodeWarmer();

    BytecodeWarmer(const BytecodeWarmer&) = delete;
    BytecodeWarmer& operator=(const BytecodeWarmer&) = delete;

    // Starts building `names`' bytecode, in order, on `threads` threads
    // (clamped to [1, 8]), against `mod`. Once per warmer; later calls do
    // nothing.
    void warm(const Module& mod, std::vector<std::string> names, unsigned threads);

    // The prebuilt bytecode of `fn`, or null (then the caller compiles `fn`
    // and the warmer will not). Waits while a warmer thread is building it.
    std::shared_ptr<const BytecodeFunction> take(const Function& fn);

    // Whether take() can return anything: warm() ran and was not stopped.
    bool active() const noexcept { return active_.load(std::memory_order_acquire); }

    // Recording: every function an interpreter of the program compiled or
    // took, in first-use order, for `window` after the first one (and at
    // most `max_names`).
    void start_recording(std::chrono::milliseconds window, size_t max_names);
    void record(const Function& fn);
    bool recording() const noexcept { return recording_.load(std::memory_order_acquire); }
    std::vector<std::string> first_use_log() const;

    // Stops the warmer threads (a function being built is finished) and
    // drops the prebuilt bytecode. For the program's module going away: the
    // cache is keyed by Function address.
    void stop() noexcept;
    // stop() when the warmer was warming `mod`.
    void forget(const Module* mod) noexcept;

    // Statistics.
    uint64_t built() const noexcept { return built_.load(std::memory_order_relaxed); }
    uint64_t hits() const noexcept { return hits_.load(std::memory_order_relaxed); }

private:
    enum class State : uint8_t { Claimed, Building, Ready };
    struct Slot {
        State state = State::Claimed;
        std::shared_ptr<const BytecodeFunction> code;
    };
    void run();

    mutable std::mutex mutex_;
    std::condition_variable built_cv_;
    const Module* module_ = nullptr;           // under mutex_
    std::vector<std::string> names_;           // fixed once warm() ran
    size_t next_ = 0;                          // under mutex_
    bool stopping_ = false;                    // under mutex_
    std::unordered_map<const Function*, Slot> slots_; // under mutex_
    std::vector<std::thread> threads_;
    std::atomic<bool> active_{false};
    std::atomic<uint64_t> built_{0};
    std::atomic<uint64_t> hits_{0};

    mutable std::mutex log_mutex_;
    std::atomic<bool> recording_{false};
    std::chrono::steady_clock::time_point record_start_{};
    std::chrono::milliseconds record_window_{0};
    size_t record_max_ = 0;
    bool record_started_ = false;                    // under log_mutex_
    std::vector<std::string> log_;                   // under log_mutex_
    std::unordered_set<const Function*> logged_;     // under log_mutex_
};

} // namespace brass::runtime
