// The bytecode warmer (bytecode_warmer.hpp): a program's Tier-0 bytecode
// built on threads of its own, in the order the functions are expected to
// first run.

#include <brass/runtime/bytecode_warmer.hpp>
#include <brass/mir/function.hpp>
#include <brass/mir/module.hpp>
#include <brass/vm/bytecode_compiler.hpp>

#include <algorithm>
#include <exception>

namespace brass::runtime {

BytecodeWarmer::BytecodeWarmer() = default;

BytecodeWarmer::~BytecodeWarmer() { stop(); }

void BytecodeWarmer::warm(const Module& mod, std::vector<std::string> names, unsigned threads) {
    {
        std::lock_guard<std::mutex> lock(mutex_);
        if (module_ || stopping_ || names.empty()) return;
        module_ = &mod;
        names_ = std::move(names);
        next_ = 0;
    }
    active_.store(true, std::memory_order_release);
    threads = std::clamp(threads, 1u, 8u);
    threads_.reserve(threads);
    for (unsigned i = 0; i < threads; ++i) threads_.emplace_back([this] { run(); });
}

void BytecodeWarmer::run() {
    for (;;) {
        const Function* fn = nullptr;
        {
            std::lock_guard<std::mutex> lock(mutex_);
            // A name the program lacks, or a function an interpreter already
            // claimed, is skipped.
            while (!stopping_ && next_ < names_.size()) {
                Function* f = module_->get_function(names_[next_++]);
                if (!f || !f->has_body()) continue;
                auto [it, fresh] = slots_.try_emplace(f);
                if (!fresh) continue;
                it->second.state = State::Building;
                fn = f;
                break;
            }
            if (!fn) return;
        }
        std::shared_ptr<const BytecodeFunction> code;
        try {
            code = BytecodeCompiler().compile(*fn);
        } catch (const std::exception&) {
            // Left to the interpreter, which reports what it hits.
        }
        {
            std::lock_guard<std::mutex> lock(mutex_);
            auto it = slots_.find(fn);
            if (it != slots_.end()) {
                it->second.state = code ? State::Ready : State::Claimed;
                it->second.code = std::move(code);
            }
        }
        built_.fetch_add(1, std::memory_order_relaxed);
        built_cv_.notify_all();
    }
}

std::shared_ptr<const BytecodeFunction> BytecodeWarmer::take(const Function& fn) {
    std::unique_lock<std::mutex> lock(mutex_);
    if (fn.parent() != module_) return nullptr;
    if (slots_.try_emplace(&fn).second) return nullptr;  // claimed: the caller compiles it
    // Looked up again after every wait: stop() may have cleared the slots.
    const Slot* slot = nullptr;
    built_cv_.wait(lock, [&] {
        auto it = slots_.find(&fn);
        slot = it != slots_.end() ? &it->second : nullptr;
        return !slot || slot->state != State::Building;
    });
    if (!slot || slot->state != State::Ready) return nullptr;
    hits_.fetch_add(1, std::memory_order_relaxed);
    return slot->code;
}

void BytecodeWarmer::start_recording(std::chrono::milliseconds window, size_t max_names) {
    std::lock_guard<std::mutex> lock(log_mutex_);
    record_window_ = window;
    record_max_ = max_names;
    record_started_ = false;
    log_.clear();
    logged_.clear();
    recording_.store(max_names > 0, std::memory_order_release);
}

void BytecodeWarmer::record(const Function& fn) {
    std::lock_guard<std::mutex> lock(log_mutex_);
    if (!recording_.load(std::memory_order_relaxed)) return;
    const auto now = std::chrono::steady_clock::now();
    if (!record_started_) {
        record_started_ = true;
        record_start_ = now;
    } else if (now - record_start_ > record_window_) {
        recording_.store(false, std::memory_order_release);
        return;
    }
    if (!logged_.insert(&fn).second) return;
    log_.emplace_back(fn.name());
    if (log_.size() >= record_max_) recording_.store(false, std::memory_order_release);
}

std::vector<std::string> BytecodeWarmer::first_use_log() const {
    std::lock_guard<std::mutex> lock(log_mutex_);
    return log_;
}

void BytecodeWarmer::stop() noexcept {
    {
        std::lock_guard<std::mutex> lock(mutex_);
        stopping_ = true;
    }
    active_.store(false, std::memory_order_release);
    for (std::thread& t : threads_) {
        if (t.joinable()) t.join();
    }
    threads_.clear();
    std::lock_guard<std::mutex> lock(mutex_);
    slots_.clear();
    module_ = nullptr;
    built_cv_.notify_all();
}

void BytecodeWarmer::forget(const Module* mod) noexcept {
    {
        std::lock_guard<std::mutex> lock(mutex_);
        if (module_ != mod) return;
    }
    stop();
    std::lock_guard<std::mutex> lock(log_mutex_);
    logged_.clear();
}

} // namespace brass::runtime
