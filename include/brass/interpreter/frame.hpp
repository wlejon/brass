#pragma once

#include <brass/interpreter/value.hpp>
#include <brass/mir/function.hpp>
#include <brass/mir/instruction.hpp>
#include <vector>
#include <deque>
#include <cstdint>

namespace brass {

class InterpreterFrame {
public:
    explicit InterpreterFrame(const Function* fn = nullptr, InterpreterFrame* caller = nullptr)
        : function_(fn), caller_(caller) {
        values_.resize(64);
    }

    const Function* function() const noexcept { return function_; }
    void set_function(const Function* fn) noexcept { function_ = fn; }

    InterpreterFrame* caller() const noexcept { return caller_; }
    void set_caller(InterpreterFrame* caller) noexcept { caller_ = caller; }

    void set_value(const Value* ssa_val, RuntimeValue val) {
        if (!ssa_val) return;
        uint32_t id = ssa_val->id();
        if (id >= values_.size()) {
            values_.resize((std::max)(static_cast<size_t>(id + 32), values_.size() * 2));
        }
        values_[id] = val;
    }

    RuntimeValue get_value(const Value* ssa_val) const {
        if (!ssa_val) return RuntimeValue::from_void();
        uint32_t id = ssa_val->id();
        if (id < values_.size()) {
            return values_[id];
        }
        return RuntimeValue::from_void();
    }

    // Every gcref- and tagged-typed value the frame holds, and every word of
    // its `alloca.tagged` buffers.
    void collect_roots(std::vector<uintptr_t*>& roots) {
        for (size_t i = 0; i < values_.size(); ++i) {
            if (values_[i].is_gc_root() && !values_[i].is_null()) {
                roots.push_back(reinterpret_cast<uintptr_t*>(&values_[i].raw_bits_ref()));
            }
        }
        for (const auto& [words, count] : tagged_allocas_) {
            for (size_t i = 0; i < count; ++i) {
                if (words[i] != 0) roots.push_back(reinterpret_cast<uintptr_t*>(&words[i]));
            }
        }
    }

    const std::vector<RuntimeValue>& values() const noexcept { return values_; }

    // Zero-filled. A `tagged` buffer's words are roots while the frame lives.
    void* allocate(size_t size, size_t align = 16, bool tagged = false) {
        if (align == 0) align = 16;
        size_t total = size + align;
        alloca_storage_.emplace_back(total, static_cast<uint8_t>(0));
        uintptr_t addr = reinterpret_cast<uintptr_t>(alloca_storage_.back().data());
        uintptr_t aligned_addr = (addr + align - 1) & ~(align - 1);
        if (tagged) tagged_allocas_.push_back({reinterpret_cast<uint64_t*>(aligned_addr), size / 8});
        return reinterpret_cast<void*>(aligned_addr);
    }

private:
    const Function* function_ = nullptr;
    InterpreterFrame* caller_ = nullptr;
    std::vector<RuntimeValue> values_;
    std::deque<std::vector<uint8_t>> alloca_storage_;
    std::vector<std::pair<uint64_t*, size_t>> tagged_allocas_;
};

} // namespace brass
