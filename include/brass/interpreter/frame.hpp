#pragma once

#include <brass/interpreter/value.hpp>
#include <brass/mir/function.hpp>
#include <brass/mir/instruction.hpp>
#include <vector>
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
            values_.resize(std::max(static_cast<size_t>(id + 32), values_.size() * 2));
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

    void collect_roots(std::vector<uintptr_t*>& roots) {
        for (size_t i = 0; i < values_.size(); ++i) {
            if (values_[i].is_gcref() && !values_[i].is_null()) {
                roots.push_back(reinterpret_cast<uintptr_t*>(&values_[i].raw_bits_ref()));
            }
        }
    }

    const std::vector<RuntimeValue>& values() const noexcept { return values_; }

private:
    const Function* function_ = nullptr;
    InterpreterFrame* caller_ = nullptr;
    std::vector<RuntimeValue> values_;
};

} // namespace brass
