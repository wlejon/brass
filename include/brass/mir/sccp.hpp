#pragma once

#include <brass/mir/types.hpp>
#include <brass/mir/function.hpp>
#include <brass/mir/module.hpp>
#include <cstddef>
#include <cstdint>
#include <cstring>

namespace brass {

enum class LatticeState : uint8_t {
    Top,
    Constant,
    Bottom
};

class LatticeValue {
public:
    constexpr LatticeValue() noexcept : state_(LatticeState::Top), type_(Type::void_type()) {
        data_.i64_val = 0;
    }

    static LatticeValue make_top() noexcept {
        return LatticeValue();
    }

    static LatticeValue make_bottom() noexcept {
        LatticeValue v;
        v.state_ = LatticeState::Bottom;
        v.type_ = Type::void_type();
        return v;
    }

    static LatticeValue make_bottom(Type type) noexcept {
        LatticeValue v;
        v.state_ = LatticeState::Bottom;
        v.type_ = type;
        return v;
    }

    static LatticeValue make_i32(int32_t val) noexcept {
        LatticeValue v;
        v.state_ = LatticeState::Constant;
        v.type_ = Type::i32();
        v.data_.i64_val = static_cast<int64_t>(val);
        return v;
    }

    static LatticeValue make_i64(int64_t val) noexcept {
        LatticeValue v;
        v.state_ = LatticeState::Constant;
        v.type_ = Type::i64();
        v.data_.i64_val = val;
        return v;
    }

    static LatticeValue make_f32(float val) noexcept {
        LatticeValue v;
        v.state_ = LatticeState::Constant;
        v.type_ = Type::f32();
        v.data_.f32_val = val;
        return v;
    }

    static LatticeValue make_f64(double val) noexcept {
        LatticeValue v;
        v.state_ = LatticeState::Constant;
        v.type_ = Type::f64();
        v.data_.f64_val = val;
        return v;
    }

    static LatticeValue make_ptr(uint64_t val) noexcept {
        LatticeValue v;
        v.state_ = LatticeState::Constant;
        v.type_ = Type::ptr();
        v.data_.ptr_val = val;
        return v;
    }

    LatticeState state() const noexcept { return state_; }
    Type type() const noexcept { return type_; }

    bool is_top() const noexcept { return state_ == LatticeState::Top; }
    bool is_constant() const noexcept { return state_ == LatticeState::Constant; }
    bool is_bottom() const noexcept { return state_ == LatticeState::Bottom; }

    int32_t as_i32() const noexcept { return static_cast<int32_t>(data_.i64_val); }
    int64_t as_i64() const noexcept { return data_.i64_val; }
    float as_f32() const noexcept { return data_.f32_val; }
    double as_f64() const noexcept { return data_.f64_val; }
    uint64_t as_ptr() const noexcept { return data_.ptr_val; }

    bool is_int_zero() const noexcept {
        if (!is_constant()) return false;
        if (type_ == Type::i32()) return as_i32() == 0;
        if (type_ == Type::i64()) return as_i64() == 0;
        if (type_ == Type::ptr()) return as_ptr() == 0;
        return false;
    }

    bool is_int_one() const noexcept {
        if (!is_constant()) return false;
        if (type_ == Type::i32()) return as_i32() == 1;
        if (type_ == Type::i64()) return as_i64() == 1;
        return false;
    }

    LatticeValue meet(const LatticeValue& other) const noexcept {
        if (is_top()) return other;
        if (other.is_top()) return *this;
        if (is_bottom() || other.is_bottom()) {
            Type t = type_.is_void() ? other.type_ : type_;
            return make_bottom(t);
        }
        if (*this == other) {
            return *this;
        }
        Type t = type_.is_void() ? other.type_ : type_;
        return make_bottom(t);
    }

    bool operator==(const LatticeValue& other) const noexcept {
        if (state_ != other.state_) return false;
        if (state_ == LatticeState::Top || state_ == LatticeState::Bottom) return true;
        if (type_ != other.type_) return false;
        if (type_ == Type::i32()) return as_i32() == other.as_i32();
        if (type_ == Type::i64()) return as_i64() == other.as_i64();
        if (type_ == Type::ptr()) return as_ptr() == other.as_ptr();
        if (type_ == Type::f32()) {
            return std::memcmp(&data_.f32_val, &other.data_.f32_val, sizeof(float)) == 0;
        }
        if (type_ == Type::f64()) {
            return std::memcmp(&data_.f64_val, &other.data_.f64_val, sizeof(double)) == 0;
        }
        return false;
    }

    bool operator!=(const LatticeValue& other) const noexcept {
        return !(*this == other);
    }

private:
    LatticeState state_ = LatticeState::Top;
    Type type_ = Type::void_type();
    union Data {
        int64_t i64_val;
        double f64_val;
        float f32_val;
        uint64_t ptr_val;
        constexpr Data() : i64_val(0) {}
    } data_{};
};

struct SccpStats {
    size_t constants_propagated = 0;
    size_t branches_folded = 0;
    size_t guards_eliminated = 0;
    size_t guards_always_failing = 0;
    size_t instructions_folded = 0;
};

struct SccpOptions {
    bool enable_guard_elim = true;
    bool enable_branch_folding = true;
    SccpStats* stats = nullptr;
};

// Sparse Conditional Constant Propagation & Guard Elimination on a single function
bool sccp_function(Function& fn);
bool sccp_function(Function& fn, const SccpOptions& options);

// SCCP across an entire module
bool sccp_module(Module& mod);
bool sccp_module(Module& mod, const SccpOptions& options);

} // namespace brass
