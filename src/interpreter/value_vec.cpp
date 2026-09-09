#include <brass/interpreter/value.hpp>
#include <cmath>
#include <algorithm>
#include <cstring>
#include <stdexcept>

namespace brass {

RuntimeValue val_vadd(RuntimeValue lhs, RuntimeValue rhs) {
    if (lhs.is_f32x4()) {
        return RuntimeValue::from_f32x4(
            lhs.f32_lane(0) + rhs.f32_lane(0),
            lhs.f32_lane(1) + rhs.f32_lane(1),
            lhs.f32_lane(2) + rhs.f32_lane(2),
            lhs.f32_lane(3) + rhs.f32_lane(3)
        );
    }
    if (lhs.is_f64x2()) {
        return RuntimeValue::from_f64x2(
            lhs.f64_lane(0) + rhs.f64_lane(0),
            lhs.f64_lane(1) + rhs.f64_lane(1)
        );
    }
    if (lhs.is_i32x4()) {
        return RuntimeValue::from_i32x4(
            static_cast<int32_t>(static_cast<uint32_t>(lhs.i32_lane(0)) + static_cast<uint32_t>(rhs.i32_lane(0))),
            static_cast<int32_t>(static_cast<uint32_t>(lhs.i32_lane(1)) + static_cast<uint32_t>(rhs.i32_lane(1))),
            static_cast<int32_t>(static_cast<uint32_t>(lhs.i32_lane(2)) + static_cast<uint32_t>(rhs.i32_lane(2))),
            static_cast<int32_t>(static_cast<uint32_t>(lhs.i32_lane(3)) + static_cast<uint32_t>(rhs.i32_lane(3)))
        );
    }
    if (lhs.is_i64x2()) {
        return RuntimeValue::from_i64x2(
            static_cast<int64_t>(static_cast<uint64_t>(lhs.i64_lane(0)) + static_cast<uint64_t>(rhs.i64_lane(0))),
            static_cast<int64_t>(static_cast<uint64_t>(lhs.i64_lane(1)) + static_cast<uint64_t>(rhs.i64_lane(1)))
        );
    }
    return lhs;
}

RuntimeValue val_vsub(RuntimeValue lhs, RuntimeValue rhs) {
    if (lhs.is_f32x4()) {
        return RuntimeValue::from_f32x4(
            lhs.f32_lane(0) - rhs.f32_lane(0),
            lhs.f32_lane(1) - rhs.f32_lane(1),
            lhs.f32_lane(2) - rhs.f32_lane(2),
            lhs.f32_lane(3) - rhs.f32_lane(3)
        );
    }
    if (lhs.is_f64x2()) {
        return RuntimeValue::from_f64x2(
            lhs.f64_lane(0) - rhs.f64_lane(0),
            lhs.f64_lane(1) - rhs.f64_lane(1)
        );
    }
    if (lhs.is_i32x4()) {
        return RuntimeValue::from_i32x4(
            static_cast<int32_t>(static_cast<uint32_t>(lhs.i32_lane(0)) - static_cast<uint32_t>(rhs.i32_lane(0))),
            static_cast<int32_t>(static_cast<uint32_t>(lhs.i32_lane(1)) - static_cast<uint32_t>(rhs.i32_lane(1))),
            static_cast<int32_t>(static_cast<uint32_t>(lhs.i32_lane(2)) - static_cast<uint32_t>(rhs.i32_lane(2))),
            static_cast<int32_t>(static_cast<uint32_t>(lhs.i32_lane(3)) - static_cast<uint32_t>(rhs.i32_lane(3)))
        );
    }
    if (lhs.is_i64x2()) {
        return RuntimeValue::from_i64x2(
            static_cast<int64_t>(static_cast<uint64_t>(lhs.i64_lane(0)) - static_cast<uint64_t>(rhs.i64_lane(0))),
            static_cast<int64_t>(static_cast<uint64_t>(lhs.i64_lane(1)) - static_cast<uint64_t>(rhs.i64_lane(1)))
        );
    }
    return lhs;
}

RuntimeValue val_vmul(RuntimeValue lhs, RuntimeValue rhs) {
    if (lhs.is_f32x4()) {
        return RuntimeValue::from_f32x4(
            lhs.f32_lane(0) * rhs.f32_lane(0),
            lhs.f32_lane(1) * rhs.f32_lane(1),
            lhs.f32_lane(2) * rhs.f32_lane(2),
            lhs.f32_lane(3) * rhs.f32_lane(3)
        );
    }
    if (lhs.is_f64x2()) {
        return RuntimeValue::from_f64x2(
            lhs.f64_lane(0) * rhs.f64_lane(0),
            lhs.f64_lane(1) * rhs.f64_lane(1)
        );
    }
    if (lhs.is_i32x4()) {
        return RuntimeValue::from_i32x4(
            static_cast<int32_t>(static_cast<uint32_t>(lhs.i32_lane(0)) * static_cast<uint32_t>(rhs.i32_lane(0))),
            static_cast<int32_t>(static_cast<uint32_t>(lhs.i32_lane(1)) * static_cast<uint32_t>(rhs.i32_lane(1))),
            static_cast<int32_t>(static_cast<uint32_t>(lhs.i32_lane(2)) * static_cast<uint32_t>(rhs.i32_lane(2))),
            static_cast<int32_t>(static_cast<uint32_t>(lhs.i32_lane(3)) * static_cast<uint32_t>(rhs.i32_lane(3)))
        );
    }
    if (lhs.is_i64x2()) {
        return RuntimeValue::from_i64x2(
            static_cast<int64_t>(static_cast<uint64_t>(lhs.i64_lane(0)) * static_cast<uint64_t>(rhs.i64_lane(0))),
            static_cast<int64_t>(static_cast<uint64_t>(lhs.i64_lane(1)) * static_cast<uint64_t>(rhs.i64_lane(1)))
        );
    }
    return lhs;
}

RuntimeValue val_vdiv(RuntimeValue lhs, RuntimeValue rhs) {
    if (lhs.is_f32x4()) {
        return RuntimeValue::from_f32x4(
            lhs.f32_lane(0) / rhs.f32_lane(0),
            lhs.f32_lane(1) / rhs.f32_lane(1),
            lhs.f32_lane(2) / rhs.f32_lane(2),
            lhs.f32_lane(3) / rhs.f32_lane(3)
        );
    }
    if (lhs.is_f64x2()) {
        return RuntimeValue::from_f64x2(
            lhs.f64_lane(0) / rhs.f64_lane(0),
            lhs.f64_lane(1) / rhs.f64_lane(1)
        );
    }
    if (lhs.is_i32x4()) {
        auto sdiv32 = [](int32_t a, int32_t b) -> int32_t {
            if (b == 0) return 0;
            if (a == std::numeric_limits<int32_t>::min() && b == -1) return a;
            return a / b;
        };
        return RuntimeValue::from_i32x4(
            sdiv32(lhs.i32_lane(0), rhs.i32_lane(0)),
            sdiv32(lhs.i32_lane(1), rhs.i32_lane(1)),
            sdiv32(lhs.i32_lane(2), rhs.i32_lane(2)),
            sdiv32(lhs.i32_lane(3), rhs.i32_lane(3))
        );
    }
    if (lhs.is_i64x2()) {
        auto sdiv64 = [](int64_t a, int64_t b) -> int64_t {
            if (b == 0) return 0;
            if (a == std::numeric_limits<int64_t>::min() && b == -1) return a;
            return a / b;
        };
        return RuntimeValue::from_i64x2(
            sdiv64(lhs.i64_lane(0), rhs.i64_lane(0)),
            sdiv64(lhs.i64_lane(1), rhs.i64_lane(1))
        );
    }
    return lhs;
}

RuntimeValue val_vneg(RuntimeValue val) {
    if (val.is_f32x4()) {
        return RuntimeValue::from_f32x4(
            -val.f32_lane(0),
            -val.f32_lane(1),
            -val.f32_lane(2),
            -val.f32_lane(3)
        );
    }
    if (val.is_f64x2()) {
        return RuntimeValue::from_f64x2(
            -val.f64_lane(0),
            -val.f64_lane(1)
        );
    }
    if (val.is_i32x4()) {
        return RuntimeValue::from_i32x4(
            static_cast<int32_t>(-static_cast<uint32_t>(val.i32_lane(0))),
            static_cast<int32_t>(-static_cast<uint32_t>(val.i32_lane(1))),
            static_cast<int32_t>(-static_cast<uint32_t>(val.i32_lane(2))),
            static_cast<int32_t>(-static_cast<uint32_t>(val.i32_lane(3)))
        );
    }
    if (val.is_i64x2()) {
        return RuntimeValue::from_i64x2(
            static_cast<int64_t>(-static_cast<uint64_t>(val.i64_lane(0))),
            static_cast<int64_t>(-static_cast<uint64_t>(val.i64_lane(1)))
        );
    }
    return val;
}

RuntimeValue val_vmin(RuntimeValue lhs, RuntimeValue rhs) {
    if (lhs.is_f32x4()) {
        auto min_f = [](float a, float b) { return (b < a) ? b : a; };
        return RuntimeValue::from_f32x4(
            min_f(lhs.f32_lane(0), rhs.f32_lane(0)),
            min_f(lhs.f32_lane(1), rhs.f32_lane(1)),
            min_f(lhs.f32_lane(2), rhs.f32_lane(2)),
            min_f(lhs.f32_lane(3), rhs.f32_lane(3))
        );
    }
    if (lhs.is_f64x2()) {
        auto min_d = [](double a, double b) { return (b < a) ? b : a; };
        return RuntimeValue::from_f64x2(
            min_d(lhs.f64_lane(0), rhs.f64_lane(0)),
            min_d(lhs.f64_lane(1), rhs.f64_lane(1))
        );
    }
    if (lhs.is_i32x4()) {
        return RuntimeValue::from_i32x4(
            std::min(lhs.i32_lane(0), rhs.i32_lane(0)),
            std::min(lhs.i32_lane(1), rhs.i32_lane(1)),
            std::min(lhs.i32_lane(2), rhs.i32_lane(2)),
            std::min(lhs.i32_lane(3), rhs.i32_lane(3))
        );
    }
    if (lhs.is_i64x2()) {
        return RuntimeValue::from_i64x2(
            std::min(lhs.i64_lane(0), rhs.i64_lane(0)),
            std::min(lhs.i64_lane(1), rhs.i64_lane(1))
        );
    }
    return lhs;
}

RuntimeValue val_vmax(RuntimeValue lhs, RuntimeValue rhs) {
    if (lhs.is_f32x4()) {
        auto max_f = [](float a, float b) { return (b > a) ? b : a; };
        return RuntimeValue::from_f32x4(
            max_f(lhs.f32_lane(0), rhs.f32_lane(0)),
            max_f(lhs.f32_lane(1), rhs.f32_lane(1)),
            max_f(lhs.f32_lane(2), rhs.f32_lane(2)),
            max_f(lhs.f32_lane(3), rhs.f32_lane(3))
        );
    }
    if (lhs.is_f64x2()) {
        auto max_d = [](double a, double b) { return (b > a) ? b : a; };
        return RuntimeValue::from_f64x2(
            max_d(lhs.f64_lane(0), rhs.f64_lane(0)),
            max_d(lhs.f64_lane(1), rhs.f64_lane(1))
        );
    }
    if (lhs.is_i32x4()) {
        return RuntimeValue::from_i32x4(
            std::max(lhs.i32_lane(0), rhs.i32_lane(0)),
            std::max(lhs.i32_lane(1), rhs.i32_lane(1)),
            std::max(lhs.i32_lane(2), rhs.i32_lane(2)),
            std::max(lhs.i32_lane(3), rhs.i32_lane(3))
        );
    }
    if (lhs.is_i64x2()) {
        return RuntimeValue::from_i64x2(
            std::max(lhs.i64_lane(0), rhs.i64_lane(0)),
            std::max(lhs.i64_lane(1), rhs.i64_lane(1))
        );
    }
    return lhs;
}

RuntimeValue val_vsqrt(RuntimeValue val) {
    if (val.is_f32x4()) {
        return RuntimeValue::from_f32x4(
            std::sqrt(val.f32_lane(0)),
            std::sqrt(val.f32_lane(1)),
            std::sqrt(val.f32_lane(2)),
            std::sqrt(val.f32_lane(3))
        );
    }
    if (val.is_f64x2()) {
        return RuntimeValue::from_f64x2(
            std::sqrt(val.f64_lane(0)),
            std::sqrt(val.f64_lane(1))
        );
    }
    return val;
}

RuntimeValue val_vand(RuntimeValue lhs, RuntimeValue rhs) {
    uint64_t w0[2], w1[2], out[2];
    std::memcpy(w0, lhs.v128_bytes(), 16);
    std::memcpy(w1, rhs.v128_bytes(), 16);
    out[0] = w0[0] & w1[0];
    out[1] = w0[1] & w1[1];
    return RuntimeValue::from_v128(lhs.type(), out);
}

RuntimeValue val_vor(RuntimeValue lhs, RuntimeValue rhs) {
    uint64_t w0[2], w1[2], out[2];
    std::memcpy(w0, lhs.v128_bytes(), 16);
    std::memcpy(w1, rhs.v128_bytes(), 16);
    out[0] = w0[0] | w1[0];
    out[1] = w0[1] | w1[1];
    return RuntimeValue::from_v128(lhs.type(), out);
}

RuntimeValue val_vxor(RuntimeValue lhs, RuntimeValue rhs) {
    uint64_t w0[2], w1[2], out[2];
    std::memcpy(w0, lhs.v128_bytes(), 16);
    std::memcpy(w1, rhs.v128_bytes(), 16);
    out[0] = w0[0] ^ w1[0];
    out[1] = w0[1] ^ w1[1];
    return RuntimeValue::from_v128(lhs.type(), out);
}

RuntimeValue val_vnot(RuntimeValue val) {
    uint64_t w[2], out[2];
    std::memcpy(w, val.v128_bytes(), 16);
    out[0] = ~w[0];
    out[1] = ~w[1];
    return RuntimeValue::from_v128(val.type(), out);
}

RuntimeValue val_vbroadcast(Type vec_type, RuntimeValue val) {
    switch (vec_type.kind()) {
        case TypeKind::F32x4: {
            float f = val.is_f32() ? val.as_f32() : static_cast<float>(val.as_f64());
            return RuntimeValue::from_f32x4(f, f, f, f);
        }
        case TypeKind::F64x2: {
            double d = val.as_f64();
            return RuntimeValue::from_f64x2(d, d);
        }
        case TypeKind::I32x4: {
            int32_t i = val.as_i32();
            return RuntimeValue::from_i32x4(i, i, i, i);
        }
        case TypeKind::I64x2: {
            int64_t l = val.as_i64();
            return RuntimeValue::from_i64x2(l, l);
        }
        default:
            return RuntimeValue();
    }
}

RuntimeValue val_vextract_lane(RuntimeValue val, uint32_t lane) {
    if (val.is_f32x4()) {
        return RuntimeValue::from_f32(val.f32_lane(lane));
    }
    if (val.is_f64x2()) {
        return RuntimeValue::from_f64(val.f64_lane(lane));
    }
    if (val.is_i32x4()) {
        return RuntimeValue::from_i32(val.i32_lane(lane));
    }
    if (val.is_i64x2()) {
        return RuntimeValue::from_i64(val.i64_lane(lane));
    }
    return RuntimeValue();
}

RuntimeValue val_vinsert_lane(RuntimeValue vec, RuntimeValue scalar, uint32_t lane) {
    uint8_t out[16];
    std::memcpy(out, vec.v128_bytes(), 16);
    if (vec.is_f32x4()) {
        float f = scalar.is_f32() ? scalar.as_f32() : static_cast<float>(scalar.as_f64());
        if (lane < 4) std::memcpy(out + lane * sizeof(float), &f, sizeof(float));
        return RuntimeValue::from_v128(Type::f32x4(), out);
    }
    if (vec.is_f64x2()) {
        double d = scalar.as_f64();
        if (lane < 2) std::memcpy(out + lane * sizeof(double), &d, sizeof(double));
        return RuntimeValue::from_v128(Type::f64x2(), out);
    }
    if (vec.is_i32x4()) {
        int32_t i = scalar.as_i32();
        if (lane < 4) std::memcpy(out + lane * sizeof(int32_t), &i, sizeof(int32_t));
        return RuntimeValue::from_v128(Type::i32x4(), out);
    }
    if (vec.is_i64x2()) {
        int64_t l = scalar.as_i64();
        if (lane < 2) std::memcpy(out + lane * sizeof(int64_t), &l, sizeof(int64_t));
        return RuntimeValue::from_v128(Type::i64x2(), out);
    }
    return vec;
}

RuntimeValue val_vshuffle(RuntimeValue v1, RuntimeValue v2, uint32_t mask) {
    if (v1.is_f32x4()) {
        float res[4];
        res[0] = v1.f32_lane(mask & 3);
        res[1] = v1.f32_lane((mask >> 2) & 3);
        res[2] = v2.f32_lane((mask >> 4) & 3);
        res[3] = v2.f32_lane((mask >> 6) & 3);
        return RuntimeValue::from_f32x4(res[0], res[1], res[2], res[3]);
    }
    if (v1.is_i32x4()) {
        int32_t res[4];
        res[0] = v1.i32_lane(mask & 3);
        res[1] = v1.i32_lane((mask >> 2) & 3);
        res[2] = v2.i32_lane((mask >> 4) & 3);
        res[3] = v2.i32_lane((mask >> 6) & 3);
        return RuntimeValue::from_i32x4(res[0], res[1], res[2], res[3]);
    }
    if (v1.is_f64x2()) {
        double res[2];
        res[0] = (mask & 1) ? v1.f64_lane(1) : v1.f64_lane(0);
        res[1] = ((mask >> 1) & 1) ? v2.f64_lane(1) : v2.f64_lane(0);
        return RuntimeValue::from_f64x2(res[0], res[1]);
    }
    if (v1.is_i64x2()) {
        int64_t res[2];
        res[0] = (mask & 1) ? v1.i64_lane(1) : v1.i64_lane(0);
        res[1] = ((mask >> 1) & 1) ? v2.i64_lane(1) : v2.i64_lane(0);
        return RuntimeValue::from_i64x2(res[0], res[1]);
    }
    return v1;
}

RuntimeValue val_vzero(Type vec_type) {
    alignas(16) uint8_t zeros[16] = {0};
    return RuntimeValue::from_v128(vec_type, zeros);
}

} // namespace brass
