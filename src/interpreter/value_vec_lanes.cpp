// The interpreter's lane compares, selects, shifts, conversions, rounding and
// vany (opcodes.hpp), computed lane by lane over the vectors' bytes.
#include <brass/interpreter/value.hpp>
#include <cmath>
#include <cstring>
#include <limits>

namespace brass {

namespace {

struct Lanes {
    alignas(32) uint8_t bytes[32] = {};
    template <typename T> T get(uint32_t i) const { T v; std::memcpy(&v, bytes + i * sizeof(T), sizeof(T)); return v; }
    template <typename T> void set(uint32_t i, T v) { std::memcpy(bytes + i * sizeof(T), &v, sizeof(T)); }
};

Lanes lanes_of(const RuntimeValue& v) {
    Lanes l;
    std::memcpy(l.bytes, v.vec_bytes(), v.type().size_in_bytes());
    return l;
}

RuntimeValue make_vec(Type t, const Lanes& l) {
    return t.is_v256() ? RuntimeValue::from_v256(t, l.bytes) : RuntimeValue::from_v128(t, l.bytes);
}

template <typename T>
bool compare(Opcode op, T a, T b) {
    switch (op) {
        case Opcode::vcmp_eq: return a == b;
        case Opcode::vcmp_ne: return !(a == b);
        case Opcode::vcmp_lt: return a < b;
        case Opcode::vcmp_le: return a <= b;
        case Opcode::vcmp_gt: return a > b;
        default:              return a >= b;
    }
}

// A float lane to an integer lane as cvtt does: truncated, the minimum when
// out of range or NaN.
template <typename I, typename F>
I truncate_to(F f) {
    if (!(f >= static_cast<F>(std::numeric_limits<I>::min()) &&
          f < -static_cast<F>(std::numeric_limits<I>::min())))
        return std::numeric_limits<I>::min();
    return static_cast<I>(f);
}

template <typename F>
F round_lane(Opcode op, F f) {
    switch (op) {
        case Opcode::vfloor: return std::floor(f);
        case Opcode::vceil:  return std::ceil(f);
        case Opcode::vtrunc: return std::trunc(f);
        default:             return std::nearbyint(f);   // the default mode: ties to even
    }
}

} // namespace

RuntimeValue val_vlanes(Opcode op, Type result_type, const RuntimeValue* ops) {
    const Type t0 = ops[0].type();
    const Type e0 = t0.element_type();
    const uint32_t n = t0.vector_lanes();
    const Lanes a = lanes_of(ops[0]);
    Lanes out;

    switch (op) {
        case Opcode::vcmp_eq: case Opcode::vcmp_ne: case Opcode::vcmp_lt:
        case Opcode::vcmp_le: case Opcode::vcmp_gt: case Opcode::vcmp_ge: {
            const Lanes b = lanes_of(ops[1]);
            for (uint32_t i = 0; i < n; ++i) {
                bool r;
                if (e0 == Type::f32()) r = compare(op, a.get<float>(i), b.get<float>(i));
                else if (e0 == Type::f64()) r = compare(op, a.get<double>(i), b.get<double>(i));
                else if (e0 == Type::i32()) r = compare(op, a.get<int32_t>(i), b.get<int32_t>(i));
                else r = compare(op, a.get<int64_t>(i), b.get<int64_t>(i));
                if (e0.size_in_bytes() == 4) out.set<int32_t>(i, r ? -1 : 0);
                else out.set<int64_t>(i, r ? -1 : 0);
            }
            return make_vec(result_type, out);
        }

        case Opcode::vselect: {
            const Lanes x = lanes_of(ops[1]);
            const Lanes y = lanes_of(ops[2]);
            const uint32_t w = static_cast<uint32_t>(e0.size_in_bytes());
            const uint32_t vn = result_type.vector_lanes();
            for (uint32_t i = 0; i < vn; ++i) {
                const bool take = w == 4 ? a.get<int32_t>(i) < 0 : a.get<int64_t>(i) < 0;
                std::memcpy(out.bytes + i * w, (take ? x : y).bytes + i * w, w);
            }
            return make_vec(result_type, out);
        }

        case Opcode::vshl: case Opcode::vlshr: case Opcode::vashr: {
            const Lanes b = lanes_of(ops[1]);
            for (uint32_t i = 0; i < n; ++i) {
                if (e0 == Type::i32()) {
                    const uint32_t v = a.get<uint32_t>(i), s = b.get<uint32_t>(i);
                    uint32_t r;
                    if (op == Opcode::vshl) r = s >= 32 ? 0 : v << s;
                    else if (op == Opcode::vlshr) r = s >= 32 ? 0 : v >> s;
                    else r = static_cast<uint32_t>(static_cast<int32_t>(v) >> (s >= 32 ? 31 : s));
                    out.set<uint32_t>(i, r);
                } else {
                    const uint64_t v = a.get<uint64_t>(i), s = b.get<uint64_t>(i);
                    uint64_t r;
                    if (op == Opcode::vshl) r = s >= 64 ? 0 : v << s;
                    else if (op == Opcode::vlshr) r = s >= 64 ? 0 : v >> s;
                    else r = static_cast<uint64_t>(static_cast<int64_t>(v) >> (s >= 64 ? 63 : s));
                    out.set<uint64_t>(i, r);
                }
            }
            return make_vec(result_type, out);
        }

        case Opcode::vconvert: {
            const Type eo = result_type.element_type();
            for (uint32_t i = 0; i < n; ++i) {
                if (e0 == Type::i32() && eo == Type::f32()) out.set<float>(i, static_cast<float>(a.get<int32_t>(i)));
                else if (e0 == Type::f32() && eo == Type::i32()) out.set<int32_t>(i, truncate_to<int32_t>(a.get<float>(i)));
                else if (e0 == Type::i32() && eo == Type::f64()) out.set<double>(i, static_cast<double>(a.get<int32_t>(i)));
                else if (e0 == Type::f64() && eo == Type::i32()) out.set<int32_t>(i, truncate_to<int32_t>(a.get<double>(i)));
                else if (e0 == Type::f32() && eo == Type::f64()) out.set<double>(i, static_cast<double>(a.get<float>(i)));
                else if (e0 == Type::f64() && eo == Type::f32()) out.set<float>(i, static_cast<float>(a.get<double>(i)));
                else if (e0 == Type::i32() && eo == Type::i64()) out.set<int64_t>(i, a.get<int32_t>(i));
                else out.set<int32_t>(i, static_cast<int32_t>(a.get<int64_t>(i)));
            }
            return make_vec(result_type, out);
        }

        case Opcode::vbitcast:
            return make_vec(result_type, a);

        case Opcode::vfloor: case Opcode::vceil: case Opcode::vtrunc: case Opcode::vround:
            for (uint32_t i = 0; i < n; ++i) {
                if (e0 == Type::f32()) out.set<float>(i, round_lane(op, a.get<float>(i)));
                else out.set<double>(i, round_lane(op, a.get<double>(i)));
            }
            return make_vec(result_type, out);

        default: {   // vany
            for (uint32_t i = 0; i < n; ++i) {
                const bool neg = e0.size_in_bytes() == 4 ? a.get<int32_t>(i) < 0 : a.get<int64_t>(i) < 0;
                if (neg) return RuntimeValue::from_i32(1);
            }
            return RuntimeValue::from_i32(0);
        }
    }
}

} // namespace brass
