#include <brass/interpreter/memory_access.hpp>

#include <cstring>
#include <stdexcept>

namespace brass {

RuntimeValue read_memory(const gc::Heap& heap, uintptr_t base, int64_t offset, Type t) {
    if (base == 0) throw std::runtime_error("Memory Error: Null pointer dereference in read_memory");
    const size_t size = t.size_in_bytes();
    heap.check_access(base, offset, size, "read_memory");
    const auto* at = reinterpret_cast<const void*>(static_cast<uintptr_t>(static_cast<int64_t>(base) + offset));
    // i8 / i16 loads read exactly their width, zero-extended into the
    // register form (docs/semantics.md, narrow integers).
    switch (size) {
        case 1: {
            uint8_t v = 0;
            std::memcpy(&v, at, 1);
            return RuntimeValue::from_bits(t, v);
        }
        case 2: {
            uint16_t v = 0;
            std::memcpy(&v, at, 2);
            return RuntimeValue::from_bits(t, v);
        }
        case 4: {
            uint32_t v = 0;
            std::memcpy(&v, at, 4);
            return RuntimeValue::from_bits(t, v);
        }
        case 8: {
            uint64_t v = 0;
            std::memcpy(&v, at, 8);
            return RuntimeValue::from_bits(t, v);
        }
        case 16: {
            uint8_t bytes[16];
            std::memcpy(bytes, at, 16);
            return RuntimeValue::from_v128(t, bytes);
        }
        case 32: {
            uint8_t bytes[32];
            std::memcpy(bytes, at, 32);
            return RuntimeValue::from_v256(t, bytes);
        }
        default:
            throw std::runtime_error("Memory Error: Unsupported access size in read_memory");
    }
}

void write_memory(gc::Heap& heap, uintptr_t base, int64_t offset, Type t, RuntimeValue val) {
    if (base == 0) throw std::runtime_error("Memory Error: Null pointer dereference in write_memory");
    const size_t size = t.size_in_bytes();
    heap.check_access(base, offset, size, "write_memory");
    const uintptr_t address = static_cast<uintptr_t>(static_cast<int64_t>(base) + offset);
    auto* at = reinterpret_cast<void*>(address);
    // Narrow stores write exactly their width: the low bits of the value.
    switch (size) {
        case 1: {
            const auto v = static_cast<uint8_t>(val.raw_bits());
            std::memcpy(at, &v, 1);
            return;
        }
        case 2: {
            const auto v = static_cast<uint16_t>(val.raw_bits());
            std::memcpy(at, &v, 2);
            return;
        }
        case 4: {
            const auto v = static_cast<uint32_t>(val.raw_bits());
            std::memcpy(at, &v, 4);
            return;
        }
        case 8: {
            const uint64_t v = val.raw_bits();
            std::memcpy(at, &v, 8);
            // The interpreters remember every word store that may make an
            // old object name a young one, whether or not the MIR carries a
            // write_barrier for it.
            heap.write_barrier_interior(address, v);
            return;
        }
        case 16:
            std::memcpy(at, val.v128_bytes(), 16);
            return;
        case 32:
            std::memcpy(at, val.vec_bytes(), 32);
            return;
        default:
            throw std::runtime_error("Memory Error: Unsupported access size in write_memory");
    }
}

} // namespace brass
