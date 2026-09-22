#include "fast_interpreter_impl.hpp"

namespace brass {

void* fast_frame_alloca(FastFrame& frame, size_t size, size_t align) {
    return frame.allocate_alloca(size, align);
}

uint64_t fast_load(FastFrame& frame, uint8_t base_reg, uint8_t offset, size_t size_bytes) {
    uintptr_t addr = static_cast<uintptr_t>(frame.registers[base_reg]) + offset;
    switch (size_bytes) {
        case 1: {
            uint8_t val = *reinterpret_cast<const uint8_t*>(addr);
            return static_cast<uint64_t>(val);
        }
        case 2: {
            uint16_t val = 0;
            std::memcpy(&val, reinterpret_cast<const void*>(addr), 2);
            return static_cast<uint64_t>(val);
        }
        case 4: {
            uint32_t val = 0;
            std::memcpy(&val, reinterpret_cast<const void*>(addr), 4);
            return static_cast<uint64_t>(val);
        }
        case 8: {
            uint64_t val = 0;
            std::memcpy(&val, reinterpret_cast<const void*>(addr), 8);
            return val;
        }
        default:
            return 0;
    }
}

void fast_store(FastFrame& frame, uint8_t val_reg, uint8_t base_reg, uint8_t offset, size_t size_bytes) {
    uintptr_t addr = static_cast<uintptr_t>(frame.registers[base_reg]) + offset;
    uint64_t val = frame.registers[val_reg];
    switch (size_bytes) {
        case 1: {
            uint8_t b = static_cast<uint8_t>(val);
            *reinterpret_cast<uint8_t*>(addr) = b;
            break;
        }
        case 2: {
            uint16_t s = static_cast<uint16_t>(val);
            std::memcpy(reinterpret_cast<void*>(addr), &s, 2);
            break;
        }
        case 4: {
            uint32_t w = static_cast<uint32_t>(val);
            std::memcpy(reinterpret_cast<void*>(addr), &w, 4);
            break;
        }
        case 8: {
            std::memcpy(reinterpret_cast<void*>(addr), &val, 8);
            break;
        }
        default:
            break;
    }
}

uint64_t fast_load_indexed(FastFrame& frame, uint8_t base_reg, uint8_t idx_reg, size_t size_bytes) {
    uintptr_t addr = static_cast<uintptr_t>(frame.registers[base_reg]) + frame.registers[idx_reg];
    switch (size_bytes) {
        case 1: {
            uint8_t val = *reinterpret_cast<const uint8_t*>(addr);
            return static_cast<uint64_t>(val);
        }
        case 2: {
            uint16_t val = 0;
            std::memcpy(&val, reinterpret_cast<const void*>(addr), 2);
            return static_cast<uint64_t>(val);
        }
        case 4: {
            uint32_t val = 0;
            std::memcpy(&val, reinterpret_cast<const void*>(addr), 4);
            return static_cast<uint64_t>(val);
        }
        case 8: {
            uint64_t val = 0;
            std::memcpy(&val, reinterpret_cast<const void*>(addr), 8);
            return val;
        }
        default:
            return 0;
    }
}

void fast_store_indexed(FastFrame& frame, uint8_t val_reg, uint8_t base_reg, uint8_t idx_reg, size_t size_bytes) {
    uintptr_t addr = static_cast<uintptr_t>(frame.registers[base_reg]) + frame.registers[idx_reg];
    uint64_t val = frame.registers[val_reg];
    switch (size_bytes) {
        case 1: {
            uint8_t b = static_cast<uint8_t>(val);
            *reinterpret_cast<uint8_t*>(addr) = b;
            break;
        }
        case 2: {
            uint16_t s = static_cast<uint16_t>(val);
            std::memcpy(reinterpret_cast<void*>(addr), &s, 2);
            break;
        }
        case 4: {
            uint32_t w = static_cast<uint32_t>(val);
            std::memcpy(reinterpret_cast<void*>(addr), &w, 4);
            break;
        }
        case 8: {
            std::memcpy(reinterpret_cast<void*>(addr), &val, 8);
            break;
        }
        default:
            break;
    }
}

} // namespace brass
