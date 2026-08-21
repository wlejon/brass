#include <brass/target/x64/x64_operands.hpp>
#include <sstream>
#include <ostream>

namespace brass::x64 {

std::string to_string(const MemAddress& mem) {
    if (mem.is_rip_rel) {
        std::ostringstream oss;
        oss << "[rip";
        if (mem.disp > 0) {
            oss << " + " << mem.disp;
        } else if (mem.disp < 0) {
            oss << " - " << (-mem.disp);
        }
        oss << "]";
        return oss.str();
    }

    std::ostringstream oss;
    oss << "[";
    bool has_elem = false;
    if (mem.has_base()) {
        oss << to_string(mem.base, OperandSize::Qword);
        has_elem = true;
    }
    if (mem.has_index()) {
        if (has_elem) oss << " + ";
        oss << to_string(mem.index, OperandSize::Qword);
        uint8_t sc = scale_value(mem.scale);
        if (sc > 1) {
            oss << "*" << static_cast<int>(sc);
        }
        has_elem = true;
    }
    if (mem.disp != 0 || !has_elem) {
        if (has_elem) {
            if (mem.disp > 0) {
                oss << " + " << mem.disp;
            } else {
                oss << " - " << (-mem.disp);
            }
        } else {
            oss << mem.disp;
        }
    }
    oss << "]";
    return oss.str();
}

} // namespace brass::x64
