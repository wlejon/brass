#include <brass/runtime/exception.hpp>
#include <brass/object/object_writer.hpp>

namespace brass::runtime {

namespace {

constexpr uint8_t DW_EH_PE_omit = 0xFF;
constexpr uint8_t DW_EH_PE_udata4 = 0x03;

void emit_uleb128(std::vector<uint8_t>& buf, uint64_t val) {
    do {
        uint8_t byte = static_cast<uint8_t>(val & 0x7F);
        val >>= 7;
        if (val != 0) byte |= 0x80;
        buf.push_back(byte);
    } while (val != 0);
}

void emit_sleb128(std::vector<uint8_t>& buf, int64_t val) {
    bool more = true;
    while (more) {
        uint8_t byte = static_cast<uint8_t>(val & 0x7F);
        val >>= 7;
        bool sign_bit = (byte & 0x40) != 0;
        if ((val == 0 && !sign_bit) || (val == -1 && sign_bit)) {
            more = false;
        } else {
            byte |= 0x80;
        }
        buf.push_back(byte);
    }
}

} // namespace

void emit_sysv_lsda(object::Section& lsda_sec, const FunctionExceptionTable& table) {
    // 1. Header
    lsda_sec.emit8(DW_EH_PE_omit); // LPStart encoding
    lsda_sec.emit8(DW_EH_PE_omit); // TType encoding

    // 2. Build Call Site Table in temporary buffer
    std::vector<uint8_t> cs_table;
    const auto& scopes = table.scopes();

    for (const auto& s : scopes) {
        // cs_start (4 bytes)
        uint32_t start = s.begin_offset;
        cs_table.push_back(static_cast<uint8_t>(start & 0xFF));
        cs_table.push_back(static_cast<uint8_t>((start >> 8) & 0xFF));
        cs_table.push_back(static_cast<uint8_t>((start >> 16) & 0xFF));
        cs_table.push_back(static_cast<uint8_t>((start >> 24) & 0xFF));

        // cs_len (4 bytes)
        uint32_t len = (s.end_offset >= s.begin_offset) ? (s.end_offset - s.begin_offset) : 0;
        cs_table.push_back(static_cast<uint8_t>(len & 0xFF));
        cs_table.push_back(static_cast<uint8_t>((len >> 8) & 0xFF));
        cs_table.push_back(static_cast<uint8_t>((len >> 16) & 0xFF));
        cs_table.push_back(static_cast<uint8_t>((len >> 24) & 0xFF));

        // cs_lp (4 bytes)
        uint32_t lp = s.landing_pad_offset;
        cs_table.push_back(static_cast<uint8_t>(lp & 0xFF));
        cs_table.push_back(static_cast<uint8_t>((lp >> 8) & 0xFF));
        cs_table.push_back(static_cast<uint8_t>((lp >> 16) & 0xFF));
        cs_table.push_back(static_cast<uint8_t>((lp >> 24) & 0xFF));

        // cs_action (ULEB128: 1 = action record 1)
        emit_uleb128(cs_table, 1);
    }

    // Call site encoding
    lsda_sec.emit8(DW_EH_PE_udata4);

    // Call site table length
    std::vector<uint8_t> len_bytes;
    emit_uleb128(len_bytes, cs_table.size());
    lsda_sec.emit_bytes(len_bytes.data(), len_bytes.size());

    // Call site records
    lsda_sec.emit_bytes(cs_table.data(), cs_table.size());

    // 3. Action Table (Action 1: catch-all / cleanup)
    if (!scopes.empty()) {
        std::vector<uint8_t> action_table;
        emit_sleb128(action_table, 1); // Action 1 filter: 1 (catch-all)
        emit_sleb128(action_table, 0); // Next action: 0 (end of list)
        lsda_sec.emit_bytes(action_table.data(), action_table.size());
    }
}

} // namespace brass::runtime
