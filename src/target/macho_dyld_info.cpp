#include "macho_dyld_info.hpp"
#include "image_util.hpp"

#include <algorithm>
#include <memory>
#include <string_view>

namespace brass::target::macho_dyld {

using namespace brass::target::image;

namespace {

constexpr uint8_t REBASE_TYPE_POINTER = 1;
constexpr uint8_t REBASE_OPCODE_DONE = 0x00;
constexpr uint8_t REBASE_OPCODE_SET_TYPE_IMM = 0x10;
constexpr uint8_t REBASE_OPCODE_SET_SEGMENT_AND_OFFSET_ULEB = 0x20;
constexpr uint8_t REBASE_OPCODE_DO_REBASE_IMM_TIMES = 0x50;

constexpr uint8_t BIND_TYPE_POINTER = 1;
constexpr uint8_t BIND_OPCODE_DONE = 0x00;
constexpr uint8_t BIND_OPCODE_SET_DYLIB_ORDINAL_IMM = 0x10;
constexpr uint8_t BIND_OPCODE_SET_DYLIB_ORDINAL_ULEB = 0x20;
constexpr uint8_t BIND_OPCODE_SET_SYMBOL_TRAILING_FLAGS_IMM = 0x40;
constexpr uint8_t BIND_OPCODE_SET_TYPE_IMM = 0x50;
constexpr uint8_t BIND_OPCODE_SET_ADDEND_SLEB = 0x60;
constexpr uint8_t BIND_OPCODE_SET_SEGMENT_AND_OFFSET_ULEB = 0x70;
constexpr uint8_t BIND_OPCODE_DO_BIND = 0x90;

struct ExportNode {
    std::string edge;
    bool is_terminal = false;
    uint64_t address = 0;
    uint32_t offset = 0;
    std::vector<std::unique_ptr<ExportNode>> children;

    void add(std::string_view name, uint64_t addr) {
        if (name.empty()) {
            is_terminal = true;
            address = addr;
            return;
        }
        for (auto& child : children) {
            size_t match = 0;
            while (match < name.size() && match < child->edge.size() && name[match] == child->edge[match]) {
                match++;
            }
            if (match > 0) {
                if (match == child->edge.size()) {
                    child->add(name.substr(match), addr);
                    return;
                }
                auto split = std::make_unique<ExportNode>();
                split->edge = child->edge.substr(0, match);
                child->edge = child->edge.substr(match);
                std::string_view remainder = name.substr(match);
                if (remainder.empty()) {
                    split->is_terminal = true;
                    split->address = addr;
                } else {
                    auto new_child = std::make_unique<ExportNode>();
                    new_child->edge = std::string(remainder);
                    new_child->is_terminal = true;
                    new_child->address = addr;
                    split->children.push_back(std::move(new_child));
                }
                split->children.push_back(std::move(child));
                child = std::move(split);
                return;
            }
        }
        auto new_child = std::make_unique<ExportNode>();
        new_child->edge = std::string(name);
        new_child->is_terminal = true;
        new_child->address = addr;
        children.push_back(std::move(new_child));
    }

    size_t compute_size() const {
        size_t sz = 0;
        if (is_terminal) {
            size_t term_data_len = uleb128_len(0) + uleb128_len(address);
            sz += uleb128_len(term_data_len) + term_data_len;
        } else {
            sz += uleb128_len(0);
        }
        sz += 1; // child count
        for (const auto& ch : children) {
            sz += ch->edge.size() + 1;
            sz += uleb128_len(ch->offset);
        }
        return sz;
    }

    void collect(std::vector<ExportNode*>& order) {
        order.push_back(this);
        for (auto& ch : children) ch->collect(order);
    }

    void serialize(std::vector<uint8_t>& buf) const {
        if (is_terminal) {
            std::vector<uint8_t> term_data;
            encode_uleb128(term_data, 0); // EXPORT_SYMBOL_FLAGS_KIND_REGULAR
            encode_uleb128(term_data, address);
            encode_uleb128(buf, term_data.size());
            buf.insert(buf.end(), term_data.begin(), term_data.end());
        } else {
            encode_uleb128(buf, 0);
        }
        buf.push_back(static_cast<uint8_t>(children.size()));
        for (const auto& ch : children) {
            buf.insert(buf.end(), ch->edge.begin(), ch->edge.end());
            buf.push_back(0);
            encode_uleb128(buf, ch->offset);
        }
    }
};

} // namespace

std::vector<uint8_t> build_export_trie(const std::vector<std::pair<std::string, uint64_t>>& exports) {
    if (exports.empty()) return {};

    ExportNode root;
    for (const auto& [name, addr] : exports) root.add(name, addr);

    std::vector<ExportNode*> order;
    root.collect(order);

    // Child offsets are ULEBs whose width depends on the offsets themselves;
    // iterate to a fixed point.
    for (int iter = 0; iter < 10; ++iter) {
        uint32_t cur_off = 0;
        bool changed = false;
        for (auto* node : order) {
            if (node->offset != cur_off) {
                node->offset = cur_off;
                changed = true;
            }
            cur_off += static_cast<uint32_t>(node->compute_size());
        }
        if (!changed) break;
    }

    std::vector<uint8_t> trie_bytes;
    for (const auto* node : order) node->serialize(trie_bytes);
    return trie_bytes;
}

std::vector<uint8_t> build_rebase_opcodes(std::vector<RebaseEntry> entries) {
    if (entries.empty()) return {};
    std::sort(entries.begin(), entries.end(), [](const RebaseEntry& a, const RebaseEntry& b) {
        return a.segment != b.segment ? a.segment < b.segment : a.offset < b.offset;
    });
    std::vector<uint8_t> out;
    write_u8(out, REBASE_OPCODE_SET_TYPE_IMM | REBASE_TYPE_POINTER);
    for (const auto& e : entries) {
        write_u8(out, static_cast<uint8_t>(REBASE_OPCODE_SET_SEGMENT_AND_OFFSET_ULEB | (e.segment & 0xF)));
        encode_uleb128(out, e.offset);
        write_u8(out, REBASE_OPCODE_DO_REBASE_IMM_TIMES | 1);
    }
    write_u8(out, REBASE_OPCODE_DONE);
    pad_to(out, 8);
    return out;
}

std::vector<uint8_t> build_bind_opcodes(std::vector<BindEntry> entries) {
    if (entries.empty()) return {};
    std::sort(entries.begin(), entries.end(), [](const BindEntry& a, const BindEntry& b) {
        if (a.ordinal != b.ordinal) return a.ordinal < b.ordinal;
        if (a.symbol != b.symbol) return a.symbol < b.symbol;
        if (a.segment != b.segment) return a.segment < b.segment;
        return a.offset < b.offset;
    });
    std::vector<uint8_t> out;
    write_u8(out, BIND_OPCODE_SET_TYPE_IMM | BIND_TYPE_POINTER);
    uint32_t ordinal = 0;
    std::string symbol;
    int64_t addend = 0;
    bool first = true;
    for (const auto& e : entries) {
        if (first || e.ordinal != ordinal) {
            if (e.ordinal < 16) {
                write_u8(out, static_cast<uint8_t>(BIND_OPCODE_SET_DYLIB_ORDINAL_IMM | e.ordinal));
            } else {
                write_u8(out, BIND_OPCODE_SET_DYLIB_ORDINAL_ULEB);
                encode_uleb128(out, e.ordinal);
            }
            ordinal = e.ordinal;
        }
        if (first || e.symbol != symbol) {
            write_u8(out, BIND_OPCODE_SET_SYMBOL_TRAILING_FLAGS_IMM);
            write_cstring(out, e.symbol);
            symbol = e.symbol;
        }
        if (first || e.addend != addend) {
            write_u8(out, BIND_OPCODE_SET_ADDEND_SLEB);
            encode_sleb128(out, e.addend);
            addend = e.addend;
        }
        write_u8(out, static_cast<uint8_t>(BIND_OPCODE_SET_SEGMENT_AND_OFFSET_ULEB | (e.segment & 0xF)));
        encode_uleb128(out, e.offset);
        write_u8(out, BIND_OPCODE_DO_BIND);
        first = false;
    }
    write_u8(out, BIND_OPCODE_DONE);
    pad_to(out, 8);
    return out;
}

} // namespace brass::target::macho_dyld
