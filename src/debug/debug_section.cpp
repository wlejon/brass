#include <brass/debug/debug_section.hpp>
#include <algorithm>
#include <cstring>
#include <unordered_map>

namespace brass {

namespace {

void write_u32(std::vector<uint8_t>& buf, uint32_t val) {
    buf.push_back(static_cast<uint8_t>(val & 0xFF));
    buf.push_back(static_cast<uint8_t>((val >> 8) & 0xFF));
    buf.push_back(static_cast<uint8_t>((val >> 16) & 0xFF));
    buf.push_back(static_cast<uint8_t>((val >> 24) & 0xFF));
}

bool read_u32(const uint8_t*& ptr, const uint8_t* end, uint32_t& val) {
    if (ptr + 4 > end) return false;
    val = static_cast<uint32_t>(ptr[0]) |
          (static_cast<uint32_t>(ptr[1]) << 8) |
          (static_cast<uint32_t>(ptr[2]) << 16) |
          (static_cast<uint32_t>(ptr[3]) << 24);
    ptr += 4;
    return true;
}

void write_uleb128(std::vector<uint8_t>& buf, uint64_t val) {
    do {
        uint8_t byte = static_cast<uint8_t>(val & 0x7F);
        val >>= 7;
        if (val != 0) byte |= 0x80;
        buf.push_back(byte);
    } while (val != 0);
}

bool read_uleb128(const uint8_t*& ptr, const uint8_t* end, uint64_t& val) {
    val = 0;
    uint32_t shift = 0;
    while (ptr < end) {
        uint8_t byte = *ptr++;
        val |= (static_cast<uint64_t>(byte & 0x7F) << shift);
        if ((byte & 0x80) == 0) return true;
        shift += 7;
        if (shift >= 64) return false;
    }
    return false;
}

void write_sleb128(std::vector<uint8_t>& buf, int64_t val) {
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

bool read_sleb128(const uint8_t*& ptr, const uint8_t* end, int64_t& val) {
    val = 0;
    uint32_t shift = 0;
    while (ptr < end) {
        uint8_t byte = *ptr++;
        val |= (static_cast<int64_t>(byte & 0x7F) << shift);
        shift += 7;
        if ((byte & 0x80) == 0) {
            if (shift < 64 && (byte & 0x40) != 0) {
                val |= (~0ULL << shift);
            }
            return true;
        }
        if (shift >= 64) return false;
    }
    return false;
}

} // namespace

void FunctionDebugTable::add_line_entry(uint32_t code_offset, DebugLoc loc) {
    if (line_entries_.empty() || code_offset >= line_entries_.back().code_offset) {
        if (!line_entries_.empty() && line_entries_.back().code_offset == code_offset) {
            line_entries_.back().loc = loc;
        } else {
            line_entries_.push_back({code_offset, loc});
        }
        return;
    }

    auto it = std::lower_bound(
        line_entries_.begin(), line_entries_.end(), code_offset,
        [](const DebugLineEntry& e, uint32_t val) {
            return e.code_offset < val;
        }
    );

    if (it != line_entries_.end() && it->code_offset == code_offset) {
        it->loc = loc;
    } else {
        line_entries_.insert(it, {code_offset, loc});
    }
}

void FunctionDebugTable::set_line_entries(std::vector<DebugLineEntry> entries) {
    line_entries_ = std::move(entries);
    std::sort(
        line_entries_.begin(), line_entries_.end(),
        [](const DebugLineEntry& a, const DebugLineEntry& b) {
            return a.code_offset < b.code_offset;
        }
    );
}

DebugLoc FunctionDebugTable::resolve_offset(uint32_t offset) const {
    if (line_entries_.empty()) return DebugLoc();
    if (offset < line_entries_.front().code_offset) return DebugLoc();

    auto it = std::upper_bound(
        line_entries_.begin(), line_entries_.end(), offset,
        [](uint32_t val, const DebugLineEntry& e) {
            return val < e.code_offset;
        }
    );

    if (it == line_entries_.begin()) return DebugLoc();
    --it;
    return it->loc;
}

SourceMap FunctionDebugTable::to_source_map(const DebugContext& ctx, const std::string& output_name) const {
    SourceMap sm(output_name.empty() ? function_name_ : output_name);
    for (const auto& f : ctx.files()) {
        sm.add_source(f);
    }
    for (const auto& entry : line_entries_) {
        sm.add_mapping(entry.code_offset, entry.loc);
    }
    return sm;
}

std::vector<uint8_t> serialize_debug_section(const DebugContext& ctx, const std::vector<FunctionDebugTable>& tables) {
    std::vector<uint8_t> string_table;
    std::unordered_map<std::string, uint32_t> string_map;

    auto intern_str = [&](const std::string& s) -> uint32_t {
        auto it = string_map.find(s);
        if (it != string_map.end()) return it->second;
        uint32_t offset = static_cast<uint32_t>(string_table.size());
        for (char c : s) {
            string_table.push_back(static_cast<uint8_t>(c));
        }
        string_table.push_back(0);
        string_map[s] = offset;
        return offset;
    };

    // Index 0 string
    intern_str("");

    // Intern all file names
    std::vector<uint32_t> file_str_offsets;
    file_str_offsets.reserve(ctx.files().size());
    for (const auto& f : ctx.files()) {
        file_str_offsets.push_back(intern_str(f));
    }

    // Intern all inlined scope callee names
    std::vector<uint32_t> scope_name_offsets;
    scope_name_offsets.reserve(ctx.inlined_scopes().size());
    for (const auto& s : ctx.inlined_scopes()) {
        scope_name_offsets.push_back(intern_str(s.callee_name));
    }

    // Intern function names
    std::vector<uint32_t> fn_name_offsets;
    fn_name_offsets.reserve(tables.size());
    for (const auto& t : tables) {
        fn_name_offsets.push_back(intern_str(t.function_name()));
    }

    // Layout:
    // Header: 10 * 4 = 40 bytes
    // String table
    // File table
    // Scope table
    // Function table
    std::vector<uint8_t> out;
    out.reserve(1024);

    // Placeholder for header
    for (int i = 0; i < 10; ++i) write_u32(out, 0);

    // 1. String table
    uint32_t str_table_offset = static_cast<uint32_t>(out.size());
    out.insert(out.end(), string_table.begin(), string_table.end());
    uint32_t str_table_size = static_cast<uint32_t>(string_table.size());

    // 2. File table
    uint32_t file_table_offset = static_cast<uint32_t>(out.size());
    uint32_t file_count = static_cast<uint32_t>(file_str_offsets.size());
    for (uint32_t off : file_str_offsets) {
        write_u32(out, off);
    }

    // 3. Inlined Scope table
    uint32_t scope_table_offset = static_cast<uint32_t>(out.size());
    uint32_t scope_count = static_cast<uint32_t>(ctx.inlined_scopes().size());
    for (size_t i = 0; i < scope_count; ++i) {
        const auto& s = ctx.inlined_scopes()[i];
        write_u32(out, scope_name_offsets[i]);
        write_u32(out, s.callsite_loc.file_id);
        write_u32(out, s.callsite_loc.line);
        write_u32(out, s.callsite_loc.column);
        write_u32(out, s.callsite_loc.inlined_at_id);
        write_u32(out, s.parent_inlined_at_id);
    }

    // 4. Function table
    uint32_t fn_table_offset = static_cast<uint32_t>(out.size());
    uint32_t fn_count = static_cast<uint32_t>(tables.size());
    for (size_t i = 0; i < fn_count; ++i) {
        const auto& t = tables[i];
        write_u32(out, fn_name_offsets[i]);
        write_u32(out, t.code_size());
        write_u32(out, static_cast<uint32_t>(t.line_entries().size()));

        // Delta-encoded line entries
        uint32_t prev_offset = 0;
        int64_t prev_file = 0;
        int64_t prev_line = 0;
        int64_t prev_col = 0;

        for (const auto& entry : t.line_entries()) {
            uint64_t delta_off = entry.code_offset - prev_offset;
            prev_offset = entry.code_offset;

            int64_t delta_f = static_cast<int64_t>(entry.loc.file_id) - prev_file;
            prev_file = static_cast<int64_t>(entry.loc.file_id);

            int64_t delta_l = static_cast<int64_t>(entry.loc.line) - prev_line;
            prev_line = static_cast<int64_t>(entry.loc.line);

            int64_t delta_c = static_cast<int64_t>(entry.loc.column) - prev_col;
            prev_col = static_cast<int64_t>(entry.loc.column);

            write_uleb128(out, delta_off);
            write_sleb128(out, delta_f);
            write_sleb128(out, delta_l);
            write_sleb128(out, delta_c);
            write_uleb128(out, entry.loc.inlined_at_id);
        }
    }

    // Rewrite header
    std::vector<uint8_t> header;
    write_u32(header, kDebugSectionMagic);
    write_u32(header, kDebugSectionVersion);
    write_u32(header, str_table_offset);
    write_u32(header, str_table_size);
    write_u32(header, file_table_offset);
    write_u32(header, file_count);
    write_u32(header, scope_table_offset);
    write_u32(header, scope_count);
    write_u32(header, fn_table_offset);
    write_u32(header, fn_count);

    std::memcpy(out.data(), header.data(), header.size());
    return out;
}

bool deserialize_debug_section(const uint8_t* data, size_t size, DebugContext& ctx, std::vector<FunctionDebugTable>& tables) {
    if (!data || size < 40) return false;
    const uint8_t* ptr = data;
    const uint8_t* end = data + size;

    uint32_t magic = 0;
    uint32_t version = 0;
    uint32_t str_table_offset = 0;
    uint32_t str_table_size = 0;
    uint32_t file_table_offset = 0;
    uint32_t file_count = 0;
    uint32_t scope_table_offset = 0;
    uint32_t scope_count = 0;
    uint32_t fn_table_offset = 0;
    uint32_t fn_count = 0;

    if (!read_u32(ptr, end, magic) || magic != kDebugSectionMagic) return false;
    if (!read_u32(ptr, end, version) || version != kDebugSectionVersion) return false;
    if (!read_u32(ptr, end, str_table_offset) || !read_u32(ptr, end, str_table_size)) return false;
    if (!read_u32(ptr, end, file_table_offset) || !read_u32(ptr, end, file_count)) return false;
    if (!read_u32(ptr, end, scope_table_offset) || !read_u32(ptr, end, scope_count)) return false;
    if (!read_u32(ptr, end, fn_table_offset) || !read_u32(ptr, end, fn_count)) return false;

    if (str_table_offset + str_table_size > size) return false;
    const char* str_pool = reinterpret_cast<const char*>(data + str_table_offset);
    size_t str_limit = str_table_size;

    auto get_str = [&](uint32_t off) -> std::string {
        if (off >= str_limit) return "";
        const char* s = str_pool + off;
        size_t max_len = str_limit - off;
        size_t len = strnlen(s, max_len);
        return std::string(s, len);
    };

    ctx.clear();
    tables.clear();

    // Read files
    if (file_table_offset + file_count * 4 > size) return false;
    const uint8_t* f_ptr = data + file_table_offset;
    for (uint32_t i = 0; i < file_count; ++i) {
        uint32_t str_off = 0;
        if (!read_u32(f_ptr, end, str_off)) return false;
        ctx.get_or_add_file(get_str(str_off));
    }

    // Read scopes
    if (scope_table_offset + scope_count * 24 > size) return false;
    const uint8_t* s_ptr = data + scope_table_offset;
    for (uint32_t i = 0; i < scope_count; ++i) {
        uint32_t name_off = 0, f_id = 0, l = 0, c = 0, inlined_at = 0, parent_id = 0;
        if (!read_u32(s_ptr, end, name_off) ||
            !read_u32(s_ptr, end, f_id) ||
            !read_u32(s_ptr, end, l) ||
            !read_u32(s_ptr, end, c) ||
            !read_u32(s_ptr, end, inlined_at) ||
            !read_u32(s_ptr, end, parent_id)) {
            return false;
        }
        ctx.record_inlined_scope(get_str(name_off), DebugLoc(f_id, l, c, inlined_at), parent_id);
    }

    // Read functions
    if (fn_table_offset > size) return false;
    const uint8_t* fn_ptr = data + fn_table_offset;
    tables.reserve(fn_count);

    for (uint32_t i = 0; i < fn_count; ++i) {
        uint32_t name_off = 0, code_size = 0, line_count = 0;
        if (!read_u32(fn_ptr, end, name_off) ||
            !read_u32(fn_ptr, end, code_size) ||
            !read_u32(fn_ptr, end, line_count)) {
            return false;
        }

        FunctionDebugTable table(get_str(name_off), code_size);
        std::vector<DebugLineEntry> entries;
        entries.reserve(line_count);

        uint32_t prev_offset = 0;
        int64_t prev_file = 0;
        int64_t prev_line = 0;
        int64_t prev_col = 0;

        for (uint32_t j = 0; j < line_count; ++j) {
            uint64_t delta_off = 0;
            int64_t delta_f = 0;
            int64_t delta_l = 0;
            int64_t delta_c = 0;
            uint64_t inlined_id = 0;

            if (!read_uleb128(fn_ptr, end, delta_off) ||
                !read_sleb128(fn_ptr, end, delta_f) ||
                !read_sleb128(fn_ptr, end, delta_l) ||
                !read_sleb128(fn_ptr, end, delta_c) ||
                !read_uleb128(fn_ptr, end, inlined_id)) {
                return false;
            }

            uint32_t offset = prev_offset + static_cast<uint32_t>(delta_off);
            prev_offset = offset;

            int64_t f_id = prev_file + delta_f;
            prev_file = f_id;

            int64_t l_num = prev_line + delta_l;
            prev_line = l_num;

            int64_t c_num = prev_col + delta_c;
            prev_col = c_num;

            entries.push_back({
                offset,
                DebugLoc(static_cast<uint32_t>(f_id),
                         static_cast<uint32_t>(l_num),
                         static_cast<uint32_t>(c_num),
                         static_cast<uint32_t>(inlined_id))
            });
        }
        table.set_line_entries(std::move(entries));
        tables.push_back(std::move(table));
    }

    return true;
}

} // namespace brass
