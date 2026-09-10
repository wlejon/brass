#include <brass/debug/source_map.hpp>
#include <algorithm>
#include <fstream>
#include <sstream>

namespace brass {

namespace {

constexpr std::string_view kBase64Chars =
    "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

int base64_char_to_val(char c) {
    if (c >= 'A' && c <= 'Z') return c - 'A';
    if (c >= 'a' && c <= 'z') return c - 'a' + 26;
    if (c >= '0' && c <= '9') return c - '0' + 52;
    if (c == '+') return 62;
    if (c == '/') return 63;
    return -1;
}

std::string escape_json_string(const std::string& str) {
    std::string out;
    out.reserve(str.size() + 8);
    for (char c : str) {
        switch (c) {
            case '"':  out += "\\\""; break;
            case '\\': out += "\\\\"; break;
            case '\b': out += "\\b"; break;
            case '\f': out += "\\f"; break;
            case '\n': out += "\\n"; break;
            case '\r': out += "\\r"; break;
            case '\t': out += "\\t"; break;
            default:
                if (static_cast<unsigned char>(c) < 0x20) {
                    char buf[8];
                    snprintf(buf, sizeof(buf), "\\u%04x", static_cast<unsigned char>(c));
                    out += buf;
                } else {
                    out += c;
                }
                break;
        }
    }
    return out;
}

// Simple JSON string parser helper
std::string parse_json_string(std::string_view::const_iterator& it, std::string_view::const_iterator end, bool& ok) {
    ok = false;
    if (it == end || *it != '"') return "";
    ++it;
    std::string res;
    while (it != end) {
        char c = *it++;
        if (c == '"') {
            ok = true;
            return res;
        }
        if (c == '\\') {
            if (it == end) return "";
            char esc = *it++;
            switch (esc) {
                case '"': res += '"'; break;
                case '\\': res += '\\'; break;
                case '/': res += '/'; break;
                case 'b': res += '\b'; break;
                case 'f': res += '\f'; break;
                case 'n': res += '\n'; break;
                case 'r': res += '\r'; break;
                case 't': res += '\t'; break;
                case 'u': {
                    if (std::distance(it, end) < 4) return "";
                    // Basic ASCII hex check
                    char hex[5] = { *it, *(it + 1), *(it + 2), *(it + 3), '\0' };
                    it += 4;
                    unsigned long code = std::strtoul(hex, nullptr, 16);
                    if (code <= 0x7F) {
                        res += static_cast<char>(code);
                    }
                    break;
                }
                default: res += esc; break;
            }
        } else {
            res += c;
        }
    }
    return "";
}

void skip_json_ws(std::string_view::const_iterator& it, std::string_view::const_iterator end) {
    while (it != end && (*it == ' ' || *it == '\t' || *it == '\n' || *it == '\r')) {
        ++it;
    }
}

} // namespace

namespace vlq {

std::string encode(int32_t value) {
    uint32_t vlq = (value < 0)
        ? (static_cast<uint32_t>(-static_cast<int64_t>(value)) << 1) | 1
        : (static_cast<uint32_t>(value) << 1);

    std::string out;
    while (vlq > 0x1F) {
        out += kBase64Chars[(vlq & 0x1F) | 0x20];
        vlq >>= 5;
    }
    out += kBase64Chars[vlq & 0x1F];
    return out;
}

bool decode(std::string_view::const_iterator& it, std::string_view::const_iterator end, int32_t& result) {
    uint32_t val = 0;
    uint32_t shift = 0;
    bool continuation = true;

    while (continuation) {
        if (it == end) return false;
        char c = *it++;
        int digit = base64_char_to_val(c);
        if (digit < 0) return false;

        continuation = (digit & 0x20) != 0;
        uint32_t chunk = static_cast<uint32_t>(digit & 0x1F);
        val |= (chunk << shift);
        shift += 5;
    }

    bool is_neg = (val & 1) != 0;
    uint32_t mag = val >> 1;
    result = is_neg ? -static_cast<int32_t>(mag) : static_cast<int32_t>(mag);
    return true;
}

} // namespace vlq

void SourceMap::set_sources(std::vector<std::string> sources) {
    sources_ = std::move(sources);
    source_to_idx_.clear();
    for (uint32_t i = 0; i < static_cast<uint32_t>(sources_.size()); ++i) {
        source_to_idx_[sources_[i]] = i;
    }
}

void SourceMap::add_source(const std::string& src) {
    if (source_to_idx_.find(src) == source_to_idx_.end()) {
        uint32_t idx = static_cast<uint32_t>(sources_.size());
        sources_.push_back(src);
        source_to_idx_[src] = idx;
    }
}

uint32_t SourceMap::get_or_add_source(const std::string& src) {
    auto it = source_to_idx_.find(src);
    if (it != source_to_idx_.end()) {
        return it->second;
    }
    uint32_t idx = static_cast<uint32_t>(sources_.size());
    sources_.push_back(src);
    source_to_idx_[src] = idx;
    return idx;
}

void SourceMap::set_names(std::vector<std::string> names) {
    names_ = std::move(names);
    name_to_idx_.clear();
    for (uint32_t i = 0; i < static_cast<uint32_t>(names_.size()); ++i) {
        name_to_idx_[names_[i]] = i;
    }
}

void SourceMap::add_name(const std::string& name) {
    if (name_to_idx_.find(name) == name_to_idx_.end()) {
        uint32_t idx = static_cast<uint32_t>(names_.size());
        names_.push_back(name);
        name_to_idx_[name] = idx;
    }
}

uint32_t SourceMap::get_or_add_name(const std::string& name) {
    auto it = name_to_idx_.find(name);
    if (it != name_to_idx_.end()) {
        return it->second;
    }
    uint32_t idx = static_cast<uint32_t>(names_.size());
    names_.push_back(name);
    name_to_idx_[name] = idx;
    return idx;
}

void SourceMap::add_mapping(uint32_t code_offset, DebugLoc loc, const std::string& symbol) {
    if (!symbol.empty()) {
        get_or_add_name(symbol);
    }

    if (mappings_.empty() || code_offset >= mappings_.back().code_offset) {
        if (!mappings_.empty() && mappings_.back().code_offset == code_offset) {
            mappings_.back().loc = loc;
            mappings_.back().symbol = symbol;
        } else {
            mappings_.push_back({code_offset, loc, symbol});
        }
        return;
    }

    auto it = std::lower_bound(
        mappings_.begin(), mappings_.end(), code_offset,
        [](const SourceMapping& m, uint32_t val) {
            return m.code_offset < val;
        }
    );

    if (it != mappings_.end() && it->code_offset == code_offset) {
        it->loc = loc;
        it->symbol = symbol;
    } else {
        mappings_.insert(it, {code_offset, loc, symbol});
    }
}

std::string SourceMap::to_json() const {
    std::ostringstream ss;
    ss << "{\n";
    ss << "  \"version\": 3,\n";
    ss << "  \"file\": \"" << escape_json_string(file_) << "\",\n";

    ss << "  \"sources\": [";
    for (size_t i = 0; i < sources_.size(); ++i) {
        if (i > 0) ss << ", ";
        ss << "\"" << escape_json_string(sources_[i]) << "\"";
    }
    ss << "],\n";

    ss << "  \"names\": [";
    for (size_t i = 0; i < names_.size(); ++i) {
        if (i > 0) ss << ", ";
        ss << "\"" << escape_json_string(names_[i]) << "\"";
    }
    ss << "],\n";

    // Encode mappings
    std::string mappings_str;
    int32_t prev_col = 0;
    int32_t prev_src_idx = 0;
    int32_t prev_orig_line = 0;
    int32_t prev_orig_col = 0;
    int32_t prev_name_idx = 0;

    for (size_t i = 0; i < mappings_.size(); ++i) {
        if (i > 0) mappings_str += ',';
        const auto& m = mappings_[i];

        int32_t col = static_cast<int32_t>(m.code_offset);
        int32_t col_delta = col - prev_col;
        prev_col = col;

        int32_t src_idx = (m.loc.file_id > 0) ? static_cast<int32_t>(m.loc.file_id - 1) : 0;
        int32_t src_delta = src_idx - prev_src_idx;
        prev_src_idx = src_idx;

        int32_t orig_line = (m.loc.line > 0) ? static_cast<int32_t>(m.loc.line - 1) : 0;
        int32_t line_delta = orig_line - prev_orig_line;
        prev_orig_line = orig_line;

        int32_t orig_col = (m.loc.column > 0) ? static_cast<int32_t>(m.loc.column - 1) : 0;
        int32_t col_orig_delta = orig_col - prev_orig_col;
        prev_orig_col = orig_col;

        mappings_str += vlq::encode(col_delta);
        mappings_str += vlq::encode(src_delta);
        mappings_str += vlq::encode(line_delta);
        mappings_str += vlq::encode(col_orig_delta);

        if (!m.symbol.empty()) {
            auto it = name_to_idx_.find(m.symbol);
            if (it != name_to_idx_.end()) {
                int32_t name_idx = static_cast<int32_t>(it->second);
                int32_t name_delta = name_idx - prev_name_idx;
                prev_name_idx = name_idx;
                mappings_str += vlq::encode(name_delta);
            }
        }
    }

    ss << "  \"mappings\": \"" << mappings_str << "\"\n";
    ss << "}\n";
    return ss.str();
}

bool SourceMap::write_file(const std::string& path, std::string* err) const {
    std::ofstream out(path, std::ios::out | std::ios::binary);
    if (!out.is_open()) {
        if (err) *err = "Failed to open file for writing: " + path;
        return false;
    }
    std::string json = to_json();
    out.write(json.data(), static_cast<std::streamsize>(json.size()));
    if (!out.good()) {
        if (err) *err = "Failed to write data to file: " + path;
        return false;
    }
    return true;
}

std::unique_ptr<SourceMap> SourceMap::parse_json(const std::string& json, std::string* err) {
    auto sm = std::make_unique<SourceMap>();
    std::string_view view(json);
    auto it = view.begin();
    auto end = view.end();

    auto set_error = [&](const std::string& msg) {
        if (err) *err = msg;
    };

    skip_json_ws(it, end);
    if (it == end || *it != '{') {
        set_error("Expected '{' at start of JSON");
        return nullptr;
    }
    ++it;

    std::string mappings_content;

    while (it != end) {
        skip_json_ws(it, end);
        if (it == end || *it == '}') {
            if (it != end) ++it;
            break;
        }
        if (*it == ',') {
            ++it;
            continue;
        }

        bool ok = false;
        std::string key = parse_json_string(it, end, ok);
        if (!ok) {
            set_error("Failed to parse JSON key");
            return nullptr;
        }

        skip_json_ws(it, end);
        if (it == end || *it != ':') {
            set_error("Expected ':' after key");
            return nullptr;
        }
        ++it;
        skip_json_ws(it, end);

        if (key == "version") {
            int ver = 0;
            while (it != end && *it >= '0' && *it <= '9') {
                ver = ver * 10 + (*it - '0');
                ++it;
            }
            if (ver != 3) {
                set_error("Unsupported source map version: " + std::to_string(ver));
                return nullptr;
            }
        } else if (key == "file") {
            std::string file_val = parse_json_string(it, end, ok);
            if (ok) sm->set_file(file_val);
        } else if (key == "sources") {
            if (it == end || *it != '[') {
                set_error("Expected '[' for sources");
                return nullptr;
            }
            ++it;
            std::vector<std::string> srcs;
            while (it != end) {
                skip_json_ws(it, end);
                if (it == end || *it == ']') {
                    if (it != end) ++it;
                    break;
                }
                if (*it == ',') {
                    ++it;
                    continue;
                }
                std::string s = parse_json_string(it, end, ok);
                if (ok) srcs.push_back(std::move(s));
            }
            sm->set_sources(std::move(srcs));
        } else if (key == "names") {
            if (it == end || *it != '[') {
                set_error("Expected '[' for names");
                return nullptr;
            }
            ++it;
            std::vector<std::string> nms;
            while (it != end) {
                skip_json_ws(it, end);
                if (it == end || *it == ']') {
                    if (it != end) ++it;
                    break;
                }
                if (*it == ',') {
                    ++it;
                    continue;
                }
                std::string n = parse_json_string(it, end, ok);
                if (ok) nms.push_back(std::move(n));
            }
            sm->set_names(std::move(nms));
        } else if (key == "mappings") {
            mappings_content = parse_json_string(it, end, ok);
            if (!ok) {
                set_error("Failed to parse mappings string");
                return nullptr;
            }
        } else {
            // Skip unknown value
            if (*it == '"') {
                parse_json_string(it, end, ok);
            } else if (*it == '[' || *it == '{') {
                char open = *it++;
                char close = (open == '[') ? ']' : '}';
                int depth = 1;
                while (it != end && depth > 0) {
                    if (*it == open) depth++;
                    else if (*it == close) depth--;
                    ++it;
                }
            } else {
                while (it != end && *it != ',' && *it != '}') ++it;
            }
        }
    }

    // Now decode mappings_content
    std::string_view m_view(mappings_content);
    auto m_it = m_view.begin();
    auto m_end = m_view.end();

    int32_t prev_col = 0;
    int32_t prev_src_idx = 0;
    int32_t prev_orig_line = 0;
    int32_t prev_orig_col = 0;
    int32_t prev_name_idx = 0;

    while (m_it != m_end) {
        if (*m_it == ';') {
            prev_col = 0;
            ++m_it;
            continue;
        }
        if (*m_it == ',') {
            ++m_it;
            continue;
        }

        int32_t col_delta = 0;
        if (!vlq::decode(m_it, m_end, col_delta)) break;
        int32_t col = prev_col + col_delta;
        prev_col = col;

        if (m_it == m_end || *m_it == ',' || *m_it == ';') {
            // 1-field segment
            sm->add_mapping(static_cast<uint32_t>(col), DebugLoc());
            continue;
        }

        int32_t src_delta = 0;
        int32_t line_delta = 0;
        int32_t orig_col_delta = 0;
        if (!vlq::decode(m_it, m_end, src_delta) ||
            !vlq::decode(m_it, m_end, line_delta) ||
            !vlq::decode(m_it, m_end, orig_col_delta)) {
            set_error("Corrupt VLQ sequence in mappings");
            return nullptr;
        }

        int32_t src_idx = prev_src_idx + src_delta;
        prev_src_idx = src_idx;
        int32_t orig_line = prev_orig_line + line_delta;
        prev_orig_line = orig_line;
        int32_t orig_col = prev_orig_col + orig_col_delta;
        prev_orig_col = orig_col;

        std::string symbol;
        if (m_it != m_end && *m_it != ',' && *m_it != ';') {
            int32_t name_delta = 0;
            if (vlq::decode(m_it, m_end, name_delta)) {
                int32_t name_idx = prev_name_idx + name_delta;
                prev_name_idx = name_idx;
                if (name_idx >= 0 && static_cast<size_t>(name_idx) < sm->names().size()) {
                    symbol = sm->names()[static_cast<size_t>(name_idx)];
                }
            }
        }

        DebugLoc loc(
            static_cast<uint32_t>(src_idx + 1),
            static_cast<uint32_t>(orig_line + 1),
            static_cast<uint32_t>(orig_col + 1),
            0
        );
        sm->add_mapping(static_cast<uint32_t>(col), loc, symbol);
    }

    return sm;
}

DebugLoc SourceMap::resolve_offset(uint32_t code_offset) const {
    if (mappings_.empty()) return DebugLoc();
    if (code_offset < mappings_.front().code_offset) return DebugLoc();

    auto it = std::upper_bound(
        mappings_.begin(), mappings_.end(), code_offset,
        [](uint32_t val, const SourceMapping& m) {
            return val < m.code_offset;
        }
    );

    if (it == mappings_.begin()) return DebugLoc();
    --it;
    return it->loc;
}

} // namespace brass
