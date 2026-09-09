#include <brass/pgo/profile_data.hpp>
#include <fstream>
#include <sstream>

namespace brass::pgo {

std::unique_ptr<ProfileData> ProfileData::read_from_file(const std::string& path, std::string* err) {
    std::ifstream is(path, std::ios::in | std::ios::binary);
    if (!is.is_open()) {
        if (err) {
            *err = "Could not open file for reading: " + path;
        }
        return nullptr;
    }
    return read_from_stream(is, err);
}

std::unique_ptr<ProfileData> ProfileData::read_from_stream(std::istream& is, std::string* err) {
    ProfileFileHeader header;
    is.read(reinterpret_cast<char*>(&header), sizeof(header));
    if (!is || is.gcount() != sizeof(header)) {
        if (err) *err = "Corrupt profile: unexpected EOF reading header";
        return nullptr;
    }

    if (header.magic != kProfileMagic) {
        if (err) {
            std::ostringstream ss;
            ss << "Invalid profile magic header (got 0x" << std::hex << header.magic
               << ", expected 0x" << kProfileMagic << ")";
            *err = ss.str();
        }
        return nullptr;
    }

    if (header.version != kProfileVersion) {
        if (err) {
            *err = "Unsupported profile version: " + std::to_string(header.version);
        }
        return nullptr;
    }

    std::string mod_name;
    if (header.module_name_length > 0) {
        mod_name.resize(header.module_name_length);
        is.read(mod_name.data(), static_cast<std::streamsize>(header.module_name_length));
        if (!is) {
            if (err) *err = "Corrupt profile: failed to read module name";
            return nullptr;
        }
    }

    auto prof = std::make_unique<ProfileData>();
    prof->set_module_name(std::move(mod_name));
    prof->set_module_hash(header.module_hash);

    for (uint32_t i = 0; i < header.function_count; ++i) {
        FunctionProfile fn;
        uint32_t name_len = 0;
        is.read(reinterpret_cast<char*>(&name_len), sizeof(name_len));
        if (!is) {
            if (err) *err = "Corrupt profile: failed to read function name length";
            return nullptr;
        }

        if (name_len > 0) {
            fn.name.resize(name_len);
            is.read(fn.name.data(), static_cast<std::streamsize>(name_len));
            if (!is) {
                if (err) *err = "Corrupt profile: failed to read function name";
                return nullptr;
            }
        }

        is.read(reinterpret_cast<char*>(&fn.entry_count), sizeof(fn.entry_count));
        if (!is) {
            if (err) *err = "Corrupt profile: failed to read entry count";
            return nullptr;
        }

        uint32_t num_counters = 0;
        is.read(reinterpret_cast<char*>(&num_counters), sizeof(num_counters));
        if (!is) {
            if (err) *err = "Corrupt profile: failed to read counter count";
            return nullptr;
        }

        if (num_counters > 0) {
            fn.edge_counters.resize(num_counters);
            is.read(reinterpret_cast<char*>(fn.edge_counters.data()),
                    static_cast<std::streamsize>(num_counters * sizeof(uint64_t)));
            if (!is) {
                if (err) *err = "Corrupt profile: failed to read edge counters";
                return nullptr;
            }
        }

        uint32_t num_indirect = 0;
        is.read(reinterpret_cast<char*>(&num_indirect), sizeof(num_indirect));
        if (!is) {
            if (err) *err = "Corrupt profile: failed to read indirect target count";
            return nullptr;
        }

        for (uint32_t j = 0; j < num_indirect; ++j) {
            uint32_t tlen = 0;
            is.read(reinterpret_cast<char*>(&tlen), sizeof(tlen));
            if (!is) {
                if (err) *err = "Corrupt profile: failed to read indirect target name length";
                return nullptr;
            }
            std::string tname;
            if (tlen > 0) {
                tname.resize(tlen);
                is.read(tname.data(), static_cast<std::streamsize>(tlen));
                if (!is) {
                    if (err) *err = "Corrupt profile: failed to read indirect target name";
                    return nullptr;
                }
            }
            uint64_t count = 0;
            is.read(reinterpret_cast<char*>(&count), sizeof(count));
            if (!is) {
                if (err) *err = "Corrupt profile: failed to read indirect target count";
                return nullptr;
            }
            fn.add_indirect_target(std::move(tname), count);
        }

        prof->add_function(std::move(fn));
    }

    return prof;
}

} // namespace brass::pgo
