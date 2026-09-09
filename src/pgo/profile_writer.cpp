#include <brass/pgo/profile_data.hpp>
#include <fstream>
#include <sstream>

namespace brass::pgo {

const FunctionProfile* ProfileData::find_function(const std::string& name) const {
    auto it = functions_.find(name);
    if (it != functions_.end()) {
        return &it->second;
    }
    return nullptr;
}

void ProfileData::add_function(FunctionProfile prof) {
    functions_[prof.name] = std::move(prof);
}

bool ProfileData::write_to_file(const std::string& path, std::string* err) const {
    std::ofstream os(path, std::ios::out | std::ios::binary);
    if (!os.is_open()) {
        if (err) {
            *err = "Could not open file for writing: " + path;
        }
        return false;
    }
    return write_to_stream(os, err);
}

bool ProfileData::write_to_stream(std::ostream& os, std::string* err) const {
    ProfileFileHeader header;
    header.magic = kProfileMagic;
    header.version = kProfileVersion;
    header.module_hash = module_hash_;
    header.module_name_length = static_cast<uint32_t>(module_name_.size());
    header.function_count = static_cast<uint32_t>(functions_.size());

    os.write(reinterpret_cast<const char*>(&header), sizeof(header));
    if (!os) {
        if (err) *err = "Failed to write profile header";
        return false;
    }

    if (header.module_name_length > 0) {
        os.write(module_name_.data(), static_cast<std::streamsize>(header.module_name_length));
        if (!os) {
            if (err) *err = "Failed to write module name";
            return false;
        }
    }

    for (const auto& kv : functions_) {
        const FunctionProfile& fn = kv.second;
        uint32_t name_len = static_cast<uint32_t>(fn.name.size());
        os.write(reinterpret_cast<const char*>(&name_len), sizeof(name_len));
        if (name_len > 0) {
            os.write(fn.name.data(), static_cast<std::streamsize>(name_len));
        }

        uint64_t entry_count = fn.entry_count;
        os.write(reinterpret_cast<const char*>(&entry_count), sizeof(entry_count));

        uint32_t num_counters = static_cast<uint32_t>(fn.edge_counters.size());
        os.write(reinterpret_cast<const char*>(&num_counters), sizeof(num_counters));
        if (num_counters > 0) {
            os.write(reinterpret_cast<const char*>(fn.edge_counters.data()),
                     static_cast<std::streamsize>(num_counters * sizeof(uint64_t)));
        }

        uint32_t num_indirect = static_cast<uint32_t>(fn.indirect_targets.size());
        os.write(reinterpret_cast<const char*>(&num_indirect), sizeof(num_indirect));
        for (const auto& it : fn.indirect_targets) {
            uint32_t tlen = static_cast<uint32_t>(it.target_name.size());
            os.write(reinterpret_cast<const char*>(&tlen), sizeof(tlen));
            if (tlen > 0) {
                os.write(it.target_name.data(), static_cast<std::streamsize>(tlen));
            }
            uint64_t count = it.count;
            os.write(reinterpret_cast<const char*>(&count), sizeof(count));
        }

        if (!os) {
            if (err) *err = "Failed to write function profile: " + fn.name;
            return false;
        }
    }

    return true;
}

} // namespace brass::pgo
