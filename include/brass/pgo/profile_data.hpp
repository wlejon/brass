#pragma once

#include <brass/pgo/profile_format.hpp>
#include <string>
#include <vector>
#include <unordered_map>
#include <memory>
#include <cstdint>
#include <iosfwd>

namespace brass::pgo {

struct IndirectCallProfile {
    std::string target_name;
    uint64_t count = 0;
};

struct FunctionProfile {
    std::string name;
    uint64_t entry_count = 0;
    std::vector<uint64_t> edge_counters;
    std::vector<IndirectCallProfile> indirect_targets;

    uint32_t edge_counter_count() const noexcept {
        return static_cast<uint32_t>(edge_counters.size());
    }

    void add_indirect_target(std::string target, uint64_t count) {
        indirect_targets.push_back(IndirectCallProfile{std::move(target), count});
    }

    uint64_t get_indirect_target_count(const std::string& target) const {
        for (const auto& it : indirect_targets) {
            if (it.target_name == target) return it.count;
        }
        return 0;
    }
};

class ProfileData {
public:
    ProfileData() = default;

    const std::string& module_name() const noexcept { return module_name_; }
    void set_module_name(std::string name) { module_name_ = std::move(name); }

    uint64_t module_hash() const noexcept { return module_hash_; }
    void set_module_hash(uint64_t hash) noexcept { module_hash_ = hash; }

    const FunctionProfile* find_function(const std::string& name) const;
    void add_function(FunctionProfile prof);

    const std::unordered_map<std::string, FunctionProfile>& functions() const noexcept {
        return functions_;
    }

    size_t function_count() const noexcept { return functions_.size(); }

    bool write_to_file(const std::string& path, std::string* err = nullptr) const;
    bool write_to_stream(std::ostream& os, std::string* err = nullptr) const;

    static std::unique_ptr<ProfileData> read_from_file(const std::string& path, std::string* err = nullptr);
    static std::unique_ptr<ProfileData> read_from_stream(std::istream& is, std::string* err = nullptr);

private:
    std::string module_name_;
    uint64_t module_hash_ = 0;
    std::unordered_map<std::string, FunctionProfile> functions_;
};

} // namespace brass::pgo
