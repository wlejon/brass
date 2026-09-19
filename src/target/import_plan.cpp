#include "import_plan.hpp"

#include <algorithm>
#include <map>
#include <set>
#include <sstream>

namespace brass::target::imports {

const Imported* Plan::find(std::string_view name) const {
    for (const auto& s : symbols) {
        if (s.name == name) return &s;
    }
    return nullptr;
}

std::vector<std::vector<size_t>> Plan::by_library() const {
    std::vector<std::vector<size_t>> groups(libraries.size());
    for (const auto& s : symbols) groups[s.library].push_back(s.index);
    return groups;
}

bool is_defined_in(const object::ObjectFile& obj, std::string_view name) {
    if (const auto* sym = obj.find_symbol(name)) {
        if (sym->section_index >= 0) return true;
    }
    for (const auto& sec : obj.sections) {
        if (sec.name == name) return true;
    }
    return false;
}

bool plan(const object::ObjectFile& obj, const std::vector<std::string_view>& section_names,
          const std::vector<ImportLibrary>& libraries, Plan& out, std::string& error) {
    out = Plan{};

    // symbol -> library index, first library wins so a symbol listed twice is
    // not an error here but a fixed choice.
    std::map<std::string, size_t, std::less<>> provider;
    for (size_t i = 0; i < libraries.size(); ++i) {
        for (const auto& name : libraries[i].symbols) provider.emplace(name, i);
    }

    std::set<std::string> seen;
    std::vector<std::pair<size_t, std::string>> found;   // (library, name)
    for (const auto& sec : obj.sections) {
        if (std::find(section_names.begin(), section_names.end(), sec.name) == section_names.end()) {
            continue;
        }
        for (const auto& r : sec.relocations) {
            if (is_defined_in(obj, r.symbol_name)) continue;
            if (!seen.insert(r.symbol_name).second) continue;
            auto it = provider.find(r.symbol_name);
            if (it == provider.end()) {
                std::ostringstream msg;
                msg << "unresolved symbol '" << r.symbol_name << "' (referenced from " << sec.name
                    << "+0x" << std::hex << r.offset << std::dec << ") is not defined by the object";
                if (libraries.empty()) {
                    msg << " and no import library was given";
                } else {
                    msg << " and is not provided by ";
                    for (size_t i = 0; i < libraries.size(); ++i) {
                        msg << (i ? ", " : "") << libraries[i].library;
                    }
                }
                error = msg.str();
                return false;
            }
            found.emplace_back(it->second, r.symbol_name);
        }
    }

    std::sort(found.begin(), found.end());
    std::vector<size_t> library_slot(libraries.size(), SIZE_MAX);
    for (const auto& [lib, name] : found) {
        if (library_slot[lib] == SIZE_MAX) {
            library_slot[lib] = out.libraries.size();
            out.libraries.push_back(libraries[lib].library);
        }
        Imported imp;
        imp.name = name;
        imp.library = library_slot[lib];
        imp.index = out.symbols.size();
        out.symbols.push_back(std::move(imp));
    }
    return true;
}

} // namespace brass::target::imports
