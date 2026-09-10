#include <brass/debug/source_loc.hpp>

namespace brass {

namespace {
const std::string kEmptyString;
}

uint32_t DebugContext::get_or_add_file(const std::string& path) {
    if (path.empty()) return 0;
    auto it = file_to_id_.find(path);
    if (it != file_to_id_.end()) {
        return it->second;
    }
    files_.push_back(path);
    uint32_t new_id = static_cast<uint32_t>(files_.size());
    file_to_id_[path] = new_id;
    return new_id;
}

uint32_t DebugContext::get_file_id(const std::string& path) const {
    auto it = file_to_id_.find(path);
    if (it != file_to_id_.end()) {
        return it->second;
    }
    return 0;
}

const std::string& DebugContext::get_file(uint32_t file_id) const {
    if (file_id == 0 || file_id > files_.size()) {
        return kEmptyString;
    }
    return files_[file_id - 1];
}

uint32_t DebugContext::record_inlined_scope(std::string callee, DebugLoc callsite, uint32_t parent_id) {
    inlined_scopes_.emplace_back(std::move(callee), callsite, parent_id);
    return static_cast<uint32_t>(inlined_scopes_.size());
}

const InlinedScope* DebugContext::get_inlined_scope(uint32_t inlined_id) const {
    if (inlined_id == 0 || inlined_id > inlined_scopes_.size()) {
        return nullptr;
    }
    return &inlined_scopes_[inlined_id - 1];
}

uint32_t DebugContext::wrap_inlined_scope(uint32_t existing_inlined_id, uint32_t outer_inlined_id) {
    if (existing_inlined_id == 0) return outer_inlined_id;
    if (outer_inlined_id == 0) return existing_inlined_id;
    const InlinedScope* existing = get_inlined_scope(existing_inlined_id);
    if (!existing) return outer_inlined_id;

    uint32_t new_parent_id = outer_inlined_id;
    if (existing->parent_inlined_at_id != 0) {
        new_parent_id = wrap_inlined_scope(existing->parent_inlined_at_id, outer_inlined_id);
    }
    return record_inlined_scope(existing->callee_name, existing->callsite_loc, new_parent_id);
}

void DebugContext::clear() {
    files_.clear();
    file_to_id_.clear();
    inlined_scopes_.clear();
}

} // namespace brass
