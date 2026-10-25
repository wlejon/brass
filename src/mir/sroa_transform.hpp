#pragma once

#include <brass/mir/sroa.hpp>
#include <brass/mir/function.hpp>
#include <brass/mir/dominators.hpp>
#include <brass/mir/builder.hpp>
#include <brass/mir/escape_analysis.hpp>
#include <cstdint>
#include <map>
#include <optional>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace brass {

struct FieldInfo {
    int32_t offset = 0;
    Type type = Type::void_type();

    bool operator==(const FieldInfo& o) const noexcept {
        return offset == o.offset && type == o.type;
    }
};

class SroaTransformer {
public:
    SroaTransformer(Function& fn, const SroaOptions& options, SroaStats* stats);
    bool run();

private:
    bool process_candidate(const Value* alloc_val);
    Value* get_or_create_zero_constant(Builder& b, BasicBlock* entry, Type type);

    Function& fn_;
    SroaOptions options_;
    SroaStats* stats_ = nullptr;
    std::unordered_map<uint8_t, Value*> zero_constants_;
    // Promotion adds block parameters and edge arguments but never a block or
    // an edge, so the dominator tree and frontiers hold for the whole run.
    std::optional<DominatorTree> dom_;
    std::unordered_map<const BasicBlock*, std::vector<BasicBlock*>> df_;
};

} // namespace brass
