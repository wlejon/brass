#pragma once

#include <brass/mir/sroa.hpp>
#include <brass/mir/function.hpp>
#include <brass/mir/dominators.hpp>
#include <brass/mir/builder.hpp>
#include <brass/mir/escape_analysis.hpp>
#include <cstdint>
#include <map>
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
    bool process_candidate(const Value* alloc_val, const EscapeAnalysis& ea);
    Value* get_or_create_zero_constant(Builder& b, BasicBlock* entry, Type type);

    Function& fn_;
    SroaOptions options_;
    SroaStats* stats_ = nullptr;
    std::unordered_map<uint8_t, Value*> zero_constants_;
};

} // namespace brass
