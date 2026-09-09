#pragma once

#include <brass/mir/instruction.hpp>
#include <brass/mir/types.hpp>
#include <brass/mir/opcodes.hpp>
#include <vector>
#include <unordered_map>
#include <string_view>
#include <cstring>
#include <cstdint>

namespace brass {

bool is_pure_gvn_op(const Instruction* inst) noexcept;
bool is_commutative_op(Opcode op) noexcept;

struct GvnExpression {
    Opcode opcode = Opcode::unreachable;
    Type type = Type::void_type();
    const Value* op0 = nullptr;
    const Value* op1 = nullptr;
    const Value* op2 = nullptr;
    uint64_t imm_bits = 0;
    int32_t offset = 0;
    uint8_t scale = 1;
    std::string_view symbol;

    static GvnExpression from_instruction(const Instruction* inst);

    bool operator==(const GvnExpression& other) const noexcept;
};

struct GvnExprHash {
    size_t operator()(const GvnExpression& expr) const noexcept;
};

struct AvailableLoadKey {
    const Value* base = nullptr;
    int32_t offset = 0;
    Type memory_type = Type::void_type();
    uint32_t memory_access_id = 0;

    bool operator==(const AvailableLoadKey& o) const noexcept {
        return base == o.base && offset == o.offset &&
               memory_type == o.memory_type &&
               memory_access_id == o.memory_access_id;
    }
};

struct AvailableLoadHash {
    size_t operator()(const AvailableLoadKey& k) const noexcept {
        size_t h = std::hash<const void*>()(k.base);
        h = h * 31 + static_cast<size_t>(k.offset);
        h = h * 31 + static_cast<size_t>(k.memory_type.kind());
        h = h * 31 + static_cast<size_t>(k.memory_access_id);
        return h;
    }
};

class GvnTable {
public:
    void enter_scope();
    void exit_scope();

    Value* lookup_expression(const GvnExpression& expr) const;
    void insert_expression(const GvnExpression& expr, Value* val);

    Value* lookup_load(const AvailableLoadKey& key) const;
    void insert_load(const AvailableLoadKey& key, Value* val);

private:
    struct ScopeFrame {
        std::vector<GvnExpression> exprs;
        std::vector<AvailableLoadKey> loads;
    };

    std::vector<ScopeFrame> scopes_;
    std::unordered_map<GvnExpression, Value*, GvnExprHash> expr_map_;
    std::unordered_map<AvailableLoadKey, Value*, AvailableLoadHash> load_map_;
};

} // namespace brass
