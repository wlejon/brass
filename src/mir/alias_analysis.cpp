#include <brass/mir/alias_analysis.hpp>
#include <brass/mir/escape_analysis.hpp>
#include <ostream>
#include <cstdlib>

namespace brass {

std::string_view alias_result_name(AliasResult result) noexcept {
    switch (result) {
        case AliasResult::NoAlias: return "NoAlias";
        case AliasResult::MayAlias: return "MayAlias";
        case AliasResult::MustAlias: return "MustAlias";
    }
    return "Unknown";
}

std::ostream& operator<<(std::ostream& os, AliasResult result) {
    return os << alias_result_name(result);
}

namespace {

bool get_const_integer_val(const Value* val, int64_t& out_c) {
    if (!val || !val->is_instruction()) return false;
    const Instruction* def = val->defining_instruction();
    if (!def) return false;
    if (def->opcode() == Opcode::iconst_i32) {
        out_c = static_cast<int64_t>(def->imm_i32());
        return true;
    }
    if (def->opcode() == Opcode::iconst_i64) {
        out_c = def->imm_i64();
        return true;
    }
    return false;
}

bool are_incompatible_memory_types(Type t1, Type t2) noexcept {
    if (t1.is_void() || t2.is_void()) return false;
    if (t1 == t2) return false;

    // Pointer vs GC reference heap disambiguation
    if ((t1.is_gcref() && t2.is_pointer()) || (t1.is_pointer() && t2.is_gcref())) {
        return true;
    }

    // Floating-point vs Integer / Reference
    if ((t1.is_float() && !t2.is_float()) || (!t1.is_float() && t2.is_float())) {
        return true;
    }

    // Vector vs Scalar
    if (t1.is_vector() != t2.is_vector()) {
        return true;
    }

    // Different sized scalars
    if (t1.size_in_bytes() != t2.size_in_bytes()) {
        return true;
    }

    return false;
}

} // namespace

AliasAnalysis::AliasAnalysis(const Function& fn)
    : fn_(&fn), external_ea_(nullptr), owned_ea_(std::make_unique<EscapeAnalysis>(fn)) {}

AliasAnalysis::AliasAnalysis(const Function& fn, const EscapeAnalysis* ea)
    : fn_(&fn), external_ea_(ea) {
    if (!external_ea_) {
        owned_ea_ = std::make_unique<EscapeAnalysis>(fn);
    }
}

AliasAnalysis::~AliasAnalysis() = default;
AliasAnalysis::AliasAnalysis(AliasAnalysis&&) noexcept = default;
AliasAnalysis& AliasAnalysis::operator=(AliasAnalysis&&) noexcept = default;

const EscapeAnalysis* AliasAnalysis::escape_analysis() const noexcept {
    return external_ea_ ? external_ea_ : owned_ea_.get();
}

bool AliasAnalysis::is_allocation(const Value* val) const {
    if (!val) return false;
    const EscapeAnalysis* ea = escape_analysis();
    if (ea && ea->is_allocation(val)) return true;
    if (val->is_instruction()) {
        const Instruction* inst = val->defining_instruction();
        if (inst && inst->opcode() == Opcode::call) {
            return is_allocation_callee(inst->symbol());
        }
    }
    return false;
}

const Value* AliasAnalysis::get_underlying_base(const Value* ptr, int64_t& out_offset) const {
    out_offset = 0;
    const Value* cur = ptr;

    while (cur && cur->is_instruction()) {
        const Instruction* inst = cur->defining_instruction();
        if (!inst) break;

        if (inst->opcode() == Opcode::add) {
            int64_t c = 0;
            if (get_const_integer_val(inst->operand(1), c)) {
                out_offset += c;
                cur = inst->operand(0);
                continue;
            } else if (get_const_integer_val(inst->operand(0), c)) {
                out_offset += c;
                cur = inst->operand(1);
                continue;
            }
        } else if (inst->opcode() == Opcode::sub) {
            int64_t c = 0;
            if (get_const_integer_val(inst->operand(1), c)) {
                out_offset -= c;
                cur = inst->operand(0);
                continue;
            }
        }
        break;
    }
    return cur;
}

bool AliasAnalysis::is_distinct_allocation(const Value* base1, const Value* base2) const {
    if (!base1 || !base2 || base1 == base2) return false;
    return is_allocation(base1) && is_allocation(base2);
}

bool AliasAnalysis::is_non_escaping(const Value* base) const {
    if (!base) return false;
    const EscapeAnalysis* ea = escape_analysis();
    if (!ea) return false;
    return ea->get_escape_state(base) == EscapeState::NoEscape;
}

AliasResult AliasAnalysis::alias(const Value* ptr1, int32_t off1, const Value* ptr2, int32_t off2) const {
    return alias(ptr1, off1, Type::void_type(), ptr2, off2, Type::void_type());
}

AliasResult AliasAnalysis::alias(
    const Value* ptr1, int32_t off1, Type type1,
    const Value* ptr2, int32_t off2, Type type2
) const {
    if (!ptr1 || !ptr2) return AliasResult::MayAlias;

    // 1. Pointer vs GC reference heap disambiguation
    bool is_gc1 = ptr1->type().is_gcref();
    bool is_gc2 = ptr2->type().is_gcref();
    if (is_gc1 != is_gc2) {
        return AliasResult::NoAlias;
    }

    // 2. Type-based alias disambiguation (incompatible memory types)
    if (are_incompatible_memory_types(type1, type2)) {
        return AliasResult::NoAlias;
    }

    // 3. Extract underlying base allocations and accumulated constant offsets
    int64_t accum1 = 0;
    int64_t accum2 = 0;
    const Value* base1 = get_underlying_base(ptr1, accum1);
    const Value* base2 = get_underlying_base(ptr2, accum2);

    if (base1 && base2) {
        // Base heap type check
        if (base1->type().is_gcref() != base2->type().is_gcref()) {
            return AliasResult::NoAlias;
        }

        int64_t total_off1 = static_cast<int64_t>(off1) + accum1;
        int64_t total_off2 = static_cast<int64_t>(off2) + accum2;

        if (base1 == base2) {
            if (total_off1 == total_off2) {
                return AliasResult::MustAlias;
            }
            // Same base pointer with provably distinct constant byte offsets
            uint32_t sz1 = type1.is_void() ? 4U : static_cast<uint32_t>(type1.size_in_bytes());
            uint32_t sz2 = type2.is_void() ? 4U : static_cast<uint32_t>(type2.size_in_bytes());
            if (sz1 == 0) sz1 = 4U;
            if (sz2 == 0) sz2 = 4U;

            if (total_off1 + sz1 <= total_off2 || total_off2 + sz2 <= total_off1 || total_off1 != total_off2) {
                return AliasResult::NoAlias;
            }
            return AliasResult::MayAlias;
        }

        // Distinct base allocations or distinct non-escaping objects -> NoAlias
        if (is_distinct_allocation(base1, base2)) {
            return AliasResult::NoAlias;
        }

        if (is_non_escaping(base1) || is_non_escaping(base2)) {
            return AliasResult::NoAlias;
        }
    }

    return AliasResult::MayAlias;
}

bool AliasAnalysis::can_clobber(const Instruction* write_inst, const Instruction* read_inst) const {
    if (!write_inst || !read_inst) return false;

    Opcode w_op = write_inst->opcode();
    bool writes_mem = (w_op == Opcode::store || w_op == Opcode::store_indexed ||
                       w_op == Opcode::vstore || is_call(w_op));
    if (!writes_mem) return false;

    Opcode r_op = read_inst->opcode();
    bool reads_mem = (r_op == Opcode::load || r_op == Opcode::load_indexed ||
                      r_op == Opcode::vload || is_call(r_op));
    if (!reads_mem) return false;

    // Allocation calls (brass_gc_alloc, malloc) produce fresh memory and do not clobber existing memory
    if (is_call(w_op)) {
        if (is_allocation_callee(write_inst->symbol())) {
            return false;
        }

        // Check if read is from a non-escaping allocation not passed to this call
        if (r_op == Opcode::load || r_op == Opcode::load_indexed || r_op == Opcode::vload) {
            const Value* r_ptr = read_inst->operand(0);
            int64_t dummy = 0;
            const Value* r_base = get_underlying_base(r_ptr, dummy);
            if (is_non_escaping(r_base)) {
                bool passed_as_arg = false;
                for (const Value* op : write_inst->operands()) {
                    int64_t op_off = 0;
                    if (get_underlying_base(op, op_off) == r_base) {
                        passed_as_arg = true;
                        break;
                    }
                }
                if (!passed_as_arg) {
                    return false;
                }
            }
        }
        return true;
    }

    // Direct store vs load
    if (w_op == Opcode::store || w_op == Opcode::vstore) {
        if (r_op == Opcode::load || r_op == Opcode::vload) {
            const Value* w_ptr = write_inst->operand(0);
            int32_t w_off = write_inst->offset();
            Type w_type = write_inst->memory_type();

            const Value* r_ptr = read_inst->operand(0);
            int32_t r_off = read_inst->offset();
            Type r_type = read_inst->memory_type();

            AliasResult res = alias(w_ptr, w_off, w_type, r_ptr, r_off, r_type);
            return res != AliasResult::NoAlias;
        }

        if (r_op == Opcode::load_indexed) {
            AliasResult res = alias(write_inst->operand(0), write_inst->offset(),
                                    read_inst->operand(0), read_inst->offset());
            return res != AliasResult::NoAlias;
        }
        return true;
    }

    // Store indexed vs loads
    if (w_op == Opcode::store_indexed) {
        if (r_op == Opcode::load || r_op == Opcode::vload) {
            AliasResult res = alias(write_inst->operand(0), write_inst->offset(),
                                    read_inst->operand(0), read_inst->offset());
            return res != AliasResult::NoAlias;
        }

        if (r_op == Opcode::load_indexed) {
            AliasResult base_res = alias(write_inst->operand(0), 0, read_inst->operand(0), 0);
            if (base_res == AliasResult::NoAlias) return false;

            if (write_inst->operand(0) == read_inst->operand(0) &&
                write_inst->operand(1) == read_inst->operand(1) &&
                write_inst->scale() == read_inst->scale()) {
                return write_inst->offset() == read_inst->offset();
            }
            return true;
        }
        return true;
    }

    return true;
}

} // namespace brass
