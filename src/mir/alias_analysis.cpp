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

// MIR memory is untyped: the same bytes may be read as i32, i64, f64, a
// pointer or a GC reference (bronze reads a header word both as i32 and as
// i64). The access type therefore says how many bytes an access touches and
// nothing about which bytes; it never proves two accesses disjoint on its own.
// 0 means the size is unknown.
uint64_t access_size(Type t) noexcept {
    return t.is_void() ? 0 : static_cast<uint64_t>(t.size_in_bytes());
}

// One memory access in address form: `base + offset + index * scale`, over
// `size` bytes. `index` is null when the address has no variable part.
struct Access {
    const Value* ptr = nullptr;
    int64_t offset = 0;
    const Value* index = nullptr;
    int64_t scale = 0;
    uint64_t size = 0;
};

bool is_memory_reader(Opcode op) noexcept {
    return op == Opcode::load || op == Opcode::load_indexed || op == Opcode::vload ||
           is_call(op) || is_coro_op(op);
}

bool is_memory_writer(Opcode op) noexcept {
    return op == Opcode::store || op == Opcode::store_indexed || op == Opcode::vstore ||
           is_call(op) || is_coro_op(op);
}

bool is_plain_access(Opcode op) noexcept {
    return op == Opcode::load || op == Opcode::load_indexed || op == Opcode::vload ||
           op == Opcode::store || op == Opcode::store_indexed || op == Opcode::vstore;
}

Access describe(const Instruction* inst) {
    Access a;
    a.ptr = inst->operand(0);
    a.offset = inst->offset();
    a.size = access_size(inst->memory_type());
    Opcode op = inst->opcode();
    if (op == Opcode::load_indexed || op == Opcode::store_indexed) {
        const Value* idx = inst->operand(1);
        const int64_t scale = static_cast<int64_t>(inst->scale());
        int64_t c = 0;
        if (get_const_integer_val(idx, c)) {
            a.offset += c * scale;
        } else {
            a.index = idx;
            a.scale = scale;
        }
    }
    return a;
}

// Same-base comparison of two byte ranges at known start offsets.
AliasResult compare_ranges(int64_t off1, uint64_t size1, int64_t off2, uint64_t size2) noexcept {
    if (off1 == off2) return AliasResult::MustAlias;
    if (size1 != 0 && size2 != 0) {
        if (off1 + static_cast<int64_t>(size1) <= off2 || off2 + static_cast<int64_t>(size2) <= off1) {
            return AliasResult::NoAlias;
        }
    }
    return AliasResult::MayAlias;
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
        if (inst) {
            if (inst->opcode() == Opcode::alloca_) return true;
            if (inst->opcode() == Opcode::call) {
                return is_allocation_call(inst);
            }
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
    if (is_allocation(base1) && is_allocation(base2)) return true;
    if (fn_ && fn_->entry_block()) {
        const BasicBlock* entry = fn_->entry_block();
        if (base1->is_block_param() && base2->is_block_param() &&
            base1->defining_block() == entry && base2->defining_block() == entry &&
            base1 != base2) {
            bool noalias1 = base1->is_noalias() || fn_->is_param_noalias(base1->param_index());
            bool noalias2 = base2->is_noalias() || fn_->is_param_noalias(base2->param_index());
            if (noalias1 || noalias2) {
                return true;
            }
            return false;
        }
    }
    return false;
}

bool AliasAnalysis::is_non_escaping(const Value* base) const {
    // Only a fresh allocation has an identity escape analysis can vouch for;
    // a select, block parameter or loaded pointer may name any object.
    if (!base || !is_allocation(base)) return false;
    const EscapeAnalysis* ea = escape_analysis();
    if (!ea) return false;
    return ea->get_escape_state(base) == EscapeState::NoEscape;
}

bool AliasAnalysis::is_global_or_external_arg(const Value* base) const {
    if (!base) return false;
    if (fn_ && fn_->entry_block()) {
        const BasicBlock* entry = fn_->entry_block();
        if (base->is_block_param() && base->defining_block() == entry) {
            return true;
        }
    }
    if (base->is_instruction()) {
        const Instruction* inst = base->defining_instruction();
        if (inst) {
            Opcode op = inst->opcode();
            if (op == Opcode::pinned_tls_read || op == Opcode::func_addr) {
                return true;
            }
        }
    }
    const EscapeAnalysis* ea = escape_analysis();
    if (ea && ea->get_escape_state(base) == EscapeState::GlobalEscape && !is_allocation(base)) {
        return true;
    }
    return false;
}

AliasResult AliasAnalysis::alias(const Value* ptr1, const Value* ptr2) const {
    return alias(ptr1, 0, Type::void_type(), ptr2, 0, Type::void_type());
}

AliasResult AliasAnalysis::alias(const Value* ptr1, int32_t off1, const Value* ptr2, int32_t off2) const {
    return alias(ptr1, off1, Type::void_type(), ptr2, off2, Type::void_type());
}

bool AliasAnalysis::are_distinct_objects(const Value* ptr1, const Value* ptr2) const {
    auto separated = [&](const Value* b1, const Value* b2) {
        if (!b1 || !b2 || b1 == b2) return false;
        if (is_distinct_allocation(b1, b2)) return true;
        if (is_non_escaping(b1) && is_global_or_external_arg(b2)) return true;
        if (is_non_escaping(b2) && is_global_or_external_arg(b1)) return true;
        return false;
    };

    int64_t ignored1 = 0;
    int64_t ignored2 = 0;
    if (separated(get_underlying_base(ptr1, ignored1), get_underlying_base(ptr2, ignored2))) return true;

    // Pointer arithmetic by a variable amount stays inside the object it
    // started from, so the objects the two addresses were derived from decide.
    auto origin = [](const Value* ptr) -> const Value* {
        const Value* cur = ptr;
        while (cur && cur->is_instruction()) {
            const Instruction* inst = cur->defining_instruction();
            if (!inst) break;
            auto is_addr = [](const Value* v) { return v && v->type().is_pointer_or_gcref(); };
            if (inst->opcode() == Opcode::add) {
                if (is_addr(inst->operand(0))) { cur = inst->operand(0); continue; }
                if (is_addr(inst->operand(1))) { cur = inst->operand(1); continue; }
            } else if (inst->opcode() == Opcode::sub) {
                if (is_addr(inst->operand(0))) { cur = inst->operand(0); continue; }
            }
            break;
        }
        return cur;
    };
    return separated(origin(ptr1), origin(ptr2));
}

AliasResult AliasAnalysis::alias(
    const Value* ptr1, int32_t off1, Type type1,
    const Value* ptr2, int32_t off2, Type type2
) const {
    if (!ptr1 || !ptr2) return AliasResult::MayAlias;

    int64_t accum1 = 0;
    int64_t accum2 = 0;
    const Value* base1 = get_underlying_base(ptr1, accum1);
    const Value* base2 = get_underlying_base(ptr2, accum2);

    if (base1 && base1 == base2) {
        return compare_ranges(static_cast<int64_t>(off1) + accum1, access_size(type1),
                              static_cast<int64_t>(off2) + accum2, access_size(type2));
    }

    return are_distinct_objects(ptr1, ptr2) ? AliasResult::NoAlias : AliasResult::MayAlias;
}

bool AliasAnalysis::can_clobber(const Instruction* write_inst, const Instruction* read_inst) const {
    if (!write_inst || !read_inst) return false;

    const Opcode w_op = write_inst->opcode();
    const Opcode r_op = read_inst->opcode();
    if (!is_memory_writer(w_op) || !is_memory_reader(r_op)) return false;

    if (is_call(w_op)) {
        // A call cannot reach an allocation whose address never left this
        // function, unless the address is one of its own arguments.
        if (is_plain_access(r_op)) {
            int64_t ignored = 0;
            const Value* r_base = get_underlying_base(read_inst->operand(0), ignored);
            if (is_non_escaping(r_base)) {
                for (const Value* op : write_inst->operands()) {
                    int64_t op_off = 0;
                    if (get_underlying_base(op, op_off) == r_base) return true;
                }
                return false;
            }
        }
        return true;
    }

    if (!is_plain_access(w_op) || !is_plain_access(r_op)) return true;

    const Access w = describe(write_inst);
    const Access r = describe(read_inst);

    // Without a variable part, or with the same variable index scaled the
    // same way (which moves both addresses together), the constant parts of
    // two addresses off one base decide.
    const bool same_variable_part = w.index == r.index && (!w.index || w.scale == r.scale);
    if (same_variable_part) {
        int64_t acc_w = 0;
        int64_t acc_r = 0;
        const Value* bw = get_underlying_base(w.ptr, acc_w);
        const Value* br = get_underlying_base(r.ptr, acc_r);
        if (bw && bw == br) {
            return compare_ranges(w.offset + acc_w, w.size, r.offset + acc_r, r.size) != AliasResult::NoAlias;
        }
    }

    // A variable index can reach any byte of the object; only distinct
    // objects keep the accesses apart.
    return !are_distinct_objects(w.ptr, r.ptr);
}

} // namespace brass
