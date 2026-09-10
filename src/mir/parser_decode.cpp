#include "parser_decode.hpp"
#include "parser_vec.hpp"
#include "parser_coro.hpp"
#include <unordered_map>

namespace brass {

Type parse_type_from_string(std::string_view s) {
    if (s == "i32") return Type::i32();
    if (s == "i64") return Type::i64();
    if (s == "f32") return Type::f32();
    if (s == "f64") return Type::f64();
    if (s == "ptr") return Type::ptr();
    if (s == "gcref") return Type::gcref();
    if (s == "void") return Type::void_type();
    if (s == "f32x4") return Type::f32x4();
    if (s == "f64x2") return Type::f64x2();
    if (s == "i32x4") return Type::i32x4();
    if (s == "i64x2") return Type::i64x2();
    if (s == "f32x8") return Type::f32x8();
    if (s == "f64x4") return Type::f64x4();
    if (s == "i32x8") return Type::i32x8();
    if (s == "i64x4") return Type::i64x4();
    return Type::void_type();
}

bool is_identifier_or_keyword(TokenKind k) noexcept {
    return k == TokenKind::Ident || (k >= TokenKind::Kw_func && k <= TokenKind::Kw_i64x4);
}

bool decode_opcode_string(std::string_view str, Opcode& op, Type& type_suffix, Type& mem_type) {
    type_suffix = Type::void_type();
    mem_type = Type::void_type();

    // Check exact matches first
    if (str == "func" || str == "extern" || str == "module" || str == "resume_table" || str == "entry") {
        return false;
    }

    if (decode_vector_opcode(str, op, type_suffix, mem_type)) {
        return true;
    }

    if (decode_coro_opcode(str, op, type_suffix)) {
        return true;
    }

    if (str == "ret") { op = Opcode::ret; return true; }
    if (str == "br") { op = Opcode::br; return true; }
    if (str == "br_if") { op = Opcode::br_if; return true; }
    if (str == "switch") { op = Opcode::switch_; return true; }
    if (str == "guard") { op = Opcode::guard; return true; }
    if (str == "resume_point") { op = Opcode::resume_point; return true; }
    if (str == "osr_entry") { op = Opcode::osr_entry; return true; }
    if (str == "safepoint") { op = Opcode::safepoint; return true; }
    if (str == "write_barrier") { op = Opcode::write_barrier; return true; }
    if (str == "unreachable") { op = Opcode::unreachable; return true; }
    if (str == "call") { op = Opcode::call; return true; }
    if (str == "throw") { op = Opcode::throw_; return true; }
    if (str == "resume") { op = Opcode::resume; return true; }
    if (str == "landing_pad" || str == "landingpad") { op = Opcode::landing_pad; return true; }
    if (str == "invoke") { op = Opcode::invoke; return true; }

    // Special conversions
    if (str == "sitofp.f64.i32" || str == "sitofp_f64_i32") { op = Opcode::sitofp_f64_i32; type_suffix = Type::f64(); return true; }
    if (str == "sitofp.f64.i64" || str == "sitofp_f64_i64") { op = Opcode::sitofp_f64_i64; type_suffix = Type::f64(); return true; }
    if (str == "sitofp.f64") { op = Opcode::sitofp_f64_i32; type_suffix = Type::f64(); return true; }

    if (str == "bitcast.i64.f64" || str == "bitcast_i64_f64") { op = Opcode::bitcast_i64_f64; type_suffix = Type::i64(); return true; }
    if (str == "bitcast.f64.i64" || str == "bitcast_f64_i64") { op = Opcode::bitcast_f64_i64; type_suffix = Type::f64(); return true; }

    if (str == "fma.f32" || str == "fma_f32") { op = Opcode::fma_f32; type_suffix = Type::f32(); return true; }
    if (str == "fma.f64" || str == "fma_f64") { op = Opcode::fma_f64; type_suffix = Type::f64(); return true; }

    // Check dot separation: base.suffix or underscore separation
    std::string_view base = str;
    std::string_view suffix = "";
    size_t dot_pos = str.find('.');
    if (dot_pos != std::string_view::npos) {
        base = str.substr(0, dot_pos);
        suffix = str.substr(dot_pos + 1);
    } else {
        size_t underscore_pos = str.rfind('_');
        if (underscore_pos != std::string_view::npos) {
            std::string_view potential_suffix = str.substr(underscore_pos + 1);
            if (potential_suffix == "i32" || potential_suffix == "i64" || potential_suffix == "f64" ||
                potential_suffix == "ptr" || potential_suffix == "gcref" || potential_suffix == "void") {
                base = str.substr(0, underscore_pos);
                suffix = potential_suffix;
            }
        }
    }

    if (!suffix.empty()) {
        type_suffix = parse_type_from_string(suffix);
    }

    // Constants
    if (base == "iconst") {
        if (type_suffix == Type::i64()) { op = Opcode::iconst_i64; return true; }
        op = Opcode::iconst_i32;
        type_suffix = Type::i32();
        return true;
    }
    if (base == "fconst") { op = Opcode::fconst_f64; type_suffix = Type::f64(); return true; }
    if (base == "patchable_const") {
        if (type_suffix == Type::i64()) { op = Opcode::patchable_const_i64; return true; }
        op = Opcode::patchable_const_i32;
        type_suffix = Type::i32();
        return true;
    }

    // Conversions
    if (base == "sext") { op = Opcode::sext_i64; type_suffix = Type::i64(); return true; }
    if (base == "zext") { op = Opcode::zext_i64; type_suffix = Type::i64(); return true; }
    if (base == "trunc") { op = Opcode::trunc_i32; type_suffix = Type::i32(); return true; }
    if (base == "fptosi") {
        if (type_suffix == Type::i64()) { op = Opcode::fptosi_i64; return true; }
        op = Opcode::fptosi_i32;
        type_suffix = Type::i32();
        return true;
    }

    // Arithmetic / Bitwise / Comparison / Overflow
    static const std::unordered_map<std::string_view, Opcode> op_map = {
        {"add", Opcode::add}, {"sub", Opcode::sub}, {"mul", Opcode::mul},
        {"sdiv", Opcode::sdiv}, {"udiv", Opcode::udiv}, {"smod", Opcode::smod}, {"umod", Opcode::umod},
        {"neg", Opcode::neg}, {"and", Opcode::and_}, {"or", Opcode::or_}, {"xor", Opcode::xor_},
        {"shl", Opcode::shl}, {"lshr", Opcode::lshr}, {"ashr", Opcode::ashr}, {"not", Opcode::not_},
        {"clz", Opcode::clz}, {"ctz", Opcode::ctz}, {"popcnt", Opcode::popcnt},
        {"eq", Opcode::eq}, {"ne", Opcode::ne}, {"slt", Opcode::slt}, {"ult", Opcode::ult},
        {"sle", Opcode::sle}, {"ule", Opcode::ule}, {"sgt", Opcode::sgt}, {"ugt", Opcode::ugt},
        {"sge", Opcode::sge}, {"uge", Opcode::uge},
        {"sadd_overflow", Opcode::sadd_overflow}, {"ssub_overflow", Opcode::ssub_overflow},
        {"smul_overflow", Opcode::smul_overflow}, {"uadd_overflow", Opcode::uadd_overflow},
        {"usub_overflow", Opcode::usub_overflow}, {"umul_overflow", Opcode::umul_overflow},
        {"switch", Opcode::switch_}
    };
    if (base == "fma") {
        if (type_suffix == Type::f32()) { op = Opcode::fma_f32; return true; }
        op = Opcode::fma_f64;
        type_suffix = Type::f64();
        return true;
    }
    auto it = op_map.find(base);
    if (it != op_map.end()) { op = it->second; return true; }

    // Selection
    if (base == "select") { op = Opcode::select; return true; }

    // Memory
    if (base == "load") {
        op = Opcode::load;
        mem_type = type_suffix;
        return true;
    }
    if (base == "store") {
        op = Opcode::store;
        mem_type = type_suffix;
        return true;
    }
    if (base == "load_indexed") {
        op = Opcode::load_indexed;
        mem_type = type_suffix;
        return true;
    }
    if (base == "store_indexed") {
        op = Opcode::store_indexed;
        mem_type = type_suffix;
        return true;
    }

    // Calls
    if (base == "call") { op = Opcode::call; return true; }
    if (base == "call_indirect") { op = Opcode::call_indirect; return true; }
    if (base == "patchable_call") { op = Opcode::patchable_call; return true; }
    if (base == "invoke") { op = Opcode::invoke; return true; }
    if (base == "landing_pad" || base == "landingpad") { op = Opcode::landing_pad; return true; }

    return false;
}

} // namespace brass
