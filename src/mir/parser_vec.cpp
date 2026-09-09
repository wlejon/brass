#include "parser_vec.hpp"

namespace brass {

static Type parse_type_str(std::string_view s) {
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
    return Type::void_type();
}

bool decode_vector_opcode(
    std::string_view str,
    Opcode& op,
    Type& type_suffix,
    Type& mem_type
) {
    std::string_view base = str;
    std::string_view suffix = "";
    size_t dot_pos = str.find('.');
    if (dot_pos != std::string_view::npos) {
        base = str.substr(0, dot_pos);
        suffix = str.substr(dot_pos + 1);
    }
    if (!suffix.empty()) {
        type_suffix = parse_type_str(suffix);
    }

    if (base == "vadd") { op = Opcode::vadd; return true; }
    if (base == "vsub") { op = Opcode::vsub; return true; }
    if (base == "vmul") { op = Opcode::vmul; return true; }
    if (base == "vdiv") { op = Opcode::vdiv; return true; }
    if (base == "vneg") { op = Opcode::vneg; return true; }
    if (base == "vmin") { op = Opcode::vmin; return true; }
    if (base == "vmax") { op = Opcode::vmax; return true; }
    if (base == "vsqrt") { op = Opcode::vsqrt; return true; }
    if (base == "vand") { op = Opcode::vand; return true; }
    if (base == "vor") { op = Opcode::vor; return true; }
    if (base == "vxor") { op = Opcode::vxor; return true; }
    if (base == "vnot") { op = Opcode::vnot; return true; }

    if (base == "vload") {
        op = Opcode::vload;
        mem_type = type_suffix;
        return true;
    }
    if (base == "vstore") {
        op = Opcode::vstore;
        mem_type = type_suffix;
        return true;
    }
    if (base == "vbroadcast") { op = Opcode::vbroadcast; return true; }
    if (base == "vextract_lane") { op = Opcode::vextract_lane; return true; }
    if (base == "vinsert_lane") { op = Opcode::vinsert_lane; return true; }
    if (base == "vshuffle") { op = Opcode::vshuffle; return true; }
    if (base == "vzero") { op = Opcode::vzero; return true; }

    return false;
}

bool parse_vector_instruction(
    ParserVecContext& ctx,
    Opcode op,
    Type type_annotation,
    Type type_suffix,
    Type mem_type,
    Builder& b,
    Value*& res_val,
    Instruction*& res_inst
) {
    switch (op) {
        case Opcode::vadd:
        case Opcode::vsub:
        case Opcode::vmul:
        case Opcode::vdiv:
        case Opcode::vmin:
        case Opcode::vmax:
        case Opcode::vand:
        case Opcode::vor:
        case Opcode::vxor: {
            Value* lhs = ctx.parse_val();
            if (!lhs) return false;
            if (!ctx.expect(TokenKind::Comma, "','")) return false;
            Value* rhs = ctx.parse_val();
            if (!rhs) return false;

            Type op_t = type_annotation.is_vector() ? type_annotation :
                        (type_suffix.is_vector() ? type_suffix : (lhs ? lhs->type() : Type::f32x4()));

            switch (op) {
                case Opcode::vadd: res_val = b.build_vadd(lhs, rhs); break;
                case Opcode::vsub: res_val = b.build_vsub(lhs, rhs); break;
                case Opcode::vmul: res_val = b.build_vmul(lhs, rhs); break;
                case Opcode::vdiv: res_val = b.build_vdiv(lhs, rhs); break;
                case Opcode::vmin: res_val = b.build_vmin(lhs, rhs); break;
                case Opcode::vmax: res_val = b.build_vmax(lhs, rhs); break;
                case Opcode::vand: res_val = b.build_vand(lhs, rhs); break;
                case Opcode::vor:  res_val = b.build_vor(lhs, rhs); break;
                case Opcode::vxor: res_val = b.build_vxor(lhs, rhs); break;
                default: break;
            }
            if (res_val && op_t.is_vector()) {
                res_val->set_type(op_t);
                if (res_val->defining_instruction()) res_val->defining_instruction()->set_type(op_t);
            }
            return true;
        }

        case Opcode::vneg:
        case Opcode::vsqrt:
        case Opcode::vnot: {
            Value* val = ctx.parse_val();
            if (!val) return false;

            Type op_t = type_annotation.is_vector() ? type_annotation :
                        (type_suffix.is_vector() ? type_suffix : (val ? val->type() : Type::f32x4()));

            switch (op) {
                case Opcode::vneg:  res_val = b.build_vneg(val); break;
                case Opcode::vsqrt: res_val = b.build_vsqrt(val); break;
                case Opcode::vnot:  res_val = b.build_vnot(val); break;
                default: break;
            }
            if (res_val && op_t.is_vector()) {
                res_val->set_type(op_t);
                if (res_val->defining_instruction()) res_val->defining_instruction()->set_type(op_t);
            }
            return true;
        }

        case Opcode::vload: {
            Value* base = ctx.parse_val();
            if (!base) return false;
            int32_t offset = 0;
            if (ctx.match(TokenKind::Comma)) {
                if (!ctx.peek().is(TokenKind::IntLiteral)) {
                    ctx.error(ctx.peek().location, "Expected integer offset for vload");
                    return false;
                }
                offset = static_cast<int32_t>(ctx.advance().int_val);
            }
            Type vt = mem_type.is_vector() ? mem_type :
                      (type_annotation.is_vector() ? type_annotation :
                      (type_suffix.is_vector() ? type_suffix : Type::f32x4()));
            res_val = b.build_vload(vt, base, offset);
            return true;
        }

        case Opcode::vstore: {
            Value* base = ctx.parse_val();
            if (!base) return false;
            if (!ctx.expect(TokenKind::Comma, "','")) return false;

            int32_t offset = 0;
            Value* val = nullptr;
            if (ctx.peek().is(TokenKind::IntLiteral)) {
                offset = static_cast<int32_t>(ctx.advance().int_val);
                if (!ctx.expect(TokenKind::Comma, "','")) return false;
                val = ctx.parse_val();
            } else {
                val = ctx.parse_val();
            }
            if (!val) return false;

            Type vt = mem_type.is_vector() ? mem_type :
                      (type_suffix.is_vector() ? type_suffix :
                      (val ? val->type() : Type::f32x4()));
            res_inst = b.build_vstore(vt, base, offset, val);
            return true;
        }

        case Opcode::vbroadcast: {
            Value* scalar = ctx.parse_val();
            if (!scalar) return false;
            Type vt = type_suffix.is_vector() ? type_suffix :
                      (type_annotation.is_vector() ? type_annotation : Type::f32x4());
            res_val = b.build_vbroadcast(vt, scalar);
            return true;
        }

        case Opcode::vextract_lane: {
            Value* vec = ctx.parse_val();
            if (!vec) return false;
            if (!ctx.expect(TokenKind::Comma, "','")) return false;
            if (!ctx.peek().is(TokenKind::IntLiteral)) {
                ctx.error(ctx.peek().location, "Expected integer lane index for vextract_lane");
                return false;
            }
            uint32_t lane = static_cast<uint32_t>(ctx.advance().int_val);
            res_val = b.build_vextract_lane(vec, lane);
            if (type_annotation != Type::void_type() && res_val) {
                res_val->set_type(type_annotation);
                if (res_val->defining_instruction()) res_val->defining_instruction()->set_type(type_annotation);
            }
            return true;
        }

        case Opcode::vinsert_lane: {
            Value* vec = ctx.parse_val();
            if (!vec) return false;
            if (!ctx.expect(TokenKind::Comma, "','")) return false;
            Value* scalar = ctx.parse_val();
            if (!scalar) return false;
            if (!ctx.expect(TokenKind::Comma, "','")) return false;
            if (!ctx.peek().is(TokenKind::IntLiteral)) {
                ctx.error(ctx.peek().location, "Expected integer lane index for vinsert_lane");
                return false;
            }
            uint32_t lane = static_cast<uint32_t>(ctx.advance().int_val);
            res_val = b.build_vinsert_lane(vec, scalar, lane);
            Type vt = type_annotation.is_vector() ? type_annotation :
                      (vec ? vec->type() : Type::f32x4());
            if (res_val) {
                res_val->set_type(vt);
                if (res_val->defining_instruction()) res_val->defining_instruction()->set_type(vt);
            }
            return true;
        }

        case Opcode::vshuffle: {
            Value* v1 = ctx.parse_val();
            if (!v1) return false;
            if (!ctx.expect(TokenKind::Comma, "','")) return false;
            Value* v2 = ctx.parse_val();
            if (!v2) return false;
            if (!ctx.expect(TokenKind::Comma, "','")) return false;
            if (!ctx.peek().is(TokenKind::IntLiteral)) {
                ctx.error(ctx.peek().location, "Expected integer mask for vshuffle");
                return false;
            }
            uint32_t mask = static_cast<uint32_t>(ctx.advance().int_val);
            res_val = b.build_vshuffle(v1, v2, mask);
            Type vt = type_annotation.is_vector() ? type_annotation :
                      (v1 ? v1->type() : Type::f32x4());
            if (res_val) {
                res_val->set_type(vt);
                if (res_val->defining_instruction()) res_val->defining_instruction()->set_type(vt);
            }
            return true;
        }

        case Opcode::vzero: {
            Type vt = type_suffix.is_vector() ? type_suffix :
                      (type_annotation.is_vector() ? type_annotation : Type::f32x4());
            res_val = b.build_vzero(vt);
            return true;
        }

        default:
            return false;
    }
}

} // namespace brass
