#include "mir_sym_exec.hpp"
#include <brass/mir/instruction.hpp>
#include <brass/mir/opcodes.hpp>
#include <algorithm>

namespace brass::verify {

namespace {

BVNodeId vec_coerce(BVContext& ctx, BVNodeId node, uint32_t target_w) {
    uint32_t w = ctx.width(node);
    if (w == target_w) return node;
    if (w > target_w) return ctx.make_extract(node, target_w - 1, 0);
    return ctx.make_zext(node, target_w - w);
}

BVNodeId vec_eval_addr(BVContext& ctx, const Instruction* inst,
                       const std::unordered_map<const Value*, BVNodeId>& vals,
                       bool& ok) {
    const Value* ptr = inst->operand(0);
    if (!ptr || !vals.count(ptr)) {
        ok = false;
        return ctx.make_const(0, 64);
    }
    BVNodeId addr = vec_coerce(ctx, vals.at(ptr), 64);
    if (inst->offset() != 0) {
        addr = ctx.make_binary(BVOp::Add, addr,
            ctx.make_const(static_cast<uint64_t>(static_cast<int64_t>(inst->offset())), 64), 64);
    }
    return addr;
}

bool get_vec_operand(const Instruction* inst, size_t idx, const SymVecMap& vec_vals,
                     SymExecOutput& out, std::vector<BVNodeId>& lanes_out) {
    const Value* v = inst->operand(idx);
    if (!v || !v->type().is_vector()) {
        out.has_type_error = true;
        out.error_msg = "Expected vector operand";
        return false;
    }
    auto it = vec_vals.find(v);
    if (it == vec_vals.end()) {
        out.has_type_error = true;
        out.error_msg = "Use of undefined/undominated vector SSA value";
        return false;
    }
    lanes_out = it->second;
    return true;
}

std::vector<BVNodeId> reinterpret_lanes(BVContext& ctx, const std::vector<BVNodeId>& src,
                                        uint32_t src_w, uint32_t dst_lanes, uint32_t dst_w) {
    // Pack into contiguous 64-bit words and unpack into dst_w-bit lanes.
    std::vector<BVNodeId> words64;
    const uint32_t sper = 64u / src_w;
    for (size_t base = 0; base < src.size(); base += sper) {
        size_t cnt = std::min<size_t>(sper, src.size() - base);
        BVNodeId w = src[base + cnt - 1];
        for (size_t k = cnt - 1; k-- > 0;) {
            w = ctx.make_concat(w, src[base + k]);
        }
        if (cnt < sper) {
            w = ctx.make_zext(w, static_cast<uint32_t>(64u - cnt * src_w));
        }
        words64.push_back(w);
    }
    std::vector<BVNodeId> dst;
    dst.reserve(dst_lanes);
    const uint32_t dper = 64u / dst_w;
    for (size_t i = 0; i < dst_lanes; ++i) {
        size_t wi = i / dper;
        uint32_t bi = static_cast<uint32_t>(i % dper) * dst_w;
        if (wi >= words64.size()) {
            dst.push_back(ctx.make_const(0, dst_w));
        } else if (dst_w == 64) {
            dst.push_back(words64[wi]);
        } else {
            dst.push_back(ctx.make_extract(words64[wi], bi + dst_w - 1, bi));
        }
    }
    return dst;
}

} // namespace

bool symbolically_execute_vec_inst(
    BVContext& ctx,
    InitMemEnv& init_mem,
    SymExecOutput& out,
    BVNodeId cur_cond,
    const Instruction* inst,
    std::unordered_map<const Value*, BVNodeId>& vals,
    SymVecMap& vec_vals,
    const std::unordered_set<uint64_t>& alloca_bases) {

    const Opcode op = inst->opcode();
    const Type rt = inst->type();
    const uint32_t res_lanes = rt.is_vector() ? rt.vector_lanes() : 0;
    const Type res_elem = rt.is_vector() ? rt.element_type() : rt;
    const uint32_t res_ew = type_bv_width(res_elem);

    auto set_vec = [&](std::vector<BVNodeId> lanes) {
        if (inst->result()) vec_vals[inst->result()] = std::move(lanes);
    };
    auto set_scalar = [&](BVNodeId id) {
        if (inst->result()) vals[inst->result()] = id;
    };
    auto get_scalar = [&](size_t idx, uint32_t w) -> BVNodeId {
        const Value* v = inst->operand(idx);
        if (!v) { out.has_type_error = true; return ctx.make_const(0, w); }
        auto it = vals.find(v);
        if (it == vals.end()) { out.has_type_error = true; return ctx.make_const(0, w); }
        return vec_coerce(ctx, it->second, w);
    };

    switch (op) {
        case Opcode::vzero: {
            if (!rt.is_vector() || res_ew == 0) { out.has_type_error = true; return true; }
            set_vec(std::vector<BVNodeId>(res_lanes, ctx.make_const(0, res_ew)));
            return true;
        }

        case Opcode::vbroadcast: {
            if (!rt.is_vector() || res_ew == 0) { out.has_type_error = true; return true; }
            BVNodeId s = get_scalar(0, res_ew);
            set_vec(std::vector<BVNodeId>(res_lanes, s));
            return true;
        }

        case Opcode::vload: {
            if (!rt.is_vector() || res_ew == 0) { out.has_type_error = true; return true; }
            bool ok = true;
            BVNodeId addr = vec_eval_addr(ctx, inst, vals, ok);
            if (!ok) { out.has_type_error = true; out.error_msg = "Undefined pointer in vload"; return true; }
            uint32_t eb = res_ew / 8;
            std::vector<BVNodeId> loaded;
            loaded.reserve(res_lanes);
            for (uint32_t i = 0; i < res_lanes; ++i) {
                BVNodeId la = (i == 0) ? addr
                    : ctx.make_binary(BVOp::Add, addr, ctx.make_const(static_cast<uint64_t>(i) * eb, 64), 64);
                loaded.push_back(read_memory_bv(ctx, init_mem, out.stores, la, eb));
            }
            set_vec(std::move(loaded));
            return true;
        }

        case Opcode::vstore: {
            bool ok = true;
            BVNodeId base = vec_eval_addr(ctx, inst, vals, ok);
            std::vector<BVNodeId> src;
            if (!ok || !get_vec_operand(inst, 1, vec_vals, out, src)) {
                out.has_type_error = true;
                return true;
            }
            Type mt = inst->memory_type().is_vector() ? inst->memory_type() : inst->operand(1)->type();
            uint32_t lanes = mt.vector_lanes();
            uint32_t eb = static_cast<uint32_t>(mt.element_type().size_in_bytes());
            if (lanes == 0 || src.size() != lanes || (eb != 1 && eb != 2 && eb != 4 && eb != 8)) {
                out.has_type_error = true;
                return true;
            }
            const Value* ptr_val = inst->operand(0);
            BVNodeId raw_base = vals.at(ptr_val);
            bool is_local = (ctx.nodes()[raw_base].op == BVOp::Const &&
                             alloca_bases.count(ctx.nodes()[raw_base].imm) > 0);
            for (uint32_t i = 0; i < lanes; ++i) {
                BVNodeId la = (i == 0) ? base
                    : ctx.make_binary(BVOp::Add, base, ctx.make_const(static_cast<uint64_t>(i) * eb, 64), 64);
                out.stores.push_back({cur_cond, la, eb, vec_coerce(ctx, src[i], eb * 8), is_local});
            }
            return true;
        }

        case Opcode::vextract_lane: {
            std::vector<BVNodeId> src;
            if (!get_vec_operand(inst, 0, vec_vals, out, src)) return true;
            uint32_t lane = inst->lane();
            if (lane >= src.size()) { out.has_type_error = true; return true; }
            set_scalar(vec_coerce(ctx, src[lane], type_bv_width(rt)));
            return true;
        }

        case Opcode::vinsert_lane: {
            std::vector<BVNodeId> src;
            if (!get_vec_operand(inst, 0, vec_vals, out, src)) return true;
            uint32_t lane = inst->lane();
            if (lane >= src.size() || !rt.is_vector()) { out.has_type_error = true; return true; }
            src[lane] = get_scalar(1, res_ew);
            set_vec(std::move(src));
            return true;
        }

        case Opcode::vshuffle: {
            std::vector<BVNodeId> a, b;
            if (!get_vec_operand(inst, 0, vec_vals, out, a) ||
                !get_vec_operand(inst, 1, vec_vals, out, b) ||
                !rt.is_vector() || a.size() != res_lanes || b.size() != res_lanes ||
                res_elem.size_in_bytes() < 4) {
                out.has_type_error = true;
                return true;
            }
            uint32_t mask = inst->shuffle_mask();
            uint32_t half_lanes = static_cast<uint32_t>(16 / res_elem.size_in_bytes());
            std::vector<BVNodeId> dst(res_lanes);
            for (uint32_t l = 0; l < res_lanes; ++l) {
                uint32_t half = l / half_lanes;
                uint32_t in_half = l % half_lanes;
                uint32_t sel = (half_lanes == 4) ? ((mask >> (2 * in_half)) & 3u)
                                                 : ((mask >> (half * 2 + in_half)) & 1u);
                bool from_b = (half_lanes == 4) ? (in_half >= 2) : (in_half == 1);
                uint32_t src_lane = half * half_lanes + sel;
                dst[l] = (from_b ? b : a)[src_lane];
            }
            set_vec(std::move(dst));
            return true;
        }

        case Opcode::vadd: case Opcode::vsub: case Opcode::vmul:
        case Opcode::vdiv: case Opcode::vmin: case Opcode::vmax: {
            std::vector<BVNodeId> a, b;
            if (!get_vec_operand(inst, 0, vec_vals, out, a) ||
                !get_vec_operand(inst, 1, vec_vals, out, b) ||
                a.size() != res_lanes || b.size() != res_lanes) {
                out.has_type_error = true;
                return true;
            }
            bool is_fl = res_elem.is_float();
            std::vector<BVNodeId> dst(res_lanes);
            for (uint32_t i = 0; i < res_lanes; ++i) {
                BVNodeId ai = vec_coerce(ctx, a[i], res_ew);
                BVNodeId bi = vec_coerce(ctx, b[i], res_ew);
                if (op == Opcode::vmin || op == Opcode::vmax) {
                    if (is_fl) {
                        dst[i] = ctx.make_binary(op == Opcode::vmin ? BVOp::FMin : BVOp::FMax, ai, bi, res_ew);
                    } else {
                        BVNodeId cond = ctx.make_binary(op == Opcode::vmin ? BVOp::Slt : BVOp::Sgt, ai, bi, 1);
                        dst[i] = ctx.make_ite(cond, ai, bi);
                    }
                } else if (op == Opcode::vdiv && !is_fl) {
                    out.supported = false;
                    out.error_msg = "Integer vector division is not supported";
                    return true;
                } else {
                    BVOp bop = is_fl
                        ? (op == Opcode::vadd ? BVOp::FAdd : op == Opcode::vsub ? BVOp::FSub
                           : op == Opcode::vmul ? BVOp::FMul : BVOp::FDiv)
                        : (op == Opcode::vadd ? BVOp::Add : op == Opcode::vsub ? BVOp::Sub : BVOp::Mul);
                    dst[i] = ctx.make_binary(bop, ai, bi, res_ew);
                }
            }
            set_vec(std::move(dst));
            return true;
        }

        case Opcode::vfma: {
            std::vector<BVNodeId> a, b, c;
            if (!get_vec_operand(inst, 0, vec_vals, out, a) ||
                !get_vec_operand(inst, 1, vec_vals, out, b) ||
                !get_vec_operand(inst, 2, vec_vals, out, c) ||
                a.size() != res_lanes || b.size() != res_lanes || c.size() != res_lanes) {
                out.has_type_error = true;
                return true;
            }
            std::vector<BVNodeId> dst(res_lanes);
            for (uint32_t i = 0; i < res_lanes; ++i) {
                BVNodeId ai = vec_coerce(ctx, a[i], res_ew);
                BVNodeId bi = vec_coerce(ctx, b[i], res_ew);
                BVNodeId ci = vec_coerce(ctx, c[i], res_ew);
                if (res_elem.is_float()) {
                    dst[i] = init_mem.eval_uninterpreted_call(
                        ctx, res_ew == 32 ? "fmaf" : "fma", {ai, bi, ci}, res_elem);
                } else {
                    dst[i] = ctx.make_binary(BVOp::Add, ctx.make_binary(BVOp::Mul, ai, bi, res_ew), ci, res_ew);
                }
            }
            set_vec(std::move(dst));
            return true;
        }

        case Opcode::vneg: case Opcode::vsqrt:
        case Opcode::vfloor: case Opcode::vceil: case Opcode::vtrunc: case Opcode::vround: {
            std::vector<BVNodeId> a;
            if (!get_vec_operand(inst, 0, vec_vals, out, a) || a.size() != res_lanes) {
                out.has_type_error = true;
                return true;
            }
            if (op == Opcode::vtrunc) {
                // Truncate toward zero: x - sign * floor(|x|) is exact via uninterpreted call or FpToSi?
                // Use pure uninterpreted float trunc matching ieee ftrunc to preserve bit exactness.
                std::vector<BVNodeId> dst(res_lanes);
                for (uint32_t i = 0; i < res_lanes; ++i) {
                    dst[i] = init_mem.eval_uninterpreted_call(
                        ctx, res_ew == 32 ? "truncf" : "trunc", {vec_coerce(ctx, a[i], res_ew)}, res_elem);
                }
                set_vec(std::move(dst));
                return true;
            }
            bool is_fl = res_elem.is_float();
            BVOp uop = (op == Opcode::vneg) ? (is_fl ? BVOp::FNeg : BVOp::Neg)
                     : (op == Opcode::vsqrt) ? BVOp::FSqrt
                     : (op == Opcode::vfloor) ? BVOp::FFloor
                     : (op == Opcode::vceil) ? BVOp::FCeil : BVOp::FRound;
            std::vector<BVNodeId> dst(res_lanes);
            for (uint32_t i = 0; i < res_lanes; ++i) {
                dst[i] = ctx.make_unary(uop, vec_coerce(ctx, a[i], res_ew), res_ew);
            }
            set_vec(std::move(dst));
            return true;
        }

        case Opcode::vand: case Opcode::vor: case Opcode::vxor: case Opcode::vnot: {
            std::vector<BVNodeId> a, b;
            if (!get_vec_operand(inst, 0, vec_vals, out, a) || a.size() != res_lanes) {
                out.has_type_error = true;
                return true;
            }
            if (op != Opcode::vnot && (!get_vec_operand(inst, 1, vec_vals, out, b) || b.size() != res_lanes)) {
                out.has_type_error = true;
                return true;
            }
            std::vector<BVNodeId> dst(res_lanes);
            for (uint32_t i = 0; i < res_lanes; ++i) {
                BVNodeId ai = vec_coerce(ctx, a[i], res_ew);
                if (op == Opcode::vnot) {
                    dst[i] = ctx.make_unary(BVOp::Not, ai, res_ew);
                } else {
                    BVOp bop = (op == Opcode::vand) ? BVOp::And : (op == Opcode::vor) ? BVOp::Or : BVOp::Xor;
                    dst[i] = ctx.make_binary(bop, ai, vec_coerce(ctx, b[i], res_ew), res_ew);
                }
            }
            set_vec(std::move(dst));
            return true;
        }

        case Opcode::vcmp_eq: case Opcode::vcmp_ne: case Opcode::vcmp_lt:
        case Opcode::vcmp_le: case Opcode::vcmp_gt: case Opcode::vcmp_ge: {
            std::vector<BVNodeId> a, b;
            if (!get_vec_operand(inst, 0, vec_vals, out, a) ||
                !get_vec_operand(inst, 1, vec_vals, out, b) ||
                a.size() != res_lanes || b.size() != res_lanes) {
                out.has_type_error = true;
                return true;
            }
            Type in_elem = inst->operand(0)->type().element_type();
            uint32_t ew = type_bv_width(in_elem);
            bool is_fl = in_elem.is_float();
            BVOp cmp_op = is_fl
                ? ((op == Opcode::vcmp_eq) ? BVOp::FEq : (op == Opcode::vcmp_ne) ? BVOp::FNe
                   : (op == Opcode::vcmp_lt) ? BVOp::FLt : (op == Opcode::vcmp_le) ? BVOp::FLe
                   : (op == Opcode::vcmp_gt) ? BVOp::FGt : BVOp::FGe)
                : ((op == Opcode::vcmp_eq) ? BVOp::Eq : (op == Opcode::vcmp_ne) ? BVOp::Ne
                   : (op == Opcode::vcmp_lt) ? BVOp::Slt : (op == Opcode::vcmp_le) ? BVOp::Sle
                   : (op == Opcode::vcmp_gt) ? BVOp::Sgt : BVOp::Sge);
            BVNodeId ones = ctx.make_const(width_mask(ew), ew);
            BVNodeId zeros = ctx.make_const(0, ew);
            std::vector<BVNodeId> dst(res_lanes);
            for (uint32_t i = 0; i < res_lanes; ++i) {
                BVNodeId c1 = ctx.make_binary(cmp_op, vec_coerce(ctx, a[i], ew), vec_coerce(ctx, b[i], ew), 1);
                dst[i] = ctx.make_ite(c1, ones, zeros);
            }
            set_vec(std::move(dst));
            return true;
        }

        case Opcode::vselect: {
            std::vector<BVNodeId> m, a, b;
            if (!get_vec_operand(inst, 0, vec_vals, out, m) ||
                !get_vec_operand(inst, 1, vec_vals, out, a) ||
                !get_vec_operand(inst, 2, vec_vals, out, b) ||
                m.size() != res_lanes || a.size() != res_lanes || b.size() != res_lanes) {
                out.has_type_error = true;
                return true;
            }
            std::vector<BVNodeId> dst(res_lanes);
            for (uint32_t i = 0; i < res_lanes; ++i) {
                uint32_t mw = ctx.width(m[i]);
                BVNodeId msb = ctx.make_extract(m[i], mw - 1, mw - 1);
                dst[i] = ctx.make_ite(msb, vec_coerce(ctx, a[i], res_ew), vec_coerce(ctx, b[i], res_ew));
            }
            set_vec(std::move(dst));
            return true;
        }

        case Opcode::vshl: case Opcode::vlshr: case Opcode::vashr: {
            std::vector<BVNodeId> a, b;
            if (!get_vec_operand(inst, 0, vec_vals, out, a) ||
                !get_vec_operand(inst, 1, vec_vals, out, b) ||
                a.size() != res_lanes || b.size() != res_lanes) {
                out.has_type_error = true;
                return true;
            }
            BVNodeId w_c = ctx.make_const(res_ew, res_ew);
            BVNodeId wm1_c = ctx.make_const(res_ew - 1, res_ew);
            BVNodeId zero_c = ctx.make_const(0, res_ew);
            std::vector<BVNodeId> dst(res_lanes);
            for (uint32_t i = 0; i < res_lanes; ++i) {
                BVNodeId ai = vec_coerce(ctx, a[i], res_ew);
                BVNodeId si = vec_coerce(ctx, b[i], res_ew);
                BVNodeId past = ctx.make_binary(BVOp::Uge, si, w_c, 1);
                if (op == Opcode::vashr) {
                    BVNodeId eff = ctx.make_ite(past, wm1_c, si);
                    dst[i] = ctx.make_binary(BVOp::AShr, ai, eff, res_ew);
                } else {
                    BVOp sop = (op == Opcode::vshl) ? BVOp::Shl : BVOp::LShr;
                    dst[i] = ctx.make_ite(past, zero_c, ctx.make_binary(sop, ai, si, res_ew));
                }
            }
            set_vec(std::move(dst));
            return true;
        }

        case Opcode::vconvert: {
            std::vector<BVNodeId> a;
            if (!get_vec_operand(inst, 0, vec_vals, out, a) || !rt.is_vector()) {
                out.has_type_error = true;
                return true;
            }
            Type in_elem = inst->operand(0)->type().element_type();
            uint32_t in_ew = type_bv_width(in_elem);
            uint32_t count = std::min<uint32_t>(res_lanes, static_cast<uint32_t>(a.size()));
            std::vector<BVNodeId> dst(res_lanes, ctx.make_const(0, res_ew));
            for (uint32_t i = 0; i < count; ++i) {
                BVNodeId ai = vec_coerce(ctx, a[i], in_ew);
                if (in_elem.is_integer() && res_elem.is_float()) {
                    dst[i] = ctx.make_unary(BVOp::SiToFp, ai, res_ew);
                } else if (in_elem.is_float() && res_elem.is_integer()) {
                    dst[i] = ctx.make_unary(BVOp::FpToSi, ai, res_ew);
                } else if (in_elem == Type::f32() && res_elem == Type::f64()) {
                    dst[i] = ctx.make_unary(BVOp::FpExt, ai, 64);
                } else if (in_elem == Type::f64() && res_elem == Type::f32()) {
                    dst[i] = ctx.make_unary(BVOp::FpTrunc, ai, 32);
                } else if (in_elem == Type::i32() && res_elem == Type::i64()) {
                    dst[i] = ctx.make_sext(ai, 32);
                } else if (in_elem == Type::i64() && res_elem == Type::i32()) {
                    dst[i] = ctx.make_extract(ai, 31, 0);
                } else {
                    dst[i] = vec_coerce(ctx, ai, res_ew);
                }
            }
            set_vec(std::move(dst));
            return true;
        }

        case Opcode::vbitcast: {
            std::vector<BVNodeId> a;
            if (!get_vec_operand(inst, 0, vec_vals, out, a) || !rt.is_vector()) {
                out.has_type_error = true;
                return true;
            }
            uint32_t in_ew = type_bv_width(inst->operand(0)->type().element_type());
            set_vec(reinterpret_lanes(ctx, a, in_ew, res_lanes, res_ew));
            return true;
        }

        case Opcode::vany: {
            std::vector<BVNodeId> a;
            if (!get_vec_operand(inst, 0, vec_vals, out, a)) return true;
            uint32_t in_ew = type_bv_width(inst->operand(0)->type().element_type());
            BVNodeId any1 = ctx.make_const(0, 1);
            for (BVNodeId lane : a) {
                any1 = ctx.make_binary(BVOp::Or, any1, ctx.make_extract(lane, in_ew - 1, in_ew - 1), 1);
            }
            set_scalar(ctx.make_zext(any1, 31));
            return true;
        }

        default:
            return false;
    }
}

} // namespace brass::verify
