#include <brass/mir/gpu_superopt.hpp>
#include <brass/mir/superopt.hpp>
#include "../verify/bv_solver.hpp"
#include <brass/mir/builder.hpp>
#include <brass/mir/dominators.hpp>
#include <brass/mir/loop_analysis.hpp>
#include <brass/mir/range_analysis.hpp>
#include <brass/mir/uses.hpp>
#include <algorithm>
#include <bit>
#include <unordered_map>
#include <vector>

namespace brass::verify {

MagicConstU32 compute_magic_const_u32(uint32_t d) noexcept {
    MagicConstU32 out;
    if (d <= 1) {
        out.multiplier = 1;
        return out;
    }
    auto try_no_add = [&](uint32_t pre_sh, MagicConstU32& res, uint64_t& first_m64, uint32_t& first_s) -> bool {
        uint32_t dp = d >> pre_sh;
        uint64_t xmax = 0xFFFFFFFFULL >> pre_sh;
        uint64_t nc = xmax - ((xmax + 1ULL) % dp);
        for (uint32_t s = 0; s < 32; ++s) {
            uint64_t two_p = 1ULL << (32 + s);
            uint64_t rem = two_p % dp;
            uint64_t delta = (rem == 0) ? 0ULL : (dp - rem);
            // (nc and delta are both under 2^32: their product fits 64 bits.)
            if (nc * delta < two_p) {
                uint64_t m64 = (two_p + delta) / dp;
                first_m64 = m64;
                first_s = s;
                if (m64 <= 0xFFFFFFFFULL) {
                    res.multiplier = static_cast<uint32_t>(m64);
                    res.pre_shift = pre_sh;
                    res.post_shift = s;
                    res.needs_add = false;
                    return true;
                }
                return false;
            }
        }
        return false;
    };

    uint64_t m64_0 = 0;
    uint32_t s_0 = 0;
    if (try_no_add(0, out, m64_0, s_0)) return out;
    uint32_t tz = static_cast<uint32_t>(std::countr_zero(d));
    uint64_t dummy_m = 0;
    uint32_t dummy_s = 0;
    if (tz > 0 && try_no_add(tz, out, dummy_m, dummy_s)) return out;
    out.multiplier = static_cast<uint32_t>(m64_0 - (1ULL << 32));
    out.pre_shift = 0;
    out.post_shift = s_0;
    out.needs_add = true;
    return out;
}

BVNodeId build_magic_udiv_u32(BVContext& ctx, BVNodeId a, const MagicConstU32& mag) {
    BVNodeId x = a;
    if (mag.pre_shift > 0) {
        x = ctx.make_binary(BVOp::LShr, x, ctx.make_const(mag.pre_shift, 32), 32);
    }
    BVNodeId prod64 = ctx.make_binary(BVOp::Mul, ctx.make_zext(x, 32), ctx.make_const(mag.multiplier, 64), 64);
    BVNodeId hi32 = ctx.make_extract(prod64, 63, 32);
    if (!mag.needs_add) {
        return (mag.post_shift > 0)
            ? ctx.make_binary(BVOp::LShr, hi32, ctx.make_const(mag.post_shift, 32), 32)
            : hi32;
    }
    BVNodeId t = ctx.make_binary(BVOp::Sub, x, hi32, 32);
    BVNodeId t2 = ctx.make_binary(BVOp::LShr, t, ctx.make_const(1, 32), 32);
    BVNodeId t3 = ctx.make_binary(BVOp::Add, t2, hi32, 32);
    return (mag.post_shift > 1)
        ? ctx.make_binary(BVOp::LShr, t3, ctx.make_const(mag.post_shift - 1, 32), 32)
        : t3;
}

BVNodeId expand_const_div_rem_bv(BVContext& ctx, BVOp op, BVNodeId a, BVNodeId b, uint32_t w) {
    const BVNode cb = ctx.nodes()[b];
    if (op == BVOp::UDiv || op == BVOp::SDiv) {
        if (cb.op == BVOp::Const && cb.imm == 1) return a;
        if (a == b && ctx.interval(b).umin > 0) return ctx.make_const(1, w);
        if (cb.op == BVOp::Const && std::has_single_bit(cb.imm)) {
            uint32_t sh = static_cast<uint32_t>(std::countr_zero(cb.imm));
            if (op == BVOp::UDiv) return ctx.make_binary(BVOp::LShr, a, ctx.make_const(sh, w), w);
            if (sh > 0 && sh + 1 < w) {
                BVNodeId sh_c = ctx.make_const(sh, w);
                BVNodeId non_neg = ctx.make_binary(BVOp::Sge, a, ctx.make_const(0, w), 1);
                BVNodeId pos_q = ctx.make_binary(BVOp::LShr, a, sh_c, w);
                BVNodeId neg_a = ctx.make_unary(BVOp::Neg, a, w);
                BVNodeId neg_q = ctx.make_unary(BVOp::Neg, ctx.make_binary(BVOp::LShr, neg_a, sh_c, w), w);
                return ctx.make_ite(non_neg, pos_q, neg_q);
            }
        }
        if (w == 32 && cb.op == BVOp::Const) {
            uint32_t d = static_cast<uint32_t>(cb.imm & 0xFFFFFFFFULL);
            if (op == BVOp::UDiv && d > 1 && !std::has_single_bit(d)) {
                return build_magic_udiv_u32(ctx, a, compute_magic_const_u32(d));
            }
            if (op == BVOp::SDiv && static_cast<int32_t>(d) > 1 && !std::has_single_bit(d)) {
                MagicConstU32 mag = compute_magic_const_u32(d);
                BVNodeId non_neg = ctx.make_binary(BVOp::Sge, a, ctx.make_const(0, 32), 1);
                BVNodeId pos_q = build_magic_udiv_u32(ctx, a, mag);
                BVNodeId neg_a = ctx.make_unary(BVOp::Neg, a, 32);
                BVNodeId neg_q = ctx.make_unary(BVOp::Neg, build_magic_udiv_u32(ctx, neg_a, mag), 32);
                return ctx.make_ite(non_neg, pos_q, neg_q);
            }
        }
        return UINT32_MAX;
    }
    if (cb.op == BVOp::Const && cb.imm == 1) return ctx.make_const(0, w);
    if (a == b && ctx.interval(b).umin > 0) return ctx.make_const(0, w);
    if (cb.op == BVOp::Const && std::has_single_bit(cb.imm)) {
        BVNodeId mask_c = ctx.make_const(cb.imm - 1, w);
        if (op == BVOp::URem) return ctx.make_binary(BVOp::And, a, mask_c, w);
        uint32_t sh = static_cast<uint32_t>(std::countr_zero(cb.imm));
        if (sh > 0 && sh + 1 < w) {
            BVNodeId non_neg = ctx.make_binary(BVOp::Sge, a, ctx.make_const(0, w), 1);
            BVNodeId pos_r = ctx.make_binary(BVOp::And, a, mask_c, w);
            BVNodeId neg_a = ctx.make_unary(BVOp::Neg, a, w);
            BVNodeId neg_r = ctx.make_unary(BVOp::Neg, ctx.make_binary(BVOp::And, neg_a, mask_c, w), w);
            return ctx.make_ite(non_neg, pos_r, neg_r);
        }
    }
    if (w == 32 && cb.op == BVOp::Const) {
        uint32_t d = static_cast<uint32_t>(cb.imm & 0xFFFFFFFFULL);
        if (op == BVOp::URem && d > 1 && !std::has_single_bit(d)) {
            BVNodeId q = build_magic_udiv_u32(ctx, a, compute_magic_const_u32(d));
            return ctx.make_binary(BVOp::Sub, a, ctx.make_binary(BVOp::Mul, q, b, 32), 32);
        }
        if (op == BVOp::SRem && static_cast<int32_t>(d) > 1 && !std::has_single_bit(d)) {
            MagicConstU32 mag = compute_magic_const_u32(d);
            BVNodeId non_neg = ctx.make_binary(BVOp::Sge, a, ctx.make_const(0, 32), 1);
            BVNodeId pos_q = build_magic_udiv_u32(ctx, a, mag);
            BVNodeId pos_r = ctx.make_binary(BVOp::Sub, a, ctx.make_binary(BVOp::Mul, pos_q, b, 32), 32);
            BVNodeId neg_a = ctx.make_unary(BVOp::Neg, a, 32);
            BVNodeId neg_q = build_magic_udiv_u32(ctx, neg_a, mag);
            BVNodeId neg_r = ctx.make_unary(
                BVOp::Neg, ctx.make_binary(BVOp::Sub, neg_a, ctx.make_binary(BVOp::Mul, neg_q, b, 32), 32), 32);
            return ctx.make_ite(non_neg, pos_r, neg_r);
        }
    }
    return UINT32_MAX;
}

BVNodeId try_eval_gpu_call_bv(BVContext& ctx, std::string_view sym, const std::vector<BVNodeId>& args) {
    auto coerce = [&](BVNodeId n, uint32_t w, bool s) -> BVNodeId {
        uint32_t nw = ctx.width(n);
        if (nw == w) return n;
        if (nw > w) return ctx.make_extract(n, w - 1, 0);
        return s ? ctx.make_sext(n, w - nw) : ctx.make_zext(n, w - nw);
    };
    if (sym == "ptx_mul_hi_u32" && args.size() == 2) {
        BVNodeId a64 = ctx.make_zext(coerce(args[0], 32, false), 32);
        BVNodeId b64 = ctx.make_zext(coerce(args[1], 32, false), 32);
        return ctx.make_extract(ctx.make_binary(BVOp::Mul, a64, b64, 64), 63, 32);
    }
    if (sym == "ptx_mul_wide_u32" && args.size() == 2) {
        BVNodeId a64 = ctx.make_zext(coerce(args[0], 32, false), 32);
        BVNodeId b64 = ctx.make_zext(coerce(args[1], 32, false), 32);
        return ctx.make_binary(BVOp::Mul, a64, b64, 64);
    }
    if (sym == "ptx_mul_wide_s32" && args.size() == 2) {
        BVNodeId a64 = ctx.make_sext(coerce(args[0], 32, true), 32);
        BVNodeId b64 = ctx.make_sext(coerce(args[1], 32, true), 32);
        return ctx.make_binary(BVOp::Mul, a64, b64, 64);
    }
    if (sym == "ptx_mad_lo_u32" && args.size() == 3) {
        BVNodeId a = coerce(args[0], 32, false);
        BVNodeId b = coerce(args[1], 32, false);
        BVNodeId c = coerce(args[2], 32, false);
        return ctx.make_binary(BVOp::Add, ctx.make_binary(BVOp::Mul, a, b, 32), c, 32);
    }
    return UINT32_MAX;
}

} // namespace brass::verify

namespace brass {

namespace {

using verify::BVContext;
using verify::BVNodeId;
using verify::BVOp;
using verify::MagicConstU32;
using verify::SolveOutcome;

constexpr BVNodeId kInvalidBV = UINT32_MAX;

uint32_t gpu_bv_width(Type t) noexcept {
    switch (t.kind()) {
        case TypeKind::I8: return 8;
        case TypeKind::I16: return 16;
        case TypeKind::I32: return 32;
        case TypeKind::I64:
        case TypeKind::Ptr:
        case TypeKind::GCRef:
        case TypeKind::Tagged: return 64;
        default: return 0;
    }
}

BVNodeId coerce_gpu_bv(BVContext& ctx, BVNodeId n, uint32_t target_w, bool sign_ext) {
    uint32_t w = ctx.width(n);
    if (w == target_w) return n;
    if (w > target_w) return ctx.make_extract(n, target_w - 1, 0);
    return sign_ext ? ctx.make_sext(n, target_w - w) : ctx.make_zext(n, target_w - w);
}

bool get_const_int(const Value* v, int64_t& out_imm) noexcept {
    if (!v || !v->is_instruction()) return false;
    const Instruction* inst = v->defining_instruction();
    if (!inst) return false;
    if (inst->opcode() == Opcode::iconst_i32 || inst->opcode() == Opcode::patchable_const_i32) {
        out_imm = inst->imm_i32();
        return true;
    }
    if (inst->opcode() == Opcode::iconst_i64 || inst->opcode() == Opcode::patchable_const_i64) {
        out_imm = inst->imm_i64();
        return true;
    }
    return false;
}

struct GpuSuperoptEncoder {
    BVContext& ctx;
    const RangeAnalysis& ra;
    const BasicBlock* cur_bb;
    uint32_t max_depth;
    std::unordered_map<const Value*, BVNodeId> cache;
    std::unordered_map<std::string, BVNodeId> special_vars;
    std::vector<BVNodeId> range_assumptions;

    void add_range_constraint(const Value* v, BVNodeId node) {
        if (!v || !cur_bb || node == kInvalidBV) return;
        uint32_t w = ctx.width(node);
        if (w != 32 && w != 64) return;
        ValueRange r = ra.get_range_at(v, cur_bb);
        if (r.is_empty()) return;
        int64_t type_min = (w == 32) ? static_cast<int64_t>(INT32_MIN) : INT64_MIN;
        int64_t type_max = (w == 32) ? static_cast<int64_t>(INT32_MAX) : INT64_MAX;
        if (r.min_val > type_min && r.min_val <= type_max) {
            BVNodeId lo = ctx.make_const(static_cast<uint64_t>(r.min_val), w);
            range_assumptions.push_back(ctx.make_binary(BVOp::Sge, node, lo, 1));
        }
        if (r.max_val < type_max && r.max_val >= type_min) {
            BVNodeId hi = ctx.make_const(static_cast<uint64_t>(r.max_val), w);
            range_assumptions.push_back(ctx.make_binary(BVOp::Sle, node, hi, 1));
        }
    }

    BVNodeId get_special_u32(std::string_view name, uint32_t mask, uint32_t min_val, uint32_t max_val) {
        std::string key(name);
        auto it = special_vars.find(key);
        if (it != special_vars.end()) return it->second;
        BVNodeId raw = ctx.make_var(key, 32, false);
        BVNodeId masked = ctx.make_binary(BVOp::And, raw, ctx.make_const(mask, 32), 32);
        if (min_val > 0) {
            range_assumptions.push_back(ctx.make_binary(BVOp::Uge, masked, ctx.make_const(min_val, 32), 1));
        }
        if (max_val < mask) {
            range_assumptions.push_back(ctx.make_binary(BVOp::Ule, masked, ctx.make_const(max_val, 32), 1));
        }
        special_vars[key] = masked;
        return masked;
    }

    BVNodeId encode(const Value* v, uint32_t depth = 0) {
        if (!v) return kInvalidBV;
        auto it = cache.find(v);
        if (it != cache.end()) return it->second;

        uint32_t w = gpu_bv_width(v->type());
        if (w == 0) return kInvalidBV;

        auto make_leaf = [&]() -> BVNodeId {
            BVNodeId raw = ctx.make_var("v" + std::to_string(v->id()), w, false);
            BVNodeId var = raw;
            if (cur_bb && (w == 32 || w == 64)) {
                ValueRange r = ra.get_range_at(v, cur_bb);
                int64_t type_max = (w == 32) ? static_cast<int64_t>(INT32_MAX) : INT64_MAX;
                if (!r.is_empty() && r.min_val >= 0 && r.max_val >= 0 && r.max_val <= type_max) {
                    uint64_t umax = static_cast<uint64_t>(r.max_val);
                    uint32_t bits = (umax == 0) ? 1u : static_cast<uint32_t>(64 - std::countl_zero(umax));
                    if (bits < w) {
                        uint64_t mask = (1ULL << bits) - 1ULL;
                        var = ctx.make_binary(BVOp::And, raw, ctx.make_const(mask, w), w);
                    }
                }
            }
            cache[v] = var;
            add_range_constraint(v, var);
            return var;
        };

        if (!v->is_instruction() || depth >= max_depth) return make_leaf();
        const Instruction* inst = v->defining_instruction();
        if (!inst) return make_leaf();

        Opcode op = inst->opcode();
        BVNodeId res = kInvalidBV;

        switch (op) {
            case Opcode::iconst_i32:
            case Opcode::patchable_const_i32:
                res = ctx.make_const(static_cast<uint32_t>(inst->imm_i32()), 32);
                break;
            case Opcode::iconst_i64:
            case Opcode::patchable_const_i64:
                res = ctx.make_const(static_cast<uint64_t>(inst->imm_i64()), 64);
                break;
            case Opcode::sext_i64: {
                BVNodeId a = encode(inst->operand(0), depth + 1);
                if (a == kInvalidBV) return make_leaf();
                res = coerce_gpu_bv(ctx, a, 64, true);
                break;
            }
            case Opcode::zext_i64: {
                BVNodeId a = encode(inst->operand(0), depth + 1);
                if (a == kInvalidBV) return make_leaf();
                res = coerce_gpu_bv(ctx, a, 64, false);
                break;
            }
            case Opcode::trunc_i32: {
                BVNodeId a = encode(inst->operand(0), depth + 1);
                if (a == kInvalidBV) return make_leaf();
                res = coerce_gpu_bv(ctx, a, 32, false);
                break;
            }
            case Opcode::add: case Opcode::sub: case Opcode::mul:
            case Opcode::sdiv: case Opcode::udiv: case Opcode::smod: case Opcode::umod:
            case Opcode::and_: case Opcode::or_: case Opcode::xor_:
            case Opcode::shl: case Opcode::lshr: case Opcode::ashr: {
                if (inst->type().is_float()) return make_leaf();
                BVNodeId a = encode(inst->operand(0), depth + 1);
                BVNodeId b = encode(inst->operand(1), depth + 1);
                if (a == kInvalidBV || b == kInvalidBV) return make_leaf();
                a = coerce_gpu_bv(ctx, a, w, true);
                b = coerce_gpu_bv(ctx, b, w, true);
                BVOp bop = (op == Opcode::add)  ? BVOp::Add  : (op == Opcode::sub)  ? BVOp::Sub
                         : (op == Opcode::mul)  ? BVOp::Mul  : (op == Opcode::sdiv) ? BVOp::SDiv
                         : (op == Opcode::udiv) ? BVOp::UDiv : (op == Opcode::smod) ? BVOp::SRem
                         : (op == Opcode::umod) ? BVOp::URem : (op == Opcode::and_) ? BVOp::And
                         : (op == Opcode::or_)  ? BVOp::Or   : (op == Opcode::xor_) ? BVOp::Xor
                         : (op == Opcode::shl)  ? BVOp::Shl  : (op == Opcode::lshr) ? BVOp::LShr : BVOp::AShr;
                res = ctx.make_binary(bop, a, b, w);
                break;
            }
            case Opcode::neg: case Opcode::not_: {
                if (inst->type().is_float()) return make_leaf();
                BVNodeId a = encode(inst->operand(0), depth + 1);
                if (a == kInvalidBV) return make_leaf();
                a = coerce_gpu_bv(ctx, a, w, false);
                res = ctx.make_unary(op == Opcode::neg ? BVOp::Neg : BVOp::Not, a, w);
                break;
            }
            case Opcode::eq: case Opcode::ne:
            case Opcode::slt: case Opcode::ult: case Opcode::sle: case Opcode::ule:
            case Opcode::sgt: case Opcode::ugt: case Opcode::sge: case Opcode::uge: {
                if (!inst->operand(0) || inst->operand(0)->type().is_float()) return make_leaf();
                BVNodeId a = encode(inst->operand(0), depth + 1);
                BVNodeId b = encode(inst->operand(1), depth + 1);
                if (a == kInvalidBV || b == kInvalidBV) return make_leaf();
                uint32_t cw = std::max(ctx.width(a), ctx.width(b));
                a = coerce_gpu_bv(ctx, a, cw, false);
                b = coerce_gpu_bv(ctx, b, cw, false);
                BVOp cop = (op == Opcode::eq)  ? BVOp::Eq  : (op == Opcode::ne)  ? BVOp::Ne
                         : (op == Opcode::slt) ? BVOp::Slt : (op == Opcode::ult) ? BVOp::Ult
                         : (op == Opcode::sle) ? BVOp::Sle : (op == Opcode::ule) ? BVOp::Ule
                         : (op == Opcode::sgt) ? BVOp::Sgt : (op == Opcode::ugt) ? BVOp::Ugt
                         : (op == Opcode::sge) ? BVOp::Sge : BVOp::Uge;
                res = ctx.make_zext(ctx.make_binary(cop, a, b, 1), 31);
                break;
            }
            case Opcode::select: {
                if (inst->type().is_float()) return make_leaf();
                BVNodeId c = encode(inst->operand(0), depth + 1);
                BVNodeId t = encode(inst->operand(1), depth + 1);
                BVNodeId f = encode(inst->operand(2), depth + 1);
                if (c == kInvalidBV || t == kInvalidBV || f == kInvalidBV) return make_leaf();
                BVNodeId c_bit = ctx.make_binary(BVOp::Ne, c, ctx.make_const(0, ctx.width(c)), 1);
                res = ctx.make_ite(c_bit, coerce_gpu_bv(ctx, t, w, false), coerce_gpu_bv(ctx, f, w, false));
                break;
            }
            case Opcode::call: {
                std::string_view sym = inst->symbol();
                if (sym == "ptx_tid_x" || sym == "ptx_tid_y" || sym == "ptx_tid_z") {
                    res = get_special_u32(sym, 0x3FFu, 0, 1023);
                    break;
                }
                if (sym == "ptx_ntid_x" || sym == "ptx_ntid_y" || sym == "ptx_ntid_z") {
                    res = get_special_u32(sym, 0x7FFu, 1, 1024);
                    break;
                }
                if (sym == "ptx_ctaid_x" || sym == "ptx_ctaid_y" || sym == "ptx_ctaid_z") {
                    res = get_special_u32(sym, 0xFFFFu, 0, 65535);
                    break;
                }
                if (sym == "ptx_nctaid_x" || sym == "ptx_nctaid_y" || sym == "ptx_nctaid_z") {
                    res = get_special_u32(sym, 0xFFFFu, 1, 65535);
                    break;
                }
                if (sym == "ptx_laneid" || sym == "ptx_lane_id") {
                    res = get_special_u32("ptx_laneid", 31u, 0, 31);
                    break;
                }
                if (sym == "ptx_global_tid_x" || sym == "ptx_global_id_x") {
                    res = get_special_u32("ptx_global_tid_x", 0x3FFFFFFu, 0, 0x3FFFFFFu);
                    break;
                }
                std::vector<BVNodeId> args;
                bool ok = true;
                for (size_t i = 0; i < inst->operand_count(); ++i) {
                    BVNodeId ai = encode(inst->operand(i), depth + 1);
                    if (ai == kInvalidBV) { ok = false; break; }
                    args.push_back(ai);
                }
                if (ok) {
                    res = verify::try_eval_gpu_call_bv(ctx, sym, args);
                    if (res != kInvalidBV) break;
                }
                return make_leaf();
            }
            default:
                return make_leaf();
        }

        cache[v] = res;
        add_range_constraint(v, res);
        return res;
    }

    BVNodeId encode_bool(const Value* cond_val) {
        BVNodeId bv = encode(cond_val, 0);
        if (bv == kInvalidBV) return kInvalidBV;
        if (ctx.width(bv) == 1) return bv;
        return ctx.make_binary(BVOp::Ne, bv, ctx.make_const(0, ctx.width(bv)), 1);
    }
};

void remove_dead_pure_insts(Function& fn) {
    bool progress = true;
    while (progress) {
        progress = false;
        auto uses = compute_use_counts(fn);
        for (BasicBlock* bb : fn.blocks()) {
            if (!bb) continue;
            Instruction* cur = bb->head();
            while (cur) {
                Instruction* next = cur->next();
                bool is_pure_gpu_call =
                    (cur->opcode() == Opcode::call &&
                     (cur->symbol() == "ptx_mul_hi_u32" || cur->symbol() == "ptx_mul_wide_u32" ||
                      cur->symbol() == "ptx_mul_wide_s32" || cur->symbol() == "ptx_mad_lo_u32" ||
                      cur->symbol() == "ptx_tid_x" || cur->symbol() == "ptx_tid_y" || cur->symbol() == "ptx_tid_z" ||
                      cur->symbol() == "ptx_ntid_x" || cur->symbol() == "ptx_ntid_y" || cur->symbol() == "ptx_ntid_z" ||
                      cur->symbol() == "ptx_ctaid_x" || cur->symbol() == "ptx_ctaid_y" || cur->symbol() == "ptx_ctaid_z" ||
                      cur->symbol() == "ptx_nctaid_x" || cur->symbol() == "ptx_nctaid_y" || cur->symbol() == "ptx_nctaid_z" ||
                      cur->symbol() == "ptx_laneid" || cur->symbol() == "ptx_lane_id" ||
                      cur->symbol() == "ptx_global_tid_x" || cur->symbol() == "ptx_global_id_x"));
                if ((!cur->has_side_effects() || is_pure_gpu_call) &&
                    cur->produces_value() && uses[cur->result()] == 0) {
                    bb->remove_instruction(cur);
                    progress = true;
                }
                cur = next;
            }
        }
    }
}

Value* emit_magic_udiv_mir(Builder& b, Value* lhs, const MagicConstU32& mag, GpuSuperoptEncoder& enc, BVNodeId lhs_bv) {
    Value* x = lhs;
    BVNodeId x_bv = lhs_bv;
    if (mag.pre_shift > 0) {
        Value* sh_c = b.build_iconst_i32(static_cast<int32_t>(mag.pre_shift));
        x = b.build_lshr(x, sh_c);
        x_bv = enc.ctx.make_binary(BVOp::LShr, x_bv, enc.ctx.make_const(mag.pre_shift, 32), 32);
        enc.cache[x] = x_bv;
    }
    Value* m_c = b.build_iconst_i32(static_cast<int32_t>(mag.multiplier));
    Value* hi = b.build_call("ptx_mul_hi_u32", Type::i32(), {x, m_c});
    BVNodeId prod64 = enc.ctx.make_binary(
        BVOp::Mul, enc.ctx.make_zext(x_bv, 32), enc.ctx.make_const(mag.multiplier, 64), 64);
    BVNodeId hi_bv = enc.ctx.make_extract(prod64, 63, 32);
    enc.cache[hi] = hi_bv;

    if (!mag.needs_add) {
        if (mag.post_shift > 0) {
            Value* psh_c = b.build_iconst_i32(static_cast<int32_t>(mag.post_shift));
            Value* q = b.build_lshr(hi, psh_c);
            BVNodeId q_bv = enc.ctx.make_binary(BVOp::LShr, hi_bv, enc.ctx.make_const(mag.post_shift, 32), 32);
            enc.cache[q] = q_bv;
            return q;
        }
        return hi;
    }

    Value* t = b.build_sub(x, hi);
    BVNodeId t_bv = enc.ctx.make_binary(BVOp::Sub, x_bv, hi_bv, 32);
    enc.cache[t] = t_bv;

    Value* one = b.build_iconst_i32(1);
    Value* t2 = b.build_lshr(t, one);
    BVNodeId t2_bv = enc.ctx.make_binary(BVOp::LShr, t_bv, enc.ctx.make_const(1, 32), 32);
    enc.cache[t2] = t2_bv;

    Value* t3 = b.build_add(t2, hi);
    BVNodeId t3_bv = enc.ctx.make_binary(BVOp::Add, t2_bv, hi_bv, 32);
    enc.cache[t3] = t3_bv;

    if (mag.post_shift > 1) {
        Value* psh_c = b.build_iconst_i32(static_cast<int32_t>(mag.post_shift - 1));
        Value* q = b.build_lshr(t3, psh_c);
        BVNodeId q_bv = enc.ctx.make_binary(BVOp::LShr, t3_bv, enc.ctx.make_const(mag.post_shift - 1, 32), 32);
        enc.cache[q] = q_bv;
        return q;
    }
    return t3;
}

} // namespace

MagicU32 compute_magic_u32(uint32_t divisor) noexcept {
    MagicConstU32 mc = verify::compute_magic_const_u32(divisor);
    return MagicU32{mc.multiplier, mc.pre_shift, mc.post_shift, mc.needs_add};
}

bool verify_magic_u32_with_solver(
    uint32_t divisor,
    const MagicU32& magic,
    uint32_t max_sat_conflicts) {
    if (divisor <= 1) return false;
    BVContext ctx;
    BVNodeId x = ctx.make_var("x", 32, false);
    MagicConstU32 mc{magic.multiplier, magic.pre_shift, magic.post_shift, magic.needs_add};
    BVNodeId cand_q = verify::build_magic_udiv_u32(ctx, x, mc);
    BVNodeId ref_q = ctx.make_binary(BVOp::UDiv, x, ctx.make_const(divisor, 32), 32);

    // 1. Full 32-bit symbolic equivalence against reference UDiv lowering
    BVNodeId bad_ref = ctx.make_binary(BVOp::Ne, cand_q, ref_q, 1);
    if (verify::solve_nonzero(ctx, bad_ref, max_sat_conflicts).status != SolveOutcome::Unsat) {
        return false;
    }

    // 2. Direct Euclidean division specification check (0 <= x - q*d < d) on symbolic window
    BVNodeId x_low = ctx.make_binary(BVOp::And, x, ctx.make_const(0xFFu, 32), 32);
    BVNodeId q_low = verify::build_magic_udiv_u32(ctx, x_low, mc);
    BVNodeId x64 = ctx.make_zext(x_low, 32);
    BVNodeId q64 = ctx.make_zext(q_low, 32);
    BVNodeId d64 = ctx.make_const(divisor, 64);
    BVNodeId qd64 = ctx.make_binary(BVOp::Mul, q64, d64, 64);
    BVNodeId rem64 = ctx.make_binary(BVOp::Sub, x64, qd64, 64);
    BVNodeId bad_over = ctx.make_binary(BVOp::Ugt, qd64, x64, 1);
    BVNodeId bad_rem = ctx.make_binary(BVOp::Uge, rem64, d64, 1);
    BVNodeId bad_euclid = ctx.make_binary(BVOp::Or, bad_over, bad_rem, 1);

    SolveOutcome res = verify::solve_nonzero(ctx, bad_euclid, max_sat_conflicts);
    return res.status == SolveOutcome::Unsat;
}

bool run_gpu_mir_superopt(
    Function& fn,
    Module& mod,
    const SuperoptOptions& opts) {

    if (fn.blocks().empty() || !fn.entry_block()) return false;

    fn.rebuild_cfg_predecessors();
    DominatorTree dom(fn);
    LoopAnalysis loops(fn, dom);
    RangeAnalysis ra(fn, dom, loops);

    bool changed = false;

    for (BasicBlock* bb : fn.blocks()) {
        if (!bb) continue;

        std::vector<const BasicBlock*> dom_chain;
        for (const BasicBlock* d : fn.blocks()) {
            if (d && d != bb && dom.dominates(d, bb)) {
                dom_chain.push_back(d);
            }
        }

        BVContext ctx;
        GpuSuperoptEncoder enc{ctx, ra, bb, std::max<uint32_t>(opts.max_window_depth, 8u), {}, {}, {}};
        std::vector<BVNodeId> assumptions;

        for (const BasicBlock* dbb : dom_chain) {
            for (const Instruction* inst : *dbb) {
                if (inst && inst->opcode() == Opcode::guard && inst->operand(0)) {
                    BVNodeId g = enc.encode_bool(inst->operand(0));
                    if (g != kInvalidBV) assumptions.push_back(g);
                }
            }
        }
        std::vector<const BasicBlock*> all_doms = dom_chain;
        all_doms.push_back(bb);
        for (const BasicBlock* anc : all_doms) {
            if (anc->predecessors().size() == 1) {
                const BasicBlock* pred = anc->predecessors()[0];
                const Instruction* term = pred ? pred->terminator() : nullptr;
                if (term && term->opcode() == Opcode::br_if &&
                    term->true_target().block != term->false_target().block &&
                    term->operand(0)) {
                    BVNodeId c_bit = enc.encode_bool(term->operand(0));
                    if (c_bit != kInvalidBV) {
                        if (term->true_target().block == anc) {
                            assumptions.push_back(c_bit);
                        } else if (term->false_target().block == anc) {
                            assumptions.push_back(ctx.make_unary(BVOp::Not, c_bit, 1));
                        }
                    }
                }
            }
        }

        auto build_path_cond = [&]() -> BVNodeId {
            BVNodeId pc = ctx.make_const(1, 1);
            for (BVNodeId a : assumptions) pc = ctx.make_binary(BVOp::And, pc, a, 1);
            for (BVNodeId r : enc.range_assumptions) pc = ctx.make_binary(BVOp::And, pc, r, 1);
            return pc;
        };

        auto prove_unsat = [&](BVNodeId bad_cond) -> bool {
            if (opts.stats) opts.stats->solver_queries++;
            SolveOutcome out = verify::solve_nonzero(ctx, bad_cond, opts.max_sat_conflicts);
            if (out.status == SolveOutcome::Unsat) {
                verify::debug_verify_unsat_samples(ctx, bad_cond);
                return true;
            }
            return false;
        };

        Instruction* cur = bb->head();
        while (cur) {
            Instruction* next = cur->next();

            if (cur->opcode() == Opcode::guard && cur->operand(0)) {
                BVNodeId g = enc.encode_bool(cur->operand(0));
                if (g != kInvalidBV) assumptions.push_back(g);
                cur = next;
                continue;
            }

            if (cur->has_side_effects() || !cur->produces_value()) {
                cur = next;
                continue;
            }

            Opcode op = cur->opcode();
            Type res_t = cur->type();

            // ---------------------------------------------------------------
            // 1. 32-bit Constant UDiv / URem & Non-Negative SDiv / SMod
            //    Magic-Multiplier Strength Reduction
            // ---------------------------------------------------------------
            if (res_t == Type::i32() &&
                (op == Opcode::udiv || op == Opcode::umod || op == Opcode::sdiv || op == Opcode::smod)) {
                Value* lhs = cur->operand(0);
                int64_t div_imm = 0;
                if (lhs && get_const_int(cur->operand(1), div_imm)) {
                    uint32_t d = static_cast<uint32_t>(div_imm);
                    bool is_signed = (op == Opcode::sdiv || op == Opcode::smod);
                    bool valid_d = is_signed ? (static_cast<int32_t>(d) > 1) : (d > 1);
                    if (valid_d) {
                        BVNodeId orig_bv = enc.encode(cur->result(), 0);
                        BVNodeId lhs_bv = enc.encode(lhs, 0);
                        BVNodeId pc = build_path_cond();
                        if (orig_bv != kInvalidBV && lhs_bv != kInvalidBV) {
                            bool sign_ok = true;
                            if (is_signed) {
                                BVNodeId is_neg = ctx.make_binary(BVOp::Slt, lhs_bv, ctx.make_const(0, 32), 1);
                                sign_ok = prove_unsat(ctx.make_binary(BVOp::And, pc, is_neg, 1));
                            }
                            if (sign_ok) {
                                bool is_rem = (op == Opcode::umod || op == Opcode::smod);
                                if (std::has_single_bit(d)) {
                                    uint32_t sh = static_cast<uint32_t>(std::countr_zero(d));
                                    BVNodeId cand_bv = is_rem
                                        ? ctx.make_binary(BVOp::And, lhs_bv, ctx.make_const(d - 1, 32), 32)
                                        : ctx.make_binary(BVOp::LShr, lhs_bv, ctx.make_const(sh, 32), 32);
                                    BVNodeId diff = ctx.make_binary(BVOp::Ne, orig_bv, cand_bv, 1);
                                    if (prove_unsat(ctx.make_binary(BVOp::And, pc, diff, 1))) {
                                        Builder b(mod);
                                        b.set_function(&fn);
                                        b.position_before(cur);
                                        Value* repl = is_rem
                                            ? b.build_and(lhs, b.build_iconst_i32(static_cast<int32_t>(d - 1)))
                                            : b.build_lshr(lhs, b.build_iconst_i32(static_cast<int32_t>(sh)));
                                        enc.cache[repl] = cand_bv;
                                        replace_all_uses(fn, cur->result(), repl);
                                        if (opts.stats) opts.stats->dags_synthesized++;
                                        changed = true;
                                        cur = next;
                                        continue;
                                    }
                                } else {
                                    MagicConstU32 mag = verify::compute_magic_const_u32(d);
                                    BVNodeId q_bv = verify::build_magic_udiv_u32(ctx, lhs_bv, mag);
                                    BVNodeId cand_bv = is_rem
                                        ? ctx.make_binary(
                                              BVOp::Sub, lhs_bv,
                                              ctx.make_binary(BVOp::Mul, q_bv, ctx.make_const(d, 32), 32), 32)
                                        : q_bv;
                                    BVNodeId ref_u_bv = ctx.make_binary(
                                        is_rem ? BVOp::URem : BVOp::UDiv, lhs_bv, ctx.make_const(d, 32), 32);
                                    BVNodeId diff = ctx.make_binary(BVOp::Ne, ref_u_bv, cand_bv, 1);
                                    if (prove_unsat(ctx.make_binary(BVOp::And, pc, diff, 1))) {
                                        Builder b(mod);
                                        b.set_function(&fn);
                                        b.position_before(cur);
                                        Value* q_val = emit_magic_udiv_mir(b, lhs, mag, enc, lhs_bv);
                                        Value* repl = q_val;
                                        if (is_rem) {
                                            Value* d_val = b.build_iconst_i32(static_cast<int32_t>(d));
                                            Value* prod_val = b.build_mul(q_val, d_val);
                                            enc.cache[prod_val] = ctx.make_binary(BVOp::Mul, q_bv, ctx.make_const(d, 32), 32);
                                            repl = b.build_sub(lhs, prod_val);
                                            enc.cache[repl] = cand_bv;
                                        }
                                        replace_all_uses(fn, cur->result(), repl);
                                        if (opts.stats) {
                                            opts.stats->magic_divs_reduced++;
                                            opts.stats->dags_synthesized++;
                                        }
                                        changed = true;
                                        cur = next;
                                        continue;
                                    }
                                }
                            }
                        }
                    }
                }
            }

            // ---------------------------------------------------------------
            // 2. 32-bit Index Narrowing for 64-bit Add / Sub / Mul / Shl
            // ---------------------------------------------------------------
            if (res_t == Type::i64() &&
                (op == Opcode::add || op == Opcode::sub || op == Opcode::mul || op == Opcode::shl)) {
                Value* op0 = cur->operand(0);
                Value* op1 = cur->operand(1);

                auto extract_i32_source = [&](Value* v, Value*& out_v32, int64_t& out_c32, bool& is_ext) -> bool {
                    out_v32 = nullptr;
                    out_c32 = 0;
                    is_ext = false;
                    if (!v) return false;
                    int64_t c = 0;
                    if (get_const_int(v, c)) {
                        if (op == Opcode::shl) {
                            if (c >= 0 && c < 31) { out_c32 = c; return true; }
                            return false;
                        }
                        if (c >= 0 && c <= INT32_MAX) { out_c32 = c; return true; }
                        return false;
                    }
                    if (v->is_instruction()) {
                        Instruction* def = v->defining_instruction();
                        if (def && (def->opcode() == Opcode::sext_i64 || def->opcode() == Opcode::zext_i64) &&
                            def->operand(0) && def->operand(0)->type() == Type::i32()) {
                            out_v32 = def->operand(0);
                            is_ext = true;
                            return true;
                        }
                    }
                    return false;
                };

                Value* v0_32 = nullptr;
                Value* v1_32 = nullptr;
                int64_t c0_32 = 0, c1_32 = 0;
                bool ext0 = false, ext1 = false;
                if (extract_i32_source(op0, v0_32, c0_32, ext0) &&
                    extract_i32_source(op1, v1_32, c1_32, ext1) &&
                    (ext0 || ext1)) {
                    BVNodeId orig_bv = enc.encode(cur->result(), 0);
                    BVNodeId bv0_32 = v0_32 ? enc.encode(v0_32, 0) : ctx.make_const(static_cast<uint32_t>(c0_32), 32);
                    BVNodeId bv1_32 = v1_32 ? enc.encode(v1_32, 0) : ctx.make_const(static_cast<uint32_t>(c1_32), 32);
                    BVNodeId pc = build_path_cond();

                    if (orig_bv != kInvalidBV && bv0_32 != kInvalidBV && bv1_32 != kInvalidBV) {
                        BVOp bop = (op == Opcode::add) ? BVOp::Add
                                 : (op == Opcode::sub) ? BVOp::Sub
                                 : (op == Opcode::mul) ? BVOp::Mul : BVOp::Shl;
                        BVNodeId narrow_bv32 = ctx.make_binary(bop, bv0_32, bv1_32, 32);
                        BVNodeId cand_zext64 = ctx.make_zext(narrow_bv32, 32);
                        // Prove that the 64-bit index fits in [0, INT32_MAX] (bits [63:31] == 0)
                        // so both zext_i64(narrow_bv32) and sext_i64(narrow_bv32) equal orig_bv.
                        BVNodeId hi_bits = ctx.make_extract(orig_bv, 63, 31);
                        BVNodeId bad = ctx.make_binary(
                            BVOp::And, pc, ctx.make_binary(BVOp::Ne, hi_bits, ctx.make_const(0, 33), 1), 1);
                        if (prove_unsat(bad)) {
                            Builder b(mod);
                            b.set_function(&fn);
                            b.position_before(cur);
                            if (!v0_32) {
                                v0_32 = b.build_iconst_i32(static_cast<int32_t>(c0_32));
                                enc.cache[v0_32] = bv0_32;
                            }
                            if (!v1_32) {
                                v1_32 = b.build_iconst_i32(static_cast<int32_t>(c1_32));
                                enc.cache[v1_32] = bv1_32;
                            }
                            Value* res32 = (op == Opcode::add) ? b.build_add(v0_32, v1_32)
                                         : (op == Opcode::sub) ? b.build_sub(v0_32, v1_32)
                                         : (op == Opcode::mul) ? b.build_mul(v0_32, v1_32)
                                                               : b.build_shl(v0_32, v1_32);
                            enc.cache[res32] = narrow_bv32;
                            Value* ext64 = b.build_zext_i64(res32);
                            enc.cache[ext64] = cand_zext64;
                            replace_all_uses(fn, cur->result(), ext64);
                            if (opts.stats) {
                                opts.stats->indices_narrowed++;
                                opts.stats->dags_synthesized++;
                            }
                            changed = true;
                            cur = next;
                            continue;
                        }
                    }
                }
            }

            cur = next;
        }
    }

    if (changed) {
        remove_dead_pure_insts(fn);
    }
    return changed;
}

bool run_gpu_mir_superopt(Function& fn, Module& mod) {
    SuperoptOptions opts;
    opts.enable_gpu_superopt = true;
    return run_gpu_mir_superopt(fn, mod, opts);
}

bool run_gpu_mir_superopt(Module& mod, const SuperoptOptions& opts) {
    bool changed = false;
    for (Function* fn : mod.functions()) {
        if (fn && !fn->blocks().empty()) {
            changed |= run_gpu_mir_superopt(*fn, mod, opts);
        }
    }
    return changed;
}

bool run_gpu_mir_superopt(Module& mod) {
    SuperoptOptions opts;
    opts.enable_gpu_superopt = true;
    return run_gpu_mir_superopt(mod, opts);
}

} // namespace brass
