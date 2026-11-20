#include "lir_sym_exec_arm64.hpp"
#include <array>

namespace brass::verify {

namespace {

using namespace brass::codegen;

std::array<BVNodeId, 4> extract_4x32(BVContext& ctx, BVNodeId lo, BVNodeId hi) {
    return {
        ctx.make_extract(lo, 31, 0),
        ctx.make_extract(lo, 63, 32),
        ctx.make_extract(hi, 31, 0),
        ctx.make_extract(hi, 63, 32)
    };
}

std::pair<BVNodeId, BVNodeId> pack_4x32(BVContext& ctx, const std::array<BVNodeId, 4>& lanes) {
    return {ctx.make_concat(lanes[1], lanes[0]), ctx.make_concat(lanes[3], lanes[2])};
}

BVNodeId cmp_f_mask(BVContext& ctx, int pred, BVNodeId a, BVNodeId b, uint32_t w) {
    BVNodeId cond = ctx.make_const(0, 1);
    switch (pred) {
        case 0x00: cond = ctx.make_binary(BVOp::FEq, a, b, 1); break;
        case 0x04: cond = ctx.make_binary(BVOp::FNe, a, b, 1); break;
        case 0x11: cond = ctx.make_binary(BVOp::FLt, a, b, 1); break;
        case 0x12: cond = ctx.make_binary(BVOp::FLe, a, b, 1); break;
        case 0x1E: cond = ctx.make_binary(BVOp::FGt, a, b, 1); break;
        case 0x1D: cond = ctx.make_binary(BVOp::FGe, a, b, 1); break;
        default:   cond = ctx.make_const(0, 1); break;
    }
    return ctx.make_sext(cond, w - 1);
}

BVNodeId mask_from_cond(BVContext& ctx, BVNodeId cond, uint32_t w) {
    return ctx.make_sext(cond, w - 1);
}

BVNodeId round_bv_op_for_mode(int mode) noexcept {
    switch (mode & 3) {
        case 1: return static_cast<BVNodeId>(BVOp::FFloor);
        case 2: return static_cast<BVNodeId>(BVOp::FCeil);
        case 3: return static_cast<BVNodeId>(BVOp::FTrunc);
        default: return static_cast<BVNodeId>(BVOp::FNearbyInt);
    }
}

BVNodeId fp_to_si_clamped32(BVContext& ctx, BVNodeId f, uint32_t src_w) {
    // Matches MIR/x64/AArch64-fixed Vcvttps2dq / Vcvttpd2dq:
    // Values in [-2^31, +2^31) convert via truncation; NaN or out-of-range -> INT32_MIN (0x80000000).
    BVNodeId lo_bound = ctx.make_const(src_w == 32 ? 0xCF000000ULL : 0xC1E0000000000000ULL, src_w); // -2^31
    BVNodeId hi_bound = ctx.make_const(src_w == 32 ? 0x4F000000ULL : 0x41E0000000000000ULL, src_w); // +2^31
    BVNodeId ge_lo = ctx.make_binary(BVOp::FGe, f, lo_bound, 1);
    BVNodeId lt_hi = ctx.make_binary(BVOp::FLt, f, hi_bound, 1);
    BVNodeId in_range = ctx.make_binary(BVOp::And, ge_lo, lt_hi, 1);
    BVNodeId conv = ctx.make_unary(BVOp::FpToSi, f, 32);
    return ctx.make_ite(in_range, conv, ctx.make_const(0x80000000ULL, 32));
}

} // namespace

bool execute_arm64_vec_inst(BVContext& ctx, Arm64InitEnv& env, Arm64ExecState& st, const LirInst& inst) {
    auto read_op = [&](const LirOperand& o, uint32_t w) { return arm64_read_op(ctx, env, st, o, w); };
    auto write_op = [&](const LirOperand& d, BVNodeId v) { arm64_write_op(ctx, env, st, d, v); };
    auto read_v = [&](const LirOperand& o) { return arm64_read_vec128(ctx, env, st, o); };
    auto write_v = [&](const LirOperand& d, BVNodeId lo, BVNodeId hi) { arm64_write_vec128(ctx, env, st, d, lo, hi); };

    const LirOpcode op = inst.opcode;

    switch (op) {
        case LirOpcode::Movaps:  case LirOpcode::Movups:
        case LirOpcode::Vmovaps: case LirOpcode::Vmovups: {
            if (inst.defs.empty() || inst.uses.empty()) return true;
            auto [lo, hi] = read_v(inst.uses[0]);
            write_v(inst.defs[0], lo, hi);
            return true;
        }

        case LirOpcode::Xorpd:  case LirOpcode::Xorps:  case LirOpcode::Pxor:
        case LirOpcode::Vxorps: case LirOpcode::Vxorpd: case LirOpcode::Vpxor:
        case LirOpcode::Pand:   case LirOpcode::Vandps: case LirOpcode::Vandpd: case LirOpcode::Vpand:
        case LirOpcode::Por:    case LirOpcode::Vorps:  case LirOpcode::Vorpd:  case LirOpcode::Vpor: {
            if (inst.defs.empty() || inst.uses.empty()) return true;
            const auto& lhs_op = (inst.uses.size() >= 2) ? inst.uses[0] : inst.defs[0];
            auto [a_lo, a_hi] = read_v(lhs_op);
            auto [b_lo, b_hi] = read_v(inst.uses.back());
            BVOp bop = (op == LirOpcode::Pand || op == LirOpcode::Vandps ||
                        op == LirOpcode::Vandpd || op == LirOpcode::Vpand) ? BVOp::And
                     : (op == LirOpcode::Por || op == LirOpcode::Vorps ||
                        op == LirOpcode::Vorpd || op == LirOpcode::Vpor) ? BVOp::Or : BVOp::Xor;
            write_v(inst.defs[0], ctx.make_binary(bop, a_lo, b_lo, 64), ctx.make_binary(bop, a_hi, b_hi, 64));
            return true;
        }

        case LirOpcode::Pandn: {
            if (inst.defs.empty() || inst.uses.empty()) return true;
            const auto& lhs_op = (inst.uses.size() >= 2) ? inst.uses[0] : inst.defs[0];
            auto [a_lo, a_hi] = read_v(lhs_op);
            auto [b_lo, b_hi] = read_v(inst.uses.back());
            write_v(inst.defs[0],
                    ctx.make_binary(BVOp::And, ctx.make_unary(BVOp::Not, a_lo, 64), b_lo, 64),
                    ctx.make_binary(BVOp::And, ctx.make_unary(BVOp::Not, a_hi, 64), b_hi, 64));
            return true;
        }

        case LirOpcode::Pnot: {
            if (inst.defs.empty() || inst.uses.empty()) return true;
            auto [lo, hi] = read_v(inst.uses[0]);
            write_v(inst.defs[0], ctx.make_unary(BVOp::Not, lo, 64), ctx.make_unary(BVOp::Not, hi, 64));
            return true;
        }

        // 4x32 float & integer binary lane ops
        case LirOpcode::Addps: case LirOpcode::Vaddps:
        case LirOpcode::Subps: case LirOpcode::Vsubps:
        case LirOpcode::Mulps: case LirOpcode::Vmulps:
        case LirOpcode::Divps: case LirOpcode::Vdivps:
        case LirOpcode::Minps: case LirOpcode::Vminps:
        case LirOpcode::Maxps: case LirOpcode::Vmaxps:
        case LirOpcode::Paddd: case LirOpcode::Vpaddd:
        case LirOpcode::Psubd: case LirOpcode::Vpsubd:
        case LirOpcode::Pmulld: case LirOpcode::Vpmulld:
        case LirOpcode::Pminsd: case LirOpcode::Vpminsd:
        case LirOpcode::Pmaxsd: case LirOpcode::Vpmaxsd:
        case LirOpcode::Pcmpeqd: case LirOpcode::Vpcmpeqd:
        case LirOpcode::Vpcmpgtd: {
            if (inst.defs.empty() || inst.uses.empty()) return true;
            const auto& lhs_op = (inst.uses.size() >= 2) ? inst.uses[0] : inst.defs[0];
            auto [a_lo, a_hi] = read_v(lhs_op);
            auto [b_lo, b_hi] = read_v(inst.uses.back());
            auto a = extract_4x32(ctx, a_lo, a_hi);
            auto b = extract_4x32(ctx, b_lo, b_hi);
            std::array<BVNodeId, 4> r{};
            for (size_t i = 0; i < 4; ++i) {
                switch (op) {
                    case LirOpcode::Addps: case LirOpcode::Vaddps: r[i] = ctx.make_binary(BVOp::FAdd, a[i], b[i], 32); break;
                    case LirOpcode::Subps: case LirOpcode::Vsubps: r[i] = ctx.make_binary(BVOp::FSub, a[i], b[i], 32); break;
                    case LirOpcode::Mulps: case LirOpcode::Vmulps: r[i] = ctx.make_binary(BVOp::FMul, a[i], b[i], 32); break;
                    case LirOpcode::Divps: case LirOpcode::Vdivps: r[i] = ctx.make_binary(BVOp::FDiv, a[i], b[i], 32); break;
                    case LirOpcode::Minps: case LirOpcode::Vminps: r[i] = ctx.make_binary(BVOp::FMin, a[i], b[i], 32); break;
                    case LirOpcode::Maxps: case LirOpcode::Vmaxps: r[i] = ctx.make_binary(BVOp::FMax, a[i], b[i], 32); break;
                    case LirOpcode::Paddd: case LirOpcode::Vpaddd: r[i] = ctx.make_binary(BVOp::Add, a[i], b[i], 32); break;
                    case LirOpcode::Psubd: case LirOpcode::Vpsubd: r[i] = ctx.make_binary(BVOp::Sub, a[i], b[i], 32); break;
                    case LirOpcode::Pmulld: case LirOpcode::Vpmulld: r[i] = ctx.make_binary(BVOp::Mul, a[i], b[i], 32); break;
                    case LirOpcode::Pminsd: case LirOpcode::Vpminsd:
                        r[i] = ctx.make_ite(ctx.make_binary(BVOp::Slt, a[i], b[i], 1), a[i], b[i]);
                        break;
                    case LirOpcode::Pmaxsd: case LirOpcode::Vpmaxsd:
                        r[i] = ctx.make_ite(ctx.make_binary(BVOp::Sgt, a[i], b[i], 1), a[i], b[i]);
                        break;
                    case LirOpcode::Pcmpeqd: case LirOpcode::Vpcmpeqd:
                        r[i] = mask_from_cond(ctx, ctx.make_binary(BVOp::Eq, a[i], b[i], 1), 32);
                        break;
                    case LirOpcode::Vpcmpgtd:
                        r[i] = mask_from_cond(ctx, ctx.make_binary(BVOp::Sgt, a[i], b[i], 1), 32);
                        break;
                    default: break;
                }
            }
            auto [r_lo, r_hi] = pack_4x32(ctx, r);
            write_v(inst.defs[0], r_lo, r_hi);
            return true;
        }

        // 4x32 float unary ops
        case LirOpcode::Sqrtps: case LirOpcode::Vsqrtps:
        case LirOpcode::Fneg4s: {
            if (inst.defs.empty() || inst.uses.empty()) return true;
            auto [lo, hi] = read_v(inst.uses[0]);
            auto a = extract_4x32(ctx, lo, hi);
            BVOp uop = (op == LirOpcode::Fneg4s) ? BVOp::FNeg : BVOp::FSqrt;
            std::array<BVNodeId, 4> r{};
            for (size_t i = 0; i < 4; ++i) r[i] = ctx.make_unary(uop, a[i], 32);
            auto [r_lo, r_hi] = pack_4x32(ctx, r);
            write_v(inst.defs[0], r_lo, r_hi);
            return true;
        }

        // 4x32 float FMA
        case LirOpcode::Vfmadd213ps:
        case LirOpcode::Vfmadd231ps: {
            if (inst.defs.empty() || inst.uses.size() < 2) return true;
            bool is_213 = (op == LirOpcode::Vfmadd213ps);
            const auto& op1 = (inst.uses.size() >= 3) ? inst.uses[0] : inst.defs[0];
            size_t s2_idx = (inst.uses.size() >= 3) ? 1u : 0u;
            size_t s3_idx = (inst.uses.size() >= 3) ? 2u : 1u;
            auto [l1, h1] = read_v(op1);
            auto [l2, h2] = read_v(inst.uses[s2_idx]);
            auto [l3, h3] = read_v(inst.uses[s3_idx]);
            auto a1 = extract_4x32(ctx, l1, h1);
            auto a2 = extract_4x32(ctx, l2, h2);
            auto a3 = extract_4x32(ctx, l3, h3);
            std::array<BVNodeId, 4> r{};
            for (size_t i = 0; i < 4; ++i) {
                BVNodeId sn = is_213 ? a1[i] : a2[i];
                BVNodeId sm = is_213 ? a2[i] : a3[i];
                BVNodeId sa = is_213 ? a3[i] : a1[i];
                r[i] = (env.shared_mem != nullptr)
                    ? env.shared_mem->eval_uninterpreted_call(ctx, "fmaf", {sn, sm, sa}, Type::f32())
                    : ctx.make_ternary(BVOp::FFma, sn, sm, sa, 32);
            }
            auto [r_lo, r_hi] = pack_4x32(ctx, r);
            write_v(inst.defs[0], r_lo, r_hi);
            return true;
        }

        // 2x64 float & integer binary lane ops
        case LirOpcode::Paddq:  case LirOpcode::Vpaddq:
        case LirOpcode::Psubq:  case LirOpcode::Vpsubq:
        case LirOpcode::Addpd:  case LirOpcode::Vaddpd:
        case LirOpcode::Subpd:  case LirOpcode::Vsubpd:
        case LirOpcode::Mulpd:  case LirOpcode::Vmulpd:
        case LirOpcode::Divpd:  case LirOpcode::Vdivpd:
        case LirOpcode::Minpd:  case LirOpcode::Vminpd:
        case LirOpcode::Maxpd:  case LirOpcode::Vmaxpd: {
            if (inst.defs.empty() || inst.uses.empty()) return true;
            const auto& lhs_op = (inst.uses.size() >= 2) ? inst.uses[0] : inst.defs[0];
            auto [a_lo, a_hi] = read_v(lhs_op);
            auto [b_lo, b_hi] = read_v(inst.uses.back());
            BVOp vop = (op == LirOpcode::Paddq || op == LirOpcode::Vpaddq) ? BVOp::Add
                     : (op == LirOpcode::Psubq || op == LirOpcode::Vpsubq) ? BVOp::Sub
                     : (op == LirOpcode::Addpd || op == LirOpcode::Vaddpd) ? BVOp::FAdd
                     : (op == LirOpcode::Subpd || op == LirOpcode::Vsubpd) ? BVOp::FSub
                     : (op == LirOpcode::Mulpd || op == LirOpcode::Vmulpd) ? BVOp::FMul
                     : (op == LirOpcode::Divpd || op == LirOpcode::Vdivpd) ? BVOp::FDiv
                     : (op == LirOpcode::Minpd || op == LirOpcode::Vminpd) ? BVOp::FMin : BVOp::FMax;
            write_v(inst.defs[0], ctx.make_binary(vop, a_lo, b_lo, 64), ctx.make_binary(vop, a_hi, b_hi, 64));
            return true;
        }

        case LirOpcode::Vpcmpeqq:
        case LirOpcode::Vpcmpgtq: {
            if (inst.defs.empty() || inst.uses.size() < 2) return true;
            auto [a_lo, a_hi] = read_v(inst.uses[0]);
            auto [b_lo, b_hi] = read_v(inst.uses[1]);
            BVOp cop = (op == LirOpcode::Vpcmpeqq) ? BVOp::Eq : BVOp::Sgt;
            write_v(inst.defs[0],
                    mask_from_cond(ctx, ctx.make_binary(cop, a_lo, b_lo, 1), 64),
                    mask_from_cond(ctx, ctx.make_binary(cop, a_hi, b_hi, 1), 64));
            return true;
        }

        // 2x64 float unary & FMA ops
        case LirOpcode::Sqrtpd: case LirOpcode::Vsqrtpd:
        case LirOpcode::Fneg2d: {
            if (inst.defs.empty() || inst.uses.empty()) return true;
            auto [lo, hi] = read_v(inst.uses[0]);
            BVOp uop = (op == LirOpcode::Fneg2d) ? BVOp::FNeg : BVOp::FSqrt;
            write_v(inst.defs[0], ctx.make_unary(uop, lo, 64), ctx.make_unary(uop, hi, 64));
            return true;
        }

        case LirOpcode::Vfmadd213pd:
        case LirOpcode::Vfmadd231pd: {
            if (inst.defs.empty() || inst.uses.size() < 2) return true;
            bool is_213 = (op == LirOpcode::Vfmadd213pd);
            const auto& op1 = (inst.uses.size() >= 3) ? inst.uses[0] : inst.defs[0];
            size_t s2_idx = (inst.uses.size() >= 3) ? 1u : 0u;
            size_t s3_idx = (inst.uses.size() >= 3) ? 2u : 1u;
            auto [l1, h1] = read_v(op1);
            auto [l2, h2] = read_v(inst.uses[s2_idx]);
            auto [l3, h3] = read_v(inst.uses[s3_idx]);
            BVNodeId sn0 = is_213 ? l1 : l2, sm0 = is_213 ? l2 : l3, sa0 = is_213 ? l3 : l1;
            BVNodeId sn1 = is_213 ? h1 : h2, sm1 = is_213 ? h2 : h3, sa1 = is_213 ? h3 : h1;
            BVNodeId lo = (env.shared_mem != nullptr)
                ? env.shared_mem->eval_uninterpreted_call(ctx, "fma", {sn0, sm0, sa0}, Type::f64())
                : ctx.make_ternary(BVOp::FFma, sn0, sm0, sa0, 64);
            BVNodeId hi = (env.shared_mem != nullptr)
                ? env.shared_mem->eval_uninterpreted_call(ctx, "fma", {sn1, sm1, sa1}, Type::f64())
                : ctx.make_ternary(BVOp::FFma, sn1, sm1, sa1, 64);
            write_v(inst.defs[0], lo, hi);
            return true;
        }

        // Float vector compares (Vcmpps / Vcmppd)
        case LirOpcode::Vcmpps: {
            if (inst.defs.empty() || inst.uses.size() < 3) return true;
            auto [a_lo, a_hi] = read_v(inst.uses[0]);
            auto [b_lo, b_hi] = read_v(inst.uses[1]);
            int pred = static_cast<int>(inst.uses[2].imm_int);
            auto a = extract_4x32(ctx, a_lo, a_hi);
            auto b = extract_4x32(ctx, b_lo, b_hi);
            std::array<BVNodeId, 4> r{};
            for (size_t i = 0; i < 4; ++i) r[i] = cmp_f_mask(ctx, pred, a[i], b[i], 32);
            auto [r_lo, r_hi] = pack_4x32(ctx, r);
            write_v(inst.defs[0], r_lo, r_hi);
            return true;
        }

        case LirOpcode::Vcmppd: {
            if (inst.defs.empty() || inst.uses.size() < 3) return true;
            auto [a_lo, a_hi] = read_v(inst.uses[0]);
            auto [b_lo, b_hi] = read_v(inst.uses[1]);
            int pred = static_cast<int>(inst.uses[2].imm_int);
            write_v(inst.defs[0], cmp_f_mask(ctx, pred, a_lo, b_lo, 64), cmp_f_mask(ctx, pred, a_hi, b_hi, 64));
            return true;
        }

        // Blend by lane MSB: uses[0]=b (false), uses[1]=a (true), uses[2]=mask
        case LirOpcode::Vblendvps: {
            if (inst.defs.empty() || inst.uses.size() < 3) return true;
            auto [bl, bh] = read_v(inst.uses[0]);
            auto [al, ah] = read_v(inst.uses[1]);
            auto [ml, mh] = read_v(inst.uses[2]);
            auto vb = extract_4x32(ctx, bl, bh);
            auto va = extract_4x32(ctx, al, ah);
            auto vm = extract_4x32(ctx, ml, mh);
            std::array<BVNodeId, 4> r{};
            for (size_t i = 0; i < 4; ++i) r[i] = ctx.make_ite(ctx.make_extract(vm[i], 31, 31), va[i], vb[i]);
            auto [r_lo, r_hi] = pack_4x32(ctx, r);
            write_v(inst.defs[0], r_lo, r_hi);
            return true;
        }

        case LirOpcode::Vblendvpd: {
            if (inst.defs.empty() || inst.uses.size() < 3) return true;
            auto [bl, bh] = read_v(inst.uses[0]);
            auto [al, ah] = read_v(inst.uses[1]);
            auto [ml, mh] = read_v(inst.uses[2]);
            BVNodeId lo = ctx.make_ite(ctx.make_extract(ml, 63, 63), al, bl);
            BVNodeId hi = ctx.make_ite(ctx.make_extract(mh, 63, 63), ah, bh);
            write_v(inst.defs[0], lo, hi);
            return true;
        }

        // Shifts (immediate & per-lane variable)
        case LirOpcode::Pslld: {
            if (inst.defs.empty() || inst.uses.size() < 2) return true;
            auto [lo, hi] = read_v(inst.uses[0]);
            uint32_t sh = static_cast<uint32_t>(inst.uses.back().imm_int) & 31u;
            BVNodeId sh_c = ctx.make_const(sh, 32);
            auto a = extract_4x32(ctx, lo, hi);
            std::array<BVNodeId, 4> r{};
            for (size_t i = 0; i < 4; ++i) r[i] = ctx.make_binary(BVOp::Shl, a[i], sh_c, 32);
            auto [r_lo, r_hi] = pack_4x32(ctx, r);
            write_v(inst.defs[0], r_lo, r_hi);
            return true;
        }

        case LirOpcode::Psrld:
        case LirOpcode::Psrad: {
            if (inst.defs.empty() || inst.uses.size() < 2) return true;
            auto [lo, hi] = read_v(inst.uses[0]);
            uint32_t raw_sh = static_cast<uint32_t>(inst.uses.back().imm_int);
            auto a = extract_4x32(ctx, lo, hi);
            std::array<BVNodeId, 4> r{};
            if (raw_sh >= 32u) {
                if (op == LirOpcode::Psrld) {
                    BVNodeId z = ctx.make_const(0, 32);
                    r = {z, z, z, z};
                } else {
                    BVNodeId sh_c = ctx.make_const(31, 32);
                    for (size_t i = 0; i < 4; ++i) r[i] = ctx.make_binary(BVOp::AShr, a[i], sh_c, 32);
                }
            } else {
                BVNodeId sh_c = ctx.make_const(raw_sh, 32);
                BVOp sop = (op == LirOpcode::Psrld) ? BVOp::LShr : BVOp::AShr;
                for (size_t i = 0; i < 4; ++i) r[i] = ctx.make_binary(sop, a[i], sh_c, 32);
            }
            auto [r_lo, r_hi] = pack_4x32(ctx, r);
            write_v(inst.defs[0], r_lo, r_hi);
            return true;
        }

        case LirOpcode::Psllq: {
            if (inst.defs.empty() || inst.uses.size() < 2) return true;
            auto [lo, hi] = read_v(inst.uses[0]);
            uint32_t sh = static_cast<uint32_t>(inst.uses.back().imm_int) & 63u;
            BVNodeId sh_c = ctx.make_const(sh, 64);
            write_v(inst.defs[0], ctx.make_binary(BVOp::Shl, lo, sh_c, 64), ctx.make_binary(BVOp::Shl, hi, sh_c, 64));
            return true;
        }

        case LirOpcode::Psrlq:
        case LirOpcode::Psraq: {
            if (inst.defs.empty() || inst.uses.size() < 2) return true;
            auto [lo, hi] = read_v(inst.uses[0]);
            uint32_t raw_sh = static_cast<uint32_t>(inst.uses.back().imm_int);
            if (raw_sh >= 64u) {
                if (op == LirOpcode::Psrlq) {
                    BVNodeId z = ctx.make_const(0, 64);
                    write_v(inst.defs[0], z, z);
                } else {
                    BVNodeId sh_c = ctx.make_const(63, 64);
                    write_v(inst.defs[0], ctx.make_binary(BVOp::AShr, lo, sh_c, 64),
                            ctx.make_binary(BVOp::AShr, hi, sh_c, 64));
                }
            } else {
                BVNodeId sh_c = ctx.make_const(raw_sh, 64);
                BVOp sop = (op == LirOpcode::Psrlq) ? BVOp::LShr : BVOp::AShr;
                write_v(inst.defs[0], ctx.make_binary(sop, lo, sh_c, 64), ctx.make_binary(sop, hi, sh_c, 64));
            }
            return true;
        }

        case LirOpcode::Vpsllvd: case LirOpcode::Vpsrlvd: case LirOpcode::Vpsravd: {
            if (inst.defs.empty() || inst.uses.size() < 2) return true;
            auto [al, ah] = read_v(inst.uses[0]);
            auto [sl, sh] = read_v(inst.uses[1]);
            auto va = extract_4x32(ctx, al, ah);
            auto vs = extract_4x32(ctx, sl, sh);
            std::array<BVNodeId, 4> r{};
            BVNodeId c32 = ctx.make_const(32, 32), c31 = ctx.make_const(31, 32);
            for (size_t i = 0; i < 4; ++i) {
                BVNodeId ge32 = ctx.make_binary(BVOp::Uge, vs[i], c32, 1);
                if (op == LirOpcode::Vpsravd) {
                    BVNodeId amt = ctx.make_ite(ge32, c31, vs[i]);
                    r[i] = ctx.make_binary(BVOp::AShr, va[i], amt, 32);
                } else {
                    BVOp sop = (op == LirOpcode::Vpsllvd) ? BVOp::Shl : BVOp::LShr;
                    r[i] = ctx.make_ite(ge32, ctx.make_const(0, 32), ctx.make_binary(sop, va[i], vs[i], 32));
                }
            }
            auto [r_lo, r_hi] = pack_4x32(ctx, r);
            write_v(inst.defs[0], r_lo, r_hi);
            return true;
        }

        case LirOpcode::Vpsllvq: case LirOpcode::Vpsrlvq: case LirOpcode::Vpsravq: {
            if (inst.defs.empty() || inst.uses.size() < 2) return true;
            auto [al, ah] = read_v(inst.uses[0]);
            auto [sl, sh] = read_v(inst.uses[1]);
            BVNodeId c64 = ctx.make_const(64, 64), c63 = ctx.make_const(63, 64), z64 = ctx.make_const(0, 64);
            if (op == LirOpcode::Vpsravq) {
                BVNodeId amt_lo = ctx.make_ite(ctx.make_binary(BVOp::Uge, sl, c64, 1), c63, sl);
                BVNodeId amt_hi = ctx.make_ite(ctx.make_binary(BVOp::Uge, sh, c64, 1), c63, sh);
                write_v(inst.defs[0], ctx.make_binary(BVOp::AShr, al, amt_lo, 64),
                        ctx.make_binary(BVOp::AShr, ah, amt_hi, 64));
            } else {
                BVOp sop = (op == LirOpcode::Vpsllvq) ? BVOp::Shl : BVOp::LShr;
                BVNodeId lo = ctx.make_ite(ctx.make_binary(BVOp::Uge, sl, c64, 1), z64, ctx.make_binary(sop, al, sl, 64));
                BVNodeId hi = ctx.make_ite(ctx.make_binary(BVOp::Uge, sh, c64, 1), z64, ctx.make_binary(sop, ah, sh, 64));
                write_v(inst.defs[0], lo, hi);
            }
            return true;
        }

        // Roundings & Conversions (128-bit slice)
        case LirOpcode::Vroundps: {
            if (inst.defs.empty() || inst.uses.size() < 2) return true;
            auto [lo, hi] = read_v(inst.uses[0]);
            int mode = static_cast<int>(inst.uses[1].imm_int);
            auto a = extract_4x32(ctx, lo, hi);
            std::array<BVNodeId, 4> r{};
            if (env.shared_mem != nullptr && (mode & 3) == 3) {
                for (size_t i = 0; i < 4; ++i) {
                    r[i] = env.shared_mem->eval_uninterpreted_call(ctx, "truncf", {a[i]}, Type::f32());
                }
            } else {
                BVOp rop = (env.shared_mem != nullptr && (mode & 3) == 0)
                    ? BVOp::FRound
                    : static_cast<BVOp>(round_bv_op_for_mode(mode));
                for (size_t i = 0; i < 4; ++i) r[i] = ctx.make_unary(rop, a[i], 32);
            }
            auto [r_lo, r_hi] = pack_4x32(ctx, r);
            write_v(inst.defs[0], r_lo, r_hi);
            return true;
        }

        case LirOpcode::Vroundpd: {
            if (inst.defs.empty() || inst.uses.size() < 2) return true;
            auto [lo, hi] = read_v(inst.uses[0]);
            int mode = static_cast<int>(inst.uses[1].imm_int);
            if (env.shared_mem != nullptr && (mode & 3) == 3) {
                write_v(inst.defs[0],
                        env.shared_mem->eval_uninterpreted_call(ctx, "trunc", {lo}, Type::f64()),
                        env.shared_mem->eval_uninterpreted_call(ctx, "trunc", {hi}, Type::f64()));
            } else {
                BVOp rop = (env.shared_mem != nullptr && (mode & 3) == 0)
                    ? BVOp::FRound
                    : static_cast<BVOp>(round_bv_op_for_mode(mode));
                write_v(inst.defs[0], ctx.make_unary(rop, lo, 64), ctx.make_unary(rop, hi, 64));
            }
            return true;
        }

        case LirOpcode::Vcvtdq2ps: {
            if (inst.defs.empty() || inst.uses.empty()) return true;
            auto [lo, hi] = read_v(inst.uses[0]);
            auto a = extract_4x32(ctx, lo, hi);
            std::array<BVNodeId, 4> r{};
            for (size_t i = 0; i < 4; ++i) r[i] = ctx.make_unary(BVOp::SiToFp, a[i], 32);
            auto [r_lo, r_hi] = pack_4x32(ctx, r);
            write_v(inst.defs[0], r_lo, r_hi);
            return true;
        }

        case LirOpcode::Vcvttps2dq: {
            if (inst.defs.empty() || inst.uses.empty()) return true;
            auto [lo, hi] = read_v(inst.uses[0]);
            auto a = extract_4x32(ctx, lo, hi);
            std::array<BVNodeId, 4> r{};
            for (size_t i = 0; i < 4; ++i) {
                r[i] = (env.shared_mem != nullptr) ? ctx.make_unary(BVOp::FpToSi, a[i], 32)
                                                   : fp_to_si_clamped32(ctx, a[i], 32);
            }
            auto [r_lo, r_hi] = pack_4x32(ctx, r);
            write_v(inst.defs[0], r_lo, r_hi);
            return true;
        }

        case LirOpcode::Vcvtdq2pd: {
            if (inst.defs.empty() || inst.uses.empty()) return true;
            auto [lo, hi] = read_v(inst.uses[0]);
            size_t base = (inst.uses.size() >= 2 && inst.uses[1].is_imm_int() && (inst.uses[1].imm_int & 1)) ? 2u : 0u;
            auto a = extract_4x32(ctx, lo, hi);
            write_v(inst.defs[0], ctx.make_unary(BVOp::SiToFp, a[base], 64), ctx.make_unary(BVOp::SiToFp, a[base + 1], 64));
            return true;
        }

        case LirOpcode::Vcvtps2pd: {
            if (inst.defs.empty() || inst.uses.empty()) return true;
            auto [lo, hi] = read_v(inst.uses[0]);
            size_t base = (inst.uses.size() >= 2 && inst.uses[1].is_imm_int() && (inst.uses[1].imm_int & 1)) ? 2u : 0u;
            auto a = extract_4x32(ctx, lo, hi);
            write_v(inst.defs[0], ctx.make_unary(BVOp::FpExt, a[base], 64), ctx.make_unary(BVOp::FpExt, a[base + 1], 64));
            return true;
        }

        case LirOpcode::Vpmovsxdq: {
            if (inst.defs.empty() || inst.uses.empty()) return true;
            auto [lo, hi] = read_v(inst.uses[0]);
            size_t base = (inst.uses.size() >= 2 && inst.uses[1].is_imm_int() && (inst.uses[1].imm_int & 1)) ? 2u : 0u;
            auto a = extract_4x32(ctx, lo, hi);
            write_v(inst.defs[0], ctx.make_sext(a[base], 32), ctx.make_sext(a[base + 1], 32));
            return true;
        }

        case LirOpcode::Vcvttpd2dq: {
            if (inst.defs.empty() || inst.uses.size() < 2) return true;
            auto [l0, h0] = read_v(inst.uses[0]);
            auto [l1, h1] = read_v(inst.uses[1]);
            auto cvt = [&](BVNodeId f) {
                return (env.shared_mem != nullptr) ? ctx.make_unary(BVOp::FpToSi, f, 32)
                                                   : fp_to_si_clamped32(ctx, f, 64);
            };
            std::array<BVNodeId, 4> r{cvt(l0), cvt(h0), cvt(l1), cvt(h1)};
            auto [r_lo, r_hi] = pack_4x32(ctx, r);
            write_v(inst.defs[0], r_lo, r_hi);
            return true;
        }

        case LirOpcode::Vcvtpd2ps: {
            if (inst.defs.empty() || inst.uses.size() < 2) return true;
            auto [l0, h0] = read_v(inst.uses[0]);
            auto [l1, h1] = read_v(inst.uses[1]);
            std::array<BVNodeId, 4> r{
                ctx.make_unary(BVOp::FpTrunc, l0, 32), ctx.make_unary(BVOp::FpTrunc, h0, 32),
                ctx.make_unary(BVOp::FpTrunc, l1, 32), ctx.make_unary(BVOp::FpTrunc, h1, 32)
            };
            auto [r_lo, r_hi] = pack_4x32(ctx, r);
            write_v(inst.defs[0], r_lo, r_hi);
            return true;
        }

        case LirOpcode::Vmovmskps: {
            if (inst.defs.empty() || inst.uses.empty()) return true;
            auto [lo, hi] = read_v(inst.uses[0]);
            auto a = extract_4x32(ctx, lo, hi);
            BVNodeId b0 = ctx.make_zext(ctx.make_extract(a[0], 31, 31), 31);
            BVNodeId b1 = ctx.make_binary(BVOp::Shl, ctx.make_zext(ctx.make_extract(a[1], 31, 31), 31), ctx.make_const(1, 32), 32);
            BVNodeId b2 = ctx.make_binary(BVOp::Shl, ctx.make_zext(ctx.make_extract(a[2], 31, 31), 31), ctx.make_const(2, 32), 32);
            BVNodeId b3 = ctx.make_binary(BVOp::Shl, ctx.make_zext(ctx.make_extract(a[3], 31, 31), 31), ctx.make_const(3, 32), 32);
            write_op(inst.defs[0], ctx.make_binary(BVOp::Or, ctx.make_binary(BVOp::Or, b0, b1, 32),
                                                    ctx.make_binary(BVOp::Or, b2, b3, 32), 32));
            return true;
        }

        case LirOpcode::Vmovmskpd: {
            if (inst.defs.empty() || inst.uses.empty()) return true;
            auto [lo, hi] = read_v(inst.uses[0]);
            BVNodeId b0 = ctx.make_zext(ctx.make_extract(lo, 63, 63), 31);
            BVNodeId b1 = ctx.make_binary(BVOp::Shl, ctx.make_zext(ctx.make_extract(hi, 63, 63), 31), ctx.make_const(1, 32), 32);
            write_op(inst.defs[0], ctx.make_binary(BVOp::Or, b0, b1, 32));
            return true;
        }

        // Broadcasts, extracts, inserts, shuffles
        case LirOpcode::Vbroadcastss: {
            if (inst.defs.empty() || inst.uses.empty()) return true;
            BVNodeId s = read_op(inst.uses[0], 32);
            BVNodeId half = ctx.make_concat(s, s);
            write_v(inst.defs[0], half, half);
            return true;
        }

        case LirOpcode::Vbroadcastsd:
        case LirOpcode::Movddup: {
            if (inst.defs.empty() || inst.uses.empty()) return true;
            BVNodeId s = read_op(inst.uses[0], 64);
            write_v(inst.defs[0], s, s);
            return true;
        }

        case LirOpcode::Vpbroadcastd: {
            if (inst.defs.empty() || inst.uses.empty()) return true;
            BVNodeId s = read_op(inst.uses[0], 32);
            BVNodeId half = ctx.make_concat(s, s);
            write_v(inst.defs[0], half, half);
            return true;
        }

        case LirOpcode::Vpbroadcastq: {
            if (inst.defs.empty() || inst.uses.empty()) return true;
            BVNodeId s = read_op(inst.uses[0], 64);
            write_v(inst.defs[0], s, s);
            return true;
        }

        case LirOpcode::Pinsrd: {
            if (inst.defs.empty() || inst.uses.size() < 2) return true;
            const auto& base_op = (inst.uses.size() >= 3) ? inst.uses[0] : inst.defs[0];
            const auto& scalar_op = (inst.uses.size() >= 3) ? inst.uses[1] : inst.uses[0];
            auto [lo, hi] = read_v(base_op);
            auto lanes = extract_4x32(ctx, lo, hi);
            uint32_t lane = static_cast<uint32_t>(inst.uses.back().imm_int) & 3u;
            lanes[lane] = read_op(scalar_op, 32);
            auto [r_lo, r_hi] = pack_4x32(ctx, lanes);
            write_v(inst.defs[0], r_lo, r_hi);
            return true;
        }

        case LirOpcode::Pinsrq: {
            if (inst.defs.empty() || inst.uses.size() < 2) return true;
            const auto& base_op = (inst.uses.size() >= 3) ? inst.uses[0] : inst.defs[0];
            const auto& scalar_op = (inst.uses.size() >= 3) ? inst.uses[1] : inst.uses[0];
            auto [lo, hi] = read_v(base_op);
            BVNodeId sv = read_op(scalar_op, 64);
            uint32_t lane = static_cast<uint32_t>(inst.uses.back().imm_int) & 1u;
            write_v(inst.defs[0], lane == 0 ? sv : lo, lane == 1 ? sv : hi);
            return true;
        }

        case LirOpcode::Pextrd:
        case LirOpcode::Extractps: {
            if (inst.defs.empty() || inst.uses.size() < 2) return true;
            auto [lo, hi] = read_v(inst.uses[0]);
            auto lanes = extract_4x32(ctx, lo, hi);
            uint32_t lane = static_cast<uint32_t>(inst.uses.back().imm_int) & 3u;
            write_op(inst.defs[0], lanes[lane]);
            return true;
        }

        case LirOpcode::Pextrq: {
            if (inst.defs.empty() || inst.uses.size() < 2) return true;
            auto [lo, hi] = read_v(inst.uses[0]);
            uint32_t lane = static_cast<uint32_t>(inst.uses.back().imm_int) & 1u;
            write_op(inst.defs[0], lane == 0 ? lo : hi);
            return true;
        }

        case LirOpcode::Insertps: {
            if (inst.defs.empty() || inst.uses.size() < 2) return true;
            const auto& base_op = (inst.uses.size() >= 3) ? inst.uses[0] : inst.defs[0];
            const auto& src_op = (inst.uses.size() >= 3) ? inst.uses[1] : inst.uses[0];
            auto [bl, bh] = read_v(base_op);
            auto [sl, sh] = read_v(src_op);
            auto dst_l = extract_4x32(ctx, bl, bh);
            auto src_l = extract_4x32(ctx, sl, sh);
            uint32_t imm = static_cast<uint32_t>(inst.uses.back().imm_int);
            dst_l[(imm >> 4) & 3u] = src_l[(imm >> 6) & 3u];
            for (uint32_t i = 0; i < 4; ++i) {
                if (imm & (1u << i)) dst_l[i] = ctx.make_const(0, 32);
            }
            auto [r_lo, r_hi] = pack_4x32(ctx, dst_l);
            write_v(inst.defs[0], r_lo, r_hi);
            return true;
        }

        case LirOpcode::Pshufd: {
            if (inst.defs.empty() || inst.uses.size() < 2) return true;
            auto [lo, hi] = read_v(inst.uses[0]);
            auto s = extract_4x32(ctx, lo, hi);
            uint32_t mask = static_cast<uint32_t>(inst.uses.back().imm_int);
            std::array<BVNodeId, 4> r{s[mask & 3u], s[(mask >> 2) & 3u], s[(mask >> 4) & 3u], s[(mask >> 6) & 3u]};
            auto [r_lo, r_hi] = pack_4x32(ctx, r);
            write_v(inst.defs[0], r_lo, r_hi);
            return true;
        }

        case LirOpcode::Shufps:
        case LirOpcode::Vshufps: {
            if (inst.defs.empty() || inst.uses.size() < 2) return true;
            const auto& op0 = (inst.uses.size() >= 3) ? inst.uses[0] : inst.defs[0];
            const auto& op1 = inst.uses[inst.uses.size() - 2];
            auto [al, ah] = read_v(op0);
            auto [bl, bh] = read_v(op1);
            auto a = extract_4x32(ctx, al, ah);
            auto b = extract_4x32(ctx, bl, bh);
            uint32_t mask = static_cast<uint32_t>(inst.uses.back().imm_int);
            std::array<BVNodeId, 4> r{a[mask & 3u], a[(mask >> 2) & 3u], b[(mask >> 4) & 3u], b[(mask >> 6) & 3u]};
            auto [r_lo, r_hi] = pack_4x32(ctx, r);
            write_v(inst.defs[0], r_lo, r_hi);
            return true;
        }

        case LirOpcode::Shufpd: {
            if (inst.defs.empty() || inst.uses.size() < 2) return true;
            const auto& op0 = (inst.uses.size() >= 3) ? inst.uses[0] : inst.defs[0];
            const auto& op1 = inst.uses[inst.uses.size() - 2];
            auto [al, ah] = read_v(op0);
            auto [bl, bh] = read_v(op1);
            uint32_t mask = static_cast<uint32_t>(inst.uses.back().imm_int);
            write_v(inst.defs[0], (mask & 1u) ? ah : al, (mask & 2u) ? bh : bl);
            return true;
        }

        default:
            return false;
    }
}

} // namespace brass::verify
