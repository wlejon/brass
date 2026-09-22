#include <brass/codegen/emit_context.hpp>
#include <brass/target/x64/x64_frame.hpp>
#include <algorithm>
#include <vector>
#include <cstdint>

namespace brass::codegen {

using namespace brass::x64;

void EmitContext::emit_parallel_copy(const LirInst& inst) {
    size_t n = inst.defs.size();
    if (n == 0 || n != inst.uses.size()) return;

    struct Move {
        LirOperand dst;
        LirOperand src;
        bool done = false;
    };

    std::vector<Move> moves;
    for (size_t i = 0; i < n; ++i) {
        if (inst.defs[i].is_preg() && inst.uses[i].is_preg() &&
            inst.defs[i].preg_val == inst.uses[i].preg_val) {
            continue; // Skip self moves
        }
        if (inst.defs[i].is_spill_slot() && inst.uses[i].is_spill_slot() &&
            inst.defs[i].spill_slot == inst.uses[i].spill_slot) {
            continue; // Skip self moves
        }
        moves.push_back({inst.defs[i], inst.uses[i], false});
    }

    auto emit_move = [&](const LirOperand& dst, const LirOperand& src) {
        if (dst.is_preg() && src.is_preg() && dst.preg_val == src.preg_val) return;
        if (dst.is_spill_slot() && src.is_spill_slot() && dst.spill_slot == src.spill_slot) return;
        if (dst.is_preg()) {
            if (dst.preg_val.is_gpr()) {
                GPR dst_gpr = dst.preg_val.as_gpr();
                if (src.is_preg()) {
                    GPR src_gpr = src.preg_val.as_gpr();
                    if (dst.size == 4 && src.size == 4) enc_.mov32(dst_gpr, src_gpr);
                    else enc_.mov(dst_gpr, src_gpr);
                } else if (src.is_imm_int()) {
                    if (dst.size == 4) enc_.mov32(dst_gpr, static_cast<uint32_t>(src.imm_int));
                    else enc_.mov(dst_gpr, src.imm_int);
                } else {
                    if (dst.size == 4) enc_.mov32(dst_gpr, to_mem_address(src));
                    else enc_.mov(dst_gpr, to_mem_address(src));
                }
            } else {
                XMM dst_xmm = dst.preg_val.as_xmm();
                if (dst.size == 32 || src.size == 32) {
                    if (src.is_preg()) enc_.vmovaps(dst_xmm, src.preg_val.as_xmm());
                    else enc_.vmovups(dst_xmm, to_mem_address(src));
                } else if (dst.size == 16 || src.size == 16) {
                    if (src.is_preg()) enc_.movaps(dst_xmm, src.preg_val.as_xmm());
                    else enc_.movups(dst_xmm, to_mem_address(src));
                } else if (dst.size == 4 && src.size == 4) {
                    if (src.is_preg()) enc_.movss(dst_xmm, src.preg_val.as_xmm());
                    else enc_.movss(dst_xmm, to_mem_address(src));
                } else {
                    if (src.is_preg()) enc_.movsd(dst_xmm, src.preg_val.as_xmm());
                    else enc_.movsd(dst_xmm, to_mem_address(src));
                }
            }
        } else {
            MemAddress dst_mem = to_mem_address(dst);
            if (src.is_preg()) {
                if (src.preg_val.is_gpr()) {
                    if (src.size == 4) enc_.mov32(dst_mem, src.preg_val.as_gpr());
                    else enc_.mov(dst_mem, src.preg_val.as_gpr());
                } else {
                    if (dst.size == 32 || src.size == 32) {
                        enc_.vmovups(dst_mem, src.preg_val.as_xmm());
                    } else if (dst.size == 16 || src.size == 16) {
                        enc_.movups(dst_mem, src.preg_val.as_xmm());
                    } else if (dst.size == 4 && src.size == 4) {
                        enc_.movss(dst_mem, src.preg_val.as_xmm());
                    } else {
                        enc_.movsd(dst_mem, src.preg_val.as_xmm());
                    }
                }
            } else if (src.is_imm_int()) {
                if (dst.size == 4) {
                    enc_.mov32(dst_mem, static_cast<int32_t>(src.imm_int));
                } else if (src.imm_int >= INT32_MIN && src.imm_int <= INT32_MAX) {
                    enc_.mov(dst_mem, static_cast<int32_t>(src.imm_int));
                } else {
                    // A store immediate is sign-extended from 32 bits, and
                    // R11 may hold a cycle's saved value: store two halves.
                    const uint64_t bits = static_cast<uint64_t>(src.imm_int);
                    MemAddress hi = dst_mem;
                    hi.disp += 4;
                    enc_.mov32(dst_mem, static_cast<int32_t>(static_cast<uint32_t>(bits)));
                    enc_.mov32(hi, static_cast<int32_t>(static_cast<uint32_t>(bits >> 32)));
                }
            } else {
                // Memory-to-memory move using scratch register
                MemAddress src_mem = to_mem_address(src);
                if (dst.size == 32 || src.size == 32) {
                    enc_.vmovups(XMM::XMM15, src_mem);
                    enc_.vmovups(dst_mem, XMM::XMM15);
                } else if (dst.size == 16 || src.size == 16) {
                    enc_.movups(XMM::XMM15, src_mem);
                    enc_.movups(dst_mem, XMM::XMM15);
                } else if (dst.size == 4 && src.size == 4) {
                    enc_.mov32(GPR::R11, src_mem);
                    enc_.mov32(dst_mem, GPR::R11);
                } else {
                    enc_.mov(GPR::R11, src_mem);
                    enc_.mov(dst_mem, GPR::R11);
                }
            }
        }
    };

    // A cycle's saved value must survive the moves that follow it, and a
    // memory-to-memory move goes through R11 / XMM15; so the cycle keeps
    // its value in the other reserved scratch register of the class.
    auto cycle_scratch = [](bool is_xmm) {
        return is_xmm ? PReg::xmm(XMM::XMM14) : PReg::gpr(GPR::R10);
    };

    auto peel_acyclic = [&]() -> bool {
        bool progress = false;
        for (auto& m : moves) {
            if (m.done) continue;
            bool dst_used = false;
            for (const auto& other : moves) {
                if (other.done) continue;
                if (m.dst.is_preg() && other.src.is_preg() &&
                    m.dst.preg_val == other.src.preg_val) {
                    dst_used = true;
                    break;
                }
                if (m.dst.is_spill_slot() && other.src.is_spill_slot() &&
                    m.dst.spill_slot == other.src.spill_slot) {
                    dst_used = true;
                    break;
                }
            }
            if (!dst_used) {
                emit_move(m.dst, m.src);
                m.done = true;
                progress = true;
            }
        }
        return progress;
    };

    while (true) {
        while (peel_acyclic()) {}

        size_t start_idx = SIZE_MAX;
        for (size_t i = 0; i < moves.size(); ++i) {
            if (!moves[i].done) {
                start_idx = i;
                break;
            }
        }
        if (start_idx == SIZE_MAX) break;

        // Trace permutation cycle
        std::vector<size_t> cycle;
        size_t curr = start_idx;
        while (true) {
            cycle.push_back(curr);
            size_t next_idx = SIZE_MAX;
            for (size_t i = 0; i < moves.size(); ++i) {
                if (!moves[i].done) {
                    bool match = false;
                    if (moves[i].dst.is_preg() && moves[curr].src.is_preg() &&
                        moves[i].dst.preg_val == moves[curr].src.preg_val) {
                        match = true;
                    } else if (moves[i].dst.is_spill_slot() && moves[curr].src.is_spill_slot() &&
                               moves[i].dst.spill_slot == moves[curr].src.spill_slot) {
                        match = true;
                    }
                    if (match) {
                        next_idx = i;
                        break;
                    }
                }
            }
            if (next_idx == SIZE_MAX || next_idx == start_idx) {
                break;
            }
            bool already_in_cycle = false;
            for (size_t c : cycle) {
                if (c == next_idx) {
                    already_in_cycle = true;
                    break;
                }
            }
            if (already_in_cycle) break;
            curr = next_idx;
        }

        if (cycle.size() == 2) {
            size_t idx0 = cycle[0];
            size_t idx1 = cycle[1];
            auto& m0 = moves[idx0];
            auto& m1 = moves[idx1];

            if (m0.dst.is_preg() && m0.dst.preg_val.is_gpr() &&
                m1.dst.is_preg() && m1.dst.preg_val.is_gpr() &&
                m0.src.is_preg() && m0.src.preg_val.is_gpr() &&
                m1.src.is_preg() && m1.src.preg_val.is_gpr()) {
                GPR r0 = m0.dst.preg_val.as_gpr();
                GPR r1 = m1.dst.preg_val.as_gpr();
                if (m0.dst.size == 4 && m1.dst.size == 4) {
                    enc_.xchg32(r0, r1);
                } else {
                    enc_.xchg(r0, r1);
                }
                m0.done = true;
                m1.done = true;
            } else {
                bool is_xmm = (m0.dst.is_preg() && m0.dst.preg_val.is_xmm()) ||
                              (m1.dst.is_preg() && m1.dst.preg_val.is_xmm()) ||
                              (m0.src.is_preg() && m0.src.preg_val.is_xmm()) ||
                              (m1.src.is_preg() && m1.src.preg_val.is_xmm()) ||
                              (m0.dst.size == 16 || m0.dst.size == 32);
                PReg scratch = cycle_scratch(is_xmm);
                uint8_t sz = m0.dst.size;
                emit_move(LirOperand::preg(scratch, sz), m0.src);
                emit_move(m1.dst, m0.dst);
                emit_move(m0.dst, LirOperand::preg(scratch, sz));
                m0.done = true;
                m1.done = true;
            }
        } else if (!cycle.empty()) {
            size_t idx0 = cycle[0];
            auto& m0 = moves[idx0];
            bool is_xmm = (m0.dst.is_preg() && m0.dst.preg_val.is_xmm()) ||
                          (m0.src.is_preg() && m0.src.preg_val.is_xmm()) ||
                          (m0.dst.size == 16 || m0.dst.size == 32);
            PReg scratch = cycle_scratch(is_xmm);
            uint8_t sz = m0.dst.size;

            emit_move(LirOperand::preg(scratch, sz), m0.dst);
            size_t last_idx = cycle.back();
            moves[last_idx].src = LirOperand::preg(scratch, sz);
        } else {
            emit_move(moves[start_idx].dst, moves[start_idx].src);
            moves[start_idx].done = true;
        }
    }
}

} // namespace brass::codegen
