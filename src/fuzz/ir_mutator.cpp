#include <brass/fuzz/ir_mutator.hpp>
#include <brass/mir/verifier.hpp>
#include <brass/mir/builder.hpp>
#include <limits>
#include <cmath>
#include <vector>
#include <algorithm>

namespace brass::fuzz {

// ============================================================================
// FuzzRng Implementation
// ============================================================================

FuzzRng::FuzzRng(uint64_t seed_val) noexcept
    : state_(seed_val ? seed_val : 0x853c49e6748fea9bULL) {}

void FuzzRng::seed(uint64_t s) noexcept {
    state_ = (s ? s : 0x853c49e6748fea9bULL);
}

uint64_t FuzzRng::next_u64() noexcept {
    // SplitMix64 generator
    uint64_t z = (state_ += 0x9e3779b97f4a7c15ULL);
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
    return z ^ (z >> 31);
}

uint32_t FuzzRng::next_u32() noexcept {
    return static_cast<uint32_t>(next_u64());
}

int64_t FuzzRng::next_i64() noexcept {
    return static_cast<int64_t>(next_u64());
}

int32_t FuzzRng::next_i32() noexcept {
    return static_cast<int32_t>(next_u32());
}

double FuzzRng::next_f64() noexcept {
    return static_cast<double>(next_u64() >> 11) * (1.0 / 9007199254740992.0);
}

float FuzzRng::next_f32() noexcept {
    return static_cast<float>(next_u32() >> 8) * (1.0f / 16777216.0f);
}

uint64_t FuzzRng::range_u64(uint64_t min_v, uint64_t max_v) noexcept {
    if (min_v >= max_v) return min_v;
    return min_v + (next_u64() % (max_v - min_v + 1));
}

int64_t FuzzRng::range_i64(int64_t min_v, int64_t max_v) noexcept {
    if (min_v >= max_v) return min_v;
    uint64_t span = static_cast<uint64_t>(max_v - min_v + 1);
    return min_v + static_cast<int64_t>(next_u64() % span);
}

size_t FuzzRng::pick_index(size_t size) noexcept {
    if (size == 0) return 0;
    return static_cast<size_t>(next_u64() % size);
}

bool FuzzRng::coin_flip(double p) noexcept {
    return next_f64() < p;
}

int32_t FuzzRng::boundary_i32() noexcept {
    static const int32_t boundaries[] = {
        0, 1, -1, 2, -2,
        std::numeric_limits<int32_t>::min(),
        std::numeric_limits<int32_t>::max(),
        std::numeric_limits<int32_t>::min() + 1,
        std::numeric_limits<int32_t>::max() - 1,
        0x7FFF, -0x8000, 256, -256, 42
    };
    constexpr size_t count = sizeof(boundaries) / sizeof(boundaries[0]);
    return boundaries[pick_index(count)];
}

int64_t FuzzRng::boundary_i64() noexcept {
    static const int64_t boundaries[] = {
        0, 1, -1, 2, -2,
        std::numeric_limits<int64_t>::min(),
        std::numeric_limits<int64_t>::max(),
        std::numeric_limits<int64_t>::min() + 1,
        std::numeric_limits<int64_t>::max() - 1,
        static_cast<int64_t>(std::numeric_limits<int32_t>::min()),
        static_cast<int64_t>(std::numeric_limits<int32_t>::max()),
        0x100000000LL, -0x100000000LL, 42
    };
    constexpr size_t count = sizeof(boundaries) / sizeof(boundaries[0]);
    return boundaries[pick_index(count)];
}

float FuzzRng::boundary_f32() noexcept {
    static const float boundaries[] = {
        0.0f, -0.0f, 1.0f, -1.0f, 0.5f, -0.5f,
        std::numeric_limits<float>::quiet_NaN(),
        std::numeric_limits<float>::infinity(),
        -std::numeric_limits<float>::infinity(),
        std::numeric_limits<float>::denorm_min(),
        std::numeric_limits<float>::min(),
        std::numeric_limits<float>::max()
    };
    constexpr size_t count = sizeof(boundaries) / sizeof(boundaries[0]);
    return boundaries[pick_index(count)];
}

double FuzzRng::boundary_f64() noexcept {
    static const double boundaries[] = {
        0.0, -0.0, 1.0, -1.0, 0.5, -0.5,
        std::numeric_limits<double>::quiet_NaN(),
        std::numeric_limits<double>::infinity(),
        -std::numeric_limits<double>::infinity(),
        std::numeric_limits<double>::denorm_min(),
        std::numeric_limits<double>::min(),
        std::numeric_limits<double>::max()
    };
    constexpr size_t count = sizeof(boundaries) / sizeof(boundaries[0]);
    return boundaries[pick_index(count)];
}

// ============================================================================
// IrGenerator Implementation
// ============================================================================

Function* IrGenerator::generate(Module& mod, std::string_view name, uint64_t seed) {
    FuzzRng rng(seed);

    // If exception handling enabled, ensure helper callee exists
    if (options_.enable_exceptions && !mod.get_function("fuzz_eh_callee")) {
        Function* eh_fn = mod.create_function("fuzz_eh_callee", Type::i64(), {Type::i64()});
        Builder eb(mod);
        eb.set_function(eh_fn);

        BasicBlock* eh_entry = eb.append_block("entry");
        BasicBlock* eh_throw = eb.create_block("throw_bb");
        BasicBlock* eh_norm = eb.create_block("norm_bb");
        eh_fn->append_block(eh_throw);
        eh_fn->append_block(eh_norm);

        eb.position_at_end(eh_entry);
        Value* arg = eb.add_block_param(eh_entry, Type::i64());
        Value* zero = eb.build_iconst_i64(0);
        Value* is_zero = eb.build_eq(arg, zero);
        eb.build_br_if(is_zero, eh_throw, {}, eh_norm, {});

        eb.position_at_end(eh_throw);
        eb.build_throw(arg);

        eb.position_at_end(eh_norm);
        Value* two = eb.build_iconst_i64(2);
        Value* doubled = eb.build_mul(arg, two);
        eb.build_ret(doubled);

        eh_fn->rebuild_cfg_predecessors();
    }

    Function* fn = mod.create_function(name, options_.return_type, options_.param_types);
    Builder b(mod);
    b.set_function(fn);

    BasicBlock* entry = b.append_block("entry");
    b.position_at_end(entry);

    std::vector<Value*> pool_i64;
    std::vector<Value*> pool_i32;
    std::vector<Value*> pool_f64;

    for (size_t i = 0; i < options_.param_types.size(); ++i) {
        Type pty = options_.param_types[i];
        Value* param = b.add_block_param(entry, pty);
        if (pty == Type::i64()) pool_i64.push_back(param);
        else if (pty == Type::i32()) pool_i32.push_back(param);
        else if (pty == Type::f64()) pool_f64.push_back(param);
    }

    // Seed boundary values into the pools
    pool_i64.push_back(b.build_iconst_i64(rng.boundary_i64()));
    pool_i64.push_back(b.build_iconst_i64(rng.boundary_i64()));
    pool_i32.push_back(b.build_iconst_i32(rng.boundary_i32()));
    pool_i32.push_back(b.build_iconst_i32(rng.boundary_i32()));
    pool_f64.push_back(b.build_fconst_f64(rng.boundary_f64()));

    // Generate basic arithmetic and bitwise expressions
    size_t num_exprs = rng.range_u64(options_.min_instructions, options_.max_instructions);
    for (size_t i = 0; i < num_exprs; ++i) {
        uint32_t op_choice = rng.next_u32() % 14;
        Value* a = pool_i64[rng.pick_index(pool_i64.size())];
        Value* c_val = pool_i64[rng.pick_index(pool_i64.size())];

        Value* res = nullptr;
        switch (op_choice) {
            case 0: res = b.build_add(a, c_val); break;
            case 1: res = b.build_sub(a, c_val); break;
            case 2: res = b.build_mul(a, c_val); break;
            case 3: res = b.build_and(a, c_val); break;
            case 4: res = b.build_or(a, c_val); break;
            case 5: res = b.build_xor(a, c_val); break;
            case 6: res = b.build_not(a); break;
            case 7: {
                Value* sh = b.build_iconst_i64(static_cast<int64_t>(rng.next_u32() % 63));
                res = b.build_shl(a, sh);
                break;
            }
            case 8: {
                Value* sh = b.build_iconst_i64(static_cast<int64_t>(rng.next_u32() % 63));
                res = b.build_lshr(a, sh);
                break;
            }
            case 9: {
                Value* sh = b.build_iconst_i64(static_cast<int64_t>(rng.next_u32() % 63));
                res = b.build_ashr(a, sh);
                break;
            }
            case 10: res = b.build_popcnt(a); break;
            case 11: res = b.build_clz(a); break;
            case 12: res = b.build_ctz(a); break;
            case 13: {
                // Safe division by masking denominator with 1
                Value* one = b.build_iconst_i64(1);
                Value* safe_denom = b.build_or(c_val, one);
                res = b.build_sdiv(a, safe_denom);
                break;
            }
        }
        if (res) pool_i64.push_back(res);
    }

    // Vector operations
    if (options_.enable_vectors) {
        Value* f_scalar = pool_f64.back();
        Value* vzero = b.build_vzero(Type::f64x4());
        Value* vbroadcast = b.build_vbroadcast(Type::f64x4(), f_scalar);
        Value* vadd = b.build_vadd(vzero, vbroadcast);
        Value* vmul = b.build_vmul(vadd, vbroadcast);
        Value* vfma = b.build_vfma(vadd, vmul, vbroadcast);
        Value* lane0 = b.build_vextract_lane(vfma, 0);
        Value* lane_int = b.build_bitcast_i64_f64(lane0);
        pool_i64.push_back(lane_int);
    }

    // Memory operations (alloc, store, load)
    if (options_.enable_memory) {
        Value* sz = b.build_iconst_i64(32);
        Value* mask = b.build_iconst_i64(0);
        Value* tag = b.build_iconst_i32(1);
        Value* obj = b.build_call("brass_gc_alloc", Type::gcref(), {sz, mask, tag});
        Value* to_store = pool_i64.back();
        b.build_store(Type::i64(), obj, 0, to_store);
        Value* loaded = b.build_load(Type::i64(), obj, 0);
        pool_i64.push_back(loaded);
    }

    BasicBlock* cur_bb = entry;

    // Diamond branch
    if (options_.enable_diamonds) {
        BasicBlock* then_bb = b.create_block("then_bb");
        BasicBlock* else_bb = b.create_block("else_bb");
        BasicBlock* merge_bb = b.create_block("merge_bb");

        fn->append_block(then_bb);
        fn->append_block(else_bb);
        fn->append_block(merge_bb);

        b.position_at_end(cur_bb);
        Value* cond_val = b.build_slt(pool_i64[rng.pick_index(pool_i64.size())],
                                      pool_i64[rng.pick_index(pool_i64.size())]);
        b.build_br_if(cond_val, then_bb, {}, else_bb, {});

        b.position_at_end(then_bb);
        Value* then_res = b.build_add(pool_i64.back(), b.build_iconst_i64(7));
        b.build_br(merge_bb, {then_res});

        b.position_at_end(else_bb);
        Value* else_res = b.build_xor(pool_i64.back(), b.build_iconst_i64(0xFF));
        b.build_br(merge_bb, {else_res});

        b.position_at_end(merge_bb);
        Value* merge_phi = b.add_block_param(merge_bb, Type::i64());
        pool_i64.push_back(merge_phi);
        cur_bb = merge_bb;
    }

    // Nested loops
    if (options_.enable_loops) {
        BasicBlock* loop_hdr = b.create_block("loop_hdr");
        BasicBlock* loop_body = b.create_block("loop_body");
        BasicBlock* loop_exit = b.create_block("loop_exit");

        fn->append_block(loop_hdr);
        fn->append_block(loop_body);
        fn->append_block(loop_exit);

        b.position_at_end(cur_bb);
        Value* zero_i64 = b.build_iconst_i64(0);
        Value* init_acc = pool_i64.back();
        b.build_br(loop_hdr, {zero_i64, init_acc});

        b.position_at_end(loop_hdr);
        Value* iv = b.add_block_param(loop_hdr, Type::i64());
        Value* acc = b.add_block_param(loop_hdr, Type::i64());
        Value* limit = b.build_iconst_i64(static_cast<int64_t>(4 + (rng.next_u32() % 8)));
        Value* loop_cond = b.build_slt(iv, limit);
        b.build_br_if(loop_cond, loop_body, {}, loop_exit, {acc});

        b.position_at_end(loop_body);
        Value* one_i64 = b.build_iconst_i64(1);
        Value* next_iv = b.build_add(iv, one_i64);
        Value* step_val = b.build_mul(iv, b.build_iconst_i64(3));
        Value* next_acc = b.build_add(acc, step_val);
        b.build_br(loop_hdr, {next_iv, next_acc});

        b.position_at_end(loop_exit);
        Value* exit_acc = b.add_block_param(loop_exit, Type::i64());
        pool_i64.push_back(exit_acc);
        cur_bb = loop_exit;
    }

    // Switch statement
    if (options_.enable_switches) {
        BasicBlock* sw_c0 = b.create_block("sw_c0");
        BasicBlock* sw_c1 = b.create_block("sw_c1");
        BasicBlock* sw_def = b.create_block("sw_def");
        BasicBlock* sw_join = b.create_block("sw_join");

        fn->append_block(sw_c0);
        fn->append_block(sw_c1);
        fn->append_block(sw_def);
        fn->append_block(sw_join);

        b.position_at_end(cur_bb);
        Value* sw_sel = b.build_and(pool_i64.back(), b.build_iconst_i64(3));
        b.build_switch(sw_sel, sw_def, {}, {SwitchCase(0, sw_c0), SwitchCase(1, sw_c1)});

        b.position_at_end(sw_c0);
        b.build_br(sw_join, {b.build_iconst_i64(100)});

        b.position_at_end(sw_c1);
        b.build_br(sw_join, {b.build_iconst_i64(200)});

        b.position_at_end(sw_def);
        b.build_br(sw_join, {b.build_iconst_i64(300)});

        b.position_at_end(sw_join);
        Value* sw_res = b.add_block_param(sw_join, Type::i64());
        Value* combined = b.build_add(pool_i64.back(), sw_res);
        pool_i64.push_back(combined);
        cur_bb = sw_join;
    }

    // Exception handling (invoke and landing_pad)
    if (options_.enable_exceptions) {
        BasicBlock* norm_bb = b.create_block("norm_bb");
        BasicBlock* unw_bb = b.create_block("unw_bb");
        BasicBlock* eh_join = b.create_block("eh_join");

        fn->append_block(norm_bb);
        fn->append_block(unw_bb);
        fn->append_block(eh_join);

        b.position_at_end(cur_bb);
        // Pass value > 0 so normal path is taken deterministically
        Value* eh_arg = b.build_or(pool_i64.back(), b.build_iconst_i64(1));
        b.build_invoke("fuzz_eh_callee", Type::i64(), {eh_arg}, norm_bb, {}, unw_bb, {});

        b.position_at_end(norm_bb);
        b.build_br(eh_join, {eh_arg});

        b.position_at_end(unw_bb);
        Value* exc = b.build_landing_pad(Type::i64());
        b.build_br(eh_join, {exc});

        b.position_at_end(eh_join);
        Value* eh_res = b.add_block_param(eh_join, Type::i64());
        pool_i64.push_back(eh_res);
        cur_bb = eh_join;
    }

    b.position_at_end(cur_bb);
    b.build_ret(pool_i64.back());

    fn->rebuild_cfg_predecessors();
    return fn;
}

// ============================================================================
// IrMutator Implementation
// ============================================================================

bool IrMutator::mutate_constant_immediates(Function& fn, FuzzRng& rng) {
    std::vector<Instruction*> const_insts;
    for (BasicBlock* bb : fn.blocks()) {
        if (!bb) continue;
        for (Instruction* inst : *bb) {
            if (inst && is_constant(inst->opcode())) {
                const_insts.push_back(inst);
            }
        }
    }

    if (const_insts.empty()) return false;
    Instruction* target = const_insts[rng.pick_index(const_insts.size())];

    if (target->opcode() == Opcode::iconst_i64) {
        uint32_t choice = rng.next_u32() % 3;
        if (choice == 0) {
            target->set_imm_i64(rng.boundary_i64());
        } else if (choice == 1) {
            uint64_t bit = 1ULL << (rng.next_u32() % 64);
            target->set_imm_i64(static_cast<int64_t>(static_cast<uint64_t>(target->imm_i64()) ^ bit));
        } else {
            target->set_imm_i64(target->imm_i64() + static_cast<int64_t>(rng.coin_flip() ? 1 : -1));
        }
        return true;
    }

    if (target->opcode() == Opcode::iconst_i32) {
        uint32_t choice = rng.next_u32() % 3;
        if (choice == 0) {
            target->set_imm_i32(rng.boundary_i32());
        } else if (choice == 1) {
            uint32_t bit = 1U << (rng.next_u32() % 32);
            target->set_imm_i32(static_cast<int32_t>(static_cast<uint32_t>(target->imm_i32()) ^ bit));
        } else {
            target->set_imm_i32(target->imm_i32() + static_cast<int32_t>(rng.coin_flip() ? 1 : -1));
        }
        return true;
    }

    if (target->opcode() == Opcode::fconst_f64) {
        target->set_imm_f64(rng.boundary_f64());
        return true;
    }

    return false;
}

bool IrMutator::swap_commutative_operands(Function& fn, FuzzRng& rng) {
    std::vector<Instruction*> comm_insts;
    for (BasicBlock* bb : fn.blocks()) {
        if (!bb) continue;
        for (Instruction* inst : *bb) {
            if (!inst || inst->operand_count() < 2) continue;
            Opcode op = inst->opcode();
            if (op == Opcode::add || op == Opcode::mul || op == Opcode::and_ ||
                op == Opcode::or_  || op == Opcode::xor_ || op == Opcode::eq ||
                op == Opcode::ne   || op == Opcode::vadd || op == Opcode::vmul ||
                op == Opcode::vand || op == Opcode::vor  || op == Opcode::vxor) {
                comm_insts.push_back(inst);
            }
        }
    }

    if (comm_insts.empty()) return false;
    Instruction* inst = comm_insts[rng.pick_index(comm_insts.size())];
    Value* op0 = inst->operand(0);
    Value* op1 = inst->operand(1);
    inst->set_operand(0, op1);
    inst->set_operand(1, op0);
    return true;
}

bool IrMutator::replace_opcodes(Function& fn, FuzzRng& rng) {
    std::vector<Instruction*> repl_insts;
    for (BasicBlock* bb : fn.blocks()) {
        if (!bb) continue;
        for (Instruction* inst : *bb) {
            if (!inst) continue;
            Opcode op = inst->opcode();
            if (op == Opcode::add || op == Opcode::sub ||
                op == Opcode::and_ || op == Opcode::or_ || op == Opcode::xor_ ||
                op == Opcode::slt || op == Opcode::sgt || op == Opcode::sle || op == Opcode::sge ||
                op == Opcode::vadd || op == Opcode::vsub) {
                repl_insts.push_back(inst);
            }
        }
    }

    if (repl_insts.empty()) return false;
    Instruction* inst = repl_insts[rng.pick_index(repl_insts.size())];
    Opcode op = inst->opcode();

    if (op == Opcode::add) inst->set_opcode(Opcode::sub);
    else if (op == Opcode::sub) inst->set_opcode(Opcode::add);
    else if (op == Opcode::and_) inst->set_opcode(rng.coin_flip() ? Opcode::or_ : Opcode::xor_);
    else if (op == Opcode::or_)  inst->set_opcode(rng.coin_flip() ? Opcode::and_ : Opcode::xor_);
    else if (op == Opcode::xor_) inst->set_opcode(rng.coin_flip() ? Opcode::and_ : Opcode::or_);
    else if (op == Opcode::slt) inst->set_opcode(Opcode::sgt);
    else if (op == Opcode::sgt) inst->set_opcode(Opcode::slt);
    else if (op == Opcode::sle) inst->set_opcode(Opcode::sge);
    else if (op == Opcode::sge) inst->set_opcode(Opcode::sle);
    else if (op == Opcode::vadd) inst->set_opcode(Opcode::vsub);
    else if (op == Opcode::vsub) inst->set_opcode(Opcode::vadd);

    return true;
}

bool IrMutator::split_basic_blocks(Function& fn, FuzzRng& rng) {
    std::vector<BasicBlock*> candidates;
    for (BasicBlock* bb : fn.blocks()) {
        if (!bb || bb->instruction_count() < 3) continue;
        size_t non_term = 0;
        for (Instruction* inst : *bb) {
            if (inst && !inst->is_terminator() && inst->opcode() != Opcode::landing_pad) {
                non_term++;
            }
        }
        if (non_term >= 2) {
            candidates.push_back(bb);
        }
    }

    if (candidates.empty()) return false;
    BasicBlock* bb = candidates[rng.pick_index(candidates.size())];

    std::vector<Instruction*> non_term_list;
    for (Instruction* inst : *bb) {
        if (inst && !inst->is_terminator() && inst->opcode() != Opcode::landing_pad) {
            non_term_list.push_back(inst);
        }
    }
    if (non_term_list.size() < 2) return false;

    size_t split_idx = 1 + (rng.next_u32() % (non_term_list.size() - 1));
    Instruction* split_start = non_term_list[split_idx];

    // Collect all instructions from split_start to the end of bb (including terminator)
    std::vector<Instruction*> to_move;
    bool found = false;
    for (Instruction* inst : *bb) {
        if (inst == split_start) found = true;
        if (found) to_move.push_back(inst);
    }
    if (to_move.empty()) return false;

    Builder b(fn);
    BasicBlock* new_bb = b.create_block("split_bb");
    auto& blist = fn.blocks();
    auto it = std::find(blist.begin(), blist.end(), bb);
    if (it != blist.end()) {
        blist.insert(it + 1, new_bb);
    } else {
        fn.append_block(new_bb);
    }

    for (Instruction* inst : to_move) {
        bb->remove_instruction(inst);
        new_bb->append_instruction(inst);
    }

    // Insert unconditional jump from bb to new_bb
    b.position_at_end(bb);
    b.build_br(new_bb);

    fn.rebuild_cfg_predecessors();

    if (!verify_function(fn)) {
        // Rollback
        for (Instruction* inst : to_move) {
            new_bb->remove_instruction(inst);
            bb->append_instruction(inst);
        }
        Instruction* br_inst = bb->tail();
        if (br_inst && br_inst->opcode() == Opcode::br && br_inst->branch_target().block == new_bb) {
            bb->remove_instruction(br_inst);
        }
        fn.remove_block(new_bb);
        fn.rebuild_cfg_predecessors();
        return false;
    }

    return true;
}

bool IrMutator::inject_speculative_guards(Function& fn, FuzzRng& rng) {
    std::vector<Instruction*> non_term_insts;
    for (BasicBlock* bb : fn.blocks()) {
        if (!bb) continue;
        for (Instruction* inst : *bb) {
            if (inst && !inst->is_terminator() && inst->opcode() != Opcode::landing_pad) {
                non_term_insts.push_back(inst);
            }
        }
    }

    if (non_term_insts.empty()) return false;
    Instruction* target = non_term_insts[rng.pick_index(non_term_insts.size())];

    Builder b(fn);
    b.position_before(target);
    Value* cond = b.build_iconst_i32(1); // guard always succeeds
    b.build_guard(cond, "fuzz_deopt_exit");

    return true;
}

bool IrMutator::mutate_function(Function& fn, FuzzRng& rng) {
    uint32_t pass = rng.next_u32() % 5;
    bool changed = false;

    switch (pass) {
        case 0: changed = mutate_constant_immediates(fn, rng); break;
        case 1: changed = swap_commutative_operands(fn, rng); break;
        case 2: changed = replace_opcodes(fn, rng); break;
        case 3: changed = split_basic_blocks(fn, rng); break;
        case 4: changed = inject_speculative_guards(fn, rng); break;
    }

    if (changed) {
        fn.rebuild_cfg_predecessors();
        if (!verify_function(fn)) {
            return false;
        }
    }
    return changed;
}

} // namespace brass::fuzz
