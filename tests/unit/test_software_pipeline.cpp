#include "test_framework.hpp"
#include <brass/codegen/software_pipeline.hpp>
#include <brass/codegen/lir.hpp>
#include <brass/mir/builder.hpp>
#include <brass/mir/module.hpp>
#include <brass/codegen/jit_exec.hpp>
#include <brass/target/x64/x64_registers.hpp>

using namespace brass;
using namespace brass::codegen;
using namespace brass::x64;

TEST_CASE("Software Pipeline - Candidate Legality Verification") {
    LirFunction fn;
    LirBlock* preheader = fn.create_block("preheader");
    LirBlock* loop = fn.create_block("loop");
    LirBlock* exit = fn.create_block("exit");

    preheader->successors.push_back(loop);
    loop->predecessors.push_back(preheader);
    loop->predecessors.push_back(loop);
    loop->successors.push_back(loop);
    loop->successors.push_back(exit);
    exit->predecessors.push_back(loop);

    VReg v_ptr = fn.allocate_vreg(RegClass::GPR, 8);
    VReg v_val = fn.allocate_vreg(RegClass::XMM, 16);
    VReg v_acc = fn.allocate_vreg(RegClass::XMM, 16);

    // Inst 0: v_val = load [v_ptr]
    auto l0 = std::make_unique<LirInst>(LirOpcode::Movaps);
    l0->add_def(LirOperand::vreg(v_val, 16));
    l0->add_use(LirOperand::mem(v_ptr, 0, 16));
    loop->append_inst(std::move(l0));

    // Inst 1: v_acc = addps v_acc, v_val
    auto a1 = std::make_unique<LirInst>(LirOpcode::Addps);
    a1->add_def(LirOperand::vreg(v_acc, 16));
    a1->add_use(LirOperand::vreg(v_acc, 16));
    a1->add_use(LirOperand::vreg(v_val, 16));
    loop->append_inst(std::move(a1));

    // Inst 2: v_ptr = add v_ptr, 16
    auto iv = std::make_unique<LirInst>(LirOpcode::Add);
    iv->add_def(LirOperand::vreg(v_ptr, 8));
    iv->add_use(LirOperand::vreg(v_ptr, 8));
    iv->add_use(LirOperand::imm(16, 8));
    loop->append_inst(std::move(iv));

    // Inst 3: cmp v_ptr, 64 (trip count 4 >= 2)
    auto cmp = std::make_unique<LirInst>(LirOpcode::Cmp);
    cmp->add_use(LirOperand::vreg(v_ptr, 8));
    cmp->add_use(LirOperand::imm(64, 8));
    loop->append_inst(std::move(cmp));

    // Inst 4: jcc Less, loop
    auto jcc = std::make_unique<LirInst>(LirOpcode::Jcc);
    jcc->condition = x64::Condition::L;
    jcc->add_use(LirOperand::label(loop->id));
    loop->append_inst(std::move(jcc));

    PipelineOptions opts;
    opts.enable_software_pipelining = true;
    CHECK(is_pipelinable_loop(fn, *loop, opts));

    // A loop with a call barrier must not be pipelinable
    auto call_inst = std::make_unique<LirInst>(LirOpcode::Call);
    call_inst->callee_symbol = "barrier_func";
    loop->instructions.insert(loop->instructions.begin(), std::move(call_inst));
    CHECK(!is_pipelinable_loop(fn, *loop, opts));
}

TEST_CASE("Software Pipeline - Prologue, Kernel, Epilogue Generation") {
    LirFunction fn;
    LirBlock* preheader = fn.create_block("preheader");
    LirBlock* loop = fn.create_block("loop");
    LirBlock* exit = fn.create_block("exit");

    preheader->successors.push_back(loop);
    loop->predecessors.push_back(preheader);
    loop->predecessors.push_back(loop);
    loop->successors.push_back(loop);
    loop->successors.push_back(exit);
    exit->predecessors.push_back(loop);

    VReg v_ptr = fn.allocate_vreg(RegClass::GPR, 8);
    VReg v_val = fn.allocate_vreg(RegClass::XMM, 16);
    VReg v_acc = fn.allocate_vreg(RegClass::XMM, 16);

    // Load: movaps v_val, [v_ptr]
    auto l0 = std::make_unique<LirInst>(LirOpcode::Movaps);
    l0->add_def(LirOperand::vreg(v_val, 16));
    l0->add_use(LirOperand::mem(v_ptr, 0, 16));
    loop->append_inst(std::move(l0));

    // Compute: addps v_acc, v_val
    auto a1 = std::make_unique<LirInst>(LirOpcode::Addps);
    a1->add_def(LirOperand::vreg(v_acc, 16));
    a1->add_use(LirOperand::vreg(v_acc, 16));
    a1->add_use(LirOperand::vreg(v_val, 16));
    loop->append_inst(std::move(a1));

    // IV: add v_ptr, 16
    auto iv = std::make_unique<LirInst>(LirOpcode::Add);
    iv->add_def(LirOperand::vreg(v_ptr, 8));
    iv->add_use(LirOperand::vreg(v_ptr, 8));
    iv->add_use(LirOperand::imm(16, 8));
    loop->append_inst(std::move(iv));

    // Cmp: cmp v_ptr, 128
    auto cmp = std::make_unique<LirInst>(LirOpcode::Cmp);
    cmp->add_use(LirOperand::vreg(v_ptr, 8));
    cmp->add_use(LirOperand::imm(128, 8));
    loop->append_inst(std::move(cmp));

    // Jcc: jcc Less, loop
    auto jcc = std::make_unique<LirInst>(LirOpcode::Jcc);
    jcc->condition = x64::Condition::L;
    jcc->add_use(LirOperand::label(loop->id));
    loop->append_inst(std::move(jcc));

    PipelineOptions opts;
    opts.enable_software_pipelining = true;
    bool ok = pipeline_loop(fn, loop, opts);
    CHECK(ok);

    // Total blocks should now be 5: preheader, loop, exit + prologue, epilogue
    CHECK_EQ(fn.blocks.size(), size_t(5));

    // Verify prologue block
    LirBlock* prologue = nullptr;
    LirBlock* epilogue = nullptr;
    for (const auto& b : fn.blocks) {
        if (b->name == "loop_prologue") prologue = b.get();
        if (b->name == "loop_epilogue") epilogue = b.get();
    }
    REQUIRE(prologue != nullptr);
    REQUIRE(epilogue != nullptr);

    // Prologue primes the initial load (Movaps) and jumps to loop (kernel)
    CHECK_EQ(prologue->instructions.size(), size_t(2));
    CHECK(prologue->instructions[0]->opcode == LirOpcode::Movaps);
    CHECK(prologue->instructions[1]->opcode == LirOpcode::Jmp);
    CHECK_EQ(prologue->instructions[1]->uses[0].label_id, loop->id);

    // Epilogue drains the final computation (Addps) and jumps to exit
    bool has_epilogue_addps = false;
    for (const auto& inst : epilogue->instructions) {
        if (inst->opcode == LirOpcode::Addps) has_epilogue_addps = true;
    }
    CHECK(has_epilogue_addps);

    // Preheader connects to prologue
    CHECK_EQ(preheader->successors[0], prologue);
    // Prologue connects to kernel
    CHECK_EQ(prologue->successors[0], loop);
    // Epilogue connects to exit
    CHECK_EQ(epilogue->successors[0], exit);
}

TEST_CASE("Software Pipeline - End-to-End Vector Sum Execution") {
    Module mod("sw_pipe_test");
    Builder b(mod);

    // Function vec_sum(ptr array, i64 count) -> f32
    Function* fn = mod.create_function("vec_sum", Type::f32(), {Type::ptr(), Type::i64()});
    b.set_function(fn);

    BasicBlock* entry = b.append_block("entry");
    Value* ptr = b.add_block_param(entry, Type::ptr());
    Value* count = b.add_block_param(entry, Type::i64());

    BasicBlock* loop_bb = b.create_block("loop");
    BasicBlock* exit_bb = b.create_block("exit");

    Value* zero_idx = b.build_iconst_i64(0);
    Value* zero_acc = b.build_vzero(Type::f32x4());
    b.build_br(loop_bb, {zero_idx, zero_acc});

    fn->append_block(loop_bb);
    b.position_at_end(loop_bb);
    Value* idx = b.add_block_param(loop_bb, Type::i64());
    Value* acc = b.add_block_param(loop_bb, Type::f32x4());

    Value* elem = b.build_load_indexed(Type::f32x4(), ptr, idx, 16, 0);
    Value* next_acc = b.build_vadd(acc, elem);
    Value* one = b.build_iconst_i64(1);
    Value* next_idx = b.build_add(idx, one);

    Value* cond = b.build_slt(next_idx, count);
    b.build_br_if(cond, loop_bb, {next_idx, next_acc}, exit_bb, {next_acc});

    fn->append_block(exit_bb);
    b.position_at_end(exit_bb);
    Value* final_vec = b.add_block_param(exit_bb, Type::f32x4());
    Value* lane0 = b.build_vextract_lane(final_vec, 0);
    b.build_ret(lane0);

    fn->rebuild_cfg_predecessors();

    // Compile with JIT and software pipelining enabled
    JitExecutionEngine jit(Target::host());
    SchedOptions sched_opts;
    sched_opts.enable_pre_ra = true;
    sched_opts.enable_post_ra = true;
    sched_opts.enable_software_pipelining = true;
    jit.set_sched_options(sched_opts);

    REQUIRE(jit.compile_and_load(mod));

    alignas(16) float data[16] = {
        1.0f, 0.0f, 0.0f, 0.0f,
        2.0f, 0.0f, 0.0f, 0.0f,
        3.0f, 0.0f, 0.0f, 0.0f,
        4.0f, 0.0f, 0.0f, 0.0f
    };

    RuntimeValue res = jit.invoke("vec_sum", {
        RuntimeValue::from_ptr(reinterpret_cast<uintptr_t>(data)),
        RuntimeValue::from_i64(4)
    });

    // Sum of lane 0 elements: 1 + 2 + 3 + 4 = 10.0f
    CHECK_EQ(res.as_f32(), 10.0f);
}
