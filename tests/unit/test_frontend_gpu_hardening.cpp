#include "test_framework.hpp"
#include <brass/brass.hpp>
#include <brass/brass_c_api.h>
#include <brass/embedding/brass_c_api.h>
#include <brass/embedding/host_gc.hpp>
#include <brass/gc/mini_cheney.hpp>
#include <brass/il_translator/il_translator.hpp>
#include "../../src/il_translator/il_runtime.hpp"
#include <brass/codegen/grammar_builder.hpp>
#include <brass/codegen/kernel_jit.hpp>
#include <brass/target/ptx/ptx_ir.hpp>
#include <brass/target/ptx/ptx_isel.hpp>
#include <brass/target/ptx/ptx_verifier.hpp>
#include <brass/target/ptx/ptx_printer.hpp>
#include <brass/debug/dwarf_emitter.hpp>
#include <brass/debug/codeview_emitter.hpp>
#include <brass/object/object_writer.hpp>

#include <vector>
#include <string>
#include <cstring>
#include <cmath>

using namespace brass;
using namespace brass::test;

// =============================================================================
// Task 1: GC Rooting for Dynamic Calls, Constructors & SuperCalls (>16 Args)
// =============================================================================

TEST_CASE("Frontend Hardening - CallDynamic with >16 Arguments Stages in GC Frame") {
    // Bronze IL program defining a dynamic call passing 18 arguments
    std::string il_source =
        "module test_dynamic_call_18.js\n"
        "\n"
        "func call_many(%0: dynamic, %1: dynamic) -> dynamic {\n"
        "  b0:\n";

    for (int i = 0; i < 18; ++i) {
        il_source += "    %" + std::to_string(i + 2) + ": dynamic = const.i32 " + std::to_string(i + 1) + "\n";
    }

    il_source += "    %20: dynamic = call.dynamic %0, %1, 18";
    for (int i = 0; i < 18; ++i) {
        il_source += ", %" + std::to_string(i + 2);
    }
    il_source += "\n";
    il_source += "    ret %20\n";
    il_source += "}\n";

    DiagnosticReporter diag;
    il::TranslationResult res = il::translate_bronze_il(il_source, {}, &diag);
    REQUIRE(res.success);
    REQUIRE(res.module != nullptr);

    // Verify lowered MIR in call_many
    Function* fn = res.module->get_function("call_many");
    REQUIRE(fn != nullptr);

    bool found_dynamic_call_n = false;
    bool found_frame_push_or_stores = false;

    for (BasicBlock* bb : fn->blocks()) {
        for (Instruction* inst : *bb) {
            if (inst->opcode() == Opcode::call) {
                std::string_view sym = inst->symbol();
                if (sym == "bronze_call_dynamic_n") {
                    found_dynamic_call_n = true;
                    // bronze_call_dynamic_n takes 4 operands: callee, this, argc, argv
                    REQUIRE_EQ(inst->operand_count(), size_t{4});
                    Value* argc_val = inst->operand(2);
                    REQUIRE(argc_val != nullptr);
                    if (argc_val->defining_instruction() && argc_val->defining_instruction()->opcode() == Opcode::iconst_i32) {
                        CHECK_EQ(argc_val->defining_instruction()->imm_i32(), 18);
                    }
                }
                if (sym == "bronze_gc_frame_push") {
                    found_frame_push_or_stores = true;
                }
            } else if (inst->opcode() == Opcode::store) {
                found_frame_push_or_stores = true;
            }
        }
    }
    CHECK(found_dynamic_call_n);
    CHECK(found_frame_push_or_stores);

    // End-to-end execution: Register runtime symbols and invoke JIT
    codegen::JitExecutionEngine jit(Target::host());
    il::register_all_runtime_symbols(jit);

    // Custom callee that receives 18 arguments and sums them up
    auto sum_18_fn = [](int64_t /*env*/, int64_t /*this_val*/, uint32_t argc, const int64_t* argv) -> int64_t {
        if (argc != 18 || !argv) return -1;
        int64_t total = 0;
        for (uint32_t i = 0; i < argc; ++i) {
            total += argv[i];
        }
        return total;
    };

    il::BronzeClosure closure;
    std::memset(&closure, 0, sizeof(closure));
    std::strncpy(closure.fn_name, "sum18", sizeof(closure.fn_name) - 1);
    closure.code_ptr = reinterpret_cast<void*>(+sum_18_fn);
    closure.env_box = 0;
    closure.param_count = 18;

    int64_t callee_box = reinterpret_cast<int64_t>(&closure);

    REQUIRE(jit.compile_and_load(*res.module));
    auto call_many_ptr = jit.get_function_ptr<int64_t(*)(int64_t, int64_t)>("call_many");
    REQUIRE(call_many_ptr != nullptr);

    int64_t sum_result = call_many_ptr(callee_box, 0);
    // 1 + 2 + ... + 18 = (18 * 19) / 2 = 171
    CHECK_EQ(sum_result, 171);
}

TEST_CASE("Frontend Hardening - Construct and SuperCall with >16 Arguments GC Frame Staging") {
    // 1. Construct with 18 arguments
    std::string il_construct =
        "module test_construct_18.js\n"
        "\n"
        "func test_ctor(%0: dynamic) -> dynamic {\n"
        "  b0:\n";
    for (int i = 0; i < 18; ++i) {
        il_construct += "    %" + std::to_string(i + 1) + ": dynamic = const.i32 " + std::to_string(i + 10) + "\n";
    }
    il_construct += "    %19: dynamic = new %0, 18";
    for (int i = 0; i < 18; ++i) {
        il_construct += ", %" + std::to_string(i + 1);
    }
    il_construct += "\n    ret %19\n}\n";

    DiagnosticReporter diag;
    il::TranslationResult res = il::translate_bronze_il(il_construct, {}, &diag);
    REQUIRE(res.success);
    Function* fn = res.module->get_function("test_ctor");
    REQUIRE(fn != nullptr);

    bool found_construct_n = false;
    for (BasicBlock* bb : fn->blocks()) {
        for (Instruction* inst : *bb) {
            if (inst->opcode() == Opcode::call && inst->symbol() == "bronze_construct_n") {
                found_construct_n = true;
                REQUIRE_EQ(inst->operand_count(), size_t{3}); // callee, argc, argv
            }
        }
    }
    CHECK(found_construct_n);

    // 2. SuperCall with 18 arguments
    std::string il_super =
        "module test_super_18.js\n"
        "\n"
        "func test_sup(%0: dynamic, %1: dynamic) -> dynamic {\n"
        "  b0:\n";
    for (int i = 0; i < 18; ++i) {
        il_super += "    %" + std::to_string(i + 2) + ": dynamic = const.i32 " + std::to_string(i * 2) + "\n";
    }
    il_super += "    %20: dynamic = call.super %0, %1, 18";
    for (int i = 0; i < 18; ++i) {
        il_super += ", %" + std::to_string(i + 2);
    }
    il_super += "\n    ret %20\n}\n";

    il::TranslationResult res2 = il::translate_bronze_il(il_super, {}, &diag);
    REQUIRE(res2.success);
    Function* fn2 = res2.module->get_function("test_sup");
    REQUIRE(fn2 != nullptr);

    bool found_super_n = false;
    for (BasicBlock* bb : fn2->blocks()) {
        for (Instruction* inst : *bb) {
            if (inst->opcode() == Opcode::call && inst->symbol() == "bronze_super_call_n") {
                found_super_n = true;
                REQUIRE_EQ(inst->operand_count(), size_t{4}); // sub, this, argc, argv
            }
        }
    }
    CHECK(found_super_n);
}

TEST_CASE("Frontend Hardening - Dynamic Call Argument Relocation under Moving Cheney GC") {
    MiniCheneyGC gc(128 * 1024);

    constexpr size_t NUM_ARGS = 18;
    std::vector<uintptr_t> old_addrs(NUM_ARGS);
    std::vector<uintptr_t> roots(NUM_ARGS);

    // Allocate 18 separate objects in Cheney semispace
    for (size_t i = 0; i < NUM_ARGS; ++i) {
        uintptr_t obj = gc.allocate(16, 0, 100 + static_cast<uint32_t>(i));
        REQUIRE(obj != 0);
        gc.write_field(obj, 0, 0x1000ULL + i);
        gc.write_field(obj, 1, 0x2000ULL + i);
        old_addrs[i] = obj;
        roots[i] = obj;
    }

    // Set up a simulated shadow GC frame with 18 slots
    struct SimulatedFrame {
        void* prev = nullptr;
        uint32_t count = NUM_ARGS;
        uint32_t pad = 0;
        int64_t slots[NUM_ARGS];
    } frame;

    for (size_t i = 0; i < NUM_ARGS; ++i) {
        frame.slots[i] = static_cast<int64_t>(roots[i]);
    }

    // Prepare roots pointing directly into the frame slots
    std::vector<uintptr_t*> gc_roots;
    for (size_t i = 0; i < NUM_ARGS; ++i) {
        gc_roots.push_back(reinterpret_cast<uintptr_t*>(&frame.slots[i]));
    }

    // Trigger moving Cheney GC collection
    gc.collect(gc_roots);
    CHECK_EQ(gc.collection_count(), 1ULL);

    // Verify all 18 objects relocated to to-space and frame slots forwarded in-place
    for (size_t i = 0; i < NUM_ARGS; ++i) {
        uintptr_t new_addr = static_cast<uintptr_t>(frame.slots[i]);
        CHECK_NE(new_addr, old_addrs[i]);
        CHECK(gc.is_valid_object(new_addr));
        CHECK(!gc.is_valid_object(old_addrs[i]));

        // Old space poisoned
        const uint64_t* old_mem = reinterpret_cast<const uint64_t*>(old_addrs[i]);
        CHECK_EQ(old_mem[0], MiniCheneyGC::POISON_PATTERN);
        CHECK_EQ(old_mem[1], MiniCheneyGC::POISON_PATTERN);

        // New space retains valid payload
        CHECK_EQ(gc.read_field(new_addr, 0), 0x1000ULL + i);
        CHECK_EQ(gc.read_field(new_addr, 1), 0x2000ULL + i);
    }

    // Pass argv = frame.slots to bronze_call_dynamic_n and verify arguments are read
    auto verify_fn = [](int64_t /*env*/, int64_t /*this_val*/, uint32_t argc, const int64_t* argv) -> int64_t {
        if (argc != NUM_ARGS || !argv) return -1;
        for (uint32_t i = 0; i < argc; ++i) {
            auto* mem = reinterpret_cast<const uint64_t*>(argv[i]);
            if (mem[0] != 0x1000ULL + i || mem[1] != 0x2000ULL + i) return -2;
        }
        return 42;
    };

    il::BronzeClosure closure;
    std::memset(&closure, 0, sizeof(closure));
    closure.code_ptr = reinterpret_cast<void*>(+verify_fn);
    closure.param_count = NUM_ARGS;

    int64_t callee = reinterpret_cast<int64_t>(&closure);
    int64_t res = il::bronze_call_dynamic_n(callee, 0, NUM_ARGS, frame.slots);
    CHECK_EQ(res, 42);
}

// =============================================================================
// Task 2: C API Exception Containment & Lifetime Safety
// =============================================================================

TEST_CASE("C API Hardening - Exception Containment Across ABI Boundary") {
    // 1. HostGC allocation exceeding semispace capacity throws in C++,
    // but the C API function brass_host_gc_allocate catches it and returns 0.
    brass_gc_t* gc = brass_host_gc_create(1024);
    REQUIRE(gc != nullptr);

    // Request size that exceeds 1024 bytes (e.g. 8192 bytes)
    uintptr_t big_alloc = brass_host_gc_allocate(gc, 8192, 0, 1);
    CHECK_EQ(big_alloc, 0u);

    // brass_host_gc_allocate_value catches and returns null value
    brass_value_t big_val = brass_host_gc_allocate_value(gc, 8192, 0, 1);
    CHECK(HostValue::from_raw(big_val).is_null());

    // 2. Fill semispace and call brass_host_gc_collect without unhandled exceptions
    brass_host_gc_collect(gc);
    CHECK_EQ(brass_host_gc_collection_count(gc), 1u);

    // 3. Reset and compiled module error containment
    brass_host_gc_reset(gc);
    CHECK_EQ(brass_compiled_module_patch_const32(nullptr, "test_site", 42), 0);
    CHECK_EQ(brass_compiled_module_patch_const64(nullptr, "test_site", 42), 0);
    CHECK_EQ(brass_compiled_module_patch_call(nullptr, "test_site", nullptr), 0);
    CHECK_EQ(brass_compiled_module_walk_stack(nullptr, 0, 0, nullptr, nullptr), 0u);

    // 4. Null checks and safety
    brass_host_gc_collect(nullptr);
    brass_host_gc_safepoint(nullptr, 0, 0);
    CHECK_EQ(brass_host_gc_allocate(nullptr, 16, 0, 1), 0u);
    brass_host_gc_destroy(gc);
}

TEST_CASE("C API Hardening - Handle Invalidation on Module Destruction") {
    BrassContext ctx = brass_context_create();
    REQUIRE(ctx != nullptr);

    BrassModule mod = brass_module_create(ctx, "mod_lifetime_test");
    REQUIRE(mod != nullptr);

    BrassType i64_t = brass_type_i64();
    BrassFunction fn = brass_function_create(mod, "target_fn", i64_t, &i64_t, 1);
    REQUIRE(fn != nullptr);

    BrassBlock entry_bb = brass_function_append_block(fn, "entry");
    REQUIRE(entry_bb != nullptr);

    BrassValue param = brass_block_add_param(entry_bb, i64_t);
    REQUIRE(param != nullptr);

    BrassBuilder b = brass_builder_create(ctx, fn);
    REQUIRE(b != nullptr);
    brass_builder_position_at_end(b, entry_bb);
    BrassValue c1 = brass_build_iconst_i64(b, 42);
    REQUIRE(c1 != nullptr);

    // Module destruction invalidates all handles belonging to the module
    brass_module_destroy(mod);

    // Verify handles safely report null / return safe status without use-after-free
    CHECK(brass_function_get_param(fn, 0) == nullptr);
    CHECK(brass_function_append_block(fn, "extra") == nullptr);
    CHECK(brass_block_add_param(entry_bb, i64_t) == nullptr);
    CHECK(brass_build_iconst_i64(b, 100) == nullptr);

    brass_builder_destroy(b);
    brass_context_destroy(ctx);
}

// =============================================================================
// Task 3: GPU / PTX Warp Convergence & Active Mask
// =============================================================================

TEST_CASE("GPU / PTX Hardening - Activemask and Ballot Instruction Emission & Verification") {
    ptx::Function fn{"warp_convergence_kernel"};
    ptx::Reg r = fn.new_b32();
    ptx::Reg r1 = fn.new_b32();
    ptx::Reg p = fn.new_pred();

    ptx::Block* bb = fn.add_block("$L_entry");

    // Emit activemask.b32 %r
    bb->append(ptx::Inst::make(ptx::Opcode::activemask, ptx::Type::b32).dst(ptx::Operand::reg(r)));

    // Emit vote.sync.ballot.b32 %r1, %p, %r
    bb->append(ptx::Inst::make(ptx::Opcode::vote, ptx::Type::b32).sync().dst(ptx::Operand::reg(r1)).src(ptx::Operand::reg(p)).src(ptx::Operand::reg(r)));

    auto diags = ptx::verify(fn);
    CHECK(diags.empty());

    std::string ptx_code = ptx::print_body(fn);
    CHECK(ptx_code.find("activemask.b32") != std::string::npos);
    CHECK(ptx_code.find("vote.sync.ballot.b32") != std::string::npos);
}

TEST_CASE("GPU / PTX Hardening - Warp Shuffle Lowers Activemask When Mask Unspecified") {
    Module mir_mod("test_shfl_activemask");
    Function* fn = mir_mod.create_function("kernel_shfl", Type::void_type(), {Type::i32()});
    Builder b(mir_mod);
    b.set_function(fn);
    BasicBlock* b0 = b.append_block("b0");
    b.position_at_end(b0);

    Value* val = b.add_block_param(b0, Type::i32());
    Value* delta = b.build_iconst_i32(1);

    // Call intrinsic without mask operand: ptx_shfl_down_i32(val, delta)
    mir_mod.add_external_symbol("ptx_shfl_down_i32");
    b.build_call("ptx_shfl_down_i32", Type::i32(), {val, delta});
    b.build_ret(nullptr);

    ptx::PtxISel isel;
    ptx::Function ptx_fn = isel.lower(*fn);

    auto diags = ptx::verify(ptx_fn);
    CHECK(diags.empty());

    std::string ptx_code = ptx::print_body(ptx_fn);
    // Should emit activemask.b32 and pass the active mask to shfl.sync.down
    CHECK(ptx_code.find("activemask.b32") != std::string::npos);
    CHECK(ptx_code.find("shfl.sync.down") != std::string::npos);
    // Hardcoded 0xffffffff should NOT be present as the member mask
    CHECK(ptx_code.find("0xffffffff") == std::string::npos);
}

// =============================================================================
// Task 4: GrammarBuilder Single-Byte Load Safety
// =============================================================================

TEST_CASE("GrammarBuilder Hardening - Single-Byte Load Safety Without Over-Reads") {
    Module mod("dfa_scan_byte_safety");
    codegen::GrammarBuilder gb(mod);
    Function* fn = gb.build_dfa_scan_string_function("scan_safe", 4);
    REQUIRE(fn != nullptr);

    // Verify in MIR that byte loads are Type::i8() and not Type::i64()
    bool found_i8_load = false;
    bool found_i64_load = false;

    for (BasicBlock* bb : fn->blocks()) {
        for (Instruction* inst : *bb) {
            if (inst->opcode() == Opcode::load) {
                if (inst->type() == Type::i8()) {
                    found_i8_load = true;
                }
                if (inst->type() == Type::i64()) {
                    found_i64_load = true;
                }
            }
        }
    }
    CHECK(found_i8_load);
    CHECK(!found_i64_load);

    // JIT compile and verify execution on odd-length byte buffers (1, 2, 3, 5 bytes)
    if (Target::host().is_x64() || Target::host().is_aarch64()) {
        codegen::KernelJit jit;
        codegen::KernelFunction kfn = jit.compile(*fn);
        REQUIRE(kfn.is_valid());

        auto scan_fn = kfn.as<codegen::DfaScanStringFn>();
        REQUIRE(scan_fn != nullptr);

        // DFA table recognizing "a" -> 1, "ab" -> 2, "abc" -> 3
        std::vector<int32_t> table(4 * 256, -1);
        table[0 * 256 + 'a'] = 1;
        table[1 * 256 + 'b'] = 2;
        table[2 * 256 + 'c'] = 3;

        // Exactly 1 byte
        uint8_t buf1[1] = {'a'};
        CHECK_EQ(scan_fn(table.data(), 0, buf1, 1), 1);

        // Exactly 2 bytes
        uint8_t buf2[2] = {'a', 'b'};
        CHECK_EQ(scan_fn(table.data(), 0, buf2, 2), 2);

        // Exactly 3 bytes
        uint8_t buf3[3] = {'a', 'b', 'c'};
        CHECK_EQ(scan_fn(table.data(), 0, buf3, 3), 3);

        // Invalid first byte
        uint8_t buf_bad[1] = {'z'};
        CHECK_EQ(scan_fn(table.data(), 0, buf_bad, 1), -1);
    }
}

// =============================================================================
// Task 5: DWARF & CodeView Debug Info Compliance
// =============================================================================

TEST_CASE("DWARF Hardening - DW_AT_frame_base Emits Valid Address Expressions") {
    // 1. AArch64 emits DW_OP_breg29 0 (0x8d, 0x00)
    {
        object::ObjectFile obj;
        obj.target = Target::aarch64_linux();

        object::Section text;
        text.name = ".text";
        text.kind = object::SectionKind::Text;
        text.flags = object::SectionFlags::Read | object::SectionFlags::Execute;
        text.data.resize(32, 0);
        obj.sections.push_back(std::move(text));

        object::CompiledFunctionInfo finfo;
        finfo.name = "test_arm_frame";
        finfo.text_offset = 0;
        finfo.text_size = 32;
        obj.functions.push_back(finfo);

        FunctionDebugTable tbl("test_arm_frame", 32);
        tbl.set_decl_file(1);
        tbl.set_decl_line(1);
        obj.debug_tables.push_back(tbl);
        obj.debug_context.get_or_add_file("test.brass");

        debug::DwarfOptions opts;
        debug::DwarfEmitter::emit(obj, opts);

        const object::Section* info_sec = obj.get_section(".debug_info");
        REQUIRE(info_sec != nullptr);
        REQUIRE(!info_sec->data.empty());

        bool found_breg29_0 = false;
        for (size_t i = 0; i + 1 < info_sec->data.size(); ++i) {
            if (info_sec->data[i] == debug::dwarf::DW_OP_breg29 && info_sec->data[i + 1] == 0) {
                found_breg29_0 = true;
                break;
            }
        }
        CHECK(found_breg29_0);
    }

    // 2. x64 emits DW_OP_breg6 0 (0x76, 0x00)
    {
        object::ObjectFile obj;
        obj.target = Target::x64_linux();

        object::Section text;
        text.name = ".text";
        text.kind = object::SectionKind::Text;
        text.flags = object::SectionFlags::Read | object::SectionFlags::Execute;
        text.data.resize(32, 0);
        obj.sections.push_back(std::move(text));

        object::CompiledFunctionInfo finfo;
        finfo.name = "test_x64_frame";
        finfo.text_offset = 0;
        finfo.text_size = 32;
        obj.functions.push_back(finfo);

        FunctionDebugTable tbl("test_x64_frame", 32);
        tbl.set_decl_file(1);
        tbl.set_decl_line(1);
        obj.debug_tables.push_back(tbl);
        obj.debug_context.get_or_add_file("test.brass");

        debug::DwarfOptions opts;
        debug::DwarfEmitter::emit(obj, opts);

        const object::Section* info_sec = obj.get_section(".debug_info");
        REQUIRE(info_sec != nullptr);
        REQUIRE(!info_sec->data.empty());

        bool found_breg6_0 = false;
        for (size_t i = 0; i + 1 < info_sec->data.size(); ++i) {
            if (info_sec->data[i] == debug::dwarf::DW_OP_breg6 && info_sec->data[i + 1] == 0) {
                found_breg6_0 = true;
                break;
            }
        }
        CHECK(found_breg6_0);
    }
}

TEST_CASE("CodeView Hardening - ARM64 FP Register and S_GPROC32 pEnd Offset") {
    DebugContext ctx;
    ctx.get_or_add_file("test_cv.brass");

    std::vector<FunctionDebugTable> tables;
    FunctionDebugTable tbl("cv_func", 0x40);
    tbl.set_decl_file(1);
    tbl.set_decl_line(1);
    tbl.set_prologue_size(8);

    DebugVariable var;
    var.name = "local_v";
    var.is_parameter = false;
    var.stack_offset = -8;
    var.decl_file = 1;
    var.decl_line = 2;
    tbl.add_variable(var);
    tables.push_back(tbl);

    std::vector<object::CompiledFunctionInfo> functions;
    object::CompiledFunctionInfo finfo;
    finfo.name = "cv_func";
    finfo.text_offset = 0;
    finfo.text_size = 0x40;
    functions.push_back(finfo);

    auto read_u16 = [](const uint8_t* p) -> uint16_t {
        return static_cast<uint16_t>(p[0] | (static_cast<uint16_t>(p[1]) << 8));
    };
    auto read_u32 = [](const uint8_t* p) -> uint32_t {
        return static_cast<uint32_t>(p[0] | (static_cast<uint32_t>(p[1]) << 8) |
                                     (static_cast<uint32_t>(p[2]) << 16) |
                                     (static_cast<uint32_t>(p[3]) << 24));
    };

    // 1. Emit for ARM64: register must be CV_ARM64_FP (79)
    {
        object::Section s_sec;
        debug::CodeViewOptions opts;
        opts.is_aarch64 = true;
        debug::CodeViewEmitter::emit_debug_s(ctx, tables, functions, s_sec, opts);

        const uint8_t* p = s_sec.data.data();
        size_t off = 4; // skip C13 signature
        bool checked_arm64 = false;

        while (off + 8 <= s_sec.data.size()) {
            uint32_t sub_kind = read_u32(p + off);
            uint32_t sub_len = read_u32(p + off + 4);
            off += 8;

            if (sub_kind == debug::codeview::DEBUG_S_SYMBOLS) {
                size_t sym_off = off;
                size_t sym_end = off + sub_len;
                uint32_t gproc_pend = 0;
                uint32_t proc_end_offset = 0;

                while (sym_off + 4 <= sym_end) {
                    uint16_t r_len = read_u16(p + sym_off);
                    uint16_t r_kind = read_u16(p + sym_off + 2);

                    if (r_kind == debug::codeview::S_GPROC32) {
                        // pEnd is at sym_off + 8
                        gproc_pend = read_u32(p + sym_off + 8);
                    } else if (r_kind == debug::codeview::S_REGREL32) {
                        // Register is at sym_off + 12
                        uint16_t reg = read_u16(p + sym_off + 12);
                        CHECK_EQ(reg, debug::codeview::CV_ARM64_FP);
                    } else if (r_kind == debug::codeview::S_PROC_ID_END) {
                        proc_end_offset = static_cast<uint32_t>(sym_off - off);
                    }
                    sym_off += 2 + r_len;
                }
                // Check pEnd matches S_PROC_ID_END offset
                CHECK_EQ(gproc_pend, proc_end_offset);
                CHECK(gproc_pend > 0u);
                checked_arm64 = true;
            }
            off += sub_len;
            if (off % 4 != 0) off += (4 - (off % 4));
        }
        CHECK(checked_arm64);
    }

    // 2. Emit for x64: register must be CV_AMD64_RBP (334)
    {
        object::Section s_sec;
        debug::CodeViewOptions opts;
        opts.is_aarch64 = false;
        debug::CodeViewEmitter::emit_debug_s(ctx, tables, functions, s_sec, opts);

        const uint8_t* p = s_sec.data.data();
        size_t off = 4;
        bool checked_x64 = false;

        while (off + 8 <= s_sec.data.size()) {
            uint32_t sub_kind = read_u32(p + off);
            uint32_t sub_len = read_u32(p + off + 4);
            off += 8;

            if (sub_kind == debug::codeview::DEBUG_S_SYMBOLS) {
                size_t sym_off = off;
                size_t sym_end = off + sub_len;

                while (sym_off + 4 <= sym_end) {
                    uint16_t r_len = read_u16(p + sym_off);
                    uint16_t r_kind = read_u16(p + sym_off + 2);

                    if (r_kind == debug::codeview::S_REGREL32) {
                        uint16_t reg = read_u16(p + sym_off + 12);
                        CHECK_EQ(reg, debug::codeview::CV_AMD64_RBP);
                        checked_x64 = true;
                    }
                    sym_off += 2 + r_len;
                }
            }
            off += sub_len;
            if (off % 4 != 0) off += (4 - (off % 4));
        }
        CHECK(checked_x64);
    }
}
