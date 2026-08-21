#include <brass/mir/module.hpp>
#include <brass/mir/builder.hpp>
#include <brass/mir/verifier.hpp>
#include <brass/mir/printer.hpp>
#include <brass/object/object_writer.hpp>
#include <brass/object/coff_writer.hpp>
#include <brass/codegen/jit_exec.hpp>
#include <brass/gc/mini_cheney.hpp>
#include <brass/gc/runtime_gc.hpp>
#include <brass/gc/stack_walker.hpp>
#include "test_framework.hpp"
#include "msvc_toolchain_helper.hpp"

#include <vector>
#include <string>
#include <fstream>
#include <filesystem>
#include <iostream>
#include <cstdlib>

using namespace brass;
using namespace brass::object;
using namespace brass::test;

// =============================================================================
// Milestone Item (a): Emit Brass .obj and link with MSVC link.exe into a DLL
// =============================================================================
TEST_CASE("MSVC Milestone (a) - Emit Brass .obj, link into DLL with MSVC link.exe, LoadLibrary & execute") {
    if (!MsvcToolchain::is_available()) {
        std::cout << "       [SKIPPED] MSVC toolchain not available in environment\n";
        return;
    }

    auto tmp_dir = MsvcToolchain::temp_dir() / "milestone_a";
    std::filesystem::create_directories(tmp_dir);

    // 1. Build Brass MIR Module
    Module mod("brass_math_module");
    {
        // fn brass_compute(a: i64, b: i64, c: i64) -> i64
        // computes (a + b) * c - (a ^ b)
        Function* fn1 = mod.create_function("brass_compute", Type::i64(), {Type::i64(), Type::i64(), Type::i64()});
        Builder b(mod);
        b.set_function(fn1);
        BasicBlock* entry = b.append_block("entry");
        Value* a = b.add_block_param(entry, Type::i64());
        Value* val_b = b.add_block_param(entry, Type::i64());
        Value* c = b.add_block_param(entry, Type::i64());

        Value* sum = b.build_add(a, val_b);
        Value* prod = b.build_mul(sum, c);
        Value* xor_val = b.build_xor(a, val_b);
        Value* res = b.build_sub(prod, xor_val);
        b.build_ret(res);

        fn1->rebuild_cfg_predecessors();
        REQUIRE(verify_function(*fn1));
    }

    {
        // fn brass_accumulate_array(ptr: ptr, len: i64) -> i64
        Function* fn2 = mod.create_function("brass_accumulate_array", Type::i64(), {Type::ptr(), Type::i64()});
        Builder b(mod);
        b.set_function(fn2);

        BasicBlock* entry = b.append_block("entry");
        BasicBlock* loop_hdr = b.append_block("loop_hdr");
        BasicBlock* loop_body = b.append_block("loop_body");
        BasicBlock* exit_bb = b.append_block("exit");

        b.position_at_end(entry);
        Value* arr_ptr = b.add_block_param(entry, Type::ptr());
        Value* count = b.add_block_param(entry, Type::i64());

        Value* zero = b.build_iconst_i64(0);
        b.build_br(loop_hdr, {zero, zero});

        b.position_at_end(loop_hdr);
        Value* idx = b.add_block_param(loop_hdr, Type::i64());
        Value* acc = b.add_block_param(loop_hdr, Type::i64());

        Value* cmp = b.build_slt(idx, count);
        b.build_br_if(cmp, loop_body, {}, exit_bb, {});

        b.position_at_end(loop_body);
        Value* elem = b.build_load_indexed(Type::i64(), arr_ptr, idx, 8, 0);
        Value* new_acc = b.build_add(acc, elem);
        Value* one = b.build_iconst_i64(1);
        Value* next_idx = b.build_add(idx, one);
        b.build_br(loop_hdr, {next_idx, new_acc});

        b.position_at_end(exit_bb);
        b.build_ret(acc);

        fn2->rebuild_cfg_predecessors();
        REQUIRE(verify_function(*fn2));
    }

    // 2. Compile to COFF .obj
    ObjectFile obj = compile_module_to_object(mod, Target::x64_windows());
    std::vector<uint8_t> coff_bytes = emit_coff_object(obj);
    REQUIRE(!coff_bytes.empty());

    auto obj_path = tmp_dir / "brass_math.obj";
    {
        std::ofstream ofs(obj_path, std::ios::binary);
        ofs.write(reinterpret_cast<const char*>(coff_bytes.data()), static_cast<std::streamsize>(coff_bytes.size()));
    }
    REQUIRE(std::filesystem::exists(obj_path));

    // 3. Write MSVC C++ wrapper source
    auto cpp_path = tmp_dir / "msvc_wrapper_a.cpp";
    {
        std::ofstream ofs(cpp_path);
        ofs << "#include <cstdint>\n"
            << "#include <vector>\n"
            << "extern \"C\" {\n"
            << "    int64_t brass_compute(int64_t a, int64_t b, int64_t c);\n"
            << "    int64_t brass_accumulate_array(const int64_t* ptr, int64_t count);\n"
            << "\n"
            << "    __declspec(dllexport) int64_t msvc_call_brass_compute(int64_t a, int64_t b, int64_t c) {\n"
            << "        return brass_compute(a, b, c);\n"
            << "    }\n"
            << "\n"
            << "    __declspec(dllexport) int64_t msvc_call_brass_array(int64_t count) {\n"
            << "        std::vector<int64_t> data(count);\n"
            << "        for (int64_t i = 0; i < count; ++i) {\n"
            << "            data[i] = (i + 1) * 3;\n"
            << "        }\n"
            << "        return brass_accumulate_array(data.data(), count);\n"
            << "    }\n"
            << "}\n";
    }
    REQUIRE(std::filesystem::exists(cpp_path));

    // 4. Compile and link with MSVC cl.exe / link.exe into DLL
    auto dll_path = tmp_dir / "test_milestone_a.dll";
    std::string build_cmd = "cl.exe /nologo /LD /EHsc /MD /O2 \"" + cpp_path.string() + "\" \"" + obj_path.string() + "\" /Fe:\"" + dll_path.string() + "\"";
    int compile_res = MsvcToolchain::run_msvc_cmd(build_cmd);
    REQUIRE_EQ(compile_res, 0);
    REQUIRE(std::filesystem::exists(dll_path));

    // 5. Load DLL, execute exported functions via GetProcAddress, and verify
    HMODULE hDll = LoadLibraryA(dll_path.string().c_str());
    REQUIRE(hDll != nullptr);

    typedef int64_t (*compute_fn_t)(int64_t, int64_t, int64_t);
    typedef int64_t (*array_fn_t)(int64_t);

    auto p_compute = reinterpret_cast<compute_fn_t>(reinterpret_cast<void*>(GetProcAddress(hDll, "msvc_call_brass_compute")));
    auto p_array = reinterpret_cast<array_fn_t>(reinterpret_cast<void*>(GetProcAddress(hDll, "msvc_call_brass_array")));

    REQUIRE(p_compute != nullptr);
    REQUIRE(p_array != nullptr);

    // Compute test: (12 + 34) * 5 - (12 ^ 34) = 46 * 5 - 46 = 230 - 46 = 184
    int64_t res_compute = p_compute(12, 34, 5);
    CHECK_EQ(res_compute, 184LL);

    // Array test: sum of (i + 1) * 3 for i in 0..49 = 3 * (50 * 51 / 2) = 3825
    int64_t res_array = p_array(50);
    CHECK_EQ(res_array, 3825LL);

    BOOL freed = FreeLibrary(hDll);
    CHECK(freed != 0);
}

// =============================================================================
// Milestone Item (b): Bi-directional Calls Across Custom Calling Conventions
// =============================================================================
TEST_CASE("MSVC Milestone (b) - Bi-directional calls between MSVC and Brass via custom calling convention") {
    if (!MsvcToolchain::is_available()) {
        std::cout << "       [SKIPPED] MSVC toolchain not available in environment\n";
        return;
    }

    auto tmp_dir = MsvcToolchain::temp_dir() / "milestone_b";
    std::filesystem::create_directories(tmp_dir);

    // Define custom CallingConvention: args in R10, R11, R12; return in RAX
    using namespace brass::x64;
    CustomCallingConvConfig cfg;
    cfg.arg_gprs = { GPR::R10, GPR::R11, GPR::R12 };
    cfg.ret_gprs = { GPR::RAX };
    cfg.callee_saved_gprs = reg_mask(GPR::RBX) | reg_mask(GPR::RBP) | reg_mask(GPR::R13) |
                            reg_mask(GPR::R14) | reg_mask(GPR::R15) | reg_mask(GPR::RSI) |
                            reg_mask(GPR::RDI);
    cfg.caller_saved_gprs = reg_mask(GPR::R10) | reg_mask(GPR::R11) | reg_mask(GPR::R12) |
                            reg_mask(GPR::RCX) | reg_mask(GPR::RDX) | reg_mask(GPR::R8)  |
                            reg_mask(GPR::R9)  | reg_mask(GPR::RAX);
    cfg.shadow_space = 0;

    CallingConvention custom_cc = CallingConvention::custom(cfg);

    // 1. Build Brass MIR Module compiled with custom calling convention
    Module mod("brass_custom_cc_module");
    mod.add_external_symbol("msvc_custom_target");

    {
        // fn brass_custom_worker(a: i64, b: i64, c: i64) -> i64
        // computes (a * 7) + (b * 13) + (c * 17)
        Function* fn1 = mod.create_function("brass_custom_worker", Type::i64(), {Type::i64(), Type::i64(), Type::i64()});
        Builder b(mod);
        b.set_function(fn1);
        BasicBlock* entry = b.append_block("entry");
        Value* a = b.add_block_param(entry, Type::i64());
        Value* val_b = b.add_block_param(entry, Type::i64());
        Value* c = b.add_block_param(entry, Type::i64());

        Value* c7 = b.build_iconst_i64(7);
        Value* c13 = b.build_iconst_i64(13);
        Value* c17 = b.build_iconst_i64(17);

        Value* t1 = b.build_mul(a, c7);
        Value* t2 = b.build_mul(val_b, c13);
        Value* t3 = b.build_mul(c, c17);

        Value* s1 = b.build_add(t1, t2);
        Value* res = b.build_add(s1, t3);
        b.build_ret(res);

        fn1->rebuild_cfg_predecessors();
        REQUIRE(verify_function(*fn1));
    }

    {
        // fn brass_custom_caller(a: i64, b: i64, c: i64) -> i64
        // calls external msvc_custom_target(a, b, c) and adds 100
        Function* fn2 = mod.create_function("brass_custom_caller", Type::i64(), {Type::i64(), Type::i64(), Type::i64()});
        Builder b(mod);
        b.set_function(fn2);
        BasicBlock* entry = b.append_block("entry");
        Value* a = b.add_block_param(entry, Type::i64());
        Value* val_b = b.add_block_param(entry, Type::i64());
        Value* c = b.add_block_param(entry, Type::i64());

        Value* call_res = b.build_call("msvc_custom_target", Type::i64(), {a, val_b, c});
        Value* c100 = b.build_iconst_i64(100);
        Value* res = b.build_add(call_res, c100);
        b.build_ret(res);

        fn2->rebuild_cfg_predecessors();
        REQUIRE(verify_function(*fn2));
    }

    // Compile Brass Module with custom CC
    ModuleCompiler compiler(Target::x64_windows(), custom_cc);
    ObjectFile obj = compiler.compile(mod);
    std::vector<uint8_t> coff_bytes = emit_coff_object(obj);
    REQUIRE(!coff_bytes.empty());

    auto obj_path = tmp_dir / "brass_custom_cc.obj";
    {
        std::ofstream ofs(obj_path, std::ios::binary);
        ofs.write(reinterpret_cast<const char*>(coff_bytes.data()), static_cast<std::streamsize>(coff_bytes.size()));
    }

    // 2. Write MASM helper for invoking and handling custom calling convention
    auto asm_path = tmp_dir / "custom_cc_stubs.asm";
    {
        std::ofstream ofs(asm_path);
        ofs << ".code\n"
            << "\n"
            << "; int64_t invoke_custom_brass(fn_ptr [rcx], a [rdx], b [r8], c [r9])\n"
            << "invoke_custom_brass PROC\n"
            << "    push rbp\n"
            << "    mov rbp, rsp\n"
            << "    push rbx\n"
            << "    push rsi\n"
            << "    push rdi\n"
            << "    push r12\n"
            << "    push r13\n"
            << "    push r14\n"
            << "    push r15\n"
            << "    sub rsp, 40\n"
            << "\n"
            << "    ; Save fn_ptr in rax, args into r10, r11, r12\n"
            << "    mov rax, rcx\n"
            << "    mov r10, rdx\n"
            << "    mov r11, r8\n"
            << "    mov r12, r9\n"
            << "\n"
            << "    ; Set known canary values in non-volatile registers\n"
            << "    mov rbx, 1111222233334444h\n"
            << "    mov rsi, 2222333344445555h\n"
            << "    mov rdi, 3333444455556666h\n"
            << "    mov r13, 5555666677778888h\n"
            << "    mov r14, 6666777788889999h\n"
            << "    mov r15, 777788889999AAAAh\n"
            << "\n"
            << "    call rax\n"
            << "\n"
            << "    ; Check if non-volatile registers were preserved\n"
            << "    ; rax has return value, preserve it in r10\n"
            << "    mov r10, rax\n"
            << "    mov r11, 1111222233334444h\n"
            << "    cmp rbx, r11\n"
            << "    jne fail_regs\n"
            << "    mov r11, 2222333344445555h\n"
            << "    cmp rsi, r11\n"
            << "    jne fail_regs\n"
            << "    mov r11, 3333444455556666h\n"
            << "    cmp rdi, r11\n"
            << "    jne fail_regs\n"
            << "    mov r11, 5555666677778888h\n"
            << "    cmp r13, r11\n"
            << "    jne fail_regs\n"
            << "    mov r11, 6666777788889999h\n"
            << "    cmp r14, r11\n"
            << "    jne fail_regs\n"
            << "    mov r11, 777788889999AAAAh\n"
            << "    cmp r15, r11\n"
            << "    jne fail_regs\n"
            << "\n"
            << "    mov rax, r10 ; restore return value\n"
            << "    jmp done\n"
            << "fail_regs:\n"
            << "    mov rax, -999999\n"
            << "done:\n"
            << "    add rsp, 40\n"
            << "    pop r15\n"
            << "    pop r14\n"
            << "    pop r13\n"
            << "    pop r12\n"
            << "    pop rdi\n"
            << "    pop rsi\n"
            << "    pop rbx\n"
            << "    pop rbp\n"
            << "    ret\n"
            << "invoke_custom_brass ENDP\n"
            << "\n"
            << "; int64_t msvc_custom_target(r10: a, r11: b, r12: c) -> rax\n"
            << "; computes (r10 + r11) ^ r12 and clobbers rcx, rdx, r8, r9\n"
            << "msvc_custom_target PROC\n"
            << "    mov rcx, 0FFFFFFFFh\n"
            << "    mov rdx, 0AAAAAAAAh\n"
            << "    mov r8,  055555555h\n"
            << "    mov r9,  012345678h\n"
            << "    mov rax, r10\n"
            << "    add rax, r11\n"
            << "    xor rax, r12\n"
            << "    ret\n"
            << "msvc_custom_target ENDP\n"
            << "\n"
            << "END\n";
    }

    // 3. Write MSVC C++ wrapper source
    auto cpp_path = tmp_dir / "msvc_wrapper_b.cpp";
    {
        std::ofstream ofs(cpp_path);
        ofs << "#include <cstdint>\n"
            << "extern \"C\" {\n"
            << "    int64_t brass_custom_worker(int64_t a, int64_t b, int64_t c);\n"
            << "    int64_t brass_custom_caller(int64_t a, int64_t b, int64_t c);\n"
            << "    int64_t invoke_custom_brass(void* fn, int64_t a, int64_t b, int64_t c);\n"
            << "\n"
            << "    __declspec(dllexport) int64_t test_msvc_to_brass(int64_t a, int64_t b, int64_t c) {\n"
            << "        return invoke_custom_brass((void*)brass_custom_worker, a, b, c);\n"
            << "    }\n"
            << "\n"
            << "    __declspec(dllexport) int64_t test_brass_to_msvc(int64_t a, int64_t b, int64_t c) {\n"
            << "        return invoke_custom_brass((void*)brass_custom_caller, a, b, c);\n"
            << "    }\n"
            << "}\n";
    }

    // 4. Assemble MASM stubs
    auto asm_obj_path = tmp_dir / "custom_cc_stubs.obj";
    std::string ml_cmd = "ml64.exe /nologo /c /Fo\"" + asm_obj_path.string() + "\" \"" + asm_path.string() + "\"";
    int ml_res = MsvcToolchain::run_msvc_cmd(ml_cmd);
    REQUIRE_EQ(ml_res, 0);

    // 5. Compile and link DLL
    auto dll_path = tmp_dir / "test_milestone_b.dll";
    std::string cl_cmd = "cl.exe /nologo /LD /EHsc /MD /O2 \"" + cpp_path.string() + "\" \"" +
                         obj_path.string() + "\" \"" + asm_obj_path.string() + "\" /Fe:\"" + dll_path.string() + "\"";
    int cl_res = MsvcToolchain::run_msvc_cmd(cl_cmd);
    REQUIRE_EQ(cl_res, 0);

    // 6. Load DLL and verify bi-directional calls
    HMODULE hDll = LoadLibraryA(dll_path.string().c_str());
    REQUIRE(hDll != nullptr);

    typedef int64_t (*test_fn_t)(int64_t, int64_t, int64_t);
    auto p_m2b = reinterpret_cast<test_fn_t>(reinterpret_cast<void*>(GetProcAddress(hDll, "test_msvc_to_brass")));
    auto p_b2m = reinterpret_cast<test_fn_t>(reinterpret_cast<void*>(GetProcAddress(hDll, "test_brass_to_msvc")));

    REQUIRE(p_m2b != nullptr);
    REQUIRE(p_b2m != nullptr);

    // 1. MSVC -> Brass: (10 * 7) + (20 * 13) + (30 * 17) = 70 + 260 + 510 = 840
    int64_t res1 = p_m2b(10, 20, 30);
    CHECK_EQ(res1, 840LL);

    // 2. Brass -> MSVC: ((15 + 25) ^ 7) + 100 = (40 ^ 7) + 100 = 47 + 100 = 147
    int64_t res2 = p_b2m(15, 25, 7);
    CHECK_EQ(res2, 147LL);

    FreeLibrary(hDll);
}

// =============================================================================
// Milestone Item (c): Win64 SEH & C++ Exception Unwind Interoperability
// =============================================================================
TEST_CASE("MSVC Milestone (c) - Real Windows OS SEH & C++ exception unwinding through Brass frame") {
    if (!MsvcToolchain::is_available()) {
        std::cout << "       [SKIPPED] MSVC toolchain not available in environment\n";
        return;
    }

    auto tmp_dir = MsvcToolchain::temp_dir() / "milestone_c";
    std::filesystem::create_directories(tmp_dir);

    // 1. Build Brass MIR Module:
    // fn brass_unwind_frame(callback_fn: ptr, code: i64) -> i64
    // Allocates frame, saves callee-saved registers (forces non-leaf frame with .pdata / .xdata unwind info),
    // calls callback_fn(code), and returns.
    Module mod("brass_unwind_module");
    mod.add_external_symbol("msvc_unwind_callback");

    Function* fn = mod.create_function("brass_unwind_frame", Type::i64(), {Type::ptr(), Type::i64()});
    Builder b(mod);
    b.set_function(fn);
    BasicBlock* entry = b.append_block("entry");
    Value* cb = b.add_block_param(entry, Type::ptr());
    Value* code = b.add_block_param(entry, Type::i64());

    // Generate complex expressions to force callee-saved register spills
    Value* c1 = b.build_iconst_i64(111);
    Value* c2 = b.build_iconst_i64(222);
    Value* c3 = b.build_iconst_i64(333);
    Value* a1 = b.build_add(code, c1);
    Value* a2 = b.build_add(code, c2);
    Value* a3 = b.build_add(code, c3);

    // Call callback_fn(code) indirectly through function pointer
    Value* call_res = b.build_call_indirect(cb, Type::i64(), {code});

    // Sum everything up
    Value* s1 = b.build_add(call_res, a1);
    Value* s2 = b.build_add(s1, a2);
    Value* res = b.build_add(s2, a3);
    b.build_ret(res);

    fn->rebuild_cfg_predecessors();
    REQUIRE(verify_function(*fn));

    // Compile to COFF .obj
    ObjectFile obj = compile_module_to_object(mod, Target::x64_windows());
    std::vector<uint8_t> coff_bytes = emit_coff_object(obj);
    REQUIRE(!coff_bytes.empty());

    auto obj_path = tmp_dir / "brass_unwind.obj";
    {
        std::ofstream ofs(obj_path, std::ios::binary);
        ofs.write(reinterpret_cast<const char*>(coff_bytes.data()), static_cast<std::streamsize>(coff_bytes.size()));
    }

    // 2. Write MSVC C++ wrapper that sets up try/catch and __try/__except
    auto cpp_path = tmp_dir / "msvc_wrapper_c.cpp";
    {
        std::ofstream ofs(cpp_path);
        ofs << "#define NOMINMAX\n"
            << "#define WIN32_LEAN_AND_MEAN\n"
            << "#include <windows.h>\n"
            << "#include <cstdint>\n"
            << "#include <stdexcept>\n"
            << "#include <string>\n"
            << "\n"
            << "extern \"C\" {\n"
            << "    int64_t brass_unwind_frame(void* cb, int64_t code);\n"
            << "\n"
            << "    // Callback target called by Brass\n"
            << "    __declspec(noinline) int64_t msvc_unwind_callback(int64_t code) {\n"
            << "        if (code == 42) {\n"
            << "            throw std::runtime_error(\"brass_seh_unwind_verified\");\n"
            << "        } else if (code == 99) {\n"
            << "            RaiseException(0xE0000001, 0, 0, nullptr);\n"
            << "        }\n"
            << "        return code * 2;\n"
            << "    }\n"
            << "\n"
            << "    // Test C++ exception propagation through Brass frame\n"
            << "    __declspec(dllexport) int64_t test_cpp_unwind(int64_t code, char* out_msg, int max_len) {\n"
            << "        try {\n"
            << "            return brass_unwind_frame((void*)msvc_unwind_callback, code);\n"
            << "        } catch (const std::runtime_error& ex) {\n"
            << "            const char* what = ex.what();\n"
            << "            int i = 0;\n"
            << "            while (what[i] && i < max_len - 1) {\n"
            << "                out_msg[i] = what[i];\n"
            << "                i++;\n"
            << "            }\n"
            << "            out_msg[i] = '\\0';\n"
            << "            return 10042;\n"
            << "        }\n"
            << "    }\n"
            << "\n"
            << "    // Test Windows SEH exception propagation through Brass frame\n"
            << "    __declspec(dllexport) int64_t test_seh_unwind(int64_t code) {\n"
            << "        __try {\n"
            << "            return brass_unwind_frame((void*)msvc_unwind_callback, code);\n"
            << "        } __except (GetExceptionCode() == 0xE0000001 ? EXCEPTION_EXECUTE_HANDLER : EXCEPTION_CONTINUE_SEARCH) {\n"
            << "            return 10099;\n"
            << "        }\n"
            << "    }\n"
            << "}\n";
    }

    // 3. Compile and link DLL with MSVC using /EHs (standard C++ exception handling allowing extern "C" throwing)
    auto dll_path = tmp_dir / "test_milestone_c.dll";
    std::string cl_cmd = "cl.exe /nologo /LD /EHs /MD /O2 \"" + cpp_path.string() + "\" \"" +
                         obj_path.string() + "\" /Fe:\"" + dll_path.string() + "\"";
    int cl_res = MsvcToolchain::run_msvc_cmd(cl_cmd);
    REQUIRE_EQ(cl_res, 0);

    // 4. Load DLL and execute tests
    HMODULE hDll = LoadLibraryA(dll_path.string().c_str());
    REQUIRE(hDll != nullptr);

    typedef int64_t (*cpp_unwind_fn)(int64_t, char*, int);
    typedef int64_t (*seh_unwind_fn)(int64_t);

    auto p_cpp = reinterpret_cast<cpp_unwind_fn>(reinterpret_cast<void*>(GetProcAddress(hDll, "test_cpp_unwind")));
    auto p_seh = reinterpret_cast<seh_unwind_fn>(reinterpret_cast<void*>(GetProcAddress(hDll, "test_seh_unwind")));

    REQUIRE(p_cpp != nullptr);
    REQUIRE(p_seh != nullptr);

    // 1. Normal execution without throwing (code = 10)
    char msg_buf[128] = {0};
    int64_t norm_res = p_cpp(10, msg_buf, sizeof(msg_buf));
    // call_res = 20; a1 = 121; a2 = 232; a3 = 343; sum = 20 + 121 + 232 + 343 = 716
    CHECK_EQ(norm_res, 716LL);

    // 2. C++ Exception thrown from MSVC callback, unwinds through Brass frame, caught in MSVC catch block!
    std::memset(msg_buf, 0, sizeof(msg_buf));
    int64_t cpp_res = p_cpp(42, msg_buf, sizeof(msg_buf));
    CHECK_EQ(cpp_res, 10042LL);
    CHECK_EQ(std::string(msg_buf), "brass_seh_unwind_verified");

    // 3. Windows SEH Exception (RaiseException) unwinds through Brass frame, caught in MSVC __except block!
    int64_t seh_res = p_seh(99);
    CHECK_EQ(seh_res, 10099LL);

    FreeLibrary(hDll);
}

// =============================================================================
// Milestone Item (d): Live GC Stack Walker Across Brass -> MSVC -> Brass Sandwich
// =============================================================================

TEST_CASE("MSVC Milestone (d) - Stack walker finds live gcrefs across Brass -> MSVC -> Brass sandwich") {
    if (!MsvcToolchain::is_available()) {
        std::cout << "  [SKIP] MSVC toolchain not found, skipping milestone test (d)\n";
        return;
    }

    auto tmp_dir = MsvcToolchain::temp_dir() / "milestone_d";
    std::filesystem::create_directories(tmp_dir);

    // 1. Write MSVC MASM stub that establishes standard RBP frame and calls Brass Frame 3
    auto asm_path = tmp_dir / "sandwich_stub.asm";
    {
        std::ofstream ofs(asm_path);
        ofs << ".code\n"
            << "\n"
            << "PUBLIC msvc_sandwich_frame2\n"
            << "\n"
            << "; int64_t msvc_sandwich_frame2(fn_frame3 [rcx], val [rdx])\n"
            << "msvc_sandwich_frame2 PROC\n"
            << "    push rbp\n"
            << "    mov rbp, rsp\n"
            << "    push rbx\n"
            << "    sub rsp, 40\n"
            << "    mov rax, rcx\n"
            << "    mov rcx, rdx\n"
            << "    call rax\n"
            << "    add rax, 10000\n"
            << "    add rsp, 40\n"
            << "    pop rbx\n"
            << "    pop rbp\n"
            << "    ret\n"
            << "msvc_sandwich_frame2 ENDP\n"
            << "\n"
            << "END\n";
    }

    // 2. Assemble MASM stub and link into test_milestone_d.dll with MSVC
    auto obj_asm_path = tmp_dir / "sandwich_stub.obj";
    std::string ml_cmd = "ml64.exe /nologo /c /Fo\"" + obj_asm_path.string() + "\" \"" + asm_path.string() + "\"";
    int ml_res = MsvcToolchain::run_msvc_cmd(ml_cmd);
    REQUIRE_EQ(ml_res, 0);

    auto def_path = tmp_dir / "sandwich.def";
    {
        std::ofstream ofs(def_path);
        ofs << "EXPORTS\n    msvc_sandwich_frame2\n";
    }

    auto dll_path = tmp_dir / "test_milestone_d.dll";
    std::string link_cmd = "link.exe /nologo /DLL /NOENTRY /DEF:\"" + def_path.string() + "\" \"" +
                           obj_asm_path.string() + "\" /OUT:\"" + dll_path.string() + "\"";
    int link_res = MsvcToolchain::run_msvc_cmd(link_cmd);
    REQUIRE_EQ(link_res, 0);

    HMODULE hDll = LoadLibraryA(dll_path.string().c_str());
    REQUIRE(hDll != nullptr);

    typedef int64_t (*Frame2Func)(void*, int64_t);
    auto p_sandwich_frame2 = reinterpret_cast<Frame2Func>(reinterpret_cast<void*>(GetProcAddress(hDll, "msvc_sandwich_frame2")));
    REQUIRE(p_sandwich_frame2 != nullptr);

    // 3. Build Brass MIR Module:
    // Frame 3: brass_frame3(val: i64) -> i64
    // - Allocates GC object ref2 with value val (e.g. 2222)
    // - Triggers GC collection while ref2 is live on stack
    // - Reads ref2 after GC relocation, adds 10, returns.
    Module mod("sandwich_gc_module");
    mod.add_external_symbol("brass_gc_alloc");
    mod.add_external_symbol("brass_gc_safepoint");

    {
        Function* fn3 = mod.create_function("brass_frame3", Type::i64(), {Type::i64()});
        Builder b(mod);
        b.set_function(fn3);
        BasicBlock* entry = b.append_block("entry");
        Value* v = b.add_block_param(entry, Type::i64());

        Value* sz16 = b.build_iconst_i64(16);
        Value* mask0 = b.build_iconst_i64(0);
        Value* tag1 = b.build_iconst_i32(1);

        // Allocate ref2
        Value* ref2 = b.build_call("brass_gc_alloc", Type::gcref(), {sz16, mask0, tag1});
        b.build_store(Type::i64(), ref2, 0, v);

        // Explicit safepoint trigger while ref2 is live on Frame 3 stack
        b.build_safepoint();

        // Read ref2 from relocated address
        Value* read_v = b.build_load(Type::i64(), ref2, 0);
        Value* ten = b.build_iconst_i64(10);
        Value* res3 = b.build_add(read_v, ten);
        b.build_ret(res3);

        fn3->rebuild_cfg_predecessors();
        REQUIRE(verify_function(*fn3));
    }

    {
        // Frame 1: brass_frame1(msvc_fn_ptr: ptr, frame3_fn_ptr: ptr) -> i64
        // - Allocates GC object ref1 with value 1111
        // - Calls MSVC Frame 2: msvc_sandwich_frame2(frame3_fn_ptr, 2222) while ref1 is live!
        // - Reads ref1 after returning from MSVC Frame 2, adds result, returns.
        Function* fn1 = mod.create_function("brass_frame1", Type::i64(), {Type::ptr(), Type::ptr()});
        Builder b(mod);
        b.set_function(fn1);
        BasicBlock* entry = b.append_block("entry");
        Value* msvc_fn = b.add_block_param(entry, Type::ptr());
        Value* f3_fn = b.add_block_param(entry, Type::ptr());

        Value* sz16 = b.build_iconst_i64(16);
        Value* mask0 = b.build_iconst_i64(0);
        Value* tag1 = b.build_iconst_i32(1);

        // Allocate ref1
        Value* ref1 = b.build_call("brass_gc_alloc", Type::gcref(), {sz16, mask0, tag1});
        Value* v1111 = b.build_iconst_i64(1111);
        b.build_store(Type::i64(), ref1, 0, v1111);

        // Call MSVC Frame 2 indirectly through function pointer
        Value* v2222 = b.build_iconst_i64(2222);
        Value* call_res = b.build_call_indirect(msvc_fn, Type::i64(), {f3_fn, v2222});

        // Read ref1 from relocated address
        Value* read_v1 = b.build_load(Type::i64(), ref1, 0);
        Value* total = b.build_add(read_v1, call_res);
        b.build_ret(total);

        fn1->rebuild_cfg_predecessors();
        REQUIRE(verify_function(*fn1));
    }

    // 4. Run with Native JIT Execution & MiniCheney Moving GC
    MiniCheneyGC gc(128 * 1024);
    gc.set_stress_mode(false); // Clean explicit safepoints
    brass_set_active_gc(&gc);

    codegen::JitExecutionEngine jit(Target::host());

    bool ok = jit.compile_and_load(mod);
    REQUIRE(ok);

    brass_set_active_stack_maps(&jit.stack_maps());

    auto* f3_addr = jit.get_symbol_address("brass_frame3");
    REQUIRE(f3_addr != nullptr);

    uint64_t initial_collections = gc.collection_count();

    // Invoke Frame 1: Frame 1 (1111) -> MSVC Frame 2 (+10000) -> Frame 3 (2222 + 10 = 2232)
    // Expected return: 1111 + 10000 + 2232 = 13343
    RuntimeValue res = jit.invoke("brass_frame1", {
        RuntimeValue::from_ptr(reinterpret_cast<uintptr_t>(p_sandwich_frame2)),
        RuntimeValue::from_ptr(reinterpret_cast<uintptr_t>(f3_addr))
    });

    CHECK_EQ(res.as_i64(), 13343LL);
    CHECK(gc.collection_count() > initial_collections);

    FreeLibrary(hDll);
}
