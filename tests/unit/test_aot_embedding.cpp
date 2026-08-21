#include "test_framework.hpp"
#include <brass/brass.hpp>
#include <brass/mir/module.hpp>
#include <brass/mir/builder.hpp>
#include <brass/mir/verifier.hpp>
#include <brass/embedding/embedding.hpp>
#include <brass/embedding/host_gc.hpp>
#include <brass/gc/stack_map.hpp>
#include <brass/gc/stack_walker.hpp>
#include <brass/runtime/patcher.hpp>
#include "msvc_toolchain_helper.hpp"

#include <vector>
#include <string>
#include <fstream>
#include <filesystem>
#include <iostream>
#include <cstdint>
#include <cstring>

#if defined(_MSC_VER)
#include <intrin.h>
extern "C" uintptr_t brass_get_rbp();
#endif

using namespace brass;
using namespace brass::test;

namespace {

inline void get_aot_caller_frame(uintptr_t& caller_rbp, uintptr_t& caller_ip) noexcept {
#if defined(_MSC_VER) && !defined(__clang__)
    void** ret_addr_slot = reinterpret_cast<void**>(_AddressOfReturnAddress());
    caller_ip = reinterpret_cast<uintptr_t>(*ret_addr_slot);
    caller_rbp = brass_get_rbp();
#elif defined(__GNUC__) || defined(__clang__)
    void* cur_frame = __builtin_frame_address(0);
    if (cur_frame) {
        caller_rbp = *reinterpret_cast<uintptr_t*>(cur_frame);
        caller_ip = *reinterpret_cast<uintptr_t*>(reinterpret_cast<uintptr_t>(cur_frame) + 8);
    }
#else
    caller_rbp = 0;
    caller_ip = 0;
#endif
}

static HostGC* g_aot_test_gc = nullptr;
static int64_t g_subroutine_call_count = 0;
static uintptr_t g_last_r1_old_addr = 0;
static uintptr_t g_last_r2_old_addr = 0;

[[maybe_unused]] static int64_t host_hook_in_process(int64_t val) {
    return val * 7;
}

int64_t host_subroutine_gc_trigger(uintptr_t r1, uintptr_t r2) {
    g_subroutine_call_count++;
    g_last_r1_old_addr = r1;
    g_last_r2_old_addr = r2;

    if (g_aot_test_gc) {
        uintptr_t caller_rbp = 0;
        uintptr_t caller_ip = 0;
        get_aot_caller_frame(caller_rbp, caller_ip);

        // Perform Cheney GC collection walking the AOT caller frame
        g_aot_test_gc->collect(caller_rbp, caller_ip);
    }

    return 50;
}

} // anonymous namespace

TEST_CASE("Embedding API - HostEngine::compile_to_object AOT Parity with In-Memory JIT") {
    HostEngine engine;

    Module mod("test_aot_parity_mod");
    mod.add_external_symbol("host_gc_alloc");
    mod.add_external_symbol("host_gc_safepoint");

    // 1. Function with patchable call and constant
    Function* f_hook = mod.create_function("hook_default", Type::i64(), {Type::i64()});
    {
        Builder b(mod);
        b.set_function(f_hook);
        BasicBlock* entry = b.append_block("entry");
        b.position_at_end(entry);
        Value* x = b.add_block_param(entry, Type::i64());
        Value* c100 = b.build_iconst_i64(100);
        b.build_ret(b.build_add(x, c100));
        f_hook->rebuild_cfg_predecessors();
        REQUIRE(verify_function(*f_hook));
    }

    Function* f_worker = mod.create_function("worker_fn", Type::i64(), {Type::i64()});
    {
        Builder b(mod);
        b.set_function(f_worker);
        BasicBlock* entry = b.append_block("entry");
        b.position_at_end(entry);
        Value* x = b.add_block_param(entry, Type::i64());

        Value* bias = b.build_patchable_const_i64("bias_site", 42);
        Value* biased_x = b.build_add(x, bias);
        Value* call_res = b.build_patchable_call("call_site", "hook_default", Type::i64(), {biased_x});
        b.build_ret(call_res);

        f_worker->rebuild_cfg_predecessors();
        REQUIRE(verify_function(*f_worker));
    }

    // 2. In-memory JIT compile
    std::unique_ptr<CompiledModule> jit_mod = engine.compile(mod);
    REQUIRE(jit_mod != nullptr);

    const auto& jit_stack_maps = jit_mod->stack_maps();
    const auto& jit_patch_sites = jit_mod->patch_sites();
    const auto& jit_resume_tables = jit_mod->resume_tables();

    CHECK(jit_patch_sites.has_site("bias_site"));
    CHECK(jit_patch_sites.has_site("call_site"));

    // 3. AOT Object file compile
    auto tmp_dir = MsvcToolchain::temp_dir() / "aot_parity";
    std::filesystem::create_directories(tmp_dir);
    auto obj_path = tmp_dir / "parity_test.obj";

    bool ok = engine.compile_to_object(mod, obj_path.string());
    REQUIRE(ok);
    REQUIRE(std::filesystem::exists(obj_path));
    REQUIRE(std::filesystem::file_size(obj_path) > 0);

    // Verify in-memory compiler produces matching metadata
    object::ModuleCompiler compiler(engine.target());
    object::ObjectFile obj = compiler.compile(mod);

    CHECK_EQ(obj.functions.size(), 2ULL);
    CHECK_EQ(obj.stack_maps.functions().size(), jit_stack_maps.functions().size());
    CHECK_EQ(obj.patch_sites.size(), jit_patch_sites.size());
    CHECK_EQ(obj.resume_tables.size(), jit_resume_tables.size());
}

TEST_CASE("Embedding API - End-to-End AOT Linking, Dynamic Patching, and Moving GC Stack Walker") {
    if (!MsvcToolchain::is_available()) {
        std::cout << "  [SKIPPED] MSVC toolchain not available in environment\n";
        return;
    }

    auto tmp_dir = MsvcToolchain::temp_dir() / "aot_e2e";
    std::filesystem::create_directories(tmp_dir);

    // 1. Build Brass MIR Module:
    // hook_default(val: i64) -> i64 => val + 100
    // aot_gc_worker(val: i64, subroutine_fn: ptr) -> i64:
    //   ref1 = host_gc_alloc(16, 0, 1)
    //   ref1[0] = val
    //   patched_res = patchable_call "aot_hook_site" @hook_default(val)
    //   ref2 = host_gc_alloc(16, 0, 1)
    //   ref2[0] = patched_res
    //   sub_res = call_indirect subroutine_fn(ref1, ref2)  <-- GC collection triggers while ref1, ref2 are live!
    //   v1 = load ref1[0]
    //   v2 = load ref2[0]
    //   ret v1 + v2 + sub_res
    Module mod("brass_aot_e2e_mod");
    mod.add_external_symbol("host_gc_alloc");
    mod.add_external_symbol("host_gc_safepoint");

    {
        Function* f_hook = mod.create_function("hook_default", Type::i64(), {Type::i64()});
        Builder b(mod);
        b.set_function(f_hook);
        BasicBlock* entry = b.append_block("entry");
        b.position_at_end(entry);
        Value* x = b.add_block_param(entry, Type::i64());
        Value* c100 = b.build_iconst_i64(100);
        b.build_ret(b.build_add(x, c100));
        f_hook->rebuild_cfg_predecessors();
        REQUIRE(verify_function(*f_hook));
    }

    {
        Function* f_hook2 = mod.create_function("hook_v2", Type::i64(), {Type::i64()});
        Builder b(mod);
        b.set_function(f_hook2);
        BasicBlock* entry = b.append_block("entry");
        b.position_at_end(entry);
        Value* x = b.add_block_param(entry, Type::i64());
        Value* c7 = b.build_iconst_i64(7);
        b.build_ret(b.build_mul(x, c7));
        f_hook2->rebuild_cfg_predecessors();
        REQUIRE(verify_function(*f_hook2));
    }

    {
        Function* f_worker = mod.create_function("aot_gc_worker", Type::i64(), {Type::i64(), Type::ptr()});
        Builder b(mod);
        b.set_function(f_worker);
        BasicBlock* entry = b.append_block("entry");
        b.position_at_end(entry);
        Value* val = b.add_block_param(entry, Type::i64());
        Value* sub_fn = b.add_block_param(entry, Type::ptr());

        Value* sz16 = b.build_iconst_i64(16);
        Value* mask0 = b.build_iconst_i64(0);
        Value* tag1 = b.build_iconst_i32(1);

        // Allocate ref1
        Value* ref1 = b.build_call("host_gc_alloc", Type::gcref(), {sz16, mask0, tag1});
        b.build_store(Type::i64(), ref1, 0, val);

        // Patchable call site to hook_default
        Value* patched_res = b.build_patchable_call("aot_hook_site", "hook_default", Type::i64(), {val});

        // Allocate ref2
        Value* ref2 = b.build_call("host_gc_alloc", Type::gcref(), {sz16, mask0, tag1});
        b.build_store(Type::i64(), ref2, 0, patched_res);

        // Call subroutine indirectly: ref1 and ref2 are live across this subroutine call!
        Value* sub_res = b.build_call_indirect(sub_fn, Type::i64(), {ref1, ref2});

        // Load through relocated references
        Value* v1 = b.build_load(Type::i64(), ref1, 0);
        Value* v2 = b.build_load(Type::i64(), ref2, 0);

        Value* sum1 = b.build_add(v1, v2);
        Value* total = b.build_add(sum1, sub_res);
        b.build_ret(total);

        f_worker->rebuild_cfg_predecessors();
        REQUIRE(verify_function(*f_worker));
    }

    // 2. Compile to COFF .obj using HostEngine::compile_to_object
    HostEngine engine;
    auto obj_path = tmp_dir / "brass_aot_e2e.obj";
    bool compile_ok = engine.compile_to_object(mod, obj_path.string());
    REQUIRE(compile_ok);
    REQUIRE(std::filesystem::exists(obj_path));

    // Also compile to in-memory ObjectFile to retrieve patch-site offsets
    object::ModuleCompiler compiler(engine.target());
    object::ObjectFile compiled_obj = compiler.compile(mod);

    // 3. Write C++ wrapper and .def file
    auto cpp_path = tmp_dir / "aot_wrapper.cpp";
    {
        std::ofstream ofs(cpp_path);
        ofs << "#define NOMINMAX\n"
            << "#define WIN32_LEAN_AND_MEAN\n"
            << "#include <windows.h>\n"
            << "#include <cstdint>\n"
            << "#include <cstddef>\n"
            << "\n"
            << "extern \"C\" {\n"
            << "    // Export functions from Brass object\n"
            << "    __declspec(dllexport) int64_t aot_gc_worker(int64_t val, void* sub_fn);\n"
            << "    __declspec(dllexport) int64_t hook_default(int64_t val);\n"
            << "    __declspec(dllexport) int64_t hook_v2(int64_t val);\n"
            << "    __declspec(dllexport) extern const uint8_t __brass_stack_maps[];\n"
            << "\n"
            << "    // Export in-DLL C++ hook target for dynamic patching\n"
            << "    __declspec(dllexport) int64_t hook_dll_target(int64_t val) {\n"
            << "        return val * 9;\n"
            << "    }\n"
            << "\n"
            << "    typedef uintptr_t (*host_alloc_fn_t)(size_t, uint64_t, uint32_t);\n"
            << "    typedef void (*host_safepoint_fn_t)();\n"
            << "\n"
            << "    static host_alloc_fn_t g_host_alloc = nullptr;\n"
            << "    static host_safepoint_fn_t g_host_safepoint = nullptr;\n"
            << "\n"
            << "    __declspec(dllexport) void set_host_gc_callbacks(host_alloc_fn_t alloc_fn, host_safepoint_fn_t safepoint_fn) {\n"
            << "        g_host_alloc = alloc_fn;\n"
            << "        g_host_safepoint = safepoint_fn;\n"
            << "    }\n"
            << "\n"
            << "    uintptr_t host_gc_alloc(size_t size, uint64_t pointer_mask, uint32_t type_tag) {\n"
            << "        if (g_host_alloc) return g_host_alloc(size, pointer_mask, type_tag);\n"
            << "        return 0;\n"
            << "    }\n"
            << "\n"
            << "    uintptr_t brass_gc_alloc(size_t size, uint64_t pointer_mask, uint32_t type_tag) {\n"
            << "        return host_gc_alloc(size, pointer_mask, type_tag);\n"
            << "    }\n"
            << "\n"
            << "    void host_gc_safepoint() {\n"
            << "        if (g_host_safepoint) g_host_safepoint();\n"
            << "    }\n"
            << "\n"
            << "    void brass_gc_safepoint() {\n"
            << "        host_gc_safepoint();\n"
            << "    }\n"
            << "}\n";
    }

    auto def_path = tmp_dir / "aot_e2e.def";
    {
        std::ofstream ofs(def_path);
        ofs << "EXPORTS\n"
            << "    aot_gc_worker\n"
            << "    hook_default\n"
            << "    hook_v2\n"
            << "    hook_dll_target\n"
            << "    __brass_stack_maps\n"
            << "    set_host_gc_callbacks\n";
    }

    // 4. Link into DLL with MSVC
    auto dll_path = tmp_dir / "test_aot_e2e.dll";
    std::string cl_cmd = "cl.exe /nologo /LD /EHsc /MD /O2 \"" + cpp_path.string() + "\" \"" +
                         obj_path.string() + "\" /Fe:\"" + dll_path.string() + "\" /link /DEF:\"" + def_path.string() + "\"";
    int cl_res = MsvcToolchain::run_msvc_cmd(cl_cmd);
    REQUIRE_EQ(cl_res, 0);
    REQUIRE(std::filesystem::exists(dll_path));

    // 5. Load DLL and retrieve exported symbols
    HMODULE hDll = LoadLibraryA(dll_path.string().c_str());
    REQUIRE(hDll != nullptr);

    typedef void (*set_callbacks_fn_t)(uintptr_t(*)(size_t, uint64_t, uint32_t), void(*)());
    typedef int64_t (*worker_fn_t)(int64_t, void*);
    typedef int64_t (*hook_fn_t)(int64_t);

    auto p_set_callbacks = reinterpret_cast<set_callbacks_fn_t>(reinterpret_cast<void*>(GetProcAddress(hDll, "set_host_gc_callbacks")));
    auto p_worker = reinterpret_cast<worker_fn_t>(reinterpret_cast<void*>(GetProcAddress(hDll, "aot_gc_worker")));
    auto p_hook_default = reinterpret_cast<hook_fn_t>(reinterpret_cast<void*>(GetProcAddress(hDll, "hook_default")));
    auto p_hook_v2 = reinterpret_cast<hook_fn_t>(reinterpret_cast<void*>(GetProcAddress(hDll, "hook_v2")));
    auto p_hook_dll = reinterpret_cast<hook_fn_t>(reinterpret_cast<void*>(GetProcAddress(hDll, "hook_dll_target")));
    auto p_stack_maps = reinterpret_cast<const uint8_t*>(reinterpret_cast<void*>(GetProcAddress(hDll, "__brass_stack_maps")));

    REQUIRE(p_set_callbacks != nullptr);
    REQUIRE(p_worker != nullptr);
    REQUIRE(p_hook_default != nullptr);
    REQUIRE(p_hook_v2 != nullptr);
    REQUIRE(p_hook_dll != nullptr);
    REQUIRE(p_stack_maps != nullptr);

    // 6. Setup HostGC and attach decoded stack maps
    HostGC host_gc(256 * 1024);
    host_gc.set_stress_mode(false);
    g_aot_test_gc = &host_gc;

    // Decode stack maps exported by the mapped DLL
    ModuleStackMap decoded_maps = decode_stack_maps(std::span<const uint8_t>(p_stack_maps, 65536));
    CHECK(!decoded_maps.empty());

    // Register runtime function addresses from the mapped DLL
    for (const auto& fn_info : compiled_obj.functions) {
        void* addr = reinterpret_cast<void*>(GetProcAddress(hDll, fn_info.name.c_str()));
        if (addr) {
            decoded_maps.register_function_address(fn_info.name, reinterpret_cast<uintptr_t>(addr), static_cast<uint32_t>(fn_info.text_size));
        }
    }
    host_gc.set_stack_maps(&decoded_maps);

    // Connect host GC allocation callback to the DLL
    auto my_alloc = [](size_t sz, uint64_t mask, uint32_t tag) -> uintptr_t {
        return g_aot_test_gc->allocate(sz, mask, tag);
    };
    auto my_safepoint = []() {
        if (g_aot_test_gc) g_aot_test_gc->safepoint();
    };
    p_set_callbacks(+my_alloc, +my_safepoint);

    // =========================================================================
    // Phase 1: Unpatched Execution + Cheney GC Stack Walking
    // =========================================================================
    g_subroutine_call_count = 0;
    // Worker runs:
    // ref1 = 20
    // hook_default(20) = 20 + 100 = 120
    // ref2 = 120
    // subroutine called -> Cheney GC walks stack, relocates ref1 & ref2, returns 50
    // worker reads relocated ref1 (20) + ref2 (120) + 50 = 190
    int64_t res_unpatched = p_worker(20, reinterpret_cast<void*>(&host_subroutine_gc_trigger));
    CHECK_EQ(res_unpatched, 190LL);
    CHECK_EQ(g_subroutine_call_count, 1LL);
    CHECK_EQ(host_gc.collection_count(), 1ULL);

    // Verify old From-Space memory was poisoned
    CHECK_EQ(*reinterpret_cast<uint64_t*>(g_last_r1_old_addr), HostGC::POISON_PATTERN);
    CHECK_EQ(*reinterpret_cast<uint64_t*>(g_last_r2_old_addr), HostGC::POISON_PATTERN);

    // =========================================================================
    // Phase 2: Dynamic In-Memory Patching of Mapped DLL Image to hook_v2 (*7)
    // =========================================================================
    const auto* patch_site = compiled_obj.patch_sites.find_site("aot_hook_site");
    REQUIRE(patch_site != nullptr);

    // Find the CompiledFunctionInfo for "aot_gc_worker"
    const object::CompiledFunctionInfo* worker_info = nullptr;
    for (const auto& fn_info : compiled_obj.functions) {
        if (fn_info.name == "aot_gc_worker") {
            worker_info = &fn_info;
            break;
        }
    }
    REQUIRE(worker_info != nullptr);

    // Patch site offset within aot_gc_worker is patch_site->code_offset - worker_info->text_offset
    size_t site_offset_in_worker = patch_site->code_offset - worker_info->text_offset;
    uint8_t* call_site_addr = reinterpret_cast<uint8_t*>(p_worker) + site_offset_in_worker;

    DWORD old_protect = 0;
    BOOL prot_ok = VirtualProtect(call_site_addr, 32, PAGE_EXECUTE_READWRITE, &old_protect);
    REQUIRE(prot_ok != 0);

    bool patch_ok = brass_patch_call(call_site_addr, reinterpret_cast<const void*>(p_hook_v2));
    REQUIRE(patch_ok);

    VirtualProtect(call_site_addr, 32, old_protect, &old_protect);
    FlushInstructionCache(GetCurrentProcess(), call_site_addr, 32);

    // =========================================================================
    // Phase 3: Patched Execution Verification + Second GC Collection
    // =========================================================================
    // Worker runs with patched call to hook_v2 (*7):
    // ref1 = 20
    // hook_v2(20) = 20 * 7 = 140
    // ref2 = 140
    // subroutine called -> Cheney GC walks stack, relocates ref1 & ref2, returns 50
    // worker reads relocated ref1 (20) + ref2 (140) + 50 = 210
    int64_t res_patched = p_worker(20, reinterpret_cast<void*>(&host_subroutine_gc_trigger));
    CHECK_EQ(res_patched, 210LL);
    CHECK_EQ(g_subroutine_call_count, 2LL);
    CHECK_EQ(host_gc.collection_count(), 2ULL);

    // =========================================================================
    // Phase 4: Patching to in-DLL C++ Hook Target (*9)
    // =========================================================================
    prot_ok = VirtualProtect(call_site_addr, 32, PAGE_EXECUTE_READWRITE, &old_protect);
    REQUIRE(prot_ok != 0);

    bool patch_ok2 = brass_patch_call(call_site_addr, reinterpret_cast<const void*>(p_hook_dll));
    REQUIRE(patch_ok2);

    VirtualProtect(call_site_addr, 32, old_protect, &old_protect);
    FlushInstructionCache(GetCurrentProcess(), call_site_addr, 32);

    // Worker runs with patched call to hook_dll_target (*9):
    // ref1 = 20
    // hook_dll_target(20) = 20 * 9 = 180
    // ref2 = 180
    // subroutine called -> Cheney GC walks stack, relocates ref1 & ref2, returns 50
    // worker reads relocated ref1 (20) + ref2 (180) + 50 = 250
    int64_t res_patched2 = p_worker(20, reinterpret_cast<void*>(&host_subroutine_gc_trigger));
    CHECK_EQ(res_patched2, 250LL);
    CHECK_EQ(g_subroutine_call_count, 3LL);
    CHECK_EQ(host_gc.collection_count(), 3ULL);

    FreeLibrary(hDll);
    g_aot_test_gc = nullptr;
}
