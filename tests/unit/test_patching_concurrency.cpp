#include "test_framework.hpp"
#include <brass/brass.hpp>
#include <thread>
#include <atomic>
#include <vector>
#include <chrono>

using namespace brass;
using namespace brass::runtime;
using namespace brass::codegen;

TEST_CASE("Patching - Low Level Cache Line Alignment and Atomic APIs") {
    alignas(64) uint8_t buffer[128];
    std::memset(buffer, 0x90, sizeof(buffer));

    // Test 1: is_cache_line_safe helper
    void* safe_32 = buffer + 16;
    CHECK(is_cache_line_safe(safe_32, 4));

    void* straddle_32 = buffer + 62; // spans bytes 62, 63, 64, 65 -> crosses boundary 64
    CHECK(!is_cache_line_safe(straddle_32, 4));

    void* safe_64 = buffer + 40;
    CHECK(is_cache_line_safe(safe_64, 8));

    void* straddle_64 = buffer + 60; // spans bytes 60..67 -> crosses boundary 64
    CHECK(!is_cache_line_safe(straddle_64, 8));

    // Test 2: compute_cache_line_padding
    // If instruction at offset 60 has imm_offset=1 and imm_size=4: imm is at 61..64 (straddles!)
    size_t pad = compute_cache_line_padding(60, 1, 4);
    CHECK_EQ(pad, size_t(3)); // with pad=3, new_offset=63, imm is at 64..67 (safe!)
    CHECK(is_cache_line_safe(reinterpret_cast<void*>(60 + pad + 1), 4));

    // If instruction at offset 63 has imm_offset=1 and imm_size=4: imm is at 64..67 (safe!)
    size_t pad_zero = compute_cache_line_padding(63, 1, 4);
    CHECK_EQ(pad_zero, size_t(0));

    // Test 3: brass_patch_const32 atomic store
    int32_t val32 = 1234;
    CHECK(brass_patch_const32(&val32, 5678));
    CHECK_EQ(val32, 5678);

    // Test 4: brass_patch_const64 atomic store
    int64_t val64 = 0x1122334455667788LL;
    CHECK(brass_patch_const64(&val64, static_cast<int64_t>(0x99AABBCCDDEEFF00ULL)));
    CHECK_EQ(val64, static_cast<int64_t>(0x99AABBCCDDEEFF00ULL));
}

TEST_CASE("Patching - Single-Threaded Patchable Constants in JIT") {
    Module mod("patch_const_mod");

    // func @get_bias(%x: i32) -> i32
    //   %bias = patchable_const.i32 @bias_site, 10
    //   %res = add %x, %bias
    //   ret %res
    Function* fn32 = mod.create_function("get_bias32", Type::i32(), {Type::i32()});
    Builder b32(*fn32);
    BasicBlock* bb32 = b32.append_block("entry");
    b32.position_at_end(bb32);
    Value* x32 = b32.add_param(Type::i32());
    Value* bias32 = b32.build_patchable_const_i32("bias_site32", 10);
    Value* res32 = b32.build_add(x32, bias32);
    b32.build_ret(res32);

    // func @get_bias64(%x: i64) -> i64
    //   %bias = patchable_const.i64 @bias_site64, 1000
    //   %res = add %x, %bias
    //   ret %res
    Function* fn64 = mod.create_function("get_bias64", Type::i64(), {Type::i64()});
    Builder b64(*fn64);
    BasicBlock* bb64 = b64.append_block("entry");
    b64.position_at_end(bb64);
    Value* x64 = b64.add_param(Type::i64());
    Value* bias64 = b64.build_patchable_const_i64("bias_site64", 1000);
    Value* res64 = b64.build_add(x64, bias64);
    b64.build_ret(res64);

    JitExecutionEngine engine;
    REQUIRE(engine.compile_and_load(mod));

    // Check initial values
    CHECK_EQ(engine.invoke("get_bias32", {RuntimeValue::from_i32(5)}).as_i32(), 15);
    CHECK_EQ(engine.invoke("get_bias64", {RuntimeValue::from_i64(5)}).as_i64(), 1005);

    // Patch 32-bit constant to 50
    CHECK(engine.patch_const32("bias_site32", 50));
    CHECK_EQ(engine.invoke("get_bias32", {RuntimeValue::from_i32(5)}).as_i32(), 55);

    // Patch 64-bit constant to 9000
    CHECK(engine.patch_const64("bias_site64", 9000));
    CHECK_EQ(engine.invoke("get_bias64", {RuntimeValue::from_i64(5)}).as_i64(), 9005);
}

TEST_CASE("Patching - Single-Threaded Patchable Call Redirection in JIT") {
    Module mod("patch_call_mod");

    // Target stub A: returns x + 10
    Function* stub_a = mod.create_function("stub_add_10", Type::i64(), {Type::i64()});
    Builder ba(*stub_a);
    BasicBlock* bba = ba.append_block("entry");
    ba.position_at_end(bba);
    Value* xa = ba.add_param(Type::i64());
    ba.build_ret(ba.build_add(xa, ba.build_iconst_i64(10)));

    // Target stub B: returns x * 3
    Function* stub_b = mod.create_function("stub_mul_3", Type::i64(), {Type::i64()});
    Builder bb(*stub_b);
    BasicBlock* bbb = bb.append_block("entry");
    bb.position_at_end(bbb);
    Value* xb = bb.add_param(Type::i64());
    bb.build_ret(bb.build_mul(xb, bb.build_iconst_i64(3)));

    // Caller function with patchable_call:
    // func @dispatch_call(%x: i64) -> i64
    //   %res = patchable_call.i64 @ic_site_0, @stub_add_10(%x)
    //   ret %res
    Function* caller = mod.create_function("dispatch_call", Type::i64(), {Type::i64()});
    Builder bc(*caller);
    BasicBlock* bbc = bc.append_block("entry");
    bc.position_at_end(bbc);
    Value* xc = bc.add_param(Type::i64());
    Value* call_res = bc.build_patchable_call("ic_site_0", "stub_add_10", Type::i64(), {xc});
    bc.build_ret(call_res);

    JitExecutionEngine engine;
    REQUIRE(engine.compile_and_load(mod));

    // Initially points to stub_add_10: 7 + 10 = 17
    CHECK_EQ(engine.invoke("dispatch_call", {RuntimeValue::from_i64(7)}).as_i64(), 17);

    // Dynamically patch to stub_mul_3: 7 * 3 = 21
    CHECK(engine.patch_call("ic_site_0", "stub_mul_3"));
    CHECK_EQ(engine.invoke("dispatch_call", {RuntimeValue::from_i64(7)}).as_i64(), 21);

    // Patch back to stub_add_10: 7 + 10 = 17
    CHECK(engine.patch_call("ic_site_0", "stub_add_10"));
    CHECK_EQ(engine.invoke("dispatch_call", {RuntimeValue::from_i64(7)}).as_i64(), 17);
}

TEST_CASE("Patching - Cache Line Alignment Verification Across Various Offsets") {
    // Generate a module with multiple patchable constants and calls
    Module mod("alignment_test_mod");

    Function* dummy_callee = mod.create_function("dummy_target", Type::i64(), {Type::i64()});
    Builder bd(*dummy_callee);
    BasicBlock* bbd = bd.append_block("entry");
    bd.position_at_end(bbd);
    bd.build_ret(bd.add_param(Type::i64()));

    Function* test_fn = mod.create_function("alignment_fn", Type::i64(), {Type::i64()});
    Builder bt(*test_fn);
    BasicBlock* bbt = bt.append_block("entry");
    bt.position_at_end(bbt);
    Value* acc = bt.add_param(Type::i64());

    // Insert multiple patchable constants and calls interspersed with arithmetic to shift offsets
    for (int i = 0; i < 20; ++i) {
        std::string c32_name = "site_c32_" + std::to_string(i);
        std::string c64_name = "site_c64_" + std::to_string(i);
        std::string call_name = "site_call_" + std::to_string(i);

        Value* p32 = bt.build_patchable_const_i32(c32_name, i * 10);
        Value* p32_ext = bt.build_zext_i64(p32);
        acc = bt.build_add(acc, p32_ext);

        Value* p64 = bt.build_patchable_const_i64(c64_name, i * 100);
        acc = bt.build_add(acc, p64);

        acc = bt.build_patchable_call(call_name, "dummy_target", Type::i64(), {acc});
    }
    bt.build_ret(acc);

    JitExecutionEngine engine;
    REQUIRE(engine.compile_and_load(mod));

    // Verify all patch sites are cache-line safe
    void* fn_ptr = engine.get_symbol_address("alignment_fn");
    REQUIRE(fn_ptr != nullptr);

    for (const auto& site : engine.patch_sites().sites()) {
        uint8_t* inst_addr = static_cast<uint8_t*>(fn_ptr) + site.code_offset;
        uint8_t* imm_addr = inst_addr + site.imm_offset;
        size_t imm_size = (site.kind == PatchKind::Const64) ? 8 : 4;

        bool safe = is_cache_line_safe(imm_addr, imm_size);
        CHECK(safe);
    }
}

TEST_CASE("Patching - Multithreaded Concurrency Stress Test") {
    Module mod("concurrency_stress_mod");

    // Stub 1: adds 100
    Function* s1 = mod.create_function("stub1", Type::i64(), {Type::i64()});
    Builder b1(*s1);
    BasicBlock* bb1 = b1.append_block("entry");
    b1.position_at_end(bb1);
    b1.build_ret(b1.build_add(b1.add_param(Type::i64()), b1.build_iconst_i64(100)));

    // Stub 2: multiplies by 2
    Function* s2 = mod.create_function("stub2", Type::i64(), {Type::i64()});
    Builder b2(*s2);
    BasicBlock* bb2 = b2.append_block("entry");
    b2.position_at_end(bb2);
    b2.build_ret(b2.build_mul(b2.add_param(Type::i64()), b2.build_iconst_i64(2)));

    // Stub 3: adds 500
    Function* s3 = mod.create_function("stub3", Type::i64(), {Type::i64()});
    Builder b3(*s3);
    BasicBlock* bb3 = b3.append_block("entry");
    b3.position_at_end(bb3);
    b3.build_ret(b3.build_add(b3.add_param(Type::i64()), b3.build_iconst_i64(500)));

    // Worker function with patchable call and patchable constant:
    // func @concurrent_worker(%x: i64) -> i64
    //   %c = patchable_const.i64 @c_site, 1
    //   %v = add %x, %c
    //   %res = patchable_call.i64 @call_site, @stub1(%v)
    //   ret %res
    Function* worker_fn = mod.create_function("concurrent_worker", Type::i64(), {Type::i64()});
    Builder bw(*worker_fn);
    BasicBlock* bbw = bw.append_block("entry");
    bw.position_at_end(bbw);
    Value* xw = bw.add_param(Type::i64());
    Value* cw = bw.build_patchable_const_i64("c_site", 1);
    Value* vw = bw.build_add(xw, cw);
    Value* resw = bw.build_patchable_call("call_site", "stub1", Type::i64(), {vw});
    bw.build_ret(resw);

    JitExecutionEngine engine;
    REQUIRE(engine.compile_and_load(mod));

    auto fn_ptr = engine.get_function_ptr<int64_t(*)(int64_t)>("concurrent_worker");
    REQUIRE(fn_ptr != nullptr);

    std::atomic<bool> running{true};
    std::atomic<uint64_t> total_executions{0};
    std::atomic<uint64_t> total_patches{0};
    std::atomic<bool> test_passed{true};

    const int num_workers = 6;
    std::vector<std::thread> workers;

    // Worker threads: continuously invoke concurrent_worker(10)
    // Possible valid results:
    // c_site in {1, 2, 3}
    // stub in {stub1 (+100), stub2 (*2), stub3 (+500)}
    // Any returned result MUST be of the form stub(10 + c)
    for (int i = 0; i < num_workers; ++i) {
        workers.emplace_back([&, i]() {
            uint32_t iter = 0;
            while (running.load(std::memory_order_relaxed)) {
                int64_t input = 10;
                int64_t output = fn_ptr(input);
                total_executions.fetch_add(1, std::memory_order_relaxed);
                if ((++iter & 0x3FF) == 0) {
                    std::this_thread::yield();
                }

                // Verify output is one of the valid mathematical combinations (no torn reads)
                bool valid = false;
                for (int64_t c : {1LL, 2LL, 3LL}) {
                    int64_t v = input + c;
                    if (output == (v + 100) || output == (v * 2) || output == (v + 500)) {
                        valid = true;
                        break;
                    }
                }

                if (!valid) {
                    test_passed.store(false, std::memory_order_release);
                }
            }
        });
    }

    // Patcher thread: dynamically modifies call target and constant in a tight loop
    std::thread patcher([&]() {
        const char* stubs[] = {"stub1", "stub2", "stub3"};
        const int64_t consts[] = {1, 2, 3};
        size_t idx = 0;

        while (running.load(std::memory_order_relaxed)) {
            idx = (idx + 1) % 3;
            engine.patch_call("call_site", stubs[idx]);
            engine.patch_const64("c_site", consts[idx]);
            total_patches.fetch_add(1, std::memory_order_relaxed);
            std::this_thread::yield();
        }
    });

    // Run concurrency test until at least 5 patches completed or timeout (up to 200 ms)
    auto start_time = std::chrono::steady_clock::now();
    while (total_patches.load(std::memory_order_relaxed) < 5) {
        if (std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::steady_clock::now() - start_time).count() > 200) {
            break;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }
    running.store(false, std::memory_order_relaxed);

    patcher.join();
    for (auto& w : workers) {
        w.join();
    }

    CHECK(test_passed.load());
    CHECK(total_executions.load() > 1000);
    CHECK(total_patches.load() >= 1);

    std::cout << "  Concurrency stats: " << total_executions.load() << " worker executions, "
              << total_patches.load() << " dynamic patches completed safely.\n";
}
