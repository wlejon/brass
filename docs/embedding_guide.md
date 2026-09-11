# Brass Embedding Guide (libbrass SDK & Public C-ABI)

The **Brass Compiler Backend** provides a high-performance, standalone, zero-dependency, and zero-LLVM code-generation infrastructure for garbage-collected dynamic languages, JIT runtimes, and ahead-of-time (AOT) toolchains.

The public C-ABI (`<brass/brass_c_api.h>`) and embedder SDK (`libbrass`) allow C, C++, Rust, Zig, Go, Python, and other language runtimes to easily embed Brass for:
- In-memory JIT code generation and dynamic function pointer execution.
- SSA IR construction with basic blocks, instructions, and block parameters.
- External host symbol registration and interop.
- Bronze textual IL translation and execution.
- Relocatable object file emission in memory (`COFF`, `ELF64`, `Mach-O`).
- Standalone shared library linking (`.dll`, `.so`, `.dylib`) without external linkers.

---

## 1. Quick Start (Hello World)

To embed Brass in pure C99/C11:

```c
#include <brass/brass_c_api.h>
#include <stdio.h>

int main(void) {
    // 1. Create compiler context
    BrassContext ctx = brass_context_create();

    // 2. Create module and function: add(a: i64, b: i64) -> i64
    BrassModule mod = brass_module_create(ctx, "hello_module");
    BrassType i64_t = brass_type_i64();
    BrassType param_types[2] = { i64_t, i64_t };
    BrassFunction fn = brass_function_create(mod, "add", i64_t, param_types, 2);

    // 3. Build function body
    BrassBlock entry = brass_function_append_block(fn, "entry");
    BrassValue a = brass_block_add_param(entry, i64_t);
    BrassValue b = brass_block_add_param(entry, i64_t);

    BrassBuilder builder = brass_builder_create(ctx, fn);
    brass_builder_position_at_end(builder, entry);
    BrassValue sum = brass_build_add(builder, a, b);
    brass_build_ret(builder, sum);

    // 4. Verify module
    char err[256];
    if (brass_module_verify(mod, err, sizeof(err)) != BRASS_OK) {
        fprintf(stderr, "Verification error: %s\n", err);
        return 1;
    }

    // 5. JIT compile and execute
    BrassJitEngine jit = brass_jit_create(ctx);
    brass_jit_compile_module(jit, mod);

    typedef int64_t (*AddFn)(int64_t, int64_t);
    AddFn add_fn = (AddFn)brass_jit_get_function_address(jit, "add");
    printf("Result: 40 + 2 = %lld\n", (long long)add_fn(40, 2));

    // 6. Clean up
    brass_builder_destroy(builder);
    brass_jit_destroy(jit);
    brass_module_destroy(mod);
    brass_context_destroy(ctx);
    return 0;
}
```

---

## 2. In-Memory JIT Compilation: Loops & Block Parameters

Brass uses canonical SSA block parameters rather than phi nodes. Values are forwarded to successor blocks via branch arguments.

```c
#include <brass/brass_c_api.h>
#include <stdio.h>

// Builds loop: computes sum_{i=1..n} (i * 3 + 1)
void build_and_run_loop(void) {
    BrassContext ctx = brass_context_create();
    BrassModule mod = brass_module_create(ctx, "loop_kernel_mod");

    BrassType i64_t = brass_type_i64();
    BrassFunction fn = brass_function_create(mod, "loop_kernel", i64_t, &i64_t, 1);

    BrassBlock entry = brass_function_append_block(fn, "entry");
    BrassValue n = brass_block_add_param(entry, i64_t);

    BrassBlock check_bb = brass_function_append_block(fn, "check");
    BrassValue i_param = brass_block_add_param(check_bb, i64_t);
    BrassValue acc_param = brass_block_add_param(check_bb, i64_t);

    BrassBlock body_bb = brass_function_append_block(fn, "body");
    BrassBlock exit_bb = brass_function_append_block(fn, "exit");
    BrassValue res_param = brass_block_add_param(exit_bb, i64_t);

    BrassBuilder b = brass_builder_create(ctx, fn);

    // entry: jump to check with i=0, acc=0
    brass_builder_position_at_end(b, entry);
    BrassValue zero = brass_build_iconst_i64(b, 0);
    BrassValue init_args[2] = { zero, zero };
    brass_build_br(b, check_bb, init_args, 2);

    // check: if (i < n) goto body; else goto exit(acc);
    brass_builder_position_at_end(b, check_bb);
    BrassValue cond = brass_build_cmp(b, BRASS_CMP_SLT, i_param, n);
    BrassValue exit_args[1] = { acc_param };
    brass_build_br_if(b, cond, body_bb, NULL, 0, exit_bb, exit_args, 1);

    // body: next_acc = acc + (i * 3 + 1); next_i = i + 1; goto check(next_i, next_acc);
    brass_builder_position_at_end(b, body_bb);
    BrassValue c3 = brass_build_iconst_i64(b, 3);
    BrassValue c1 = brass_build_iconst_i64(b, 1);
    BrassValue term = brass_build_add(b, brass_build_mul(b, i_param, c3), c1);
    BrassValue next_acc = brass_build_add(b, acc_param, term);
    BrassValue next_i = brass_build_add(b, i_param, c1);
    BrassValue loop_args[2] = { next_i, next_acc };
    brass_build_br(b, check_bb, loop_args, 2);

    // exit: ret res_param
    brass_builder_position_at_end(b, exit_bb);
    brass_build_ret(b, res_param);

    brass_builder_destroy(b);

    // JIT compile and invoke
    BrassJitEngine jit = brass_jit_create(ctx);
    brass_jit_compile_module(jit, mod);

    typedef int64_t (*LoopFn)(int64_t);
    LoopFn kernel = (LoopFn)brass_jit_get_function_address(jit, "loop_kernel");

    printf("Kernel(10) = %lld\n", (long long)kernel(10));

    brass_jit_destroy(jit);
    brass_module_destroy(mod);
    brass_context_destroy(ctx);
}
```

---

## 3. External Host Symbol Registration

Host applications can register C/C++ callbacks and runtime helper functions so JIT-compiled code can invoke them via `brass_build_call`.

```c
#include <brass/brass_c_api.h>
#include <stdio.h>

// Host runtime callback
static int64_t host_log_number(int64_t x) {
    printf("[Host Log] Processed value: %lld\n", (long long)x);
    return x * 2;
}

void run_with_host_symbol(void) {
    BrassContext ctx = brass_context_create();
    BrassModule mod = brass_module_create(ctx, "extern_test_mod");

    // Declare symbol in module
    brass_module_add_external_symbol(mod, "host_log_number");

    BrassType i64_t = brass_type_i64();
    BrassFunction fn = brass_function_create(mod, "run_log", i64_t, &i64_t, 1);
    BrassBlock entry = brass_function_append_block(fn, "entry");
    BrassValue in_val = brass_block_add_param(entry, i64_t);

    BrassBuilder b = brass_builder_create(ctx, fn);
    brass_builder_position_at_end(b, entry);

    BrassValue call_args[1] = { in_val };
    BrassValue call_res = brass_build_call(b, "host_log_number", i64_t, call_args, 1);
    brass_build_ret(b, call_res);
    brass_builder_destroy(b);

    // JIT Engine & symbol binding
    BrassJitEngine jit = brass_jit_create(ctx);
    brass_jit_register_symbol(jit, "host_log_number", (void*)&host_log_number);
    brass_jit_compile_module(jit, mod);

    typedef int64_t (*LogFn)(int64_t);
    LogFn run_log = (LogFn)brass_jit_get_function_address(jit, "run_log");
    int64_t out = run_log(21);
    printf("Result returned from JIT: %lld\n", (long long)out); // 42

    brass_jit_destroy(jit);
    brass_module_destroy(mod);
    brass_context_destroy(ctx);
}
```

---

## 4. Bronze IL Translation Bridge

Embedders targeting high-level dynamic languages can pass textual Bronze IL directly to `brass_translate_bronze_il` for automated lowering, SSA construction, and optimization.

```c
#include <brass/brass_c_api.h>
#include <stdio.h>
#include <string.h>

void compile_bronze_il_snippet(void) {
    BrassContext ctx = brass_context_create();

    const char* il_text =
        "module math_ops.js\n"
        "func hypot_sq(%0: f64, %1: f64) -> f64 {\n"
        "  b0:\n"
        "    %2: f64 = mul %0, %0\n"
        "    %3: f64 = mul %1, %1\n"
        "    %4: f64 = add %2, %3\n"
        "    ret %4\n"
        "}\n";

    BrassOptions* opts = brass_options_create();
    brass_options_set_optimize(opts, 1);

    BrassModule mod = NULL;
    BrassStatus s = brass_translate_bronze_il(ctx, il_text, strlen(il_text), opts, &mod);
    if (s != BRASS_OK) {
        fprintf(stderr, "Translation failed: %s\n", brass_context_get_last_error(ctx));
        brass_options_destroy(opts);
        brass_context_destroy(ctx);
        return;
    }
    brass_options_destroy(opts);

    // Compile and execute
    BrassJitEngine jit = brass_jit_create(ctx);
    brass_jit_compile_module(jit, mod);

    typedef double (*HypotFn)(double, double);
    HypotFn fn = (HypotFn)brass_jit_get_function_address(jit, "hypot_sq");
    printf("hypot_sq(3.0, 4.0) = %f\n", fn(3.0, 4.0)); // 25.0

    brass_jit_destroy(jit);
    brass_module_destroy(mod);
    brass_context_destroy(ctx);
}
```

---

## 5. In-Memory AOT Object File Emission

Brass can compile modules ahead-of-time directly into memory buffers without writing temporary files to disk.

```c
#include <brass/brass_c_api.h>
#include <stdio.h>

void emit_relocatable_object(BrassModule mod) {
    void* object_bytes = NULL;
    size_t object_size = 0;

    // Target formats:
    //   BRASS_OBJECT_AUTO  = 0 (Host platform)
    //   BRASS_OBJECT_COFF  = 1 (Windows x64 COFF)
    //   BRASS_OBJECT_ELF   = 2 (Linux x64 ELF)
    //   BRASS_OBJECT_MACHO = 3 (macOS x64 Mach-O)
    BrassStatus s = brass_compile_to_object(mod, BRASS_OBJECT_ELF, &object_bytes, &object_size);
    if (s == BRASS_OK && object_bytes != NULL) {
        printf("Emitted %zu relocatable ELF bytes in memory.\n", object_size);

        // Process or persist buffer...

        // Free memory buffer via Brass allocator
        brass_free_buffer(object_bytes);
    }
}
```

---

## 6. Standalone AOT Shared Library Linking

Brass embeds standalone PE DLL, ELF `.so`, and Mach-O `.dylib` linkers that link emitted objects directly into shared libraries without requiring `link.exe`, `ld`, or `lld`.

```c
#include <brass/brass_c_api.h>
#include <stdio.h>

void compile_to_dll(BrassModule mod, const char* out_path) {
    BrassOptions* opts = brass_options_create();
    brass_options_set_optimize(opts, 1);

    BrassStatus s = brass_compile_to_shared_lib(mod, out_path, opts);
    if (s != BRASS_OK) {
        fprintf(stderr, "Failed to link shared library\n");
    } else {
        printf("Generated standalone dynamic library: %s\n", out_path);
    }

    brass_options_destroy(opts);
}
```

---

## 7. Error Handling Model

Every C-ABI boundary function is guarded against null pointers, invalid arguments, and C++ exceptions:
1. Functions returning `BrassStatus` return `BRASS_OK` (0) on success, or negative `BRASS_ERR_*` codes on failure.
2. Context-bound functions record human-readable error diagnostics into `BrassContext`.
3. Query the latest error message using `brass_context_get_last_error(ctx)`.
4. Inject custom diagnostics using `brass_context_set_error(ctx, msg)`.

---

## 8. CMake Integration

To integrate `libbrass` into a CMake project:

```cmake
# Using installed Brass package
find_package(brass REQUIRED)
target_link_libraries(my_host PRIVATE brass_shared)

# Or when Brass is a submodule/subdirectory:
add_subdirectory(brass)
target_link_libraries(my_host PRIVATE brass_shared)
target_include_directories(my_host PRIVATE ${BRASS_INCLUDE_DIR})
```
