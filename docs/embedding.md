# Brass Embedding API & Host Integration Guide

Brass provides a clean, zero-LLVM, thread-safe embedding surface for embedding dynamic language runtimes, AOT/JIT execution engines, and moving garbage collectors.

---

## 1. Overview & Architecture

```
+-------------------------------------------------------------------------------+
|                             Host Application                                  |
|  - C++ API: #include <brass/embedding/embedding.hpp>                          |
|  - C   API: #include <brass/embedding/brass_c_api.h>                         |
+-------------------------------------------------------------------------------+
       |                                             |
       v                                             v
+-------------------------------+             +-------------------------------+
|    HostEngine / Embedding     |             |      Host Cheney Moving GC    |
|  - Register host callbacks    |             |  - 2-Space Relocating Compact |
|  - In-memory MIR compilation  |             |  - Heap Poisoning (0xDEADBEEF)|
|  - Target ABI configuration   |             |  - Stack Walk Root Relocation |
+-------------------------------+             +-------------------------------+
       |                                             |
       v                                             v
+-------------------------------------------------------------------------------+
|                       CompiledModule / JIT Execution                          |
|  - Function entry pointers: `get_function_ptr<T>("foo")`                      |
|  - Stack maps query: `stack_maps()`                                           |
|  - Dynamic patching: `patch_constant(...)`, `patch_call(...)`                 |
|  - Deopt/Resume table points: `resume_tables()`                               |
|  - Stack walking helper: `walk_stack(...)`                                    |
+-------------------------------------------------------------------------------+
```

---

## 2. C++ Embedding API

### 2.1 `brass::HostEngine`

`HostEngine` is the top-level builder and compiler manager for host applications.

```cpp
#include <brass/brass.hpp>
#include <brass/embedding/embedding.hpp>

// Initialize engine
brass::HostEngine engine(brass::Target::host());

// Register a host C++ callback callable from JIT code
auto host_print = [](int64_t val) -> void {
    std::cout << "Host received: " << val << "\n";
};
engine.register_external_symbol("host_print", reinterpret_cast<void*>(+host_print));

// Optionally attach a HostGC for automatic safepoint and allocator binding
brass::HostGC gc(1024 * 1024);
engine.register_host_gc(&gc);

// Compile in-memory MIR Module into an executable CompiledModule
std::unique_ptr<brass::CompiledModule> compiled = engine.compile(my_module);
```

### 2.2 `brass::CompiledModule`

`CompiledModule` holds executable JIT code, stack maps, patch registries, and resume tables.

```cpp
// 1. Direct Typed Function Pointer Retrieval
using MyFunc = int64_t (*)(int64_t, int64_t);
MyFunc fn = compiled->get_function_ptr<MyFunc>("compute");
int64_t result = fn(10, 20);

// 2. Dynamic Invocation Helper
brass::RuntimeValue dyn_res = compiled->invoke("compute", {
    brass::RuntimeValue::from_i64(10),
    brass::RuntimeValue::from_i64(20)
});

// 3. Runtime Patching (Thread-Safe, Cache-Line Aligned)
compiled->patch_constant("bias_site", int64_t(42));
compiled->patch_call("ic_call_site", "optimized_stub_v2");

// 4. Stack Walking
compiled->walk_stack(top_rbp, top_ip, [](void** root_slot) {
    std::cout << "Live GC root found at: " << root_slot << " (val=" << *root_slot << ")\n";
});
```

---

## 3. C-Callable ABI Layer (`brass_c_api.h`)

Brass exposes an `extern "C"` ABI designed for FFI bindings (Rust, Zig, C, Go, Python):

```c
#include <brass/embedding/brass_c_api.h>

// Create engine and GC
brass_engine_t* engine = brass_engine_create();
brass_gc_t* gc = brass_host_gc_create(256 * 1024);
brass_engine_register_gc(engine, gc);

// Register external host symbol
brass_engine_register_symbol(engine, "my_host_fn", (void*)&my_host_fn);

// Compile module
brass_module_t* compiled = brass_compile_module(engine, mir_module_ptr);

// Retrieve function pointer
typedef int64_t (*compute_fn_t)(int64_t);
compute_fn_t fn = (compute_fn_t)brass_module_get_function_ptr(compiled, "compute");
int64_t res = fn(42);

// Runtime patching via C API
brass_module_patch_const64(compiled, "bias_site", 100);
brass_module_patch_call(compiled, "call_site", (const void*)&new_stub);

// Cleanup
brass_module_destroy(compiled);
brass_host_gc_destroy(gc);
brass_engine_destroy(engine);
```

---

## 4. NaN-Boxed 64-Bit Value System

`brass::HostValue` implements an IEEE 754 NaN-boxing encoding:
- All canonical double-precision floating point values are represented directly (raw unsigned 64-bit value `< TAG_BASE = 0x7FF8000000000000ULL`).
- Non-floating values are encoded inside the quiet NaN payload space with a 3-bit tag:

| Value Type | Tag Bits (48..50) | Lower 32/48-bit Payload |
|---|---|---|
| **Double (f64)** | `000` (non-NaN) | Standard IEEE 754 float bits |
| **Int32** | `001` (`TAG_INT32`) | 32-bit signed integer in bits 0..31 |
| **Bool** | `010` (`TAG_BOOL`) | `1` for true, `0` for false in bit 0 |
| **Null** | `011` (`TAG_NULL`) | `0` |
| **Undefined** | `100` (`TAG_UNDEFINED`) | `0` |
| **GCRef / Object** | `101` (`TAG_GCREF`) | 48-bit heap payload pointer |
| **Raw Pointer** | `110` (`TAG_POINTER`) | 48-bit virtual address |

```cpp
#include <brass/embedding/nanbox.hpp>

brass::HostValue v_num = brass::HostValue::from_f64(3.14159);
brass::HostValue v_int = brass::HostValue::from_i32(42);
brass::HostValue v_obj = brass::HostValue::from_gcref(heap_addr);

if (v_int.is_i32()) {
    int32_t val = v_int.as_i32();
}
```

---

## 5. Moving Cheney GC Contract & Stack Map Coordination

The `HostGC` is a Cheney moving semispace garbage collector designed to coordinate seamlessly with Brass stack maps:

1. **Heap Poisoning**:
   - Whenever semispaces swap, the old space is immediately filled with `0xDEADBEEFDEADBEEFULL` to catch dangling pointers instantly.
2. **Stack Frame Traversal**:
   - During allocation or safepoint, `brass_stack_walk` unrolls native stack frames using RBP chains and Brass `ModuleStackMap` records.
3. **In-Place Root Relocation**:
   - Live `gcref` spill slots and `HostValue` NaN-boxed object references are automatically evacuated into To-Space and rewritten in-place.
4. **Stress Mode**:
   - Setting `gc.set_stress_mode(true)` forces a collection on every allocation and safepoint to catch root retention bugs.

---

## 6. Runtime Patching Protocol

Brass guarantees atomic, cache-line-safe runtime patching on x64 without stopping mutator threads:

- **`patch_constant(name, val)`**: Writes a new 32-bit or 64-bit immediate with a cache-line aligned atomic store.
- **`patch_call(name, target)`**: Atomically updates the 32-bit relative displacement of direct `call` instructions (`0xE8 <disp32>`).
