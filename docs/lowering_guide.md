# Consumer's Guide to Lowering (Bronze -> Brass)

This document guides frontend language compilers (such as `bronze`, the JS AOT compiler) on mapping high-level typed SSA IR to Brass MIR and using the Brass C++ and C APIs.

---

## 1. Division of Responsibility & Available Capabilities

| Task | Owned By | Notes |
| :--- | :--- | :--- |
| Lexical Analysis, Parsing & AST Lowering | Consumer (`bronze`) | Frontend language syntax & semantics |
| Scope & Environment Frame Allocation | Consumer / Runtime | Lexical environment chain lowering (`BronzeEnv`, `BronzeClosure`) |
| Unboxing & Type Specialization | Consumer (`bronze`) | Speculative unboxing and type tag checks |
| Function Inlining & Speculative Devirtualization | Brass (or Consumer) | Static IPO inliner and Type Feedback Vector (TFV) speculative devirtualizer |
| Escape Analysis & Allocation Sinking | Brass | Hot-path scalarization and cold-path allocation sinking (PEA) |
| Redundant Guard & Bounds Check Elimination | Brass | SCCP guard pruning and interval value range analysis (BCE) |
| SROA, GVN, PRE & Loop Optimizations | Brass | Scalar replacement, memory SSA forwarding, loop tiling, fusion, unrolling |
| Auto-Parallelization & Vectorization | Brass | Polyhedral dependence analysis, SIMD loop unrolling (SSE/AVX2), and SLP |
| Instruction Selection & Register Allocation | Brass | x64 pattern matching, linear-scan with live-range splitting |
| LIR Peephole & Machine Scheduling | Brass | Post-regalloc peephole optimizations, DAG critical-path instruction scheduling |
| Stack Map Generation & Moving GC Coordination | Brass | Out-of-band compact binary stack maps for live `gcref` registers and spill slots |
| Frame Layout & Unwind Metadata Emission | Brass | Zero-cost Win64 SEH (`.pdata`/`.xdata`), Linux SysV CFI (`.eh_frame`), macOS Compact Unwind |
| Object File & Shared Library Emission | Brass | Deterministic COFF (`.obj`), ELF64 (`.o`), Mach-O (`.o`), PE DLL, and ELF `.so` |

---

## 2. Mapping Frontend IL to Brass MIR

1. **Values & Types**:
   - Boxed JS Objects / Managed References -> `Type::gcref()`
   - Unboxed 32-bit Integers / Booleans -> `Type::i32()`
   - Unboxed 64-bit Integers / Raw Addresses -> `Type::i64()` / `Type::ptr()`
   - IEEE-754 Floats -> `Type::f32()`, `Type::f64()`
   - Fixed-width SIMD vectors -> `Type::f32x4()`, `Type::f64x2()`, `Type::f32x8()`, `Type::f64x4()`, `Type::i32x4()`, `Type::i64x2()`, etc.
2. **Phi Nodes -> Block Arguments**:
   - Transform SSA Phi nodes into canonical basic block parameters during lowering.
3. **Calls & Inline Caches**:
   - Direct static calls -> `b.build_call("func_name", return_type, args)`
   - Indirect pointer calls -> `b.build_call_indirect(callee_ptr, return_type, args)`
   - Inline cache sites -> `b.build_patchable_call("ic_site_1", "slow_stub", return_type, args)`
4. **Exception Handling & Unwinding**:
   - Throwing expressions -> `b.build_throw(exception_val)`
   - Potentially throwing calls -> `b.build_invoke("func_name", return_type, args, normal_block, normal_args, unwind_block, unwind_args)`
   - Catch block entry -> `b.build_landing_pad(Type::gcref())`
   - Re-throwing / finally continuation -> `b.build_resume(exception_val)`
5. **Coroutines & Generators**:
   - Create coroutine frame -> `b.build_coro_create("fn_symbol", args)`
   - Yield value -> `b.build_coro_suspend(yield_val, resume_id)`
   - Resume coroutine -> `b.build_coro_resume(coro_ptr, arg_val)`
   - Destroy coroutine -> `b.build_coro_destroy(coro_ptr)`
6. **Speculation, Guards & Deoptimization**:
   - Lower speculative checks into `b.build_guard(cond, "generic_twin_exit", live_state_values)`.
   - Implement interior resume targets in generic twin functions with `twin->add_resume_point(resume_id, target_block)`.
7. **Generational GC Write Barriers**:
   - When writing a managed pointer into an object field, emit `b.build_write_barrier(object_ref, value_ref)` to mark the generational card table.

---

## 3. End-to-End API Example

```cpp
#include <brass/brass.hpp>

using namespace brass;

// 1. Construct Module and Function
Module mod("my_module");
Builder b(mod);

Function* fn = mod.create_function("compute", Type::i64(), {Type::i64(), Type::i64()});
b.set_function(fn);

BasicBlock* entry = b.append_block("entry");
Value* x = b.add_block_param(entry, Type::i64());
Value* y = b.add_block_param(entry, Type::i64());

Value* sum = b.build_add(x, y);
b.build_ret(sum);

fn->rebuild_cfg_predecessors();
assert(verify_module(mod));

// 2. Compile to Relocatable Object File (.obj / .o)
object::ObjectFile obj = object::compile_module_to_object(mod, Target::host());
std::vector<uint8_t> binary_bytes;
if (Target::host().is_windows()) {
    binary_bytes = object::emit_coff_object(obj);
} else if (Target::host().is_macos()) {
    binary_bytes = object::emit_macho_object(obj);
} else {
    binary_bytes = object::emit_elf_object(obj);
}

// 3. Or JIT Execute in Memory
codegen::JitExecutionEngine jit;
jit.compile_and_load(mod);
auto compute_fn = jit.get_function_ptr<int64_t(*)(int64_t, int64_t)>("compute");
int64_t result = compute_fn(10, 32); // returns 42
```
