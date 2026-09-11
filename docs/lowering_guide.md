# Consumer's Guide to Lowering (Bronze -> Brass)

This document guides frontend language compilers (such as `bronze`, the JS AOT compiler) on mapping high-level typed SSA IR to Brass MIR and using the Brass C++ API.

---

## 1. Division of Responsibility

| Task | Owned By | Notes |
| :--- | :--- | :--- |
| Inlining, Escape Analysis | Consumer (`bronze`) | High-level semantic transformations |
| Unboxing, Type Specialization | Consumer (`bronze`) | Type inference & speculative specialization |
| Guard Placement & Check Elimination | Consumer (`bronze`) | Redundant guard removal |
| Instruction Selection | Brass | Pattern matching & x64 addressing modes |
| Register Allocation | Brass | Linear-scan with live-range splitting |
| LIR Peephole Optimization | Brass | Redundant moves, dead stores, zeroing, branch fold |
| Stack Map Generation | Brass | Live `gcref` tracking at call safepoints |
| Frame Layout & Unwind Emission | Brass | Win64 SEH (`.pdata`/`.xdata`), Linux SysV CFI (`.eh_frame`), macOS Compact Unwind |
| Object File Emission | Brass | Deterministic COFF (`.obj`), ELF64 (`.o`), and Mach-O (`.o`) |

---

## 2. Mapping Frontend IL to Brass MIR

1. **Values & Types**:
   - Boxed JS Objects / Managed References -> `Type::gcref()`
   - Unboxed 32-bit Integers / Booleans -> `Type::i32()`
   - Unboxed 64-bit Integers / Raw Addresses -> `Type::i64()` / `Type::ptr()`
   - IEEE 754 Doubles -> `Type::f64()`
2. **Phi Nodes -> Block Arguments**:
   - Transform SSA Phi nodes into block parameters during lowering.
3. **Calls & Inline Caches**:
   - Monomorphic calls -> `b.build_call("func_name", return_type, args)`
   - Inline cache sites -> `b.build_patchable_call("ic_site_1", "slow_stub", return_type, args)`
4. **Guards & Deoptimization**:
   - Lower speculative checks into `b.build_guard(cond, "deopt_stub_1", live_state_values)`.
   - Implement interior resume targets in generic twin functions with `b.build_resume_point(resume_id)`.

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
