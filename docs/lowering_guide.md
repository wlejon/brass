# Consumer's Guide to Lowering (Bronze -> Brass)

This document guides frontend compilers (such as `bronze`, the JS AOT compiler) on mapping high-level typed SSA IR to Brass MIR.

---

## 1. Division of Responsibility

| Task | Owned By | Notes |
| :--- | :--- | :--- |
| Inlining, Escape Analysis | Consumer (`bronze`) | Semantic high-level optimizations |
| Unboxing, Type Specialization | Consumer (`bronze`) | Type inference & specialization |
| Guard Placement & Check Elimination | Consumer (`bronze`) | Redundant guard removal |
| Instruction Selection | Brass | Machine pattern matching & addressing modes |
| Register Allocation | Brass | Linear-scan with live-range splitting |
| Stack Map Generation | Brass | Live `gcref` tracking at call safepoints |
| Frame Layout & Unwind Emission | Brass | Win64 SEH `.pdata`/`.xdata`, SysV CFI |
| Object File Emission | Brass | COFF / ELF64 relocatable objects |

---

## 2. Mapping bronze IL to Brass MIR

1. **Values & Types**:
   - Boxed JS Values (`Value` / `HeapObject*`) -> `gcref`
   - Unboxed 32-bit ints -> `i32`
   - Unboxed 64-bit ints / pointers -> `i64` / `ptr`
   - Unboxed floats -> `f64`
2. **Phi Nodes -> Block Arguments**:
   - Transform SSA Phi nodes into block parameters during lowering.
3. **Calls & Inline Caches**:
   - Lower direct calls to `call @target(...)`
   - Lower IC call sites to `patchable_call @ic_name, @slow_path(...)`
4. **Guards & Deoptimization**:
   - Lower type checks into `guard %is_type, @deopt_stub, [%live_vars...]`
