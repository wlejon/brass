# Brass MIR Reference Specification

Brass Machine Intermediate Representation (MIR) is a typed SSA intermediate representation designed specifically for machine-level code generation with first-class support for precise moving garbage collection, speculation, inline caching, vectorization, coroutines, and zero-cost exception handling.

---

## 1. Type System

Brass MIR provides scalar, pointer, reference, and fixed-width SIMD vector types:

### Scalar and Pointer Types

| Type | Size (Bytes) | Register Class | Description |
| :--- | :--- | :--- | :--- |
| `i32` | 4 | GPR | 32-bit two's-complement integer |
| `i64` | 8 | GPR | 64-bit two's-complement integer |
| `f32` | 4 | XMM / FPR | 32-bit IEEE-754 single-precision floating point |
| `f64` | 8 | XMM / FPR | 64-bit IEEE-754 double-precision floating point |
| `ptr` | 8 | GPR | Raw unmanaged machine pointer (not traced by GC) |
| `gcref` | 8 | GPR | Managed object reference (participates in regalloc, recorded in stack maps) |
| `void` | 0 | None | Void return type for functions |

### 128-Bit SIMD Vector Types (`v128`)

| Type | Size (Bytes) | Register Class | Description |
| :--- | :--- | :--- | :--- |
| `f32x4` | 16 | XMM | Vector of 4 single-precision floats |
| `f64x2` | 16 | XMM | Vector of 2 double-precision floats |
| `i32x4` | 16 | XMM | Vector of 4 32-bit integers |
| `i64x2` | 16 | XMM | Vector of 2 64-bit integers |

### 256-Bit SIMD Vector Types (`v256`, AVX2)

| Type | Size (Bytes) | Register Class | Description |
| :--- | :--- | :--- | :--- |
| `f32x8` | 32 | YMM | Vector of 8 single-precision floats |
| `f64x4` | 32 | YMM | Vector of 4 double-precision floats |
| `i32x8` | 32 | YMM | Vector of 8 32-bit integers |
| `i64x4` | 32 | YMM | Vector of 4 64-bit integers |

---

## 2. SSA & Control Flow

- **Block Structured CFG**: A function body consists of one or more basic blocks (`bb0`, `bb1`, etc.).
- **Block Parameters**: Block arguments replace Phi nodes. Every branch to a target block passes values matching that block's parameter types.
- **Strict Dominance**: Values are defined once and must dominate all uses. In the text, a use may appear before its definition when the defining block comes later in the file (the parser resolves such forward references after reading the whole function); a use before its definition within one block, or a name no block defines, is a parse error. Dominance itself is checked by the verifier.
- **Explicit Terminators**: Every basic block must end in a single terminator instruction (`br`, `br_if`, `switch`, `ret`, `unreachable`, `throw`, `invoke`, `resume`).
- **Exception Unwinding**: Call sites that may throw use `invoke`, specifying both a normal successor block and an unwind landing pad block.

---

## 3. Instruction Set Summary

### Constants & Conversions

Integer literals must fit the field they fill, or the parser reports "out of range": `iconst.i32` and 32-bit offsets take [-2^31, 2^31-1], `iconst.i64` takes [-2^63, 2^63-1], and a `switch` case value must fit the condition's type (the verifier checks the same, and rejects duplicate case values). Decimal literals are written signed, as the printer writes them (`iconst.i32 4294967295` is an error; write `-1`); for `iconst`, `patchable_const` and `switch` cases a hex literal may spell the value's bit pattern (`iconst.i32 0xFFFFFFFF` is -1).
- `iconst.i32 <imm32>` -> `i32`
- `iconst.i64 <imm64>` -> `i64`
- `fconst.f64 <imm64>` -> `f64`
- `patchable_const.i32 @sym, <imm32>` -> `i32`
- `patchable_const.i64 @sym, <imm64>` -> `i64`
- `sext.i64 <val:i32>` -> `i64`
- `zext.i64 <val:i32>` -> `i64`
- `trunc.i32 <val:i64>` -> `i32`
- `fptosi.i32 <val:f64>` -> `i32`
- `fptosi.i64 <val:f64>` -> `i64`
- `sitofp.f64 <val:i32|i64>` -> `f64` (the bare form takes its source width from the operand; `sitofp.f64.i32` / `sitofp.f64.i64` name it explicitly; `sitofp.f32` likewise)
- `bitcast.i64 <val:f64>` -> `i64` (also spelled `bitcast.i64.f64`, as the printer writes it)
- `bitcast.f64 <val:i64>` -> `f64` (also spelled `bitcast.f64.i64`, as the printer writes it)

### Arithmetic, Logic & Bitwise
- `add.<type> <lhs>, <rhs>`, `sub.<type> <lhs>, <rhs>`, `mul.<type> <lhs>, <rhs>` (`i32`, `i64`, `f64`)
- `fma.f32 <a:f32>, <b:f32>, <c:f32>` -> `f32` ($a \times b + c$)
- `fma.f64 <a:f64>, <b:f64>, <c:f64>` -> `f64` ($a \times b + c$)
- `sdiv.<type> <lhs>, <rhs>`, `udiv.<type> <lhs>, <rhs>`, `smod.<type> <lhs>, <rhs>`, `umod.<type> <lhs>, <rhs>` (`i32`, `i64`)
- `sdiv.f64 <lhs>, <rhs>`, `sdiv.f32 <lhs>, <rhs>`: floating-point division (there is no separate `fdiv`)
- `neg.<type> <val>` (`i32`, `i64`, `f64`)
- `and.<type> <lhs>, <rhs>`, `or.<type> <lhs>, <rhs>`, `xor.<type> <lhs>, <rhs>` (`i32`, `i64`)
- `shl.<type> <lhs>, <rhs>`, `lshr.<type> <lhs>, <rhs>`, `ashr.<type> <lhs>, <rhs>`, `not.<type> <val>` (`i32`, `i64`)
- `clz.<type> <val>`, `ctz.<type> <val>`, `popcnt.<type> <val>` (`i32`, `i64`)
- `select.<type> <cond:i32>, <true_val>, <false_val>` (`i32`, `i64`, `f32`, `f64`, `ptr`, `gcref`)
- `sadd_overflow.<type> <lhs>, <rhs>`, `ssub_overflow.<type>`, `smul_overflow.<type>` (`i32`, `i64` -> `i32` overflow flag)
- `uadd_overflow.<type> <lhs>, <rhs>`, `usub_overflow.<type>`, `umul_overflow.<type>` (`i32`, `i64` -> `i32` overflow flag)

### Comparisons
- `eq.<type>`, `ne.<type>`, `slt.<type>`, `ult.<type>`, `sle.<type>`, `ule.<type>`, `sgt.<type>`, `ugt.<type>`, `sge.<type>`, `uge.<type>` (`i32`, `i64`, `f64` -> `i32` condition)

### Memory Operations & GC Barriers
- `load.<type> <base:ptr|gcref> [, <offset:i32>]` -> `<type>`
- `store.<type> <base:ptr|gcref> [, <offset:i32>], <val>`
- `load_indexed.<type> <base>, <index:i64>, <scale:1|2|4|8> [, <offset:i32>]` -> `<type>`
- `store_indexed.<type> <base>, <index:i64>, <scale:1|2|4|8> [, <offset:i32>], <val>`
- `write_barrier <obj:gcref>, <val:gcref>` (marks generational card table for old-to-young references)

### Function Calls, Addresses & Safepoints
- `call.<type> @fn_symbol(<args...>)` -> `<type>`
- `call_indirect.<type> <callee_ptr:ptr>(<args...>)` -> `<type>`
- `patchable_call.<type> @patch_sym, @default_target(<args...>)` -> `<type>`
- `func_addr @fn_symbol` -> `ptr`
- `safepoint`

### Speculation, Deoptimization & OSR
- `guard <cond:i32>, @exit_stub [, [<val1>, <val2>, ...]]`
- `resume_point <resume_id:i32>`
- `osr_entry <imm64> [[<val1>, <val2>, ...]]`

#### Guard exits

When a guard's condition is 0, the guard exits with its state values `[v0, ..., vN-1]`. A guard can exit in two ways:

- **Exit stub.** The guard's `@label` names a function of the same module. That function is called as `stub(v0, ..., vN-1)`, and its result becomes the guarded function's result. The verifier requires the stub to have exactly N parameters, where parameter i has the type of vi, and to return the guarded function's return type. A label that names no function (for example `@exit_stub`) is only a label, and the guard then has no exit stub.
- **Resume target.** The function's `resume_table` maps the guard's resume id to a block. Execution continues in that block, in the same function, and block parameter i takes vi. The verifier requires the block to have at most N parameters, with matching types.

A guard with an exit stub always takes the stub, even when it also has a resume target. A guard with neither can only exit through a deopt handler. The tiers behave as follows:

| Tier | Exit stub | Resume target only | Neither |
|---|---|---|---|
| Interpreter | calls `stub(v...)` | branches to the block | the installed deopt handler handles it; with no handler, `DeoptException` |
| Baseline JIT (x64) | calls `stub(v...)` | branches to the block | compile error |
| Tier 2, with a pipeline resumer or deopt handler (OSR) | the lower tier finishes the call and takes the same exits as the interpreter, in the same order | the same | the resumer or handler handles it; with neither, a fatal error |
| Tier 2, standalone (no resumer and no handler) | native call `stub(v...)`; the stub's return registers are the function's | fatal error: nothing can resume the frame | fatal error |

Tier 2 rejects a guard in a function that returns a vector, because a lower tier returns its result as one 64-bit word. On AArch64, tier 2 passes stub arguments only in registers (x0-x7 and d0-d7), and a stub that needs stack arguments is a compile error. The AArch64 baseline JIT does not implement guard exits: a guard that fails there traps (`brk`).

### Terminators
- `br <target_block>(<args...>)`
- `br_if <cond:i32>, <true_block>(<true_args...>), <false_block>(<false_args...>)`
- `switch.<type> <val>, default: <def_block>(<def_args...>), [<val_1>: <target_1>(<args_1...>), ...]`
- `ret [<val>]`
- `unreachable`
- `throw <val>`
- `invoke.<type> @fn_symbol(<args...>), <normal_block>(<normal_args...>), <unwind_block>(<unwind_args...>)`
- `resume [<val>]`

### Exception Handling (Non-Terminator)
- `landing_pad [.<type>]` (retrieves active exception reference at catch block entry)

### Vector & SIMD Operations
- `vadd <lhs>, <rhs>`, `vsub <lhs>, <rhs>`, `vmul <lhs>, <rhs>`, `vdiv <lhs>, <rhs>`
- `vmin <lhs>, <rhs>`, `vmax <lhs>, <rhs>`
- `vfma <a:vec>, <b:vec>, <c:vec>`
- `vand <lhs>, <rhs>`, `vor <lhs>, <rhs>`, `vxor <lhs>, <rhs>`, `vnot <val>`
- `vneg <val>`, `vsqrt <val>`
- `vload.<type> <base:ptr|gcref>, <offset:i32>`
- `vstore.<type> <base:ptr|gcref>, <offset:i32>, <val>`
- `vbroadcast.<vec_type> <scalar_val>`
- `vextract_lane <vec_val>, <lane:i32>`
- `vinsert_lane <vec_val>, <scalar_val>, <lane:i32>`
- `vshuffle <lhs>, <rhs>, <mask:hex_imm>`
- `vzero.<vec_type>`

### Coroutines
- `coro_create @fn_symbol(<args...>)` -> `ptr`
- `coro_suspend.<type> <coro_ptr:ptr> [, <resume_id:i32>]` -> `<type>`
- `coro_resume.<type> <coro_ptr:ptr> [, <yield_val>]` -> `<type>`
- `coro_destroy <coro_ptr:ptr>`

---

## 4. Textual Format Example

```mir
module @fib_module

func @fibonacci(%0: i64) -> i64 {
bb0:
  %1 = iconst.i64 2
  %2 = slt.i64 %0, %1
  br_if %2, bb_base, bb_rec

bb_base:
  ret %0

bb_rec:
  %3 = iconst.i64 1
  %4 = sub.i64 %0, %3
  %5 = call.i64 @fibonacci(%4)
  %6 = iconst.i64 2
  %7 = sub.i64 %0, %6
  %8 = call.i64 @fibonacci(%7)
  %9 = add.i64 %5, %8
  ret %9
}
```
