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
- **Strict Dominance**: Values are defined once and must dominate all uses.
- **Explicit Terminators**: Every basic block must end in a single terminator instruction (`br`, `br_if`, `switch`, `ret`, `unreachable`, `throw`, `invoke`, `resume`).
- **Exception Unwinding**: Call sites that may throw use `invoke`, specifying both a normal successor block and an unwind landing pad block.

---

## 3. Instruction Set Summary

### Constants & Conversions
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
- `sitofp.f64 <val:i32|i64>` -> `f64`
- `bitcast.i64 <val:f64>` -> `i64`
- `bitcast.f64 <val:i64>` -> `f64`

### Arithmetic, Logic & Bitwise
- `add.<type> <lhs>, <rhs>`, `sub.<type> <lhs>, <rhs>`, `mul.<type> <lhs>, <rhs>` (`i32`, `i64`, `f64`)
- `fma.f32 <a:f32>, <b:f32>, <c:f32>` -> `f32` ($a \times b + c$)
- `fma.f64 <a:f64>, <b:f64>, <c:f64>` -> `f64` ($a \times b + c$)
- `sdiv.<type> <lhs>, <rhs>`, `udiv.<type> <lhs>, <rhs>`, `smod.<type> <lhs>, <rhs>`, `umod.<type> <lhs>, <rhs>` (`i32`, `i64`)
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
