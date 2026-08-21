# Brass MIR Reference Specification

Brass Machine Intermediate Representation (MIR) is a typed SSA intermediate representation designed specifically for machine-level code generation with first-class support for precise moving garbage collection, speculation, and inline caching.

---

## 1. Type System

Brass MIR has a lean, machine-oriented type system:

| Type | Size (Bytes) | Register Class | Description |
| :--- | :--- | :--- | :--- |
| `i32` | 4 | GPR | 32-bit two's-complement integer |
| `i64` | 8 | GPR | 64-bit two's-complement integer |
| `f64` | 8 | XMM/FPR | 64-bit IEEE-754 floating point |
| `ptr` | 8 | GPR | Raw unmanaged machine pointer (not traced by GC) |
| `gcref` | 8 | GPR | Managed object reference (participates in regalloc, recorded in stack maps) |
| `void` | 0 | None | Void return type for functions |

---

## 2. SSA & Control Flow

- **Block Structured CFG**: A function body consists of one or more basic blocks (`bb0`, `bb1`, etc.).
- **Block Parameters**: Block arguments replace Phi nodes. Every branch to a target block passes values matching that block's parameter types.
- **Strict Dominance**: Values are defined once and must dominate all uses.
- **Explicit Terminators**: Every basic block must end in a single terminator instruction (`br`, `br_if`, `ret`, `unreachable`).

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

### Arithmetic & Logic
- `add.<type> <lhs>, <rhs>`, `sub.<type> <lhs>, <rhs>`, `mul.<type> <lhs>, <rhs>` (`i32`, `i64`, `f64`)
- `sdiv.<type> <lhs>, <rhs>`, `udiv.<type> <lhs>, <rhs>`, `smod.<type> <lhs>, <rhs>`, `umod.<type> <lhs>, <rhs>` (`i32`, `i64`)
- `neg.<type> <val>` (`i32`, `i64`, `f64`)
- `and.<type> <lhs>, <rhs>`, `or.<type> <lhs>, <rhs>`, `xor.<type> <lhs>, <rhs>` (`i32`, `i64`)
- `shl.<type> <lhs>, <rhs>`, `lshr.<type> <lhs>, <rhs>`, `ashr.<type> <lhs>, <rhs>`, `not.<type> <val>` (`i32`, `i64`)
- `clz.<type> <val>`, `ctz.<type> <val>`, `popcnt.<type> <val>` (`i32`, `i64`)
- `select.<type> <cond:i32>, <true_val>, <false_val>` (`i32`, `i64`, `f64`, `ptr`, `gcref`)
- `sadd_overflow.<type> <lhs>, <rhs>`, `ssub_overflow.<type>`, `smul_overflow.<type>` (`i32`, `i64` -> `i32` overflow flag)
- `uadd_overflow.<type> <lhs>, <rhs>`, `usub_overflow.<type>`, `umul_overflow.<type>` (`i32`, `i64` -> `i32` overflow flag)

### Comparisons
- `eq.<type>`, `ne.<type>`, `slt.<type>`, `ult.<type>`, `sle.<type>`, `ule.<type>`, `sgt.<type>`, `ugt.<type>`, `sge.<type>`, `uge.<type>` (`i32`, `i64`, `f64` -> `i32` condition)

### Memory Operations
- `load.<type> <base:ptr|gcref>` -> `<type>`
- `load.<type> <base:ptr|gcref>, <offset:i32>` -> `<type>`
- `store.<type> <base:ptr|gcref>, <val>`
- `store.<type> <base:ptr|gcref>, <offset:i32>, <val>`
- `load_indexed.<type> <base>, <index:i64>, <scale:1|2|4|8>` -> `<type>`
- `load_indexed.<type> <base>, <index:i64>, <scale:1|2|4|8>, <offset:i32>` -> `<type>`
- `store_indexed.<type> <base>, <index:i64>, <scale:1|2|4|8>, <val>`
- `store_indexed.<type> <base>, <index:i64>, <scale:1|2|4|8>, <offset:i32>, <val>`

### Function Calls & Safepoints
- `call.<type> @fn_symbol(<args...>)` -> `<type>`
- `call_indirect.<type> <callee_ptr:ptr>(<args...>)` -> `<type>`
- `patchable_call.<type> @patch_sym, @default_target(<args...>)` -> `<type>`
- `safepoint`

### Speculation & Deoptimization
- `guard %cond, @exit_stub, [%val1, %val2, ...]`
- `resume_point <resume_id:i32>`

### Terminators
- `br <target_block>(<args...>)`
- `br_if <cond:i32>, <true_block>(<true_args...>), <false_block>(<false_args...>)`
- `switch.<type> <val>, default: <def_block>(<def_args...>), [<val_1>: <target_1>(<args_1...>), ...]`
- `ret [<val>]`
- `unreachable`

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
