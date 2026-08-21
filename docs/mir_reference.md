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
- `patchable_const.i32 <sym>, <imm32>` -> `i32`
- `patchable_const.i64 <sym>, <imm64>` -> `i64`
- `sext.i64 <i32>` -> `i64`
- `zext.i64 <i32>` -> `i64`
- `trunc.i32 <i64>` -> `i32`
- `fptosi.i32/i64 <f64>` -> `i32/i64`
- `sitofp.f64 <i32/i64>` -> `f64`
- `bitcast.i64 <f64>` -> `i64`
- `bitcast.f64 <i64>` -> `f64`

### Arithmetic & Logic
- `add`, `sub`, `mul`, `sdiv`, `udiv`, `smod`, `umod`, `neg` (`i32`, `i64`, `f64`)
- `and`, `or`, `xor`, `shl`, `lshr`, `ashr`, `not` (`i32`, `i64`)
- `clz`, `ctz`, `popcnt` (`i32`, `i64`)

### Comparisons
- `eq`, `ne`, `slt`, `ult`, `sle`, `ule`, `sgt`, `ugt`, `sge`, `uge` (`i32`, `i64`, `f64` -> `i32` condition)

### Memory Operations
- `load.<type> <base:ptr/gcref>, <offset:i32>` -> `<type>`
- `store.<type> <base:ptr/gcref>, <offset:i32>, <val>`
- `load_indexed.<type> <base>, <index:i64>, <scale:1|2|4|8>, <offset:i32>` -> `<type>`
- `store_indexed.<type> <base>, <index:i64>, <scale:1|2|4|8>, <offset:i32>, <val>`

### Function Calls & Safepoints
- `call <fn_symbol>(<args...>)` -> `<return_type>`
- `call_indirect <callee_ptr:ptr>(<args...>)` -> `<return_type>`
- `patchable_call <sym>, <default_target>(<args...>)` -> `<return_type>`
- `safepoint`

### Speculation & Deoptimization
- `guard <cond:i32>, exit_label, [<state_map_values...>]`
- `resume_point <resume_id:i32>`

### Terminators
- `br <target_block>(<args...>)`
- `br_if <cond:i32>, <true_block>(<args...>), <false_block>(<args...>)`
- `ret [<val>]`
- `unreachable`

---

## 4. Textual Format Example

```mir
func @fibonacci(%n: i32) -> i32 {
bb0:
  %c2 = iconst.i32 2
  %cond = slt.i32 %n, %c2
  br_if %cond, bb1, bb2

bb1:
  ret %n

bb2:
  %c1 = iconst.i32 1
  %n1 = sub.i32 %n, %c1
  %r1 = call @fibonacci(%n1)
  %n2 = sub.i32 %n, %c2
  %r2 = call @fibonacci(%n2)
  %sum = add.i32 %r1, %r2
  ret %sum
}
```
