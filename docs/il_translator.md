# Bronze IL Textual Translator & Brass Integration Report

## 1. Overview
The Bronze IL Translator (`brass::il::translate_bronze_il` and `brass-il` CLI) provides a direct front door from Bronze's textual Intermediate Language to Brass Machine Intermediate Representation (MIR). It enables Bronze IL programs to be parsed, lowered into typed SSA basic blocks, optimized via Brass's loop and scalar pipelines, executed in-memory via the Brass JIT engine, or compiled ahead-of-time to linkable `.obj` COFF/ELF files.

---

## 2. Supported / Translated Construct Subset

The translator currently supports the complete **numeric and scalar computational subset** of Bronze IL:

| Category | Bronze IL Instructions | Brass MIR Lowering |
| :--- | :--- | :--- |
| **Module & Functions** | `module <name>`, `func name(...) -> type [export]` | `Module`, `Function` with typed signatures |
| **Basic Blocks** | `b0:`, `b1(%0: type, %1: type):` | `BasicBlock`, `add_block_param` |
| **Constants** | `const.f64`, `const.i32`, `const.bool`, `const.undefined`, `const.null`, `const.bigint` | `build_fconst_f64`, `build_iconst_i32`, `build_iconst_i64` |
| **Arithmetic** | `add`, `sub`, `neg`, `mul`, `div`, `mod` | `build_add`, `build_sub`, `build_mul`, `build_div`, `build_srem` / `bronze_f64_mod` |
| **Bitwise Operations** | `and`, `or`, `xor`, `shl`, `shr`, `ushr`, `bitnot` | `build_and`, `build_or`, `build_xor`, `build_shl`, `build_ashr`, `build_lshr` with auto-coercion |
| **Type Conversions** | `to.int32`, `to.numeric` | `build_fptosi_i32_f64`, `build_sitofp_f64_i32`, identity |
| **Boxing & Unboxing** | `box.f64`, `box.i32`, `box.bool`, `unbox.f64`, `unbox.i32`, `unbox.bool` | `build_bitcast_i64_f64`, NaN-tag embedding, `build_bitcast_f64_i64` |
| **Comparisons** | `cmp.lt`, `cmp.gt`, `cmp.le`, `cmp.ge`, `cmp.eq`, `cmp.ne`, `strict.eq`, `loose.eq`, `rel.*` | `build_slt`, `build_sgt`, `build_sle`, `build_sge`, `build_eq`, `build_ne` |
| **Control Flow** | `jump bX(...)`, `br %cond, bX(...), bY(...)`, `ret %val`, `ret` | `build_br`, `build_br_if`, `build_ret` |
| **Calls & Prints** | `call @name(...)`, `print %0, ...`, `print.err %0, ...` | `build_call` to internal/external functions & `bronze_print_*` helpers |

---

## 3. Untranslated Constructs (Subset Boundary)

The following constructs belong to JavaScript runtime object and dynamic engine layers and are not yet lowered in this scalar pass:

1. **Object & Shape Inline Caches (`prop.get`, `prop.set`, `elem.get`, `elem.set`, `method.def`, `accessor.def`)**:
   - *Reason*: Requires polymorphic IC cache slot allocation and shape transition tables.
2. **Lexical Scope & Closures (`env.create`, `env.get`, `env.set`, `create.func`)**:
   - *Reason*: Requires runtime heap-allocated environment frames.
3. **Async / Generator Coroutines (`create.async_machine`, `async.start`, `async.await`, `iter.open`, `iter.step`)**:
   - *Reason*: Requires coroutine state machine transformation and resume point descriptors.
4. **Exception Tables (`handler b<id>`, `throw`, `exc.take`)**:
   - *Reason*: Requires Windows SEH / C++ landing pad metadata generation.

---

## 4. Impedance Mismatches Identified & Resolved

1. **Pure Block-Parameter SSA Alignment**:
   - *Insight*: Both Bronze IL and Brass MIR represent SSA join values using block parameters rather than $\phi$ nodes.
   - *Result*: Zero impedance mismatch; direct 1-to-1 lowering of branch arguments to basic block parameter values without edge splitting.
2. **Implicit Numeric Type Coercion**:
   - *Insight*: In JS semantics, bitwise operations (`&`, `|`, `^`, `<<`, `>>`) operate on 32-bit integers but produce `f64` number results when bound to `f64` variables.
   - *Resolution*: Lowering automatically inserts `sitofp_f64_i32` or `uitofp_f64_i32` (for `ushr`) when the destination type is `f64`.
3. **Floating-Point Remainder**:
   - *Insight*: JavaScript `%` operator on floats implements IEEE-754 remainder.
   - *Resolution*: Lowering dispatches `f64` `mod` to `bronze_f64_mod` (`std::fmod`).
4. **NaN-Box Value Interoperability**:
   - *Insight*: Bronze dynamic values are 64-bit NaN-boxed integers.
   - *Resolution*: Mapped directly to Brass `Type::i64()`, enabling seamless bitcast conversions to/from `f64`.
