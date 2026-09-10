# Bronze IL Textual Translator & Brass Integration Report

## 1. Overview
The Bronze IL Translator (`brass::il::translate_bronze_il` and `brass-il` CLI) provides a direct, production-grade front door from Bronze's textual Intermediate Language to Brass Machine Intermediate Representation (MIR). It parses raw `bronze.exe il` compiler output verbatim, lowers block-parameter SSA graphs into typed Brass basic blocks, resolves host and dynamic calls, and supports in-memory JIT execution as well as AOT COFF/ELF object compilation.

---

## 2. Supported / Translated Construct Matrix

| Category | Bronze IL Instructions | Brass MIR Lowering | Status |
| :--- | :--- | :--- | :--- |
| **Module & Functions** | `module <path>`, `func name(...) -> type [export]` | `Module`, `Function` with typed signatures | Supported |
| **Basic Blocks** | `b0:`, `b1(%0: type, %1: type):` | `BasicBlock`, `add_block_param` | Supported |
| **Constants** | `const.f64`, `const.i32`, `const.bool`, `const.undefined`, `const.null`, `const.bigint` | `build_fconst_f64`, `build_iconst_i32`, `build_iconst_i64` | Supported |
| **Arithmetic** | `add`, `sub`, `neg`, `mul`, `div`, `mod` | `build_add`, `build_sub`, `build_mul`, `build_sdiv`, `build_smod` / `bronze_f64_mod` | Supported |
| **Bitwise Operations** | `and`, `or`, `xor`, `shl`, `shr`, `ushr`, `bitnot` | `build_and`, `build_or`, `build_xor`, `build_shl`, `build_ashr`, `build_lshr` with auto-coercion | Supported |
| **Type Conversions** | `to.int32`, `to.numeric` | `build_fptosi_i32`, `build_sitofp_f64_i32`, identity | Supported |
| **Boxing & Unboxing** | `box.f64`, `box.i32`, `box.bool`, `unbox.f64`, `unbox.i32`, `unbox.bool` | `build_bitcast_i64_f64`, NaN-tag embedding, `build_bitcast_f64_i64` | Supported |
| **Comparisons** | `cmp.lt`, `cmp.gt`, `cmp.le`, `cmp.ge`, `cmp.eq`, `cmp.ne`, `strict.eq`, `loose.eq`, `rel.*` | `build_slt`, `build_sgt`, `build_sle`, `build_sge`, `build_eq`, `build_ne` | Supported |
| **Control Flow** | `jump bX(...)`, `br %cond, bX(...), bY(...)`, `ret %val`, `ret` | `build_br`, `build_br_if`, `build_ret` | Supported |
| **Global Resolution** | `name.resolve "<name>"` | Lowers host symbols (`print`, `console.log`, `print.err`) to tagged host callable descriptors | Supported |
| **Dynamic Calls** | `call.dynamic %callee, %this, <argc>, %args...` | Marshalling through `bronze_call_dynamic_*` dispatcher with JS-compliant number formatting | Supported |
| **Environments & Scope**| `env.create`, `env.get`, `env.set`, `env.get.tdz`, `env.init.tdz` | Heap-allocated lexical environment chains (`BronzeEnv`) | Supported |
| **Closures** | `create.func @fn, <param_count>, %env` | Code pointer + environment pairing (`BronzeClosure`) | Supported |
| **Direct Calls & Prints**| `call @name(...)`, `print %0, ...`, `print.err %0, ...` | Direct internal/external subroutine calls & formatted printers | Supported |
| **Objects & Properties**| `create.object`, `create.array`, `prop.get`, `prop.set`, `elem.get`, `elem.set`, `method.def` | Polymorphic inline caches (PICs), Shape hidden class transitions, moving GC DynamicObject | Supported |
| **Exception Handling**| `handler b<id>`, `throw %val`, `exc.take` | MIR `invoke`, `throw`, `landing_pad` with zero-cost Win64 SEH & SysV DWARF LSDA unwinding | Supported |

---

## 3. Untranslated Constructs (Subset Boundary)

1. **Accessor Property Descriptors (`accessor.def`)**:
   - *Reason*: Requires getter/setter dynamic property dispatch and call stub synthesis.
2. **Async / Generator Coroutines (`create.async_machine`, `async.start`, `async.await`, `iter.open`, `iter.step`)**:
   - *Reason*: Requires coroutine state machine transformation and resume point descriptors.

---

## 4. Impedance Mismatches Identified & Resolved

1. **Dynamic Call Edge (`name.resolve "print"` + `call.dynamic`)**:
   - *Issue*: Bronze compiles JavaScript `print(...)` into a dynamic global name lookup (`name.resolve "print"`) followed by a variable-argument `call.dynamic` rather than a static opcode.
   - *Resolution*: Lowered to host symbol descriptors and dynamic dispatchers that unpack NaN-boxed argument payloads and format output identically to Node.js.
2. **ECMA-262 Number Formatting**:
   - *Issue*: C/C++ default `%g` or `%f` prints integer doubles as `1.62412e+06` or `55.0`.
   - *Resolution*: Implemented `std::to_chars`-backed JS-style formatting where integer-valued floats print without exponents or trailing decimals, matching Node.js `ToString(Number)` byte-for-byte.
3. **Lexical Scope & Per-Iteration Capture**:
   - *Issue*: Closures capture mutable outer scopes (including per-iteration variables in `for (let i...)` loops).
   - *Resolution*: Lowered to `BronzeEnv` frame chains and `BronzeClosure` handles passing the environment context as hidden first argument.
