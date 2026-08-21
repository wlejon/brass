# Brass Compiler Optimization & Semantics Specification

This document audits all optimization passes in Brass, specifying observable floating-point (IEEE-754) and integer behaviors, algebraic transformation rules, and explicit opt-in gating flags.

---

## 1. Core Principles & Semantic Laws

1. **Semantics-Changing Optimizations Are Opt-In, Never Default**:
   Any transformation that alters observable floating-point rounding, exception state, precision, or non-wrapping integer behavior is strictly disabled by default.
2. **IEEE-754 Strictness by Default**:
   Floating-point arithmetic adheres strictly to IEEE-754 double-precision (`f64`) semantics:
   - Serial summation order is preserved bit-for-bit.
   - Operations are non-associative: $(a + b) + c \neq a + (b + c)$.
   - Signed zeroes, infinities, and NaN propagations are maintained.
3. **Deterministic Integer Modulo Arithmetic**:
   Integer operations execute two's complement wrapping arithmetic modulo $2^{w}$ ($w \in \{1, 8, 16, 32, 64\}$), with associative and commutative properties valid under modular ring equivalence.

---

## 2. Pass-by-Pass Semantic Audit

### 2.1. Constant Folding (`constant_folding_pass`)
- **Pass Type**: Scalar simplification.
- **Integer Behavior**:
  - Constant integer expressions (`iconst`, `add`, `sub`, `mul`, `and`, `or`, `xor`, `shl`, `lshr`, `ashr`, `icmp`) are evaluated at compile time.
  - Division by zero and out-of-range shifts are never folded; they remain runtime instructions.
- **Floating-Point Behavior**:
  - Constant floating-point expressions (`fconst`, `add`, `sub`, `mul`, `div`, `neg`, `fcmp`) are evaluated under IEEE-754 rules using host compiler intrinsics with default rounding mode.
  - Folds respect NaN payloads, signed zeroes (`+0.0` vs `-0.0`), and infinity arithmetic.
- **Flags**: Always active as part of basic optimization; preserves strict semantics.

---

### 2.2. Common Subexpression Elimination (`cse_pass`)
- **Pass Type**: Dominator-tree scoped Global Value Numbering (GVN).
- **Integer Behavior**:
  - Replaces redundant computations with dominating equivalent value results.
  - Identical expressions compute identical integer values; no algebraic reassociation is performed.
- **Floating-Point Behavior**:
  - Only eliminates byte-identical subexpressions (same opcode, identical SSA value operands).
  - Does NOT commute non-identical subexpressions or assume $(a \times b) \times c \equiv a \times (b \times c)$.
- **Flags**: Enabled via `LoopOptOptions::enable_dce` / default pipeline; strict semantics preserved.

---

### 2.3. Dead Code Elimination (`dead_code_elimination_pass`)
- **Pass Type**: Liveness-based dead instruction pruning.
- **Integer & Float Behavior**:
  - Removes unused instructions that have no side effects (i.e. not memory stores, volatile loads, calls, or control flow terminators).
  - Trapping instructions without effects are preserved if side-effectful.
- **Flags**: Enabled via `LoopOptOptions::enable_dce`; strict semantics preserved.

---

### 2.4. Loop Invariant Code Motion (`licm_pass`)
- **Pass Type**: Loop optimization (Dominator & Loop Tree).
- **Integer Behavior**:
  - Hoists loop-invariant integer expressions and read-only memory loads (when proven unaliased) to preheader blocks.
- **Floating-Point Behavior**:
  - Hoists loop-invariant floating-point calculations to the preheader.
  - Does NOT alter the execution order of loop-carried floating-point operations relative to one another.
- **Flags**: Enabled via `LoopOptOptions::enable_licm`; strict semantics preserved.

---

### 2.5. Induction Variable Strength Reduction (`ivsr_pass`)
- **Pass Type**: Loop induction optimization.
- **Integer Behavior**:
  - Transforms linear induction expressions ($v = i \times C + B$) into incremental additions across iterations.
  - Valid because integer multiplication distributes over addition in modular arithmetic.
- **Floating-Point Behavior**:
  - **Untouched**: Float variables are strictly excluded from IVSR to avoid cumulative floating-point drift.
- **Flags**: Enabled via `LoopOptOptions::enable_ivsr`; strict semantics preserved.

---

### 2.6. CFG Diamond Select Simplification (`simplify_cfg_diamonds`)
- **Pass Type**: Control flow simplification.
- **Integer & Float Behavior**:
  - Replaces conditional branches around single-assignment phi nodes with branchless `select` (x86-64 `cmov` / SSE `blendvpd`).
  - Only applied when neither branch contains memory stores, loads that can fault, or throwing subroutines.
  - Both operands must be pre-evaluated values with no speculative side effects.
- **Flags**: Enabled via `LoopOptOptions::enable_diamond_select`; strict semantics preserved.

---

### 2.7. Loop Unrolling & Reduction Jam (`loop_unroll_pass`)
- **Pass Type**: Counted loop unrolling and parallel reduction jamming.
- **Integer Behavior (Default: ON)**:
  - Counted loops with integer accumulation (`add`) are unrolled by factor $F$ (default 4).
  - Accumulators are split into $F$ parallel sub-accumulators in the unrolled header and combined via tree reduction at loop exit.
  - Modulo $2^{64}$ two's complement addition is associative and commutative; integer reduction jam produces bit-exact identical results to serial execution.
  - Gated by: `LoopUnrollOptions::enable_reduction_jam` (default `true`).
- **Floating-Point Behavior (Default: OFF / Strict IEEE-754)**:
  - IEEE-754 floating-point addition is non-associative: $(a + b) + c \neq a + (b + c)$.
  - **Strict Mode (Default)**: Floating-point loops are either not reassociated or unrolled with strict serial dependence chains, preserving exact serial rounding and bit-identical outputs.
  - **Opt-In Reassociation Mode (Flagged)**: When explicitly enabled, $F$-way accumulator splitting and tree reduction are applied to `f64` accumulation loops, achieving $2\times\text{--}3\times$ instruction throughput via parallel execution pipelines.
  - **Gating Controls**:
    - `Function::set_allow_fp_reassociation(bool)`
    - `Module::set_allow_fp_reassociation(bool)`
    - `LoopOptOptions::enable_fp_reassociation = true`
    - `LoopUnrollOptions::enable_fp_reduction_jam = true`

---

## 3. Optimization Pass Summary Matrix

| Pass Name | Applies to Types | Semantic Impact | Default Status | Opt-In / Gating Flag |
| :--- | :--- | :--- | :--- | :--- |
| **Constant Folding** | `i8`..`i64`, `f32`, `f64` | Bit-exact IEEE-754 & 2's complement | ON | Standard pipeline |
| **CSE** | All types | Bit-exact; non-reassociating | ON | `LoopOptOptions::enable_dce` |
| **DCE** | All types | Prunes dead values | ON | `LoopOptOptions::enable_dce` |
| **LICM** | All types | Bit-exact loop-invariant hoisting | ON | `LoopOptOptions::enable_licm` |
| **IVSR** | `i32`, `i64` only | Bit-exact modular arithmetic | ON | `LoopOptOptions::enable_ivsr` |
| **Diamond Select** | All types | Branchless select (`cmov`) | ON | `LoopOptOptions::enable_diamond_select` |
| **Integer Reduction Jam** | `i8`..`i64` | Bit-exact 2's complement | ON | `LoopUnrollOptions::enable_reduction_jam` |
| **FP Reduction Jam** | `f32`, `f64` | Reassociates IEEE-754 additions | **OFF** (Strict) | `Function::set_allow_fp_reassociation(true)` |

---

## 4. Verification and Differential Testing

To guarantee compliance:
1. **Bit-Exact Differential Tests**: Matrix computations in strict mode are compared against interpreter serial reference using deterministic pseudorandom mantissas. Exact bit equality (`std::memcmp == 0`) is verified.
2. **Bounded ULP Tolerance in Reassoc Mode**: In opt-in reassociation mode, results are verified to stay within $\le 64$ ULPs of strict reference across arbitrary odd and even matrix dimensions.
