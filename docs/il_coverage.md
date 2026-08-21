# Bronze IL to Brass MIR Coverage Matrix

This document defines the complete coverage mapping of every data type and instruction opcode in Bronze IL (`D:/projects/bronze/src/il/il.h`) to Brass MIR (`include/brass/mir/`).

---

## 1. Type Coverage Matrix

| Bronze IL Type | Size (Bytes) | MIR Representation | Category | Notes & Test Reference |
| :--- | :--- | :--- | :--- | :--- |
| `Void` | 0 | `Type::void_type()` | **Expressible today** | Used for void-returning functions. [`test_sanity.cpp`](file:///D:/projects/brass/tests/unit/test_sanity.cpp), [`test_mir_builder.cpp`](file:///D:/projects/brass/tests/unit/test_mir_builder.cpp) |
| `Int32` | 4 | `Type::i32()` | **Expressible today** | 32-bit signed/unsigned integer. [`test_mir_builder.cpp`](file:///D:/projects/brass/tests/unit/test_mir_builder.cpp), [`test_il_expressibility.cpp`](file:///D:/projects/brass/tests/unit/test_il_expressibility.cpp) |
| `Int64` | 8 | `Type::i64()` | **Expressible today** | 64-bit signed/unsigned integer. [`test_mir_builder.cpp`](file:///D:/projects/brass/tests/unit/test_mir_builder.cpp), [`test_il_expressibility.cpp`](file:///D:/projects/brass/tests/unit/test_il_expressibility.cpp) |
| `Float64` | 8 | `Type::f64()` | **Expressible today** | IEEE 754 64-bit double precision float. [`test_il_expressibility.cpp`](file:///D:/projects/brass/tests/unit/test_il_expressibility.cpp), [`test_mir_builder.cpp`](file:///D:/projects/brass/tests/unit/test_mir_builder.cpp) |
| `Boolean` | 4 | `Type::i32()` | **Expressible today** | Booleans represented as 32-bit `0` (false) and `1` (true). [`test_il_expressibility.cpp`](file:///D:/projects/brass/tests/unit/test_il_expressibility.cpp), [`test_mir_verifier.cpp`](file:///D:/projects/brass/tests/unit/test_mir_verifier.cpp) |
| `Pointer` | 8 | `Type::ptr()` | **Expressible today** | Raw unmanaged machine address. [`test_lir_isel.cpp`](file:///D:/projects/brass/tests/unit/test_lir_isel.cpp), [`test_x64_codegen.cpp`](file:///D:/projects/brass/tests/unit/test_x64_codegen.cpp) |
| `GCRef` | 8 | `Type::gcref()` | **Expressible today** | Precise GC-managed heap object reference tracked in stack maps. [`test_mini_cheney.cpp`](file:///D:/projects/brass/tests/unit/test_mini_cheney.cpp), [`test_stack_map.cpp`](file:///D:/projects/brass/tests/unit/test_stack_map.cpp), [`test_moving_gc_native.cpp`](file:///D:/projects/brass/tests/differential/test_moving_gc_native.cpp) |

---

## 2. Opcode Coverage Matrix

### 2.1 Constants & Literals

| Bronze IL Opcode | Category | MIR Lowering | Test Reference |
| :--- | :--- | :--- | :--- |
| `ConstInt32` | **Expressible today** | `b.build_iconst_i32(val)` (`iconst.i32`) | [`test_mir_builder.cpp`](file:///D:/projects/brass/tests/unit/test_mir_builder.cpp), [`test_il_expressibility.cpp`](file:///D:/projects/brass/tests/unit/test_il_expressibility.cpp) |
| `ConstInt64` | **Expressible today** | `b.build_iconst_i64(val)` (`iconst.i64`) | [`test_mir_builder.cpp`](file:///D:/projects/brass/tests/unit/test_mir_builder.cpp), [`test_il_expressibility.cpp`](file:///D:/projects/brass/tests/unit/test_il_expressibility.cpp) |
| `ConstFloat64` | **Expressible today** | `b.build_fconst_f64(val)` (`fconst.f64`) | [`test_mir_builder.cpp`](file:///D:/projects/brass/tests/unit/test_mir_builder.cpp), [`test_il_expressibility.cpp`](file:///D:/projects/brass/tests/unit/test_il_expressibility.cpp) |
| `ConstBool` | **Expressible today** | `b.build_iconst_i32(val ? 1 : 0)` | [`test_il_expressibility.cpp`](file:///D:/projects/brass/tests/unit/test_il_expressibility.cpp) |
| `ConstNull` | **Expressible today** | `b.build_iconst_i64(0)` (`ptr`/`gcref` zero) | [`test_mini_cheney.cpp`](file:///D:/projects/brass/tests/unit/test_mini_cheney.cpp), [`test_mir_builder.cpp`](file:///D:/projects/brass/tests/unit/test_mir_builder.cpp) |
| `ConstUndefined` | **Expressible today** | `b.build_iconst_i64(tag)` or `patchable_const.i64` | [`test_embedding_differential.cpp`](file:///D:/projects/brass/tests/differential/test_embedding_differential.cpp), [`test_fuzz_js_shapes.cpp`](file:///D:/projects/brass/tests/differential/test_fuzz_js_shapes.cpp) |

### 2.2 Arithmetic, Bitwise & Primitives

| Bronze IL Opcode | Category | MIR Lowering | Test Reference |
| :--- | :--- | :--- | :--- |
| `Add` | **Expressible today** | `b.build_add(lhs, rhs)` (`add.i32`, `add.i64`, `add.f64`) | [`test_mir_builder.cpp`](file:///D:/projects/brass/tests/unit/test_mir_builder.cpp), [`test_il_expressibility.cpp`](file:///D:/projects/brass/tests/unit/test_il_expressibility.cpp) |
| `Sub` | **Expressible today** | `b.build_sub(lhs, rhs)` (`sub.i32`, `sub.i64`, `sub.f64`) | [`test_mir_builder.cpp`](file:///D:/projects/brass/tests/unit/test_mir_builder.cpp), [`test_il_expressibility.cpp`](file:///D:/projects/brass/tests/unit/test_il_expressibility.cpp) |
| `Mul` | **Expressible today** | `b.build_mul(lhs, rhs)` (`mul.i32`, `mul.i64`, `mul.f64`) | [`test_mir_builder.cpp`](file:///D:/projects/brass/tests/unit/test_mir_builder.cpp), [`test_il_expressibility.cpp`](file:///D:/projects/brass/tests/unit/test_il_expressibility.cpp) |
| `Div` | **Expressible today** | `b.build_sdiv`, `b.build_udiv`, `b.build_div` (`f64`) | [`test_mir_builder.cpp`](file:///D:/projects/brass/tests/unit/test_mir_builder.cpp), [`test_lir_isel.cpp`](file:///D:/projects/brass/tests/unit/test_lir_isel.cpp) |
| `Mod` | **Expressible today** | `b.build_smod(lhs, rhs)`, `b.build_umod(lhs, rhs)` | [`test_mir_builder.cpp`](file:///D:/projects/brass/tests/unit/test_mir_builder.cpp), [`test_lir_isel.cpp`](file:///D:/projects/brass/tests/unit/test_lir_isel.cpp) |
| `Neg` | **Expressible today** | `b.build_neg(val)` (`neg.i32`, `neg.i64`, `neg.f64`) | [`test_mir_builder.cpp`](file:///D:/projects/brass/tests/unit/test_mir_builder.cpp), [`test_lir_isel.cpp`](file:///D:/projects/brass/tests/unit/test_lir_isel.cpp) |
| `BitAnd` | **Expressible today** | `b.build_and(lhs, rhs)` (`and_.i32`, `and_.i64`) | [`test_mir_builder.cpp`](file:///D:/projects/brass/tests/unit/test_mir_builder.cpp), [`test_lir_isel.cpp`](file:///D:/projects/brass/tests/unit/test_lir_isel.cpp) |
| `BitOr` | **Expressible today** | `b.build_or(lhs, rhs)` (`or_.i32`, `or_.i64`) | [`test_mir_builder.cpp`](file:///D:/projects/brass/tests/unit/test_mir_builder.cpp), [`test_lir_isel.cpp`](file:///D:/projects/brass/tests/unit/test_lir_isel.cpp) |
| `BitXor` | **Expressible today** | `b.build_xor(lhs, rhs)` (`xor_.i32`, `xor_.i64`) | [`test_mir_builder.cpp`](file:///D:/projects/brass/tests/unit/test_mir_builder.cpp), [`test_lir_isel.cpp`](file:///D:/projects/brass/tests/unit/test_lir_isel.cpp) |
| `BitNot` | **Expressible today** | `b.build_not(val)` (`not_.i32`, `not_.i64`) | [`test_mir_builder.cpp`](file:///D:/projects/brass/tests/unit/test_mir_builder.cpp), [`test_lir_isel.cpp`](file:///D:/projects/brass/tests/unit/test_lir_isel.cpp) |
| `ShiftLeft` | **Expressible today** | `b.build_shl(lhs, rhs)` (`shl.i32`, `shl.i64`) | [`test_mir_builder.cpp`](file:///D:/projects/brass/tests/unit/test_mir_builder.cpp), [`test_lir_isel.cpp`](file:///D:/projects/brass/tests/unit/test_lir_isel.cpp) |
| `ShiftRight` | **Expressible today** | `b.build_ashr(lhs, rhs)` (`ashr.i32`, `ashr.i64`) | [`test_mir_builder.cpp`](file:///D:/projects/brass/tests/unit/test_mir_builder.cpp), [`test_lir_isel.cpp`](file:///D:/projects/brass/tests/unit/test_lir_isel.cpp) |
| `ShiftRightUnsigned` | **Expressible today** | `b.build_lshr(lhs, rhs)` (`lshr.i32`, `lshr.i64`) | [`test_mir_builder.cpp`](file:///D:/projects/brass/tests/unit/test_mir_builder.cpp), [`test_lir_isel.cpp`](file:///D:/projects/brass/tests/unit/test_lir_isel.cpp) |
| `RotateLeft` | **Expressible today** | Lowerable via `shl` + `lshr` + `or` bit combination | [`test_mir_builder.cpp`](file:///D:/projects/brass/tests/unit/test_mir_builder.cpp) |
| `RotateRight` | **Expressible today** | Lowerable via `lshr` + `shl` + `or` bit combination | [`test_mir_builder.cpp`](file:///D:/projects/brass/tests/unit/test_mir_builder.cpp) |
| `CountLeadingZeros` | **Expressible today** | `b.build_clz(val)` (`clz.i32`, `clz.i64`) | [`test_mir_builder.cpp`](file:///D:/projects/brass/tests/unit/test_mir_builder.cpp), [`test_lir_isel.cpp`](file:///D:/projects/brass/tests/unit/test_lir_isel.cpp) |
| `CountTrailingZeros` | **Expressible today** | `b.build_ctz(val)` (`ctz.i32`, `ctz.i64`) | [`test_mir_builder.cpp`](file:///D:/projects/brass/tests/unit/test_mir_builder.cpp), [`test_lir_isel.cpp`](file:///D:/projects/brass/tests/unit/test_lir_isel.cpp) |
| `PopCount` | **Expressible today** | `b.build_popcnt(val)` (`popcnt.i32`, `popcnt.i64`) | [`test_mir_builder.cpp`](file:///D:/projects/brass/tests/unit/test_mir_builder.cpp), [`test_lir_isel.cpp`](file:///D:/projects/brass/tests/unit/test_lir_isel.cpp) |
| `Abs` | **Expressible today** | Integer: `slt 0` + `select` + `neg`; Float: bitwise sign clear | [`test_mir_builder.cpp`](file:///D:/projects/brass/tests/unit/test_mir_builder.cpp), [`test_mir_unroll_select.cpp`](file:///D:/projects/brass/tests/unit/test_mir_unroll_select.cpp) |
| `Min` | **Expressible today** | `slt`/`ult` + `select` | [`test_mir_unroll_select.cpp`](file:///D:/projects/brass/tests/unit/test_mir_unroll_select.cpp) |
| `Max` | **Expressible today** | `sgt`/`ugt` + `select` | [`test_mir_unroll_select.cpp`](file:///D:/projects/brass/tests/unit/test_mir_unroll_select.cpp) |
| `Sqrt` | **Expressible today** | `b.build_sqrt(val)` (`sqrt.f64` -> `sqrtsd`) | [`test_mir_builder.cpp`](file:///D:/projects/brass/tests/unit/test_mir_builder.cpp), [`test_lir_isel.cpp`](file:///D:/projects/brass/tests/unit/test_lir_isel.cpp) |
| `Floor` | **Expressible with helper call** | Runtime helper call `floor(double)` / x64 SSE4.1 `roundsd` | [`lowering_guide.md`](file:///D:/projects/brass/docs/lowering_guide.md) |
| `Ceil` | **Expressible with helper call** | Runtime helper call `ceil(double)` / x64 SSE4.1 `roundsd` | [`lowering_guide.md`](file:///D:/projects/brass/docs/lowering_guide.md) |
| `Trunc` | **Expressible with helper call** | Runtime helper call `trunc(double)` / `fptosi` conversion | [`test_il_expressibility.cpp`](file:///D:/projects/brass/tests/unit/test_il_expressibility.cpp) |
| `Sin` | **Expressible with helper call** | Runtime helper call `sin(double)` via C runtime | [`lowering_guide.md`](file:///D:/projects/brass/docs/lowering_guide.md) |
| `Cos` | **Expressible with helper call** | Runtime helper call `cos(double)` via C runtime | [`lowering_guide.md`](file:///D:/projects/brass/docs/lowering_guide.md) |
| `Pow` | **Expressible with helper call** | Runtime helper call `pow(double, double)` via C runtime | [`lowering_guide.md`](file:///D:/projects/brass/docs/lowering_guide.md) |
| `Exp` | **Expressible with helper call** | Runtime helper call `exp(double)` via C runtime | [`lowering_guide.md`](file:///D:/projects/brass/docs/lowering_guide.md) |
| `Log` | **Expressible with helper call** | Runtime helper call `log(double)` via C runtime | [`lowering_guide.md`](file:///D:/projects/brass/docs/lowering_guide.md) |

### 2.3 Overflow-Checked Arithmetic

| Feature | Category | MIR Lowering | Test Reference |
| :--- | :--- | :--- | :--- |
| Signed Add Overflow Check | **Expressible today** | `b.build_sadd_overflow(lhs, rhs)` (`sadd_overflow.i32|i64`) | [`test_il_expressibility.cpp`](file:///D:/projects/brass/tests/unit/test_il_expressibility.cpp) |
| Signed Sub Overflow Check | **Expressible today** | `b.build_ssub_overflow(lhs, rhs)` (`ssub_overflow.i32|i64`) | [`test_il_expressibility.cpp`](file:///D:/projects/brass/tests/unit/test_il_expressibility.cpp) |
| Signed Mul Overflow Check | **Expressible today** | `b.build_smul_overflow(lhs, rhs)` (`smul_overflow.i32|i64`) | [`test_il_expressibility.cpp`](file:///D:/projects/brass/tests/unit/test_il_expressibility.cpp) |
| Unsigned Add Overflow Check | **Expressible today** | `b.build_uadd_overflow(lhs, rhs)` (`uadd_overflow.i32|i64`) | [`test_il_expressibility.cpp`](file:///D:/projects/brass/tests/unit/test_il_expressibility.cpp) |
| Unsigned Sub Overflow Check | **Expressible today** | `b.build_usub_overflow(lhs, rhs)` (`usub_overflow.i32|i64`) | [`test_il_expressibility.cpp`](file:///D:/projects/brass/tests/unit/test_il_expressibility.cpp) |
| Unsigned Mul Overflow Check | **Expressible today** | `b.build_umul_overflow(lhs, rhs)` (`umul_overflow.i32|i64`) | [`test_il_expressibility.cpp`](file:///D:/projects/brass/tests/unit/test_il_expressibility.cpp) |

### 2.4 Comparisons

| Bronze IL Opcode | Category | MIR Lowering | Test Reference |
| :--- | :--- | :--- | :--- |
| `Equal` | **Expressible today** | `b.build_eq(lhs, rhs)` (`eq.i32`, `eq.i64`, `eq.f64`) | [`test_mir_builder.cpp`](file:///D:/projects/brass/tests/unit/test_mir_builder.cpp), [`test_il_expressibility.cpp`](file:///D:/projects/brass/tests/unit/test_il_expressibility.cpp) |
| `NotEqual` | **Expressible today** | `b.build_ne(lhs, rhs)` (`ne.i32`, `ne.i64`, `ne.f64`) | [`test_mir_builder.cpp`](file:///D:/projects/brass/tests/unit/test_mir_builder.cpp), [`test_il_expressibility.cpp`](file:///D:/projects/brass/tests/unit/test_il_expressibility.cpp) |
| `LessThan` | **Expressible today** | `b.build_slt`, `b.build_ult` (`i32`, `i64`, `f64`) | [`test_mir_builder.cpp`](file:///D:/projects/brass/tests/unit/test_mir_builder.cpp), [`test_il_expressibility.cpp`](file:///D:/projects/brass/tests/unit/test_il_expressibility.cpp) |
| `LessThanOrEqual` | **Expressible today** | `b.build_sle`, `b.build_ule` (`i32`, `i64`, `f64`) | [`test_mir_builder.cpp`](file:///D:/projects/brass/tests/unit/test_mir_builder.cpp), [`test_il_expressibility.cpp`](file:///D:/projects/brass/tests/unit/test_il_expressibility.cpp) |
| `GreaterThan` | **Expressible today** | `b.build_sgt`, `b.build_ugt` (`i32`, `i64`, `f64`) | [`test_mir_builder.cpp`](file:///D:/projects/brass/tests/unit/test_mir_builder.cpp), [`test_il_expressibility.cpp`](file:///D:/projects/brass/tests/unit/test_il_expressibility.cpp) |
| `GreaterThanOrEqual` | **Expressible today** | `b.build_sge`, `b.build_uge` (`i32`, `i64`, `f64`) | [`test_mir_builder.cpp`](file:///D:/projects/brass/tests/unit/test_mir_builder.cpp), [`test_il_expressibility.cpp`](file:///D:/projects/brass/tests/unit/test_il_expressibility.cpp) |

### 2.5 Conversions & Bitcasts

| Bronze IL Opcode | Category | MIR Lowering | Test Reference |
| :--- | :--- | :--- | :--- |
| `Int32ToFloat64` | **Expressible today** | `b.build_sitofp_f64_i32(val)` (`sitofp_f64_i32` -> `cvtsi2sd`) | [`test_il_expressibility.cpp`](file:///D:/projects/brass/tests/unit/test_il_expressibility.cpp), [`test_mir_builder.cpp`](file:///D:/projects/brass/tests/unit/test_mir_builder.cpp) |
| `Int64ToFloat64` | **Expressible today** | `b.build_sitofp_f64_i64(val)` (`sitofp_f64_i64` -> `cvtsi2sd`) | [`test_il_expressibility.cpp`](file:///D:/projects/brass/tests/unit/test_il_expressibility.cpp), [`test_mir_builder.cpp`](file:///D:/projects/brass/tests/unit/test_mir_builder.cpp) |
| `Float64ToInt32` | **Expressible today** | `b.build_fptosi_i32(val)` (`fptosi_i32` -> `cvttsd2si`) | [`test_il_expressibility.cpp`](file:///D:/projects/brass/tests/unit/test_il_expressibility.cpp), [`test_mir_builder.cpp`](file:///D:/projects/brass/tests/unit/test_mir_builder.cpp) |
| `Float64ToInt64` | **Expressible today** | `b.build_fptosi_i64(val)` (`fptosi_i64` -> `cvttsd2si`) | [`test_il_expressibility.cpp`](file:///D:/projects/brass/tests/unit/test_il_expressibility.cpp), [`test_mir_builder.cpp`](file:///D:/projects/brass/tests/unit/test_mir_builder.cpp) |
| `Int32ToInt64` | **Expressible today** | `b.build_sext_i64(val)` or `b.build_zext_i64(val)` | [`test_il_expressibility.cpp`](file:///D:/projects/brass/tests/unit/test_il_expressibility.cpp), [`test_mir_builder.cpp`](file:///D:/projects/brass/tests/unit/test_mir_builder.cpp) |
| `Int64ToInt32` | **Expressible today** | `b.build_trunc_i32(val)` (`trunc.i32`) | [`test_il_expressibility.cpp`](file:///D:/projects/brass/tests/unit/test_il_expressibility.cpp), [`test_mir_builder.cpp`](file:///D:/projects/brass/tests/unit/test_mir_builder.cpp) |
| `BitcastInt64ToFloat64` | **Expressible today** | `b.build_bitcast_f64_i64(val)` (`movq_gx`) | [`test_il_expressibility.cpp`](file:///D:/projects/brass/tests/unit/test_il_expressibility.cpp), [`test_mir_builder.cpp`](file:///D:/projects/brass/tests/unit/test_mir_builder.cpp) |
| `BitcastFloat64ToInt64` | **Expressible today** | `b.build_bitcast_i64_f64(val)` (`movq_xg`) | [`test_il_expressibility.cpp`](file:///D:/projects/brass/tests/unit/test_il_expressibility.cpp), [`test_mir_builder.cpp`](file:///D:/projects/brass/tests/unit/test_mir_builder.cpp) |

### 2.6 Memory, Heap Objects & Fields

| Bronze IL Opcode | Category | MIR Lowering | Test Reference |
| :--- | :--- | :--- | :--- |
| `Alloc` | **Expressible today** | `b.build_call("brass_gc_alloc", Type::gcref(), {size, type_id})` | [`test_mini_cheney.cpp`](file:///D:/projects/brass/tests/unit/test_mini_cheney.cpp), [`test_moving_gc_native.cpp`](file:///D:/projects/brass/tests/differential/test_moving_gc_native.cpp) |
| `Load` | **Expressible today** | `b.build_load`, `b.build_load_offset`, `b.build_load_indexed` | [`test_mir_builder.cpp`](file:///D:/projects/brass/tests/unit/test_mir_builder.cpp), [`test_lir_isel.cpp`](file:///D:/projects/brass/tests/unit/test_lir_isel.cpp) |
| `Store` | **Expressible today** | `b.build_store`, `b.build_store_offset`, `b.build_store_indexed` | [`test_mir_builder.cpp`](file:///D:/projects/brass/tests/unit/test_mir_builder.cpp), [`test_lir_isel.cpp`](file:///D:/projects/brass/tests/unit/test_lir_isel.cpp) |
| `LoadField` | **Expressible today** | `b.build_load_offset(field_type, obj_ref, field_offset)` | [`test_mini_cheney.cpp`](file:///D:/projects/brass/tests/unit/test_mini_cheney.cpp), [`test_mir_builder.cpp`](file:///D:/projects/brass/tests/unit/test_mir_builder.cpp) |
| `StoreField` | **Expressible today** | `b.build_store_offset(obj_ref, field_offset, val)` | [`test_mini_cheney.cpp`](file:///D:/projects/brass/tests/unit/test_mini_cheney.cpp), [`test_mir_builder.cpp`](file:///D:/projects/brass/tests/unit/test_mir_builder.cpp) |
| `LoadElement` | **Expressible today** | `b.build_load_indexed(elem_type, arr_ref, idx, scale, base_offset)` | [`test_mir_builder.cpp`](file:///D:/projects/brass/tests/unit/test_mir_builder.cpp), [`test_lir_isel.cpp`](file:///D:/projects/brass/tests/unit/test_lir_isel.cpp) |
| `StoreElement` | **Expressible today** | `b.build_store_indexed(arr_ref, idx, val, scale, base_offset)` | [`test_mir_builder.cpp`](file:///D:/projects/brass/tests/unit/test_mir_builder.cpp), [`test_lir_isel.cpp`](file:///D:/projects/brass/tests/unit/test_lir_isel.cpp) |
| `MemoryBarrier` | **Expressible with helper call** | Runtime barrier call / inline `mfence` | [`lowering_guide.md`](file:///D:/projects/brass/docs/lowering_guide.md) |

### 2.7 Control Flow, Branches & Calls

| Bronze IL Opcode | Category | MIR Lowering | Test Reference |
| :--- | :--- | :--- | :--- |
| `Branch` | **Expressible today** | `b.build_br(target_block, block_args)` | [`test_mir_builder.cpp`](file:///D:/projects/brass/tests/unit/test_mir_builder.cpp), [`test_mir_verifier.cpp`](file:///D:/projects/brass/tests/unit/test_mir_verifier.cpp) |
| `BranchIf` | **Expressible today** | `b.build_br_if(cond, true_bb, true_args, false_bb, false_args)` | [`test_mir_builder.cpp`](file:///D:/projects/brass/tests/unit/test_mir_builder.cpp), [`test_mir_verifier.cpp`](file:///D:/projects/brass/tests/unit/test_mir_verifier.cpp) |
| `Switch` | **Expressible today** | `b.build_switch(val, def_bb, def_args, cases)` (`switch.<type>`) | [`test_il_expressibility.cpp`](file:///D:/projects/brass/tests/unit/test_il_expressibility.cpp) |
| `Call` | **Expressible today** | `b.build_call(callee_sym, ret_type, args)` (Multi-arg stack passing & alignment) | [`test_il_expressibility.cpp`](file:///D:/projects/brass/tests/unit/test_il_expressibility.cpp), [`test_target_calling_conv.cpp`](file:///D:/projects/brass/tests/unit/test_target_calling_conv.cpp) |
| `CallIndirect` | **Expressible today** | `b.build_call_indirect(target_ptr, ret_type, args)` | [`test_x64_codegen.cpp`](file:///D:/projects/brass/tests/unit/test_x64_codegen.cpp), [`test_mir_builder.cpp`](file:///D:/projects/brass/tests/unit/test_mir_builder.cpp) |
| `Return` | **Expressible today** | `b.build_ret(val)` or `b.build_ret_void()` | [`test_mir_builder.cpp`](file:///D:/projects/brass/tests/unit/test_mir_builder.cpp), [`test_mir_verifier.cpp`](file:///D:/projects/brass/tests/unit/test_mir_verifier.cpp) |
| `Unreachable` | **Expressible today** | `b.build_unreachable()` | [`test_mir_builder.cpp`](file:///D:/projects/brass/tests/unit/test_mir_builder.cpp), [`test_mir_verifier.cpp`](file:///D:/projects/brass/tests/unit/test_mir_verifier.cpp) |
| `Phi` | **Expressible today** | Transformed to SSA Basic Block Parameters | [`test_mir_builder.cpp`](file:///D:/projects/brass/tests/unit/test_mir_builder.cpp), [`test_il_expressibility.cpp`](file:///D:/projects/brass/tests/unit/test_il_expressibility.cpp) |
| `Select` | **Expressible today** | `b.build_select(cond, true_val, false_val)` (`select.<type>`) | [`test_mir_unroll_select.cpp`](file:///D:/projects/brass/tests/unit/test_mir_unroll_select.cpp) |

---

## 3. Coverage Summary & Metrics

- **Total Bronze IL Types**: 7
  - **Expressible Today**: 7 / 7 (**100.0%**)
- **Total Bronze IL Opcodes**: 67
  - **Expressible Today (Pure MIR & Codegen)**: 59 / 67 (**88.1%**)
  - **Expressible via Runtime Helper Call**: 8 / 67 (**11.9%**) (Transcendental math helpers `sin`, `cos`, `pow`, `exp`, `log`, `floor`, `ceil`, and `MemoryBarrier`)
  - **Unimplemented Gaps**: 0 / 67 (**0.0%**)
- **Total Coverage**: **100% of Bronze IL is expressible and lowerable today.**
