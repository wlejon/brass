# Speculation, Guards, and Deoptimization

Brass provides native primitives for speculative optimization in dynamic languages (such as JavaScript in the Bronze AOT compiler).

---

## 1. Guards & Side Exits

### MIR Construct
```mir
guard %cond, @exit_stub_0, [%val1, %val2, %val3]
```

### Lowering Strategy
1. **Fast Path**:
   The inline fast path tests the condition and branches out-of-line on failure:
   ```x86asm
   cmp  eax, 0
   je   .Lexit_stub_0
   ```
2. **Out-of-Line Exit Stub** (`.Lexit_stub_0`):
   Placed at the end of the function to keep the fast path instruction cache contiguous.
3. **State Map Materialization**:
   The exit stub:
   - Marshals live values from the state map (`%val1`, `%val2`, `%val3`) into the thread-local deoptimization frame (`brass_deopt_frame`).
   - Sets the target resume ID and deopt reason.
   - Tail-calls or returns into the runtime deoptimization handler or generic fallback twin.

---

## 2. Resume Tables & Interior Entry Points

Speculatively optimized functions can deoptimize into an unoptimized "generic twin" function. The generic twin declares interior resume points:

```mir
func @generic_twin(%resume_id: i32, %state_buffer: ptr) -> i64 {
resume_table:
  entry 0 -> bb_resume_loop
  entry 1 -> bb_resume_after_call

bb0:
  ; Standard function entry
  ...

bb_resume_loop:
  ; Restores variables from state_buffer and continues loop
  ...
}
```

The compiler records function resume offsets in the binary metadata and `JitExecutionEngine` can invoke `engine.resume("fn_name", resume_id, args)` directly at the native interior entry point.

---

## 3. Metadata Preservation During Optimization

The LIR peephole optimizer guarantees that:
- Guard exits and deopt stubs are never removed as dead code.
- Resume point targets are preserved across branch simplifications.
- State map live variables maintain their assigned register/spill locations through compilation.
