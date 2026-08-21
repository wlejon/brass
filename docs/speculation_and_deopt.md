# Speculation, Guards, and Deoptimization

Brass provides native primitives for speculative optimization in dynamic languages (e.g. JavaScript in Bronze).

---

## 1. Guards & Side Exits

### MIR Construct
```mir
guard %cond, @exit_stub_0, [%val1, %val2, %val3]
```

### Lowering Strategy
1. The inline fast path executes a conditional branch:
   ```x86asm
   test  eax, eax
   jz    .Lexit_stub_0
   ```
2. The slow path (`.Lexit_stub_0`) is placed out-of-line at the end of the function.
3. The exit stub:
   - Spills/marshals the specified state map values into the thread's deoptimization buffer (`brass_deopt_frame`).
   - Sets the target resume ID / metadata pointer.
   - Tail-calls or jumps to the runtime deoptimization handler or fallback twin function.

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

The compiler emits a dispatch jump table at function entry for resume points.
