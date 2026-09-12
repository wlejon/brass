# Speculation, Guards, Deoptimization & OSR

Brass provides native primitives for speculative optimization, deoptimization into generic fallbacks, and On-Stack Replacement (OSR) in dynamic language runtimes (such as JavaScript in the Bronze AOT compiler).

---

## 1. Guards & Side Exits

### MIR Construct
```mir
guard %cond, @fallback_exit, [%val1, %val2, %val3]
```

### Lowering Strategy
1. **Fast Path**:
   The inline fast path tests the boolean condition and branches out-of-line on failure:
   ```x86asm
   test eax, eax
   jz   .Lexit_stub_0
   ```
2. **Out-of-Line Exit Stub** (`.Lexit_stub_0`):
   Placed at the end of the function to keep the fast path instruction cache contiguous.
3. **State Map Materialization**:
   The exit stub:
   - Marshals live values from the state map (`%val1`, `%val2`, `%val3`) into the thread-local deoptimization frame (`brass::runtime::DeoptFrame`).
   - Sets the target `resume_id` and `DeoptReason`.
   - Transfers control to the runtime deoptimization handler or invokes the generic fallback twin.

---

## 2. Deoptimization Frame & Reason Model

The runtime deopt frame preserves live state across the speculation boundary:

```cpp
enum class DeoptReason : uint32_t {
    None = 0,
    Generic = 1,
    TypeCheckFailed = 2,
    Overflow = 3,
    ShapeCheckFailed = 4,
    BoundsCheckFailed = 5,
    DivisionByZero = 6,
    NullCheckFailed = 7,
    Custom = 8
};

struct DeoptValue {
    DeoptValueKind kind; // Int32, Int64, Float64, Pointer, GcRef
    uint64_t raw;
};

class DeoptFrame {
public:
    static constexpr size_t kMaxSlots = 64;
    uint32_t resume_id = 0;
    DeoptReason reason = DeoptReason::Generic;
    void* target_fn = nullptr;
    size_t count = 0;
    std::array<uint64_t, kMaxSlots> slots;
    std::array<DeoptValueKind, kMaxSlots> kinds;
};

// Thread-local accessors
DeoptFrame* brass_get_thread_deopt_frame();
void* brass_deopt_exit(uint32_t resume_id, uint32_t reason, uint32_t count, const uint64_t* raw_slots);
```

---

## 3. Resume Tables & Interior Entry Points

Speculatively optimized functions can deoptimize into an unoptimized "generic twin" function. The generic twin declares interior resume points at block targets:

```mir
func @generic_twin(%resume_id: i32, %state_buf: ptr) -> i64 {
bb0:
  ret 0

bb_resume_0:
  %v0 = load.i64 %state_buf, 0
  %v1 = load.i64 %state_buf, 8
  %sum = add.i64 %v0, %v1
  ret %sum
}
```

In the C++ API:
```cpp
twin->add_resume_point(0, bb_resume_0);
```

The compiler records function resume offsets in `ResumeTableRegistry`. The `JitExecutionEngine` can invoke `engine.resume("generic_twin", resume_id, args)` directly at the native interior entry point.

---

## 4. On-Stack Replacement (OSR)

Brass supports On-Stack Replacement to promote long-running interpreter loops directly into Tier-2 optimized JIT code while actively running:

1. **`osr_entry` Instruction**:
   Placed at candidate loop headers in compiled functions:
   ```mir
   osr_entry 100 [%val1, %val2]
   ```
2. **Loop Backedge Profiling**:
   The interpreter tracks loop backedge counts against a configurable threshold (e.g. `osr_threshold = 100`).
3. **`OsrMigrationFrame`**:
   Upon reaching the threshold, active interpreter variables are packed into an `OsrMigrationFrame` containing slot indices and NaN-boxed `HostValue` entries.
4. **`OsrCoordinator`**:
   The `OsrCoordinator` retrieves the compiled loop header target address via `engine.get_osr_entry_address("fn_name")` and pivots execution into the JIT frame with state intact.

---

## 5. Metadata Preservation During Optimization

The optimization and LIR peephole pipelines guarantee that:
- Guard exits and deopt stubs are never removed as dead code unless the condition is proven constant true by SCCP.
- Resume point targets and block layouts are preserved across branch simplifications.
- State map live variables maintain their assigned register/spill locations through register allocation.
