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
    DeoptValueKind kind; // Int32, Int64, Float64, Pointer, GcRef, Float32, Boolean, Tagged
    uint64_t raw;
};

class DeoptFrame {
public:
    uint32_t resume_id = 0;
    DeoptReason reason = DeoptReason::Generic;
    void* target_fn = nullptr;
    void* code_entry = nullptr;          // optimized code that deoptimized
    size_t count = 0;                    // no limit
    std::vector<uint64_t> slots;
    std::vector<DeoptValueKind> kinds;
};

// x64 optimized code builds a DeoptExitRecord (header + slots + kinds) on
// its stack and calls brass_deopt_exit_record. Tier-2 code installed by
// CodeInstaller has a registered resumer: the failed guard's state is
// rebuilt as typed values and the call finishes in Tier 0 (exit stub
// function, else resume target, as the interpreter's guard does). After
// deopt_threshold failures at one guard the optimized code is invalidated
// and the function bails out of tier 2.

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

The compiler records function resume offsets in `ResumeTableRegistry` (`engine.get_resume_target_address(fn, resume_id)`). A resume block is entered only by a guard exit (see "Guard exits" in `mir_reference.md`); a function's entry never dispatches on its arguments, so `%resume_id` above is an ordinary parameter and calling `generic_twin(0, buf)` runs `bb0`.

---

## 4. On-Stack Replacement (OSR)

On-Stack Replacement moves a long-running interpreted loop into Tier-2 optimized code mid-loop. It happens in a program whose `MultiTierPipeline` runs it, on the `FastInterpreter` the pipeline runs it with; the program's `OsrCoordinator` (`FunctionDispatchTable::osr()`) is enabled and given a backedge threshold (`set_enabled`, `set_threshold`).

1. **Backedge counting**: the interpreter counts a function's backedges. Past the threshold, a backedge to a loop header asks the coordinator for the header's OSR entry.
2. **The OSR entry function** (`mir/osr_entry.hpp`): `plan_osr_entry` finds the values live into the header, and `build_osr_entry_function` builds a function of its own, `(ptr buffer) -> the function's return type`, whose entry block loads those values from the buffer and branches to the header, followed by every block reachable from it. Its control flow is ordinary, so the whole optimizer runs on it.
3. **Background compile**: the entry function, with the bodies of what it calls, is copied on the interpreter's thread, then optimized with the program's tier-2 passes and compiled on the shared `CompilePool`. The interpreter keeps running the loop meanwhile.
4. **Entry**: a later backedge to the header, once the code is ready, fills the buffer from the frame's registers and calls the entry; its result is the activation's. A failed guard in the code finishes the call in Tier 0, as tier-2 code of the program does.

A loop in a function with resume points (a coroutine, or a guard with a resume target), or with a gcref or vector value live into its header, stays interpreted. The reference `Interpreter`, and a `FastInterpreter` whose program's pipeline is not initialized, do not OSR.

---

## 5. Metadata Preservation During Optimization

The optimization and LIR peephole pipelines guarantee that:
- Guard exits and deopt stubs are never removed as dead code unless the condition is proven constant true by SCCP.
- Resume point targets and block layouts are preserved across branch simplifications.
- State map live variables maintain their assigned register/spill locations through register allocation.
