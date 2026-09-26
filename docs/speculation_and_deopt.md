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

**Allocas in a guard's state.** A value defined by an `alloca` is an address in the frame that computed it. When tier-2 (or OSR) code deoptimizes, the call finishes in a new Tier-0 frame, so a resume from native code (`Interpreter::resume`, `FastInterpreter::resume`) re-creates every alloca in the guard's state as a buffer of the new frame, with the alignment and `tagged`-ness of the Tier-0 alloca, and copies into it the contents of the deoptimized frame's buffer (that frame is still on the stack below the resume, and nothing allocates between the exit and the copy). The state value and any resume-block parameter it fills then name the new buffer, so a `tagged` buffer's words are roots of the frame that uses them. A guard failing inside its own interpreter frame keeps the frame's buffers as they are.

The compiler records function resume offsets in `ResumeTableRegistry` (`engine.get_resume_target_address(fn, resume_id)`). A resume block is entered only by a guard exit (see "Guard exits" in `mir_reference.md`); a function's entry never dispatches on its arguments, so `%resume_id` above is an ordinary parameter and calling `generic_twin(0, buf)` runs `bb0`.

---

## 4. On-Stack Replacement (OSR)

On-Stack Replacement moves a long-running interpreted loop into Tier-2 optimized code mid-loop. It happens in a program whose `MultiTierPipeline` runs it, on the `FastInterpreter` the pipeline runs it with; the program's `OsrCoordinator` (`FunctionDispatchTable::osr()`) is enabled and given a backedge threshold (`set_enabled`, `set_threshold`).

1. **Backedge counting**: the interpreter counts a function's backedges. Past the threshold, a backedge to a loop header asks the coordinator for the header's OSR entry.
2. **The OSR entry function** (`mir/osr_entry.hpp`): `plan_osr_entry` finds the values live into the header, and `build_osr_entry_function` builds a function of its own, `(ptr buffer) -> the function's return type`, whose entry block loads those values from the buffer and branches to the header, followed by every block reachable from it. Its control flow is ordinary, so the whole optimizer runs on it.
3. **Background compile**: the entry function, with the bodies of what it calls, is copied on the interpreter's thread, then optimized with the program's tier-2 passes and compiled on the shared `CompilePool`. The interpreter keeps running the loop meanwhile.
4. **Entry**: a later backedge to the header, once the code is ready, fills the buffer from the frame's registers and calls the entry; its result is the activation's. A failed guard in the code finishes the call in Tier 0, as tier-2 code of the program does.
5. **Invalidation**: that continuation runs the loop in a Tier-0 frame nested under the failed OSR code, so re-entering the same code at its next backedge would fail the same guard and nest another frame without bound. After `deopt_threshold` failures the entry is never entered again (`OsrCoordinator::note_deopt`, run by the resumer before the continuation); its code stays alive for frames still in it. With a front pass the header may then get a new entry, compiled against the feedback the failures produced, up to `kMaxReoptimizations` times.

A loop in a coroutine body, in a function with a resume point that no guard of it names, or with a gcref or vector value live into its header, stays interpreted. A function whose resume points are all its guards' resume targets does OSR: the entry function carries the guards (not the resume table), and one failing in the OSR code finishes the call in Tier 0 at the Tier-0 guard of the same resume id, as tier-2 code does. The reference `Interpreter`, and a `FastInterpreter` whose program's pipeline is not initialized, do not OSR.

Every program function the OSR copy carries a body of gets its handle when the copy is made, and each of those bodies gets a deopt resumer, so a callee's guard failing inside OSR code finishes the callee in Tier 0 and returns into the OSR code. A copy with a guarded body that has no Tier-0 function to resume in is not compiled.

A tier-up copy follows the same rule for the callee bodies it carries: after the pipeline, each one must pass `deopt_targets_valid` against its own Tier-0 function or the compile is rejected, and each one gets a resumer whether or not its code is published as that callee's entry.

## 4b. Front-end speculation passes

A front end whose Tier-0 functions carry speculation sites registers one pass with `MultiTierPipeline::set_tier2_front_pass`. `run_tier2_optimization_pipeline` runs it first on every tier-2 copy, tier-up and OSR alike, before the program's tier-2 passes. The pass sees the copy only and may use whatever feedback the front end has collected in Tier 0. It can decide per site to arm a guard (give it a real condition), or drop one, along with the copy's resume points. The Tier-0 function keeps its guards and resume table, so a guard armed in the copy still has a Tier-0 guard of its resume id to finish at.

The contract for a site's Tier-0 form: a guard whose condition always holds in Tier 0, whose resume target is the block the site's slow path starts in, and whose state is every value live into that block. That block is then correct to resume from with only the guard's state, so arming the guard in the copy is always sound. Bronze's sites are the reference (`il2mir/il_speculation.h` in bronze).

**Reoptimization.** A tier-2 function whose speculation is invalidated (its deopt count passes the threshold) normally bails out to Tier 1 for good. With a front pass registered, it instead drops the invalid code and clears its deopt counts, up to `FunctionFeedback::kMaxReoptimizations` (3) times, and tiers up again. The front pass then sees the feedback the failures produced and stops arming the sites that failed. After the last reoptimization it bails out as before.

---

## 4a. Deopt stress

`BRASS_DEOPT_STRESS` (`runtime/deopt_stress.hpp`) forces guards to fail, the way `BRASS_GC_STRESS` forces collections: `1`/`all` fails every eligible guard evaluation, `<N>` every Nth (one counter shared by every tier), `site:<R>` only guards with resume id R, `site:<R>:<N>` every Nth of those. Tests set it with `DeoptStressScope`. An eligible guard is one with an exit to take: an exit stub, or a resume target in a tier that can continue one (interpreters, the baseline JIT, pipeline tier-2 and OSR code; standalone tier 2 forces only stub guards).

The interpreters check at each evaluation. The baseline JIT, compiled while stress is on, calls `brass_deopt_stress_poll` before each selected guard. Tier 2 places, after the optimizer, a guard of its own before each eligible guard with the same exits, state values and resume id (`mir/deopt_stress.hpp`), counting inline, so the real guard's code and the values live at it are unchanged.

Under period 1 or one site, every tier exits where the stressed interpreter does and must give its answer; for a program whose exits compute what their fast paths compute (speculation's contract), every period must give the unstressed answer. `assert_diff_tiers` (`tests/differential/diff_harness.hpp`) checks both across the reference interpreter, the fast interpreter, Tier 1, pipeline Tier 2 and the standalone JIT.

---

## 5. Metadata Preservation During Optimization

The optimization and LIR peephole pipelines guarantee that:
- Guard exits and deopt stubs are never removed as dead code unless the condition is proven constant true by SCCP.
- Resume point targets and block layouts are preserved across branch simplifications.
- State map live variables maintain their assigned register/spill locations through register allocation.
