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

A guard that names a resume twin skips all of that: its stub stores the state under the frame and calls the twin (section 7), which is how an AOT image's guards exit.

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
6. **Leaving**: a guard of a callee body the code carries finishes only the callee and returns into the loop, so a frame still looping in invalidated code would fail the same copy on every iteration. An entry whose region calls anything is built with `OsrEntryPlan::leave_check`: the buffer carries the address of the entry's leave flag, which `note_deopt` sets when it invalidates the entry, and every backedge to the header polls it. Set, the code writes the loop's live values back to the buffer, marks it left and returns; `enter` copies them into the interpreter frame's registers and the frame goes on with the backedge in the interpreter, where a later backedge may enter a fresh entry. The failures a frame sees are then bounded by the threshold rather than by the loop's trip count. A frame in an inner loop of the entry's region leaves only when it next reaches the entry's header.

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

The interpreters check at each evaluation; the fast interpreter finds a guard's resume point when it compiles the function (`GuardInfo::resume_index`), so an evaluation that does not fail costs only the counter. The baseline JIT, compiled while stress is on, calls `brass_deopt_stress_poll` before each selected guard. Tier 2 places, after the optimizer, a guard of its own before each eligible guard with the same exits, state values and resume id (`mir/deopt_stress.hpp`), counting inline, so the real guard's code and the values live at it are unchanged.

Under period 1 or one site, every tier exits where the stressed interpreter does and must give its answer; for a program whose exits compute what their fast paths compute (speculation's contract), every period must give the unstressed answer. `assert_diff_tiers` (`tests/differential/diff_harness.hpp`) checks both across the reference interpreter, the fast interpreter, Tier 1, pipeline Tier 2 and the standalone JIT.

---

## 5. Metadata Preservation During Optimization

The optimization and LIR peephole pipelines guarantee that:
- Guard exits and deopt stubs are never removed as dead code unless the condition is proven constant true by SCCP.
- Resume point targets and block layouts are preserved across branch simplifications.
- State map live variables maintain their assigned register/spill locations through register allocation.
- A guard's state is live up to the guard's conditional branch, not to its exit block. Isel places each exit block (one `GuardExit`, entered only by that `Jcc`) after every other block, and nothing runs between the branch and the exit, so liveness counts the exit's reads at the `Jcc` (`LivenessAnalysis::find_guard_exits`). Counted where the block stands, every state value held a register or a spill slot from its definition to the end of the function. On a bronze nbody kernel with its arithmetic guards armed that was 176 spill slots, and it is now 45.

## 6. Guards in coroutine bodies

Coroutine lowering (`mir/coro_transform.hpp`) leaves a body's guards working in every tier:
- A suspend's state id is not a resume id. Only the entry's dispatch switch enters a suspend's resume block. The block's resume-table entry, which keeps loop passes and allocation sinking off a body that is re-entered mid-body, is `coro_suspend_resume_id(state)`, the state id with the top bit set. A front end numbers its guards below 2^31 however it likes, including with the same numbers as its suspends.
- After lowering, each guard that resumes in the body gets appended to its state, in id order, whatever its resume target now reads and the state lacks. That is the frame, which every spill store, reload and completion store addresses, plus anything the target's successors reload. A guard in a block the lowered body cannot reach (code after a `throw`) never runs and has no provable state, so lowering removes it, and its resume point unless another guard names that id. Values reloaded after a suspend are renamed in the state like any other use.
- A failed guard in tier-2 code of a body finishes that resume in Tier 0 over the same heap frame. The next resume dispatches on the frame's state as usual, in whatever tier the body has. `CoroTiers` in `tests/unit/test_coro_tiers.cpp` holds guards before and after a suspend to the interpreter's answer in every placement, under stress and with real deopts.

---

## 7. Guards in AOT images: resume twins

An image has no interpreter to finish a call in and cannot recompile. There, a function that speculates is compiled twice into the same image: the speculated function, whose guards have exits only, and its **resume twin**, the function's generic code with every resume point as an entry. A failed guard calls the twin with its resume id and its state, and the twin's result is the function's. Nothing of the runtime is on that path.

### 7.1. The pieces

- **Site form.** The front end lowers each function once, exactly as it does for the JIT tiers (4b, `speculation_sites.hpp`): a site is a guard whose resume target is a block of the same function (the site's generic code), whose state is every value live into that block, with the target's parameters first. Every tier runs this form.
- **The twin** (`build_resume_twin`, `mir/resume_twin.hpp`): a function `(i32 resume id, ptr state) -> R` marked `resume_twin` (`Function::is_resume_twin`). Its entry block switches on the resume id. Each case loads that guard's state values from `state` (value i at byte `8 * i`, in the guard's state order, each in its own type) and branches to the resume block with the first of them as its parameters. The rest are the values the code from there on reads under their own names. The twin holds every block reachable from a resume block. A site-form guard inside it is a branch (on to the rest of its block, or to its resume block when the condition is false), so the twin has no guard of its own and no resume table: it is ordinary code, and the whole optimizer runs on it, loop passes included.
- **The speculated function** (`speculate_against_twin`): a copy of the site form in which the front end has armed the guards it trusts (the site's test as the guard's condition, the branch behind it pinned to the fast side) and dropped the others, as `arm_speculation_sites` does for tier 2. Every remaining guard then names the twin, `guard %c, @twin, [state]`, the resume table is dropped and the slow blocks it kept alive go with it.
- **The generic entry** (`ResumeTwinOptions::entry_id`, `build_twin_entry`): the twin can also take the function's own entry as an entry, whose state is the arguments. `build_twin_entry` makes a function with the original signature that calls it. That is the unspeculated function without a third copy of the code.

`speculate_module_against_twins` does the first three for every function of a module that has a site, with no arming (each guard keeps the condition it has).

### 7.2. What a twin exit compiles to (x64)

```x86asm
  test  r8, r8              ; the guard: its fused compare ...
  jl    .Lexit              ; ... and a branch that is not taken
  ...
.Lexit:                     ; after the function's other blocks
  sub   rsp, 8*N (rounded to 16)
  mov   [rsp + 8*i], <state value i>        ; from the register or spill slot it is in
  mov   edi, <resume id>
  lea   rsi, [rsp]
  call  twin
  add   rsp, 8*N
  <epilogue>; ret           ; the twin's return registers are the function's
```

The guard costs the fast path its compare and one branch; a state value is read wherever the allocator has it at the branch, so a guard forces nothing into a spill slot (5). The exit is a call, not a jump: the speculated frame stays on the stack under the twin until the twin returns. So

- an `alloca` in a guard's state is still the speculated frame's block while the twin uses it, and a `tagged` block's words are that frame's roots at the exit's call, which has a stack map like any call;
- any number of state values pass, with no calling convention besides the two arguments;
- every frame is an ordinary frame at an ordinary call for the stack walk and for unwinding. An exception the twin raises is caught by the twin's own landing pads (it holds the same `invoke`s the speculated code had from the resume block on) and otherwise passes through the speculated frame as through any frame with no handler at that call.

The cost is the speculated frame's stack for the rest of the call. A call exits at most once (after the exit it is in the twin, which has no exits), so the depth is bounded by two frames per activation.

A reference in the state words needs no root while it sits there: the twin loads every word in the entry it dispatches to, before its first GC point, and the words lie in a live frame, which the collector's stack scan pins against (gc_contract.md 2.5).

After an exit the next call of the function runs the speculated code again; nothing is patched or counted there.

### 7.3. The contract for a front end

1. **Build the twin from the function the speculated copy is made from**, before arming or dropping any guard of the copy. A guard's resume id and the order and types of its state are what tie the two together. The twin records the state types of each id and the verifier holds every guard that names it to them (`Function::guard_exit_stub_matches`); passes keep a guard's state positions.
2. **A resume block may assume** exactly what a Tier-0 resume may: its parameters are the first state values, and every other value live into it is one of the guard's state values (the verifier's rule for a resume target, unchanged). It must be correct to enter with those values and no others. It may read memory, call, loop back into the function's loops, raise. It needs no resume table of its own.
3. **State values** are scalars: `i32`, `i64`, `f32`, `f64`, `ptr`, `gcref`, `tagged`. Each takes one 8-byte word (a narrower one its low bytes). A vector is rejected. `SpecSiteEmitter::finish_function` forms the list (the values live into the slow block, by id); a hand-built guard lists the target's parameters first.
4. **Between the last side effect the generic code must not repeat and the guard, nothing.** The twin continues at the resume block with the heap as the speculated code left it. Whatever the speculated code did before the guard that the resume block will do again (a store, a call, a counter) happens twice. In site form the guard stands directly in front of the site's branch and the slow block starts with the site's generic operation, so this holds by construction; a front end that moves or merges guards (hoisting a check out of a loop, say) must give the hoisted guard a resume block that is right at the hoisted position.
5. **Calling convention.** `twin(i32 resume id, ptr state)`, the platform's C convention, returning the function's return type in the function's return register. The state block belongs to the caller and is dead once the twin has read it. The twin and the speculated function must be in one module when the guard is compiled (the guard names the twin by symbol). Both are ordinary functions to the linker and to `FunctionPlacement`: placing twins among the cold functions keeps them out of the hot text.
6. **Inlining.** When brass inlines a speculated function, its guard becomes a branch to a cold block that stores the state into a block of the caller's frame, calls the twin and continues after the inlined body with the twin's result. The twin finishes the callee only; the caller stays in its speculated code. That is sound without the caller's twin knowing the callee, and it is the restriction: such a site is a cold call that rejoins, not an exit, so values of the caller that are live after the inlined body are live across it. A front end that wants exits only across an inlined call inlines in its own IL before lowering, so that the site form (and with it the twin) holds the callee's sites as sites of the caller, each with the caller's values in its state.
7. **Pinning to generic.** `ResumeTwinOptions::on_exit` emits front-end code into each of the twin's guard entries (after the entry has read its state): a counter in memory the front end can address, and whatever it does when the count passes its limit. The speculated function carries none of it. The remedy is the front end's as well, since it owns how its functions are reached: store `build_twin_entry`'s function where its calls load their target. For direct calls that place is an **entry cell**: `call @f via @cell` (and `invoke`) calls through a word of the image's data (`call [rip + cell]` on x64, loaded via `x16` + `blr` on AArch64). The front end defines the word holding `f`'s address, and the code `on_exit` emits stores the generic entry there once its count says so; every call through the cell is then generic, with nothing in `f`'s code and nothing more at its callers. The cell is a hint to every pass (they see a call of `f`; whatever the cell holds must compute what `f` computes), and a tier without the form calls `f`.
8. **Coroutine bodies have no twin yet** (`build_resume_twin` refuses one): lower their sites as branches.

### 7.4. Other tiers

A twin exit is an exit stub with another signature, and each tier takes it that way: both interpreters call `twin(id, &words)` with the state as raw words (`interpreter/twin_exit.hpp`), so a speculated module runs anywhere a site-form one does. The baseline JITs reject a guard that names a twin (`UnsupportedOperation`): tier 2 (x64 and AArch64) and the interpreters implement the direct exit. A twin exit never asks the runtime for a lower tier, in tier-2 JIT code either, so its failures are not counted toward invalidation; JIT front ends keep resume targets.

### 7.5. Deopt stress

`ModuleCompiler::set_deopt_stress` instruments twin exits like exit stubs: the stress guard has the same twin, id and state. Code that runs in the process that compiled it counts in place. An image built for another process sets `DeoptStressPlan::runtime_poll`: each stress guard calls `brass_deopt_stress_poll_site(resume id)`, an import of the image, and `BRASS_DEOPT_STRESS` (or `brass_set_deopt_stress`) in the process the image runs in decides which guards fail. A forced exit continues in the twin, whose copies of the sites are branches, so the rest of that call runs unstressed: every period must give the unstressed answer for a program whose exits compute what their fast paths compute. `assert_diff_tiers` has an image leg (`DiffTier::Aot`) that checks this for any module with a site-form guard whose image needs nothing from the process.
