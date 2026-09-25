; MSVC Windows-ARM64 (armasm64) counterparts of gc_msvc_x64.asm: the entry
; stubs generated code calls that need their caller's frame, the throw
; stubs, the landing-pad jump and the JIT invoke thunk. The same routines
; for GCC/Clang are C++ (coroutine.cpp, runtime_gc.cpp,
; runtime/exception_throw_aarch64.cpp, codegen/jit_invoke.cpp);
; every build CMake treats as MSVC (cl or clang-cl) takes them from here.
;
; A frame stub pushes a frame record {x29, x30} and sets x29 to it, as the
; AArch64 frame chain expects, so [x29] is the caller's frame pointer and
; [x29 + 8] the return address into the caller: the pair each C++ bridge
; takes as (caller_fp, caller_ip), as __builtin_frame_address(0) and
; __builtin_return_address(0) give on GCC/Clang.
;
; Unwind data is written out here (no kxarm64.h, which needs the C
; preprocessor): a .pdata entry per framed function and an .xdata record with
; its prolog codes, in reverse prolog order, and one epilog scope when it has
; an epilog. Codes used: 0xE1 set_fp (mov x29, sp), 0x80|(N/8 - 1)
; save_fplr_x (stp x29, x30, [sp, #-N]!), save_regp / save_reg, 0xE4 end.
; C++ exceptions and SEH raised in the bridges (a collection that throws,
; brass_throw_impl's brass_seh_raise and BrassException) unwind through them.

        AREA    |.text|, ALIGN=4, CODE, READONLY

        IMPORT  brass_runtime_gc_safepoint_bridge
        IMPORT  brass_runtime_gc_alloc_bridge
        IMPORT  brass_coro_create_at
        IMPORT  brass_throw_impl
        IMPORT  brass_current_exception_bits
        IMPORT  brass_runtime_gc_collect_bridge

; .pdata and .xdata of a function whose whole prolog is
;     stp x29, x30, [sp, #-$Frame]!
;     mov x29, sp
; ($Frame a multiple of 16, at most 512). With $Epilog "1" the function has
; one epilog, at label $Fn._epilog:
;     ldp x29, x30, [sp], #$Frame
;     ret (or a tail branch)
; described by the codes from index 1 (save_fplr_x, end). Labels $Fn and
; $Fn._end bound the function.
        MACRO
        BRASS_FRAME_UNWIND $Fn, $Frame, $Epilog
        AREA    |.pdata|, ALIGN=2, READONLY
        DCD     $Fn
        RELOC   2
        DCD     $Fn._xdata
        RELOC   2
        AREA    |.xdata|, ALIGN=2, READONLY
        ALIGN   4
$Fn._xdata
        IF "$Epilog" == "1"
        ; 1 code word, 1 epilog scope, no handler, function length in words
        DCD     (1:SHL:27) :OR: (1:SHL:22) :OR: (($Fn._end - $Fn)/4)
        ; epilog scope: first code index 1, start offset in words
        DCD     (1:SHL:22) :OR: (($Fn._epilog - $Fn)/4)
        ELSE
        DCD     (1:SHL:27) :OR: (($Fn._end - $Fn)/4)
        ENDIF
        DCB     0xE1, (0x80 :OR: (($Frame/8) - 1)), 0xE4, 0xE4
        AREA    |.text|, CODE, READONLY
        MEND

; uintptr_t brass_get_rbp(): the frame pointer of the calling C++ function.
        ALIGN   4
        EXPORT  brass_get_rbp
brass_get_rbp PROC
        mov     x0, x29
        ret
        ENDP

; void brass_gc_safepoint()
        ALIGN   4
        EXPORT  brass_gc_safepoint
brass_gc_safepoint PROC
        stp     x29, x30, [sp, #-16]!
        mov     x29, sp
        ldp     x0, x1, [x29]               ; caller_fp, caller_ip
        bl      brass_runtime_gc_safepoint_bridge
brass_gc_safepoint_epilog
        ldp     x29, x30, [sp], #16
        ret
brass_gc_safepoint_end
        ENDP
        BRASS_FRAME_UNWIND brass_gc_safepoint, 16, 1

; uintptr_t brass_gc_alloc(size_t size, uint64_t pointer_mask, uint32_t type_tag)
; x0 = size, x1 = pointer_mask, w2 = type_tag; x3 = caller_fp, x4 = caller_ip
        ALIGN   4
        EXPORT  brass_gc_alloc
brass_gc_alloc PROC
        stp     x29, x30, [sp, #-16]!
        mov     x29, sp
        ldp     x3, x4, [x29]
        bl      brass_runtime_gc_alloc_bridge
brass_gc_alloc_epilog
        ldp     x29, x30, [sp], #16
        ret
brass_gc_alloc_end
        ENDP
        BRASS_FRAME_UNWIND brass_gc_alloc, 16, 1

; uintptr_t brass_coro_create(void* fn_ptr, uint32_t slot_count, uint64_t pointer_mask)
; Generated code's coroutine-frame allocation: as brass_gc_alloc, passes the
; calling frame so a collection it triggers updates that frame's gcrefs.
; x0 = fn_ptr, w1 = slot_count, x2 = pointer_mask; x3 = caller_fp, x4 = caller_ip
        ALIGN   4
        EXPORT  brass_coro_create
brass_coro_create PROC
        stp     x29, x30, [sp, #-16]!
        mov     x29, sp
        ldp     x3, x4, [x29]
        bl      brass_coro_create_at
brass_coro_create_epilog
        ldp     x29, x30, [sp], #16
        ret
brass_coro_create_end
        ENDP
        BRASS_FRAME_UNWIND brass_coro_create, 16, 1

; void brass_gc_collect()
        ALIGN   4
        EXPORT  brass_gc_collect
brass_gc_collect PROC
        stp     x29, x30, [sp, #-16]!
        mov     x29, sp
        ldp     x0, x1, [x29]
        bl      brass_runtime_gc_collect_bridge
brass_gc_collect_epilog
        ldp     x29, x30, [sp], #16
        ret
brass_gc_collect_end
        ENDP
        BRASS_FRAME_UNWIND brass_gc_collect, 16, 1

; void brass_throw(HostValue val)
; x0 = val. Frame record + SavedRegisters (runtime/exception.hpp: 7 x86-64
; words, then x19..x28, fp, lr at 56, then d8..d15 at 152; 216 bytes) =
; 232, rounded to 240. The callee-saved registers are stored exactly as the
; throwing frame left them; brass_throw_impl(val = x0, regs = x1,
; caller_fp = x2, caller_ip = x3) walks from the throwing frame and does
; not return.
        ALIGN   4
        EXPORT  brass_throw
brass_throw PROC
        stp     x29, x30, [sp, #-240]!
        mov     x29, sp
        add     x1, sp, #16
        stp     x19, x20, [x1, #56]
        stp     x21, x22, [x1, #72]
        stp     x23, x24, [x1, #88]
        stp     x25, x26, [x1, #104]
        stp     x27, x28, [x1, #120]
        stp     d8, d9, [x1, #152]
        stp     d10, d11, [x1, #168]
        stp     d12, d13, [x1, #184]
        stp     d14, d15, [x1, #200]
        ldp     x2, x3, [x29]
        bl      brass_throw_impl
        brk     #0xF000
brass_throw_end
        ENDP
        BRASS_FRAME_UNWIND brass_throw, 240, 0

; void brass_rethrow()
; Re-raises the pending exception. Tail-branches into brass_throw so the
; walker sees the JIT frame's return address rather than a C++ helper's.
        ALIGN   4
        EXPORT  brass_rethrow
brass_rethrow PROC
        stp     x29, x30, [sp, #-16]!
        mov     x29, sp
        bl      brass_current_exception_bits    ; x0 = the pending value
brass_rethrow_epilog
        ldp     x29, x30, [sp], #16
        b       brass_throw
brass_rethrow_end
        ENDP
        BRASS_FRAME_UNWIND brass_rethrow, 16, 1

; void brass_jump_to_landing_pad_msvc(void* ip, void* fp, void* sp, uint64_t val,
;                                     const SavedRegisters* regs)
; x0 = ip, x1 = fp, x2 = sp, x3 = val, x4 = regs (may be null). Restores the
; callee-saved registers the walker collected, installs the landing frame and
; enters the pad with the value in x0. Never returns.
        ALIGN   4
        EXPORT  brass_jump_to_landing_pad_msvc
brass_jump_to_landing_pad_msvc PROC
        mov     x16, x0
        cbz     x4, brass_jump_skip_regs
        ldp     x19, x20, [x4, #56]
        ldp     x21, x22, [x4, #72]
        ldp     x23, x24, [x4, #88]
        ldp     x25, x26, [x4, #104]
        ldp     x27, x28, [x4, #120]
        ldp     d8, d9, [x4, #152]
        ldp     d10, d11, [x4, #168]
        ldp     d12, d13, [x4, #184]
        ldp     d14, d15, [x4, #200]
brass_jump_skip_regs
        mov     x29, x1
        mov     sp, x2
        mov     x0, x3
        br      x16
        ENDP

; void aarch64_invoke_thunk(const AArch64InvokeArgs* args, AArch64InvokeResult* result)
; x0 = args, x1 = result (codegen/jit_exec.hpp). AArch64InvokeArgs: x0..x7 at
; 0, v0..v7 at 64, stack_words at 192, stack_word_count at 200, target at
; 208, pinned TLS block (x28) at 216. AArch64InvokeResult: x0 at 0, x1 at 8,
; q0 at 16, q1 at 32. partition_aarch64_invoke_args pads the stack words to
; an even count, so sp stays 16-byte aligned. x29 is set last in the prolog,
; so the unwinder restores sp from it before the saves, whatever the stack
; arguments took.
        ALIGN   4
        EXPORT  aarch64_invoke_thunk
aarch64_invoke_thunk PROC
        stp     x29, x30, [sp, #-48]!
        stp     x19, x20, [sp, #16]
        str     x28, [sp, #32]
        mov     x29, sp
        mov     x19, x1                     ; x19 = result
        mov     x20, x0                     ; x20 = args
        ldr     x2, [x20, #200]             ; stack_word_count
        cbz     x2, aarch64_invoke_no_stack
        lsl     x3, x2, #3
        sub     sp, sp, x3
        ldr     x1, [x20, #192]             ; stack_words
        mov     x4, sp
aarch64_invoke_copy
        ldr     x5, [x1], #8
        str     x5, [x4], #8
        subs    x2, x2, #1
        b.ne    aarch64_invoke_copy
aarch64_invoke_no_stack
        add     x1, x20, #64                ; v0..v7
        ldp     q0, q1, [x1, #0]
        ldp     q2, q3, [x1, #32]
        ldp     q4, q5, [x1, #64]
        ldp     q6, q7, [x1, #96]
        ldr     x16, [x20, #208]            ; target
        ldr     x28, [x20, #216]            ; pinned TLS block
        ldp     x0, x1, [x20, #0]           ; x0..x7
        ldp     x2, x3, [x20, #16]
        ldp     x4, x5, [x20, #32]
        ldp     x6, x7, [x20, #48]
        blr     x16
        str     x0, [x19, #0]
        str     x1, [x19, #8]
        str     q0, [x19, #16]
        str     q1, [x19, #32]
aarch64_invoke_thunk_epilog
        mov     sp, x29
        ldr     x28, [sp, #32]
        ldp     x19, x20, [sp, #16]
        ldp     x29, x30, [sp], #48
        ret
aarch64_invoke_thunk_end
        ENDP

        AREA    |.pdata|, ALIGN=2, READONLY
        DCD     aarch64_invoke_thunk
        RELOC   2
        DCD     aarch64_invoke_thunk_xdata
        RELOC   2
        AREA    |.xdata|, ALIGN=2, READONLY
        ALIGN   4
aarch64_invoke_thunk_xdata
        ; 2 code words, 1 epilog scope, function length in words
        DCD     (2:SHL:27) :OR: (1:SHL:22) :OR: ((aarch64_invoke_thunk_end - aarch64_invoke_thunk)/4)
        ; the epilog mirrors the prolog: its codes are the prolog's, from index 0
        DCD     (0:SHL:22) :OR: ((aarch64_invoke_thunk_epilog - aarch64_invoke_thunk)/4)
        ; set_fp; save_reg x28 at [sp, #32] (X = 9, Z = 4); save_regp x19, x20
        ; at [sp, #16] (X = 0, Z = 2); save_fplr_x 48; end; pad
        DCB     0xE1, 0xD2, 0x44, 0xC8, 0x02, 0x85, 0xE4, 0xE4

        END
