.code

EXTERN brass_runtime_gc_safepoint_bridge: PROC
EXTERN brass_runtime_gc_alloc_bridge: PROC
EXTERN brass_runtime_gc_collect_bridge: PROC
EXTERN brass_coro_create_at: PROC
EXTERN brass_coro_create_body_at: PROC
EXTERN brass_throw_impl: PROC
EXTERN brass_current_exception_bits: PROC

brass_get_rbp PROC
    mov rax, rbp
    ret
brass_get_rbp ENDP

; void brass_gc_safepoint()
brass_gc_safepoint PROC FRAME
    push rbp
    .pushreg rbp
    mov rbp, rsp
    .setframe rbp, 0
    sub rsp, 32
    .allocstack 32
    .endprolog
    mov rcx, qword ptr [rbp]
    mov rdx, qword ptr [rbp + 8]
    call brass_runtime_gc_safepoint_bridge
    add rsp, 32
    pop rbp
    ret
brass_gc_safepoint ENDP

; uintptr_t brass_gc_alloc(size_t size, uint64_t pointer_mask, uint32_t type_tag)
brass_gc_alloc PROC FRAME
    push rbp
    .pushreg rbp
    mov rbp, rsp
    .setframe rbp, 0
    sub rsp, 48
    .allocstack 48
    .endprolog
    ; rcx = size
    ; rdx = pointer_mask
    ; r8  = type_tag
    ; r9  = caller_rbp
    mov r9, qword ptr [rbp]
    ; [rsp + 32] = caller_ip
    mov rax, qword ptr [rbp + 8]
    mov qword ptr [rsp + 32], rax
    call brass_runtime_gc_alloc_bridge
    add rsp, 48
    pop rbp
    ret
brass_gc_alloc ENDP

; uintptr_t brass_coro_create(void* fn_ptr, uint32_t slot_count, uint64_t pointer_mask)
; Generated code's coroutine-frame allocation: as brass_gc_alloc, passes the
; calling frame so a collection it triggers updates that frame's gcrefs.
brass_coro_create PROC FRAME
    push rbp
    .pushreg rbp
    mov rbp, rsp
    .setframe rbp, 0
    sub rsp, 48
    .allocstack 48
    .endprolog
    ; rcx = fn_ptr
    ; edx = slot_count
    ; r8  = pointer_mask
    ; r9  = caller_rbp
    mov r9, qword ptr [rbp]
    ; [rsp + 32] = caller_ip
    mov rax, qword ptr [rbp + 8]
    mov qword ptr [rsp + 32], rax
    call brass_coro_create_at
    add rsp, 48
    pop rbp
    ret
brass_coro_create ENDP

; uintptr_t brass_coro_create_body(const void* body)
; The same for a frame of a body descriptor: rcx = body, rdx = caller_rbp,
; r8 = caller_ip for brass_coro_create_body_at.
brass_coro_create_body PROC FRAME
    push rbp
    .pushreg rbp
    mov rbp, rsp
    .setframe rbp, 0
    sub rsp, 32
    .allocstack 32
    .endprolog
    mov rdx, qword ptr [rbp]
    mov r8, qword ptr [rbp + 8]
    call brass_coro_create_body_at
    add rsp, 32
    pop rbp
    ret
brass_coro_create_body ENDP

; void brass_gc_collect()
brass_gc_collect PROC FRAME
    push rbp
    .pushreg rbp
    mov rbp, rsp
    .setframe rbp, 0
    sub rsp, 32
    .allocstack 32
    .endprolog
    mov rcx, qword ptr [rbp]
    mov rdx, qword ptr [rbp + 8]
    call brass_runtime_gc_collect_bridge
    add rsp, 32
    pop rbp
    ret
brass_gc_collect ENDP

; void brass_throw(HostValue val)
; rcx = val
; 256 = 32 shadow + sizeof(SavedRegisters) (216), rounded for alignment;
; brass_throw_impl copies the whole struct, so all of it must be in-frame.
brass_default_throw PROC FRAME
    push rbp
    .pushreg rbp
    mov rbp, rsp
    .setframe rbp, 0
    sub rsp, 256
    .allocstack 256
    .endprolog
    mov qword ptr [rsp + 32], r15
    mov qword ptr [rsp + 40], r14
    mov qword ptr [rsp + 48], r13
    mov qword ptr [rsp + 56], r12
    mov qword ptr [rsp + 64], rdi
    mov qword ptr [rsp + 72], rsi
    mov qword ptr [rsp + 80], rbx
    lea rdx, [rsp + 32]         ; rdx = SavedRegisters*
    mov r8, qword ptr [rbp]     ; r8  = caller_rbp
    mov r9, qword ptr [rbp + 8] ; r9  = caller_ip
    call brass_throw_impl
    add rsp, 256
    pop rbp
    ret
brass_default_throw ENDP

; void brass_rethrow()
; Re-raises the pending exception. Tail-jumps into brass_throw so the walker
; sees the JIT frame's return address rather than a C++ helper's.
brass_default_rethrow PROC FRAME
    sub rsp, 40
    .allocstack 40
    .endprolog
    call brass_current_exception_bits
    add rsp, 40
    mov rcx, rax
    jmp brass_default_throw
brass_default_rethrow ENDP

; void brass_jump_to_landing_pad_msvc(void* ip, void* rbp, void* rsp, uint64_t val, const SavedRegisters* regs)
; rcx = ip, rdx = rbp, r8 = rsp, r9 = val, [rsp + 40] = regs
brass_jump_to_landing_pad_msvc PROC
    mov r10, qword ptr [rsp + 40]
    mov rax, r9
    mov r11, rcx
    test r10, r10
    jz skip_regs
    mov r15, qword ptr [r10 + 0]
    mov r14, qword ptr [r10 + 8]
    mov r13, qword ptr [r10 + 16]
    mov r12, qword ptr [r10 + 24]
    mov rdi, qword ptr [r10 + 32]
    mov rsi, qword ptr [r10 + 40]
    mov rbx, qword ptr [r10 + 48]
skip_regs:
    mov rbp, rdx
    mov rsp, r8
    jmp r11
brass_jump_to_landing_pad_msvc ENDP

; --- MSVC x64 JIT calling bridge routines for 256-bit vector results ---
; (JitExecutionEngine::invoke; the invoke thunk below captures only XMM0)

; The v256_* stubs return through an `out` pointer (rdx). rdx is
; volatile in the Win64 ABI, so the JIT callee may clobber it: the pointer is
; kept in callee-saved rbx across the call. push rbx + 32 bytes of shadow
; space keeps rsp 16-byte aligned at the call; the 5th argument (a2) is then
; at [rsp + 80].
brass_call_jit_v256_0 PROC FRAME
    push rbx
    .pushreg rbx
    sub rsp, 32
    .allocstack 32
    .endprolog
    mov rbx, rdx
    mov rax, rcx
    call rax
    vmovups ymmword ptr [rbx], ymm0
    vzeroupper
    add rsp, 32
    pop rbx
    ret
brass_call_jit_v256_0 ENDP

brass_call_jit_v256_1 PROC FRAME
    push rbx
    .pushreg rbx
    sub rsp, 32
    .allocstack 32
    .endprolog
    mov rbx, rdx
    mov rax, rcx
    vmovups ymm0, ymmword ptr [r8]
    call rax
    vmovups ymmword ptr [rbx], ymm0
    vzeroupper
    add rsp, 32
    pop rbx
    ret
brass_call_jit_v256_1 ENDP

brass_call_jit_v256_2 PROC FRAME
    push rbx
    .pushreg rbx
    sub rsp, 32
    .allocstack 32
    .endprolog
    mov rbx, rdx
    mov rax, rcx
    vmovups ymm0, ymmword ptr [r8]
    vmovups ymm1, ymmword ptr [r9]
    call rax
    vmovups ymmword ptr [rbx], ymm0
    vzeroupper
    add rsp, 32
    pop rbx
    ret
brass_call_jit_v256_2 ENDP

brass_call_jit_v256_3 PROC FRAME
    push rbx
    .pushreg rbx
    sub rsp, 32
    .allocstack 32
    .endprolog
    mov rbx, rdx
    mov rax, rcx
    vmovups ymm0, ymmword ptr [r8]
    vmovups ymm1, ymmword ptr [r9]
    mov r10, qword ptr [rsp + 80]
    vmovups ymm2, ymmword ptr [r10]
    call rax
    vmovups ymmword ptr [rbx], ymm0
    vzeroupper
    add rsp, 32
    pop rbx
    ret
brass_call_jit_v256_3 ENDP

; void x64_win64_invoke_thunk(const X64Win64InvokeArgs* args, X64Win64InvokeResult* result)
; rcx = args
; rdx = result
; Unwind data describes the frame, so a C++ exception thrown below the called
; JIT code unwinds through here: RBP (the saved-RBP slot, as a frame chain
; expects) is set after the pushes and the 8-byte pad, the last prolog step,
; because the argument area below it is sized at run time.
x64_win64_invoke_thunk PROC FRAME
    push rbp
    .pushreg rbp
    push rbx
    .pushreg rbx
    push rsi
    .pushreg rsi
    push rdi
    .pushreg rdi
    push r12
    .pushreg r12
    push r13
    .pushreg r13
    sub rsp, 8
    .allocstack 8
    lea rbp, [rsp + 48]
    .setframe rbp, 48
    .endprolog

    mov r12, rcx
    mov rbx, rdx                   ; result (r13 is the pinned TLS register)

    ; RSP is 16-byte aligned here; keep it so with an even word count.
    mov rcx, qword ptr [r12 + 104] ; stack_word_count
    lea rax, [rcx + 4]             ; 4 + stack_word_count
    test rax, 1
    jz alloc_even
    inc rax
alloc_even:
    shl rax, 3
    sub rsp, rax

    test rcx, rcx
    jz skip_stack_copy
    mov rsi, qword ptr [r12 + 96]  ; args->stack_words
    lea rdi, [rsp + 32]            ; [rsp + 32]
    rep movsq
skip_stack_copy:

    movdqu xmm0, xmmword ptr [r12 + 32]
    movdqu xmm1, xmmword ptr [r12 + 48]
    movdqu xmm2, xmmword ptr [r12 + 64]
    movdqu xmm3, xmmword ptr [r12 + 80]

    mov rcx, qword ptr [r12 + 0]
    mov rdx, qword ptr [r12 + 8]
    mov r8,  qword ptr [r12 + 16]
    mov r9,  qword ptr [r12 + 24]

    mov r11, qword ptr [r12 + 112]
    mov r13, qword ptr [r12 + 120] ; args->pinned_tls
    call r11

    mov qword ptr [rbx], rax
    movdqu xmmword ptr [rbx + 16], xmm0

    lea rsp, [rbp - 40]
    pop r13
    pop r12
    pop rdi
    pop rsi
    pop rbx
    pop rbp
    ret
x64_win64_invoke_thunk ENDP

END
