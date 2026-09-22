.code

EXTERN brass_runtime_gc_safepoint_bridge: PROC
EXTERN brass_runtime_gc_alloc_bridge: PROC
EXTERN brass_throw_impl: PROC
EXTERN brass_current_exception_bits: PROC

brass_get_rbp PROC
    mov rax, rbp
    ret
brass_get_rbp ENDP

; void brass_gc_safepoint()
brass_gc_safepoint PROC
    push rbp
    mov rbp, rsp
    sub rsp, 32
    mov rcx, qword ptr [rbp]
    mov rdx, qword ptr [rbp + 8]
    call brass_runtime_gc_safepoint_bridge
    add rsp, 32
    pop rbp
    ret
brass_gc_safepoint ENDP

; uintptr_t brass_gc_alloc(size_t size, uint64_t pointer_mask, uint32_t type_tag)
brass_gc_alloc PROC
    push rbp
    mov rbp, rsp
    sub rsp, 48
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

; void brass_gc_collect()
brass_gc_collect PROC
    push rbp
    mov rbp, rsp
    sub rsp, 32
    mov rcx, qword ptr [rbp]
    mov rdx, qword ptr [rbp + 8]
    call brass_runtime_gc_safepoint_bridge
    add rsp, 32
    pop rbp
    ret
brass_gc_collect ENDP

EXTERN host_gc_safepoint_bridge: PROC
EXTERN host_gc_alloc_bridge: PROC
EXTERN host_gc_alloc_nanbox_bridge: PROC

; void host_gc_safepoint()
host_gc_safepoint PROC
    push rbp
    mov rbp, rsp
    sub rsp, 32
    mov rcx, qword ptr [rbp]
    mov rdx, qword ptr [rbp + 8]
    call host_gc_safepoint_bridge
    add rsp, 32
    pop rbp
    ret
host_gc_safepoint ENDP

; uintptr_t host_gc_alloc(size_t size, uint64_t pointer_mask, uint32_t type_tag)
host_gc_alloc PROC
    push rbp
    mov rbp, rsp
    sub rsp, 48
    ; rcx = size
    ; rdx = pointer_mask
    ; r8  = type_tag
    ; r9  = caller_rbp
    mov r9, qword ptr [rbp]
    ; [rsp + 32] = caller_ip
    mov rax, qword ptr [rbp + 8]
    mov qword ptr [rsp + 32], rax
    call host_gc_alloc_bridge
    add rsp, 48
    pop rbp
    ret
host_gc_alloc ENDP

; uint64_t host_gc_alloc_nanbox(size_t size, uint64_t pointer_mask, uint32_t type_tag)
host_gc_alloc_nanbox PROC
    push rbp
    mov rbp, rsp
    sub rsp, 48
    ; rcx = size
    ; rdx = pointer_mask
    ; r8  = type_tag
    ; r9  = caller_rbp
    mov r9, qword ptr [rbp]
    ; [rsp + 32] = caller_ip
    mov rax, qword ptr [rbp + 8]
    mov qword ptr [rsp + 32], rax
    call host_gc_alloc_nanbox_bridge
    add rsp, 48
    pop rbp
    ret
host_gc_alloc_nanbox ENDP

; void host_gc_collect()
host_gc_collect PROC
    push rbp
    mov rbp, rsp
    sub rsp, 32
    mov rcx, qword ptr [rbp]
    mov rdx, qword ptr [rbp + 8]
    call host_gc_safepoint_bridge
    add rsp, 32
    pop rbp
    ret
host_gc_collect ENDP

; void brass_throw(HostValue val)
; rcx = val
; 256 = 32 shadow + sizeof(SavedRegisters) (216), rounded for alignment;
; brass_throw_impl copies the whole struct, so all of it must be in-frame.
brass_throw PROC FRAME
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
brass_throw ENDP

; void brass_rethrow()
; Re-raises the pending exception. Tail-jumps into brass_throw so the walker
; sees the JIT frame's return address rather than a C++ helper's.
brass_rethrow PROC FRAME
    sub rsp, 40
    .allocstack 40
    .endprolog
    call brass_current_exception_bits
    add rsp, 40
    mov rcx, rax
    jmp brass_throw
brass_rethrow ENDP

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

; --- MSVC x64 JIT calling bridge routines for SIMD vector arguments ---

brass_call_jit_int_vec_ret_vec PROC
    sub rsp, 40
    mov rax, rcx
    mov rcx, rdx
    movdqu xmm1, xmmword ptr [r8]
    call rax
    add rsp, 40
    ret
brass_call_jit_int_vec_ret_vec ENDP

brass_call_jit_int_vec_ret_void PROC
    sub rsp, 40
    mov rax, rcx
    mov rcx, rdx
    movdqu xmm1, xmmword ptr [r8]
    call rax
    add rsp, 40
    ret
brass_call_jit_int_vec_ret_void ENDP

brass_call_jit_int_vec_ret_i64 PROC
    sub rsp, 40
    mov rax, rcx
    mov rcx, rdx
    movdqu xmm1, xmmword ptr [r8]
    call rax
    add rsp, 40
    ret
brass_call_jit_int_vec_ret_i64 ENDP

brass_call_jit_vec_int_ret_vec PROC
    sub rsp, 40
    mov rax, rcx
    movdqu xmm0, xmmword ptr [rdx]
    mov rdx, r8
    call rax
    add rsp, 40
    ret
brass_call_jit_vec_int_ret_vec ENDP

brass_call_jit_vec_int_ret_void PROC
    sub rsp, 40
    mov rax, rcx
    movdqu xmm0, xmmword ptr [rdx]
    mov rdx, r8
    call rax
    add rsp, 40
    ret
brass_call_jit_vec_int_ret_void ENDP

brass_call_jit_vec_int_ret_i64 PROC
    sub rsp, 40
    mov rax, rcx
    movdqu xmm0, xmmword ptr [rdx]
    mov rdx, r8
    call rax
    add rsp, 40
    ret
brass_call_jit_vec_int_ret_i64 ENDP

brass_call_jit_vec1_ret_vec PROC
    sub rsp, 40
    mov rax, rcx
    movdqu xmm0, xmmword ptr [rdx]
    call rax
    add rsp, 40
    ret
brass_call_jit_vec1_ret_vec ENDP

brass_call_jit_vec1_ret_f32 PROC
    sub rsp, 40
    mov rax, rcx
    movdqu xmm0, xmmword ptr [rdx]
    call rax
    add rsp, 40
    ret
brass_call_jit_vec1_ret_f32 ENDP

brass_call_jit_vec1_ret_f64 PROC
    sub rsp, 40
    mov rax, rcx
    movdqu xmm0, xmmword ptr [rdx]
    call rax
    add rsp, 40
    ret
brass_call_jit_vec1_ret_f64 ENDP

brass_call_jit_vec1_ret_i64 PROC
    sub rsp, 40
    mov rax, rcx
    movdqu xmm0, xmmword ptr [rdx]
    call rax
    add rsp, 40
    ret
brass_call_jit_vec1_ret_i64 ENDP

brass_call_jit_vec2_ret_vec PROC
    sub rsp, 40
    mov rax, rcx
    movdqu xmm0, xmmword ptr [rdx]
    movdqu xmm1, xmmword ptr [r8]
    call rax
    add rsp, 40
    ret
brass_call_jit_vec2_ret_vec ENDP

brass_call_jit_vec2_ret_f32 PROC
    sub rsp, 40
    mov rax, rcx
    movdqu xmm0, xmmword ptr [rdx]
    movdqu xmm1, xmmword ptr [r8]
    call rax
    add rsp, 40
    ret
brass_call_jit_vec2_ret_f32 ENDP

brass_call_jit_vec2_ret_i64 PROC
    sub rsp, 40
    mov rax, rcx
    movdqu xmm0, xmmword ptr [rdx]
    movdqu xmm1, xmmword ptr [r8]
    call rax
    add rsp, 40
    ret
brass_call_jit_vec2_ret_i64 ENDP

brass_call_jit_v256_0 PROC
    sub rsp, 40
    mov rax, rcx
    call rax
    vmovups ymmword ptr [rdx], ymm0
    vzeroupper
    add rsp, 40
    ret
brass_call_jit_v256_0 ENDP

brass_call_jit_v256_1 PROC
    sub rsp, 40
    mov rax, rcx
    vmovups ymm0, ymmword ptr [r8]
    call rax
    vmovups ymmword ptr [rdx], ymm0
    vzeroupper
    add rsp, 40
    ret
brass_call_jit_v256_1 ENDP

brass_call_jit_v256_2 PROC
    sub rsp, 40
    mov rax, rcx
    vmovups ymm0, ymmword ptr [r8]
    vmovups ymm1, ymmword ptr [r9]
    call rax
    vmovups ymmword ptr [rdx], ymm0
    vzeroupper
    add rsp, 40
    ret
brass_call_jit_v256_2 ENDP

brass_call_jit_v256_3 PROC
    sub rsp, 48
    mov rax, rcx
    vmovups ymm0, ymmword ptr [r8]
    vmovups ymm1, ymmword ptr [r9]
    mov r10, qword ptr [rsp + 88]
    vmovups ymm2, ymmword ptr [r10]
    call rax
    vmovups ymmword ptr [rdx], ymm0
    vzeroupper
    add rsp, 48
    ret
brass_call_jit_v256_3 ENDP

brass_call_jit_v128_3 PROC
    sub rsp, 48
    mov rax, rcx
    movdqu xmm0, xmmword ptr [r8]
    movdqu xmm1, xmmword ptr [r9]
    mov r10, qword ptr [rsp + 88]
    movdqu xmm2, xmmword ptr [r10]
    call rax
    movdqu xmmword ptr [rdx], xmm0
    add rsp, 48
    ret
brass_call_jit_v128_3 ENDP

; void x64_win64_invoke_thunk(const X64Win64InvokeArgs* args, X64Win64InvokeResult* result)
; rcx = args
; rdx = result
x64_win64_invoke_thunk PROC
    push rbp
    mov rbp, rsp
    push rbx
    push rsi
    push rdi
    push r12
    push r13

    mov r12, rcx
    mov r13, rdx

    mov rcx, qword ptr [r12 + 104] ; stack_word_count
    lea rax, [rcx + 4]             ; 4 + stack_word_count
    test rax, 1
    jnz alloc_odd
    inc rax
alloc_odd:
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
    call r11

    mov qword ptr [r13], rax
    movdqu xmmword ptr [r13 + 16], xmm0

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
