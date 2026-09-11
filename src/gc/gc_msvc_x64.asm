.code

EXTERN brass_runtime_gc_safepoint_bridge: PROC
EXTERN brass_runtime_gc_alloc_bridge: PROC
EXTERN brass_throw_impl: PROC

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
brass_throw PROC FRAME
    push rbp
    .pushreg rbp
    mov rbp, rsp
    .setframe rbp, 0
    sub rsp, 128
    .allocstack 128
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
    add rsp, 128
    pop rbp
    ret
brass_throw ENDP

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

END
