.code

EXTERN brass_runtime_gc_safepoint_bridge: PROC
EXTERN brass_runtime_gc_alloc_bridge: PROC

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

END
