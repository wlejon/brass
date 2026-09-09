#include <brass/codegen/jit_exec.hpp>
#include <brass/object/coff_writer.hpp>
#include <brass/object/elf_writer.hpp>
#include <brass/gc/runtime_gc.hpp>
#include <brass/runtime/deopt.hpp>
#include <brass/runtime/resume_table.hpp>
#include <brass/runtime/patcher.hpp>
#include <stdexcept>
#include <cstring>
#include <emmintrin.h>
#include <iostream>

#if defined(_WIN32)
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <sys/mman.h>
#include <unistd.h>
#endif

namespace brass::codegen {

JitMemoryBlock::JitMemoryBlock(size_t size) {
    if (size == 0) return;
    size_t page_aligned = (size + 4095) & ~size_t(4095);
#if defined(_WIN32)
    ptr_ = static_cast<uint8_t*>(VirtualAlloc(nullptr, page_aligned, MEM_COMMIT | MEM_RESERVE, PAGE_EXECUTE_READWRITE));
#else
    ptr_ = static_cast<uint8_t*>(mmap(nullptr, page_aligned, PROT_READ | PROT_WRITE | PROT_EXEC, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0));
    if (ptr_ == MAP_FAILED) ptr_ = nullptr;
#endif
    if (ptr_) size_ = page_aligned;
}

JitMemoryBlock::~JitMemoryBlock() {
    reset();
}

JitMemoryBlock::JitMemoryBlock(JitMemoryBlock&& other) noexcept
    : ptr_(other.ptr_), size_(other.size_) {
    other.ptr_ = nullptr;
    other.size_ = 0;
}

JitMemoryBlock& JitMemoryBlock::operator=(JitMemoryBlock&& other) noexcept {
    if (this != &other) {
        reset();
        ptr_ = other.ptr_;
        size_ = other.size_;
        other.ptr_ = nullptr;
        other.size_ = 0;
    }
    return *this;
}

void JitMemoryBlock::reset() {
    if (ptr_) {
#if defined(_WIN32)
        VirtualFree(ptr_, 0, MEM_RELEASE);
#else
        munmap(ptr_, size_);
#endif
        ptr_ = nullptr;
        size_ = 0;
    }
}

void JitMemoryBlock::make_executable() {
#if defined(_WIN32)
    if (ptr_) {
        DWORD old_protect;
        VirtualProtect(ptr_, size_, PAGE_EXECUTE_READWRITE, &old_protect);
        FlushInstructionCache(GetCurrentProcess(), ptr_, size_);
    }
#else
    if (ptr_) {
        mprotect(ptr_, size_, PROT_READ | PROT_WRITE | PROT_EXEC);
        __builtin___clear_cache(reinterpret_cast<char*>(ptr_), reinterpret_cast<char*>(ptr_ + size_));
    }
#endif
}

void JitMemoryBlock::make_read_write() {
#if defined(_WIN32)
    if (ptr_) {
        DWORD old_protect;
        VirtualProtect(ptr_, size_, PAGE_READWRITE, &old_protect);
    }
#else
    if (ptr_) {
        mprotect(ptr_, size_, PROT_READ | PROT_WRITE);
    }
#endif
}

JitExecutionEngine::JitExecutionEngine(const Target& target)
    : target_(target) {
    register_external_symbol("brass_gc_alloc", reinterpret_cast<void*>(&brass_gc_alloc));
    register_external_symbol("brass_gc_safepoint", reinterpret_cast<void*>(&brass_gc_safepoint));
    register_external_symbol("brass_gc_collect", reinterpret_cast<void*>(&brass_gc_collect));
    register_external_symbol("brass_runtime_gc_safepoint", reinterpret_cast<void*>(&brass_gc_safepoint));
    register_external_symbol("brass_deopt_exit", reinterpret_cast<void*>(&brass_deopt_exit));
    register_external_symbol("brass_get_thread_deopt_frame", reinterpret_cast<void*>(&brass_get_thread_deopt_frame));
    register_external_symbol("brass_set_thread_deopt_frame", reinterpret_cast<void*>(&brass_set_thread_deopt_frame));
    register_external_symbol("brass_patch_const32", reinterpret_cast<void*>(&brass_patch_const32));
    register_external_symbol("brass_patch_const64", reinterpret_cast<void*>(&brass_patch_const64));
    register_external_symbol("brass_patch_call", reinterpret_cast<void*>(&brass_patch_call));
}

JitExecutionEngine::JitExecutionEngine()
    : target_(Target::host()) {
    register_external_symbol("brass_gc_alloc", reinterpret_cast<void*>(&brass_gc_alloc));
    register_external_symbol("brass_gc_safepoint", reinterpret_cast<void*>(&brass_gc_safepoint));
    register_external_symbol("brass_gc_collect", reinterpret_cast<void*>(&brass_gc_collect));
    register_external_symbol("brass_runtime_gc_safepoint", reinterpret_cast<void*>(&brass_gc_safepoint));
    register_external_symbol("brass_deopt_exit", reinterpret_cast<void*>(&brass_deopt_exit));
    register_external_symbol("brass_get_thread_deopt_frame", reinterpret_cast<void*>(&brass_get_thread_deopt_frame));
    register_external_symbol("brass_set_thread_deopt_frame", reinterpret_cast<void*>(&brass_set_thread_deopt_frame));
    register_external_symbol("brass_patch_const32", reinterpret_cast<void*>(&brass_patch_const32));
    register_external_symbol("brass_patch_const64", reinterpret_cast<void*>(&brass_patch_const64));
    register_external_symbol("brass_patch_call", reinterpret_cast<void*>(&brass_patch_call));
}

JitExecutionEngine::~JitExecutionEngine() {
    unregister_seh_tables();
    if (brass_get_active_stack_maps() == &stack_maps_) {
        brass_set_active_stack_maps(nullptr);
    }
}

JitExecutionEngine::JitExecutionEngine(JitExecutionEngine&& other) noexcept
    : target_(other.target_),
      code_mem_(std::move(other.code_mem_)),
      symbol_table_(std::move(other.symbol_table_)),
      external_symbols_(std::move(other.external_symbols_)),
      function_signatures_(std::move(other.function_signatures_)),
      stack_maps_(std::move(other.stack_maps_)),
      pdata_table_(other.pdata_table_),
      pdata_count_(other.pdata_count_),
      code_base_(other.code_base_) {
    other.pdata_table_ = nullptr;
    other.pdata_count_ = 0;
    other.code_base_ = 0;
}

JitExecutionEngine& JitExecutionEngine::operator=(JitExecutionEngine&& other) noexcept {
    if (this != &other) {
        unregister_seh_tables();
        target_ = other.target_;
        code_mem_ = std::move(other.code_mem_);
        symbol_table_ = std::move(other.symbol_table_);
        external_symbols_ = std::move(other.external_symbols_);
        function_signatures_ = std::move(other.function_signatures_);
        stack_maps_ = std::move(other.stack_maps_);
        pdata_table_ = other.pdata_table_;
        pdata_count_ = other.pdata_count_;
        code_base_ = other.code_base_;
        other.pdata_table_ = nullptr;
        other.pdata_count_ = 0;
        other.code_base_ = 0;
    }
    return *this;
}

void JitExecutionEngine::register_external_symbol(std::string_view name, void* address) {
    external_symbols_[std::string(name)] = address;
}

bool JitExecutionEngine::compile_and_load(const Module& mod, size_t code_padding) {
    // Record signatures of functions in the module
    for (const auto* fn : mod.functions()) {
        if (!fn) continue;
        std::vector<Type> params = fn->param_types();
        function_signatures_[std::string(fn->name())] = {fn->return_type(), std::move(params)};
    }

    object::ObjectFile obj = object::compile_module_to_object(mod, target_);
    return load_object(obj, code_padding);
}

bool JitExecutionEngine::load_object(const object::ObjectFile& obj, size_t code_padding) {
    unregister_seh_tables();
    symbol_table_.clear();

    object::ObjectFile working_obj = obj;

    // Generate unwind tables for Windows / Linux
    if (target_.is_windows()) {
        if (!working_obj.functions.empty()) {
            working_obj.get_or_create_section(
                ".xdata",
                object::SectionKind::XData,
                object::SectionFlags::Read | object::SectionFlags::Alloc,
                4
            );
            working_obj.get_or_create_section(
                ".pdata",
                object::SectionKind::PData,
                object::SectionFlags::Read | object::SectionFlags::Alloc,
                4
            );
            auto* pdata_sec = working_obj.get_section(".pdata");
            auto* xdata_sec = working_obj.get_section(".xdata");
            if (pdata_sec && xdata_sec) {
                object::CoffUnwindBuilder::build_unwind_info(working_obj, *pdata_sec, *xdata_sec);
            }
        }
    }

    // Compute memory size and section offsets
    size_t total_size = code_padding;
    std::vector<size_t> sec_offsets(working_obj.sections.size(), 0);

    for (size_t i = 0; i < working_obj.sections.size(); ++i) {
        const auto& sec = working_obj.sections[i];
        if (sec.alignment > 1) {
            total_size = (total_size + (sec.alignment - 1)) & ~(size_t(sec.alignment) - 1);
        }
        sec_offsets[i] = total_size;
        total_size += sec.data.size();
    }

    if (total_size == 0) return true;

    // Reserve extra space for PLT far-call trampolines
    size_t trampoline_capacity = 4096;
    size_t trampoline_offset = (total_size + 15) & ~size_t(15);
    total_size = trampoline_offset + trampoline_capacity;

    // Allocate memory block
    code_mem_ = JitMemoryBlock(total_size);
    if (!code_mem_.is_valid()) {
        return false;
    }

    uint8_t* base_ptr = code_mem_.data();
    uint8_t* trampoline_ptr = base_ptr + trampoline_offset;
    size_t trampoline_used = 0;
    std::unordered_map<std::string, void*> trampolines;

    // Copy section data to memory block
    for (size_t i = 0; i < working_obj.sections.size(); ++i) {
        const auto& sec = working_obj.sections[i];
        if (!sec.data.empty()) {
            std::memcpy(base_ptr + sec_offsets[i], sec.data.data(), sec.data.size());
        }
        symbol_table_[sec.name] = base_ptr + sec_offsets[i];
    }

    // Register symbols
    for (const auto& sym : working_obj.symbols) {
        if (sym.section_index >= 0 && sym.section_index < static_cast<int32_t>(working_obj.sections.size())) {
            symbol_table_[sym.name] = base_ptr + sec_offsets[sym.section_index] + sym.value;
        }
    }
    for (const auto& fn : working_obj.functions) {
        int32_t text_idx = working_obj.get_section_index(".text");
        if (text_idx >= 0) {
            symbol_table_[fn.name] = base_ptr + sec_offsets[text_idx] + fn.text_offset;
        }
    }

    // Resolve relocations
    for (size_t i = 0; i < working_obj.sections.size(); ++i) {
        const auto& sec = working_obj.sections[i];
        uint8_t* sec_runtime_base = base_ptr + sec_offsets[i];

        for (const auto& r : sec.relocations) {
            uint8_t* patch_loc = sec_runtime_base + r.offset;
            void* target_addr = nullptr;

            auto it = symbol_table_.find(r.symbol_name);
            if (it != symbol_table_.end()) {
                target_addr = it->second;
            } else {
                auto ext_it = external_symbols_.find(r.symbol_name);
                if (ext_it != external_symbols_.end()) {
                    target_addr = ext_it->second;
                }
            }

            if (!target_addr) {
                std::cerr << "JIT Error: Unresolved symbol '" << r.symbol_name << "'\n";
                return false;
            }

            switch (r.kind) {
                case object::RelocKind::PCRel32:
                case object::RelocKind::Plt32: {
                    int64_t disp = reinterpret_cast<int64_t>(target_addr) + r.addend - reinterpret_cast<int64_t>(patch_loc);
                    if (disp < INT32_MIN || disp > INT32_MAX) {
                        // Out of 32-bit reach: generate or reuse a 64-bit indirect jump PLT trampoline
                        void* tramp_addr = nullptr;
                        auto tramp_it = trampolines.find(r.symbol_name);
                        if (tramp_it != trampolines.end()) {
                            tramp_addr = tramp_it->second;
                        } else {
                            if (trampoline_used + 16 <= trampoline_capacity) {
                                uint8_t* t = trampoline_ptr + trampoline_used;
                                trampoline_used += 16;
                                // Emit: FF 25 00 00 00 00 (jmp qword ptr [rip + 0]) ; [64-bit target_addr]
                                t[0] = 0xFF; t[1] = 0x25;
                                t[2] = 0x00; t[3] = 0x00; t[4] = 0x00; t[5] = 0x00;
                                *reinterpret_cast<uint64_t*>(t + 6) = reinterpret_cast<uint64_t>(target_addr);
                                trampolines[r.symbol_name] = t;
                                tramp_addr = t;
                            }
                        }
                        if (tramp_addr) {
                            disp = reinterpret_cast<int64_t>(tramp_addr) + r.addend - reinterpret_cast<int64_t>(patch_loc);
                        }
                    }
                    *reinterpret_cast<int32_t*>(patch_loc) = static_cast<int32_t>(disp);
                    break;
                }
                case object::RelocKind::Abs64: {
                    uint64_t val = reinterpret_cast<uint64_t>(target_addr) + static_cast<uint64_t>(r.addend);
                    *reinterpret_cast<uint64_t*>(patch_loc) = val;
                    break;
                }
                case object::RelocKind::Addr32NB:
                case object::RelocKind::SecRel32: {
                    uint32_t rva = static_cast<uint32_t>(reinterpret_cast<uint8_t*>(target_addr) - base_ptr + r.addend);
                    *reinterpret_cast<uint32_t*>(patch_loc) = rva;
                    break;
                }
                case object::RelocKind::Abs32: {
                    uint32_t val = static_cast<uint32_t>(reinterpret_cast<uintptr_t>(target_addr) + static_cast<uintptr_t>(r.addend));
                    *reinterpret_cast<uint32_t*>(patch_loc) = val;
                    break;
                }
            }
        }
    }

    // Windows SEH Registration
    if (target_.is_windows()) {
        register_seh_tables(working_obj, base_ptr);
    }

    // Register and relocate Stack Maps
    stack_maps_ = working_obj.stack_maps;
    int32_t text_idx = working_obj.get_section_index(".text");
    if (text_idx >= 0) {
        text_section_base_ = base_ptr + sec_offsets[text_idx];
        uintptr_t text_base = reinterpret_cast<uintptr_t>(text_section_base_);
        stack_maps_.relocate(text_base);
        for (const auto& fn : working_obj.functions) {
            uintptr_t fn_addr = reinterpret_cast<uintptr_t>(base_ptr + sec_offsets[text_idx] + fn.text_offset);
            stack_maps_.register_function_address(fn.name, fn_addr, static_cast<uint32_t>(fn.text_size));
        }
    } else {
        text_section_base_ = base_ptr;
    }
    brass_set_active_stack_maps(&stack_maps_);

    // Register Resume Tables and Patch Sites
    resume_tables_ = working_obj.resume_tables;
    patch_sites_ = working_obj.patch_sites;

    code_mem_.make_executable();
    return true;
}

void* JitExecutionEngine::get_symbol_address(std::string_view name) const {
    auto it = symbol_table_.find(std::string(name));
    if (it != symbol_table_.end()) {
        return it->second;
    }
    auto ext_it = external_symbols_.find(std::string(name));
    if (ext_it != external_symbols_.end()) {
        return ext_it->second;
    }
    return nullptr;
}

void JitExecutionEngine::register_seh_tables(const object::ObjectFile& obj, uint8_t* base_ptr) {
#if defined(_WIN32) && defined(_M_X64)
    unregister_seh_tables();
    int32_t pdata_idx = obj.get_section_index(".pdata");
    if (pdata_idx != object::SECTION_UNDEF) {
        const auto& pdata_sec = obj.sections[pdata_idx];
        if (!pdata_sec.data.empty()) {
            auto it = symbol_table_.find(".pdata");
            if (it != symbol_table_.end()) {
                pdata_table_ = it->second;
                pdata_count_ = pdata_sec.data.size() / sizeof(RUNTIME_FUNCTION);
                code_base_ = reinterpret_cast<uintptr_t>(base_ptr);
                RtlAddFunctionTable(
                    reinterpret_cast<PRUNTIME_FUNCTION>(pdata_table_),
                    static_cast<DWORD>(pdata_count_),
                    static_cast<DWORD64>(code_base_)
                );
            }
        }
    }
#else
    (void)obj;
    (void)base_ptr;
#endif
}

void JitExecutionEngine::unregister_seh_tables() {
#if defined(_WIN32) && defined(_M_X64)
    if (pdata_table_) {
        RtlDeleteFunctionTable(reinterpret_cast<PRUNTIME_FUNCTION>(pdata_table_));
        pdata_table_ = nullptr;
        pdata_count_ = 0;
        code_base_ = 0;
    }
#endif
}

#if defined(__GNUC__) || defined(__clang__)
inline __m128 call_jit_vec1_ret_vec(void* addr, __m128 a0) {
    register __m128 r_xmm0 asm("xmm0") = a0;
    register __m128 res asm("xmm0");
    asm volatile(
        "subq $32, %%rsp\n\t"
        "call *%1\n\t"
        "addq $32, %%rsp"
        : "=x"(res)
        : "r"(addr), "x"(r_xmm0)
        : "rax", "rcx", "rdx", "r8", "r9", "r10", "r11", "xmm1", "xmm2", "xmm3", "xmm4", "xmm5", "memory"
    );
    return res;
}

inline float call_jit_vec1_ret_f32(void* addr, __m128 a0) {
    register __m128 r_xmm0 asm("xmm0") = a0;
    register float res asm("xmm0");
    asm volatile(
        "subq $32, %%rsp\n\t"
        "call *%1\n\t"
        "addq $32, %%rsp"
        : "=x"(res)
        : "r"(addr), "x"(r_xmm0)
        : "rax", "rcx", "rdx", "r8", "r9", "r10", "r11", "xmm1", "xmm2", "xmm3", "xmm4", "xmm5", "memory"
    );
    return res;
}

inline double call_jit_vec1_ret_f64(void* addr, __m128 a0) {
    register __m128 r_xmm0 asm("xmm0") = a0;
    register double res asm("xmm0");
    asm volatile(
        "subq $32, %%rsp\n\t"
        "call *%1\n\t"
        "addq $32, %%rsp"
        : "=x"(res)
        : "r"(addr), "x"(r_xmm0)
        : "rax", "rcx", "rdx", "r8", "r9", "r10", "r11", "xmm1", "xmm2", "xmm3", "xmm4", "xmm5", "memory"
    );
    return res;
}

inline int64_t call_jit_vec1_ret_i64(void* addr, __m128 a0) {
    register __m128 r_xmm0 asm("xmm0") = a0;
    register int64_t res asm("rax");
    asm volatile(
        "subq $32, %%rsp\n\t"
        "call *%1\n\t"
        "addq $32, %%rsp"
        : "=r"(res)
        : "r"(addr), "x"(r_xmm0)
        : "rcx", "rdx", "r8", "r9", "r10", "r11", "xmm1", "xmm2", "xmm3", "xmm4", "xmm5", "memory"
    );
    return res;
}

inline __m128 call_jit_vec2_ret_vec(void* addr, __m128 a0, __m128 a1) {
    register __m128 r_xmm0 asm("xmm0") = a0;
    register __m128 r_xmm1 asm("xmm1") = a1;
    register __m128 res asm("xmm0");
    asm volatile(
        "subq $32, %%rsp\n\t"
        "call *%1\n\t"
        "addq $32, %%rsp"
        : "=x"(res)
        : "r"(addr), "x"(r_xmm0), "x"(r_xmm1)
        : "rax", "rcx", "rdx", "r8", "r9", "r10", "r11", "xmm2", "xmm3", "xmm4", "xmm5", "memory"
    );
    return res;
}

inline float call_jit_vec2_ret_f32(void* addr, __m128 a0, __m128 a1) {
    register __m128 r_xmm0 asm("xmm0") = a0;
    register __m128 r_xmm1 asm("xmm1") = a1;
    register float res asm("xmm0");
    asm volatile(
        "subq $32, %%rsp\n\t"
        "call *%1\n\t"
        "addq $32, %%rsp"
        : "=x"(res)
        : "r"(addr), "x"(r_xmm0), "x"(r_xmm1)
        : "rax", "rcx", "rdx", "r8", "r9", "r10", "r11", "xmm2", "xmm3", "xmm4", "xmm5", "memory"
    );
    return res;
}

inline int64_t call_jit_vec2_ret_i64(void* addr, __m128 a0, __m128 a1) {
    register __m128 r_xmm0 asm("xmm0") = a0;
    register __m128 r_xmm1 asm("xmm1") = a1;
    register int64_t res asm("rax");
    asm volatile(
        "subq $32, %%rsp\n\t"
        "call *%1\n\t"
        "addq $32, %%rsp"
        : "=r"(res)
        : "r"(addr), "x"(r_xmm0), "x"(r_xmm1)
        : "rcx", "rdx", "r8", "r9", "r10", "r11", "xmm2", "xmm3", "xmm4", "xmm5", "memory"
    );
    return res;
}

inline void call_jit_vec_int_ret_void(void* addr, __m128 a0, int64_t a1) {
    register __m128 r_xmm0 asm("xmm0") = a0;
    register int64_t r_rdx asm("rdx") = a1;
    asm volatile(
        "subq $32, %%rsp\n\t"
        "call *%0\n\t"
        "addq $32, %%rsp"
        :
        : "r"(addr), "x"(r_xmm0), "r"(r_rdx)
        : "rax", "rcx", "r8", "r9", "r10", "r11", "xmm1", "xmm2", "xmm3", "xmm4", "xmm5", "memory"
    );
}

inline __m128 call_jit_vec_int_ret_vec(void* addr, __m128 a0, int64_t a1) {
    register __m128 r_xmm0 asm("xmm0") = a0;
    register int64_t r_rdx asm("rdx") = a1;
    register __m128 res asm("xmm0");
    asm volatile(
        "subq $32, %%rsp\n\t"
        "call *%1\n\t"
        "addq $32, %%rsp"
        : "=x"(res)
        : "r"(addr), "x"(r_xmm0), "r"(r_rdx)
        : "rax", "rcx", "r8", "r9", "r10", "r11", "xmm1", "xmm2", "xmm3", "xmm4", "xmm5", "memory"
    );
    return res;
}

inline int64_t call_jit_vec_int_ret_i64(void* addr, __m128 a0, int64_t a1) {
    register __m128 r_xmm0 asm("xmm0") = a0;
    register int64_t r_rdx asm("rdx") = a1;
    register int64_t res asm("rax");
    asm volatile(
        "subq $32, %%rsp\n\t"
        "call *%1\n\t"
        "addq $32, %%rsp"
        : "=r"(res)
        : "r"(addr), "x"(r_xmm0), "r"(r_rdx)
        : "rcx", "r8", "r9", "r10", "r11", "xmm1", "xmm2", "xmm3", "xmm4", "xmm5", "memory"
    );
    return res;
}

inline void call_jit_int_vec_ret_void(void* addr, int64_t a0, __m128 a1) {
    register int64_t r_rcx asm("rcx") = a0;
    register __m128 r_xmm1 asm("xmm1") = a1;
    asm volatile(
        "subq $32, %%rsp\n\t"
        "call *%0\n\t"
        "addq $32, %%rsp"
        :
        : "r"(addr), "r"(r_rcx), "x"(r_xmm1)
        : "rax", "rdx", "r8", "r9", "r10", "r11", "xmm0", "xmm2", "xmm3", "xmm4", "xmm5", "memory"
    );
}

inline __m128 call_jit_int_vec_ret_vec(void* addr, int64_t a0, __m128 a1) {
    register int64_t r_rcx asm("rcx") = a0;
    register __m128 r_xmm1 asm("xmm1") = a1;
    register __m128 res asm("xmm0");
    asm volatile(
        "subq $32, %%rsp\n\t"
        "call *%1\n\t"
        "addq $32, %%rsp"
        : "=x"(res)
        : "r"(addr), "r"(r_rcx), "x"(r_xmm1)
        : "rax", "rdx", "r8", "r9", "r10", "r11", "xmm2", "xmm3", "xmm4", "xmm5", "memory"
    );
    return res;
}

inline int64_t call_jit_int_vec_ret_i64(void* addr, int64_t a0, __m128 a1) {
    register int64_t r_rcx asm("rcx") = a0;
    register __m128 r_xmm1 asm("xmm1") = a1;
    register int64_t res asm("rax");
    asm volatile(
        "subq $32, %%rsp\n\t"
        "call *%1\n\t"
        "addq $32, %%rsp"
        : "=r"(res)
        : "r"(addr), "r"(r_rcx), "x"(r_xmm1)
        : "rdx", "r8", "r9", "r10", "r11", "xmm0", "xmm2", "xmm3", "xmm4", "xmm5", "memory"
    );
    return res;
}
#elif defined(_MSC_VER)
inline __m128 call_jit_vec1_ret_vec(void* addr, __m128 a0) {
    return reinterpret_cast<__m128(__vectorcall*)(__m128)>(addr)(a0);
}
inline float call_jit_vec1_ret_f32(void* addr, __m128 a0) {
    return reinterpret_cast<float(__vectorcall*)(__m128)>(addr)(a0);
}
inline double call_jit_vec1_ret_f64(void* addr, __m128 a0) {
    return reinterpret_cast<double(__vectorcall*)(__m128)>(addr)(a0);
}
inline int64_t call_jit_vec1_ret_i64(void* addr, __m128 a0) {
    return reinterpret_cast<int64_t(__vectorcall*)(__m128)>(addr)(a0);
}
inline __m128 call_jit_vec2_ret_vec(void* addr, __m128 a0, __m128 a1) {
    return reinterpret_cast<__m128(__vectorcall*)(__m128, __m128)>(addr)(a0, a1);
}
inline float call_jit_vec2_ret_f32(void* addr, __m128 a0, __m128 a1) {
    return reinterpret_cast<float(__vectorcall*)(__m128, __m128)>(addr)(a0, a1);
}
inline int64_t call_jit_vec2_ret_i64(void* addr, __m128 a0, __m128 a1) {
    return reinterpret_cast<int64_t(__vectorcall*)(__m128, __m128)>(addr)(a0, a1);
}
inline void call_jit_vec_int_ret_void(void* addr, __m128 a0, int64_t a1) {
    reinterpret_cast<void(__vectorcall*)(__m128, int64_t)>(addr)(a0, a1);
}
inline __m128 call_jit_vec_int_ret_vec(void* addr, __m128 a0, int64_t a1) {
    return reinterpret_cast<__m128(__vectorcall*)(__m128, int64_t)>(addr)(a0, a1);
}
inline int64_t call_jit_vec_int_ret_i64(void* addr, __m128 a0, int64_t a1) {
    return reinterpret_cast<int64_t(__vectorcall*)(__m128, int64_t)>(addr)(a0, a1);
}
inline void call_jit_int_vec_ret_void(void* addr, int64_t a0, __m128 a1) {
    reinterpret_cast<void(__vectorcall*)(int64_t, __m128)>(addr)(a0, a1);
}
inline __m128 call_jit_int_vec_ret_vec(void* addr, int64_t a0, __m128 a1) {
    return reinterpret_cast<__m128(__vectorcall*)(int64_t, __m128)>(addr)(a0, a1);
}
inline int64_t call_jit_int_vec_ret_i64(void* addr, int64_t a0, __m128 a1) {
    return reinterpret_cast<int64_t(__vectorcall*)(int64_t, __m128)>(addr)(a0, a1);
}
#endif

RuntimeValue JitExecutionEngine::invoke(std::string_view name) {
    return invoke(name, {});
}

RuntimeValue JitExecutionEngine::invoke(std::string_view name, const std::vector<RuntimeValue>& args) {
    void* addr = get_symbol_address(name);
    if (!addr) {
        throw std::runtime_error("JIT Error: Function '" + std::string(name) + "' not found");
    }

    Type ret_type = Type::i64();
    auto sig_it = function_signatures_.find(std::string(name));
    if (sig_it != function_signatures_.end()) {
        ret_type = sig_it->second.first;
    }

    // Helper lambdas to extract typed values
    auto get_int = [&](size_t idx) -> int64_t {
        if (idx >= args.size()) return 0;
        return args[idx].as_i64();
    };

    auto get_float = [&](size_t idx) -> double {
        if (idx >= args.size()) return 0.0;
        return args[idx].as_f64();
    };

    // Check if arguments or return value contain floats
    bool has_float_arg = false;
    for (const auto& a : args) {
        if (a.is_f64()) has_float_arg = true;
    }

    // 0 arguments
    if (args.empty()) {
        if (ret_type.is_void()) {
            reinterpret_cast<void(*)()>(addr)();
            return RuntimeValue::from_void();
        } else if (ret_type.is_vector()) {
            __m128 r = reinterpret_cast<__m128(*)()>(addr)();
            alignas(16) uint8_t b[16];
            std::memcpy(b, &r, 16);
            return RuntimeValue::from_v128(ret_type, b);
        } else if (ret_type.is_float()) {
            if (ret_type.kind() == TypeKind::F32) {
                float r = reinterpret_cast<float(*)()>(addr)();
                return RuntimeValue::from_f32(r);
            }
            double r = reinterpret_cast<double(*)()>(addr)();
            return RuntimeValue::from_f64(r);
        } else if (ret_type.kind() == TypeKind::I32) {
            int32_t r = reinterpret_cast<int32_t(*)()>(addr)();
            return RuntimeValue::from_i32(r);
        } else {
            int64_t r = reinterpret_cast<int64_t(*)()>(addr)();
            return RuntimeValue::from_i64(r);
        }
    }

    // 1 argument
    if (args.size() == 1) {
        if (args[0].is_vector()) {
            __m128 a0;
            std::memcpy(&a0, args[0].v128_bytes(), 16);
            if (ret_type.is_vector()) {
                __m128 r = call_jit_vec1_ret_vec(addr, a0);
                alignas(16) uint8_t b[16];
                std::memcpy(b, &r, 16);
                return RuntimeValue::from_v128(ret_type, b);
            } else if (ret_type.is_float()) {
                if (ret_type.kind() == TypeKind::F32) {
                    float r = call_jit_vec1_ret_f32(addr, a0);
                    return RuntimeValue::from_f32(r);
                }
                double r = call_jit_vec1_ret_f64(addr, a0);
                return RuntimeValue::from_f64(r);
            } else {
                int64_t r = call_jit_vec1_ret_i64(addr, a0);
                if (ret_type.kind() == TypeKind::I32) return RuntimeValue::from_i32(static_cast<int32_t>(r));
                return RuntimeValue::from_i64(r);
            }
        } else if (args[0].is_f64()) {
            if (ret_type.is_float()) {
                double r = reinterpret_cast<double(*)(double)>(addr)(get_float(0));
                return RuntimeValue::from_f64(r);
            } else {
                int64_t r = reinterpret_cast<int64_t(*)(double)>(addr)(get_float(0));
                return RuntimeValue::from_i64(r);
            }
        } else {
            if (ret_type.is_vector()) {
                __m128 r = reinterpret_cast<__m128(*)(int64_t)>(addr)(get_int(0));
                alignas(16) uint8_t b[16];
                std::memcpy(b, &r, 16);
                return RuntimeValue::from_v128(ret_type, b);
            } else if (ret_type.is_float()) {
                double r = reinterpret_cast<double(*)(int64_t)>(addr)(get_int(0));
                return RuntimeValue::from_f64(r);
            } else {
                int64_t r = reinterpret_cast<int64_t(*)(int64_t)>(addr)(get_int(0));
                return RuntimeValue::from_i64(r);
            }
        }
    }

    // 2 arguments
    if (args.size() == 2) {
        if (args[0].is_vector() && args[1].is_vector()) {
            __m128 a0, a1;
            std::memcpy(&a0, args[0].v128_bytes(), 16);
            std::memcpy(&a1, args[1].v128_bytes(), 16);
            if (ret_type.is_vector()) {
                __m128 r = call_jit_vec2_ret_vec(addr, a0, a1);
                alignas(16) uint8_t b[16];
                std::memcpy(b, &r, 16);
                return RuntimeValue::from_v128(ret_type, b);
            } else if (ret_type.is_float()) {
                if (ret_type.kind() == TypeKind::F32) {
                    float r = call_jit_vec2_ret_f32(addr, a0, a1);
                    return RuntimeValue::from_f32(r);
                }
                double r = reinterpret_cast<double(*)(__m128, __m128)>(addr)(a0, a1);
                return RuntimeValue::from_f64(r);
            } else {
                int64_t r = call_jit_vec2_ret_i64(addr, a0, a1);
                return RuntimeValue::from_i64(r);
            }
        } else if (args[0].is_vector() && !args[1].is_vector()) {
            __m128 a0;
            std::memcpy(&a0, args[0].v128_bytes(), 16);
            if (ret_type.is_void()) {
                call_jit_vec_int_ret_void(addr, a0, get_int(1));
                return RuntimeValue::from_void();
            } else if (ret_type.is_vector()) {
                __m128 r = call_jit_vec_int_ret_vec(addr, a0, get_int(1));
                alignas(16) uint8_t b[16];
                std::memcpy(b, &r, 16);
                return RuntimeValue::from_v128(ret_type, b);
            } else {
                int64_t r = call_jit_vec_int_ret_i64(addr, a0, get_int(1));
                return RuntimeValue::from_i64(r);
            }
        } else if (!args[0].is_vector() && args[1].is_vector()) {
            __m128 a1;
            std::memcpy(&a1, args[1].v128_bytes(), 16);
            if (ret_type.is_void()) {
                call_jit_int_vec_ret_void(addr, get_int(0), a1);
                return RuntimeValue::from_void();
            } else if (ret_type.is_vector()) {
                __m128 r = call_jit_int_vec_ret_vec(addr, get_int(0), a1);
                alignas(16) uint8_t b[16];
                std::memcpy(b, &r, 16);
                return RuntimeValue::from_v128(ret_type, b);
            } else {
                int64_t r = call_jit_int_vec_ret_i64(addr, get_int(0), a1);
                return RuntimeValue::from_i64(r);
            }
        } else if (args[0].is_f64() && args[1].is_f64()) {
            if (ret_type.is_float()) {
                double r = reinterpret_cast<double(*)(double, double)>(addr)(get_float(0), get_float(1));
                return RuntimeValue::from_f64(r);
            } else {
                int64_t r = reinterpret_cast<int64_t(*)(double, double)>(addr)(get_float(0), get_float(1));
                return RuntimeValue::from_i64(r);
            }
        } else {
            if (ret_type.is_vector()) {
                __m128 r = reinterpret_cast<__m128(*)(int64_t, int64_t)>(addr)(get_int(0), get_int(1));
                alignas(16) uint8_t b[16];
                std::memcpy(b, &r, 16);
                return RuntimeValue::from_v128(ret_type, b);
            } else if (ret_type.is_float()) {
                double r = reinterpret_cast<double(*)(int64_t, int64_t)>(addr)(get_int(0), get_int(1));
                return RuntimeValue::from_f64(r);
            } else {
                int64_t r = reinterpret_cast<int64_t(*)(int64_t, int64_t)>(addr)(get_int(0), get_int(1));
                return RuntimeValue::from_i64(r);
            }
        }
    }

    // 3 arguments
    if (args.size() == 3) {
        if (args[0].is_f64() && args[1].is_f64() && !args[2].is_f64()) {
            if (ret_type.is_float()) {
                double r = reinterpret_cast<double(*)(double, double, int64_t)>(addr)(get_float(0), get_float(1), get_int(2));
                return RuntimeValue::from_f64(r);
            } else {
                int64_t r = reinterpret_cast<int64_t(*)(double, double, int64_t)>(addr)(get_float(0), get_float(1), get_int(2));
                return RuntimeValue::from_i64(r);
            }
        } else if (args[0].is_f64() && args[1].is_f64() && args[2].is_f64()) {
            if (ret_type.is_float()) {
                double r = reinterpret_cast<double(*)(double, double, double)>(addr)(get_float(0), get_float(1), get_float(2));
                return RuntimeValue::from_f64(r);
            } else {
                int64_t r = reinterpret_cast<int64_t(*)(double, double, double)>(addr)(get_float(0), get_float(1), get_float(2));
                return RuntimeValue::from_i64(r);
            }
        }
    }

    // 3 to 8 arguments (Integer / Pointer paths)
    if (!has_float_arg) {
        int64_t a0 = get_int(0), a1 = get_int(1), a2 = get_int(2), a3 = get_int(3);
        int64_t a4 = get_int(4), a5 = get_int(5), a6 = get_int(6), a7 = get_int(7);

        if (ret_type.is_float()) {
            auto fn8_f = reinterpret_cast<double(*)(int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t)>(addr);
            double r = fn8_f(a0, a1, a2, a3, a4, a5, a6, a7);
            return RuntimeValue::from_f64(r);
        }

        auto fn8 = reinterpret_cast<int64_t(*)(int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t)>(addr);
        int64_t r = fn8(a0, a1, a2, a3, a4, a5, a6, a7);

        if (ret_type.is_void()) return RuntimeValue::from_void();
        if (ret_type.kind() == TypeKind::I32) return RuntimeValue::from_i32(static_cast<int32_t>(r));
        if (ret_type.is_pointer()) return RuntimeValue::from_ptr(static_cast<uintptr_t>(r));
        if (ret_type.is_gcref()) return RuntimeValue::from_gcref(static_cast<uintptr_t>(r));
        return RuntimeValue::from_i64(r);
    } else {
        // Multi-arg Float path
        double f0 = get_float(0), f1 = get_float(1), f2 = get_float(2), f3 = get_float(3);
        double f4 = get_float(4), f5 = get_float(5), f6 = get_float(6), f7 = get_float(7);

        auto fn_f8 = reinterpret_cast<double(*)(double, double, double, double, double, double, double, double)>(addr);
        double r = fn_f8(f0, f1, f2, f3, f4, f5, f6, f7);
        return RuntimeValue::from_f64(r);
    }
}

const runtime::FunctionResumeTable* JitExecutionEngine::get_resume_table(std::string_view fn_name) const noexcept {
    return resume_tables_.get_table(fn_name);
}

void* JitExecutionEngine::get_resume_target_address(std::string_view fn_name, uint32_t resume_id) const {
    void* fn_addr = get_symbol_address(fn_name);
    if (!fn_addr) return nullptr;
    const auto* table = get_resume_table(fn_name);
    if (!table) return nullptr;
    return table->get_target_address(fn_addr, resume_id);
}

bool JitExecutionEngine::patch_const32(std::string_view site_name, int32_t new_val) {
    if (!text_section_base_) return false;
    bool ok = patch_sites_.patch_const32(text_section_base_, site_name, new_val);
#if defined(_WIN32)
    if (ok && code_mem_.data()) {
        FlushInstructionCache(GetCurrentProcess(), code_mem_.data(), code_mem_.size());
    }
#endif
    return ok;
}

bool JitExecutionEngine::patch_const64(std::string_view site_name, int64_t new_val) {
    if (!text_section_base_) return false;
    bool ok = patch_sites_.patch_const64(text_section_base_, site_name, new_val);
#if defined(_WIN32)
    if (ok && code_mem_.data()) {
        FlushInstructionCache(GetCurrentProcess(), code_mem_.data(), code_mem_.size());
    }
#endif
    return ok;
}

bool JitExecutionEngine::patch_call(std::string_view site_name, const void* new_target) {
    if (!text_section_base_) return false;
    bool ok = patch_sites_.patch_call(text_section_base_, site_name, new_target);
#if defined(_WIN32)
    if (ok && code_mem_.data()) {
        FlushInstructionCache(GetCurrentProcess(), code_mem_.data(), code_mem_.size());
    }
#endif
    return ok;
}

bool JitExecutionEngine::patch_call(std::string_view site_name, std::string_view new_target_fn) {
    void* target_addr = get_symbol_address(new_target_fn);
    if (!target_addr) return false;
    return patch_call(site_name, target_addr);
}

RuntimeValue JitExecutionEngine::resume(std::string_view name, uint32_t resume_id) {
    std::vector<RuntimeValue> empty_args;
    return resume(name, resume_id, empty_args);
}

RuntimeValue JitExecutionEngine::resume(std::string_view name, uint32_t resume_id, const std::vector<RuntimeValue>& args) {
    std::vector<RuntimeValue> full_args;
    full_args.reserve(args.size() + 1);
    full_args.push_back(RuntimeValue::from_i32(static_cast<int32_t>(resume_id)));
    for (const auto& a : args) {
        full_args.push_back(a);
    }
    return invoke(name, full_args);
}

} // namespace brass::codegen
