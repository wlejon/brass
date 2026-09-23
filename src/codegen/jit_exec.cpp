#include <brass/codegen/jit_exec.hpp>
#include <brass/object/coff_writer.hpp>
#include <brass/object/elf_writer.hpp>
#include <brass/object/aarch64_reloc.hpp>
#include <brass/gc/runtime_gc.hpp>
#include <brass/runtime/deopt.hpp>
#include <brass/runtime/resume_table.hpp>
#include <brass/runtime/patcher.hpp>
#include <brass/runtime/exception.hpp>
#include <brass/runtime/coroutine.hpp>
#include <brass/runtime/multi_tier_pipeline.hpp>
#include <brass/runtime/parallel_runtime.hpp>
#include "../il_translator/il_runtime.hpp"
#include <algorithm>
#include <mutex>
#include <vector>
#include <stdexcept>
#include <cstring>
#include <iostream>
#include <unordered_set>

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
#if defined(__APPLE__)
#include <pthread.h>
#include <libkern/OSCacheControl.h>
#endif
#endif

namespace brass::codegen {

static void* brass_exit_stub(uint32_t, const uint64_t*) {
    return nullptr;
}

JitExecutionEngine::JitExecutionEngine(const Target& target)
    : target_(target) {
    register_external_symbol("@exit_stub", reinterpret_cast<void*>(&brass_exit_stub));
    register_external_symbol("exit_stub", reinterpret_cast<void*>(&brass_exit_stub));
    register_external_symbol("brass_gc_alloc", reinterpret_cast<void*>(&brass_gc_alloc));
    register_external_symbol("brass_gc_safepoint", reinterpret_cast<void*>(&brass_gc_safepoint));
    register_external_symbol("brass_gc_collect", reinterpret_cast<void*>(&brass_gc_collect));
    register_external_symbol("brass_runtime_gc_safepoint", reinterpret_cast<void*>(&brass_gc_safepoint));
    register_external_symbol("brass_deopt_exit", reinterpret_cast<void*>(&brass_deopt_exit));
    register_external_symbol("brass_deopt_exit_record", reinterpret_cast<void*>(&brass_deopt_exit_record));
    register_external_symbol("brass_get_thread_deopt_slots", reinterpret_cast<void*>(&brass_get_thread_deopt_slots));
    register_external_symbol("brass_get_thread_deopt_frame", reinterpret_cast<void*>(&brass_get_thread_deopt_frame));
    register_external_symbol("brass_set_thread_deopt_frame", reinterpret_cast<void*>(&brass_set_thread_deopt_frame));
    register_external_symbol("brass_patch_const32", reinterpret_cast<void*>(&brass_patch_const32));
    register_external_symbol("brass_patch_const64", reinterpret_cast<void*>(&brass_patch_const64));
    register_external_symbol("brass_patch_call", reinterpret_cast<void*>(&brass_patch_call));
    register_external_symbol("brass_throw", reinterpret_cast<void*>(&runtime::brass_throw));
    register_external_symbol("brass_rethrow", reinterpret_cast<void*>(&runtime::brass_rethrow));
#if defined(_WIN32)
    register_external_symbol("brass_seh_personality", reinterpret_cast<void*>(&runtime::brass_seh_personality));
#endif
    register_external_symbol("brass_coro_create", reinterpret_cast<void*>(&brass_coro_create));
    register_external_symbol("brass_coro_resume", reinterpret_cast<void*>(&brass_coro_resume));
    register_external_symbol("brass_coro_is_done", reinterpret_cast<void*>(&brass_coro_is_done));
    register_external_symbol("brass_coro_destroy", reinterpret_cast<void*>(&brass_coro_destroy));
    register_external_symbol("bronze_iter_open", reinterpret_cast<void*>(&runtime::bronze_iter_open));
    register_external_symbol("bronze_iter_step", reinterpret_cast<void*>(&runtime::bronze_iter_step));
    register_external_symbol("bronze_create_async_machine", reinterpret_cast<void*>(&runtime::bronze_create_async_machine));
    register_external_symbol("bronze_async_start", reinterpret_cast<void*>(&runtime::bronze_async_start));
    register_external_symbol("bronze_async_await", reinterpret_cast<void*>(&runtime::bronze_async_await));
    register_external_symbol("brass_gc_write_barrier", reinterpret_cast<void*>(&brass_default_gc_write_barrier));
    register_external_symbol("brass_gc_card_table_base", reinterpret_cast<void*>(&brass_gc_card_table_base));
    register_external_symbol("brass_gc_heap_base", reinterpret_cast<void*>(&brass_gc_heap_base));
    register_external_symbol("brass_parallel_for", reinterpret_cast<void*>(&brass_parallel_for));
    register_external_symbol("brass_set_parallel_workers", reinterpret_cast<void*>(&brass_set_parallel_workers));
    register_external_symbol("brass_get_parallel_workers", reinterpret_cast<void*>(&brass_get_parallel_workers));
    register_external_symbol("brass_parallel_reduce_i64", reinterpret_cast<void*>(&brass_parallel_reduce_i64));
    register_external_symbol("brass_parallel_reduce_f64", reinterpret_cast<void*>(&brass_parallel_reduce_f64));
    register_external_symbol("brass_parallel_alloc_context", reinterpret_cast<void*>(&brass_parallel_alloc_context));
    register_external_symbol("brass_parallel_free_context", reinterpret_cast<void*>(&brass_parallel_free_context));
}

JitExecutionEngine::JitExecutionEngine()
    : target_(Target::host()) {
    register_external_symbol("@exit_stub", reinterpret_cast<void*>(&brass_exit_stub));
    register_external_symbol("exit_stub", reinterpret_cast<void*>(&brass_exit_stub));
    register_external_symbol("brass_gc_alloc", reinterpret_cast<void*>(&brass_gc_alloc));
    register_external_symbol("brass_gc_safepoint", reinterpret_cast<void*>(&brass_gc_safepoint));
    register_external_symbol("brass_gc_collect", reinterpret_cast<void*>(&brass_gc_collect));
    register_external_symbol("brass_runtime_gc_safepoint", reinterpret_cast<void*>(&brass_gc_safepoint));
    register_external_symbol("brass_deopt_exit", reinterpret_cast<void*>(&brass_deopt_exit));
    register_external_symbol("brass_deopt_exit_record", reinterpret_cast<void*>(&brass_deopt_exit_record));
    register_external_symbol("brass_get_thread_deopt_slots", reinterpret_cast<void*>(&brass_get_thread_deopt_slots));
    register_external_symbol("brass_get_thread_deopt_frame", reinterpret_cast<void*>(&brass_get_thread_deopt_frame));
    register_external_symbol("brass_set_thread_deopt_frame", reinterpret_cast<void*>(&brass_set_thread_deopt_frame));
    register_external_symbol("brass_patch_const32", reinterpret_cast<void*>(&brass_patch_const32));
    register_external_symbol("brass_patch_const64", reinterpret_cast<void*>(&brass_patch_const64));
    register_external_symbol("brass_patch_call", reinterpret_cast<void*>(&brass_patch_call));
    register_external_symbol("brass_throw", reinterpret_cast<void*>(&runtime::brass_throw));
    register_external_symbol("brass_rethrow", reinterpret_cast<void*>(&runtime::brass_rethrow));
#if defined(_WIN32)
    register_external_symbol("brass_seh_personality", reinterpret_cast<void*>(&runtime::brass_seh_personality));
#endif
    register_external_symbol("brass_coro_create", reinterpret_cast<void*>(&brass_coro_create));
    register_external_symbol("brass_coro_resume", reinterpret_cast<void*>(&brass_coro_resume));
    register_external_symbol("brass_coro_is_done", reinterpret_cast<void*>(&brass_coro_is_done));
    register_external_symbol("brass_coro_destroy", reinterpret_cast<void*>(&brass_coro_destroy));
    register_external_symbol("bronze_iter_open", reinterpret_cast<void*>(&runtime::bronze_iter_open));
    register_external_symbol("bronze_iter_step", reinterpret_cast<void*>(&runtime::bronze_iter_step));
    register_external_symbol("bronze_create_async_machine", reinterpret_cast<void*>(&runtime::bronze_create_async_machine));
    register_external_symbol("bronze_async_start", reinterpret_cast<void*>(&runtime::bronze_async_start));
    register_external_symbol("bronze_async_await", reinterpret_cast<void*>(&runtime::bronze_async_await));
    register_external_symbol("brass_gc_write_barrier", reinterpret_cast<void*>(&brass_default_gc_write_barrier));
    register_external_symbol("brass_gc_card_table_base", reinterpret_cast<void*>(&brass_gc_card_table_base));
    register_external_symbol("brass_gc_heap_base", reinterpret_cast<void*>(&brass_gc_heap_base));
    register_external_symbol("brass_parallel_for", reinterpret_cast<void*>(&brass_parallel_for));
    register_external_symbol("brass_set_parallel_workers", reinterpret_cast<void*>(&brass_set_parallel_workers));
    register_external_symbol("brass_get_parallel_workers", reinterpret_cast<void*>(&brass_get_parallel_workers));
    register_external_symbol("brass_parallel_reduce_i64", reinterpret_cast<void*>(&brass_parallel_reduce_i64));
    register_external_symbol("brass_parallel_reduce_f64", reinterpret_cast<void*>(&brass_parallel_reduce_f64));
    register_external_symbol("brass_parallel_alloc_context", reinterpret_cast<void*>(&brass_parallel_alloc_context));
    register_external_symbol("brass_parallel_free_context", reinterpret_cast<void*>(&brass_parallel_free_context));
}

JitExecutionEngine::~JitExecutionEngine() {
    unregister_seh_tables();
    unregister_eh_frame();
    for (uintptr_t fn_addr : registered_exception_fns_) {
        runtime::get_global_exception_registry().unregister_function_mapping(fn_addr);
    }
    registered_exception_fns_.clear();
    if (brass_get_active_stack_maps() == &stack_maps_) {
        brass_set_active_stack_maps(nullptr);
    }
    if (il::get_active_jit() == this) {
        il::unregister_all_runtime_symbols();
    }
}

JitExecutionEngine::JitExecutionEngine(JitExecutionEngine&& other) noexcept
    : target_(other.target_),
      code_mem_(std::move(other.code_mem_)),
      data_mem_(std::move(other.data_mem_)),
      symbol_table_(std::move(other.symbol_table_)),
      external_symbols_(std::move(other.external_symbols_)),
      function_signatures_(std::move(other.function_signatures_)),
      stack_maps_(std::move(other.stack_maps_)),
      osr_entry_offsets_(std::move(other.osr_entry_offsets_)),
      pdata_table_(other.pdata_table_),
      pdata_count_(other.pdata_count_),
      code_base_(other.code_base_),
      registered_fdes_(std::move(other.registered_fdes_)),
      sched_opts_(other.sched_opts_) {
    other.registered_fdes_.clear();
    other.pdata_table_ = nullptr;
    other.pdata_count_ = 0;
    other.code_base_ = 0;
}

JitExecutionEngine& JitExecutionEngine::operator=(JitExecutionEngine&& other) noexcept {
    if (this != &other) {
        unregister_seh_tables();
        unregister_eh_frame();
        registered_fdes_ = std::move(other.registered_fdes_);
        other.registered_fdes_.clear();
        target_ = other.target_;
        code_mem_ = std::move(other.code_mem_);
        data_mem_ = std::move(other.data_mem_);
        symbol_table_ = std::move(other.symbol_table_);
        external_symbols_ = std::move(other.external_symbols_);
        function_signatures_ = std::move(other.function_signatures_);
        stack_maps_ = std::move(other.stack_maps_);
        osr_entry_offsets_ = std::move(other.osr_entry_offsets_);
        pdata_table_ = other.pdata_table_;
        pdata_count_ = other.pdata_count_;
        code_base_ = other.code_base_;
        sched_opts_ = other.sched_opts_;
        other.pdata_table_ = nullptr;
        other.pdata_count_ = 0;
        other.code_base_ = 0;
    }
    return *this;
}

void JitExecutionEngine::register_external_symbol(std::string_view name, void* address) {
    external_symbols_[std::string(name)] = address;
}

void JitExecutionEngine::register_function_signature(std::string_view name, Type ret_type, std::vector<Type> param_types) {
    function_signatures_[std::string(name)] = {ret_type, std::move(param_types)};
}

bool JitExecutionEngine::compile_and_load(const Module& mod, size_t code_padding) {
    // Record signatures of functions in the module
    for (const auto* fn : mod.functions()) {
        if (!fn) continue;
        std::vector<Type> params = fn->param_types();
        function_signatures_[std::string(fn->name())] = {fn->return_type(), std::move(params)};
    }

    object::ObjectFile obj = object::compile_module_to_object(mod, target_, sched_opts_);
    return load_object(obj, code_padding);
}

bool JitExecutionEngine::compile_and_load(const Module& mod, size_t code_padding, const SchedOptions& sched_opts) {
    sched_opts_ = sched_opts;
    return compile_and_load(mod, code_padding);
}

bool JitExecutionEngine::load_object(const object::ObjectFile& obj, size_t code_padding) {
    unregister_seh_tables();
    unregister_eh_frame();
    symbol_table_.clear();

    object::ObjectFile working_obj = obj;
    // Every GOT load whose symbol turns out to be within reach of the code
    // — the object's own, and in practice the process's — is relaxed to a
    // `lea` once addresses are known; the rest read a slot this engine owns
    // beside the code. Slots are reserved for all of them, since which are
    // which is only settled below. In-process, unlike in an image, "defined
    // here" is not the criterion: a separately mapped data block can be out
    // of a lea's reach and a host symbol within it.
    size_t got_slots = 0;
    {
        std::unordered_set<std::string> got_symbols;
        for (const auto& sec : working_obj.sections) {
            for (const auto& r : sec.relocations) {
                if (r.kind == object::RelocKind::GotPCRel32 || object::a64::is_got_kind(r.kind)) {
                    got_symbols.insert(r.symbol_name);
                }
            }
        }
        got_slots = got_symbols.size();
    }

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
    } else if (!working_obj.functions.empty() && !working_obj.get_section(".eh_frame")) {
        // DWARF CFI for every function, laid out with the data pages (its
        // pc-relative FDE addresses are relocated like any other section)
        // and handed to the unwinder below, so that a C++ exception thrown by
        // a host function called from JIT code unwinds through the JIT frames.
        auto& eh = working_obj.get_or_create_section(
            ".eh_frame",
            object::SectionKind::EhFrame,
            object::SectionFlags::Read | object::SectionFlags::Alloc,
            8
        );
        object::ElfCfiBuilder::build_eh_frame(working_obj, eh);
    }

    size_t page_sz = jit_system_page_size();

    // Compute memory size and section offsets:
    // 1. Executable code sections (.text) first
    size_t code_size = code_padding;
    std::vector<size_t> sec_offsets(working_obj.sections.size(), 0);
    std::vector<bool> is_code_sec(working_obj.sections.size(), false);

    for (size_t i = 0; i < working_obj.sections.size(); ++i) {
        const auto& sec = working_obj.sections[i];
        if (sec.kind == object::SectionKind::Text || object::has_flag(sec.flags, object::SectionFlags::Execute)) {
            is_code_sec[i] = true;
            if (sec.alignment > 1) {
                code_size = (code_size + (sec.alignment - 1)) & ~(size_t(sec.alignment) - 1);
            }
            sec_offsets[i] = code_size;
            code_size += sec.data.size();
        }
    }

    // Reserve space for PLT far-call trampolines in the executable region,
    // and after them the GOT: one 8-byte slot per external symbol whose
    // address the code loads. Both sit in the code mapping so that every
    // rip-relative displacement onto them fits in 32 bits.
    size_t trampoline_capacity = 4096;
    size_t trampoline_offset = (code_size + 15) & ~size_t(15);
    size_t got_offset = trampoline_offset + trampoline_capacity;
    if (code_size > 0 || !working_obj.functions.empty()) {
        code_size = got_offset + got_slots * 8;
    }

    // Code pages end at system page boundary
    size_t code_pages_size = (code_size > 0) ? ((code_size + page_sz - 1) & ~(page_sz - 1)) : 0;

    // 2. Non-executable data sections (.rodata, .data, .bss, .pdata, .xdata, etc.)
    size_t data_size = 0;
    for (size_t i = 0; i < working_obj.sections.size(); ++i) {
        const auto& sec = working_obj.sections[i];
        if (!is_code_sec[i]) {
            if (sec.alignment > 1) {
                data_size = (data_size + (sec.alignment - 1)) & ~(size_t(sec.alignment) - 1);
            }
            sec_offsets[i] = data_size;
            data_size += sec.data.size();
        }
    }
    size_t data_pages_size = (data_size > 0) ? ((data_size + page_sz - 1) & ~(page_sz - 1)) : 0;

    if (code_pages_size == 0 && data_pages_size == 0) return true;

    // Allocate memory: the data pages directly after the code pages in the
    // SAME mapping wherever the platform allows it (JitMemoryBlock says why:
    // .pdata and Addr32NB are 32-bit offsets from the code base, and a
    // separately placed data block can land below the code or beyond 4 GB,
    // where the truncated offset sends the unwinder into unmapped memory).
    // The hint-placed separate block remains for the MAP_JIT case only.
    uint8_t* data_base = nullptr;
    if (code_pages_size > 0) {
        code_mem_ = JitMemoryBlock(code_pages_size, data_pages_size);
        if (!code_mem_.is_valid()) {
            return false;
        }
    } else {
        code_mem_.reset();
    }

    if (data_pages_size > 0 && code_mem_.is_valid() &&
        code_mem_.size() >= code_pages_size + data_pages_size) {
        data_mem_.reset();
        data_base = code_mem_.data() + code_pages_size;
    } else if (data_pages_size > 0) {
        void* hint = code_mem_.is_valid() ? (code_mem_.data() + code_mem_.size()) : nullptr;
        data_mem_ = DataMemoryBlock(data_pages_size, hint);
        if (!data_mem_.is_valid()) {
            code_mem_.reset();
            return false;
        }
        data_base = data_mem_.data();
    } else {
        data_mem_.reset();
    }

    std::vector<uint8_t*> sec_bases(working_obj.sections.size(), nullptr);
    for (size_t i = 0; i < working_obj.sections.size(); ++i) {
        if (is_code_sec[i]) {
            sec_bases[i] = code_mem_.data() + sec_offsets[i];
        } else {
            sec_bases[i] = data_base ? data_base + sec_offsets[i] : nullptr;
        }
    }

    uint8_t* trampoline_ptr = code_mem_.is_valid() ? (code_mem_.data() + trampoline_offset) : nullptr;
    size_t trampoline_used = 0;
    std::unordered_map<std::string, void*> trampolines;
    uint8_t* got_ptr = code_mem_.is_valid() ? (code_mem_.data() + got_offset) : nullptr;
    size_t got_used = 0;
    std::unordered_map<std::string, uint8_t*> got_slot_of;

    // Copy section data to memory block
    for (size_t i = 0; i < working_obj.sections.size(); ++i) {
        const auto& sec = working_obj.sections[i];
        if (!sec.data.empty() && sec_bases[i]) {
            std::memcpy(sec_bases[i], sec.data.data(), sec.data.size());
        }
        symbol_table_[sec.name] = sec_bases[i];
    }

    // Register symbols
    for (const auto& sym : working_obj.symbols) {
        if (sym.section_index >= 0 && sym.section_index < static_cast<int32_t>(working_obj.sections.size())) {
            symbol_table_[sym.name] = sec_bases[sym.section_index] + sym.value;
        }
    }
    osr_entry_offsets_.clear();
    for (const auto& fn : working_obj.functions) {
        int32_t text_idx = working_obj.get_section_index(".text");
        if (text_idx >= 0 && sec_bases[text_idx]) {
            symbol_table_[fn.name] = sec_bases[text_idx] + fn.text_offset;
            if (fn.osr_entry_offset > 0) {
                osr_entry_offsets_[fn.name] = fn.osr_entry_offset;
            }
        }
    }

    uint8_t* module_base = code_mem_.is_valid() ? code_mem_.data() : data_mem_.data();

    // Resolve relocations
    for (size_t i = 0; i < working_obj.sections.size(); ++i) {
        const auto& sec = working_obj.sections[i];
        uint8_t* sec_runtime_base = sec_bases[i];
        if (!sec_runtime_base) continue;

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
                case object::RelocKind::Plt32: {
                    if (target_.is_aarch64()) {
                        int64_t disp = reinterpret_cast<int64_t>(target_addr) + r.addend - reinterpret_cast<int64_t>(patch_loc);
                        int64_t disp_words = disp >> 2;
                        if (disp_words < -33554432 || disp_words > 33554431) {
                            // Out of 26-bit reach (+-128MB): generate or reuse AArch64 trampoline
                            void* tramp_addr = nullptr;
                            auto tramp_it = trampolines.find(r.symbol_name);
                            if (tramp_it != trampolines.end()) {
                                tramp_addr = tramp_it->second;
                            } else {
                                if (trampoline_ptr && trampoline_used + 16 <= trampoline_capacity) {
                                    uint8_t* t = trampoline_ptr + trampoline_used;
                                    trampoline_used += 16;
                                    // AArch64 trampoline:
                                    // ldr x16, #8 (0x58000050)
                                    // br x16      (0xD61F0200)
                                    // [64-bit target_addr]
                                    *reinterpret_cast<uint32_t*>(t) = 0x58000050u;
                                    *reinterpret_cast<uint32_t*>(t + 4) = 0xD61F0200u;
                                    *reinterpret_cast<uint64_t*>(t + 8) = reinterpret_cast<uint64_t>(target_addr);
                                    trampolines[r.symbol_name] = t;
                                    tramp_addr = t;
                                }
                            }
                            if (tramp_addr) {
                                disp = reinterpret_cast<int64_t>(tramp_addr) + r.addend - reinterpret_cast<int64_t>(patch_loc);
                            }
                        }
                        uint32_t inst = 0;
                        std::memcpy(&inst, patch_loc, 4);
                        const std::string err = object::a64::patch(
                            object::RelocKind::Plt32, inst, reinterpret_cast<uint64_t>(patch_loc),
                            reinterpret_cast<uint64_t>(patch_loc) + static_cast<uint64_t>(disp));
                        if (!err.empty()) {
                            std::cerr << "JIT Error: call to '" << r.symbol_name << "': " << err << "\n";
                            return false;
                        }
                        std::memcpy(patch_loc, &inst, 4);
                    } else {
                        int64_t disp = reinterpret_cast<int64_t>(target_addr) + r.addend - reinterpret_cast<int64_t>(patch_loc);
                        if (disp < INT32_MIN || disp > INT32_MAX) {
                            // Out of 32-bit reach: generate or reuse a 64-bit indirect jump PLT trampoline
                            void* tramp_addr = nullptr;
                            auto tramp_it = trampolines.find(r.symbol_name);
                            if (tramp_it != trampolines.end()) {
                                tramp_addr = tramp_it->second;
                            } else {
                                if (trampoline_ptr && trampoline_used + 16 <= trampoline_capacity) {
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
                        if (disp < INT32_MIN || disp > INT32_MAX) {
                            std::cerr << "JIT Error: call to '" << r.symbol_name
                                      << "' is out of range and the trampoline area is full\n";
                            return false;
                        }
                        *reinterpret_cast<int32_t*>(patch_loc) = static_cast<int32_t>(disp);
                    }
                    break;
                }
                case object::RelocKind::AdrPage21:
                case object::RelocKind::AddLo12:
                case object::RelocKind::LdSt8Lo12:
                case object::RelocKind::LdSt16Lo12:
                case object::RelocKind::LdSt32Lo12:
                case object::RelocKind::LdSt64Lo12:
                case object::RelocKind::LdSt128Lo12:
                case object::RelocKind::GotPage21:
                case object::RelocKind::GotLo12: {
                    // AArch64 instruction relocations. A GOT pair always goes
                    // through a slot this engine owns in the code mapping:
                    // the two halves are patched separately, so relaxing one
                    // on a distance test the other did not see could split
                    // the pair.
                    uint64_t value = reinterpret_cast<uint64_t>(target_addr) + static_cast<uint64_t>(r.addend);
                    if (object::a64::is_got_kind(r.kind)) {
                        uint8_t* slot = nullptr;
                        auto slot_it = got_slot_of.find(r.symbol_name);
                        if (slot_it != got_slot_of.end()) {
                            slot = slot_it->second;
                        } else if (got_ptr && got_used < got_slots) {
                            slot = got_ptr + got_used * 8;
                            ++got_used;
                            *reinterpret_cast<uint64_t*>(slot) = reinterpret_cast<uint64_t>(target_addr);
                            got_slot_of[r.symbol_name] = slot;
                        } else {
                            std::cerr << "JIT Error: no GOT slot for '" << r.symbol_name << "'\n";
                            return false;
                        }
                        value = reinterpret_cast<uint64_t>(slot);
                    }
                    uint32_t inst = 0;
                    std::memcpy(&inst, patch_loc, 4);
                    const std::string err =
                        object::a64::patch(r.kind, inst, reinterpret_cast<uint64_t>(patch_loc), value);
                    if (!err.empty()) {
                        std::cerr << "JIT Error: relocation against '" << r.symbol_name << "': " << err << "\n";
                        return false;
                    }
                    std::memcpy(patch_loc, &inst, 4);
                    break;
                }
                case object::RelocKind::PCRel32: {
                    int64_t disp = reinterpret_cast<int64_t>(target_addr) + r.addend - reinterpret_cast<int64_t>(patch_loc);
                    if (disp < INT32_MIN || disp > INT32_MAX) {
                        std::cerr << "JIT Error: PC-relative reference to '" << r.symbol_name
                                  << "' is out of 32-bit range\n";
                        return false;
                    }
                    *reinterpret_cast<int32_t*>(patch_loc) = static_cast<int32_t>(disp);
                    break;
                }
                case object::RelocKind::Abs64: {
                    uint64_t val = reinterpret_cast<uint64_t>(target_addr) + static_cast<uint64_t>(r.addend);
                    *reinterpret_cast<uint64_t*>(patch_loc) = val;
                    break;
                }
                case object::RelocKind::GotPCRel32: {
                    // Within reach: the load becomes a lea of the symbol
                    // (8B -> 8D, the same relaxation an image writer does
                    // for a defined symbol). Otherwise the address goes into
                    // this symbol's slot and the load reads the slot.
                    const int64_t direct = reinterpret_cast<int64_t>(target_addr) + r.addend -
                                           reinterpret_cast<int64_t>(patch_loc);
                    if (direct >= INT32_MIN && direct <= INT32_MAX && r.offset >= 2 &&
                        patch_loc[-2] == 0x8B) {
                        patch_loc[-2] = 0x8D;
                        *reinterpret_cast<int32_t*>(patch_loc) = static_cast<int32_t>(direct);
                        break;
                    }
                    uint8_t* slot = nullptr;
                    auto slot_it = got_slot_of.find(r.symbol_name);
                    if (slot_it != got_slot_of.end()) {
                        slot = slot_it->second;
                    } else if (got_ptr && got_used < got_slots) {
                        slot = got_ptr + got_used * 8;
                        ++got_used;
                        *reinterpret_cast<uint64_t*>(slot) = reinterpret_cast<uint64_t>(target_addr);
                        got_slot_of[r.symbol_name] = slot;
                    } else {
                        std::cerr << "JIT Error: no GOT slot for '" << r.symbol_name << "'\n";
                        return false;
                    }
                    int64_t disp = reinterpret_cast<int64_t>(slot) + r.addend - reinterpret_cast<int64_t>(patch_loc);
                    *reinterpret_cast<int32_t*>(patch_loc) = static_cast<int32_t>(disp);
                    break;
                }
                case object::RelocKind::SecRel32:
                case object::RelocKind::Addr32NB: {
                    uint32_t rva = static_cast<uint32_t>(reinterpret_cast<uint8_t*>(target_addr) - module_base + r.addend);
                    *reinterpret_cast<uint32_t*>(patch_loc) = rva;
                    break;
                }
                case object::RelocKind::Abs32: {
                    uint32_t val = static_cast<uint32_t>(reinterpret_cast<uintptr_t>(target_addr) + static_cast<uintptr_t>(r.addend));
                    *reinterpret_cast<uint32_t*>(patch_loc) = val;
                    break;
                }
                case object::RelocKind::SecIdx: {
                    *reinterpret_cast<uint16_t*>(patch_loc) = 1;
                    break;
                }
            }
        }
    }

    // Windows SEH Registration
    if (target_.is_windows()) {
        register_seh_tables(working_obj, module_base);
    } else {
        auto eh_it = symbol_table_.find(".eh_frame");
        if (eh_it != symbol_table_.end() && eh_it->second) {
            register_eh_frame(static_cast<uint8_t*>(eh_it->second));
        }
    }

    // Register and relocate Stack Maps
    stack_maps_ = working_obj.stack_maps;
    int32_t text_idx = working_obj.get_section_index(".text");
    if (text_idx >= 0 && sec_bases[text_idx]) {
        text_section_base_ = sec_bases[text_idx];
        uintptr_t text_base = reinterpret_cast<uintptr_t>(text_section_base_);
        stack_maps_.relocate(text_base);
        for (const auto& fn : working_obj.functions) {
            uintptr_t fn_addr = reinterpret_cast<uintptr_t>(sec_bases[text_idx] + fn.text_offset);
            stack_maps_.register_function_address(fn.name, fn_addr, static_cast<uint32_t>(fn.text_size));
        }
    } else {
        text_section_base_ = module_base;
    }
    brass_set_active_stack_maps(&stack_maps_);

    // Register Resume Tables and Patch Sites
    resume_tables_ = working_obj.resume_tables;
    patch_sites_ = working_obj.patch_sites;
    exception_tables_ = working_obj.exception_tables;

    if (text_section_base_) {
        for (const auto& fn : working_obj.functions) {
            if (fn.text_size > 0) {
                uintptr_t fn_addr = reinterpret_cast<uintptr_t>(text_section_base_) + fn.text_offset;
                runtime::get_global_exception_registry().register_function_mapping(
                    fn_addr,
                    fn.text_size,
                    fn.exception_table
                );
                registered_exception_fns_.push_back(fn_addr);
            }
        }
    }

    // W^X: the code pages go from read-write to read-execute; the data pages
    // after them stay read-write.
    if (code_mem_.is_valid() && !code_mem_.make_executable_read_only(code_pages_size)) {
        std::cerr << "JIT Error: could not make the code pages executable\n";
        return false;
    }
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

// RtlAddFunctionTable takes the native RUNTIME_FUNCTION: 12 bytes on x64,
// 8 on ARM64, matching the .pdata each target's builder emits.
#if defined(_WIN32) && (defined(_M_X64) || defined(__x86_64__) || defined(_M_ARM64) || defined(__aarch64__))
#define BRASS_JIT_SEH_REGISTRATION 1
#endif

void JitExecutionEngine::register_seh_tables(const object::ObjectFile& obj, uint8_t* base_ptr) {
#if defined(BRASS_JIT_SEH_REGISTRATION)
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

#if !defined(_WIN32)
// The unwinder's dynamic registration interface (libgcc, and libunwind on
// Apple platforms).
extern "C" void __register_frame(void*);
extern "C" void __deregister_frame(void*);
#endif

void JitExecutionEngine::register_eh_frame(uint8_t* eh_frame) {
    unregister_eh_frame();
#if !defined(_WIN32)
    // Frames are only described to this process's unwinder when this process
    // can run the code.
    const Target host = Target::host();
    if (target_.is_aarch64() != host.is_aarch64() || target_.is_windows()) return;
#if defined(__APPLE__)
    // Apple's libunwind registers a single FDE per call.
    uint8_t* p = eh_frame;
    for (;;) {
        uint32_t len = 0;
        std::memcpy(&len, p, 4);
        if (len == 0) break;
        if (len == 0xFFFFFFFFu) {
            throw std::runtime_error("JIT .eh_frame uses a 64-bit length, which it never emits");
        }
        uint32_t cie_id = 0;
        std::memcpy(&cie_id, p + 4, 4);
        if (cie_id != 0) {
            __register_frame(p);
            registered_fdes_.push_back(p);
        }
        p += 4 + len;
    }
#else
    // libgcc takes the whole zero-terminated section.
    __register_frame(eh_frame);
    registered_fdes_.push_back(eh_frame);
#endif
#else
    (void)eh_frame;
#endif
}

void JitExecutionEngine::unregister_eh_frame() {
#if !defined(_WIN32)
    for (auto it = registered_fdes_.rbegin(); it != registered_fdes_.rend(); ++it) {
        __deregister_frame(*it);
    }
#endif
    registered_fdes_.clear();
}

void JitExecutionEngine::unregister_seh_tables() {
#if defined(BRASS_JIT_SEH_REGISTRATION)
    if (pdata_table_) {
        RtlDeleteFunctionTable(reinterpret_cast<PRUNTIME_FUNCTION>(pdata_table_));
        pdata_table_ = nullptr;
        pdata_count_ = 0;
        code_base_ = 0;
    }
#endif
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
    const runtime::CodeArch arch = target_.is_aarch64() ? runtime::CodeArch::AArch64 : runtime::CodeArch::X64;
    bool ok = patch_sites_.patch_call(arch, text_section_base_, site_name, new_target);
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

size_t JitExecutionEngine::get_osr_entry_offset(std::string_view fn_name) const {
    auto it = osr_entry_offsets_.find(std::string(fn_name));
    if (it != osr_entry_offsets_.end()) {
        return it->second;
    }
    return 0;
}

void* JitExecutionEngine::get_osr_entry_address(std::string_view fn_name) const {
    size_t off = get_osr_entry_offset(fn_name);
    if (off == 0) return nullptr;
    void* fn_addr = get_symbol_address(fn_name);
    if (!fn_addr) return nullptr;
    return reinterpret_cast<uint8_t*>(fn_addr) + off;
}

runtime::MultiTierPipeline* JitExecutionEngine::multi_tier_pipeline() noexcept {
    return &runtime::MultiTierPipeline::instance();
}

const runtime::MultiTierPipeline* JitExecutionEngine::multi_tier_pipeline() const noexcept {
    return &runtime::MultiTierPipeline::instance();
}

void JitExecutionEngine::set_multi_tier_enabled(bool enabled) {
    if (enabled) {
        runtime::MultiTierPipeline::instance().initialize();
    } else {
        runtime::MultiTierPipeline::instance().shutdown();
    }
}

bool JitExecutionEngine::is_multi_tier_enabled() const noexcept {
    return runtime::MultiTierPipeline::instance().is_initialized();
}

} // namespace brass::codegen
