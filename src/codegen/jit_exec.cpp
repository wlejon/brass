#include <brass/codegen/jit_exec.hpp>
#include <brass/object/coff_writer.hpp>
#include <brass/object/elf_writer.hpp>
#include <stdexcept>
#include <cstring>
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
    }
#else
    if (ptr_) {
        mprotect(ptr_, size_, PROT_READ | PROT_WRITE | PROT_EXEC);
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
    : target_(target) {}

JitExecutionEngine::JitExecutionEngine()
    : target_(Target::host()) {}

JitExecutionEngine::~JitExecutionEngine() {
    unregister_seh_tables();
}

JitExecutionEngine::JitExecutionEngine(JitExecutionEngine&& other) noexcept
    : target_(other.target_),
      code_mem_(std::move(other.code_mem_)),
      symbol_table_(std::move(other.symbol_table_)),
      external_symbols_(std::move(other.external_symbols_)),
      function_signatures_(std::move(other.function_signatures_)),
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

bool JitExecutionEngine::compile_and_load(const Module& mod) {
    // Record signatures of functions in the module
    for (const auto* fn : mod.functions()) {
        if (!fn) continue;
        std::vector<Type> params = fn->param_types();
        function_signatures_[std::string(fn->name())] = {fn->return_type(), std::move(params)};
    }

    object::ObjectFile obj = object::compile_module_to_object(mod, target_);
    return load_object(obj);
}

bool JitExecutionEngine::load_object(const object::ObjectFile& obj) {
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
    size_t total_size = 0;
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

    // Allocate memory block
    code_mem_ = JitMemoryBlock(total_size);
    if (!code_mem_.is_valid()) {
        return false;
    }

    uint8_t* base_ptr = code_mem_.data();

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
        } else if (ret_type.is_float()) {
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
        if (args[0].is_f64()) {
            if (ret_type.is_float()) {
                double r = reinterpret_cast<double(*)(double)>(addr)(get_float(0));
                return RuntimeValue::from_f64(r);
            } else {
                int64_t r = reinterpret_cast<int64_t(*)(double)>(addr)(get_float(0));
                return RuntimeValue::from_i64(r);
            }
        } else {
            if (ret_type.is_float()) {
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
        if (args[0].is_f64() && args[1].is_f64()) {
            if (ret_type.is_float()) {
                double r = reinterpret_cast<double(*)(double, double)>(addr)(get_float(0), get_float(1));
                return RuntimeValue::from_f64(r);
            } else {
                int64_t r = reinterpret_cast<int64_t(*)(double, double)>(addr)(get_float(0), get_float(1));
                return RuntimeValue::from_i64(r);
            }
        } else {
            if (ret_type.is_float()) {
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

} // namespace brass::codegen
