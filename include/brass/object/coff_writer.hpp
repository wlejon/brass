#pragma once

#include <brass/object/object_writer.hpp>
#include <vector>
#include <string>
#include <string_view>
#include <cstdint>

namespace brass::object {

namespace coff {
    constexpr uint16_t IMAGE_FILE_MACHINE_AMD64 = 0x8664;

    constexpr uint32_t IMAGE_SCN_CNT_CODE               = 0x00000020;
    constexpr uint32_t IMAGE_SCN_CNT_INITIALIZED_DATA  = 0x00000040;
    constexpr uint32_t IMAGE_SCN_CNT_UNINITIALIZED_DATA = 0x00000080;
    constexpr uint32_t IMAGE_SCN_ALIGN_1BYTES           = 0x00100000;
    constexpr uint32_t IMAGE_SCN_ALIGN_2BYTES           = 0x00200000;
    constexpr uint32_t IMAGE_SCN_ALIGN_4BYTES           = 0x00300000;
    constexpr uint32_t IMAGE_SCN_ALIGN_8BYTES           = 0x00400000;
    constexpr uint32_t IMAGE_SCN_ALIGN_16BYTES          = 0x00500000;
    constexpr uint32_t IMAGE_SCN_MEM_DISCARDABLE        = 0x02000000;
    constexpr uint32_t IMAGE_SCN_MEM_EXECUTE            = 0x20000000;
    constexpr uint32_t IMAGE_SCN_MEM_READ               = 0x40000000;
    constexpr uint32_t IMAGE_SCN_MEM_WRITE              = 0x80000000;

    constexpr uint16_t IMAGE_REL_AMD64_ABSOLUTE = 0x0000;
    constexpr uint16_t IMAGE_REL_AMD64_ADDR64   = 0x0001;
    constexpr uint16_t IMAGE_REL_AMD64_ADDR32   = 0x0002;
    constexpr uint16_t IMAGE_REL_AMD64_ADDR32NB = 0x0003;
    constexpr uint16_t IMAGE_REL_AMD64_REL32    = 0x0004;
    constexpr uint16_t IMAGE_REL_AMD64_SECTION  = 0x000A;
    constexpr uint16_t IMAGE_REL_AMD64_SECREL   = 0x000B;

    constexpr uint8_t IMAGE_SYM_CLASS_EXTERNAL = 2;
    constexpr uint8_t IMAGE_SYM_CLASS_STATIC   = 3;
    constexpr uint16_t IMAGE_SYM_DTYPE_FUNCTION = 0x20;

    // Win64 SEH Unwind Opcodes
    constexpr uint8_t UWOP_PUSH_NONVOL      = 0;
    constexpr uint8_t UWOP_ALLOC_LARGE       = 1;
    constexpr uint8_t UWOP_ALLOC_SMALL       = 2;
    constexpr uint8_t UWOP_SET_FPREG         = 3;
    constexpr uint8_t UWOP_SAVE_NONVOL       = 4;
    constexpr uint8_t UWOP_SAVE_NONVOL_FAR   = 5;
    constexpr uint8_t UWOP_SAVE_XMM128       = 8;
    constexpr uint8_t UWOP_SAVE_XMM128_FAR   = 9;
    constexpr uint8_t UWOP_PUSH_MACHFRAME    = 10;
}

class CoffUnwindBuilder {
public:
    static void build_unwind_info(
        ObjectFile& obj,
        Section& pdata_sec,
        Section& xdata_sec
    );
};

class CoffWriter {
public:
    explicit CoffWriter(const ObjectFile& obj);

    std::vector<uint8_t> write();
    bool write_to_file(const std::string& path);

private:
    ObjectFile obj_;
};

std::vector<uint8_t> emit_coff_object(const ObjectFile& obj);

} // namespace brass::object
