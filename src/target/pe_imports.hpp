#pragma once

// The PE import machinery: the `.idata` section (import directory, lookup and
// address tables, hint/name table) and one thunk per imported function.
//
// A relocation against an imported symbol resolves to the symbol's THUNK — a
// `jmp [IAT slot]` appended to .text — exactly as MSVC's linker treats a
// function that was not declared dllimport: a call reaches the callee through
// the thunk, and taking its address yields the thunk. The loader fills the IAT
// at load time; nothing in the image holds an import's real address before
// then, which is why data references cannot be pointed at the callee
// directly.

#include "import_plan.hpp"

#include <cstdint>
#include <vector>

namespace brass::target::pe_imports {

struct Table {
    imports::Plan plan;
    bool aarch64 = false;
    size_t thunk_base = 0;      // offset of the first thunk in .text
    size_t thunk_stride = 8;    // bytes per thunk: 8 on x64, 16 on AArch64
    // Filled by `emit`, for the data directories.
    uint32_t directory_rva = 0;
    uint32_t directory_size = 0;
    uint32_t iat_rva = 0;
    uint32_t iat_size = 0;

    bool empty() const { return plan.empty(); }
    // Size of the `.idata` section, known before layout.
    uint32_t idata_size() const;
    // Appends the thunks to `text` (which must already hold every function),
    // recording where they start; their IAT displacements are patched by
    // `emit` once RVAs exist.
    void append_thunks(std::vector<uint8_t>& text);
    // Builds `.idata` in place and patches the thunks against the IAT.
    void emit(std::vector<uint8_t>& idata, uint32_t idata_rva, std::vector<uint8_t>& text,
              uint32_t text_rva);
    uint32_t thunk_rva(size_t index, uint32_t text_rva) const {
        return text_rva + static_cast<uint32_t>(thunk_base + index * thunk_stride);
    }
};

} // namespace brass::target::pe_imports
