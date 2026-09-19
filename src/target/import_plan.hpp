#pragma once

// Which undefined symbols an image actually IMPORTS, and from where.
//
// Shared by the three image writers. An object's symbol table declares more
// externals than its relocations use (a module declares every runtime helper
// it might call), and only the referenced ones become imports — the same rule
// a system linker applies. A referenced symbol that no library provides is an
// error naming it: the writers used to resolve such a symbol to address zero
// and produce an image that loaded fine and crashed on first call.

#include <brass/object/object_writer.hpp>
#include <brass/target/image_imports.hpp>
#include <string>
#include <string_view>
#include <vector>

namespace brass::target::imports {

struct Imported {
    std::string name;
    size_t library = 0;   // index into Plan::libraries
    size_t index = 0;     // position in Plan::symbols, which is the stub/slot index
};

struct Plan {
    // Only the libraries at least one referenced symbol came from, in the
    // order the options listed them.
    std::vector<std::string> libraries;
    // Sorted by (library, name): deterministic output for a given object.
    std::vector<Imported> symbols;

    bool empty() const { return symbols.empty(); }
    const Imported* find(std::string_view name) const;
    // Symbol indices grouped per library, in Plan::symbols order.
    std::vector<std::vector<size_t>> by_library() const;
};

// Is `name` something the writer resolves from the object itself — a defined
// symbol, or the name of a section (section-relative relocations name their
// section)?
bool is_defined_in(const object::ObjectFile& obj, std::string_view name);

// Walks the relocations of every section in `section_names` (the ones the
// writer will place in the image) and plans the imports. False, with `error`
// set, when a referenced symbol is undefined and in no library.
bool plan(const object::ObjectFile& obj, const std::vector<std::string_view>& section_names,
          const std::vector<ImportLibrary>& libraries, Plan& out, std::string& error);

} // namespace brass::target::imports
