#pragma once

// Lazy linking for the x64 baseline tier. A call or func_addr whose symbol
// is not resolvable when the caller is compiled (a sibling the host compiles
// and registers later) goes through a per-symbol stub:
//
//     stub:  movabs r11, &cell ; jmp [r11]
//
// The cell starts out pointing at a shared resolver thunk. The first call
// through the stub resolves the symbol (register_external_symbol, the
// compiler's custom resolver, the dispatch table), stores the address in the
// cell and continues into it with the caller's arguments intact; later calls
// jump straight through. Registering the symbol fills the cell eagerly.
//
// The stub's address is what func_addr yields: stable and callable before
// the target exists. A call through a symbol that is still unresolved when
// it runs is a hard error: the name goes to stderr and to
// last_unresolved_symbol(), and the thunk executes ud2 - never a call
// through null.
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace brass::codegen {

class JitMemoryBlock;
class LazySymbolTable;

struct LazySymbolCell {
    // Read by the stub's `jmp [r11]`: must stay the first member.
    std::atomic<void*> target{nullptr};
    LazySymbolTable* table = nullptr;
    std::string name;
};

class LazySymbolTable {
public:
    using Resolver = std::function<void*(std::string_view)>;

    explicit LazySymbolTable(Resolver resolver);
    ~LazySymbolTable();

    LazySymbolTable(const LazySymbolTable&) = delete;
    LazySymbolTable& operator=(const LazySymbolTable&) = delete;

    // The stub for `name`, created on first use; the same address for the
    // table's lifetime. Throws std::runtime_error if executable memory
    // cannot be allocated.
    void* stub_for(std::string_view name);

    // Points `name`'s cell (if one exists) at `addr`; null re-arms lazy
    // resolution.
    void define(std::string_view name, void* addr);

    // Drops the resolver (its owner is going away). Stubs stay valid; a cell
    // not yet resolved then fails at call time.
    void detach();

    // Current target of `name`'s cell: null if there is no cell or it is
    // still unresolved.
    void* resolved_target(std::string_view name) const;

    // The symbol of the last call on this thread that hit an unresolved
    // stub ("" if none).
    static std::string last_unresolved_symbol();
    static void clear_last_unresolved_symbol();

    // Called by the resolver thunk; returns the target, or null (after
    // reporting) if the symbol is still unresolved.
    void* resolve(LazySymbolCell& cell);

private:
    struct Chunk;
    static constexpr size_t kStubSize = 16;
    static constexpr size_t kStubsPerChunk = 256;

    mutable std::mutex mutex_;
    Resolver resolver_;
    std::vector<std::unique_ptr<Chunk>> chunks_;
    size_t used_in_last_chunk_ = kStubsPerChunk;
    std::unordered_map<std::string, std::pair<LazySymbolCell*, void*>> by_name_;
};

} // namespace brass::codegen
