// Lazy linking stubs for the x64 baseline tier (see lazy_symbols.hpp).
#include <brass/codegen/lazy_symbols.hpp>
#include <brass/codegen/jit_exec.hpp>
#include <brass/target/x64/x64_encoder.hpp>
#include <cstdio>
#include <cstring>
#include <iterator>
#include <stdexcept>

namespace brass::codegen {

namespace {

thread_local std::string t_last_unresolved;

extern "C" void* brass_lazy_symbol_resolve(LazySymbolCell* cell) {
    if (!cell || !cell->table) {
        t_last_unresolved = cell ? cell->name : std::string("<null cell>");
        std::fprintf(stderr, "brass: call through a lazy stub with no symbol table ('%s')\n",
                     t_last_unresolved.c_str());
        std::fflush(stderr);
        return nullptr;
    }
    return cell->table->resolve(*cell);
}

#if defined(__x86_64__) || defined(_M_X64)
#define BRASS_LAZY_STUBS_SUPPORTED 1

// Entered with r11 = the cell, the caller's arguments in their registers and
// on the stack, and [rsp] = the caller's return address. Saves every
// argument register, asks brass_lazy_symbol_resolve for the target, restores
// them and tail-jumps to the target, or executes ud2 if there is none.
void* build_resolver_thunk() {
    using namespace brass::x64;
    CodeBuffer buffer;
    X64Encoder enc(buffer);
    Label trap = buffer.create_label();
    enc.push(GPR::RBP);
    enc.mov(GPR::RBP, GPR::RSP);
#if defined(_WIN32)
    static constexpr GPR kSaved[] = {GPR::RCX, GPR::RDX, GPR::R8, GPR::R9};
    constexpr int kXmms = 4;
    constexpr int32_t kShadow = 32;
    constexpr int32_t kAlloc = kShadow + kXmms * 16;  // 4 pushes keep rsp 16-aligned
    GPR arg0 = GPR::RCX;
#else
    // RAX carries the vector-register count of a variadic call.
    static constexpr GPR kSaved[] = {GPR::RDI, GPR::RSI, GPR::RDX, GPR::RCX, GPR::R8, GPR::R9, GPR::RAX};
    constexpr int kXmms = 8;
    constexpr int32_t kShadow = 0;
    constexpr int32_t kAlloc = kXmms * 16 + 8;  // 7 pushes leave rsp 8 off
    GPR arg0 = GPR::RDI;
#endif
    for (GPR r : kSaved) enc.push(r);
    enc.sub(GPR::RSP, kAlloc);
    for (int i = 0; i < kXmms; ++i) {
        enc.movups(MemAddress::base_disp(GPR::RSP, kShadow + i * 16), static_cast<XMM>(static_cast<uint8_t>(i)));
    }
    enc.mov(arg0, GPR::R11);
    enc.movabs(GPR::RAX, reinterpret_cast<uint64_t>(reinterpret_cast<void*>(&brass_lazy_symbol_resolve)));
    enc.call(GPR::RAX);
    enc.mov(GPR::R11, GPR::RAX);
    for (int i = 0; i < kXmms; ++i) {
        enc.movups(static_cast<XMM>(static_cast<uint8_t>(i)), MemAddress::base_disp(GPR::RSP, kShadow + i * 16));
    }
    enc.add(GPR::RSP, kAlloc);
    for (size_t i = std::size(kSaved); i-- > 0;) enc.pop(kSaved[i]);
    enc.pop(GPR::RBP);
    enc.test(GPR::R11, GPR::R11);
    enc.je(trap);
    enc.jmp(GPR::R11);
    buffer.bind(trap);
    enc.ud2();

    // Process lifetime: cells of every table point here until resolved.
    auto* block = new JitMemoryBlock(buffer.size());
    if (!block->is_valid()) throw std::runtime_error("lazy symbols: cannot allocate the resolver thunk");
    std::memcpy(block->data(), buffer.data(), buffer.size());
    if (!block->make_executable_read_only()) {
        throw std::runtime_error("lazy symbols: cannot make the resolver thunk executable");
    }
    return block->data();
}

void* resolver_thunk() {
    static void* thunk = build_resolver_thunk();
    return thunk;
}
#endif

} // namespace

struct LazySymbolTable::Chunk {
    std::unique_ptr<LazySymbolCell[]> cells{new LazySymbolCell[kStubsPerChunk]};
    JitMemoryBlock code{kStubSize * kStubsPerChunk};
};

LazySymbolTable::LazySymbolTable(Resolver resolver) : resolver_(std::move(resolver)) {}

LazySymbolTable::~LazySymbolTable() = default;

void* LazySymbolTable::stub_for(std::string_view name) {
#if defined(BRASS_LAZY_STUBS_SUPPORTED)
    void* thunk = resolver_thunk();
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = by_name_.find(std::string(name));
    if (it != by_name_.end()) return it->second.second;

    if (used_in_last_chunk_ == kStubsPerChunk) {
        // Every stub of a chunk is written up front: the code is read-only
        // once executable.
        auto chunk = std::make_unique<Chunk>();
        if (!chunk->code.is_valid()) throw std::runtime_error("lazy symbols: cannot allocate stub memory");
        for (size_t i = 0; i < kStubsPerChunk; ++i) {
            LazySymbolCell& cell = chunk->cells[i];
            cell.table = this;
            cell.target.store(thunk, std::memory_order_relaxed);
            uint8_t* p = chunk->code.data() + i * kStubSize;
            const uint64_t cell_addr = reinterpret_cast<uint64_t>(&cell);
            p[0] = 0x49; p[1] = 0xBB;                 // movabs r11, imm64
            std::memcpy(p + 2, &cell_addr, 8);
            p[10] = 0x41; p[11] = 0xFF; p[12] = 0x23; // jmp qword [r11]
            p[13] = 0xCC; p[14] = 0xCC; p[15] = 0xCC;
        }
        if (!chunk->code.make_executable_read_only()) {
            throw std::runtime_error("lazy symbols: cannot make stub memory executable");
        }
        chunks_.push_back(std::move(chunk));
        used_in_last_chunk_ = 0;
    }
    Chunk& chunk = *chunks_.back();
    const size_t index = used_in_last_chunk_++;
    LazySymbolCell* cell = &chunk.cells[index];
    cell->name = std::string(name);
    void* stub = chunk.code.data() + index * kStubSize;
    by_name_.emplace(cell->name, std::make_pair(cell, stub));
    return stub;
#else
    (void)name;
    throw std::runtime_error("lazy symbols: x64 stubs need an x64 host");
#endif
}

void LazySymbolTable::define(std::string_view name, void* addr) {
#if defined(BRASS_LAZY_STUBS_SUPPORTED)
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = by_name_.find(std::string(name));
    if (it == by_name_.end()) return;
    // A symbol defined as its own stub would loop forever.
    if (addr == it->second.second) addr = nullptr;
    it->second.first->target.store(addr ? addr : resolver_thunk(), std::memory_order_release);
#else
    (void)name;
    (void)addr;
#endif
}

void LazySymbolTable::detach() {
    std::lock_guard<std::mutex> lock(mutex_);
    resolver_ = nullptr;
}

void* LazySymbolTable::resolved_target(std::string_view name) const {
#if defined(BRASS_LAZY_STUBS_SUPPORTED)
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = by_name_.find(std::string(name));
    if (it == by_name_.end()) return nullptr;
    void* t = it->second.first->target.load(std::memory_order_acquire);
    return t == resolver_thunk() ? nullptr : t;
#else
    (void)name;
    return nullptr;
#endif
}

void* LazySymbolTable::resolve(LazySymbolCell& cell) {
#if defined(BRASS_LAZY_STUBS_SUPPORTED)
    {
        std::lock_guard<std::mutex> lock(mutex_);
        void* current = cell.target.load(std::memory_order_acquire);
        if (current != resolver_thunk()) return current;  // resolved meanwhile
        void* addr = resolver_ ? resolver_(cell.name) : nullptr;
        auto it = by_name_.find(cell.name);
        if (addr && it != by_name_.end() && addr == it->second.second) addr = nullptr;
        if (addr) {
            cell.target.store(addr, std::memory_order_release);
            return addr;
        }
    }
#endif
    t_last_unresolved = cell.name;
    std::fprintf(stderr, "brass: call to unresolved symbol '%s' (never registered with the baseline JIT)\n",
                 cell.name.c_str());
    std::fflush(stderr);
    return nullptr;
}

std::string LazySymbolTable::last_unresolved_symbol() { return t_last_unresolved; }

void LazySymbolTable::clear_last_unresolved_symbol() { t_last_unresolved.clear(); }

} // namespace brass::codegen
