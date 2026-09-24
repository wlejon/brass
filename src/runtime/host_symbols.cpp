#include <brass/runtime/host_symbols.hpp>

#include <atomic>

namespace brass::runtime {

namespace {
std::atomic<HostSymbolProvider*> g_provider{nullptr};
}

void set_host_symbol_provider(HostSymbolProvider* provider) noexcept {
    g_provider.store(provider, std::memory_order_release);
}

HostSymbolProvider* host_symbol_provider() noexcept {
    return g_provider.load(std::memory_order_acquire);
}

void install_host_symbols(codegen::JitExecutionEngine& engine) {
    if (HostSymbolProvider* p = host_symbol_provider()) p->install(engine);
}

void install_host_symbols(codegen::BaselineJitCompiler& compiler) {
    if (HostSymbolProvider* p = host_symbol_provider()) p->install(compiler);
}

void install_host_symbols(Interpreter& interp) {
    if (HostSymbolProvider* p = host_symbol_provider()) p->install(interp);
}

void install_host_symbols(FastInterpreter& interp) {
    if (HostSymbolProvider* p = host_symbol_provider()) p->install(interp);
}

void* host_pinned_tls_block() noexcept {
    HostSymbolProvider* p = host_symbol_provider();
    return p ? p->pinned_tls_block() : nullptr;
}

} // namespace brass::runtime
