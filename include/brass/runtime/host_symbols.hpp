#pragma once

namespace brass {

class Interpreter;
class FastInterpreter;

namespace codegen {
class JitExecutionEngine;
class BaselineJitCompiler;
}

namespace runtime {

// The embedder's runtime, as the execution engines brass creates on its own
// see it. A tier that builds an engine internally — the tier-2 installer's
// JitExecutionEngine, the pipeline's baseline compiler (at initialize), its
// fast and reference interpreters, a deoptimization's interpreter, and the C
// API's JIT engines — has no caller in the loop to register the
// host's helpers and data, so it asks the installed provider to. Each hook
// registers whatever the host's generated code names (its runtime helpers,
// module data, the thread-block accessor) into the engine it is handed.
//
// brass itself provides none: an engine an embedder builds directly is
// registered by that embedder, and a pipeline's own register_external_*
// tables are applied after the provider, so they win over it.
class HostSymbolProvider {
public:
    virtual ~HostSymbolProvider() = default;
    virtual void install(codegen::JitExecutionEngine& /*engine*/) {}
    virtual void install(codegen::BaselineJitCompiler& /*compiler*/) {}
    virtual void install(Interpreter& /*interp*/) {}
    virtual void install(FastInterpreter& /*interp*/) {}

    // The calling thread's value for the pinned TLS register (x64 R13,
    // AArch64 X28) that code of a `pinned_tls_register` module reads without
    // setting. The invoke thunks load it before entering native code from
    // C++ (an interpreter calling a tiered-up function, `invoke`), and the
    // fast interpreter's pinned_tls_read starts from it. Null by default.
    virtual void* pinned_tls_block() { return nullptr; }
};

// Process-wide, not owned. The provider must outlive every engine created
// while it is installed; installing nullptr removes it. Safe to call from
// any thread; an engine being set up concurrently sees the old or the new
// provider, never a torn one.
void set_host_symbol_provider(HostSymbolProvider* provider) noexcept;
HostSymbolProvider* host_symbol_provider() noexcept;

// What the tiers call on an engine they build: the provider's hook, or
// nothing when none is installed.
void install_host_symbols(codegen::JitExecutionEngine& engine);
void install_host_symbols(codegen::BaselineJitCompiler& compiler);
void install_host_symbols(Interpreter& interp);
void install_host_symbols(FastInterpreter& interp);

// The provider's pinned_tls_block(), or null when none is installed.
void* host_pinned_tls_block() noexcept;

} // namespace runtime
} // namespace brass
