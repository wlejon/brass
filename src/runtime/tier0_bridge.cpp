// A program's function pointer is the function's lazy stub
// (MultiTierPipeline::function_address), in every tier. Native code calls
// it directly; the first call compiles the function to Tier 1. A function
// the baseline tier rejects gets a native-to-Tier-0 bridge instead: a small
// baseline-compiled function of the same signature that passes its
// arguments as raw bits to a host entry, which runs the function in Tier 0
// (re-entering the interpreter that called into native code) and returns
// the result's bits. An exception it throws unwinds through the native
// frames to the Tier-0 or host handler above them.
#include <brass/runtime/multi_tier_pipeline.hpp>
#include <brass/runtime/code_installer.hpp>
#include <brass/codegen/baseline_jit.hpp>
#include <brass/interpreter/interpreter.hpp>
#include <brass/gc/native_frames.hpp>
#include <brass/mir/builder.hpp>
#include <brass/mir/module.hpp>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace brass::runtime {

namespace {

struct Tier0Bridge {
    MultiTierPipeline* pipeline = nullptr;
    std::string name;
    std::unique_ptr<Module> module;
    std::shared_ptr<codegen::BaselineCompiledFunction> code;
};

[[noreturn]] void bridge_fatal(const std::string& msg) {
    std::fprintf(stderr, "brass: fatal error in a native-to-Tier-0 call: %s\n", msg.c_str());
    std::fflush(stderr);
    std::abort();
}

// The bridge's host entries, one per argument count: the context, then each
// argument's bits.
template <size_t I>
using Bits = uint64_t;

// The native frames that called the bridge (its baseline body, then the
// caller holding live gcrefs across the call) are recorded for the time
// Tier 0 runs: a collection the callee triggers starts in the interpreter
// and would not otherwise see them (native_frames.hpp).
template <size_t... I>
uint64_t bridge_entry(Tier0Bridge* ctx, Bits<I>... args) {
    uintptr_t caller_rbp = 0, caller_ip = 0;
    if (!brass_capture_caller_frame(caller_rbp, caller_ip)) {
        bridge_fatal("cannot find the native frame that called '" + ctx->name + "'");
    }
    NativeFramesScope native_frames(caller_rbp, caller_ip);
    const uint64_t bits[] = {args..., 0};
    return ctx->pipeline->call_tier0_from_native(ctx->name, bits, sizeof...(I));
}

template <size_t... I>
void* bridge_entry_addr(std::index_sequence<I...>) {
    return reinterpret_cast<void*>(&bridge_entry<I...>);
}

template <size_t... N>
std::vector<void*> make_bridge_entries(std::index_sequence<N...>) {
    return {bridge_entry_addr(std::make_index_sequence<N>{})...};
}

const std::vector<void*>& bridge_entries() {
    static const std::vector<void*> entries =
        make_bridge_entries(std::make_index_sequence<MultiTierPipeline::kTier0BridgeMaxParams + 1>{});
    return entries;
}

std::string bridge_entry_symbol(size_t n) { return "brass_tier0_bridge_entry" + std::to_string(n); }

bool bridgeable(Type t) {
    switch (t.kind()) {
        case TypeKind::I8:
        case TypeKind::I32:
        case TypeKind::I64:
        case TypeKind::Ptr:
        case TypeKind::F64:
            return true;
        default:
            return false;
    }
}

// Raw bits as the value the interpreter holds for a value of type `t`.
RuntimeValue materialize(Type t, uint64_t bits) {
    if (t.kind() == TypeKind::I8) bits &= 0xFFull;
    if (t.kind() == TypeKind::I32) bits &= 0xFFFFFFFFull;
    return RuntimeValue::from_bits(t, bits);
}

} // namespace

bool MultiTierPipeline::tier0_bridge_supported(const Function& fn, std::string* why) {
    auto fail = [&](const std::string& msg) {
        if (why) *why = msg;
        return false;
    };
    if (fn.param_count() > kTier0BridgeMaxParams) {
        return fail("it has " + std::to_string(fn.param_count()) + " parameters (at most " +
                    std::to_string(kTier0BridgeMaxParams) + ")");
    }
    for (size_t i = 0; i < fn.param_count(); ++i) {
        if (!bridgeable(fn.param_type(i))) return fail("parameter " + std::to_string(i) + " has an unsupported type");
    }
    if (!fn.return_type().is_void() && !bridgeable(fn.return_type())) return fail("its result type is unsupported");
    return true;
}

void* MultiTierPipeline::function_address(std::string_view name, const Function* fn) {
    // The handle carries the MIR the stub compiles, or bridges to, on demand.
    table_->get_or_create(name, fn);
    return baseline_compiler_.module_function_stub(name);
}

uint64_t MultiTierPipeline::call_tier0_from_native(std::string_view name, const uint64_t* bits, size_t count) {
    FunctionHandle* handle = table_->find(name);
    const Function* fn = handle ? handle->mir_function() : nullptr;
    if (!fn) bridge_fatal("'" + std::string(name) + "' has no MIR to run");
    if (fn->param_count() != count) {
        bridge_fatal("'" + std::string(name) + "' called with " + std::to_string(count) + " arguments");
    }
    std::vector<RuntimeValue> args;
    args.reserve(count);
    for (size_t i = 0; i < count; ++i) args.push_back(materialize(fn->param_type(i), bits[i]));
    // Re-enter the interpreter whose code called into native code on this
    // thread, if it runs this program: the callee then shares its heap and
    // state as a Tier-0 call would. Otherwise (a host called native code
    // directly, or Tier 0 is the fast interpreter) a fresh one runs it.
    // Either way an exception it throws (a MIR throw, or a Tier-0 error)
    // propagates as a C++ exception: baseline frames carry unwind data, so
    // it unwinds through the bridge and its native callers to the Tier-0
    // invoke or host catch above them. Baseline code has no handlers of its
    // own (the tier rejects exception ops), so none is skipped.
    Interpreter* active = Interpreter::active_on_thread();
    RuntimeValue r = active && &active->dispatch_table() == table_ ? active->call_from_native(*fn, args)
                                                                   : run_fresh_tier0(*table_, fn, args);
    return r.is_void() ? 0 : r.raw_bits();
}

void* MultiTierPipeline::tier0_bridge(std::string_view name) {
    FunctionHandle* handle = table_->find(name);
    const Function* fn = handle ? handle->mir_function() : nullptr;
    if (!fn) return nullptr;
    std::lock_guard<std::mutex> lock(bridges_mutex_);
    if (auto it = tier0_bridges_.find(std::string(name)); it != tier0_bridges_.end()) {
        return static_cast<Tier0Bridge*>(it->second.get())->code->entry_point();
    }
    std::string why;
    if (!tier0_bridge_supported(*fn, &why)) {
        std::fprintf(stderr, "brass: '%s' is called through its address from native code, the baseline tier "
                             "rejects it, and it cannot be bridged into Tier 0: %s\n",
                     std::string(name).c_str(), why.c_str());
        std::fflush(stderr);
        return nullptr;
    }

    auto bridge = std::make_shared<Tier0Bridge>();
    bridge->pipeline = this;
    bridge->name = std::string(name);
    bridge->module = std::make_unique<Module>("brass_tier0_bridge");
    const std::string bname = "brass.tier0_bridge." + std::string(name);
    Function* bf = bridge->module->create_function(bname, fn->return_type(), fn->param_types());
    Builder b(*bridge->module);
    b.set_function(bf);
    BasicBlock* entry = b.append_block("entry");
    std::vector<Value*> call_args;
    call_args.push_back(b.build_iconst_i64(static_cast<int64_t>(reinterpret_cast<uintptr_t>(bridge.get()))));
    for (size_t i = 0; i < fn->param_count(); ++i) {
        Type t = fn->param_type(i);
        Value* p = b.add_block_param(entry, t);
        switch (t.kind()) {
            case TypeKind::I8: p = b.build_zext_i64(p); break;
            case TypeKind::I32: p = b.build_sext_i64(p); break;
            // (Named result-type first: bitcast_i64_f64 makes an i64.)
            case TypeKind::F64: p = b.build_bitcast_i64_f64(p); break;
            default: break;
        }
        call_args.push_back(p);
    }
    const std::string entry_sym = bridge_entry_symbol(fn->param_count());
    baseline_compiler_.register_external_symbol(entry_sym, bridge_entries()[fn->param_count()]);
    Value* r = b.build_call(entry_sym, Type::i64(), Span<Value* const>(call_args.data(), call_args.size()));
    switch (fn->return_type().kind()) {
        case TypeKind::Void: b.build_ret_void(); break;
        case TypeKind::I8: b.build_ret(b.build_trunc_i8(r)); break;
        case TypeKind::I32: b.build_ret(b.build_trunc_i32(r)); break;
        case TypeKind::F64: b.build_ret(b.build_bitcast_f64_i64(r)); break;
        default: b.build_ret(r); break;
    }
    bf->rebuild_cfg_predecessors();

    // Its invocation hook only counts: the bridge is never tiered up.
    TieringFeedback& fb = tiering().get_feedback(bname);
    fb.set_tier(TierLevel::Tier1_Baseline);
    fb.trigger_bailout("native-to-Tier-0 bridge");
    try {
        bridge->code = std::make_shared<codegen::BaselineCompiledFunction>(baseline_compiler_.compile(*bf));
    } catch (const std::exception& e) {
        std::fprintf(stderr, "brass: the native-to-Tier-0 bridge of '%s' does not compile: %s\n",
                     std::string(name).c_str(), e.what());
        std::fflush(stderr);
        return nullptr;
    }
    void* code = bridge->code->entry_point();
    tier0_bridges_.emplace(std::string(name), std::shared_ptr<void>(bridge));
    return code;
}

} // namespace brass::runtime
