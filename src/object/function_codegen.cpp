#include "function_codegen.hpp"

#include <brass/core/env.hpp>
#include <brass/core/phase_times.hpp>
#include <brass/codegen/block_layout.hpp>
#include <brass/codegen/instruction_scheduler.hpp>
#include <brass/codegen/linear_scan.hpp>
#include <brass/codegen/live_range.hpp>
#include <brass/codegen/peephole.hpp>
#include <brass/codegen/software_pipeline.hpp>
#include <brass/mir/interrupt_poll.hpp>
#include <brass/mir/loop_opt.hpp>
#include <brass/mir/superopt.hpp>
#include <brass/mir/verifier.hpp>
#include <brass/mir/write_barrier_lowering.hpp>
#include <brass/target/aarch64/aarch64_emit.hpp>
#include <brass/target/aarch64/aarch64_frame.hpp>
#include <brass/target/aarch64/aarch64_isel.hpp>
#include <brass/target/x64/x64_isel.hpp>
#include <fstream>
#include <memory>
#include <stdexcept>

namespace brass::object::detail {

namespace {

// The x64 emitter aligns a loop's head to 32 bytes from the start of its
// function (EmitContext::compile), which holds in memory only where the
// function, and the section under it, start 32-byte aligned too.
constexpr uint32_t kX64FunctionAlignment = 32;
constexpr uint32_t kAArch64FunctionAlignment = 16;

// The calls of `lir` whose callee never reaches a GC point: one declared
// `pure` or `nocollect`, and the write barrier, which by its contract
// (runtime_gc.hpp) only marks a card. The allocator clears the frame's dead
// spill slots and callee-saved registers for a conservative scan only at the
// others: no collection, so no scan of this frame, runs inside these.
void mark_no_gc_calls(codegen::LirFunction& lir, const Module& mod) {
    for (auto& block : lir.blocks) {
        for (auto& inst : block->instructions) {
            if (!inst->is_call() || inst->callee_symbol.empty() || inst->is_patchable) continue;
            const std::string_view sym = inst->callee_symbol;
            inst->no_gc_point = sym == "brass_gc_write_barrier" || mod.has_symbol_role(sym, SymbolRole::Pure) ||
                                mod.has_symbol_role(sym, SymbolRole::NoCollect);
        }
    }
}

const std::string& block_map_path() {
    static const std::string path = env_value("BRASS_BLOCK_MAP");
    return path;
}

// Diagnostic: BRASS_BLOCK_MAP=<file> gets each emitted function's blocks as
// "function<TAB>offset<TAB>block name<TAB>cold", offsets from the function's
// start, in layout order — what maps a profile's hot or out-of-line address
// back to the IR that made it.
std::string block_map_lines(std::string_view fn_name, const codegen::LirFunction& lir,
                            const std::unordered_map<uint32_t, size_t>& offsets) {
    std::string out;
    if (block_map_path().empty()) return out;
    for (const auto& b : lir.blocks) {
        if (!b) continue;
        auto it = offsets.find(b->id);
        if (it == offsets.end()) continue;
        out.append(fn_name).append("\t").append(std::to_string(it->second)).append("\t");
        out.append(b->name).append("\t").append(b->cold ? "1" : "0").append("\n");
    }
    return out;
}

// A private module for the copy of `fn` this compile rewrites.
std::unique_ptr<Module> private_module(const Module& mod) {
    auto m = std::make_unique<Module>(mod.name());
    m->set_allow_fp_reassociation(mod.allow_fp_reassociation());
    m->set_pinned_tls_register(mod.pinned_tls_register());
    m->set_precise_stack_roots(mod.precise_stack_roots());
    // Externs with their roles (passes rely on them: a Pure or Allocator
    // callee, Data symbols) and the module's string data.
    m->copy_declarations_from(mod);
    return m;
}

RelocKind object_kind(x64::RelocationKind k) {
    switch (k) {
        case x64::RelocationKind::PCRel32: return RelocKind::Plt32;
        case x64::RelocationKind::Abs64: return RelocKind::Abs64;
        case x64::RelocationKind::SecRel32: return RelocKind::SecRel32;
        case x64::RelocationKind::GotPCRel32: return RelocKind::GotPCRel32;
    }
    return RelocKind::Plt32;
}

RelocKind object_kind(aarch64::RelocationKind k) {
    switch (k) {
        case aarch64::RelocationKind::Call26:
        case aarch64::RelocationKind::Jump26: return RelocKind::Plt32;
        case aarch64::RelocationKind::Page21: return RelocKind::AdrPage21;
        case aarch64::RelocationKind::AddLo12: return RelocKind::AddLo12;
        case aarch64::RelocationKind::LdSt8Lo12: return RelocKind::LdSt8Lo12;
        case aarch64::RelocationKind::LdSt16Lo12: return RelocKind::LdSt16Lo12;
        case aarch64::RelocationKind::LdSt32Lo12: return RelocKind::LdSt32Lo12;
        case aarch64::RelocationKind::LdSt64Lo12: return RelocKind::LdSt64Lo12;
        case aarch64::RelocationKind::LdSt128Lo12: return RelocKind::LdSt128Lo12;
        case aarch64::RelocationKind::GotPage21: return RelocKind::GotPage21;
        case aarch64::RelocationKind::GotLo12: return RelocKind::GotLo12;
        case aarch64::RelocationKind::Abs64: return RelocKind::Abs64;
    }
    return RelocKind::Plt32;
}

// What both targets' compilation results carry, into `out`.
template <typename Result, typename CodeBuffer>
void take_result(EmittedFunction& out, const Function& fn, const codegen::LirFunction& lir, Result& res,
                 const CodeBuffer& code) {
    out.emitted = true;
    const auto bytes = code.span();
    out.code.assign(bytes.begin(), bytes.end());

    size_t prologue_sz = 0;
    if (!lir.blocks.empty()) {
        auto it = res.block_offsets.find(lir.blocks.front()->id);
        if (it != res.block_offsets.end()) prologue_sz = it->second;
    }

    CompiledFunctionInfo& cfi = out.info;
    cfi.name = std::string(fn.name());
    cfi.text_size = out.code.size();
    cfi.prologue_size = prologue_sz;
    cfi.return_type = fn.return_type();
    cfi.param_types = fn.param_types();
    cfi.frame_info = lir.frame;
    cfi.safepoints = std::move(res.safepoints);
    cfi.stack_map = std::move(res.stack_map);
    cfi.stack_map.code_size = static_cast<uint32_t>(out.code.size());
    cfi.resume_table = std::move(res.resume_table);
    cfi.exception_table = std::move(res.exception_table);
    cfi.patch_sites = std::move(res.patch_sites);

    out.debug_table = std::move(res.debug_table);
    out.debug_table.set_prologue_size(static_cast<uint32_t>(prologue_sz));

    for (const auto& r : code.relocations()) {
        ObjectRelocation obj_r;
        obj_r.offset = r.offset;
        obj_r.kind = object_kind(r.kind);
        obj_r.symbol_name = r.symbol_name;
        obj_r.addend = r.addend;
        out.relocations.push_back(std::move(obj_r));
    }
}

} // namespace

EmittedFunction compile_function(const Module& mod, const Function& fn, const FunctionCodegenConfig& cfg) {
    EmittedFunction out;
    const bool reassoc = fn.allow_fp_reassociation() || mod.allow_fp_reassociation();

    const Function* fn_to_lower = &fn;
    Function* opt_fn_mut = nullptr;
    std::unique_ptr<Module> opt_mod;
    if (const Pipeline* loops = cfg.loop_pipes[reassoc ? 1 : 0]) {
        opt_mod = private_module(mod);
        Function* opt_fn = clone_function(fn, *opt_mod);
        // The optimizer's guarantees hold only for well-formed input; IR the
        // verifier rejects is lowered as given, unoptimized.
        const bool input_valid = verify_function(*opt_fn);
        const bool optimized = input_valid && run_pipeline(*opt_fn, *loops).changed;
        opt_fn->rebuild_cfg_predecessors();
        opt_fn->sort_blocks_rpo();
        opt_fn->rebuild_cfg_predecessors();
        // Lowering broken IR would emit wrong code silently; an optimizer
        // bug has to surface here instead. A function the optimizer left
        // unchanged is the valid input, reordered.
        DiagnosticReporter diag;
        if (optimized && !verify_function(*opt_fn, &diag)) {
            throw std::runtime_error("MIR loop optimization produced invalid IR for '" + std::string(fn.name()) +
                                     "':\n" + diag.format_all());
        }
        fn_to_lower = opt_fn;
        opt_fn_mut = opt_fn;
    }

    // Deopt stress guards go in after the optimizer, into the copy that is
    // lowered (a copy made for them when there is none yet).
    if (has_deopt_stress_guards(*fn_to_lower, &mod, cfg.deopt_stress)) {
        if (!opt_fn_mut) {
            opt_mod = private_module(mod);
            opt_fn_mut = clone_function(fn, *opt_mod);
            fn_to_lower = opt_fn_mut;
        }
        insert_deopt_stress_guards(*opt_fn_mut, &mod, cfg.deopt_stress);
        opt_fn_mut->rebuild_cfg_predecessors();
    }

    // Interrupt polls on the back edges (Module::interrupt_poll), last, so
    // neither the optimizer nor the stress guards see them.
    if (mod.interrupt_poll() && needs_interrupt_polls(*fn_to_lower)) {
        if (!opt_fn_mut) {
            opt_mod = private_module(mod);
            opt_fn_mut = clone_function(fn, *opt_mod);
            fn_to_lower = opt_fn_mut;
        }
        insert_interrupt_polls(*opt_fn_mut);
    }

    // The barriers' inline young-target filter (Module::young_range), after
    // the optimizer too, so its passes see each barrier as one instruction.
    if (needs_write_barrier_expansion(*fn_to_lower)) {
        if (!opt_fn_mut) {
            opt_mod = private_module(mod);
            opt_fn_mut = clone_function(fn, *opt_mod);
            fn_to_lower = opt_fn_mut;
        }
        expand_write_barriers(*opt_fn_mut);
    }

    // 1. ISel to LIR
    std::unique_ptr<codegen::LirFunction> lir;
    {
        PhaseTimeScope timing("backend: isel");
        if (cfg.target.is_aarch64()) {
            aarch64::AArch64ISel isel(cfg.target, cfg.cc);
            isel.set_callee_module(&mod);
            lir = isel.lower(*fn_to_lower);
        } else {
            x64::X64ISel isel(cfg.target, cfg.cc);
            isel.set_callee_module(&mod);
            isel.set_avx(cfg.x64_avx);
            lir = isel.lower(*fn_to_lower);
        }
        if (lir) lir->sort_blocks_rpo();
    }
    if (!lir) return out;
    mark_no_gc_calls(*lir, mod);

    // 1.5 Loop Software Pipelining (if enabled)
    if (cfg.sched.enable_software_pipelining) {
        PhaseTimeScope timing("backend: software pipelining");
        codegen::run_software_pipelining(*lir);
    }

    // 1.6 Pre-RA Instruction Scheduling (if enabled)
    if (cfg.sched.enable_pre_ra) {
        PhaseTimeScope timing("backend: pre-RA scheduling");
        codegen::schedule_function(*lir, cfg.sched);
    }

    {
        PhaseTimeScope timing("backend: liveness + register allocation");
        // 2. Liveness Analysis
        codegen::LivenessAnalysis liveness(*lir);
        liveness.run();

        // 3. Linear Scan Register Allocation
        codegen::LinearScanAllocator regalloc(*lir, liveness, cfg.cc);
        regalloc.allocate();
    }

    // 3.4 Post-RA Instruction Scheduling (if enabled)
    if (cfg.sched.enable_post_ra) {
        PhaseTimeScope timing("backend: post-RA scheduling");
        codegen::schedule_function(*lir, cfg.sched);
    }

    // 3.5 LIR Trace Scheduling & Fall-Through Block Layout
    // (The loop options' enable_trace_layout was always its default, false,
    // here.)
    if (cfg.enable_trace_layout) {
        PhaseTimeScope timing("backend: block layout");
        codegen::optimize_block_layout(*lir);
    }

    // 3.6 LIR Peephole Optimization
    PhaseTimeScope peephole_timing("backend: peephole + emit");
    codegen::run_lir_peephole_optimizations(*lir);
    if (cfg.sched.enable_lir_superopt && cfg.target.is_x64()) {
        SuperoptOptions sopts;
        sopts.lir_check_flags = true;
        for (auto& blk : lir->blocks) {
            if (blk && blk->instructions.size() <= 128) run_x64_lir_superopt(*blk, sopts);
        }
    }

    if (cfg.target.is_aarch64()) {
        // 4. AArch64 Machine Code Emission
        aarch64::AArch64EmitContext emit_ctx(*lir, cfg.target);
        aarch64::AArch64CompilationResult res = emit_ctx.compile();
        out.alignment = kAArch64FunctionAlignment;
        take_result(out, fn, *lir, res, res.code_buffer);
        aarch64::AArch64FrameLayout::compute_layout(out.info.frame_info, cfg.cc);
    } else {
        // 4. Machine Code Emission
        codegen::EmitContext emit_ctx(*lir, cfg.target);
        codegen::CompilationResult res = emit_ctx.compile();
        out.alignment = kX64FunctionAlignment;
        out.block_map = block_map_lines(fn.name(), *lir, res.block_offsets);
        out.info.stack_adjust_regions = std::move(res.stack_adjust_regions);
        take_result(out, fn, *lir, res, res.code_buffer);
        x64::X64FrameLayout::compute_layout(out.info.frame_info, cfg.cc);
    }
    out.info.cc = cfg.cc;
    return out;
}

void place_function(ObjectPlacement& at, EmittedFunction&& fn) {
    ObjectFile& obj = at.obj;
    Section& text = at.text;
    text.align_to(fn.alignment);
    const size_t fn_offset = text.data.size();
    text.emit_bytes(fn.code);

    CompiledFunctionInfo& cfi = fn.info;
    cfi.text_offset = fn_offset;
    cfi.stack_map.code_offset = static_cast<uint32_t>(fn_offset);
    // Stack maps appended by name without add_function's linear search for
    // an existing map (quadratic over a large module); a repeated name still
    // replaces through it.
    if (at.mapped.insert(cfi.stack_map.function_name).second) obj.stack_maps.functions().push_back(cfi.stack_map);
    else obj.stack_maps.add_function(cfi.stack_map);

    obj.resume_tables.register_table(cfi.name, cfi.resume_table);

    cfi.exception_table.set_code_offset(static_cast<uint32_t>(fn_offset));
    obj.exception_tables.register_table(cfi.name, cfi.exception_table);

    for (runtime::PatchSite& ps : cfi.patch_sites) {
        ps.code_offset += fn_offset;
        obj.patch_sites.register_site(ps);
    }

    ObjectSymbol fn_sym;
    fn_sym.name = cfi.name;
    fn_sym.section_index = obj.get_section_index(".text");
    fn_sym.value = fn_offset;
    fn_sym.size = fn.code.size();
    fn_sym.binding = SymbolBinding::Global;
    fn_sym.type = SymbolType::Function;

    obj.functions.push_back(std::move(cfi));
    obj.debug_tables.push_back(std::move(fn.debug_table));
    obj.add_symbol(std::move(fn_sym));

    for (ObjectRelocation& r : fn.relocations) {
        r.offset += fn_offset;
        text.relocations.push_back(std::move(r));
    }
}

void write_block_map(const std::string& lines) {
    if (lines.empty() || block_map_path().empty()) return;
    static std::ofstream out(block_map_path(), std::ios::app);
    out << lines;
}

} // namespace brass::object::detail
