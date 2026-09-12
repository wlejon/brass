#include <brass/object/object_writer.hpp>
#include <brass/target/x64/x64_isel.hpp>
#include <brass/codegen/live_range.hpp>
#include <brass/codegen/linear_scan.hpp>
#include <brass/codegen/peephole.hpp>
#include <brass/codegen/block_layout.hpp>
#include <brass/codegen/instruction_scheduler.hpp>
#include <brass/codegen/software_pipeline.hpp>
#include <brass/mir/loop_opt.hpp>
#include <brass/mir/verifier.hpp>
#include <algorithm>
#include <iostream>

namespace brass::object {

Section* ObjectFile::get_section(std::string_view name) {
    for (auto& sec : sections) {
        if (sec.name == name) {
            return &sec;
        }
    }
    return nullptr;
}

const Section* ObjectFile::get_section(std::string_view name) const {
    for (const auto& sec : sections) {
        if (sec.name == name) {
            return &sec;
        }
    }
    return nullptr;
}

int32_t ObjectFile::get_section_index(std::string_view name) const {
    for (size_t i = 0; i < sections.size(); ++i) {
        if (sections[i].name == name) {
            return static_cast<int32_t>(i);
        }
    }
    return SECTION_UNDEF;
}

Section& ObjectFile::get_or_create_section(
    std::string_view name,
    SectionKind kind,
    SectionFlags flags,
    uint32_t alignment
) {
    for (auto& sec : sections) {
        if (sec.name == name) {
            return sec;
        }
    }
    Section sec;
    sec.name = std::string(name);
    sec.kind = kind;
    sec.flags = flags;
    sec.alignment = alignment;
    sections.push_back(std::move(sec));
    return sections.back();
}

Section& ObjectFile::get_or_create_section(
    std::string_view name,
    SectionKind kind,
    SectionFlags flags
) {
    return get_or_create_section(name, kind, flags, 16);
}

uint32_t ObjectFile::add_symbol(ObjectSymbol sym) {
    for (size_t i = 0; i < symbols.size(); ++i) {
        if (symbols[i].name == sym.name) {
            symbols[i] = std::move(sym);
            return static_cast<uint32_t>(i);
        }
    }
    symbols.push_back(std::move(sym));
    return static_cast<uint32_t>(symbols.size() - 1);
}

const ObjectSymbol* ObjectFile::find_symbol(std::string_view name) const {
    for (const auto& s : symbols) {
        if (s.name == name) {
            return &s;
        }
    }
    return nullptr;
}

ObjectSymbol* ObjectFile::find_symbol(std::string_view name) {
    for (auto& s : symbols) {
        if (s.name == name) {
            return &s;
        }
    }
    return nullptr;
}

ModuleCompiler::ModuleCompiler(const Target& target)
    : target_(target), cc_(CallingConvention::for_target(target)) {}

ModuleCompiler::ModuleCompiler(const Target& target, const CallingConvention& cc)
    : target_(target), cc_(cc) {}

ObjectFile ModuleCompiler::compile(const Module& mod) {
    ObjectFile obj;
    obj.target = target_;

    Section& text_sec = obj.get_or_create_section(
        ".text",
        SectionKind::Text,
        SectionFlags::Read | SectionFlags::Execute | SectionFlags::Alloc,
        16
    );

    for (const auto* fn : mod.functions()) {
        if (!fn) continue;

        // 0. Clone and optimize MIR function
        Module opt_mod(mod.name());
        opt_mod.set_allow_fp_reassociation(mod.allow_fp_reassociation());
        for (std::string_view sym : mod.external_symbols()) {
            opt_mod.add_external_symbol(sym);
        }
        Function* opt_fn = clone_function(*fn, opt_mod);
        LoopOptOptions loop_opts;
        loop_opts.enable_f64_demote = false;
        loop_opts.enable_fp_reassociation = fn->allow_fp_reassociation() || mod.allow_fp_reassociation();
        optimize_function_loops(*opt_fn, loop_opts);
        verify_function(*opt_fn);

        // 1. ISel to LIR
        x64::X64ISel isel(target_, cc_);
        auto lir = isel.lower(*opt_fn);
        if (!lir) continue;

        // 1.5 Loop Software Pipelining (if enabled)
        if (sched_opts_.enable_software_pipelining) {
            codegen::run_software_pipelining(*lir);
        }

        // 1.6 Pre-RA Instruction Scheduling (if enabled)
        if (sched_opts_.enable_pre_ra) {
            codegen::schedule_function(*lir, sched_opts_);
        }

        // 2. Liveness Analysis
        codegen::LivenessAnalysis liveness(*lir);
        liveness.run();

        // 3. Linear Scan Register Allocation
        codegen::LinearScanAllocator regalloc(*lir, liveness, cc_);
        regalloc.allocate();

        // 3.4 Post-RA Instruction Scheduling (if enabled)
        if (sched_opts_.enable_post_ra) {
            codegen::schedule_function(*lir, sched_opts_);
        }

        // 3.5 LIR Trace Scheduling & Fall-Through Block Layout
        if (enable_trace_layout_ || loop_opts.enable_trace_layout) {
            codegen::optimize_block_layout(*lir);
        }

        // 3.6 LIR Peephole Optimization
        codegen::run_lir_peephole_optimizations(*lir);

        // 4. Machine Code Emission
        codegen::EmitContext emit_ctx(*lir, target_);
        codegen::CompilationResult res = emit_ctx.compile();

        // 5. Place in .text
        text_sec.align_to(16);
        size_t fn_offset = text_sec.data.size();
        size_t fn_size = res.code_buffer.size();
        text_sec.emit_bytes(res.code_buffer.span());

        size_t prologue_sz = 0;
        if (!lir->blocks.empty()) {
            auto it = res.block_offsets.find(lir->blocks.front()->id);
            if (it != res.block_offsets.end()) {
                prologue_sz = it->second;
            }
        }

        CompiledFunctionInfo cfi;
        cfi.name = std::string(fn->name());
        cfi.text_offset = fn_offset;
        cfi.text_size = fn_size;
        cfi.prologue_size = prologue_sz;
        cfi.osr_entry_offset = res.osr_entry_offset;
        cfi.frame_info = lir->frame;
        x64::X64FrameLayout::compute_layout(cfi.frame_info, cc_);
        cfi.cc = cc_;
        cfi.safepoints = std::move(res.safepoints);
        cfi.stack_map = std::move(res.stack_map);
        cfi.stack_map.code_offset = static_cast<uint32_t>(fn_offset);
        cfi.stack_map.code_size = static_cast<uint32_t>(fn_size);
        obj.stack_maps.add_function(cfi.stack_map);

        cfi.resume_table = std::move(res.resume_table);
        obj.resume_tables.register_table(cfi.name, cfi.resume_table);

        cfi.exception_table = std::move(res.exception_table);
        cfi.exception_table.set_code_offset(static_cast<uint32_t>(fn_offset));
        obj.exception_tables.register_table(cfi.name, cfi.exception_table);

        for (const auto& ps : res.patch_sites) {
            runtime::PatchSite global_ps = ps;
            global_ps.code_offset += fn_offset;
            obj.patch_sites.register_site(global_ps);
            cfi.patch_sites.push_back(std::move(global_ps));
        }

        obj.functions.push_back(std::move(cfi));
        res.debug_table.set_prologue_size(static_cast<uint32_t>(prologue_sz));
        obj.debug_tables.push_back(std::move(res.debug_table));

        int32_t text_idx = obj.get_section_index(".text");
        ObjectSymbol fn_sym;
        fn_sym.name = std::string(fn->name());
        fn_sym.section_index = text_idx;
        fn_sym.value = fn_offset;
        fn_sym.size = fn_size;
        fn_sym.binding = SymbolBinding::Global;
        fn_sym.type = SymbolType::Function;
        obj.add_symbol(std::move(fn_sym));

        for (const auto& r : res.code_buffer.relocations()) {
            ObjectRelocation obj_r;
            obj_r.offset = fn_offset + r.offset;
            switch (r.kind) {
                case x64::RelocationKind::PCRel32:
                    obj_r.kind = RelocKind::Plt32;
                    break;
                case x64::RelocationKind::Abs64:
                    obj_r.kind = RelocKind::Abs64;
                    break;
                case x64::RelocationKind::SecRel32:
                    obj_r.kind = RelocKind::SecRel32;
                    break;
            }
            obj_r.symbol_name = r.symbol_name;
            obj_r.addend = r.addend;
            text_sec.relocations.push_back(std::move(obj_r));
        }
    }

    // Emit compact binary stack maps to .rdata / .rodata / __const section
    if (!obj.stack_maps.empty()) {
        std::string ro_sec_name = target_.is_windows() ? ".rdata" : (target_.is_macos() ? "__const" : ".rodata");
        Section& ro_sec = obj.get_or_create_section(
            ro_sec_name,
            SectionKind::RoData,
            SectionFlags::Read | SectionFlags::Alloc,
            16
        );
        ro_sec.align_to(16);
        size_t map_offset = ro_sec.data.size();
        std::vector<uint8_t> encoded_maps = encode_stack_maps(obj.stack_maps);
        ro_sec.emit_bytes(encoded_maps);

        int32_t ro_idx = obj.get_section_index(ro_sec_name);
        ObjectSymbol map_sym;
        map_sym.name = "__brass_stack_maps";
        map_sym.section_index = ro_idx;
        map_sym.value = map_offset;
        map_sym.size = encoded_maps.size();
        map_sym.binding = SymbolBinding::Global;
        map_sym.type = SymbolType::Object;
        obj.add_symbol(std::move(map_sym));
    }

    // Emit compact binary debug line section (.brass_dbg)
    obj.debug_context = mod.debug_context();
    if (!obj.debug_tables.empty() && mod.debug_context().file_count() > 0) {
        std::vector<uint8_t> dbg_bytes = serialize_debug_section(mod.debug_context(), obj.debug_tables);
        if (!dbg_bytes.empty()) {
            Section& dbg_sec = obj.get_or_create_section(
                ".brass_dbg",
                SectionKind::Custom,
                SectionFlags::Read,
                8
            );
            dbg_sec.emit_bytes(dbg_bytes);
        }
    }



    // Add any referenced relocation symbol not yet registered
    for (const auto& sec : obj.sections) {
        for (const auto& r : sec.relocations) {
            if (!r.symbol_name.empty() && !obj.find_symbol(r.symbol_name)) {
                ObjectSymbol sym;
                sym.name = r.symbol_name;
                sym.section_index = SECTION_UNDEF;
                sym.value = 0;
                sym.size = 0;
                sym.binding = SymbolBinding::Global;
                sym.type = SymbolType::Function;
                obj.add_symbol(std::move(sym));
            }
        }
    }

    return obj;
}

ObjectFile compile_module_to_object(const Module& mod, const Target& target, const codegen::SchedOptions& sched_opts) {
    ModuleCompiler compiler(target);
    compiler.set_sched_options(sched_opts);
    return compiler.compile(mod);
}

ObjectFile compile_module_to_object(const Module& mod, const Target& target) {
    codegen::SchedOptions default_opts;
    return compile_module_to_object(mod, target, default_opts);
}

} // namespace brass::object
