#include <brass/object/object_writer.hpp>
#include <brass/object/aarch64_reloc.hpp>
#include <brass/target/x64/x64_isel.hpp>
#include <brass/target/aarch64/aarch64_isel.hpp>
#include <brass/target/aarch64/aarch64_emit.hpp>
#include <brass/target/aarch64/aarch64_frame.hpp>
#include <brass/codegen/live_range.hpp>
#include <brass/codegen/linear_scan.hpp>
#include <brass/codegen/peephole.hpp>
#include <brass/codegen/block_layout.hpp>
#include <brass/codegen/instruction_scheduler.hpp>
#include <brass/codegen/software_pipeline.hpp>
#include <brass/mir/loop_opt.hpp>
#include <brass/mir/verifier.hpp>
#include <algorithm>
#include <cstring>
#include <stdexcept>
#include <iostream>
#include <fstream>

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

void ObjectFile::sync_symbol_index() const {
    if (symbol_index_.size() == symbols.size()) return;
    symbol_index_.clear();
    symbol_index_.reserve(symbols.size());
    for (size_t i = 0; i < symbols.size(); ++i) {
        // The FIRST entry of a name wins, which is what the scan answered.
        symbol_index_.try_emplace(symbols[i].name, static_cast<uint32_t>(i));
    }
}

uint32_t ObjectFile::add_symbol(ObjectSymbol sym) {
    sync_symbol_index();
    if (auto it = symbol_index_.find(std::string_view(sym.name)); it != symbol_index_.end()) {
        symbols[it->second] = std::move(sym);
        return it->second;
    }
    const uint32_t index = static_cast<uint32_t>(symbols.size());
    symbol_index_.emplace(sym.name, index);
    symbols.push_back(std::move(sym));
    return index;
}

const ObjectSymbol* ObjectFile::find_symbol(std::string_view name) const {
    sync_symbol_index();
    auto it = symbol_index_.find(name);
    return it == symbol_index_.end() ? nullptr : &symbols[it->second];
}

ObjectSymbol* ObjectFile::find_symbol(std::string_view name) {
    sync_symbol_index();
    auto it = symbol_index_.find(name);
    return it == symbol_index_.end() ? nullptr : &symbols[it->second];
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

        LoopOptOptions loop_opts;
        loop_opts.enable_f64_demote = false;
        loop_opts.enable_fp_reassociation = fn->allow_fp_reassociation() || mod.allow_fp_reassociation();

        const Function* fn_to_lower = fn;
        std::unique_ptr<Module> opt_mod;
        if (enable_mir_opts_ && !mod.has_loop_optimizations()) {
            opt_mod = std::make_unique<Module>(mod.name());
            opt_mod->set_allow_fp_reassociation(mod.allow_fp_reassociation());
            opt_mod->set_pinned_tls_register(mod.pinned_tls_register());
            // Externs with their roles (passes rely on them: a Pure or
            // Allocator callee, Data symbols) and the module's string data.
            opt_mod->copy_declarations_from(mod);
            Function* opt_fn = clone_function(*fn, *opt_mod);
            // The optimizer's guarantees hold only for well-formed input;
            // IR the verifier rejects is lowered as given, unoptimized.
            const bool input_valid = verify_function(*opt_fn);
            if (input_valid) optimize_function_loops(*opt_fn, loop_opts);
            opt_fn->rebuild_cfg_predecessors();
            opt_fn->sort_blocks_rpo();
            opt_fn->rebuild_cfg_predecessors();
            // Lowering broken IR would emit wrong code silently; an
            // optimizer bug has to surface here instead.
            DiagnosticReporter diag;
            if (input_valid && !verify_function(*opt_fn, &diag)) {
                throw std::runtime_error("MIR loop optimization produced invalid IR for '" +
                                         std::string(fn->name()) + "':\n" + diag.format_all());
            }
            fn_to_lower = opt_fn;
        }

        // 1. ISel to LIR
        std::unique_ptr<codegen::LirFunction> lir;
        if (target_.is_aarch64()) {
            aarch64::AArch64ISel isel(target_, cc_);
            isel.set_callee_module(&mod);
            lir = isel.lower(*fn_to_lower);
        } else {
            x64::X64ISel isel(target_, cc_);
            isel.set_callee_module(&mod);
            lir = isel.lower(*fn_to_lower);
        }
        if (!lir) continue;
        lir->sort_blocks_rpo();

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

        if (target_.is_aarch64()) {
            // 4. AArch64 Machine Code Emission
            aarch64::AArch64EmitContext emit_ctx(*lir, target_);
            aarch64::AArch64CompilationResult res = emit_ctx.compile();

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
            cfi.return_type = fn->return_type();
            cfi.param_types = fn->param_types();
            cfi.frame_info = lir->frame;
            aarch64::AArch64FrameLayout::compute_layout(cfi.frame_info, cc_);
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
                    case aarch64::RelocationKind::Call26:
                    case aarch64::RelocationKind::Jump26:
                        obj_r.kind = RelocKind::Plt32;
                        break;
                    case aarch64::RelocationKind::Page21:      obj_r.kind = RelocKind::AdrPage21; break;
                    case aarch64::RelocationKind::AddLo12:     obj_r.kind = RelocKind::AddLo12; break;
                    case aarch64::RelocationKind::LdSt8Lo12:   obj_r.kind = RelocKind::LdSt8Lo12; break;
                    case aarch64::RelocationKind::LdSt16Lo12:  obj_r.kind = RelocKind::LdSt16Lo12; break;
                    case aarch64::RelocationKind::LdSt32Lo12:  obj_r.kind = RelocKind::LdSt32Lo12; break;
                    case aarch64::RelocationKind::LdSt64Lo12:  obj_r.kind = RelocKind::LdSt64Lo12; break;
                    case aarch64::RelocationKind::LdSt128Lo12: obj_r.kind = RelocKind::LdSt128Lo12; break;
                    case aarch64::RelocationKind::GotPage21:   obj_r.kind = RelocKind::GotPage21; break;
                    case aarch64::RelocationKind::GotLo12:     obj_r.kind = RelocKind::GotLo12; break;
                    case aarch64::RelocationKind::Abs64:
                        obj_r.kind = RelocKind::Abs64;
                        break;
                }
                obj_r.symbol_name = r.symbol_name;
                obj_r.addend = r.addend;
                text_sec.relocations.push_back(std::move(obj_r));
            }
        } else {
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
            cfi.return_type = fn->return_type();
            cfi.param_types = fn->param_types();
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
                    case x64::RelocationKind::GotPCRel32:
                        obj_r.kind = RelocKind::GotPCRel32;
                        break;
                }
                obj_r.symbol_name = r.symbol_name;
                obj_r.addend = r.addend;
                text_sec.relocations.push_back(std::move(obj_r));
            }
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
    obj.module_name = std::string(mod.name());
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

    // The module's own string data (Module::define_string_symbol), so the
    // symbols resolve in whatever loads this object, JIT or linker.
    if (!mod.string_symbols().empty()) {
        std::string ro_sec_name = target_.is_windows() ? ".rdata" : (target_.is_macos() ? "__const" : ".rodata");
        Section& ro_sec = obj.get_or_create_section(
            ro_sec_name,
            SectionKind::RoData,
            SectionFlags::Read | SectionFlags::Alloc,
            16
        );
        int32_t ro_idx = obj.get_section_index(ro_sec_name);
        for (const auto& [sym_name, text] : mod.string_symbols()) {
            size_t offset = ro_sec.data.size();
            std::vector<uint8_t> bytes(text.begin(), text.end());
            bytes.push_back(0);
            ro_sec.emit_bytes(bytes);
            ObjectSymbol sym;
            sym.name = std::string(sym_name);
            sym.section_index = ro_idx;
            sym.value = offset;
            sym.size = bytes.size();
            sym.binding = SymbolBinding::Local;
            sym.type = SymbolType::Object;
            obj.add_symbol(std::move(sym));
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

namespace {

// `local_only`: the symbol must also be non-preemptible (local binding, or a
// section), as a relocatable object for a system linker needs.
bool names_defined(const ObjectFile& obj, std::string_view name, bool local_only) {
    if (const auto* sym = obj.find_symbol(name)) {
        if (sym->section_index >= 0 && (!local_only || sym->binding == SymbolBinding::Local)) {
            return true;
        }
    }
    for (const auto& sec : obj.sections) {
        if (sec.name == name) return true;
    }
    return false;
}

} // namespace

size_t relax_got_loads(ObjectFile& obj, bool local_only) {
    size_t left = 0;
    for (auto& sec : obj.sections) {
        for (auto& r : sec.relocations) {
            if (r.kind == RelocKind::GotPage21 || r.kind == RelocKind::GotLo12) {
                // AArch64: both halves of a pair decide on the symbol alone,
                // so they always agree. ADRP of the slot's page becomes ADRP
                // of the symbol's; the slot load becomes an ADD.
                if (!names_defined(obj, r.symbol_name, local_only)) {
                    if (r.kind == RelocKind::GotLo12) ++left;
                    continue;
                }
                if (r.kind == RelocKind::GotPage21) {
                    r.kind = RelocKind::AdrPage21;
                    continue;
                }
                if (r.offset + 4 > sec.data.size()) {
                    throw std::runtime_error("relax_got_loads: GOT load of '" + r.symbol_name +
                                             "' lies outside section " + sec.name);
                }
                uint32_t inst = 0;
                std::memcpy(&inst, sec.data.data() + r.offset, 4);
                if (!a64::relax_got_ldr_to_add(inst)) {
                    throw std::runtime_error("relax_got_loads: GOT page-offset relocation against '" +
                                             r.symbol_name + "' in " + sec.name +
                                             " is not on an LDR Xt, [Xn, #imm]");
                }
                std::memcpy(sec.data.data() + r.offset, &inst, 4);
                r.kind = RelocKind::AddLo12;
                continue;
            }
            if (r.kind != RelocKind::GotPCRel32) continue;
            // The opcode byte sits two before the displacement: REX, 8B,
            // ModRM(mod=00 reg=r rm=101), disp32.
            const bool is_mov = r.offset >= 2 && r.offset + 4 <= sec.data.size() &&
                                sec.data[r.offset - 2] == 0x8B;
            if (!is_mov || !names_defined(obj, r.symbol_name, local_only)) {
                ++left;
                continue;
            }
            sec.data[r.offset - 2] = 0x8D;
            r.kind = RelocKind::PCRel32;
        }
    }
    return left;
}

void materialize_got_slots(ObjectFile& obj, std::string_view slot_section, SectionKind kind,
                           SectionFlags flags) {
    if (relax_got_loads(obj) == 0) return;
    // The slots go at the end of the section, after whatever it holds.
    Section& slots = obj.get_or_create_section(slot_section, kind, flags, 8);
    if (slots.alignment < 8) slots.alignment = 8;
    const int32_t slots_index = obj.get_section_index(slot_section);
    std::unordered_map<std::string, std::string> slot_of;   // symbol -> slot symbol
    for (auto& sec : obj.sections) {
        if (&sec == &slots) continue;
        for (auto& r : sec.relocations) {
            if (r.kind != RelocKind::GotPCRel32 && r.kind != RelocKind::GotPage21 &&
                r.kind != RelocKind::GotLo12) {
                continue;
            }
            auto it = slot_of.find(r.symbol_name);
            if (it == slot_of.end()) {
                slots.align_to(8);
                const size_t at = slots.data.size();
                slots.relocations.push_back({at, RelocKind::Abs64, r.symbol_name, 0, 0});
                slots.emit64(0);
                ObjectSymbol slot_sym;
                slot_sym.name = r.symbol_name + "$got";
                slot_sym.section_index = slots_index;
                slot_sym.value = at;
                slot_sym.size = 8;
                slot_sym.binding = SymbolBinding::Local;
                slot_sym.type = SymbolType::Object;
                obj.add_symbol(std::move(slot_sym));
                it = slot_of.emplace(r.symbol_name, r.symbol_name + "$got").first;
            }
            // x64: the load's disp32 now addresses the slot. AArch64: ADRP of
            // the slot's page, and the LDR X of the slot itself.
            r.kind = r.kind == RelocKind::GotPage21 ? RelocKind::AdrPage21
                   : r.kind == RelocKind::GotLo12   ? RelocKind::LdSt64Lo12
                                                    : RelocKind::PCRel32;
            r.symbol_name = it->second;
        }
    }
}

} // namespace brass::object
