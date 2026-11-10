#include <brass/object/object_writer.hpp>
#include <brass/core/parallel_for.hpp>
#include <brass/object/aarch64_reloc.hpp>
#include <brass/mir/loop_opt.hpp>
#include "function_codegen.hpp"
#include <algorithm>
#include <memory>
#include <unordered_set>
#include <cstdlib>
#include <cstring>
#include <stdexcept>

namespace brass::object {

namespace {
// The x64 emitter aligns a loop's head to 32 bytes from the start of its
// function (EmitContext::compile), which holds in memory only where the
// function, and the section under it, start 32-byte aligned too.
constexpr uint32_t kTextAlignment = 32;
} // namespace

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
        kTextAlignment
    );

    // The loop pipeline, built once per FP-reassociation setting rather than
    // once per function (a module of thousands of small functions spent
    // more building step lists than running most of them).
    detail::FunctionCodegenConfig cfg;
    cfg.target = target_;
    cfg.cc = cc_;
    cfg.sched = sched_opts_;
    cfg.deopt_stress = deopt_stress_;
    cfg.enable_trace_layout = enable_trace_layout_;
    cfg.x64_avx = x64_avx_;
    std::unique_ptr<Pipeline> loop_pipes[2];
    if (enable_mir_opts_ && !mod.has_loop_optimizations()) {
        for (int reassoc = 0; reassoc < 2; ++reassoc) {
            LoopOptOptions loop_opts;
            loop_opts.enable_f64_demote = false;
            loop_opts.enable_fp_reassociation = reassoc != 0;
            loop_pipes[reassoc] = std::make_unique<Pipeline>(loop_pipeline(loop_opts));
            cfg.loop_pipes[reassoc] = loop_pipes[reassoc].get();
        }
    }

    // Every function compiled on its own (detail::compile_function reads the
    // module and writes only its result), then placed in module order: the
    // object is the same whatever the thread count.
    std::vector<const Function*> fns;
    std::vector<size_t> weights;
    size_t total_weight = 0;
    for (const Function* fn : mod.functions()) {
        if (!fn) continue;
        fns.push_back(fn);
        weights.push_back(fn->block_count() + 1);
        total_weight += weights.back();
    }
    std::vector<detail::EmittedFunction> emitted(fns.size());
    const size_t threads = threads_for_work(threads_, fns.size(), total_weight, min_blocks_per_thread_);
    parallel_for_each(heaviest_first(weights), threads, [&](size_t i) {
        emitted[i] = detail::compile_function(mod, *fns[i], cfg);
    });

    // Stack maps appended by name without add_function's linear search for
    // an existing map (quadratic over a large module); a repeated name still
    // replaces through it.
    std::unordered_set<std::string> mapped;
    detail::ObjectPlacement placement{obj, text_sec, mapped};
    for (detail::EmittedFunction& fn : emitted) {
        if (!fn.emitted) continue;
        detail::write_block_map(fn.block_map);
        detail::place_function(placement, std::move(fn));
    }

    // Emit compact binary stack maps to .rdata / .rodata / __const section
    if (emit_stack_map_symbol_ && !obj.stack_maps.empty()) {
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
