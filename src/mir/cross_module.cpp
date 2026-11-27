#include <brass/mir/cross_module.hpp>

#include <brass/core/diagnostics.hpp>
#include <brass/mir/block.hpp>
#include <brass/mir/builder.hpp>
#include <brass/mir/cfg_simplify.hpp>
#include <brass/mir/gc_refs.hpp>
#include <brass/mir/gvn.hpp>
#include <brass/mir/inline_transform.hpp>
#include <brass/mir/loop_opt.hpp>
#include <brass/mir/opcodes.hpp>
#include <brass/mir/runtime_symbols.hpp>
#include <brass/mir/scalar_opt.hpp>
#include <brass/mir/untag_reuse.hpp>
#include <brass/mir/uses.hpp>
#include <brass/mir/verifier.hpp>

#include <algorithm>
#include <sstream>
#include <unordered_map>
#include <unordered_set>

namespace brass {

namespace {

std::string format_base_loc(const DebugContext& ctx, DebugLoc loc) {
    if (!loc.is_valid()) return {};
    const std::string& file = ctx.get_file(loc.file_id);
    std::ostringstream oss;
    oss << (file.empty() ? "<unknown>" : file) << ":" << loc.line;
    if (loc.column > 0) oss << ":" << loc.column;
    return oss.str();
}

uint32_t remap_file_id(Module& dst, const Module& src, uint32_t src_file_id) {
    if (src_file_id == 0) return 0;
    const std::string& path = src.debug_context().get_file(src_file_id);
    if (path.empty()) return 0;
    return dst.debug_context().get_or_add_file(path);
}

uint32_t remap_scope_id(
    Module& dst,
    const Module& src,
    uint32_t src_scope_id,
    std::unordered_map<uint32_t, uint32_t>& cache) {
    if (src_scope_id == 0) return 0;
    auto it = cache.find(src_scope_id);
    if (it != cache.end()) return it->second;

    const InlinedScope* scope = src.debug_context().get_inlined_scope(src_scope_id);
    if (!scope) return 0;

    uint32_t dst_parent = 0;
    if (scope->parent_inlined_at_id != 0) {
        dst_parent = remap_scope_id(dst, src, scope->parent_inlined_at_id, cache);
    }
    DebugLoc callsite = scope->callsite_loc;
    callsite.file_id = remap_file_id(dst, src, callsite.file_id);
    callsite.inlined_at_id = dst_parent;
    const uint32_t dst_id = dst.debug_context().record_inlined_scope(scope->callee_name, callsite, dst_parent);
    cache[src_scope_id] = dst_id;
    return dst_id;
}

std::string unique_dest_name(const Module& dst, std::string_view base) {
    std::string candidate(base);
    if (!dst.get_function(candidate) && !dst.has_external_symbol(candidate)) {
        return candidate;
    }
    for (uint32_t i = 1;; ++i) {
        std::string next = std::string(base) + "__imp" + std::to_string(i);
        if (!dst.get_function(next) && !dst.has_external_symbol(next)) {
            return next;
        }
    }
}

bool is_gc_or_tagged_type(Type t) noexcept {
    return t.kind() == TypeKind::GCRef || t.kind() == TypeKind::Tagged;
}

size_t count_instructions(const Function& fn) noexcept {
    size_t n = 0;
    for (const BasicBlock* bb : fn.blocks()) {
        if (bb) n += bb->instruction_count();
    }
    return n;
}

std::string build_audit_diagnostic(const std::vector<DomainViolation>& violations) {
    if (violations.empty()) return {};
    std::ostringstream oss;
    for (size_t i = 0; i < violations.size(); ++i) {
        if (i > 0) oss << "; ";
        if (!violations[i].formatted_loc.empty()) {
            oss << violations[i].formatted_loc << ": ";
        }
        oss << violations[i].detail;
    }
    return oss.str();
}

Value* emit_param_conversion(Builder& b, Value* val, BoundaryConv conv, const NaNBoxSchema& nb) {
    switch (conv) {
        case BoundaryConv::Identity:
            return val;
        case BoundaryConv::UnboxedF64ToTagged:
            return b.build_bitcast_tagged_i64(b.build_bitcast_i64_f64(val));
        case BoundaryConv::UnboxedF32ToTaggedF64:
            return b.build_bitcast_tagged_i64(b.build_bitcast_i64_f64(b.build_fpext_f64_f32(val)));
        case BoundaryConv::UnboxedI32ToTagged: {
            Value* ext = b.build_zext_i64(val);
            Value* tag = b.build_iconst_i64(static_cast<int64_t>(nb.int32_tag));
            return b.build_bitcast_tagged_i64(b.build_or(ext, tag));
        }
        case BoundaryConv::UnboxedI64ToTagged:
            return b.build_bitcast_tagged_i64(val);
        default:
            return nullptr;
    }
}

Value* emit_return_conversion(Builder& b, Value* val, BoundaryConv conv) {
    switch (conv) {
        case BoundaryConv::Identity:
            return val;
        case BoundaryConv::TaggedToUnboxedF64:
            return b.build_bitcast_f64_i64(b.build_bitcast_i64_tagged(val));
        case BoundaryConv::TaggedToUnboxedF32:
            return b.build_fptrunc_f32_f64(b.build_bitcast_f64_i64(b.build_bitcast_i64_tagged(val)));
        case BoundaryConv::TaggedToUnboxedI32:
            return b.build_trunc_i32(b.build_bitcast_i64_tagged(val));
        case BoundaryConv::TaggedToUnboxedI64:
            return b.build_bitcast_i64_tagged(val);
        default:
            return nullptr;
    }
}

bool infer_param_conv(Type caller_t, Type callee_t, bool allow_nanbox, BoundaryConv& out) noexcept {
    if (caller_t == callee_t) {
        out = BoundaryConv::Identity;
        return true;
    }
    if (!allow_nanbox || callee_t.kind() != TypeKind::Tagged) {
        return false;
    }
    switch (caller_t.kind()) {
        case TypeKind::F64: out = BoundaryConv::UnboxedF64ToTagged; return true;
        case TypeKind::F32: out = BoundaryConv::UnboxedF32ToTaggedF64; return true;
        case TypeKind::I32: out = BoundaryConv::UnboxedI32ToTagged; return true;
        case TypeKind::I64: out = BoundaryConv::UnboxedI64ToTagged; return true;
        default: return false;
    }
}

bool infer_return_conv(Type caller_t, Type callee_t, bool allow_nanbox, BoundaryConv& out) noexcept {
    if (caller_t == callee_t) {
        out = BoundaryConv::Identity;
        return true;
    }
    if (!allow_nanbox || callee_t.kind() != TypeKind::Tagged) {
        return false;
    }
    switch (caller_t.kind()) {
        case TypeKind::F64: out = BoundaryConv::TaggedToUnboxedF64; return true;
        case TypeKind::F32: out = BoundaryConv::TaggedToUnboxedF32; return true;
        case TypeKind::I32: out = BoundaryConv::TaggedToUnboxedI32; return true;
        case TypeKind::I64: out = BoundaryConv::TaggedToUnboxedI64; return true;
        default: return false;
    }
}

} // namespace

std::string format_debug_loc(const DebugContext& ctx, DebugLoc loc) {
    if (!loc.is_valid()) return {};
    std::string out = format_base_loc(ctx, loc);
    uint32_t scope_id = loc.inlined_at_id;
    int guard = 0;
    while (scope_id != 0 && guard++ < 64) {
        const InlinedScope* s = ctx.get_inlined_scope(scope_id);
        if (!s) break;
        out += " [inlined in " + s->callee_name;
        const std::string callsite = format_base_loc(ctx, s->callsite_loc);
        if (!callsite.empty()) out += " at " + callsite;
        out += "]";
        scope_id = s->parent_inlined_at_id;
    }
    return out;
}

void remap_function_debug_locs(Module& dst, const Module& src, Function& cloned_fn) {
    std::unordered_map<uint32_t, uint32_t> scope_cache;
    for (BasicBlock* bb : cloned_fn.blocks()) {
        if (!bb) continue;
        for (Instruction* inst : *bb) {
            if (!inst) continue;
            DebugLoc loc = inst->loc();
            if (!loc.is_valid() && loc.inlined_at_id == 0) continue;
            loc.file_id = remap_file_id(dst, src, loc.file_id);
            if (loc.inlined_at_id != 0) {
                loc.inlined_at_id = remap_scope_id(dst, src, loc.inlined_at_id, scope_cache);
            }
            inst->set_loc(loc);
        }
    }
}

CrossModuleLinkResult import_function_into_module(
    Module& dst,
    const Module& src,
    std::string_view fn_name,
    const CrossModuleLinkOptions& opts) {
    CrossModuleLinkResult res;
    if (fn_name.empty()) {
        res.diagnostic = "import_function_into_module: empty function name";
        return res;
    }

    const Function* root = src.get_function(fn_name);
    if (!root) {
        res.diagnostic = "import_function_into_module: function '" + std::string(fn_name) + "' not found in source module";
        return res;
    }

    Module::BodyRead read(src);

    struct Pending {
        const Function* src_fn;
        size_t depth;
    };
    std::vector<Pending> worklist;
    std::vector<const Function*> ordered_funcs;
    std::unordered_map<std::string_view, std::string> name_map;
    std::unordered_set<const Function*> visited;

    auto allocate_dst_name = [&](std::string_view src_name) -> std::string {
        std::string desired = opts.symbol_prefix.empty()
            ? std::string(src_name)
            : (opts.symbol_prefix + std::string(src_name));
        if (opts.allow_overwrite) {
            if (Function* existing = dst.get_function(desired)) {
                dst.remove_function(existing);
            }
            return desired;
        }
        return unique_dest_name(dst, desired);
    };

    if (root->is_lazy()) src.materialize(*root);
    if (root->block_count() == 0) {
        res.diagnostic = "import_function_into_module: root function '" + std::string(fn_name) + "' has no body";
        return res;
    }

    visited.insert(root);
    worklist.push_back({root, 0});
    ordered_funcs.push_back(root);
    name_map[root->name()] = allocate_dst_name(root->name());

    std::unordered_set<std::string_view> referenced_externs;
    std::unordered_set<std::string_view> referenced_string_syms;

    for (size_t idx = 0; idx < worklist.size(); ++idx) {
        const Pending cur = worklist[idx];
        for (const BasicBlock* bb : cur.src_fn->blocks()) {
            if (!bb) continue;
            for (const Instruction* inst : *bb) {
                if (!inst) continue;
                auto inspect_sym = [&](std::string_view sym) {
                    if (sym.empty()) return;
                    if (src.string_symbol(sym) != nullptr) {
                        referenced_string_syms.insert(sym);
                    }
                    const Function* target = src.get_function(sym);
                    if (target && opts.import_callees && cur.depth < opts.max_callee_depth) {
                        if (target->is_lazy()) src.materialize(*target);
                        if (target->block_count() > 0) {
                            if (visited.insert(target).second) {
                                worklist.push_back({target, cur.depth + 1});
                                ordered_funcs.push_back(target);
                                name_map[target->name()] = allocate_dst_name(target->name());
                            }
                            return;
                        }
                    }
                    if (!target || src.has_external_symbol(sym) || !src.symbol_roles(sym).empty()) {
                        referenced_externs.insert(sym);
                    }
                };
                inspect_sym(inst->symbol());
                inspect_sym(inst->extra_symbol());
            }
        }
    }

    // Verify and copy referenced string symbols before mutating dst's function list cleanly.
    for (std::string_view sym : referenced_string_syms) {
        const char* src_text = src.string_symbol(sym);
        const char* dst_text = dst.string_symbol(sym);
        if (src_text && dst_text && std::string_view(src_text) != std::string_view(dst_text)) {
            res.diagnostic = "import_function_into_module: string symbol '" + std::string(sym) +
                             "' collides with differing text in destination module";
            return res;
        }
    }
    for (std::string_view sym : referenced_string_syms) {
        const char* src_text = src.string_symbol(sym);
        if (src_text && dst.string_symbol(sym) == nullptr) {
            dst.define_string_symbol(sym, src_text);
            res.imported_symbols.emplace_back(sym);
        }
    }

    // Copy external symbols & SymbolRoles referenced by the imported closure.
    for (std::string_view sym : referenced_externs) {
        if (name_map.count(sym)) continue;
        if (src.has_external_symbol(sym) && !dst.has_external_symbol(sym)) {
            dst.add_external_symbol(sym);
            res.imported_symbols.emplace_back(sym);
        }
        for (SymbolRole role : src.symbol_roles(sym)) {
            if (!dst.has_symbol_role(sym, role)) {
                dst.add_symbol_role(sym, role);
            }
        }
    }

    // Clone functions into dst, rewrite internal symbol references, and remap DebugContext IDs.
    std::vector<Function*> cloned;
    cloned.reserve(ordered_funcs.size());
    for (const Function* sf : ordered_funcs) {
        const std::string& dname = name_map.at(sf->name());
        Function* df = clone_function(*sf, dst, dname);
        if (!df) {
            res.diagnostic = "import_function_into_module: clone_function failed for '" + std::string(sf->name()) + "'";
            return res;
        }
        for (BasicBlock* bb : df->blocks()) {
            if (!bb) continue;
            for (Instruction* inst : *bb) {
                if (!inst) continue;
                if (!inst->symbol().empty()) {
                    auto it = name_map.find(inst->symbol());
                    if (it != name_map.end()) {
                        inst->set_symbol(dst.string_pool().intern(it->second));
                    }
                }
                if (!inst->extra_symbol().empty()) {
                    auto it = name_map.find(inst->extra_symbol());
                    if (it != name_map.end()) {
                        inst->set_extra_symbol(dst.string_pool().intern(it->second));
                    }
                }
            }
        }
        remap_function_debug_locs(dst, src, *df);
        cloned.push_back(df);
    }

    res.ok = true;
    res.imported = cloned.front();
    for (size_t i = 1; i < cloned.size(); ++i) {
        res.imported_callees.push_back(cloned[i]);
    }
    return res;
}

DomainAuditReport audit_function_domain(
    const Function& fn,
    const Module& mod,
    const DomainAuditOptions& opts) {
    DomainAuditReport report;
    const DebugContext& dbg = mod.debug_context();

    auto add_v = [&](DomainViolationKind kind, DebugLoc loc, std::string detail) {
        DomainViolation v;
        v.kind = kind;
        v.function_name = std::string(fn.name());
        v.loc = loc;
        v.formatted_loc = format_debug_loc(dbg, loc);
        v.detail = std::move(detail);
        report.violations.push_back(std::move(v));
        report.ok = false;
    };

    if (opts.require_no_gc) {
        if (is_gc_or_tagged_type(fn.return_type())) {
            add_v(fn.return_type().kind() == TypeKind::Tagged ? DomainViolationKind::TaggedType : DomainViolationKind::GcRefType,
                  DebugLoc(), "function returns GC/tagged type");
        }
        for (size_t i = 0; i < fn.param_count(); ++i) {
            if (is_gc_or_tagged_type(fn.param_type(i))) {
                add_v(fn.param_type(i).kind() == TypeKind::Tagged ? DomainViolationKind::TaggedType : DomainViolationKind::GcRefType,
                      DebugLoc(), "parameter " + std::to_string(i) + " has GC/tagged type");
            }
        }
    }

    for (const BasicBlock* bb : fn.blocks()) {
        if (!bb) continue;
        if (opts.require_no_gc) {
            for (const Value* p : bb->params()) {
                if (p && is_gc_or_tagged_type(p->type())) {
                    add_v(p->type().kind() == TypeKind::Tagged ? DomainViolationKind::TaggedType : DomainViolationKind::GcRefType,
                          DebugLoc(), "block '" + std::string(bb->name()) + "' has GC/tagged block parameter");
                }
            }
        }
        for (const Instruction* inst : *bb) {
            if (!inst) continue;
            const Opcode op = inst->opcode();
            const DebugLoc loc = inst->loc();

            if (opts.require_no_gc) {
                if (inst->produces_value() && is_gc_or_tagged_type(inst->type())) {
                    add_v(inst->type().kind() == TypeKind::Tagged ? DomainViolationKind::TaggedType : DomainViolationKind::GcRefType,
                          loc, "instruction '" + std::string(opcode_name(op)) + "' produces GC/tagged value");
                } else {
                    for (const Value* ov : inst->operands()) {
                        if (ov && is_gc_or_tagged_type(ov->type())) {
                            add_v(ov->type().kind() == TypeKind::Tagged ? DomainViolationKind::TaggedType : DomainViolationKind::GcRefType,
                                  loc, "instruction '" + std::string(opcode_name(op)) + "' consumes GC/tagged value");
                            break;
                        }
                    }
                }
                if (op == Opcode::write_barrier || op == Opcode::safepoint) {
                    add_v(DomainViolationKind::WriteBarrier, loc,
                          "GC instruction '" + std::string(opcode_name(op)) + "' forbidden in unboxed domain");
                }
            }

            if (opts.require_no_pinned_tls &&
                (op == Opcode::pinned_tls_read || op == Opcode::pinned_tls_write || op == Opcode::read_sp)) {
                add_v(DomainViolationKind::PinnedTls, loc,
                      "pinned TLS/stack register op '" + std::string(opcode_name(op)) + "' forbidden");
            }

            if (opts.require_no_deopt && (op == Opcode::guard || op == Opcode::resume_point)) {
                add_v(DomainViolationKind::DeoptOrGuard, loc,
                      "speculative op '" + std::string(opcode_name(op)) + "' forbidden");
            }

            if (opts.require_no_eh &&
                (op == Opcode::throw_ || op == Opcode::invoke || op == Opcode::landing_pad || op == Opcode::resume)) {
                add_v(DomainViolationKind::ExceptionHandling, loc,
                      "exception op '" + std::string(opcode_name(op)) + "' forbidden");
            }

            if (op == Opcode::call || op == Opcode::patchable_call || op == Opcode::invoke) {
                const std::string_view sym = inst->symbol();
                if (opts.require_no_alloc && mod.is_allocation_function(sym)) {
                    add_v(DomainViolationKind::AllocationCall, loc,
                          "allocation call '@" + std::string(sym) + "' forbidden");
                } else if (opts.require_no_gc) {
                    const bool mod_fn = (mod.get_function(sym) != nullptr);
                    const bool safe_ext = mod.has_symbol_role(sym, SymbolRole::Pure) ||
                                          mod.has_symbol_role(sym, SymbolRole::NoCollect) ||
                                          sym.rfind("gpu_", 0) == 0 ||
                                          sym.rfind("ptx_", 0) == 0;
                    if (!mod_fn && !safe_ext && !mod.is_allocation_function(sym)) {
                        add_v(DomainViolationKind::ImpureExternalCall, loc,
                              "external call '@" + std::string(sym) + "' lacks Pure/NoCollect contract");
                    }
                }
            } else if (op == Opcode::call_indirect && opts.require_no_gc) {
                add_v(DomainViolationKind::ImpureExternalCall, loc, "indirect call may trigger GC");
            }
        }
    }

    if (opts.require_gpu_portable) {
        DiagnosticReporter diag;
        const bool port_ok = verify_gpu_portability(fn, opts.gpu_opts, &diag);
        for (const Diagnostic& d : diag.diagnostics()) {
            if (d.severity == DiagnosticSeverity::Error || d.severity == DiagnosticSeverity::Fatal || !port_ok) {
                add_v(DomainViolationKind::GpuPortability, DebugLoc(), d.message);
            }
        }
        if (!port_ok && report.violations.empty()) {
            add_v(DomainViolationKind::GpuPortability, DebugLoc(), "GPU portability check failed");
        }
    }

    report.diagnostic = build_audit_diagnostic(report.violations);
    return report;
}

DomainSpecializationResult specialize_callee_for_domain(
    Module& dst,
    const Function& callee,
    const DomainSpecializationSpec& spec) {
    DomainSpecializationResult res;
    if (callee.block_count() == 0) {
        res.diagnostic = "specialize_callee_for_domain: callee '" + std::string(callee.name()) + "' has no body";
        return res;
    }
    if (spec.param_types.size() != callee.param_count()) {
        res.diagnostic = "specialize_callee_for_domain: parameter count mismatch";
        return res;
    }
    if (!spec.param_convs.empty() && spec.param_convs.size() != spec.param_types.size()) {
        res.diagnostic = "specialize_callee_for_domain: param_convs length mismatch";
        return res;
    }

    const std::string wrapper_name = spec.specialized_name.empty()
        ? unique_dest_name(dst, std::string(callee.name()) + "__spec")
        : unique_dest_name(dst, spec.specialized_name);

    Function* wrapper = dst.create_function(wrapper_name, spec.return_type, spec.param_types);
    Builder b(dst);
    b.set_function(wrapper);
    BasicBlock* entry = b.append_block("entry");
    b.position_at_end(entry);

    std::vector<Value*> raw_params;
    raw_params.reserve(spec.param_types.size());
    for (Type pt : spec.param_types) {
        raw_params.push_back(b.add_block_param(entry, pt));
    }

    std::vector<Value*> adapted_args;
    adapted_args.reserve(raw_params.size());
    for (size_t i = 0; i < raw_params.size(); ++i) {
        const BoundaryConv conv = spec.param_convs.empty() ? BoundaryConv::Identity : spec.param_convs[i];
        Value* converted = emit_param_conversion(b, raw_params[i], conv, spec.nanbox);
        if (!converted || converted->type() != callee.param_type(i)) {
            dst.remove_function(wrapper);
            res.diagnostic = "specialize_callee_for_domain: invalid parameter conversion at arg " + std::to_string(i);
            return res;
        }
        adapted_args.push_back(converted);
    }

    Value* call_res = b.build_call(callee.name(), callee.return_type(), adapted_args);
    Instruction* call_inst = call_res ? call_res->defining_instruction() : entry->tail();
    if (!call_inst || call_inst->opcode() != Opcode::call) {
        dst.remove_function(wrapper);
        res.diagnostic = "specialize_callee_for_domain: failed to create wrapper call";
        return res;
    }

    if (spec.return_type.is_void()) {
        b.build_ret_void();
    } else {
        Value* ret_val = emit_return_conversion(b, call_res, spec.return_conv);
        if (!ret_val || ret_val->type() != spec.return_type) {
            dst.remove_function(wrapper);
            res.diagnostic = "specialize_callee_for_domain: invalid return conversion";
            return res;
        }
        b.build_ret(ret_val);
    }

    wrapper->rebuild_cfg_predecessors();
    InlineResult ir = inline_call_site(*wrapper, call_inst, callee, &dst.debug_context());
    if (!ir.success) {
        dst.remove_function(wrapper);
        res.diagnostic = "specialize_callee_for_domain: inline_call_site failed";
        return res;
    }

    cfg_simplify_function(*wrapper);
    scalar_cleanup(*wrapper);
    gvn_function(*wrapper);
    reuse_untagged_addresses(*wrapper);
    scalar_cleanup(*wrapper);
    cfg_simplify_function(*wrapper);
    wrapper->rebuild_cfg_predecessors();

    DiagnosticReporter vdiag;
    if (!verify_function(*wrapper, &vdiag)) {
        dst.remove_function(wrapper);
        res.diagnostic = "specialize_callee_for_domain: specialized IR failed verification";
        return res;
    }

    res.audit = audit_function_domain(*wrapper, dst, spec.audit_opts);
    if (!res.audit.ok) {
        res.diagnostic = res.audit.diagnostic;
        dst.remove_function(wrapper);
        return res;
    }

    res.ok = true;
    res.specialized = wrapper;
    return res;
}

CrossDomainFusionResult fuse_cross_module_call(
    Function& caller,
    Instruction* call_inst,
    Module& dst,
    const Module& src,
    std::string_view callee_name,
    const CrossDomainFusionOptions& opts) {
    CrossDomainFusionResult res;
    if (!call_inst || call_inst->parent() == nullptr || call_inst->parent()->parent() != &caller) {
        res.diagnostic = "fuse_cross_module_call: invalid call_inst";
        return res;
    }

    CrossModuleLinkResult link = import_function_into_module(dst, src, callee_name, opts.link_opts);
    if (!link.ok || !link.imported) {
        res.diagnostic = link.diagnostic;
        return res;
    }

    Function* imported = link.imported;
    if (call_inst->operand_count() != imported->param_count()) {
        res.diagnostic = "fuse_cross_module_call: argument count mismatch (" +
                         std::to_string(call_inst->operand_count()) + " vs " +
                         std::to_string(imported->param_count()) + ")";
        return res;
    }

    DomainSpecializationSpec spec;
    spec.specialized_name = std::string(imported->name()) + "__fused";
    spec.return_type = call_inst->type();
    spec.nanbox = opts.nanbox;
    spec.audit_opts = opts.audit_opts;
    spec.param_types.reserve(call_inst->operand_count());
    spec.param_convs.reserve(call_inst->operand_count());

    for (size_t i = 0; i < call_inst->operand_count(); ++i) {
        const Value* arg = call_inst->operand(i);
        const Type caller_t = arg ? arg->type() : Type::void_type();
        spec.param_types.push_back(caller_t);
        BoundaryConv conv = BoundaryConv::Identity;
        if (!infer_param_conv(caller_t, imported->param_type(i), opts.allow_nanbox_unboxing, conv)) {
            res.diagnostic = "fuse_cross_module_call: incompatible argument type at index " + std::to_string(i);
            return res;
        }
        spec.param_convs.push_back(conv);
    }

    if (!infer_return_conv(call_inst->type(), imported->return_type(), opts.allow_nanbox_unboxing, spec.return_conv)) {
        res.diagnostic = "fuse_cross_module_call: incompatible return type";
        return res;
    }

    DomainSpecializationResult sres = specialize_callee_for_domain(dst, *imported, spec);
    res.audit = sres.audit;
    if (!sres.ok || !sres.specialized) {
        res.diagnostic = sres.diagnostic;
        return res;
    }

    if (count_instructions(*sres.specialized) > opts.max_inlined_insts) {
        res.diagnostic = "fuse_cross_module_call: specialized callee exceeds max_inlined_insts budget";
        return res;
    }

    InlineResult ir = inline_call_site(caller, call_inst, *sres.specialized, &dst.debug_context());
    if (!ir.success) {
        res.diagnostic = "fuse_cross_module_call: inline_call_site failed";
        return res;
    }

    cfg_simplify_function(caller);
    scalar_cleanup(caller);
    caller.rebuild_cfg_predecessors();

    if (!verify_function(caller)) {
        res.diagnostic = "fuse_cross_module_call: caller failed MIR verification after fusion";
        return res;
    }

    res.fused = true;
    res.specialized_callee = sres.specialized;
    return res;
}

} // namespace brass
