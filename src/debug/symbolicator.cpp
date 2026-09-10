#include <brass/debug/symbolicator.hpp>
#include <sstream>
#include <iomanip>

namespace brass {

std::string StackFrame::to_string() const {
    std::string s = "    at ";
    s += function_name.empty() ? "<anonymous>" : function_name;
    s += " (";
    if (!file.empty()) {
        s += file;
    } else {
        s += "<unknown>";
    }
    s += ":" + std::to_string(line);
    s += ":" + std::to_string(column);
    s += ")";
    return s;
}

std::string StackTrace::format() const {
    std::string out;
    for (const auto& frame : frames) {
        out += frame.to_string();
        out += "\n";
    }
    return out;
}

Symbolicator::Symbolicator(const DebugContext& ctx, const std::vector<FunctionDebugTable>& tables)
    : ctx_(ctx) {
    set_function_tables(tables);
}

void Symbolicator::set_debug_context(const DebugContext& ctx) {
    ctx_ = ctx;
}

void Symbolicator::add_function_table(const FunctionDebugTable& table) {
    size_t idx = tables_.size();
    tables_.push_back(table);
    fn_to_table_idx_[table.function_name()] = idx;
    if (table.code_base() != 0 && table.code_size() != 0) {
        register_function_address(table.function_name(), table.code_base(), table.code_size());
    }
}

void Symbolicator::set_function_tables(const std::vector<FunctionDebugTable>& tables) {
    tables_ = tables;
    fn_to_table_idx_.clear();
    for (size_t i = 0; i < tables_.size(); ++i) {
        fn_to_table_idx_[tables_[i].function_name()] = i;
        if (tables_[i].code_base() != 0 && tables_[i].code_size() != 0) {
            register_function_address(tables_[i].function_name(), tables_[i].code_base(), tables_[i].code_size());
        }
    }
}

void Symbolicator::register_function_address(const std::string& name, uintptr_t code_base, size_t code_size) {
    registered_ranges_.push_back({code_base, code_size, name});
}

void Symbolicator::expand_inlined_frames(
    const std::string& fn_name,
    DebugLoc loc,
    std::vector<StackFrame>& frames
) const {
    if (!loc.is_valid()) {
        frames.push_back({fn_name, "", 0, 0, false});
        return;
    }

    if (loc.inlined_at_id == 0) {
        frames.push_back({
            fn_name,
            ctx_.get_file(loc.file_id),
            loc.line,
            loc.column,
            false
        });
        return;
    }

    // Inlined frames expansion:
    // Innermost (leaf) frame is the inlined function at loc
    const InlinedScope* scope = ctx_.get_inlined_scope(loc.inlined_at_id);
    frames.push_back({
        scope ? scope->callee_name : "<inlined>",
        ctx_.get_file(loc.file_id),
        loc.line,
        loc.column,
        true
    });

    // Walk up the callsite chain to the root caller
    uint32_t cur_id = loc.inlined_at_id;
    while (cur_id != 0) {
        const InlinedScope* cur_scope = ctx_.get_inlined_scope(cur_id);
        if (!cur_scope) break;

        DebugLoc callsite = cur_scope->callsite_loc;
        uint32_t parent_id = cur_scope->parent_inlined_at_id;

        if (parent_id != 0) {
            const InlinedScope* parent_scope = ctx_.get_inlined_scope(parent_id);
            frames.push_back({
                parent_scope ? parent_scope->callee_name : "<inlined>",
                ctx_.get_file(callsite.file_id),
                callsite.line,
                callsite.column,
                true
            });
            cur_id = parent_id;
        } else {
            frames.push_back({
                fn_name,
                ctx_.get_file(callsite.file_id),
                callsite.line,
                callsite.column,
                false
            });
            break;
        }
    }
}

StackTrace Symbolicator::symbolize_offset(const std::string& fn_name, uint32_t offset) const {
    StackTrace trace;
    auto it = fn_to_table_idx_.find(fn_name);
    if (it == fn_to_table_idx_.end()) {
        trace.frames.push_back({fn_name, "", 0, 0, false});
        return trace;
    }

    const auto& table = tables_[it->second];
    DebugLoc loc = table.resolve_offset(offset);
    expand_inlined_frames(fn_name, loc, trace.frames);
    return trace;
}

StackTrace Symbolicator::symbolize_ip(uintptr_t code_base, uintptr_t ip) const {
    uint32_t offset = (ip >= code_base) ? static_cast<uint32_t>(ip - code_base) : 0;
    for (const auto& table : tables_) {
        if (table.code_base() == code_base) {
            return symbolize_offset(table.function_name(), offset);
        }
    }
    for (const auto& r : registered_ranges_) {
        if (r.base == code_base) {
            return symbolize_offset(r.name, offset);
        }
    }
    // Default fallback
    std::ostringstream ss;
    ss << "0x" << std::hex << ip;
    StackTrace trace;
    trace.frames.push_back({ss.str(), "", 0, 0, false});
    return trace;
}

StackTrace Symbolicator::symbolize_ip(uintptr_t ip) const {
    for (const auto& r : registered_ranges_) {
        if (ip >= r.base && ip < r.base + r.size) {
            uint32_t offset = static_cast<uint32_t>(ip - r.base);
            return symbolize_offset(r.name, offset);
        }
    }
    for (const auto& t : tables_) {
        if (t.code_base() != 0 && ip >= t.code_base() && ip < t.code_base() + t.code_size()) {
            uint32_t offset = static_cast<uint32_t>(ip - t.code_base());
            return symbolize_offset(t.function_name(), offset);
        }
    }
    std::ostringstream ss;
    ss << "0x" << std::hex << ip;
    StackTrace trace;
    trace.frames.push_back({ss.str(), "", 0, 0, false});
    return trace;
}

StackTrace Symbolicator::symbolize_stack(
    uintptr_t top_rbp,
    uintptr_t top_return_ip,
    const ModuleStackMap& stack_maps
) const {
    StackTrace trace;
    constexpr size_t MAX_FRAMES = 256;
    size_t count = 0;
    uintptr_t cur_rbp = top_rbp;
    uintptr_t cur_return_ip = top_return_ip;

    while (cur_rbp != 0 && cur_return_ip != 0 && count < MAX_FRAMES) {
        const FunctionStackMap* fn_map = stack_maps.find_function_by_ip(cur_return_ip);
        if (fn_map) {
            uintptr_t fn_base = fn_map->function_address;
            uint32_t offset = 0;
            const StackMapRecord* rec = fn_map->find_record_by_ip(cur_return_ip);
            if (rec && rec->instruction_offset > 0) {
                offset = rec->instruction_offset - 1;
            } else if (fn_base != 0 && cur_return_ip > fn_base) {
                offset = static_cast<uint32_t>(cur_return_ip - fn_base - 1);
            }
            StackTrace fn_trace = symbolize_offset(fn_map->function_name, offset);
            for (auto& f : fn_trace.frames) {
                trace.frames.push_back(std::move(f));
            }
        } else {
            StackTrace ip_trace = symbolize_ip(cur_return_ip);
            if (!ip_trace.empty()) {
                for (auto& f : ip_trace.frames) {
                    trace.frames.push_back(std::move(f));
                }
            } else {
                std::ostringstream ss;
                ss << "0x" << std::hex << cur_return_ip;
                trace.frames.push_back({ss.str(), "", 0, 0, false});
            }
        }

        uintptr_t next_rbp = *reinterpret_cast<const uintptr_t*>(cur_rbp);
        uintptr_t next_return_ip = *reinterpret_cast<const uintptr_t*>(cur_rbp + 8);
        if (next_rbp <= cur_rbp || (next_rbp % 8) != 0) break;
        cur_rbp = next_rbp;
        cur_return_ip = next_return_ip;
        count++;
    }

    return trace;
}

} // namespace brass
