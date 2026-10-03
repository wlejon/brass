// spirv::verify: the structural rules the SPIR-V backend relies on, checked
// on the typed module without spirv-val. It is not a replacement for
// spirv-val (the tests run both); it catches the mistakes an ISel bug would
// make -- undefined or duplicate ids, non-type result types, malformed
// blocks, phis that disagree with the CFG, a block named as the merge of two
// headers, a PhysicalStorageBuffer access without an alignment, a 64-bit type
// without its capability, an entry point whose interface misses a global.

#include <brass/target/spirv/spirv_ir.hpp>

#include <algorithm>
#include <set>
#include <sstream>
#include <unordered_map>
#include <unordered_set>

namespace brass::spirv {

namespace {

class Verifier {
public:
    explicit Verifier(const Module& m) : m_(m) {}

    std::vector<Diagnostic> run() {
        collect_definitions();
        check_globals();
        for (const Function& f : m_.functions) check_function(f);
        check_entry_points();
        check_capabilities();
        return std::move(diags_);
    }

private:
    const Module& m_;
    std::vector<Diagnostic> diags_;
    std::unordered_set<Id> defined_;
    std::unordered_map<Id, Id> value_type_;       // result id -> result type id
    std::unordered_set<Id> global_vars_;

    void report(const std::string& fn, Id block, size_t index, std::string msg, const Inst* inst = nullptr) {
        diags_.push_back(Diagnostic{fn, block, index, std::move(msg), inst ? to_string(*inst) : std::string()});
    }

    void define(Id id, const std::string& fn, Id block, size_t index, const Inst* inst) {
        if (id == 0 || id >= m_.bound()) {
            report(fn, block, index, "result id %" + std::to_string(id) + " is outside the id bound " +
                                         std::to_string(m_.bound()), inst);
            return;
        }
        if (!defined_.insert(id).second) report(fn, block, index, "id %" + std::to_string(id) + " is defined twice", inst);
    }

    void collect_definitions() {
        for (const auto& imp : m_.ext_imports) define(imp.first, "", 0, 0, nullptr);
        for (size_t i = 0; i < m_.globals.size(); ++i) {
            const Inst& g = m_.globals[i];
            define(g.result, "", 0, i, &g);
            if (g.type) value_type_[g.result] = g.type;
            if (g.op == spv::OpVariable) global_vars_.insert(g.result);
        }
        for (const Function& f : m_.functions) {
            define(f.id, f.name, 0, 0, nullptr);
            for (const Block& b : f.blocks) {
                define(b.label, f.name, b.label, 0, nullptr);
                for (size_t i = 0; i < b.insts.size(); ++i) {
                    const Inst& inst = b.insts[i];
                    if (inst.result) define(inst.result, f.name, b.label, i, &inst);
                    if (inst.result && inst.type) value_type_[inst.result] = inst.type;
                }
            }
        }
    }

    void check_operands(const Inst& inst, const std::string& fn, Id block, size_t index) {
        if (inst.type && !m_.type_info(inst.type)) {
            report(fn, block, index, "result type %" + std::to_string(inst.type) + " is not a type", &inst);
        }
        for (const Operand& o : inst.operands) {
            if (o.is_id() && !defined_.count(o.value)) {
                report(fn, block, index, "operand %" + std::to_string(o.value) + " is not defined", &inst);
            }
        }
    }

    void check_globals() {
        for (size_t i = 0; i < m_.globals.size(); ++i) check_operands(m_.globals[i], "", 0, i);
        for (const Inst& a : m_.annotations) check_operands(a, "", 0, 0);
        for (const Inst& e : m_.execution_modes) check_operands(e, "", 0, 0);
    }

    const TypeInfo* pointer_type_of(Id value) const {
        auto it = value_type_.find(value);
        if (it == value_type_.end()) return nullptr;
        const TypeInfo* ti = m_.type_info(it->second);
        return (ti && ti->op == spv::OpTypePointer) ? ti : nullptr;
    }

    // OpLoad: pointer at operand 0, memory operands from 1; OpStore: pointer
    // at 0, object at 1, memory operands from 2.
    void check_memory_access(const Inst& inst, const std::string& fn, Id block, size_t index) {
        size_t mask_at = inst.op == spv::OpLoad ? 1 : 2;
        if (inst.operands.empty()) return;
        const TypeInfo* pt = pointer_type_of(inst.operands[0].value);
        if (!pt || pt->storage != spv::StorageClassPhysicalStorageBuffer) return;
        bool aligned = inst.operands.size() > mask_at + 1 &&
                       (inst.operands[mask_at].value & spv::MemoryAccessAlignedMask) != 0 &&
                       inst.operands[mask_at + 1].value != 0;
        if (!aligned) report(fn, block, index, "PhysicalStorageBuffer access without an Aligned memory operand", &inst);
    }

    static void successors(const Inst& term, std::vector<Id>& out) {
        if (term.op == spv::OpBranch && !term.operands.empty()) out.push_back(term.operands[0].value);
        if (term.op == spv::OpBranchConditional && term.operands.size() >= 3) {
            out.push_back(term.operands[1].value);
            out.push_back(term.operands[2].value);
        }
        if (term.op == spv::OpSwitch && term.operands.size() >= 2) {
            out.push_back(term.operands[1].value);
            for (size_t i = 3; i < term.operands.size(); i += 2) out.push_back(term.operands[i].value);
        }
    }

    void check_function(const Function& f) {
        if (f.blocks.empty()) {
            report(f.name, 0, 0, "function has no blocks");
            return;
        }
        std::unordered_set<Id> labels;
        for (const Block& b : f.blocks) labels.insert(b.label);
        std::unordered_map<Id, std::vector<Id>> preds;
        std::unordered_map<Id, Id> merge_of; // merge or continue block -> declaring header

        for (const Block& b : f.blocks) {
            if (b.insts.empty()) {
                report(f.name, b.label, 0, "block is empty");
                continue;
            }
            bool phis_done = false;
            for (size_t i = 0; i < b.insts.size(); ++i) {
                const Inst& inst = b.insts[i];
                check_operands(inst, f.name, b.label, i);
                bool last = i + 1 == b.insts.size();
                if (is_terminator(inst.op) != last) {
                    report(f.name, b.label, i, last ? "block does not end in a terminator"
                                                    : "terminator in the middle of a block", &inst);
                }
                if (inst.op == spv::OpPhi) {
                    if (phis_done) report(f.name, b.label, i, "OpPhi after a non-phi instruction", &inst);
                } else {
                    phis_done = true;
                }
                if (inst.op == spv::OpLoad || inst.op == spv::OpStore) check_memory_access(inst, f.name, b.label, i);
                if (inst.op == spv::OpSelectionMerge || inst.op == spv::OpLoopMerge) {
                    check_merge(f, b, i, labels, merge_of);
                }
                if (is_terminator(inst.op)) {
                    std::vector<Id> succ;
                    successors(inst, succ);
                    for (Id s : succ) {
                        if (!labels.count(s)) {
                            report(f.name, b.label, i, "branch target %" + std::to_string(s) + " is not a block of the function", &inst);
                        } else if (std::find(preds[s].begin(), preds[s].end(), b.label) == preds[s].end()) {
                            preds[s].push_back(b.label);
                        }
                    }
                }
            }
        }
        if (!preds[f.blocks.front().label].empty()) {
            report(f.name, f.blocks.front().label, 0, "the entry block is the target of a branch");
        }
        for (const Block& b : f.blocks) check_phis(f, b, preds[b.label]);
    }

    void check_merge(const Function& f, const Block& b, size_t i, const std::unordered_set<Id>& labels,
                     std::unordered_map<Id, Id>& merge_of) {
        const Inst& inst = b.insts[i];
        bool loop = inst.op == spv::OpLoopMerge;
        if (i + 2 != b.insts.size()) {
            report(f.name, b.label, i, "merge instruction is not immediately before the terminator", &inst);
            return;
        }
        spv::Op next = b.insts[i + 1].op;
        bool ok_next = loop ? (next == spv::OpBranch || next == spv::OpBranchConditional)
                            : (next == spv::OpBranchConditional || next == spv::OpSwitch);
        if (!ok_next) report(f.name, b.label, i, std::string("merge instruction followed by ") + op_name(next), &inst);
        size_t targets = loop ? 2 : 1;
        for (size_t k = 0; k < targets && k < inst.operands.size(); ++k) {
            Id t = inst.operands[k].value;
            if (!labels.count(t)) {
                report(f.name, b.label, i, "merge/continue target %" + std::to_string(t) + " is not a block of the function", &inst);
                continue;
            }
            if (k == 1) continue; // a continue target may also be reached otherwise; only merges must be unique
            auto [it, fresh] = merge_of.emplace(t, b.label);
            if (!fresh) {
                report(f.name, b.label, i, "block %" + std::to_string(t) + " is already the merge block of header %" +
                                               std::to_string(it->second), &inst);
            }
        }
    }

    void check_phis(const Function& f, const Block& b, const std::vector<Id>& preds) {
        for (size_t i = 0; i < b.insts.size() && b.insts[i].op == spv::OpPhi; ++i) {
            const Inst& phi = b.insts[i];
            if (phi.operands.size() % 2 != 0) {
                report(f.name, b.label, i, "OpPhi operands are not (value, parent) pairs", &phi);
                continue;
            }
            std::set<Id> parents;
            for (size_t k = 1; k < phi.operands.size(); k += 2) parents.insert(phi.operands[k].value);
            std::set<Id> expected(preds.begin(), preds.end());
            if (parents != expected || parents.size() != phi.operands.size() / 2) {
                report(f.name, b.label, i, "OpPhi parents do not match the block's " + std::to_string(preds.size()) +
                                               " predecessor(s)", &phi);
            }
        }
    }

    void check_entry_points() {
        for (const EntryPoint& ep : m_.entry_points) {
            auto fit = std::find_if(m_.functions.begin(), m_.functions.end(),
                                    [&](const Function& f) { return f.id == ep.function; });
            if (fit == m_.functions.end()) {
                report("", 0, 0, "entry point '" + ep.name + "' names a function that does not exist");
                continue;
            }
            std::unordered_set<Id> listed(ep.interface.begin(), ep.interface.end());
            std::set<Id> missing;
            for (const Block& b : fit->blocks) {
                for (const Inst& inst : b.insts) {
                    for (const Operand& o : inst.operands) {
                        if (o.is_id() && global_vars_.count(o.value) && !listed.count(o.value)) missing.insert(o.value);
                    }
                }
            }
            for (Id v : missing) {
                report(fit->name, 0, 0, "global variable %" + std::to_string(v) + " is used but not in the entry point interface");
            }
        }
    }

    void check_capabilities() {
        for (const Inst& g : m_.globals) {
            if (g.op == spv::OpTypeInt && !g.operands.empty() && g.operands[0].value == 64 &&
                !m_.has_capability(spv::CapabilityInt64)) {
                report("", 0, 0, "64-bit integer type without the Int64 capability", &g);
            }
            if (g.op == spv::OpTypeFloat && !g.operands.empty() && g.operands[0].value == 64 &&
                !m_.has_capability(spv::CapabilityFloat64)) {
                report("", 0, 0, "64-bit float type without the Float64 capability", &g);
            }
        }
    }
};

} // namespace

std::vector<Diagnostic> verify(const Module& m) {
    return Verifier(m).run();
}

std::string format_diagnostics(const std::vector<Diagnostic>& diags) {
    std::ostringstream os;
    for (const Diagnostic& d : diags) {
        os << "spirv verify";
        if (!d.function.empty()) os << " [" << d.function << "]";
        if (d.block) os << " block %" << d.block << " #" << d.index;
        os << ": " << d.message;
        if (!d.inst_text.empty()) os << "\n    " << d.inst_text;
        os << "\n";
    }
    return os.str();
}

} // namespace brass::spirv
