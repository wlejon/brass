#include "opt_reports.hpp"
#include <brass/brass.hpp>
#include <ostream>
#include <string>
#include <vector>

namespace brass::opt {

void report_branch_probabilities(const Module& mod, const pgo::ProfileData& profile, std::ostream& os) {
    for (const Function* fn : mod.functions()) {
        if (!fn) continue;
        const auto* fp = profile.find_function(std::string(fn->name()));
        if (!fp) {
            os << "Function '" << fn->name() << "': No profile available\n";
            continue;
        }
        mir::BranchProbabilityAnalysis bpa(*fn, *fp);
        const auto& bfi = bpa.block_frequency_info();
        const auto& bpi = bpa.branch_probability_info();
        os << "Function '" << fn->name() << "' (entry count=" << bfi.entry_count() << "):\n";
        for (const BasicBlock* bb : fn->blocks()) {
            if (!bb) continue;
            os << "  block " << bb->name() << ": count=" << bfi.get_block_count(bb)
               << ", freq=" << bfi.get_block_frequency(bb)
               << (bfi.is_hot_block(bb) ? " [HOT]" : "")
               << (bfi.is_cold_block(bb) ? " [COLD]" : "") << "\n";
            for (const BasicBlock* succ : bb->successors()) {
                if (!succ) continue;
                os << "    edge -> " << succ->name()
                   << ": count=" << bpi.get_edge_count(bb, succ)
                   << ", prob=" << bpi.get_edge_probability(bb, succ) << "\n";
            }
        }
    }
}

void report_escape_analysis(const Module& mod, std::ostream& os) {
    for (const Function* fn : mod.functions()) {
        if (!fn) continue;
        EscapeAnalysis ea(*fn);
        os << "Function '" << fn->name() << "': " << ea.allocations().size() << " allocations ("
           << ea.non_escaping_allocations().size() << " non-escaping)\n";
        for (const Value* alloc_val : ea.allocations()) {
            os << "  alloc %" << alloc_val->id() << ": " << escape_state_name(ea.get_escape_state(alloc_val)) << "\n";
        }
    }
}

void report_partial_escape(const Module& mod, std::ostream& os) {
    for (const Function* fn : mod.functions()) {
        if (!fn) continue;
        PartialEscapeAnalysis pea(*fn);
        os << "Function '" << fn->name() << "': " << pea.candidate_allocations().size() << " candidate allocations\n";
        for (const Value* alloc_val : pea.candidate_allocations()) {
            auto frontier = pea.get_materialization_frontier(alloc_val);
            os << "  alloc %" << alloc_val->id() << ": frontier " << frontier.size() << " edges\n";
            for (const auto& e : frontier) {
                os << "    edge " << (e.from ? e.from->name() : "<null>") << " -> "
                   << (e.to ? e.to->name() : "<null>") << "\n";
            }
        }
    }
}

void report_alias_analysis(const Module& mod, std::ostream& os) {
    for (const Function* fn : mod.functions()) {
        if (!fn) continue;
        AliasAnalysis aa(*fn);
        os << "Alias Analysis for Function '" << fn->name() << "':\n";
        std::vector<const Instruction*> mem_insts;
        for (const BasicBlock* bb : fn->blocks()) {
            if (!bb) continue;
            for (const Instruction* inst : *bb) {
                if (inst && is_memory(inst->opcode())) mem_insts.push_back(inst);
            }
        }
        os << "  " << mem_insts.size() << " memory instructions\n";
        for (size_t i = 0; i < mem_insts.size(); ++i) {
            for (size_t j = i + 1; j < mem_insts.size(); ++j) {
                const Instruction* m1 = mem_insts[i];
                const Instruction* m2 = mem_insts[j];
                if (m1->operand_count() == 0 || m2->operand_count() == 0) continue;
                AliasResult res = aa.alias(m1->operand(0), m1->offset(), m1->memory_type(),
                                           m2->operand(0), m2->offset(), m2->memory_type());
                os << "  " << opcode_name(m1->opcode()) << " (off " << m1->offset() << ") vs "
                   << opcode_name(m2->opcode()) << " (off " << m2->offset() << "): " << alias_result_name(res) << "\n";
            }
        }
    }
}

int check_roundtrip(const Module& mod, std::ostream& os, std::ostream& err) {
    const std::string canonical1 = to_string(mod);
    DiagnosticReporter rt_diag;
    auto mod2 = parse_module(canonical1, &rt_diag, "<canonical-roundtrip>");
    if (!mod2 || rt_diag.has_errors()) {
        err << "Roundtrip parse failed:\n" << rt_diag.format_all() << "\n";
        return 1;
    }
    if (!verify_module(*mod2, &rt_diag) || rt_diag.has_errors()) {
        err << "Roundtrip verify failed:\n" << rt_diag.format_all() << "\n";
        return 1;
    }
    const std::string canonical2 = to_string(*mod2);
    if (canonical1 != canonical2) {
        err << "Roundtrip mismatch: canonical representation is not byte-identical!\n"
            << "--- First Canon ---\n" << canonical1 << "--- Second Canon ---\n" << canonical2 << "\n";
        return 1;
    }
    os << "Roundtrip verified: byte-identical canonical representation (" << canonical1.size() << " bytes)\n";
    return 0;
}

} // namespace brass::opt
