// Deopt stress guards for tier 2 (include/brass/mir/deopt_stress.hpp).
#include <brass/mir/deopt_stress.hpp>
#include <brass/mir/builder.hpp>

#include <string>
#include <vector>

namespace brass {

namespace {

bool selected(const Instruction& guard, const DeoptStressPlan& plan) {
    return plan.resume_id < 0 || static_cast<uint64_t>(plan.resume_id) == guard.resume_id();
}

} // namespace

bool deopt_stress_guard_eligible(const Function& fn, const Instruction& guard, const Module* stub_module,
                                 bool resume_targets) {
    if (guard.opcode() != Opcode::guard) return false;
    const Function* stub = nullptr;
    if (!guard.symbol().empty()) {
        stub = stub_module ? stub_module->get_function(guard.symbol()) : nullptr;
        if (!stub) stub = fn.guard_exit_stub(guard);
    }
    if (stub) {
        std::string why;
        return fn.guard_exit_stub_matches(guard, *stub, why);
    }
    return resume_targets && fn.get_resume_target(guard.resume_id()) != nullptr;
}

bool has_deopt_stress_guards(const Function& fn, const Module* stub_module, const DeoptStressPlan& plan) {
    if (plan.period == 0 || !plan.evaluations || !plan.forced) return false;
    for (const auto* bb : fn.blocks()) {
        if (!bb) continue;
        for (const auto* inst : *bb) {
            if (inst && inst->opcode() == Opcode::guard && selected(*inst, plan) &&
                deopt_stress_guard_eligible(fn, *inst, stub_module, plan.resume_targets)) {
                return true;
            }
        }
    }
    return false;
}

size_t insert_deopt_stress_guards(Function& fn, const Module* stub_module, const DeoptStressPlan& plan) {
    if (!has_deopt_stress_guards(fn, stub_module, plan)) return 0;
    std::vector<Instruction*> guards;
    for (auto* bb : fn.blocks()) {
        if (!bb) continue;
        for (auto* inst : *bb) {
            if (inst && inst->opcode() == Opcode::guard && selected(*inst, plan) &&
                deopt_stress_guard_eligible(fn, *inst, stub_module, plan.resume_targets)) {
                guards.push_back(inst);
            }
        }
    }

    Builder b(fn);
    for (Instruction* guard : guards) {
        b.position_before(guard);
        // n = ++evaluations; forced += (n % period == 0);
        // guard (n % period != 0), <the guard's exits and state>
        Value* eval_addr = b.build_iconst_i64(static_cast<int64_t>(reinterpret_cast<uintptr_t>(plan.evaluations)));
        Value* n = b.build_add(b.build_load(Type::i64(), eval_addr, 0), b.build_iconst_i64(1));
        b.build_store(Type::i64(), eval_addr, 0, n);
        Value* rem = b.build_umod(n, b.build_iconst_i64(static_cast<int64_t>(plan.period)));
        Value* fire = b.build_eq(rem, b.build_iconst_i64(0));
        Value* forced_addr = b.build_iconst_i64(static_cast<int64_t>(reinterpret_cast<uintptr_t>(plan.forced)));
        Value* forced = b.build_add(b.build_load(Type::i64(), forced_addr, 0), b.build_zext_i64(fire));
        b.build_store(Type::i64(), forced_addr, 0, forced);
        Value* keep = b.build_ne(rem, b.build_iconst_i64(0));

        const auto& state = guard->state_map();
        Instruction* stress = b.build_guard(keep, guard->symbol(), Span<Value* const>(state.data(), state.size()));
        // The same resume id: the lower tier finishes the call from the
        // guard it names, the real one, exactly as when that one fails.
        stress->set_resume_id(guard->resume_id());
        stress->set_offset(guard->offset());
    }
    return guards.size();
}

} // namespace brass
