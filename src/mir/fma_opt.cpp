#include <brass/mir/fma_opt.hpp>
#include <brass/mir/block.hpp>
#include <brass/mir/instruction.hpp>

namespace brass {

std::string FmaOptStats::format_report() const {
    std::string out;
    out += "=== FMA Optimization Stats ===\n";
    out += "Scalar fma_f32 fused: " + std::to_string(scalar_fma_f32_count) + "\n";
    out += "Scalar fma_f64 fused: " + std::to_string(scalar_fma_f64_count) + "\n";
    out += "Vector vfma fused:    " + std::to_string(vector_vfma_count) + "\n";
    out += "Total instructions fused: " + std::to_string(total_fused()) + "\n";
    return out;
}

bool fma_opt_pass(Function& fn, const FmaOptOptions& options) {
    bool changed = false;

    for (BasicBlock* bb : fn.blocks()) {
        if (!bb) continue;
        Instruction* inst = bb->head();
        while (inst) {
            Instruction* next = inst->next();

            // Scalar FMA pattern matching: add(mul(a, b), c) or add(c, mul(a, b))
            if (options.enable_scalar && inst->opcode() == Opcode::add) {
                Type ty = inst->type();
                if (ty == Type::f32() || ty == Type::f64()) {
                    Value* op0 = inst->operand(0);
                    Value* op1 = inst->operand(1);
                    Instruction* mul0 = (op0 && op0->is_instruction()) ? op0->defining_instruction() : nullptr;
                    Instruction* mul1 = (op1 && op1->is_instruction()) ? op1->defining_instruction() : nullptr;

                    Value* a = nullptr;
                    Value* b = nullptr;
                    Value* c = nullptr;

                    if (mul0 && mul0->opcode() == Opcode::mul && mul0->type() == ty && mul0->operand_count() >= 2) {
                        a = mul0->operand(0);
                        b = mul0->operand(1);
                        c = op1;
                    } else if (mul1 && mul1->opcode() == Opcode::mul && mul1->type() == ty && mul1->operand_count() >= 2) {
                        a = mul1->operand(0);
                        b = mul1->operand(1);
                        c = op0;
                    }

                    if (a && b && c) {
                        Opcode new_op = (ty == Type::f32()) ? Opcode::fma_f32 : Opcode::fma_f64;
                        inst->set_opcode(new_op);
                        inst->operands().clear();
                        inst->add_operand(a);
                        inst->add_operand(b);
                        inst->add_operand(c);

                        if (options.stats) {
                            if (ty == Type::f32()) {
                                options.stats->scalar_fma_f32_count++;
                            } else {
                                options.stats->scalar_fma_f64_count++;
                            }
                        }
                        changed = true;
                    }
                }
            }
            // Vector FMA pattern matching: vadd(vmul(a, b), c) or vadd(c, vmul(a, b))
            else if (options.enable_vector && inst->opcode() == Opcode::vadd) {
                Type ty = inst->type();
                if (ty.is_vector() && ty.element_type().is_float()) {
                    Value* op0 = inst->operand(0);
                    Value* op1 = inst->operand(1);
                    Instruction* mul0 = (op0 && op0->is_instruction()) ? op0->defining_instruction() : nullptr;
                    Instruction* mul1 = (op1 && op1->is_instruction()) ? op1->defining_instruction() : nullptr;

                    Value* a = nullptr;
                    Value* b = nullptr;
                    Value* c = nullptr;

                    if (mul0 && mul0->opcode() == Opcode::vmul && mul0->type() == ty && mul0->operand_count() >= 2) {
                        a = mul0->operand(0);
                        b = mul0->operand(1);
                        c = op1;
                    } else if (mul1 && mul1->opcode() == Opcode::vmul && mul1->type() == ty && mul1->operand_count() >= 2) {
                        a = mul1->operand(0);
                        b = mul1->operand(1);
                        c = op0;
                    }

                    if (a && b && c) {
                        inst->set_opcode(Opcode::vfma);
                        inst->operands().clear();
                        inst->add_operand(a);
                        inst->add_operand(b);
                        inst->add_operand(c);

                        if (options.stats) {
                            options.stats->vector_vfma_count++;
                        }
                        changed = true;
                    }
                }
            }

            inst = next;
        }
    }

    return changed;
}

bool fma_opt_module_pass(Module& mod, const FmaOptOptions& options) {
    bool changed = false;
    for (Function* fn : mod.functions()) {
        if (fn) changed |= fma_opt_pass(*fn, options);
    }
    return changed;
}

bool run_fma_opt(Function& fn, FmaOptStats* stats) {
    FmaOptOptions opts;
    opts.stats = stats;
    return fma_opt_pass(fn, opts);
}

} // namespace brass
