// SpirvISel core: driver, kernel parameters (push constants), builtins, value
// access and the opcode dispatch. Per-family lowering lives in
// spirv_isel_alu.cpp, spirv_isel_mem.cpp, spirv_isel_vec.cpp and
// spirv_isel_intrinsics*.cpp; control flow comes from spirv_structurize.cpp.

#include <brass/target/spirv/spirv_isel.hpp>
#include <brass/mir/block.hpp>
#include <brass/mir/opcodes.hpp>

#include "spirv_structurize.hpp"

#include <spirv/unified1/GLSL.std.450.h>

#include <stdexcept>
#include <string>

namespace brass::spirv {

namespace {

bool is_predicate_use(const brass::Instruction& inst, size_t operand_index) {
    return operand_index == 0 &&
           (inst.opcode() == brass::Opcode::br_if || inst.opcode() == brass::Opcode::select);
}

uint32_t align_up(uint32_t v, uint32_t a) { return (v + a - 1) / a * a; }

} // namespace

SpirvISel::SpirvISel(Module& m, const target::SpirvOptions& opts) : m_(m), opts_(opts) {
    m_.add_capability(spv::CapabilityPhysicalStorageBufferAddresses);
    m_.add_capability(spv::CapabilityInt64);
    m_.set_memory_model(spv::AddressingModelPhysicalStorageBuffer64, spv::MemoryModelGLSL450);
}

// ---------------------------------------------------------------------------
// Driver
// ---------------------------------------------------------------------------

KernelLayout SpirvISel::lower(const brass::Function& mir_fn) {
    const std::string name(mir_fn.name());
    if (!mir_fn.return_type().is_void()) {
        throw std::runtime_error("SpirvTarget: kernel '" + name +
                                 "' has a non-void return type; compute entry points cannot return values");
    }
    mir_ = &mir_fn;
    origin_ = nullptr;
    ids_.clear();
    bools_.clear();
    shared_.clear();
    value_uses_.clear();
    prologue_cache_.clear();
    prologue_args_.clear();
    shared_bytes_ = 0;

    StructuredCfg cfg = structurize(mir_fn);
    cfg_ = &cfg;
    analyze_uses();

    Id void_t = m_.t_void();
    m_.add_function(name, void_t, m_.t_function(void_t, {}));
    fn_index_ = m_.functions.size() - 1;
    m_.set_name(fn().id, name);

    node_block_.assign(cfg.nodes.size(), kNoNode);
    node_phis_.assign(cfg.nodes.size(), {});
    for (size_t n : cfg.order) {
        Block b;
        b.label = m_.new_id();
        m_.set_name(b.label, cfg.nodes[n].name);
        node_block_[n] = fn().blocks.size();
        fn().blocks.push_back(std::move(b));
    }

    // Workgroup size: spec constants (once per module) or a fixed LocalSize.
    if (opts_.local_size_spec_constants) workgroup_size(0);
    m_.add_execution_mode(fn().id, spv::ExecutionModeLocalSize,
                          {opts_.local_size_x, opts_.local_size_y, opts_.local_size_z});

    KernelLayout layout;
    cur_ = node_block_[0];
    lower_params(layout);
    for (size_t n : cfg.order) {
        if (cfg.nodes[n].kind != CfgNode::Kind::prologue) lower_node(n);
    }
    // The prologue's branch goes last: builtin reads were appended to it
    // while the other blocks were lowered.
    cur_ = node_block_[0];
    origin_ = nullptr;
    emit(Inst(spv::OpBranch).id(label_of(cfg.edges[cfg.nodes[0].out.at(0)].to)));
    fill_phis();

    m_.add_entry_point(spv::ExecutionModelGLCompute, fn().id, name, fn().interface);
    layout.shared_bytes = shared_bytes_;
    cfg_ = nullptr;
    mir_ = nullptr;
    return layout;
}

void SpirvISel::analyze_uses() {
    auto count = [&](const Value* v) { if (v) ++value_uses_[v]; };
    for (const BasicBlock* bb : mir_->blocks()) {
        for (const brass::Instruction* inst : *bb) {
            for (size_t i = 0; i < inst->operand_count(); ++i) {
                if (!is_predicate_use(*inst, i)) count(inst->operand(i));
            }
            for (const Value* v : inst->branch_target().args) count(v);
            for (const Value* v : inst->true_target().args) count(v);
            for (const Value* v : inst->false_target().args) count(v);
        }
    }
}

// Every kernel parameter is a member of one push-constant block, in order,
// at the next offset aligned to its size (std430 scalar rules).
void SpirvISel::lower_params(KernelLayout& layout) {
    const auto& types = mir_->param_types();
    const BasicBlock* entry = mir_->entry_block();
    if (entry->param_count() != types.size()) {
        fail("the entry block has " + std::to_string(entry->param_count()) + " parameters but the kernel declares " +
             std::to_string(types.size()));
    }
    if (types.empty()) return;

    std::vector<Id> members;
    uint32_t offset = 0;
    for (size_t i = 0; i < types.size(); ++i) {
        Type t = types[i];
        if (!(t.is_i32() || t.is_i64() || t.is_float() || t.is_pointer())) {
            fail("kernel parameter " + std::to_string(i) + " has type " + std::string(t.name()) +
                 "; SPIR-V kernel parameters are ptr, i32, i64, f32 or f64");
        }
        auto size = static_cast<uint32_t>(t.size_in_bytes());
        offset = align_up(offset, size);
        layout.params.push_back(target::SpirvParam{t, offset, size});
        members.push_back(scalar_type(t));
        offset += size;
    }
    layout.push_constant_bytes = align_up(offset, 4);

    Id block_t = m_.t_struct(members);
    m_.decorate(block_t, spv::DecorationBlock);
    for (uint32_t i = 0; i < members.size(); ++i) {
        m_.member_decorate(block_t, i, spv::DecorationOffset, {layout.params[i].offset});
    }
    Id var = m_.variable(m_.t_pointer(spv::StorageClassPushConstant, block_t), spv::StorageClassPushConstant);
    m_.set_name(var, std::string(mir_->name()) + "_params");
    use_global(var);

    bool entry_has_phis = cfg_->nodes[cfg_->edges[cfg_->nodes[0].out.at(0)].to].params != nullptr;
    for (uint32_t i = 0; i < members.size(); ++i) {
        Id member_ptr = op(spv::OpAccessChain, m_.t_pointer(spv::StorageClassPushConstant, members[i]), {var, m_.c_u32(i)});
        Id value = op(spv::OpLoad, members[i], {member_ptr});
        if (entry_has_phis) prologue_args_.push_back(value);
        else define(entry->param(i), value);
    }
}

void SpirvISel::lower_node(size_t n) {
    const CfgNode& node = cfg_->nodes[n];
    cur_ = node_block_[n];
    origin_ = nullptr;

    if (node.params) {
        for (const Value* p : node.params->params()) {
            std::vector<Id> parts;
            Id type = part_type(p->type());
            for (uint32_t k = 0; k < part_count(p->type()); ++k) {
                Id phi = m_.new_id();
                emit(Inst(spv::OpPhi, type, phi)); // incoming pairs are added by fill_phis
                parts.push_back(phi);
                node_phis_[n].push_back(phi);
            }
            if (node.kind == CfgNode::Kind::block) define(p, std::move(parts));
        }
    }

    switch (node.kind) {
        case CfgNode::Kind::block: {
            const brass::Instruction* term = node.block->terminator();
            for (const brass::Instruction* inst : *node.block) {
                origin_ = inst;
                if (inst == term) lower_terminator(n, *inst);
                else lower_instruction(*inst);
            }
            break;
        }
        case CfgNode::Kind::forward:
        case CfgNode::Kind::loop_continue:
            if (node.cont != kNoNode) {
                emit(Inst(spv::OpLoopMerge).id(label_of(node.merge)).id(label_of(node.cont)).lit(spv::LoopControlMaskNone));
            }
            emit(Inst(spv::OpBranch).id(label_of(cfg_->edges[node.out.at(0)].to)));
            break;
        case CfgNode::Kind::unreachable:
            emit(Inst(spv::OpUnreachable));
            break;
        case CfgNode::Kind::prologue:
            break;
    }
}

// Phi operands are filled once every block is lowered, so a back edge's
// arguments (defined after the header) have ids.
void SpirvISel::fill_phis() {
    for (size_t n = 0; n < cfg_->nodes.size(); ++n) {
        const CfgNode& node = cfg_->nodes[n];
        if (!node.params || node_block_[n] == kNoNode) continue;
        Block& b = fn().blocks[node_block_[n]];
        for (size_t e : node.in) {
            const CfgEdge& edge = cfg_->edges[e];
            const CfgNode& from = cfg_->nodes[edge.from];
            std::vector<Id> args;
            if (edge.target) {
                const auto& vals = edge.target->args;
                if (vals.size() != node.params->param_count()) {
                    throw std::runtime_error("SpirvISel: kernel '" + std::string(mir_ ? mir_->name() : "") + "': branch to " +
                                             block_name(edge.target->block) + " passes " + std::to_string(vals.size()) +
                                             " argument(s) but the block takes " +
                                             std::to_string(node.params->param_count()));
                }
                for (const Value* v : vals) {
                    for (Id id : ids_of(v, "branch argument")) args.push_back(id);
                }
            } else if (from.kind == CfgNode::Kind::prologue) {
                args = prologue_args_;
            } else {
                args = node_phis_[edge.from];
            }
            if (args.size() != node_phis_[n].size()) {
                throw std::runtime_error("SpirvISel: internal error: edge into " + node.name + " carries " +
                                         std::to_string(args.size()) + " value(s) for " +
                                         std::to_string(node_phis_[n].size()) + " phi(s)");
            }
            for (size_t k = 0; k < args.size(); ++k) {
                b.insts[k].id(args[k]).id(label_of(edge.from));
            }
        }
    }
}

Id SpirvISel::label_of(size_t node) const {
    return m_.functions[fn_index_].blocks.at(node_block_.at(node)).label;
}

void SpirvISel::lower_terminator(size_t n, const brass::Instruction& term) {
    const CfgNode& node = cfg_->nodes[n];
    switch (term.opcode()) {
        case brass::Opcode::br:
            emit(Inst(spv::OpBranch).id(label_of(cfg_->edges[node.out.at(0)].to)));
            break;
        case brass::Opcode::br_if: {
            Id c = cond_of(term.operand(0));
            emit(Inst(spv::OpSelectionMerge).id(label_of(node.merge)).lit(spv::SelectionControlMaskNone));
            emit(Inst(spv::OpBranchConditional).id(c)
                     .id(label_of(cfg_->edges[node.out.at(0)].to))
                     .id(label_of(cfg_->edges[node.out.at(1)].to)));
            break;
        }
        case brass::Opcode::ret:
            if (term.operand_count() != 0 && term.operand(0) != nullptr) {
                fail("ret with an operand; compute entry points cannot return values");
            }
            emit(Inst(spv::OpReturn));
            break;
        case brass::Opcode::unreachable:
            emit(Inst(spv::OpUnreachable));
            break;
        default:
            fail("unsupported terminator " + std::string(brass::opcode_name(term.opcode())));
    }
}

void SpirvISel::lower_instruction(const brass::Instruction& inst) {
    using brass::Opcode;
    switch (inst.opcode()) {
        // Constants become OpConstant on first use (ids_of).
        case Opcode::iconst_i32:
        case Opcode::iconst_i64:
        case Opcode::fconst_f64:
            break;

        case Opcode::add:
            if (shared_.count(inst.operand(0)) || shared_.count(inst.operand(1))) lower_add_ptr(inst);
            else lower_binary(inst, spv::OpIAdd, spv::OpFAdd);
            break;
        case Opcode::sub: lower_binary(inst, spv::OpISub, spv::OpFSub); break;
        case Opcode::mul: lower_binary(inst, spv::OpIMul, spv::OpFMul); break;
        case Opcode::fma_f32:
        case Opcode::fma_f64:
            define(inst.result(), ext(GLSLstd450Fma, part_type(inst.type()),
                                      {id_of(inst.operand(0), "operand 0"), id_of(inst.operand(1), "operand 1"),
                                       id_of(inst.operand(2), "operand 2")}));
            break;
        case Opcode::sdiv:
            if (inst.type().is_float()) lower_binary(inst, spv::OpFDiv, spv::OpFDiv);
            else lower_int_div_rem(inst, false, false);
            break;
        case Opcode::udiv: lower_int_div_rem(inst, false, true); break;
        case Opcode::smod:
            if (inst.type().is_float()) lower_binary(inst, spv::OpFRem, spv::OpFRem);
            else lower_int_div_rem(inst, true, false);
            break;
        case Opcode::umod: lower_int_div_rem(inst, true, true); break;
        case Opcode::neg:
            define(inst.result(), op(inst.type().is_float() ? spv::OpFNegate : spv::OpSNegate, scalar_type(inst.type()),
                                     {id_of(inst.operand(0), "operand 0")}));
            break;
        case Opcode::sqrt_f32:
        case Opcode::sqrt_f64:  lower_unary_glsl(inst, GLSLstd450Sqrt); break;
        case Opcode::floor_f32:
        case Opcode::floor_f64: lower_unary_glsl(inst, GLSLstd450Floor); break;
        case Opcode::ceil_f32:
        case Opcode::ceil_f64:  lower_unary_glsl(inst, GLSLstd450Ceil); break;
        case Opcode::round_f32:
        case Opcode::round_f64: lower_unary_glsl(inst, GLSLstd450RoundEven); break;
        case Opcode::fabs_f32:
        case Opcode::fabs_f64:  lower_unary_glsl(inst, GLSLstd450FAbs); break;
        case Opcode::fmin_f32:
        case Opcode::fmin_f64:  lower_vector_minmax(inst, false); break; // NMin
        case Opcode::fmax_f32:
        case Opcode::fmax_f64:  lower_vector_minmax(inst, true); break;  // NMax

        case Opcode::and_: lower_binary(inst, spv::OpBitwiseAnd, spv::OpNop); break;
        case Opcode::or_:  lower_binary(inst, spv::OpBitwiseOr, spv::OpNop); break;
        case Opcode::xor_: lower_binary(inst, spv::OpBitwiseXor, spv::OpNop); break;
        case Opcode::not_:
            define(inst.result(), op(spv::OpNot, scalar_type(inst.type()), {id_of(inst.operand(0), "operand 0")}));
            break;
        case Opcode::shl:  lower_binary(inst, spv::OpShiftLeftLogical, spv::OpNop); break;
        case Opcode::lshr: lower_binary(inst, spv::OpShiftRightLogical, spv::OpNop); break;
        case Opcode::ashr: lower_binary(inst, spv::OpShiftRightArithmetic, spv::OpNop); break;

        // Float `ne` is unordered (true for NaN), the rest ordered, as on x64/PTX.
        case Opcode::eq:  lower_comparison(inst, spv::OpIEqual, spv::OpFOrdEqual); break;
        case Opcode::ne:  lower_comparison(inst, spv::OpINotEqual, spv::OpFUnordNotEqual); break;
        case Opcode::slt: lower_comparison(inst, spv::OpSLessThan, spv::OpFOrdLessThan); break;
        case Opcode::ult: lower_comparison(inst, spv::OpULessThan, spv::OpFOrdLessThan); break;
        case Opcode::sle: lower_comparison(inst, spv::OpSLessThanEqual, spv::OpFOrdLessThanEqual); break;
        case Opcode::ule: lower_comparison(inst, spv::OpULessThanEqual, spv::OpFOrdLessThanEqual); break;
        case Opcode::sgt: lower_comparison(inst, spv::OpSGreaterThan, spv::OpFOrdGreaterThan); break;
        case Opcode::ugt: lower_comparison(inst, spv::OpUGreaterThan, spv::OpFOrdGreaterThan); break;
        case Opcode::sge: lower_comparison(inst, spv::OpSGreaterThanEqual, spv::OpFOrdGreaterThanEqual); break;
        case Opcode::uge: lower_comparison(inst, spv::OpUGreaterThanEqual, spv::OpFOrdGreaterThanEqual); break;
        case Opcode::select: lower_select(inst); break;

        case Opcode::sext_i64:  lower_convert(inst, spv::OpSConvert, Type::i64()); break;
        case Opcode::zext_i64:  lower_convert(inst, spv::OpUConvert, Type::i64()); break;
        case Opcode::trunc_i32: lower_convert(inst, spv::OpUConvert, Type::i32()); break;
        case Opcode::sitofp_f64_i32:
        case Opcode::sitofp_f32_i32:
        case Opcode::sitofp_f64_i64:
        case Opcode::sitofp_f32_i64: lower_convert(inst, spv::OpConvertSToF, inst.type()); break;
        case Opcode::fptosi_i32:
        case Opcode::fptosi_i32_f32:
        case Opcode::fptosi_i64:
        case Opcode::fptosi_i64_f32: lower_convert(inst, spv::OpConvertFToS, inst.type()); break;
        case Opcode::fptrunc_f32_f64:
        case Opcode::fpext_f64_f32:  lower_convert(inst, spv::OpFConvert, inst.type()); break;
        case Opcode::bitcast_i64_f64:
        case Opcode::bitcast_f64_i64:
        case Opcode::bitcast_i32_f32:
        case Opcode::bitcast_f32_i32: lower_convert(inst, spv::OpBitcast, inst.type()); break;

        case Opcode::load:          lower_load(inst); break;
        case Opcode::store:         lower_store(inst); break;
        case Opcode::vload:         lower_vload(inst); break;
        case Opcode::vstore:        lower_vstore(inst); break;
        case Opcode::load_indexed:  lower_load_indexed(inst); break;
        case Opcode::store_indexed: lower_store_indexed(inst); break;

        case Opcode::vadd:  lower_vector_binary(inst, spv::OpIAdd, spv::OpFAdd); break;
        case Opcode::vsub:  lower_vector_binary(inst, spv::OpISub, spv::OpFSub); break;
        case Opcode::vmul:  lower_vector_binary(inst, spv::OpIMul, spv::OpFMul); break;
        case Opcode::vdiv:  lower_vector_binary(inst, spv::OpSDiv, spv::OpFDiv); break;
        case Opcode::vmin:  lower_vector_minmax(inst, false); break;
        case Opcode::vmax:  lower_vector_minmax(inst, true); break;
        case Opcode::vfma:  lower_vector_fma(inst); break;
        case Opcode::vneg:
        case Opcode::vsqrt: lower_vector_unary(inst); break;
        case Opcode::vand:  lower_vector_bitwise(inst, spv::OpBitwiseAnd); break;
        case Opcode::vor:   lower_vector_bitwise(inst, spv::OpBitwiseOr); break;
        case Opcode::vxor:  lower_vector_bitwise(inst, spv::OpBitwiseXor); break;
        case Opcode::vnot:  lower_vector_bitwise(inst, spv::OpNot); break;
        case Opcode::vbroadcast:    lower_vbroadcast(inst); break;
        case Opcode::vextract_lane: lower_vextract_lane(inst); break;
        case Opcode::vinsert_lane:  lower_vinsert_lane(inst); break;
        case Opcode::vshuffle:      lower_vshuffle(inst); break;
        case Opcode::vzero:         lower_vzero(inst); break;

        case Opcode::call: lower_call(inst); break;

        case Opcode::safepoint:
        case Opcode::write_barrier:
        case Opcode::resume_point:
            break;

        default:
            fail("unsupported opcode in SPIR-V lowering: " + std::string(brass::opcode_name(inst.opcode())));
    }
}

// ---------------------------------------------------------------------------
// Diagnostics
// ---------------------------------------------------------------------------

void SpirvISel::fail(const std::string& what) const {
    std::string where = "SpirvISel: kernel '" + std::string(mir_ ? mir_->name() : "") + "'";
    if (origin_ && origin_->parent()) where += ", block " + block_name(origin_->parent());
    throw std::runtime_error(where + ": " + what);
}

void SpirvISel::malformed(const brass::Instruction& inst, const char* what) const {
    fail("malformed " + std::string(brass::opcode_name(inst.opcode())) + " (" + what + ")");
}

// ---------------------------------------------------------------------------
// Emission helpers
// ---------------------------------------------------------------------------

Function& SpirvISel::fn() { return m_.functions[fn_index_]; }
Block& SpirvISel::block() { return fn().blocks[cur_]; }

Inst& SpirvISel::emit(Inst inst) {
    if (!inst.origin) inst.origin = origin_;
    block().insts.push_back(std::move(inst));
    return block().insts.back();
}

Id SpirvISel::op(spv::Op o, Id type, std::initializer_list<Id> ids) {
    Id r = m_.new_id();
    Inst inst(o, type, r);
    for (Id i : ids) inst.id(i);
    emit(std::move(inst));
    // Float conversions are NoContraction: without it Mesa's NIR treats them
    // as inexact and folds fpext(fptrunc(x)) to x and fpext(sitofp_f32(i)) to
    // sitofp_f64(i), skipping the rounding MIR requires (found on RADV).
    if (o == spv::OpFConvert || o == spv::OpConvertSToF || o == spv::OpConvertUToF || o == spv::OpConvertFToS ||
        o == spv::OpConvertFToU)
        m_.decorate(r, spv::DecorationNoContraction);
    return r;
}

Id SpirvISel::ext(uint32_t glsl_inst, Id type, std::initializer_list<Id> ids) {
    Id r = m_.new_id();
    Inst inst(spv::OpExtInst, type, r);
    inst.id(m_.glsl_std_450()).lit(glsl_inst);
    for (Id i : ids) inst.id(i);
    emit(std::move(inst));
    return r;
}

void SpirvISel::use_global(Id var) {
    auto& iface = fn().interface;
    for (Id v : iface) if (v == var) return;
    iface.push_back(var);
}

Id SpirvISel::builtin_component(spv::BuiltIn b, uint32_t component) {
    std::string key = std::string("builtin.") + std::to_string(static_cast<uint32_t>(b));
    Id& var = m_.cached(key);
    Id v3 = m_.t_vector(t_u32(), 3);
    if (!var) {
        var = m_.variable(m_.t_pointer(spv::StorageClassInput, v3), spv::StorageClassInput);
        m_.decorate(var, spv::DecorationBuiltIn, {static_cast<uint32_t>(b)});
    }
    Id vid = var;
    use_global(vid);
    Id vec = prologue_value(key, [&] { return op(spv::OpLoad, v3, {vid}); });
    return prologue_value(key + "." + std::to_string(component), [&] {
        Id r = m_.new_id();
        emit(Inst(spv::OpCompositeExtract, t_u32(), r).id(vec).lit(component));
        return r;
    });
}

Id SpirvISel::builtin_scalar(spv::BuiltIn b) {
    std::string key = std::string("builtin.") + std::to_string(static_cast<uint32_t>(b));
    Id& var = m_.cached(key);
    if (!var) {
        var = m_.variable(m_.t_pointer(spv::StorageClassInput, t_u32()), spv::StorageClassInput);
        m_.decorate(var, spv::DecorationBuiltIn, {static_cast<uint32_t>(b)});
    }
    Id vid = var;
    use_global(vid);
    return prologue_value(key, [&] { return op(spv::OpLoad, t_u32(), {vid}); });
}

// The workgroup size: SpecId 0/1/2 constants composed into the module's one
// BuiltIn WorkgroupSize constant, or the fixed LocalSize values.
Id SpirvISel::workgroup_size(uint32_t component) {
    const uint32_t sizes[3] = {opts_.local_size_x, opts_.local_size_y, opts_.local_size_z};
    if (!opts_.local_size_spec_constants) return m_.c_u32(sizes[component]);
    Id& composite = m_.cached("workgroup_size");
    if (!composite) {
        std::vector<Id> parts;
        for (uint32_t k = 0; k < 3; ++k) {
            Id c = m_.spec_constant_u32(sizes[k], k);
            m_.cached("workgroup_size." + std::to_string(k)) = c;
            parts.push_back(c);
        }
        Id comp = m_.spec_composite(m_.t_vector(t_u32(), 3), parts);
        m_.decorate(comp, spv::DecorationBuiltIn, {spv::BuiltInWorkgroupSize});
        m_.cached("workgroup_size") = comp;
    }
    return m_.cached("workgroup_size." + std::to_string(component));
}

// ---------------------------------------------------------------------------
// Types and values
// ---------------------------------------------------------------------------

Id SpirvISel::scalar_type(Type t) {
    if (t.is_i32()) return t_u32();
    if (t.is_i64() || t.is_pointer()) return t_u64();
    if (t.kind() == TypeKind::F32) return t_f32();
    if (t.kind() == TypeKind::F64) return m_.t_float(64);
    fail("MIR type " + std::string(t.name()) + " has no SPIR-V lowering" +
         (t.is_i8() || t.is_i16() ? " (byte and halfword memory goes through ptx_load_u8/u16 and ptx_store_u8/u16)"
                                  : ""));
}

Id SpirvISel::part_type(Type t) {
    if (!t.is_vector()) return scalar_type(t);
    Type e = t.element_type();
    return m_.t_vector(scalar_type(e), static_cast<uint32_t>(16 / e.size_in_bytes()));
}

uint32_t SpirvISel::part_count(Type t) const {
    return t.is_vector() ? static_cast<uint32_t>(t.size_in_bytes() / 16) : 1;
}

Id SpirvISel::psb_pointer(Id pointee) {
    return m_.t_pointer(spv::StorageClassPhysicalStorageBuffer, pointee);
}

const std::vector<Id>& SpirvISel::ids_of(const Value* v, const char* what) {
    if (!v) fail(std::string("malformed instruction (missing ") + what + ")");
    if (shared_.count(v)) {
        fail("shared-memory pointer %" + std::to_string(v->id()) + " used as " + what +
             "; on SPIR-V a ptx_shared_alloc_* result may only be offset with add and passed to the ptx_shared_* intrinsics");
    }
    auto it = ids_.find(v);
    if (it != ids_.end()) return it->second;
    const brass::Instruction* def = v->is_instruction() ? v->defining_instruction() : nullptr;
    if (def && def->opcode() == brass::Opcode::iconst_i32) {
        define(v, m_.c_int(32, static_cast<uint32_t>(def->imm_i32())));
    } else if (def && def->opcode() == brass::Opcode::iconst_i64) {
        define(v, m_.c_int(64, static_cast<uint64_t>(def->imm_i64())));
    } else if (def && def->opcode() == brass::Opcode::fconst_f64) {
        define(v, def->type().kind() == TypeKind::F32 ? m_.c_f32(static_cast<float>(def->imm_f64())) : m_.c_f64(def->imm_f64()));
    } else {
        fail(std::string(what) + " refers to a value (%" + std::to_string(v->id()) + ") with no SPIR-V id");
    }
    return ids_.at(v);
}

Id SpirvISel::cond_of(const Value* v) {
    auto it = bools_.find(v);
    if (it != bools_.end()) return it->second;
    Id x = id_of(v, "condition");
    Type t = v->type();
    if (t.is_float()) {
        Id zero = t.kind() == TypeKind::F32 ? m_.c_f32(0.0f) : m_.c_f64(0.0);
        return op(spv::OpFUnordNotEqual, t_bool(), {x, zero});
    }
    return op(spv::OpINotEqual, t_bool(), {x, m_.c_int(t.is_i32() ? 32 : 64, 0)});
}

bool SpirvISel::has_value_uses(const Value* v) const {
    auto it = value_uses_.find(v);
    return it != value_uses_.end() && it->second > 0;
}

bool SpirvISel::const_int(const Value* v, int64_t* out) const {
    if (!v || !v->is_instruction() || !v->defining_instruction()) return false;
    const brass::Instruction* def = v->defining_instruction();
    if (def->opcode() == brass::Opcode::iconst_i32) { *out = def->imm_i32(); return true; }
    if (def->opcode() == brass::Opcode::iconst_i64) { *out = def->imm_i64(); return true; }
    return false;
}

bool SpirvISel::const_float(const Value* v, double* out) const {
    if (!v || !v->is_instruction() || !v->defining_instruction()) return false;
    const brass::Instruction* def = v->defining_instruction();
    if (def->opcode() != brass::Opcode::fconst_f64) return false;
    *out = def->imm_f64();
    return true;
}

Id SpirvISel::as_u32(const Value* v, const char* what) {
    if (!v) fail(std::string("missing ") + what);
    int64_t c = 0;
    if (const_int(v, &c)) return m_.c_u32(static_cast<uint32_t>(c));
    Id x = id_of(v, what);
    if (v->type().is_i32()) return x;
    if (v->type().is_i64() || v->type().is_pointer()) return op(spv::OpUConvert, t_u32(), {x});
    fail(std::string(what) + " must be an integer");
}

Id SpirvISel::as_u64(const Value* v, const char* what) {
    if (!v) fail(std::string("missing ") + what);
    int64_t c = 0;
    if (const_int(v, &c)) return m_.c_int(64, static_cast<uint64_t>(c));
    Id x = id_of(v, what);
    if (v->type().is_i64() || v->type().is_pointer()) return x;
    if (v->type().is_i32()) return op(spv::OpSConvert, t_u64(), {x});
    fail(std::string(what) + " must be an integer");
}

} // namespace brass::spirv
