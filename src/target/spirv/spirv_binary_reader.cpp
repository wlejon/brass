// Binary SPIR-V reader (`spirv::parse_binary`).

#define SPV_ENABLE_UTILITY_CODE 1
#include <brass/target/spirv/spirv_binary_reader.hpp>

#include <cstring>
#include <unordered_map>

namespace brass::spirv {

namespace {

bool set_err(std::string* err_out, std::string msg) {
    if (err_out) *err_out = std::move(msg);
    return false;
}

bool unpack_string(std::span<const uint32_t> words, size_t& pos, size_t end, std::string& out) {
    out.clear();
    while (pos < end) {
        uint32_t w = words[pos++];
        for (int b = 0; b < 4; ++b) {
            char c = static_cast<char>((w >> (b * 8)) & 0xFFu);
            if (c == '\0') return true;
            out.push_back(c);
        }
    }
    return false;
}

void classify_memory_access(std::span<const uint32_t> raw, size_t mask_idx, Inst& inst) {
    for (size_t i = 0; i < raw.size() && i < mask_idx; ++i) inst.id(raw[i]);
    size_t pos = mask_idx;
    while (pos < raw.size()) {
        uint32_t mask = raw[pos++];
        inst.lit(mask);
        if ((mask & static_cast<uint32_t>(spv::MemoryAccessAlignedMask)) != 0u && pos < raw.size()) {
            inst.lit(raw[pos++]);
        }
        if ((mask & static_cast<uint32_t>(spv::MemoryAccessMakePointerAvailableMask)) != 0u && pos < raw.size()) {
            inst.id(raw[pos++]);
        }
        if ((mask & static_cast<uint32_t>(spv::MemoryAccessMakePointerVisibleMask)) != 0u && pos < raw.size()) {
            inst.id(raw[pos++]);
        }
    }
}

void classify_operands(spv::Op op, std::span<const uint32_t> raw,
                       const std::unordered_map<Id, TypeInfo>& types, Inst& inst) {
    switch (op) {
        case spv::OpCapability:
        case spv::OpMemoryModel:
        case spv::OpConstant:
        case spv::OpSpecConstant:
            for (uint32_t w : raw) inst.lit(w);
            return;

        case spv::OpExecutionMode:
        case spv::OpDecorate:
            if (!raw.empty()) inst.id(raw[0]);
            for (size_t i = 1; i < raw.size(); ++i) inst.lit(raw[i]);
            return;

        case spv::OpExecutionModeId:
        case spv::OpDecorateId:
            if (raw.size() >= 1) inst.id(raw[0]);
            if (raw.size() >= 2) inst.lit(raw[1]);
            for (size_t i = 2; i < raw.size(); ++i) inst.id(raw[i]);
            return;

        case spv::OpMemberDecorate:
            if (raw.size() >= 1) inst.id(raw[0]);
            for (size_t i = 1; i < raw.size(); ++i) inst.lit(raw[i]);
            return;

        case spv::OpTypeInt:
        case spv::OpTypeFloat:
            for (uint32_t w : raw) inst.lit(w);
            return;

        case spv::OpTypeVector:
        case spv::OpTypeMatrix:
        case spv::OpTypeForwardPointer:
            if (raw.size() >= 1) inst.id(raw[0]);
            for (size_t i = 1; i < raw.size(); ++i) inst.lit(raw[i]);
            return;

        case spv::OpTypeImage:
            if (raw.size() >= 1) inst.id(raw[0]);
            for (size_t i = 1; i < raw.size(); ++i) inst.lit(raw[i]);
            return;

        case spv::OpTypePointer:
        case spv::OpVariable:
            if (raw.size() >= 1) inst.lit(raw[0]);
            for (size_t i = 1; i < raw.size(); ++i) inst.id(raw[i]);
            return;

        case spv::OpSpecConstantOp:
            if (!raw.empty()) {
                auto inner = static_cast<spv::Op>(raw[0]);
                inst.lit(raw[0]);
                if (inner == spv::OpCompositeExtract) {
                    if (raw.size() >= 2) inst.id(raw[1]);
                    for (size_t i = 2; i < raw.size(); ++i) inst.lit(raw[i]);
                } else if (inner == spv::OpCompositeInsert || inner == spv::OpVectorShuffle) {
                    if (raw.size() >= 2) inst.id(raw[1]);
                    if (raw.size() >= 3) inst.id(raw[2]);
                    for (size_t i = 3; i < raw.size(); ++i) inst.lit(raw[i]);
                } else {
                    for (size_t i = 1; i < raw.size(); ++i) inst.id(raw[i]);
                }
            }
            return;

        case spv::OpExtInst:
            if (raw.size() >= 1) inst.id(raw[0]);
            if (raw.size() >= 2) inst.lit(raw[1]);
            for (size_t i = 2; i < raw.size(); ++i) inst.id(raw[i]);
            return;

        case spv::OpCompositeExtract:
            if (raw.size() >= 1) inst.id(raw[0]);
            for (size_t i = 1; i < raw.size(); ++i) inst.lit(raw[i]);
            return;

        case spv::OpCompositeInsert:
        case spv::OpVectorShuffle:
            if (raw.size() >= 1) inst.id(raw[0]);
            if (raw.size() >= 2) inst.id(raw[1]);
            for (size_t i = 2; i < raw.size(); ++i) inst.lit(raw[i]);
            return;

        case spv::OpLoad:
            classify_memory_access(raw, 1, inst);
            return;

        case spv::OpStore:
        case spv::OpCopyMemory:
            classify_memory_access(raw, 2, inst);
            return;

        case spv::OpImageSampleImplicitLod:
        case spv::OpImageSampleExplicitLod:
        case spv::OpImageFetch:
        case spv::OpImageRead:
            for (size_t i = 0; i < raw.size(); ++i) {
                if (i == 2) inst.lit(raw[i]);
                else inst.id(raw[i]);
            }
            return;

        case spv::OpImageWrite:
            for (size_t i = 0; i < raw.size(); ++i) {
                if (i == 3) inst.lit(raw[i]);
                else inst.id(raw[i]);
            }
            return;

        case spv::OpLoopMerge:
            if (raw.size() >= 1) inst.id(raw[0]);
            if (raw.size() >= 2) inst.id(raw[1]);
            for (size_t i = 2; i < raw.size(); ++i) inst.lit(raw[i]);
            return;

        case spv::OpSelectionMerge:
            if (raw.size() >= 1) inst.id(raw[0]);
            for (size_t i = 1; i < raw.size(); ++i) inst.lit(raw[i]);
            return;

        case spv::OpBranchConditional:
            for (size_t i = 0; i < raw.size(); ++i) {
                if (i < 3) inst.id(raw[i]);
                else inst.lit(raw[i]);
            }
            return;

        case spv::OpSwitch: {
            if (raw.size() >= 1) inst.id(raw[0]);
            if (raw.size() >= 2) inst.id(raw[1]);
            // Selector may be 64-bit integer (2 literal words per case) or 32-bit (1 literal word per case).
            bool wide = false;
            if (!raw.empty()) {
                // Check if remaining words are a multiple of 3 but not 2, or check selector type if known.
                size_t rem = raw.size() - 2;
                if (rem % 3 == 0 && rem % 2 != 0) wide = true;
            }
            size_t step = wide ? 3 : 2;
            for (size_t i = 2; i < raw.size(); i += step) {
                for (size_t k = 0; k + 1 < step && i + k < raw.size(); ++k) inst.lit(raw[i + k]);
                if (i + step - 1 < raw.size()) inst.id(raw[i + step - 1]);
            }
            (void)types;
            return;
        }

        case spv::OpGroupNonUniformBallotBitCount:
        case spv::OpGroupNonUniformIAdd:
        case spv::OpGroupNonUniformFAdd:
        case spv::OpGroupNonUniformIMul:
        case spv::OpGroupNonUniformFMul:
        case spv::OpGroupNonUniformSMin:
        case spv::OpGroupNonUniformUMin:
        case spv::OpGroupNonUniformFMin:
        case spv::OpGroupNonUniformSMax:
        case spv::OpGroupNonUniformUMax:
        case spv::OpGroupNonUniformFMax:
        case spv::OpGroupNonUniformBitwiseAnd:
        case spv::OpGroupNonUniformBitwiseOr:
        case spv::OpGroupNonUniformBitwiseXor:
        case spv::OpGroupNonUniformLogicalAnd:
        case spv::OpGroupNonUniformLogicalOr:
        case spv::OpGroupNonUniformLogicalXor:
            if (raw.size() >= 1) inst.id(raw[0]);
            if (raw.size() >= 2) inst.lit(raw[1]);
            for (size_t i = 2; i < raw.size(); ++i) inst.id(raw[i]);
            return;

        case spv::OpCooperativeMatrixLoadKHR:
            classify_memory_access(raw, 3, inst);
            return;

        case spv::OpCooperativeMatrixStoreKHR:
            classify_memory_access(raw, 4, inst);
            return;

        case spv::OpCooperativeMatrixMulAddKHR:
            for (size_t i = 0; i < raw.size(); ++i) {
                if (i < 3) inst.id(raw[i]);
                else inst.lit(raw[i]);
            }
            return;

        default:
            for (uint32_t w : raw) inst.id(w);
            return;
    }
}

uint64_t eval_spec_op(spv::Op op, const std::vector<Operand>& ops,
                      const std::unordered_map<Id, uint64_t>& consts) {
    auto get = [&](size_t idx) -> uint64_t {
        if (idx >= ops.size()) return 0;
        auto it = consts.find(ops[idx].value);
        return it == consts.end() ? 0 : it->second;
    };
    uint64_t a = get(1), b = get(2), c = get(3);
    switch (op) {
        case spv::OpSNegate: return static_cast<uint64_t>(-static_cast<int64_t>(a));
        case spv::OpNot: return ~a;
        case spv::OpLogicalNot: return a == 0 ? 1 : 0;
        case spv::OpIAdd: return a + b;
        case spv::OpISub: return a - b;
        case spv::OpIMul: return a * b;
        case spv::OpUDiv: return b != 0 ? a / b : 0;
        case spv::OpSDiv: return b != 0 ? static_cast<uint64_t>(static_cast<int64_t>(a) / static_cast<int64_t>(b)) : 0;
        case spv::OpUMod: return b != 0 ? a % b : 0;
        case spv::OpSRem:
        case spv::OpSMod: return b != 0 ? static_cast<uint64_t>(static_cast<int64_t>(a) % static_cast<int64_t>(b)) : 0;
        case spv::OpShiftRightLogical: return (b < 64) ? (a >> b) : 0;
        case spv::OpShiftRightArithmetic: return (b < 64) ? static_cast<uint64_t>(static_cast<int64_t>(a) >> b) : 0;
        case spv::OpShiftLeftLogical: return (b < 64) ? (a << b) : 0;
        case spv::OpBitwiseOr: return a | b;
        case spv::OpBitwiseXor: return a ^ b;
        case spv::OpBitwiseAnd: return a & b;
        case spv::OpLogicalOr: return (a != 0 || b != 0) ? 1 : 0;
        case spv::OpLogicalAnd: return (a != 0 && b != 0) ? 1 : 0;
        case spv::OpLogicalEqual:
        case spv::OpIEqual: return a == b ? 1 : 0;
        case spv::OpLogicalNotEqual:
        case spv::OpINotEqual: return a != b ? 1 : 0;
        case spv::OpULessThan: return static_cast<uint32_t>(a) < static_cast<uint32_t>(b) ? 1 : 0;
        case spv::OpULessThanEqual: return static_cast<uint32_t>(a) <= static_cast<uint32_t>(b) ? 1 : 0;
        case spv::OpUGreaterThan: return static_cast<uint32_t>(a) > static_cast<uint32_t>(b) ? 1 : 0;
        case spv::OpUGreaterThanEqual: return static_cast<uint32_t>(a) >= static_cast<uint32_t>(b) ? 1 : 0;
        case spv::OpSLessThan: return static_cast<int32_t>(a) < static_cast<int32_t>(b) ? 1 : 0;
        case spv::OpSLessThanEqual: return static_cast<int32_t>(a) <= static_cast<int32_t>(b) ? 1 : 0;
        case spv::OpSGreaterThan: return static_cast<int32_t>(a) > static_cast<int32_t>(b) ? 1 : 0;
        case spv::OpSGreaterThanEqual: return static_cast<int32_t>(a) >= static_cast<int32_t>(b) ? 1 : 0;
        case spv::OpSelect: return a != 0 ? b : c;
        case spv::OpSConvert:
        case spv::OpUConvert: return a;
        default: return 0;
    }
}

bool is_type_opcode(spv::Op op) {
    switch (op) {
        case spv::OpTypeVoid:
        case spv::OpTypeBool:
        case spv::OpTypeInt:
        case spv::OpTypeFloat:
        case spv::OpTypeVector:
        case spv::OpTypeMatrix:
        case spv::OpTypeImage:
        case spv::OpTypeSampler:
        case spv::OpTypeSampledImage:
        case spv::OpTypeArray:
        case spv::OpTypeRuntimeArray:
        case spv::OpTypeStruct:
        case spv::OpTypeOpaque:
        case spv::OpTypePointer:
        case spv::OpTypeFunction:
        case spv::OpTypeCooperativeMatrixKHR:
        case spv::OpTypeCooperativeMatrixNV:
        case spv::OpTypeAccelerationStructureKHR:
        case spv::OpTypeRayQueryKHR:
            return true;
        default:
            return false;
    }
}

} // namespace

bool parse_binary(std::span<const uint32_t> words, Module& out_mod, std::string* err_out) {
    if (words.size() < 5) {
        return set_err(err_out, "SPIR-V binary too short (fewer than 5 header words)");
    }
    if (words[0] != spv::MagicNumber) {
        return set_err(err_out, "invalid SPIR-V magic number");
    }
    uint32_t version = words[1];
    if (version < 0x00010000u || version > 0x00010600u || (version & 0xFF0000FFu) != 0u) {
        return set_err(err_out, "unsupported or malformed SPIR-V version word");
    }
    Id bound = words[3];
    if (bound == 0) {
        return set_err(err_out, "invalid SPIR-V id bound 0");
    }
    if (words[4] != 0) {
        return set_err(err_out, "non-zero SPIR-V header schema word");
    }

    out_mod.reset_from_binary(version, bound);

    std::unordered_map<Id, std::string> id_names;
    std::unordered_map<Id, TypeInfo> parsed_types;
    std::unordered_map<Id, uint64_t> const_vals;

    Function* cur_fn = nullptr;
    Block* cur_block = nullptr;

    size_t pos = 5;
    while (pos < words.size()) {
        uint32_t w0 = words[pos];
        uint32_t wc = w0 >> spv::WordCountShift;
        auto op = static_cast<spv::Op>(w0 & spv::OpCodeMask);
        if (wc == 0) {
            return set_err(err_out, "instruction with word count 0 at word " + std::to_string(pos));
        }
        if (pos + wc > words.size()) {
            return set_err(err_out, "truncated instruction at word " + std::to_string(pos));
        }

        size_t inst_end = pos + wc;
        if (op == spv::OpNop || op == spv::OpLine || op == spv::OpNoLine ||
            op == spv::OpSource || op == spv::OpSourceContinued || op == spv::OpSourceExtension ||
            op == spv::OpString || op == spv::OpMemberName || op == spv::OpModuleProcessed) {
            pos = inst_end;
            continue;
        }

        if (op == spv::OpExtension) {
            size_t p = pos + 1;
            std::string ext;
            if (!unpack_string(words, p, inst_end, ext)) {
                return set_err(err_out, "malformed string in OpExtension");
            }
            out_mod.add_extension(ext);
            pos = inst_end;
            continue;
        }
        if (op == spv::OpExtInstImport) {
            if (wc < 3) return set_err(err_out, "truncated OpExtInstImport");
            Id id = words[pos + 1];
            size_t p = pos + 2;
            std::string name;
            if (!unpack_string(words, p, inst_end, name)) {
                return set_err(err_out, "malformed string in OpExtInstImport");
            }
            out_mod.ext_imports.emplace_back(id, std::move(name));
            pos = inst_end;
            continue;
        }
        if (op == spv::OpEntryPoint) {
            if (wc < 4) return set_err(err_out, "truncated OpEntryPoint");
            auto model = static_cast<spv::ExecutionModel>(words[pos + 1]);
            Id fn_id = words[pos + 2];
            size_t p = pos + 3;
            std::string name;
            if (!unpack_string(words, p, inst_end, name)) {
                return set_err(err_out, "malformed string in OpEntryPoint");
            }
            std::vector<Id> iface;
            while (p < inst_end) iface.push_back(words[p++]);
            out_mod.add_entry_point(model, fn_id, std::move(name), std::move(iface));
            pos = inst_end;
            continue;
        }
        if (op == spv::OpName) {
            if (wc < 3) return set_err(err_out, "truncated OpName");
            Id target = words[pos + 1];
            size_t p = pos + 2;
            std::string name;
            if (!unpack_string(words, p, inst_end, name)) {
                return set_err(err_out, "malformed string in OpName");
            }
            id_names[target] = name;
            out_mod.set_name(target, std::move(name));
            pos = inst_end;
            continue;
        }

        bool has_result = false;
        bool has_result_type = false;
        spv::HasResultAndType(op, &has_result, &has_result_type);
        size_t need = 1u + (has_result_type ? 1u : 0u) + (has_result ? 1u : 0u);
        if (wc < need) {
            return set_err(err_out, "instruction word count too small for opcode " + std::string(op_name(op)));
        }

        size_t cur = pos + 1;
        Id type_id = has_result_type ? words[cur++] : 0;
        Id result_id = has_result ? words[cur++] : 0;
        Inst inst(op, type_id, result_id);
        classify_operands(op, words.subspan(cur, inst_end - cur), parsed_types, inst);

        if (op == spv::OpCapability) {
            if (!inst.operands.empty()) {
                out_mod.add_capability(static_cast<spv::Capability>(inst.operands[0].value));
            }
        } else if (op == spv::OpMemoryModel) {
            if (inst.operands.size() >= 2) {
                out_mod.set_memory_model(static_cast<spv::AddressingModel>(inst.operands[0].value),
                                         static_cast<spv::MemoryModel>(inst.operands[1].value));
            }
        } else if (op == spv::OpExecutionMode || op == spv::OpExecutionModeId) {
            out_mod.execution_modes.push_back(std::move(inst));
        } else if (op == spv::OpDecorate || op == spv::OpMemberDecorate || op == spv::OpDecorateId) {
            out_mod.annotations.push_back(std::move(inst));
        } else if (op == spv::OpFunction) {
            Function f;
            f.id = result_id;
            f.return_type = type_id;
            f.control = inst.operands.size() >= 1 ? inst.operands[0].value
                                                  : static_cast<uint32_t>(spv::FunctionControlMaskNone);
            f.fn_type = inst.operands.size() >= 2 ? inst.operands[1].value : 0;
            auto nit = id_names.find(result_id);
            f.name = (nit != id_names.end()) ? nit->second : ("fn_" + std::to_string(result_id));
            out_mod.functions.push_back(std::move(f));
            cur_fn = &out_mod.functions.back();
            cur_block = nullptr;
        } else if (op == spv::OpFunctionParameter) {
            if (!cur_fn) return set_err(err_out, "OpFunctionParameter outside function");
            if (result_id && type_id) out_mod.register_value_type(result_id, type_id);
            cur_fn->params.push_back(std::move(inst));
        } else if (op == spv::OpFunctionEnd) {
            if (!cur_fn) return set_err(err_out, "OpFunctionEnd without matching OpFunction");
            cur_fn = nullptr;
            cur_block = nullptr;
        } else if (op == spv::OpLabel) {
            if (!cur_fn) return set_err(err_out, "OpLabel outside function");
            cur_fn->blocks.push_back(Block{result_id, {}});
            cur_block = &cur_fn->blocks.back();
        } else if (cur_fn != nullptr) {
            if (!cur_block) return set_err(err_out, "instruction inside function before OpLabel");
            if (result_id && type_id) out_mod.register_value_type(result_id, type_id);
            cur_block->insts.push_back(std::move(inst));
        } else {
            // Global section: types, constants, spec constants, global variables, OpUndef.
            if (is_type_opcode(op)) {
                TypeInfo ti{op};
                if ((op == spv::OpTypeInt || op == spv::OpTypeFloat) && !inst.operands.empty()) {
                    ti.width = inst.operands[0].value;
                } else if ((op == spv::OpTypeVector || op == spv::OpTypeMatrix) && inst.operands.size() >= 2) {
                    ti.elem = inst.operands[0].value;
                    ti.width = inst.operands[1].value;
                } else if (op == spv::OpTypeArray && inst.operands.size() >= 2) {
                    ti.elem = inst.operands[0].value;
                    auto cit = const_vals.find(inst.operands[1].value);
                    ti.width = (cit != const_vals.end()) ? static_cast<uint32_t>(cit->second) : 0;
                } else if (op == spv::OpTypeRuntimeArray && !inst.operands.empty()) {
                    ti.elem = inst.operands[0].value;
                } else if (op == spv::OpTypeStruct) {
                    ti.width = static_cast<uint32_t>(inst.operands.size());
                } else if (op == spv::OpTypePointer && inst.operands.size() >= 2) {
                    ti.storage = static_cast<spv::StorageClass>(inst.operands[0].value);
                    ti.elem = inst.operands[1].value;
                } else if (op == spv::OpTypeFunction && !inst.operands.empty()) {
                    ti.elem = inst.operands[0].value;
                }
                parsed_types[result_id] = ti;
                out_mod.register_type(result_id, ti);
            } else {
                if (result_id && type_id) out_mod.register_value_type(result_id, type_id);
                if (op == spv::OpConstant || op == spv::OpSpecConstant) {
                    uint64_t val = 0;
                    if (inst.operands.size() == 1) val = inst.operands[0].value;
                    else if (inst.operands.size() >= 2) {
                        val = static_cast<uint64_t>(inst.operands[0].value) |
                              (static_cast<uint64_t>(inst.operands[1].value) << 32);
                    }
                    const_vals[result_id] = val;
                } else if (op == spv::OpConstantTrue || op == spv::OpSpecConstantTrue) {
                    const_vals[result_id] = 1;
                } else if (op == spv::OpConstantFalse || op == spv::OpSpecConstantFalse || op == spv::OpConstantNull) {
                    const_vals[result_id] = 0;
                } else if (op == spv::OpSpecConstantOp && !inst.operands.empty()) {
                    const_vals[result_id] = eval_spec_op(static_cast<spv::Op>(inst.operands[0].value),
                                                         inst.operands, const_vals);
                }
            }
            out_mod.globals.push_back(std::move(inst));
        }

        pos = inst_end;
    }

    if (cur_fn != nullptr) {
        return set_err(err_out, "unterminated OpFunction at end of binary");
    }
    return true;
}

} // namespace brass::spirv
