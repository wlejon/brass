// spirv::Module construction (type/constant tables), the binary writer and
// the text dump. The verifier is in spirv_verifier.cpp.

#define SPV_ENABLE_UTILITY_CODE 1
#include <brass/target/spirv/spirv_ir.hpp>

#include <algorithm>
#include <cstring>
#include <sstream>
#include <stdexcept>

namespace brass::spirv {

namespace {

constexpr uint32_t kMagic = 0x07230203;

void push_string(std::vector<uint32_t>& out, const std::string& s) {
    // UTF-8 bytes, nul-terminated, padded to a word boundary.
    size_t words = s.size() / 4 + 1;
    size_t start = out.size();
    out.resize(start + words, 0);
    std::memcpy(out.data() + start, s.data(), s.size());
}

void encode(std::vector<uint32_t>& out, const Inst& inst) {
    size_t at = out.size();
    out.push_back(0);
    if (inst.type) out.push_back(inst.type);
    if (inst.result) out.push_back(inst.result);
    for (const Operand& o : inst.operands) out.push_back(o.value);
    auto count = static_cast<uint32_t>(out.size() - at);
    out[at] = (count << spv::WordCountShift) | static_cast<uint32_t>(inst.op);
}

// Encodes an instruction whose operand list ends in a literal string.
void encode_with_string(std::vector<uint32_t>& out, spv::Op op, const std::vector<uint32_t>& head,
                        const std::string& s, const std::vector<uint32_t>& tail = {}) {
    size_t at = out.size();
    out.push_back(0);
    out.insert(out.end(), head.begin(), head.end());
    push_string(out, s);
    out.insert(out.end(), tail.begin(), tail.end());
    auto count = static_cast<uint32_t>(out.size() - at);
    out[at] = (count << spv::WordCountShift) | static_cast<uint32_t>(op);
}

} // namespace

// ---------------------------------------------------------------------------
// Module
// ---------------------------------------------------------------------------

Module::Module(uint32_t version) : version_(version) {
    if (version < 0x00010500) {
        throw std::invalid_argument("spirv::Module: SPIR-V 1.5 or later is required "
                                    "(PhysicalStorageBuffer addressing and 8/16-bit storage are core there)");
    }
    add_capability(spv::CapabilityShader);
}

void Module::add_capability(spv::Capability c) {
    if (!has_capability(c)) capabilities.push_back(c);
}

bool Module::has_capability(spv::Capability c) const {
    return std::find(capabilities.begin(), capabilities.end(), c) != capabilities.end();
}

void Module::add_extension(const std::string& name) {
    if (std::find(extensions.begin(), extensions.end(), name) == extensions.end()) extensions.push_back(name);
}

Id Module::glsl_std_450() {
    for (const auto& imp : ext_imports) {
        if (imp.second == "GLSL.std.450") return imp.first;
    }
    Id id = new_id();
    ext_imports.emplace_back(id, "GLSL.std.450");
    return id;
}

EntryPoint& Module::add_entry_point(spv::ExecutionModel model, Id fn, std::string name, std::vector<Id> interface) {
    entry_points.push_back(EntryPoint{model, fn, std::move(name), std::move(interface)});
    return entry_points.back();
}

void Module::add_execution_mode(Id fn, spv::ExecutionMode mode, std::vector<uint32_t> literals) {
    Inst inst(spv::OpExecutionMode);
    inst.id(fn).lit(static_cast<uint32_t>(mode));
    for (uint32_t l : literals) inst.lit(l);
    execution_modes.push_back(std::move(inst));
}

void Module::set_name(Id target, std::string name) {
    names.emplace_back(target, std::move(name));
}

void Module::decorate(Id target, spv::Decoration d, std::vector<uint32_t> literals) {
    Inst inst(spv::OpDecorate);
    inst.id(target).lit(static_cast<uint32_t>(d));
    for (uint32_t l : literals) inst.lit(l);
    annotations.push_back(std::move(inst));
}

void Module::member_decorate(Id struct_type, uint32_t member, spv::Decoration d, std::vector<uint32_t> literals) {
    Inst inst(spv::OpMemberDecorate);
    inst.id(struct_type).lit(member).lit(static_cast<uint32_t>(d));
    for (uint32_t l : literals) inst.lit(l);
    annotations.push_back(std::move(inst));
}

Id Module::intern(Inst inst, const TypeInfo* info) {
    std::vector<uint32_t> key = {static_cast<uint32_t>(inst.op), inst.type};
    for (const Operand& o : inst.operands) key.push_back(o.value);
    auto it = interned_.find(key);
    if (it != interned_.end()) return it->second;
    Id id = new_id();
    inst.result = id;
    if (info) types_[id] = *info;
    else if (inst.type) value_types_[id] = inst.type;
    globals.push_back(std::move(inst));
    interned_.emplace(std::move(key), id);
    return id;
}

Id Module::t_void() {
    TypeInfo ti{spv::OpTypeVoid};
    return intern(Inst(spv::OpTypeVoid), &ti);
}

Id Module::t_bool() {
    TypeInfo ti{spv::OpTypeBool};
    return intern(Inst(spv::OpTypeBool), &ti);
}

Id Module::t_int(uint32_t width) {
    if (width == 64) add_capability(spv::CapabilityInt64);
    TypeInfo ti{spv::OpTypeInt, width};
    return intern(Inst(spv::OpTypeInt).lit(width).lit(0), &ti);
}

Id Module::t_float(uint32_t width) {
    if (width == 64) add_capability(spv::CapabilityFloat64);
    TypeInfo ti{spv::OpTypeFloat, width};
    return intern(Inst(spv::OpTypeFloat).lit(width), &ti);
}

Id Module::t_vector(Id elem, uint32_t count) {
    TypeInfo ti{spv::OpTypeVector, count, elem};
    return intern(Inst(spv::OpTypeVector).id(elem).lit(count), &ti);
}

Id Module::t_array(Id elem, uint32_t length) {
    Id len = c_u32(length);
    TypeInfo ti{spv::OpTypeArray, length, elem};
    return intern(Inst(spv::OpTypeArray).id(elem).id(len), &ti);
}

Id Module::t_struct(const std::vector<Id>& members) {
    Id id = new_id();
    Inst inst(spv::OpTypeStruct, 0, id);
    for (Id m : members) inst.id(m);
    types_[id] = TypeInfo{spv::OpTypeStruct, static_cast<uint32_t>(members.size())};
    globals.push_back(std::move(inst));
    return id;
}

Id Module::t_pointer(spv::StorageClass sc, Id pointee) {
    TypeInfo ti{spv::OpTypePointer, 0, pointee, sc};
    return intern(Inst(spv::OpTypePointer).lit(static_cast<uint32_t>(sc)).id(pointee), &ti);
}

Id Module::t_function(Id ret, const std::vector<Id>& params) {
    Inst inst(spv::OpTypeFunction);
    inst.id(ret);
    for (Id p : params) inst.id(p);
    TypeInfo ti{spv::OpTypeFunction, 0, ret};
    return intern(std::move(inst), &ti);
}

const TypeInfo* Module::type_info(Id type) const {
    auto it = types_.find(type);
    return it == types_.end() ? nullptr : &it->second;
}

Id Module::c_int(uint32_t width, uint64_t bits) {
    Inst inst(spv::OpConstant, t_int(width));
    if (width == 64) inst.lit(static_cast<uint32_t>(bits)).lit(static_cast<uint32_t>(bits >> 32));
    else inst.lit(static_cast<uint32_t>(bits & (width == 32 ? 0xFFFFFFFFull : ((1ull << width) - 1))));
    return intern(std::move(inst), nullptr);
}

Id Module::c_f32(float v) {
    uint32_t bits = 0;
    std::memcpy(&bits, &v, 4);
    return intern(Inst(spv::OpConstant, t_float(32)).lit(bits), nullptr);
}

Id Module::c_f64(double v) {
    uint64_t bits = 0;
    std::memcpy(&bits, &v, 8);
    return intern(Inst(spv::OpConstant, t_float(64)).lit(static_cast<uint32_t>(bits)).lit(static_cast<uint32_t>(bits >> 32)),
                  nullptr);
}

Id Module::c_bool(bool v) {
    return intern(Inst(v ? spv::OpConstantTrue : spv::OpConstantFalse, t_bool()), nullptr);
}

Id Module::c_null(Id type) {
    return intern(Inst(spv::OpConstantNull, type), nullptr);
}

Id Module::c_composite(Id type, const std::vector<Id>& parts) {
    Inst inst(spv::OpConstantComposite, type);
    for (Id p : parts) inst.id(p);
    return intern(std::move(inst), nullptr);
}

Id Module::spec_constant_u32(uint32_t default_value, uint32_t spec_id) {
    Id id = new_id();
    Id t = t_int(32);
    globals.push_back(Inst(spv::OpSpecConstant, t, id).lit(default_value));
    value_types_[id] = t;
    decorate(id, spv::DecorationSpecId, {spec_id});
    return id;
}

Id Module::spec_composite(Id type, const std::vector<Id>& parts) {
    Id id = new_id();
    Inst inst(spv::OpSpecConstantComposite, type, id);
    for (Id p : parts) inst.id(p);
    globals.push_back(std::move(inst));
    value_types_[id] = type;
    return id;
}

Id Module::type_of_value(Id value) const {
    auto it = value_types_.find(value);
    return it == value_types_.end() ? 0 : it->second;
}

Id Module::variable(Id pointer_type, spv::StorageClass sc) {
    Id id = new_id();
    globals.push_back(Inst(spv::OpVariable, pointer_type, id).lit(static_cast<uint32_t>(sc)));
    value_types_[id] = pointer_type;
    return id;
}

Function& Module::add_function(std::string name, Id return_type, Id fn_type) {
    Function f;
    f.id = new_id();
    f.return_type = return_type;
    f.fn_type = fn_type;
    f.name = std::move(name);
    functions.push_back(std::move(f));
    return functions.back();
}

// ---------------------------------------------------------------------------
// Writer
// ---------------------------------------------------------------------------

std::vector<uint32_t> write(const Module& m) {
    std::vector<uint32_t> out = {kMagic, m.version(), 0 /* generator */, m.bound(), 0 /* schema */};

    for (spv::Capability c : m.capabilities) encode(out, Inst(spv::OpCapability).lit(static_cast<uint32_t>(c)));
    for (const std::string& e : m.extensions) encode_with_string(out, spv::OpExtension, {}, e);
    for (const auto& imp : m.ext_imports) encode_with_string(out, spv::OpExtInstImport, {imp.first}, imp.second);
    encode(out, Inst(spv::OpMemoryModel).lit(static_cast<uint32_t>(m.addressing()))
                    .lit(static_cast<uint32_t>(m.memory_model())));
    for (const EntryPoint& ep : m.entry_points) {
        encode_with_string(out, spv::OpEntryPoint, {static_cast<uint32_t>(ep.model), ep.function}, ep.name, ep.interface);
    }
    for (const Inst& inst : m.execution_modes) encode(out, inst);
    for (const auto& n : m.names) encode_with_string(out, spv::OpName, {n.first}, n.second);
    for (const Inst& inst : m.annotations) encode(out, inst);
    for (const Inst& inst : m.globals) encode(out, inst);
    for (const Function& f : m.functions) {
        encode(out, Inst(spv::OpFunction, f.return_type, f.id).lit(spv::FunctionControlMaskNone).id(f.fn_type));
        for (const Block& b : f.blocks) {
            encode(out, Inst(spv::OpLabel, 0, b.label));
            for (const Inst& inst : b.insts) encode(out, inst);
        }
        encode(out, Inst(spv::OpFunctionEnd));
    }
    return out;
}

// ---------------------------------------------------------------------------
// Text
// ---------------------------------------------------------------------------

const char* op_name(spv::Op op) {
    return spv::OpToString(op);
}

bool is_terminator(spv::Op op) {
    switch (op) {
        case spv::OpBranch:
        case spv::OpBranchConditional:
        case spv::OpSwitch:
        case spv::OpReturn:
        case spv::OpReturnValue:
        case spv::OpUnreachable:
        case spv::OpKill:
        case spv::OpTerminateInvocation:
            return true;
        default:
            return false;
    }
}

std::string to_string(const Inst& inst) {
    std::ostringstream os;
    if (inst.result) os << "%" << inst.result << " = ";
    os << op_name(inst.op);
    if (inst.type) os << " %" << inst.type;
    for (const Operand& o : inst.operands) {
        if (o.is_id()) os << " %" << o.value;
        else os << " " << o.value;
    }
    return os.str();
}

std::string dump(const Module& m) {
    std::ostringstream os;
    os << "; SPIR-V " << (m.version() >> 16) << "." << ((m.version() >> 8) & 0xFF) << ", bound " << m.bound() << "\n";
    for (spv::Capability c : m.capabilities) os << "OpCapability " << spv::CapabilityToString(c) << "\n";
    for (const std::string& e : m.extensions) os << "OpExtension \"" << e << "\"\n";
    for (const auto& imp : m.ext_imports) os << "%" << imp.first << " = OpExtInstImport \"" << imp.second << "\"\n";
    os << "OpMemoryModel " << spv::AddressingModelToString(m.addressing()) << " "
       << spv::MemoryModelToString(m.memory_model()) << "\n";
    for (const EntryPoint& ep : m.entry_points) {
        os << "OpEntryPoint " << spv::ExecutionModelToString(ep.model) << " %" << ep.function << " \"" << ep.name << "\"";
        for (Id i : ep.interface) os << " %" << i;
        os << "\n";
    }
    for (const Inst& inst : m.execution_modes) os << to_string(inst) << "\n";
    for (const auto& n : m.names) os << "OpName %" << n.first << " \"" << n.second << "\"\n";
    for (const Inst& inst : m.annotations) os << to_string(inst) << "\n";
    for (const Inst& inst : m.globals) os << to_string(inst) << "\n";
    for (const Function& f : m.functions) {
        os << "%" << f.id << " = OpFunction %" << f.return_type << " None %" << f.fn_type << "   ; " << f.name << "\n";
        for (const Block& b : f.blocks) {
            os << "%" << b.label << " = OpLabel\n";
            for (const Inst& inst : b.insts) os << "    " << to_string(inst) << "\n";
        }
        os << "OpFunctionEnd\n";
    }
    return os.str();
}

} // namespace brass::spirv
