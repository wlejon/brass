#pragma once

// Typed SPIR-V module model for the SPIR-V compute target.
//
// A Module holds SPIR-V as data: ids, a deduplicating type and constant
// table, decorations, entry points and functions made of blocks of `Inst`s
// whose operands are tagged as ids or literal words. Nothing is encoded
// until `write()` produces the binary; `verify()` checks the structural
// rules that matter for the backend (see docs/spirv_backend_design.md) and
// `dump()` prints a spirv-dis-like listing for tests and diagnostics.
//
// Opcodes and enumerants come from the Khronos SPIR-V headers
// (<spirv/unified1/spirv.hpp>); nothing here hand-types an opcode number.

#include <spirv/unified1/spirv.hpp>

#include <cstdint>
#include <map>
#include <string>
#include <unordered_map>
#include <vector>

namespace brass {
class Instruction;
}

namespace brass::spirv {

using Id = uint32_t;

// One operand word after the result-type/result-id words. Ids and literals
// are distinguished so the verifier can check that every id is defined.
struct Operand {
    enum class Kind : uint8_t { id, literal };
    Kind kind = Kind::id;
    uint32_t value = 0;

    static Operand ref(Id id) { return Operand{Kind::id, id}; }
    static Operand lit(uint32_t v) { return Operand{Kind::literal, v}; }
    bool is_id() const { return kind == Kind::id; }
};

struct Inst {
    spv::Op op = spv::OpNop;
    Id type = 0;   // result type id, 0 when the opcode has none
    Id result = 0; // result id, 0 when the opcode has none
    std::vector<Operand> operands;
    const brass::Instruction* origin = nullptr; // MIR instruction this came from

    Inst() = default;
    explicit Inst(spv::Op o, Id t = 0, Id r = 0) : op(o), type(t), result(r) {}
    Inst& id(Id v) { operands.push_back(Operand::ref(v)); return *this; }
    Inst& lit(uint32_t v) { operands.push_back(Operand::lit(v)); return *this; }
};

struct Block {
    Id label = 0;
    std::vector<Inst> insts;
};

struct Function {
    Id id = 0;
    Id return_type = 0;
    Id fn_type = 0;
    std::string name;
    std::vector<Block> blocks;
    // Global variables the function references (the OpEntryPoint interface
    // must list every one of them since SPIR-V 1.4).
    std::vector<Id> interface;
};

struct EntryPoint {
    spv::ExecutionModel model = spv::ExecutionModelGLCompute;
    Id function = 0;
    std::string name;
    std::vector<Id> interface;
};

// What a type id is, for queries and the dump.
struct TypeInfo {
    spv::Op op = spv::OpNop;  // OpTypeInt, OpTypeFloat, OpTypeVector, OpTypePointer, ...
    uint32_t width = 0;        // int/float width, vector component count, array length id
    Id elem = 0;               // vector/array component type, pointer pointee
    spv::StorageClass storage = spv::StorageClassMax; // pointers
};

class Module {
public:
    explicit Module(uint32_t version = 0x00010500);

    uint32_t version() const { return version_; }
    Id new_id() { return bound_++; }
    uint32_t bound() const { return bound_; }

    // ---- module header sections ------------------------------------------
    void add_capability(spv::Capability c);
    bool has_capability(spv::Capability c) const;
    void add_extension(const std::string& name);
    Id glsl_std_450();   // OpExtInstImport "GLSL.std.450", created on first use
    void set_memory_model(spv::AddressingModel a, spv::MemoryModel m) { addressing_ = a; memory_model_ = m; }
    EntryPoint& add_entry_point(spv::ExecutionModel model, Id fn, std::string name, std::vector<Id> interface);
    void add_execution_mode(Id fn, spv::ExecutionMode mode, std::vector<uint32_t> literals);
    void set_name(Id target, std::string name);
    void decorate(Id target, spv::Decoration d, std::vector<uint32_t> literals = {});
    void member_decorate(Id struct_type, uint32_t member, spv::Decoration d, std::vector<uint32_t> literals = {});

    // ---- types (deduplicated, except structs and runtime arrays) -----------
    Id t_void();
    Id t_bool();
    Id t_int(uint32_t width);   // signedness 0; signed ops choose the opcode
    Id t_float(uint32_t width);
    Id t_vector(Id elem, uint32_t count);
    Id t_array(Id elem, uint32_t length);
    Id t_struct(const std::vector<Id>& members); // always a fresh id (structs get decorated)
    Id t_pointer(spv::StorageClass sc, Id pointee);
    Id t_function(Id ret, const std::vector<Id>& params);
    const TypeInfo* type_info(Id type) const; // nullptr when `type` is not a type

    // ---- constants (deduplicated) ----------------------------------------
    Id c_int(uint32_t width, uint64_t bits);
    Id c_u32(uint32_t v) { return c_int(32, v); }
    Id c_f32(float v);
    Id c_f64(double v);
    Id c_bool(bool v);
    Id c_null(Id type);
    Id c_composite(Id type, const std::vector<Id>& parts);
    Id spec_constant_u32(uint32_t default_value, uint32_t spec_id); // fresh, decorated SpecId
    Id spec_composite(Id type, const std::vector<Id>& parts);     // fresh
    Id type_of_value(Id value) const; // type id of a constant or global variable, 0 if unknown

    // ---- global variables ------------------------------------------------
    Id variable(Id pointer_type, spv::StorageClass sc);

    // Module-wide cache for ids that must exist once per module (builtin
    // variables, the workgroup-size spec constants); 0 until set.
    Id& cached(const std::string& key) { return cache_[key]; }

    Function& add_function(std::string name, Id return_type, Id fn_type);

    // ---- sections (read by write/verify/dump) ------------------------------
    std::vector<spv::Capability> capabilities;
    std::vector<std::string> extensions;
    std::vector<std::pair<Id, std::string>> ext_imports;
    spv::AddressingModel addressing() const { return addressing_; }
    spv::MemoryModel memory_model() const { return memory_model_; }
    std::vector<EntryPoint> entry_points;
    std::vector<Inst> execution_modes;
    std::vector<std::pair<Id, std::string>> names;
    std::vector<Inst> annotations;
    std::vector<Inst> globals; // types, constants and global variables, in definition order
    std::vector<Function> functions;

private:
    Id intern(Inst inst, const TypeInfo* info); // dedup by (op, type, operands)

    uint32_t version_;
    Id bound_ = 1;
    spv::AddressingModel addressing_ = spv::AddressingModelLogical;
    spv::MemoryModel memory_model_ = spv::MemoryModelGLSL450;
    std::map<std::vector<uint32_t>, Id> interned_;
    std::unordered_map<Id, TypeInfo> types_;
    std::unordered_map<Id, Id> value_types_;
    std::unordered_map<std::string, Id> cache_;
};

// ---- binary writer ------------------------------------------------------------
// Encodes the module in the logical layout order of SPIR-V 2.4.
std::vector<uint32_t> write(const Module& m);

// ---- verifier ---------------------------------------------------------------
struct Diagnostic {
    std::string function; // empty for module-level findings
    Id block = 0;
    size_t index = 0;     // instruction index within the block
    std::string message;
    std::string inst_text;
};

std::vector<Diagnostic> verify(const Module& m);
std::string format_diagnostics(const std::vector<Diagnostic>& diags);

// ---- text ---------------------------------------------------------------------
const char* op_name(spv::Op op);
std::string to_string(const Inst& inst);
std::string dump(const Module& m);

bool is_terminator(spv::Op op);

} // namespace brass::spirv
