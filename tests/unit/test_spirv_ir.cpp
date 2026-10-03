// spirv::Module, the binary writer, the dump and every rule of spirv::verify
// on hand-built modules; spirv-val accepts the valid ones and rejects an
// unstructured branch our verifier does not look for (which also proves the
// tool is really being run).

#include "spirv_test_support.hpp"

#include <cstring>


using namespace brass::spirv;
using namespace spvtest;

namespace {

// The smallest valid compute module: `void main() { return; }`, LocalSize
// 1x1x1. `body` may add blocks/instructions before the module is returned.
struct Mini {
    Module m;
    Id void_t = 0;
    size_t fn = 0;

    Mini() {
        void_t = m.t_void();
        m.add_function("main", void_t, m.t_function(void_t, {}));
        Block b;
        b.label = m.new_id();
        b.insts.push_back(Inst(spv::OpReturn));
        f().blocks.push_back(std::move(b));
        m.add_entry_point(spv::ExecutionModelGLCompute, f().id, "main", {});
        m.add_execution_mode(f().id, spv::ExecutionModeLocalSize, {1, 1, 1});
    }
    Function& f() { return m.functions[fn]; }
    Block& entry() { return f().blocks.front(); }
    // Inserts before the entry block's terminator.
    void add(Inst inst) { entry().insts.insert(entry().insts.end() - 1, std::move(inst)); }
};

bool has_diag(const std::vector<Diagnostic>& diags, const std::string& needle) {
    for (const Diagnostic& d : diags) {
        if (d.message.find(needle) != std::string::npos) return true;
    }
    std::cerr << "no diagnostic containing '" << needle << "' in:\n" << format_diagnostics(diags);
    return false;
}

TEST_CASE("SPIR-V IR - types and constants are deduplicated") {
    Module m;
    CHECK_EQ(m.t_int(32), m.t_int(32));
    CHECK_NE(m.t_int(32), m.t_int(64));
    CHECK_EQ(m.t_vector(m.t_float(32), 4), m.t_vector(m.t_float(32), 4));
    CHECK_EQ(m.t_pointer(spv::StorageClassWorkgroup, m.t_int(32)), m.t_pointer(spv::StorageClassWorkgroup, m.t_int(32)));
    CHECK_NE(m.t_pointer(spv::StorageClassWorkgroup, m.t_int(32)),
             m.t_pointer(spv::StorageClassPhysicalStorageBuffer, m.t_int(32)));
    CHECK_EQ(m.c_u32(7), m.c_u32(7));
    CHECK_NE(m.c_u32(7), m.c_int(64, 7));
    CHECK_EQ(m.c_f32(1.5f), m.c_f32(1.5f));
    CHECK_NE(m.c_f32(0.0f), m.c_f32(-0.0f)); // bit patterns, not values
    CHECK_NE(m.t_struct({m.t_int(32)}), m.t_struct({m.t_int(32)})); // structs are never shared
    CHECK_NE(m.spec_constant_u32(64, 0), m.spec_constant_u32(64, 1));
    // 64-bit types bring their capabilities.
    CHECK(m.has_capability(spv::CapabilityInt64));
    m.t_float(64);
    CHECK(m.has_capability(spv::CapabilityFloat64));
    const TypeInfo* ti = m.type_info(m.t_vector(m.t_float(32), 4));
    REQUIRE(ti != nullptr);
    CHECK(ti->op == spv::OpTypeVector);
    CHECK_EQ(ti->width, 4u);
    CHECK(m.type_info(m.c_u32(1)) == nullptr);
    CHECK_EQ(m.type_of_value(m.c_u32(1)), m.t_int(32));
}

TEST_CASE("SPIR-V IR - binary layout: header, word counts, strings, 64-bit literals") {
    Mini mini;
    Id c = mini.m.c_int(64, 0x1122334455667788ull);
    std::vector<uint32_t> w = write(mini.m);
    REQUIRE(w.size() > 5);
    CHECK_EQ(w[0], 0x07230203u);
    CHECK_EQ(w[1], 0x00010500u);
    CHECK_EQ(w[3], mini.m.bound());
    CHECK_EQ(w[4], 0u);
    // First instruction: OpCapability Shader, two words.
    CHECK_EQ(w[5], (2u << 16) | static_cast<uint32_t>(spv::OpCapability));
    CHECK_EQ(w[6], static_cast<uint32_t>(spv::CapabilityShader));
    // The entry point name "main" is one word plus a terminating zero word.
    bool found = false;
    for (size_t i = 5; i < w.size(); i += w[i] >> 16) {
        if ((w[i] & 0xFFFF) == spv::OpEntryPoint) {
            CHECK_EQ(w[i] >> 16, 5u); // opcode, model, function, "main", "\0"
            char name[5] = {};
            std::memcpy(name, &w[i + 3], 4);
            CHECK(std::string(name) == "main");
            CHECK_EQ(w[i + 4], 0u);
            found = true;
        }
        if ((w[i] & 0xFFFF) == spv::OpConstant && w[i + 2] == c) {
            CHECK_EQ(w[i + 3], 0x55667788u); // low word first
            CHECK_EQ(w[i + 4], 0x11223344u);
        }
        if ((w[i] >> 16) == 0) break;
    }
    CHECK(found);
    CHECK(verify(mini.m).empty());
    require_valid(w, dump(mini.m));
}

TEST_CASE("SPIR-V IR - dump names opcodes and enumerants") {
    Mini mini;
    std::string d = dump(mini.m);
    CHECK(has(d, "OpCapability Shader"));
    CHECK(has(d, "OpMemoryModel Logical GLSL450"));
    CHECK(has(d, "OpEntryPoint GLCompute"));
    CHECK(has(d, "OpReturn"));
    CHECK(has(d, "OpTypeVoid"));
    CHECK(to_string(Inst(spv::OpIAdd, 3, 9).id(4).id(5)) == "%9 = OpIAdd %3 %4 %5");
    CHECK(std::string(op_name(spv::OpGroupNonUniformShuffle)) == "OpGroupNonUniformShuffle");
}

TEST_CASE("SPIR-V IR - version below 1.5 is rejected") {
    bool threw = false;
    try {
        Module m(0x00010300);
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    CHECK(threw);
}

// ---------------------------------------------------------------------------
// Verifier rules, one failing module each
// ---------------------------------------------------------------------------

TEST_CASE("SPIR-V verify - undefined, duplicate and out-of-bound ids") {
    {
        Mini mini;
        mini.add(Inst(spv::OpIAdd, mini.m.t_int(32), mini.m.new_id()).id(999).id(mini.m.c_u32(1)));
        CHECK(has_diag(verify(mini.m), "is not defined"));
    }
    {
        Mini mini;
        Id one = mini.m.c_u32(1);
        mini.add(Inst(spv::OpIAdd, mini.m.t_int(32), one).id(one).id(one)); // redefines the constant's id
        CHECK(has_diag(verify(mini.m), "defined twice"));
    }
    {
        Mini mini;
        mini.add(Inst(spv::OpIAdd, mini.m.t_int(32), mini.m.bound() + 5).id(mini.m.c_u32(1)).id(mini.m.c_u32(1)));
        CHECK(has_diag(verify(mini.m), "outside the id bound"));
    }
    {
        Mini mini;
        mini.add(Inst(spv::OpIAdd, mini.m.c_u32(1), mini.m.new_id()).id(mini.m.c_u32(1)).id(mini.m.c_u32(1)));
        CHECK(has_diag(verify(mini.m), "is not a type"));
    }
}

TEST_CASE("SPIR-V verify - block structure") {
    {
        Mini mini;
        mini.entry().insts.clear();
        mini.entry().insts.push_back(Inst(spv::OpNop));
        CHECK(has_diag(verify(mini.m), "does not end in a terminator"));
    }
    {
        Mini mini;
        mini.entry().insts.insert(mini.entry().insts.begin(), Inst(spv::OpReturn));
        CHECK(has_diag(verify(mini.m), "terminator in the middle"));
    }
    {
        Mini mini;
        mini.entry().insts.clear();
        CHECK(has_diag(verify(mini.m), "block is empty"));
    }
    {
        Mini mini;
        mini.entry().insts.insert(mini.entry().insts.begin(), Inst(spv::OpBranch).id(mini.entry().label));
        mini.entry().insts.pop_back();
        CHECK(has_diag(verify(mini.m), "entry block is the target of a branch"));
    }
    {
        Mini mini;
        mini.entry().insts.back() = Inst(spv::OpBranch).id(mini.m.c_u32(0));
        CHECK(has_diag(verify(mini.m), "is not a block of the function"));
    }
}

// entry: br_if true, a, b; a: br c; b: br c; c: phi(...); ret
Mini diamond(bool with_merge) {
    Mini mini;
    Module& m = mini.m;
    Id a = m.new_id(), b = m.new_id(), c = m.new_id();
    Block& e = mini.entry();
    e.insts.clear();
    if (with_merge) e.insts.push_back(Inst(spv::OpSelectionMerge).id(c).lit(0));
    e.insts.push_back(Inst(spv::OpBranchConditional).id(m.c_bool(true)).id(a).id(b));
    Block ba{a, {Inst(spv::OpBranch).id(c)}};
    Block bb{b, {Inst(spv::OpBranch).id(c)}};
    Block bc{c, {Inst(spv::OpPhi, m.t_int(32), m.new_id()).id(m.c_u32(1)).id(a).id(m.c_u32(2)).id(b), Inst(spv::OpReturn)}};
    mini.f().blocks.push_back(ba);
    mini.f().blocks.push_back(bb);
    mini.f().blocks.push_back(bc);
    return mini;
}

TEST_CASE("SPIR-V verify - phis and merges") {
    {
        Mini mini = diamond(true);
        CHECK(verify(mini.m).empty());
        require_valid(write(mini.m), dump(mini.m));
    }
    {
        Mini mini = diamond(true);
        Inst& phi = mini.f().blocks[3].insts[0];
        phi.operands.resize(2); // only one of the two predecessors
        CHECK(has_diag(verify(mini.m), "OpPhi parents do not match"));
    }
    {
        Mini mini = diamond(true);
        auto& insts = mini.f().blocks[3].insts;
        insts.insert(insts.begin(), Inst(spv::OpNop));
        CHECK(has_diag(verify(mini.m), "OpPhi after a non-phi"));
    }
    {
        Mini mini = diamond(true);
        Block& e = mini.entry();
        e.insts.insert(e.insts.begin(), Inst(spv::OpSelectionMerge).id(mini.f().blocks[3].label).lit(0));
        CHECK(has_diag(verify(mini.m), "not immediately before the terminator"));
    }
    {
        // Both a and the entry name c as their merge.
        Mini mini = diamond(true);
        Block& a = mini.f().blocks[1];
        Id d = mini.m.new_id();
        a.insts.clear();
        a.insts.push_back(Inst(spv::OpSelectionMerge).id(mini.f().blocks[3].label).lit(0));
        a.insts.push_back(Inst(spv::OpBranchConditional).id(mini.m.c_bool(false)).id(mini.f().blocks[3].label).id(d));
        mini.f().blocks.push_back(Block{d, {Inst(spv::OpUnreachable)}});
        CHECK(has_diag(verify(mini.m), "already the merge block"));
    }
    {
        // Our verifier does not demand structured control flow; spirv-val
        // does, which shows the tool is really run.
        Mini mini = diamond(false);
        CHECK(verify(mini.m).empty());
        if (spirv_val_available()) CHECK(!spirv_val(write(mini.m)));
    }
}

TEST_CASE("SPIR-V verify - PhysicalStorageBuffer access needs Aligned") {
    Mini mini;
    Module& m = mini.m;
    m.add_capability(spv::CapabilityPhysicalStorageBufferAddresses);
    m.set_memory_model(spv::AddressingModelPhysicalStorageBuffer64, spv::MemoryModelGLSL450);
    Id f32 = m.t_float(32);
    Id ptr_t = m.t_pointer(spv::StorageClassPhysicalStorageBuffer, f32);
    Id p = m.new_id();
    mini.add(Inst(spv::OpConvertUToPtr, ptr_t, p).id(m.c_int(64, 256)));
    mini.add(Inst(spv::OpLoad, f32, m.new_id()).id(p));
    CHECK(has_diag(verify(m), "without an Aligned memory operand"));

    Mini ok;
    Module& m2 = ok.m;
    m2.add_capability(spv::CapabilityPhysicalStorageBufferAddresses);
    m2.set_memory_model(spv::AddressingModelPhysicalStorageBuffer64, spv::MemoryModelGLSL450);
    Id f = m2.t_float(32);
    Id q = m2.new_id();
    ok.add(Inst(spv::OpConvertUToPtr, m2.t_pointer(spv::StorageClassPhysicalStorageBuffer, f), q).id(m2.c_int(64, 256)));
    ok.add(Inst(spv::OpLoad, f, m2.new_id()).id(q).lit(spv::MemoryAccessAlignedMask).lit(4));
    CHECK(verify(m2).empty());
    require_valid(write(m2), dump(m2));
}

TEST_CASE("SPIR-V verify - entry point interface and capabilities") {
    {
        Mini mini;
        Module& m = mini.m;
        Id u32 = m.t_int(32);
        Id var = m.variable(m.t_pointer(spv::StorageClassWorkgroup, u32), spv::StorageClassWorkgroup);
        mini.add(Inst(spv::OpStore).id(var).id(m.c_u32(3)));
        CHECK(has_diag(verify(m), "not in the entry point interface"));
        m.entry_points[0].interface.push_back(var);
        CHECK(verify(m).empty());
        require_valid(write(m), dump(m));
    }
    {
        Mini mini;
        Module& m = mini.m;
        m.globals.push_back(Inst(spv::OpTypeFloat, 0, m.new_id()).lit(64)); // bypasses t_float
        CHECK(has_diag(verify(m), "without the Float64 capability"));
    }
    {
        Mini mini;
        mini.m.entry_points[0].function = 12345;
        CHECK(has_diag(verify(mini.m), "names a function that does not exist"));
    }
}

} // namespace
