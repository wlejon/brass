// Unit tests for spirv::parse_binary and spirv::audit_compute_shader:
//   - Binary header validation & malformed stream rejection
//   - Round-trip write(m) -> parse_binary -> verify -> write(m2)
//   - Static compute shader prover:
//       1. SSA dominance verification (including OpPhi predecessor dominance)
//       2. Uniform control-flow barrier & subgroup collective legality (BVSolver)
//       3. Workgroup `shared` memory array bounds verification (BVSolver)
//       4. Division / reciprocal / sqrt hazard audit
//   - Full corpus verification of hand-written GLSL .comp shaders in bro and brotensor

#include "spirv_test_support.hpp"

#include <brass/target/spirv/spirv_binary_reader.hpp>
#include <brass/target/spirv/spirv_isel.hpp>
#include <spirv/unified1/GLSL.std.450.h>

#include <filesystem>
#include <fstream>
#include <regex>
#include <sstream>

using namespace brass::spirv;
using namespace spvtest;

namespace {

struct ComputeBuilder {
    Module m;
    Id void_t = 0;
    Id bool_t = 0;
    Id u32_t = 0;
    Id f32_t = 0;
    Id uvec3_t = 0;
    Id lid_var = 0;

    explicit ComputeBuilder(uint32_t lx = 64, uint32_t ly = 1, uint32_t lz = 1) {
        void_t = m.t_void();
        bool_t = m.t_bool();
        u32_t = m.t_int(32);
        f32_t = m.t_float(32);
        uvec3_t = m.t_vector(u32_t, 3);

        Id ptr_in_uvec3 = m.t_pointer(spv::StorageClassInput, uvec3_t);
        lid_var = m.variable(ptr_in_uvec3, spv::StorageClassInput);
        m.decorate(lid_var, spv::DecorationBuiltIn, {static_cast<uint32_t>(spv::BuiltInLocalInvocationId)});

        Function& fn = m.add_function("main", void_t, m.t_function(void_t, {}));
        m.add_entry_point(spv::ExecutionModelGLCompute, fn.id, "main", {lid_var});
        m.add_execution_mode(fn.id, spv::ExecutionModeLocalSize, {lx, ly, lz});
    }

    Function& f() { return m.functions[0]; }

    Id load_lid_x(Block& b) {
        Id ptr_in_u32 = m.t_pointer(spv::StorageClassInput, u32_t);
        Id ac = m.new_id();
        b.insts.push_back(Inst(spv::OpAccessChain, ptr_in_u32, ac).id(lid_var).id(m.c_u32(0)));
        Id lid_x = m.new_id();
        b.insts.push_back(Inst(spv::OpLoad, u32_t, lid_x).id(ptr_in_u32 == 0 ? 0 : ac));
        return lid_x;
    }

    void emit_workgroup_barrier(Block& b) {
        Id wg_scope = m.c_u32(static_cast<uint32_t>(spv::ScopeWorkgroup));
        Id sem = m.c_u32(static_cast<uint32_t>(spv::MemorySemanticsAcquireReleaseMask |
                                               spv::MemorySemanticsWorkgroupMemoryMask));
        b.insts.push_back(Inst(spv::OpControlBarrier).id(wg_scope).id(wg_scope).id(sem));
    }
};

bool has_hazard(const ComputeShaderAuditReport& rep, HazardKind kind) {
    for (const ShaderHazard& h : rep.hazards) {
        if (h.kind == kind) return true;
    }
    return false;
}

std::vector<uint32_t> load_spv_file(const std::filesystem::path& p) {
    std::ifstream in(p, std::ios::binary | std::ios::ate);
    if (!in) return {};
    auto sz = static_cast<size_t>(in.tellg());
    if (sz < 20 || (sz % 4) != 0) return {};
    in.seekg(0);
    std::vector<uint32_t> words(sz / 4);
    in.read(reinterpret_cast<char*>(words.data()), static_cast<std::streamsize>(sz));
    return words;
}

std::vector<uint32_t> load_inc_words(const std::filesystem::path& p) {
    std::ifstream in(p);
    if (!in) return {};
    std::ostringstream ss;
    ss << in.rdbuf();
    std::string txt = ss.str();
    std::vector<uint32_t> words;
    words.reserve(4096);
    const char* cur = txt.data();
    const char* end = cur + txt.size();
    while (cur < end) {
        while (cur < end && !((*cur >= '0' && *cur <= '9'))) ++cur;
        if (cur >= end) break;
        char* next = nullptr;
        unsigned long v = std::strtoul(cur, &next, 0);
        if (next == cur) {
            ++cur;
            continue;
        }
        words.push_back(static_cast<uint32_t>(v));
        cur = next;
    }
    if (words.size() >= 5 && words[0] != spv::MagicNumber) {
        // Find magic number if preceded by header guard or array size numbers.
        for (size_t i = 0; i < words.size(); ++i) {
            if (words[i] == spv::MagicNumber) {
                words.erase(words.begin(), words.begin() + static_cast<std::ptrdiff_t>(i));
                break;
            }
        }
    }
    return words;
}

TEST_CASE("SPIR-V binary reader - rejects malformed headers and truncated instructions") {
    Module out;
    std::string err;

    CHECK(!parse_binary({}, out, &err));
    CHECK(has(err, "too short"));

    std::vector<uint32_t> bad_magic = {0x12345678u, 0x00010500u, 0u, 10u, 0u};
    CHECK(!parse_binary(bad_magic, out, &err));
    CHECK(has(err, "magic"));

    std::vector<uint32_t> bad_ver = {spv::MagicNumber, 0x00020100u, 0u, 10u, 0u};
    CHECK(!parse_binary(bad_ver, out, &err));
    CHECK(has(err, "version"));

    std::vector<uint32_t> zero_bound = {spv::MagicNumber, 0x00010500u, 0u, 0u, 0u};
    CHECK(!parse_binary(zero_bound, out, &err));
    CHECK(has(err, "bound"));

    std::vector<uint32_t> bad_schema = {spv::MagicNumber, 0x00010500u, 0u, 10u, 1u};
    CHECK(!parse_binary(bad_schema, out, &err));
    CHECK(has(err, "schema"));

    std::vector<uint32_t> zero_wc = {spv::MagicNumber, 0x00010500u, 0u, 10u, 0u,
                                     static_cast<uint32_t>(spv::OpNop)};
    CHECK(!parse_binary(zero_wc, out, &err));
    CHECK(has(err, "word count 0"));

    std::vector<uint32_t> trunc_inst = {spv::MagicNumber, 0x00010500u, 0u, 10u, 0u,
                                        (5u << 16) | static_cast<uint32_t>(spv::OpCapability),
                                        static_cast<uint32_t>(spv::CapabilityShader)};
    CHECK(!parse_binary(trunc_inst, out, &err));
    CHECK(has(err, "truncated"));
}

TEST_CASE("SPIR-V binary reader - round-trips hand-built and SpirvTarget modules") {
    ComputeBuilder cb(64, 1, 1);
    Module& m = cb.m;
    Id arr_t = m.t_array(cb.f32_t, 64);
    Id ptr_wg_arr = m.t_pointer(spv::StorageClassWorkgroup, arr_t);
    Id ptr_wg_f32 = m.t_pointer(spv::StorageClassWorkgroup, cb.f32_t);
    Id shm = m.variable(ptr_wg_arr, spv::StorageClassWorkgroup);
    m.entry_points[0].interface_ids.push_back(shm);

    Block entry{m.new_id(), {}};
    Id lid_x = cb.load_lid_x(entry);
    Id elem_ptr = m.new_id();
    entry.insts.push_back(Inst(spv::OpAccessChain, ptr_wg_f32, elem_ptr).id(shm).id(lid_x));
    entry.insts.push_back(Inst(spv::OpStore).id(elem_ptr).id(m.c_f32(2.5f)));
    cb.emit_workgroup_barrier(entry);
    entry.insts.push_back(Inst(spv::OpReturn));
    cb.f().blocks.push_back(std::move(entry));

    std::vector<uint32_t> bin1 = write(m);
    Module lifted;
    std::string err;
    REQUIRE(parse_binary(bin1, lifted, &err));
    CHECK(verify(lifted).empty());
    std::vector<uint32_t> bin2 = write(lifted);
    CHECK(bin1 == bin2);

    ComputeShaderAuditReport rep = audit_compute_shader(lifted);
    CHECK(rep.ok());
    CHECK_EQ(rep.local_size[0], 64u);
    CHECK_EQ(rep.num_workgroup_arrays, 1u);
    REQUIRE_EQ(rep.shared_accesses.size(), 1u);
    CHECK(rep.shared_accesses[0].proven_in_bounds);
    REQUIRE_EQ(rep.barriers.size(), 1u);
    CHECK(rep.barriers[0].uniform_control_flow);
}

TEST_CASE("SPIR-V prover - detects SSA dominance violations") {
    // Diamond CFG where branch `a` defines %val and merge block `c` uses %val directly without OpPhi.
    ComputeBuilder cb(32, 1, 1);
    Module& m = cb.m;
    Id b_entry = m.new_id(), b_a = m.new_id(), b_b = m.new_id(), b_c = m.new_id();
    Id bad_val = m.new_id();

    Block entry{b_entry, {}};
    Id lid_x = cb.load_lid_x(entry);
    Id cond = m.new_id();
    entry.insts.push_back(Inst(spv::OpULessThan, cb.bool_t, cond).id(lid_x).id(m.c_u32(16)));
    entry.insts.push_back(Inst(spv::OpSelectionMerge).id(b_c).lit(0));
    entry.insts.push_back(Inst(spv::OpBranchConditional).id(cond).id(b_a).id(b_b));

    Block blk_a{b_a, {
        Inst(spv::OpIAdd, cb.u32_t, bad_val).id(lid_x).id(m.c_u32(1)),
        Inst(spv::OpBranch).id(b_c)
    }};
    Block blk_b{b_b, {
        Inst(spv::OpBranch).id(b_c)
    }};
    Block blk_c{b_c, {
        Inst(spv::OpIAdd, cb.u32_t, m.new_id()).id(bad_val).id(m.c_u32(2)),
        Inst(spv::OpReturn)
    }};
    cb.f().blocks = {entry, blk_a, blk_b, blk_c};

    ComputeShaderAuditReport rep = audit_compute_shader_binary(write(m));
    CHECK(rep.parse_ok);
    CHECK(!rep.dominance_ok);
    CHECK(!rep.ok());
    CHECK(has_hazard(rep, HazardKind::StructuralOrDominance));
}

TEST_CASE("SPIR-V prover - barrier and subgroup control-flow uniformity") {
    // 1. NEGATIVE: OpControlBarrier inside `if (lid.x < 16)` when LocalSize.x == 64
    {
        ComputeBuilder cb(64, 1, 1);
        Module& m = cb.m;
        Id b_entry = m.new_id(), b_then = m.new_id(), b_merge = m.new_id();

        Block entry{b_entry, {}};
        Id lid_x = cb.load_lid_x(entry);
        Id cond = m.new_id();
        entry.insts.push_back(Inst(spv::OpULessThan, cb.bool_t, cond).id(lid_x).id(m.c_u32(16)));
        entry.insts.push_back(Inst(spv::OpSelectionMerge).id(b_merge).lit(0));
        entry.insts.push_back(Inst(spv::OpBranchConditional).id(cond).id(b_then).id(b_merge));

        Block blk_then{b_then, {}};
        cb.emit_workgroup_barrier(blk_then);
        blk_then.insts.push_back(Inst(spv::OpBranch).id(b_merge));

        Block blk_merge{b_merge, {Inst(spv::OpReturn)}};
        cb.f().blocks = {entry, blk_then, blk_merge};

        ComputeShaderAuditReport rep = audit_compute_shader_binary(write(m));
        CHECK(rep.parse_ok);
        CHECK(!rep.barriers_uniform);
        CHECK(!rep.ok());
        CHECK(has_hazard(rep, HazardKind::DivergentBarrier));
    }

    // 2. NEGATIVE: OpControlBarrier after divergent early return `if (lid.x == 0) return;`
    {
        ComputeBuilder cb(64, 1, 1);
        Module& m = cb.m;
        Id b_entry = m.new_id(), b_ret = m.new_id(), b_cont = m.new_id();

        Block entry{b_entry, {}};
        Id lid_x = cb.load_lid_x(entry);
        Id cond = m.new_id();
        entry.insts.push_back(Inst(spv::OpIEqual, cb.bool_t, cond).id(lid_x).id(m.c_u32(0)));
        entry.insts.push_back(Inst(spv::OpSelectionMerge).id(b_cont).lit(0));
        entry.insts.push_back(Inst(spv::OpBranchConditional).id(cond).id(b_ret).id(b_cont));

        Block blk_ret{b_ret, {Inst(spv::OpReturn)}};
        Block blk_cont{b_cont, {}};
        cb.emit_workgroup_barrier(blk_cont);
        blk_cont.insts.push_back(Inst(spv::OpReturn));
        cb.f().blocks = {entry, blk_ret, blk_cont};

        ComputeShaderAuditReport rep = audit_compute_shader_binary(write(m));
        CHECK(rep.parse_ok);
        CHECK(!rep.barriers_uniform);
        CHECK(has_hazard(rep, HazardKind::DivergentBarrier));
    }

    // 3. POSITIVE: OpControlBarrier inside tautological guard `if (lid.x < 64)` when LocalSize.x == 64
    //    (proven invariant across all threads by BVSolver!)
    {
        ComputeBuilder cb(64, 1, 1);
        Module& m = cb.m;
        Id b_entry = m.new_id(), b_then = m.new_id(), b_merge = m.new_id();

        Block entry{b_entry, {}};
        Id lid_x = cb.load_lid_x(entry);
        Id cond = m.new_id();
        entry.insts.push_back(Inst(spv::OpULessThan, cb.bool_t, cond).id(lid_x).id(m.c_u32(64)));
        entry.insts.push_back(Inst(spv::OpSelectionMerge).id(b_merge).lit(0));
        entry.insts.push_back(Inst(spv::OpBranchConditional).id(cond).id(b_then).id(b_merge));

        Block blk_then{b_then, {}};
        cb.emit_workgroup_barrier(blk_then);
        blk_then.insts.push_back(Inst(spv::OpBranch).id(b_merge));

        Block blk_merge{b_merge, {Inst(spv::OpReturn)}};
        cb.f().blocks = {entry, blk_then, blk_merge};

        ComputeShaderAuditReport rep = audit_compute_shader_binary(write(m));
        CHECK(rep.ok());
        CHECK(rep.barriers_uniform);
    }

    // 4. NEGATIVE vs POSITIVE: Subgroup collective under lane-divergent branch vs subgroupAll() branch
    {
        ComputeBuilder cb(64, 1, 1);
        Module& m = cb.m;
        m.add_capability(spv::CapabilityGroupNonUniformVote);
        m.add_capability(spv::CapabilityGroupNonUniformArithmetic);
        Id sg_scope = m.c_u32(static_cast<uint32_t>(spv::ScopeSubgroup));
        Id b_entry = m.new_id(), b_then = m.new_id(), b_merge = m.new_id();

        Block entry{b_entry, {}};
        Id lid_x = cb.load_lid_x(entry);
        Id lane_cond = m.new_id();
        entry.insts.push_back(Inst(spv::OpULessThan, cb.bool_t, lane_cond).id(lid_x).id(m.c_u32(3)));
        entry.insts.push_back(Inst(spv::OpSelectionMerge).id(b_merge).lit(0));
        entry.insts.push_back(Inst(spv::OpBranchConditional).id(lane_cond).id(b_then).id(b_merge));

        Block blk_then{b_then, {
            Inst(spv::OpGroupNonUniformIAdd, cb.u32_t, m.new_id())
                .id(sg_scope).lit(static_cast<uint32_t>(spv::GroupOperationReduce)).id(lid_x),
            Inst(spv::OpBranch).id(b_merge)
        }};
        Block blk_merge{b_merge, {Inst(spv::OpReturn)}};
        cb.f().blocks = {entry, blk_then, blk_merge};

        ComputeShaderAuditReport bad_rep = audit_compute_shader_binary(write(m));
        CHECK(!bad_rep.barriers_uniform);
        CHECK(has_hazard(bad_rep, HazardKind::DivergentSubgroupOp));

        // Now wrap `lane_cond` in OpGroupNonUniformAll (subgroupAll), making the branch subgroup-uniform!
        Id sg_all = m.new_id();
        cb.f().blocks[0].insts.insert(cb.f().blocks[0].insts.end() - 2,
                                      Inst(spv::OpGroupNonUniformAll, cb.bool_t, sg_all).id(sg_scope).id(lane_cond));
        cb.f().blocks[0].insts.back().operands[0] = Operand::ref(sg_all);

        ComputeShaderAuditReport ok_rep = audit_compute_shader_binary(write(m));
        CHECK(ok_rep.ok());
        CHECK(ok_rep.barriers_uniform);
    }
}

TEST_CASE("SPIR-V prover - shared memory bounds verification via BVSolver") {
    // 1. NEGATIVE: shared float tile[64] indexed by lid.x + 1 when LocalSize.x == 64 (OOB at lid.x == 63)
    {
        ComputeBuilder cb(64, 1, 1);
        Module& m = cb.m;
        Id arr_t = m.t_array(cb.f32_t, 64);
        Id ptr_wg_arr = m.t_pointer(spv::StorageClassWorkgroup, arr_t);
        Id ptr_wg_f32 = m.t_pointer(spv::StorageClassWorkgroup, cb.f32_t);
        Id shm = m.variable(ptr_wg_arr, spv::StorageClassWorkgroup);
        m.entry_points[0].interface_ids.push_back(shm);

        Block entry{m.new_id(), {}};
        Id lid_x = cb.load_lid_x(entry);
        Id idx = m.new_id();
        entry.insts.push_back(Inst(spv::OpIAdd, cb.u32_t, idx).id(lid_x).id(m.c_u32(1)));
        Id ptr = m.new_id();
        entry.insts.push_back(Inst(spv::OpAccessChain, ptr_wg_f32, ptr).id(shm).id(idx));
        entry.insts.push_back(Inst(spv::OpStore).id(ptr).id(m.c_f32(1.0f)));
        entry.insts.push_back(Inst(spv::OpReturn));
        cb.f().blocks.push_back(std::move(entry));

        ComputeShaderAuditReport rep = audit_compute_shader_binary(write(m));
        CHECK(!rep.shared_bounds_ok);
        CHECK(!rep.ok());
        CHECK(has_hazard(rep, HazardKind::SharedMemoryOutOfBounds));
    }

    // 2. POSITIVE: shared float tile[128] with LocalSize.x == 256 guarded by `if (lid.x < 128)`
    {
        ComputeBuilder cb(256, 1, 1);
        Module& m = cb.m;
        Id arr_t = m.t_array(cb.f32_t, 128);
        Id ptr_wg_arr = m.t_pointer(spv::StorageClassWorkgroup, arr_t);
        Id ptr_wg_f32 = m.t_pointer(spv::StorageClassWorkgroup, cb.f32_t);
        Id shm = m.variable(ptr_wg_arr, spv::StorageClassWorkgroup);
        m.entry_points[0].interface_ids.push_back(shm);

        Id b_entry = m.new_id(), b_then = m.new_id(), b_merge = m.new_id();
        Block entry{b_entry, {}};
        Id lid_x = cb.load_lid_x(entry);
        Id cond = m.new_id();
        entry.insts.push_back(Inst(spv::OpULessThan, cb.bool_t, cond).id(lid_x).id(m.c_u32(128)));
        entry.insts.push_back(Inst(spv::OpSelectionMerge).id(b_merge).lit(0));
        entry.insts.push_back(Inst(spv::OpBranchConditional).id(cond).id(b_then).id(b_merge));

        Id ptr = m.new_id();
        Block blk_then{b_then, {
            Inst(spv::OpAccessChain, ptr_wg_f32, ptr).id(shm).id(lid_x),
            Inst(spv::OpStore).id(ptr).id(m.c_f32(0.0f)),
            Inst(spv::OpBranch).id(b_merge)
        }};
        Block blk_merge{b_merge, {Inst(spv::OpReturn)}};
        cb.f().blocks = {entry, blk_then, blk_merge};

        ComputeShaderAuditReport rep = audit_compute_shader_binary(write(m));
        CHECK(rep.ok());
        CHECK(rep.shared_bounds_ok);
        REQUIRE_EQ(rep.shared_accesses.size(), 1u);
        CHECK(rep.shared_accesses[0].proven_in_bounds);
    }
}

TEST_CASE("SPIR-V prover - division and sqrt hazard detection") {
    // 1. NEGATIVE: division by constant zero
    {
        ComputeBuilder cb(64, 1, 1);
        Module& m = cb.m;
        Block entry{m.new_id(), {
            Inst(spv::OpFDiv, cb.f32_t, m.new_id()).id(m.c_f32(1.0f)).id(m.c_f32(0.0f)),
            Inst(spv::OpReturn)
        }};
        cb.f().blocks.push_back(std::move(entry));

        ComputeShaderAuditReport rep = audit_compute_shader_binary(write(m));
        CHECK(!rep.div_sqrt_ok);
        CHECK(!rep.ok());
        CHECK(has_hazard(rep, HazardKind::DefiniteDivisionByZero));
    }

    // 2. NEGATIVE: sqrt of negative constant
    {
        ComputeBuilder cb(64, 1, 1);
        Module& m = cb.m;
        Id glsl = m.glsl_std_450();
        Block entry{m.new_id(), {
            Inst(spv::OpExtInst, cb.f32_t, m.new_id()).id(glsl).lit(GLSLstd450Sqrt).id(m.c_f32(-4.0f)),
            Inst(spv::OpReturn)
        }};
        cb.f().blocks.push_back(std::move(entry));

        ComputeShaderAuditReport rep = audit_compute_shader_binary(write(m));
        CHECK(!rep.div_sqrt_ok);
        CHECK(has_hazard(rep, HazardKind::DefiniteNegativeSqrt));
    }

    // 3. POSITIVE: division guarded by FMax(x, 1e-5f) and OpSelect ternary
    {
        ComputeBuilder cb(64, 1, 1);
        Module& m = cb.m;
        Id glsl = m.glsl_std_450();
        Block entry{m.new_id(), {}};
        Id clamped = m.new_id();
        entry.insts.push_back(Inst(spv::OpExtInst, cb.f32_t, clamped)
                                  .id(glsl).lit(GLSLstd450FMax).id(m.c_f32(0.0f)).id(m.c_f32(1e-5f)));
        Id div1 = m.new_id();
        entry.insts.push_back(Inst(spv::OpFDiv, cb.f32_t, div1).id(m.c_f32(1.0f)).id(clamped));
        entry.insts.push_back(Inst(spv::OpReturn));
        cb.f().blocks.push_back(std::move(entry));

        ComputeShaderAuditReport rep = audit_compute_shader_binary(write(m));
        CHECK(rep.ok());
        CHECK(rep.div_sqrt_ok);
        REQUIRE_EQ(rep.divisions.size(), 1u);
        CHECK(rep.divisions[0].proven_safe);
    }
}

TEST_CASE("SPIR-V prover - audits compiled bro and brotensor GLSL .comp shader corpus") {
    namespace fs = std::filesystem;
    size_t audited_brotensor = 0;
    size_t audited_bro = 0;

    // 1. Audit precompiled brotensor compute shaders if present in the sibling build tree.
    fs::path bt_dir = "/home/j/projects/bro/build/brotensor/vulkan_shaders";
    if (fs::exists(bt_dir) && fs::is_directory(bt_dir)) {
        for (const auto& entry : fs::directory_iterator(bt_dir)) {
            if (entry.path().extension() != ".inc") continue;
            std::vector<uint32_t> words = load_inc_words(entry.path());
            REQUIRE(!words.empty());
            ComputeShaderAuditReport rep = audit_compute_shader_binary(words);
            if (!rep.ok()) {
                std::cerr << "brotensor shader failed audit: " << entry.path().filename() << "\n";
                for (const auto& h : rep.hazards) {
                    std::cerr << "  hazard: " << h.message << "\n";
                }
            }
            CHECK(rep.ok());
            ++audited_brotensor;
        }
    }

    // 2. Audit bro compute shaders if precompiled in /tmp/bro_spv_e or /tmp/bro_spv.
    for (const char* dir_path : {"/tmp/bro_spv_e", "/tmp/bro_spv"}) {
        fs::path d(dir_path);
        if (!fs::exists(d) || !fs::is_directory(d)) continue;
        for (const auto& entry : fs::directory_iterator(d)) {
            std::vector<uint32_t> words = (entry.path().extension() == ".spv")
                ? load_spv_file(entry.path())
                : load_inc_words(entry.path());
            if (words.empty()) continue;
            ComputeShaderAuditReport rep = audit_compute_shader_binary(words);
            if (!rep.ok()) {
                std::cerr << "bro shader failed audit: " << entry.path() << "\n";
                for (const auto& h : rep.hazards) {
                    std::cerr << "  hazard: " << h.message << "\n";
                }
            }
            CHECK(rep.ok());
            ++audited_bro;
        }
    }

    std::cout << "[SPIR-V Prover Corpus] Audited " << audited_brotensor
              << " brotensor shader binaries and " << audited_bro
              << " bro shader binaries.\n";
}

} // namespace

#ifdef BRASS_STANDALONE_TEST
BRASS_TEST_MAIN()
#endif
