#include "test_framework.hpp"
#include <brass/pgo/profile_data.hpp>
#include <sstream>
#include <vector>

using namespace brass::pgo;

TEST_CASE("PGO Profile Serialization - Basic Roundtrip") {
    ProfileData prof_out;
    prof_out.set_module_name("test_module_math");
    prof_out.set_module_hash(0x123456789ABCDEF0ULL);

    FunctionProfile fn1;
    fn1.name = "compute_sum";
    fn1.entry_count = 1000;
    fn1.edge_counters = {1000, 850, 150, 850};
    fn1.add_indirect_target("worker_a", 700);
    fn1.add_indirect_target("worker_b", 150);
    prof_out.add_function(std::move(fn1));

    FunctionProfile fn2;
    fn2.name = "cold_handler";
    fn2.entry_count = 0;
    fn2.edge_counters = {0, 0};
    prof_out.add_function(std::move(fn2));

    std::stringstream ss;
    std::string err;
    bool write_ok = prof_out.write_to_stream(ss, &err);
    REQUIRE(write_ok);
    REQUIRE(err.empty());

    auto prof_in = ProfileData::read_from_stream(ss, &err);
    REQUIRE(prof_in != nullptr);
    REQUIRE(err.empty());

    CHECK_EQ(prof_in->module_name(), "test_module_math");
    CHECK_EQ(prof_in->module_hash(), 0x123456789ABCDEF0ULL);
    CHECK_EQ(prof_in->function_count(), 2ULL);

    const FunctionProfile* read_fn1 = prof_in->find_function("compute_sum");
    REQUIRE(read_fn1 != nullptr);
    CHECK_EQ(read_fn1->name, "compute_sum");
    CHECK_EQ(read_fn1->entry_count, 1000ULL);
    CHECK_EQ(read_fn1->edge_counter_count(), 4U);
    CHECK_EQ(read_fn1->edge_counters[0], 1000ULL);
    CHECK_EQ(read_fn1->edge_counters[1], 850ULL);
    CHECK_EQ(read_fn1->edge_counters[2], 150ULL);
    CHECK_EQ(read_fn1->edge_counters[3], 850ULL);
    CHECK_EQ(read_fn1->indirect_targets.size(), 2ULL);
    CHECK_EQ(read_fn1->get_indirect_target_count("worker_a"), 700ULL);
    CHECK_EQ(read_fn1->get_indirect_target_count("worker_b"), 150ULL);
    CHECK_EQ(read_fn1->get_indirect_target_count("non_existent"), 0ULL);

    const FunctionProfile* read_fn2 = prof_in->find_function("cold_handler");
    REQUIRE(read_fn2 != nullptr);
    CHECK_EQ(read_fn2->entry_count, 0ULL);
    CHECK_EQ(read_fn2->edge_counter_count(), 2U);
    CHECK_EQ(read_fn2->edge_counters[0], 0ULL);
    CHECK_EQ(read_fn2->edge_counters[1], 0ULL);
    CHECK_EQ(read_fn2->indirect_targets.size(), 0ULL);

    CHECK(prof_in->find_function("does_not_exist") == nullptr);
}

TEST_CASE("PGO Profile Serialization - File Roundtrip") {
    ProfileData prof_out;
    prof_out.set_module_name("file_test_mod");
    prof_out.set_module_hash(0xFEDCBA9876543210ULL);

    FunctionProfile fn;
    fn.name = "kernel_loop";
    fn.entry_count = 50000;
    fn.edge_counters = {50000, 49000, 1000, 2000000};
    prof_out.add_function(std::move(fn));

    std::string test_path = "test_profile_temp.bprof";
    std::string err;
    bool write_ok = prof_out.write_to_file(test_path, &err);
    REQUIRE(write_ok);

    auto prof_in = ProfileData::read_from_file(test_path, &err);
    REQUIRE(prof_in != nullptr);
    CHECK_EQ(prof_in->module_name(), "file_test_mod");
    CHECK_EQ(prof_in->module_hash(), 0xFEDCBA9876543210ULL);

    const FunctionProfile* read_fn = prof_in->find_function("kernel_loop");
    REQUIRE(read_fn != nullptr);
    CHECK_EQ(read_fn->entry_count, 50000ULL);
    CHECK_EQ(read_fn->edge_counter_count(), 4U);
    CHECK_EQ(read_fn->edge_counters[3], 2000000ULL);

    std::remove(test_path.c_str());
}

TEST_CASE("PGO Profile Serialization - Invalid Magic and Version Error Handling") {
    // 1. Corrupt magic
    {
        ProfileFileHeader bad_hdr;
        bad_hdr.magic = 0x11223344;
        bad_hdr.version = kProfileVersion;
        bad_hdr.module_hash = 0;
        bad_hdr.module_name_length = 0;
        bad_hdr.function_count = 0;

        std::stringstream ss;
        ss.write(reinterpret_cast<const char*>(&bad_hdr), sizeof(bad_hdr));

        std::string err;
        auto prof = ProfileData::read_from_stream(ss, &err);
        CHECK(prof == nullptr);
        CHECK(err.find("magic") != std::string::npos);
    }

    // 2. Unsupported version
    {
        ProfileFileHeader bad_hdr;
        bad_hdr.magic = kProfileMagic;
        bad_hdr.version = 999;
        bad_hdr.module_hash = 0;
        bad_hdr.module_name_length = 0;
        bad_hdr.function_count = 0;

        std::stringstream ss;
        ss.write(reinterpret_cast<const char*>(&bad_hdr), sizeof(bad_hdr));

        std::string err;
        auto prof = ProfileData::read_from_stream(ss, &err);
        CHECK(prof == nullptr);
        CHECK(err.find("version") != std::string::npos);
    }

    // 3. Truncated header
    {
        std::stringstream ss;
        char buf[10] = {0};
        ss.write(buf, 10);

        std::string err;
        auto prof = ProfileData::read_from_stream(ss, &err);
        CHECK(prof == nullptr);
        CHECK(err.find("EOF") != std::string::npos);
    }

    // 4. Non-existent file
    {
        std::string err;
        auto prof = ProfileData::read_from_file("non_existent_file_path_12345.bprof", &err);
        CHECK(prof == nullptr);
        CHECK(!err.empty());
    }
}
