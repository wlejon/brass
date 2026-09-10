#include "test_framework.hpp"
#include <brass/debug/source_map.hpp>

TEST_CASE("SourceMap - Base64 VLQ encoder and decoder") {
    std::vector<int32_t> test_values = {
        0, 1, -1, 2, -2, 15, -15, 16, -16, 31, -31, 32, -32,
        63, -63, 64, -64, 127, -127, 255, -255, 1000, -1000,
        123456, -123456, 10000000, -10000000
    };

    for (int32_t val : test_values) {
        std::string encoded = brass::vlq::encode(val);
        CHECK(!encoded.empty());

        std::string_view view(encoded);
        auto it = view.begin();
        auto end = view.end();
        int32_t decoded = 0;
        bool ok = brass::vlq::decode(it, end, decoded);
        CHECK(ok);
        CHECK_EQ(decoded, val);
        CHECK(it == end);
    }
}

TEST_CASE("SourceMap - Base64 VLQ error handling") {
    // Empty
    std::string empty;
    std::string_view v1(empty);
    auto it1 = v1.begin();
    int32_t out1 = 0;
    CHECK(!brass::vlq::decode(it1, v1.end(), out1));

    // Invalid character
    std::string invalid = "?";
    std::string_view v2(invalid);
    auto it2 = v2.begin();
    int32_t out2 = 0;
    CHECK(!brass::vlq::decode(it2, v2.end(), out2));

    // Truncated multi-byte VLQ (continuation bit set without following byte)
    // 'g' has bit 5 set (value 32)
    std::string truncated = "g";
    std::string_view v3(truncated);
    auto it3 = v3.begin();
    int32_t out3 = 0;
    CHECK(!brass::vlq::decode(it3, v3.end(), out3));
}

TEST_CASE("SourceMap - Basic generation and resolution") {
    brass::SourceMap sm("output.js");
    sm.add_source("src/a.js");
    sm.add_source("src/b.js");

    // Mapping at code offset 0: src/a.js line 1, col 0
    sm.add_mapping(0, brass::DebugLoc(1, 1, 1));
    // Mapping at code offset 16: src/a.js line 5, col 10
    sm.add_mapping(16, brass::DebugLoc(1, 5, 10));
    // Mapping at code offset 32: src/b.js line 2, col 4
    sm.add_mapping(32, brass::DebugLoc(2, 2, 4));

    CHECK_EQ(sm.mappings().size(), 3u);

    // Offset queries
    // At offset 0
    brass::DebugLoc r0 = sm.resolve_offset(0);
    CHECK(r0.is_valid());
    CHECK_EQ(r0.file_id, 1u);
    CHECK_EQ(r0.line, 1u);

    // At offset 10 (falls back to offset 0)
    brass::DebugLoc r10 = sm.resolve_offset(10);
    CHECK_EQ(r10.file_id, 1u);
    CHECK_EQ(r10.line, 1u);

    // At offset 16
    brass::DebugLoc r16 = sm.resolve_offset(16);
    CHECK_EQ(r16.file_id, 1u);
    CHECK_EQ(r16.line, 5u);
    CHECK_EQ(r16.column, 10u);

    // At offset 20
    brass::DebugLoc r20 = sm.resolve_offset(20);
    CHECK_EQ(r20.file_id, 1u);
    CHECK_EQ(r20.line, 5u);

    // At offset 50 (beyond last mapping)
    brass::DebugLoc r50 = sm.resolve_offset(50);
    CHECK_EQ(r50.file_id, 2u);
    CHECK_EQ(r50.line, 2u);
    CHECK_EQ(r50.column, 4u);
}

TEST_CASE("SourceMap - JSON serialization and parse_json roundtrip") {
    brass::SourceMap original("bundle.js");
    original.add_source("module1.js");
    original.add_source("module2.js");
    original.add_name("calculateSum");

    original.add_mapping(0, brass::DebugLoc(1, 10, 5), "calculateSum");
    original.add_mapping(20, brass::DebugLoc(1, 15, 8));
    original.add_mapping(45, brass::DebugLoc(2, 3, 1));
    original.add_mapping(80, brass::DebugLoc(2, 7, 12));

    std::string json = original.to_json();
    CHECK(!json.empty());
    CHECK(json.find("\"version\": 3") != std::string::npos);
    CHECK(json.find("\"file\": \"bundle.js\"") != std::string::npos);
    CHECK(json.find("module1.js") != std::string::npos);
    CHECK(json.find("module2.js") != std::string::npos);
    CHECK(json.find("calculateSum") != std::string::npos);

    // Parse JSON back
    std::string err;
    auto parsed = brass::SourceMap::parse_json(json, &err);
    REQUIRE(parsed != nullptr);
    CHECK(err.empty());
    CHECK_EQ(parsed->file(), "bundle.js");
    CHECK_EQ(parsed->sources().size(), 2u);
    CHECK_EQ(parsed->sources()[0], "module1.js");
    CHECK_EQ(parsed->sources()[1], "module2.js");
    CHECK_EQ(parsed->names().size(), 1u);
    CHECK_EQ(parsed->names()[0], "calculateSum");

    REQUIRE_EQ(parsed->mappings().size(), 4u);
    CHECK_EQ(parsed->mappings()[0].code_offset, 0u);
    CHECK_EQ(parsed->mappings()[0].loc.file_id, 1u);
    CHECK_EQ(parsed->mappings()[0].loc.line, 10u);
    CHECK_EQ(parsed->mappings()[0].loc.column, 5u);
    CHECK_EQ(parsed->mappings()[0].symbol, "calculateSum");

    CHECK_EQ(parsed->mappings()[1].code_offset, 20u);
    CHECK_EQ(parsed->mappings()[1].loc.file_id, 1u);
    CHECK_EQ(parsed->mappings()[1].loc.line, 15u);
    CHECK_EQ(parsed->mappings()[1].loc.column, 8u);

    CHECK_EQ(parsed->mappings()[2].code_offset, 45u);
    CHECK_EQ(parsed->mappings()[2].loc.file_id, 2u);
    CHECK_EQ(parsed->mappings()[2].loc.line, 3u);
    CHECK_EQ(parsed->mappings()[2].loc.column, 1u);

    CHECK_EQ(parsed->mappings()[3].code_offset, 80u);
    CHECK_EQ(parsed->mappings()[3].loc.file_id, 2u);
    CHECK_EQ(parsed->mappings()[3].loc.line, 7u);
    CHECK_EQ(parsed->mappings()[3].loc.column, 12u);

    // Both original and parsed resolve identically
    for (uint32_t off : {0u, 10u, 20u, 30u, 45u, 60u, 80u, 100u}) {
        CHECK_EQ(original.resolve_offset(off), parsed->resolve_offset(off));
    }
}
