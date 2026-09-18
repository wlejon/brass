#include "test_framework.hpp"
#include <brass/codegen/grammar_builder.hpp>
#include <brass/codegen/kernel_jit.hpp>
#include <brass/target/target.hpp>

#include <vector>
#include <string>
#include <cstring>
#include <cmath>
#include <limits>
#include <iostream>

using namespace brass;
using namespace brass::codegen;

namespace {

// Helper to construct a simple 2-state identifier DFA:
// State 0: accepts [a-z], transitions to state 1; else -1
// State 1: accepts [a-z0-9], transitions to state 1; else -1
std::vector<int32_t> make_identifier_dfa_table() {
    std::vector<int32_t> table(2 * 256, -1);
    for (int b = 0; b < 256; ++b) {
        if ((b >= 'a' && b <= 'z') || (b >= 'A' && b <= 'Z')) {
            table[0 * 256 + b] = 1;
        }
    }
    for (int b = 0; b < 256; ++b) {
        if ((b >= 'a' && b <= 'z') || (b >= 'A' && b <= 'Z') || (b >= '0' && b <= '9')) {
            table[1 * 256 + b] = 1;
        }
    }
    return table;
}

// Helper to construct a DFA recognizing the keyword "true":
// State 0: 't' -> 1
// State 1: 'r' -> 2
// State 2: 'u' -> 3
// State 3: 'e' -> 4
// State 4: any -> -1
std::vector<int32_t> make_keyword_true_dfa_table() {
    std::vector<int32_t> table(5 * 256, -1);
    table[0 * 256 + static_cast<uint8_t>('t')] = 1;
    table[1 * 256 + static_cast<uint8_t>('r')] = 2;
    table[2 * 256 + static_cast<uint8_t>('u')] = 3;
    table[3 * 256 + static_cast<uint8_t>('e')] = 4;
    return table;
}

} // namespace

TEST_CASE("GrammarBuilder - Vectorized Logit Masking (1024 Vocabulary)") {
    if (!Target::host().is_x64() && !Target::host().is_aarch64()) {
        std::cout << "  [SKIP] non-supported host arch\n";
        return;
    }

    Module mod("logit_mask_1024_mod");
    GrammarBuilder gb(mod);
    Function* fn = gb.build_logit_mask_function("mask_1024");
    REQUIRE(fn != nullptr);

    KernelOptions opts;
    opts.enable_avx2 = true;
    KernelJit jit(opts);
    KernelFunction kfn = jit.compile(*fn);
    REQUIRE(kfn.is_valid());

    auto mask_fn = kfn.as<LogitMaskFn>();
    REQUIRE(mask_fn != nullptr);

    constexpr uint64_t kVocabSize = 1024;
    constexpr size_t kNumWords = kVocabSize / 64; // 16 words

    std::vector<float> original_logits(kVocabSize);
    std::vector<float> logits(kVocabSize);
    for (size_t i = 0; i < kVocabSize; ++i) {
        float val = -5.0f + static_cast<float>(i) * 0.01f;
        original_logits[i] = val;
        logits[i] = val;
    }

    std::vector<uint64_t> valid_mask(kNumWords, 0);
    // Word 0: all valid
    valid_mask[0] = 0xFFFFFFFFFFFFFFFFULL;
    // Word 1: all invalid
    valid_mask[1] = 0x0000000000000000ULL;
    // Word 2: alternating bits
    valid_mask[2] = 0xAAAAAAAAAAAAAAAAULL; // odd bits valid
    // Word 3: only bit 10
    valid_mask[3] = (1ULL << 10);
    // Other words: mix of patterns
    for (size_t w = 4; w < kNumWords; ++w) {
        valid_mask[w] = (w % 2 == 0) ? 0x00FF00FF00FF00FFULL : 0x123456789ABCDEF0ULL;
    }

    const float kMaskVal = -std::numeric_limits<float>::infinity();
    mask_fn(logits.data(), valid_mask.data(), kVocabSize, kMaskVal);

    for (size_t i = 0; i < kVocabSize; ++i) {
        bool is_valid = (valid_mask[i / 64] >> (i % 64)) & 1ULL;
        if (is_valid) {
            CHECK_EQ(logits[i], original_logits[i]);
        } else {
            CHECK(std::isinf(logits[i]));
            CHECK(logits[i] < 0.0f);
        }
    }
}

TEST_CASE("GrammarBuilder - Vectorized Logit Masking (32,768 Vocab & Tail 32,771)") {
    if (!Target::host().is_x64() && !Target::host().is_aarch64()) {
        std::cout << "  [SKIP] non-supported host arch\n";
        return;
    }

    Module mod("logit_mask_large_mod");
    GrammarBuilder gb(mod);
    Function* fn = gb.build_logit_mask_function("mask_large");
    REQUIRE(fn != nullptr);

    KernelOptions opts;
    opts.enable_avx2 = true;
    KernelJit jit(opts);
    KernelFunction kfn = jit.compile(*fn);
    REQUIRE(kfn.is_valid());

    auto mask_fn = kfn.as<LogitMaskFn>();
    REQUIRE(mask_fn != nullptr);

    // Test 1: Exactly 32,768 floats (512 words, multiple of 8)
    {
        constexpr uint64_t kVocabSize = 32768;
        constexpr size_t kNumWords = kVocabSize / 64;

        std::vector<float> original_logits(kVocabSize);
        std::vector<float> logits(kVocabSize);
        for (size_t i = 0; i < kVocabSize; ++i) {
            float val = static_cast<float>(i) * 0.001f;
            original_logits[i] = val;
            logits[i] = val;
        }

        std::vector<uint64_t> valid_mask(kNumWords);
        for (size_t w = 0; w < kNumWords; ++w) {
            // Mix of all-valid, all-invalid, and sparse patterns
            if (w % 3 == 0) valid_mask[w] = 0xFFFFFFFFFFFFFFFFULL;
            else if (w % 3 == 1) valid_mask[w] = 0x0000000000000000ULL;
            else valid_mask[w] = (1ULL << (w % 64)) | (1ULL << ((w + 17) % 64));
        }

        const float kMaskVal = -std::numeric_limits<float>::infinity();
        mask_fn(logits.data(), valid_mask.data(), kVocabSize, kMaskVal);

        for (size_t i = 0; i < kVocabSize; ++i) {
            bool is_valid = (valid_mask[i / 64] >> (i % 64)) & 1ULL;
            if (is_valid) {
                CHECK_EQ(logits[i], original_logits[i]);
            } else {
                CHECK(std::isinf(logits[i]));
                CHECK(logits[i] < 0.0f);
            }
        }
    }

    // Test 2: 32,771 floats (with scalar tail: 32,771 % 8 = 3)
    {
        constexpr uint64_t kVocabSize = 32771;
        constexpr size_t kNumWords = (kVocabSize + 63) / 64;

        std::vector<float> original_logits(kVocabSize);
        std::vector<float> logits(kVocabSize);
        for (size_t i = 0; i < kVocabSize; ++i) {
            float val = static_cast<float>(i) * 0.002f;
            original_logits[i] = val;
            logits[i] = val;
        }

        std::vector<uint64_t> valid_mask(kNumWords, 0);
        for (size_t w = 0; w < kNumWords; ++w) {
            valid_mask[w] = 0x5555555555555555ULL;
        }
        // Tail tokens at 32768, 32769, 32770:
        // Set bit for 32768 (valid), leave 32769 (invalid), set 32770 (valid)
        size_t tail_word = 32768 / 64;
        valid_mask[tail_word] = (1ULL << (32768 % 64)) | (1ULL << (32770 % 64));

        const float kMaskVal = -std::numeric_limits<float>::infinity();
        mask_fn(logits.data(), valid_mask.data(), kVocabSize, kMaskVal);

        for (size_t i = 0; i < kVocabSize; ++i) {
            bool is_valid = (valid_mask[i / 64] >> (i % 64)) & 1ULL;
            if (is_valid) {
                CHECK_EQ(logits[i], original_logits[i]);
            } else {
                CHECK(std::isinf(logits[i]));
                CHECK(logits[i] < 0.0f);
            }
        }
    }
}

TEST_CASE("GrammarBuilder - Fast DFA Table Step and Specialized Jump Kernels") {
    if (!Target::host().is_x64() && !Target::host().is_aarch64()) {
        std::cout << "  [SKIP] non-supported host arch\n";
        return;
    }

    auto dfa_table = make_identifier_dfa_table();

    // 1. Test DfaTableStepFn
    {
        Module mod("dfa_table_step_mod");
        GrammarBuilder gb(mod);
        Function* fn = gb.build_dfa_table_step_function("dfa_table_step");
        REQUIRE(fn != nullptr);

        KernelJit jit;
        KernelFunction kfn = jit.compile(*fn);
        REQUIRE(kfn.is_valid());

        auto step_fn = kfn.as<DfaTableStepFn>();
        REQUIRE(step_fn != nullptr);

        int32_t num_states = 2;
        // From state 0:
        CHECK_EQ(step_fn(dfa_table.data(), num_states, 0, 'a'), 1);
        CHECK_EQ(step_fn(dfa_table.data(), num_states, 0, 'z'), 1);
        CHECK_EQ(step_fn(dfa_table.data(), num_states, 0, '0'), -1);
        CHECK_EQ(step_fn(dfa_table.data(), num_states, 0, '_'), -1);

        // From state 1:
        CHECK_EQ(step_fn(dfa_table.data(), num_states, 1, 'a'), 1);
        CHECK_EQ(step_fn(dfa_table.data(), num_states, 1, '9'), 1);
        CHECK_EQ(step_fn(dfa_table.data(), num_states, 1, ' '), -1);

        // Out of bounds states:
        CHECK_EQ(step_fn(dfa_table.data(), num_states, -1, 'a'), -1);
        CHECK_EQ(step_fn(dfa_table.data(), num_states, 2, 'a'), -1);
    }

    // 2. Test DfaStepFn with embedded table
    {
        Module mod("dfa_step_embedded_mod");
        GrammarBuilder gb(mod);
        Function* fn = gb.build_dfa_step_function(dfa_table.data(), 2, "dfa_step");
        REQUIRE(fn != nullptr);

        KernelJit jit;
        KernelFunction kfn = jit.compile(*fn);
        REQUIRE(kfn.is_valid());

        auto step_fn = kfn.as<DfaStepFn>();
        REQUIRE(step_fn != nullptr);

        CHECK_EQ(step_fn(0, 'm'), 1);
        CHECK_EQ(step_fn(0, '1'), -1);
        CHECK_EQ(step_fn(1, '9'), 1);
        CHECK_EQ(step_fn(1, '!'), -1);
        CHECK_EQ(step_fn(-1, 'a'), -1);
    }
}

TEST_CASE("GrammarBuilder - DFA Unrolled String Scanner") {
    if (!Target::host().is_x64() && !Target::host().is_aarch64()) {
        std::cout << "  [SKIP] non-supported host arch\n";
        return;
    }

    auto id_table = make_identifier_dfa_table();
    auto kw_table = make_keyword_true_dfa_table();

    Module mod("dfa_scan_string_mod");
    GrammarBuilder gb(mod);
    Function* fn = gb.build_dfa_scan_string_function("scan_string", 4);
    REQUIRE(fn != nullptr);

    KernelJit jit;
    KernelFunction kfn = jit.compile(*fn);
    REQUIRE(kfn.is_valid());

    auto scan_fn = kfn.as<DfaScanStringFn>();
    REQUIRE(scan_fn != nullptr);

    // Test 1: Identifier DFA
    auto scan_id = [&](int32_t start_state, const std::string& str) -> int32_t {
        return scan_fn(id_table.data(), start_state, reinterpret_cast<const uint8_t*>(str.data()), str.size());
    };

    CHECK_EQ(scan_id(0, "var123"), 1);
    CHECK_EQ(scan_id(0, "x"), 1);
    CHECK_EQ(scan_id(0, "aVeryLongIdentifierWithMultipleChars999"), 1);
    CHECK_EQ(scan_id(0, "123var"), -1); // starts with digit
    CHECK_EQ(scan_id(0, "var!able"), -1); // contains '!'
    CHECK_EQ(scan_id(0, ""), 0); // empty string maintains start state
    CHECK_EQ(scan_id(-1, "var"), -1); // starting from dead state

    // Test 2: Keyword "true" DFA
    auto scan_kw = [&](int32_t start_state, const std::string& str) -> int32_t {
        return scan_fn(kw_table.data(), start_state, reinterpret_cast<const uint8_t*>(str.data()), str.size());
    };

    CHECK_EQ(scan_kw(0, "true"), 4);   // reaches final accepting state 4
    CHECK_EQ(scan_kw(0, "tr"), 2);     // intermediate state 2
    CHECK_EQ(scan_kw(0, "truth"), -1); // dead at 't' after 'u'
    CHECK_EQ(scan_kw(0, "false"), -1); // dead at first char
    CHECK_EQ(scan_kw(0, ""), 0);       // empty string stays 0
}

TEST_CASE("GrammarBuilder - Batched Token Accept Filter") {
    if (!Target::host().is_x64() && !Target::host().is_aarch64()) {
        std::cout << "  [SKIP] non-supported host arch\n";
        return;
    }

    auto id_table = make_identifier_dfa_table();

    Module mod("dfa_filter_tokens_mod");
    GrammarBuilder gb(mod);
    Function* fn = gb.build_dfa_filter_tokens_function("filter_tokens");
    REQUIRE(fn != nullptr);

    KernelJit jit;
    KernelFunction kfn = jit.compile(*fn);
    REQUIRE(kfn.is_valid());

    auto filter_fn = kfn.as<DfaFilterTokensFn>();
    REQUIRE(filter_fn != nullptr);

    // Mock tokens:
    std::vector<std::string> tokens = {
        "alpha",       // 0: valid (-> 1)
        "123",         // 1: invalid (-> -1)
        "beta99",      // 2: valid (-> 1)
        "",            // 3: empty (valid, stays 0)
        "gamma_delta", // 4: invalid (contains '_')
        "x",           // 5: valid (-> 1)
        "7up",         // 6: invalid (starts with '7')
        "omega"        // 7: valid (-> 1)
    };

    // Pack into contiguous byte buffer and offsets
    std::string token_bytes;
    std::vector<uint32_t> token_offsets;
    token_offsets.push_back(0);
    for (const auto& tok : tokens) {
        token_bytes += tok;
        token_offsets.push_back(static_cast<uint32_t>(token_bytes.size()));
    }

    uint64_t num_tokens = tokens.size();
    uint64_t valid_mask = 0;

    filter_fn(
        id_table.data(),
        0, // current_state = 0 (start)
        token_offsets.data(),
        reinterpret_cast<const uint8_t*>(token_bytes.data()),
        num_tokens,
        &valid_mask
    );

    // Expected valid tokens: 0 ("alpha"), 2 ("beta99"), 3 (""), 5 ("x"), 7 ("omega")
    // Expected invalid tokens: 1 ("123"), 4 ("gamma_delta"), 6 ("7up")
    uint64_t expected_mask = (1ULL << 0) | (1ULL << 2) | (1ULL << 3) | (1ULL << 5) | (1ULL << 7);
    CHECK_EQ(valid_mask, expected_mask);

    // Filter with dead start state (-1) -> must reject all tokens
    uint64_t dead_mask = 0xFFFFFFFFFFFFFFFFULL;
    filter_fn(
        id_table.data(),
        -1, // dead state
        token_offsets.data(),
        reinterpret_cast<const uint8_t*>(token_bytes.data()),
        num_tokens,
        &dead_mask
    );
    CHECK_EQ(dead_mask, 0ULL);
}

TEST_CASE("GrammarBuilder - End-to-End Pipeline: Token Filter to Logit Masking") {
    if (!Target::host().is_x64() && !Target::host().is_aarch64()) {
        std::cout << "  [SKIP] non-supported host arch\n";
        return;
    }

    auto id_table = make_identifier_dfa_table();

    Module mod("grammar_pipeline_mod");
    GrammarBuilder gb(mod);
    Function* fn_filter = gb.build_dfa_filter_tokens_function("pipeline_filter");
    Function* fn_mask = gb.build_logit_mask_function("pipeline_mask");
    REQUIRE(fn_filter != nullptr);
    REQUIRE(fn_mask != nullptr);

    KernelOptions opts;
    opts.enable_avx2 = true;
    KernelJit jit(opts);
    KernelFunction kfn_filter = jit.compile(*fn_filter);
    KernelFunction kfn_mask = jit.compile(*fn_mask);
    REQUIRE(kfn_filter.is_valid());
    REQUIRE(kfn_mask.is_valid());

    auto filter_fn = kfn_filter.as<DfaFilterTokensFn>();
    auto mask_fn = kfn_mask.as<LogitMaskFn>();

    // 16 mock tokens
    std::vector<std::string> tokens = {
        "cat", "0dog", "fish", "bird12", "!fox", "wolf", "99elk", "bear",
        "deer", "hawk", "-owl", "lynx", "hare", "3duck", "frog", "toad"
    };
    uint64_t num_tokens = tokens.size();

    std::string token_bytes;
    std::vector<uint32_t> token_offsets;
    token_offsets.push_back(0);
    for (const auto& t : tokens) {
        token_bytes += t;
        token_offsets.push_back(static_cast<uint32_t>(token_bytes.size()));
    }

    std::vector<uint64_t> valid_mask((num_tokens + 63) / 64, 0);
    std::vector<float> logits(num_tokens);
    std::vector<float> original_logits(num_tokens);
    for (size_t i = 0; i < num_tokens; ++i) {
        logits[i] = 10.0f + static_cast<float>(i);
        original_logits[i] = logits[i];
    }

    // Step 1: Filter tokens against DFA
    filter_fn(
        id_table.data(),
        0,
        token_offsets.data(),
        reinterpret_cast<const uint8_t*>(token_bytes.data()),
        num_tokens,
        valid_mask.data()
    );

    // Step 2: Apply logit mask
    const float kMaskVal = -std::numeric_limits<float>::infinity();
    mask_fn(logits.data(), valid_mask.data(), num_tokens, kMaskVal);

    // Step 3: Validate
    for (size_t i = 0; i < num_tokens; ++i) {
        bool is_valid = (valid_mask[i / 64] >> (i % 64)) & 1ULL;
        if (is_valid) {
            CHECK_EQ(logits[i], original_logits[i]);
        } else {
            CHECK(std::isinf(logits[i]));
            CHECK(logits[i] < 0.0f);
        }
    }
}
