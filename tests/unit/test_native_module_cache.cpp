// Native modules through the artifact cache
// (include/brass/cache/native_module_cache.hpp): built once, loaded from the
// store afterwards without compiling, valid after eviction, rebuilt from a
// damaged entry, one build for concurrent callers.
#include "test_framework.hpp"

#include <brass/cache/native_module_cache.hpp>
#include <brass/mir/builder.hpp>
#include <brass/mir/module.hpp>
#include <brass/mir/verifier.hpp>
#include <brass/object/object_writer.hpp>

#include <atomic>
#include <chrono>
#include <stdexcept>
#include <thread>
#include <vector>

using namespace brass;
using namespace brass::cache;

namespace {

void require_native() {
#if !BRASS_HOST_BACKEND
    throw brass::test::TestSkipped{"no code generator for this host"};
#endif
}

std::filesystem::path fresh_dir(const std::string& name) {
    const std::filesystem::path dir = brass::test::scratch_dir() / "native_module_cache" / name;
    std::error_code ec;
    std::filesystem::remove_all(dir, ec);
    return dir;
}

CacheKey client_key(const std::string& what) { return KeyBuilder("test.native/1").add("what", what).finish(); }

// mix(a, b) = a * k + b
bool build_mix(Module& mod, int64_t k, std::string& error) {
    Builder b(mod);
    Function* fn = mod.create_function("mix", Type::i64(), {Type::i64(), Type::i64()});
    b.set_function(fn);
    BasicBlock* entry = b.append_block("entry");
    Value* x = b.add_block_param(entry, Type::i64());
    Value* y = b.add_block_param(entry, Type::i64());
    b.build_ret(b.build_add(b.build_mul(x, b.build_iconst_i64(k)), y));
    fn->rebuild_cfg_predecessors();
    if (!verify_function(*fn)) {
        error = "the test module does not verify";
        return false;
    }
    return true;
}

using MixFn = int64_t (*)(int64_t, int64_t);

} // namespace

TEST_CASE("native_module_cache: built once, then loaded from the store without compiling") {
    require_native();
    ArtifactCache store(fresh_dir("once"), 0);
    const CacheKey key = client_key("mix31");
    int builds = 0;
    auto build = [&](Module& mod, std::string& err) {
        ++builds;
        return build_mix(mod, 31, err);
    };
    std::string err;
    {
        NativeModuleCache natives(store);
        std::shared_ptr<const NativeModule> m = natives.get_or_build(key, build, {}, &err);
        REQUIRE(m != nullptr);
        CHECK_FALSE(m->from_cache());
        CHECK_EQ(builds, 1);
        auto mix = m->function<MixFn>("mix");
        REQUIRE(mix != nullptr);
        CHECK_EQ(mix(2, 5), 67);
        // The same instance hands out the loaded module again.
        CHECK(natives.get_or_build(key, build) == m);
        CHECK_EQ(natives.stats().reuses, 1u);
        CHECK_EQ(natives.stats().builds, 1u);
    }
    // Another instance (as another process would): a hit, nothing built.
    NativeModuleCache again(store);
    std::shared_ptr<const NativeModule> m = again.get_or_build(key, build, {}, &err);
    REQUIRE(m != nullptr);
    CHECK(m->from_cache());
    CHECK_EQ(builds, 1);
    CHECK_EQ(again.stats().hits, 1u);
    CHECK_EQ(m->function<MixFn>("mix")(-3, 4), -89);
    CHECK(again.find(key) == m);
    // The private copy it loaded from is already gone from the load directory
    // (POSIX: unlinked once loaded; Windows: kept until the module goes).
#if !defined(_WIN32)
    size_t left = 0;
    for (const auto& e : std::filesystem::directory_iterator(again.load_dir())) {
        (void)e;
        ++left;
    }
    CHECK_EQ(left, 0u);
#endif
}

TEST_CASE("native_module_cache: a loaded module outlives its evicted entry") {
    require_native();
    const std::filesystem::path dir = fresh_dir("evicted");
    ArtifactCache store(dir, 0);
    const CacheKey key = client_key("mix7");
    int builds = 0;
    auto build = [&](Module& mod, std::string& err) {
        ++builds;
        return build_mix(mod, 7, err);
    };
    NativeModuleCache natives(store);
    std::shared_ptr<const NativeModule> m = natives.get_or_build(key, build);
    REQUIRE(m != nullptr);
    REQUIRE(store.remove(m->storage_key()));
    ArtifactCache other(dir, 1);   // and a trim from another instance
    other.trim();
    CHECK_EQ(m->function<MixFn>("mix")(3, 1), 22);
    CHECK(natives.find(key) == m);   // still loaded here
    NativeModuleCache fresh(store);
    CHECK(fresh.find(key) == nullptr);
    std::shared_ptr<const NativeModule> rebuilt = fresh.get_or_build(key, build);
    REQUIRE(rebuilt != nullptr);
    CHECK_FALSE(rebuilt->from_cache());
    CHECK_EQ(builds, 2);
    CHECK_EQ(rebuilt->function<MixFn>("mix")(3, 1), 22);
    CHECK_EQ(m->function<MixFn>("mix")(1, 1), 8);
}

TEST_CASE("native_module_cache: an image that will not load is rebuilt and replaced") {
    require_native();
    ArtifactCache store(fresh_dir("damaged"), 0);
    const CacheKey key = client_key("mix5");
    int builds = 0;
    auto build = [&](Module& mod, std::string& err) {
        ++builds;
        return build_mix(mod, 5, err);
    };
    NativeModuleCache natives(store);
    const CacheKey skey = natives.storage_key(key, {});
    REQUIRE(store.put(skey, {BlobRef("image", std::string(4096, '\x7f'))}));   // a valid entry, not an image
    std::shared_ptr<const NativeModule> m = natives.get_or_build(key, build);
    REQUIRE(m != nullptr);
    CHECK_EQ(natives.stats().load_failures, 1u);
    CHECK_EQ(builds, 1);
    CHECK_EQ(m->function<MixFn>("mix")(2, 2), 12);
    NativeModuleCache next(store);
    std::shared_ptr<const NativeModule> hit = next.get_or_build(key, build);
    REQUIRE(hit != nullptr);
    CHECK(hit->from_cache());
    CHECK_EQ(builds, 1);
}

TEST_CASE("native_module_cache: keys, link options and failures") {
    require_native();
    ArtifactCache store(fresh_dir("keys"), 0);
    NativeModuleCache natives(store);
    std::string err;
    auto m3 = natives.get_or_build(client_key("k3"), [](Module& mod, std::string& e) { return build_mix(mod, 3, e); });
    auto m4 = natives.get_or_build(client_key("k4"), [](Module& mod, std::string& e) { return build_mix(mod, 4, e); });
    REQUIRE(m3 != nullptr);
    REQUIRE(m4 != nullptr);
    CHECK_EQ(m3->function<MixFn>("mix")(10, 0), 30);
    CHECK_EQ(m4->function<MixFn>("mix")(10, 0), 40);

    // Link options are in the storage key.
    NativeOptions only_mix;
    only_mix.exports = {"mix"};
    CHECK(natives.storage_key(client_key("k3"), {}) != natives.storage_key(client_key("k3"), only_mix));
    CHECK(natives.find(client_key("k3"), only_mix) == nullptr);
    auto exported = natives.get_or_build(client_key("k3"), [](Module& mod, std::string& e) { return build_mix(mod, 3, e); },
                                         only_mix, &err);
    REQUIRE(exported != nullptr);
    CHECK_EQ(exported->function<MixFn>("mix")(1, 1), 4);

    // A failed build: an error, nothing stored, and the next call tries again.
    auto failed = natives.get_or_build(client_key("bad"), [](Module&, std::string& e) {
        e = "no such kernel";
        return false;
    }, {}, &err);
    CHECK(failed == nullptr);
    CHECK_EQ(err, std::string("no such kernel"));
    CHECK(natives.find(client_key("bad")) == nullptr);
    CHECK_EQ(natives.stats().build_failures, 1u);
    auto thrown = natives.get_or_build(client_key("bad"), [](Module&, std::string&) -> bool {
        throw std::runtime_error("boom");
    }, {}, &err);
    CHECK(thrown == nullptr);
    CHECK(err.find("boom") != std::string::npos);

    // An invalid client key is never built or stored.
    int builds = 0;
    CHECK(natives.get_or_build(CacheKey{}, [&](Module& mod, std::string& e) {
        ++builds;
        return build_mix(mod, 1, e);
    }, {}, &err) == nullptr);
    CHECK_EQ(builds, 0);
}

TEST_CASE("native_module_cache: an object file the client compiled") {
    require_native();
    ArtifactCache store(fresh_dir("object"), 0);
    const CacheKey key = client_key("object9");
    int builds = 0;
    auto build = [&](object::ObjectFile& obj, std::string& err) {
        ++builds;
        Module mod("object9");
        if (!build_mix(mod, 9, err)) return false;
        obj = object::compile_module_to_object(mod, Target::host());
        return true;
    };
    {
        NativeModuleCache natives(store);
        auto m = natives.get_or_build_object(key, build);
        REQUIRE(m != nullptr);
        CHECK_EQ(m->function<MixFn>("mix")(2, 3), 21);
    }
    NativeModuleCache natives(store);
    auto m = natives.get_or_build_object(key, build);
    REQUIRE(m != nullptr);
    CHECK(m->from_cache());
    CHECK_EQ(builds, 1);
    CHECK_EQ(m->function<MixFn>("mix")(2, 3), 21);
}

TEST_CASE("native_module_cache: concurrent callers of one key share one build") {
    require_native();
    ArtifactCache store(fresh_dir("concurrent"), 0);
    NativeModuleCache natives(store);
    const CacheKey key = client_key("shared");
    std::atomic<int> builds{0};
    std::vector<std::shared_ptr<const NativeModule>> got(8);
    std::vector<std::thread> threads;
    for (size_t t = 0; t < got.size(); ++t) {
        threads.emplace_back([&, t] {
            got[t] = natives.get_or_build(key, [&](Module& mod, std::string& e) {
                ++builds;
                std::this_thread::sleep_for(std::chrono::milliseconds(50));
                return build_mix(mod, 11, e);
            });
        });
    }
    for (auto& th : threads) th.join();
    CHECK_EQ(builds.load(), 1);
    for (const auto& m : got) {
        REQUIRE(m != nullptr);
        CHECK(m == got[0]);
    }
    CHECK_EQ(got[0]->function<MixFn>("mix")(1, 2), 13);
}
