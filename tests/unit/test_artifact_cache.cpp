// The artifact cache (include/brass/cache/artifact_cache.hpp,
// docs/artifact_cache.md): the hash, keys, the file format's checks,
// eviction, and several caches, threads and processes on one directory.
#include "test_framework.hpp"

#include <brass/cache/artifact_cache.hpp>
#include <brass/core/hash128.hpp>

#include <atomic>
#include <chrono>
#include <cstring>
#include <fstream>
#include <string>
#include <thread>
#include <vector>

#if !defined(_WIN32)
#include <sys/wait.h>
#include <unistd.h>
#endif

using namespace brass;
using namespace brass::cache;

namespace {

std::filesystem::path fresh_dir(const std::string& name) {
    const std::filesystem::path dir = brass::test::scratch_dir() / "artifact_cache" / name;
    std::error_code ec;
    std::filesystem::remove_all(dir, ec);
    return dir;
}

CacheKey key_of(const std::string& text) { return KeyBuilder("test/1").add("text", text).finish(); }

std::vector<uint8_t> file_bytes(const std::filesystem::path& p) {
    std::ifstream in(p, std::ios::binary);
    return std::vector<uint8_t>((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
}

void write_bytes(const std::filesystem::path& p, const std::vector<uint8_t>& bytes) {
    std::ofstream out(p, std::ios::binary | std::ios::trunc);
    out.write(reinterpret_cast<const char*>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
}

// What the directory's entries take on disk, as the cache counts it.
uint64_t entry_bytes(const std::filesystem::path& dir) {
    uint64_t total = 0;
    for (const auto& e : std::filesystem::directory_iterator(dir)) {
        if (e.is_regular_file() && e.path().extension() == ".bac") total += (e.file_size() + 4095) / 4096 * 4096;
    }
    return total;
}

} // namespace

TEST_CASE("artifact_cache: hash128 is MurmurHash3 x64/128, incremental equals one-shot") {
    const std::string fox = "The quick brown fox jumps over the lazy dog";
    const Digest128 d = hash128(fox);
    CHECK_EQ(d.lo, 0xe34bbc7bbc071b6cULL);
    CHECK_EQ(d.hi, 0x7a433ca9c49a9347ULL);
    CHECK(hash128("") == (Digest128{0, 0}));

    std::string text;
    for (int i = 0; i < 300; ++i) text.push_back(static_cast<char>(i * 7 + 3));
    const Digest128 whole = hash128(text, 42);
    for (size_t split : {size_t{0}, size_t{1}, size_t{7}, size_t{15}, size_t{16}, size_t{17}, size_t{150}, size_t{299}}) {
        Hasher128 h(42);
        h.update(std::string_view(text).substr(0, split));
        h.update(std::string_view(text).substr(split, 3));
        h.update(std::string_view(text).substr(std::min(text.size(), split + 3)));
        CHECK(h.finish() == whole);
    }
    Digest128 back;
    REQUIRE(Digest128::from_hex(whole.hex(), back));
    CHECK(back == whole);
    CHECK_FALSE(Digest128::from_hex("xyz", back));
}

TEST_CASE("artifact_cache: keys are sensitive to every field, its name, boundaries and order") {
    const CacheKey base = KeyBuilder("d/1").add("a", "ab").add("b", "c").finish();
    CHECK(base.valid);
    CHECK(base == KeyBuilder("d/1").add("a", "ab").add("b", "c").finish());
    CHECK(base.id != base.check);
    CHECK(base != KeyBuilder("d/1").add("a", "a").add("b", "bc").finish());   // boundaries
    CHECK(base != KeyBuilder("d/1").add("b", "c").add("a", "ab").finish());   // order
    CHECK(base != KeyBuilder("d/1").add("x", "ab").add("b", "c").finish());   // names
    CHECK(base != KeyBuilder("d/2").add("a", "ab").add("b", "c").finish());   // domain
    CHECK(KeyBuilder("d").add_u64("n", 1).finish() != KeyBuilder("d").add_u64("n", 2).finish());
    CHECK(KeyBuilder("d").add_bool("f", true).finish() != KeyBuilder("d").add_bool("f", false).finish());
    CHECK(KeyBuilder("d").add_key("k", base).finish() != KeyBuilder("d").add_key("k", key_of("x")).finish());

    // brass's own part: deterministic, changes the key, and tells targets apart.
    const CacheKey with = KeyBuilder("d/1").add_brass_build().finish();
    CHECK(with == KeyBuilder("d/1").add_brass_build().finish());
    CHECK(with.valid);
    CHECK_FALSE(brass_binary_identity().empty());
    CHECK(with != KeyBuilder("d/1").finish());
    CHECK(with != KeyBuilder("d/1").add_brass_codegen().finish());
    CHECK(KeyBuilder("d").add_brass_codegen(Target::x64_linux()).finish() !=
          KeyBuilder("d").add_brass_codegen(Target::aarch64_linux()).finish());
    // An invalid key makes whatever derives from it invalid.
    CHECK_FALSE(KeyBuilder("d").add_key("k", CacheKey{}).finish().valid);
}

TEST_CASE("artifact_cache: round trip of several blobs") {
    ArtifactCache cache(fresh_dir("roundtrip"), 0);
    const CacheKey key = key_of("roundtrip");
    std::vector<uint32_t> words = {0x07230203u, 1, 2, 3, 0xdeadbeefu};
    std::vector<uint8_t> binary(1000);
    for (size_t i = 0; i < binary.size(); ++i) binary[i] = static_cast<uint8_t>(i * 31);
    std::string err;
    REQUIRE(cache.put(key,
                      {BlobRef("il", std::string("some intermediate language")),
                       BlobRef::of<uint32_t>("spirv", std::span<const uint32_t>(words)), BlobRef("empty", ""),
                       BlobRef("binary", binary)},
                      &err));
    Miss why = Miss::Absent;
    const std::optional<Artifact> art = cache.get(key, &why);
    REQUIRE(art.has_value());
    CHECK(why == Miss::None);
    CHECK_EQ(art->size(), 4u);
    CHECK_EQ(art->text("il"), std::string_view("some intermediate language"));
    const std::span<const uint32_t> got = art->as<uint32_t>("spirv");
    REQUIRE_EQ(got.size(), words.size());
    CHECK(std::equal(got.begin(), got.end(), words.begin()));
    CHECK_EQ(reinterpret_cast<uintptr_t>(got.data()) % 16, 0u);
    CHECK(art->has("empty"));
    CHECK(art->bytes("empty").empty());
    const std::span<const uint8_t> b = art->bytes("binary");
    REQUIRE_EQ(b.size(), binary.size());
    CHECK(std::equal(b.begin(), b.end(), binary.begin()));
    CHECK_FALSE(art->has("absent"));
    CHECK_EQ(cache.stats().hits, 1u);
    CHECK_EQ(cache.stats().stores, 1u);

    // Replacing an entry, and a blob name given twice.
    REQUIRE(cache.put(key, {BlobRef("il", "second")}));
    CHECK_EQ(cache.get(key)->text("il"), std::string_view("second"));
    CHECK_FALSE(cache.put(key, {BlobRef("a", "1"), BlobRef("a", "2")}, &err));
    CHECK_FALSE(err.empty());
    CHECK_EQ(cache.get(key)->text("il"), std::string_view("second"));
}

TEST_CASE("artifact_cache: damaged, truncated, foreign and colliding files are misses") {
    const std::filesystem::path dir = fresh_dir("damaged");
    ArtifactCache cache(dir, 0);
    const CacheKey key = key_of("damaged");
    REQUIRE(cache.put(key, {BlobRef("data", std::string(5000, 'q'))}));
    const std::filesystem::path path = cache.path_for(key);
    const std::vector<uint8_t> good = file_bytes(path);
    REQUIRE(good.size() > 200);

    auto expect = [&](const std::vector<uint8_t>& bytes, Miss expected) {
        write_bytes(path, bytes);
        Miss why = Miss::None;
        CHECK_FALSE(cache.get(key, &why).has_value());
        CHECK_EQ(static_cast<int>(why), static_cast<int>(expected));
    };
    std::vector<uint8_t> b = good;
    b[3000] ^= 0x01;
    expect(b, Miss::Corrupt);                                  // a payload bit
    b = good;
    b[64] ^= 0x01;
    expect(b, Miss::Corrupt);                                  // the blob count
    b = good;
    b[100] ^= 0x01;
    expect(b, Miss::Corrupt);                                  // the blob table
    b = good;
    b.resize(good.size() - 16);
    expect(b, Miss::Truncated);
    b.resize(40);
    expect(b, Miss::Truncated);
    expect({}, Miss::Truncated);
    b = good;
    b.push_back(0);
    expect(b, Miss::Truncated);                                // longer than it says
    b = good;
    b[4] = 99;
    expect(b, Miss::Foreign);                                  // another format version
    b = good;
    b[0] = 'X';
    expect(b, Miss::Foreign);
    expect(std::vector<uint8_t>(200, 0x41), Miss::Foreign);

    // Another key whose id collides with this one's: the header's check
    // digest tells them apart.
    write_bytes(path, good);
    CacheKey collide = key;
    collide.check.lo ^= 1;
    Miss why = Miss::None;
    CHECK_FALSE(cache.get(collide, &why).has_value());
    CHECK(why == Miss::KeyMismatch);
    CHECK(cache.get(key).has_value());

    // A key that could not be made is never looked up or stored.
    CHECK_FALSE(cache.get(CacheKey{}, &why).has_value());
    CHECK(why == Miss::InvalidKey);
    CHECK_FALSE(cache.put(CacheKey{}, {BlobRef("a", "b")}));
}

TEST_CASE("artifact_cache: write_artifact_file and read_artifact_file at a chosen path") {
    const std::filesystem::path p = fresh_dir("single") / "sub" / "module.rec";
    const CacheKey key = key_of("record");
    std::string err;
    REQUIRE(write_artifact_file(p, key, std::vector<BlobRef>{BlobRef("entry", "main")}, &err));
    Miss why = Miss::None;
    std::optional<Artifact> a = read_artifact_file(p, key, &why);
    REQUIRE(a.has_value());
    CHECK_EQ(a->text("entry"), std::string_view("main"));
    CHECK_FALSE(read_artifact_file(p, key_of("other"), &why).has_value());
    CHECK(why == Miss::KeyMismatch);
    CHECK_FALSE(read_artifact_file(p.parent_path() / "none", key, &why).has_value());
    CHECK(why == Miss::Absent);
}

TEST_CASE("artifact_cache: eviction deletes the least recently used down to 7/8 of the budget") {
    const std::filesystem::path dir = fresh_dir("evict");
    const std::string blob(3000, 'e');   // with the header and table: one 4 KiB unit
    std::vector<CacheKey> keys;
    {
        ArtifactCache writer(dir, 0);
        for (int i = 0; i < 12; ++i) {
            keys.push_back(key_of("evict" + std::to_string(i)));
            REQUIRE(writer.put(keys.back(), {BlobRef("x", blob)}));
        }
        // Ages 12 s .. 1 s ago, entry 0 the oldest; then entry 0 is used.
        const auto now = std::filesystem::file_time_type::clock::now();
        for (int i = 0; i < 12; ++i) {
            std::filesystem::last_write_time(writer.path_for(keys[static_cast<size_t>(i)]),
                                             now - std::chrono::seconds(12 - i));
        }
        REQUIRE(writer.get(keys[0]).has_value());
        CHECK_EQ(writer.stats().trims, 0u);   // unbounded: never walks
    }
    CHECK_EQ(entry_bytes(dir), 12u * 4096u);

    // Budget 10 units: 12 is over, so down to 8.75, i.e. 8 units: the four
    // least recently used go — 1..4, not 0, which was just used.
    ArtifactCache cache(dir, 10 * 4096);
    CHECK_EQ(cache.trim(), 8u * 4096u);
    CHECK_EQ(cache.stats().evictions, 4u);
    for (int i = 0; i < 12; ++i) {
        const bool kept = cache.get(keys[static_cast<size_t>(i)]).has_value();
        CHECK_EQ(kept, i == 0 || i >= 5);
    }
    // Under the budget a walk deletes nothing.
    CHECK_EQ(cache.trim(), 8u * 4096u);
    CHECK_EQ(cache.stats().evictions, 4u);
}

TEST_CASE("artifact_cache: stores trim in the background, and the newest store survives a tiny budget") {
    const std::filesystem::path dir = fresh_dir("budget");
    const std::string blob(3000, 'b');
    {
        ArtifactCache cache(dir, 8 * 4096);
        CacheKey last;
        for (int i = 0; i < 40; ++i) {
            last = key_of("budget" + std::to_string(i));
            REQUIRE(cache.put(last, {BlobRef("x", blob)}));
        }
        cache.flush();
        CHECK(cache.stats().trims >= 1u);
        // A walk evicts only when the directory is over the budget (then down
        // to 7/8 of it), so what it leaves is within the budget, not under 7/8.
        CHECK(cache.trim() <= 8u * 4096u);
        CHECK(entry_bytes(dir) <= 8u * 4096u);
        CHECK(cache.get(last).has_value());
    }
    // A budget smaller than one entry: the entry just stored stays.
    ArtifactCache tiny(dir, 1024);
    const CacheKey k = key_of("tiny");
    REQUIRE(tiny.put(k, {BlobRef("x", blob)}));
    tiny.flush();
    tiny.trim();
    CHECK(tiny.get(k).has_value());
    CHECK_EQ(entry_bytes(dir), 4096u);
}

TEST_CASE("artifact_cache: abandoned writes are deleted, young ones kept, other files left alone") {
    const std::filesystem::path dir = fresh_dir("abandoned");
    std::filesystem::create_directories(dir);
    const std::string hex = key_of("abandoned").hex();
    const std::filesystem::path old_tmp = dir / (hex + ".bac.tmp-1-1");
    const std::filesystem::path young_tmp = dir / (hex + ".bac.tmp-1-2");
    const std::filesystem::path foreign = dir / "README.txt";
    write_bytes(old_tmp, std::vector<uint8_t>(100, 1));
    write_bytes(young_tmp, std::vector<uint8_t>(100, 1));
    write_bytes(foreign, std::vector<uint8_t>(100000, 1));
    std::filesystem::last_write_time(old_tmp,
                                     std::filesystem::file_time_type::clock::now() - std::chrono::hours(1));
    ArtifactCache cache(dir, 4096);
    CHECK_EQ(cache.trim(), 4096u);   // the young write counts; the foreign file does not
    CHECK_FALSE(std::filesystem::exists(old_tmp));
    CHECK(std::filesystem::exists(young_tmp));
    CHECK(std::filesystem::exists(foreign));
}

TEST_CASE("artifact_cache: files and the directory deleted behind the cache's back") {
    const std::filesystem::path dir = fresh_dir("behind");
    ArtifactCache cache(dir, 64 * 4096);
    const CacheKey key = key_of("behind");
    REQUIRE(cache.put(key, {BlobRef("x", "1")}));
    std::filesystem::remove(cache.path_for(key));
    Miss why = Miss::None;
    CHECK_FALSE(cache.get(key, &why).has_value());
    CHECK(why == Miss::Absent);
    CHECK_FALSE(cache.remove(key));
    std::filesystem::remove_all(dir);
    CHECK_EQ(cache.trim(), 0u);
    REQUIRE(cache.put(key, {BlobRef("x", "2")}));
    CHECK_EQ(cache.get(key)->text("x"), std::string_view("2"));
    CHECK(cache.remove(key));
    CHECK_FALSE(cache.get(key).has_value());
}

TEST_CASE("artifact_cache: two caches on one directory") {
    const std::filesystem::path dir = fresh_dir("two");
    ArtifactCache a(dir, 0);
    ArtifactCache b(dir, 3 * 4096);
    const std::string blob(3000, 't');
    std::vector<CacheKey> keys;
    for (int i = 0; i < 6; ++i) {
        keys.push_back(key_of("two" + std::to_string(i)));
        REQUIRE(a.put(keys.back(), {BlobRef("x", blob)}));
    }
    const auto now = std::filesystem::file_time_type::clock::now();
    for (size_t i = 0; i < keys.size(); ++i) {
        std::filesystem::last_write_time(a.path_for(keys[i]), now - std::chrono::seconds(10 - static_cast<int>(i)));
    }
    REQUIRE(b.get(keys[2]).has_value());   // b sees a's entries, and uses one
    b.trim();                              // to 2 units: keys 2 and 5 stay
    for (size_t i = 0; i < keys.size(); ++i) {
        Miss why = Miss::None;
        const bool hit = a.get(keys[i], &why).has_value();
        CHECK_EQ(hit, i == 2 || i == 5);
        if (!hit) CHECK(why == Miss::Absent);
    }
    REQUIRE(a.put(keys[0], {BlobRef("x", blob)}));
    CHECK(b.get(keys[0]).has_value());
}

TEST_CASE("artifact_cache: concurrent writers and readers of one key never see a torn entry") {
    const std::filesystem::path dir = fresh_dir("threads");
    ArtifactCache cache(dir, 0);
    const CacheKey key = key_of("contended");
    std::atomic<int> bad{0};
    std::atomic<int> hits{0};
    std::vector<std::thread> threads;
    for (int t = 0; t < 8; ++t) {
        threads.emplace_back([&, t] {
            const std::string mine(20000 + 1000 * t, static_cast<char>('a' + t));
            for (int i = 0; i < 150; ++i) {
                if (!cache.put(key, {BlobRef("who", std::string(1, static_cast<char>('a' + t))),
                                     BlobRef("payload", mine)})) {
                    ++bad;
                }
                Miss why = Miss::None;
                const std::optional<Artifact> a = cache.get(key, &why);
                if (!a) {
                    if (why != Miss::Absent) ++bad;
                    continue;
                }
                ++hits;
                const std::string_view who = a->text("who");
                const std::string_view payload = a->text("payload");
                if (who.size() != 1 || payload.size() != static_cast<size_t>(20000 + 1000 * (who[0] - 'a')) ||
                    payload.find_first_not_of(who[0]) != std::string_view::npos) {
                    ++bad;
                }
            }
        });
    }
    for (auto& th : threads) th.join();
    CHECK_EQ(bad.load(), 0);
    CHECK(hits.load() > 0);
    // No temporary file is left behind.
    size_t files = 0;
    for (const auto& e : std::filesystem::directory_iterator(dir)) {
        (void)e;
        ++files;
    }
    CHECK_EQ(files, 1u);
}

#if !defined(_WIN32)
TEST_CASE("artifact_cache: several processes writing and reading one key") {
    const std::filesystem::path dir = fresh_dir("processes");
    const CacheKey key = key_of("processes");
    std::vector<pid_t> children;
    for (int c = 0; c < 4; ++c) {
        const pid_t pid = fork();
        REQUIRE(pid >= 0);
        if (pid == 0) {
            int status = 0;
            {
                ArtifactCache cache(dir, 16 * 4096);   // with trims racing the others
                const std::string mine(30000, static_cast<char>('k' + c));
                for (int i = 0; i < 100 && status == 0; ++i) {
                    if (!cache.put(key, {BlobRef("payload", mine)})) status = 2;
                    cache.put(key_of("filler" + std::to_string(c) + "-" + std::to_string(i)),
                              {BlobRef("x", std::string(3000, 'f'))});
                    Miss why = Miss::None;
                    const std::optional<Artifact> a = cache.get(key, &why);
                    if (!a) {
                        if (why != Miss::Absent) status = 3;
                        continue;
                    }
                    const std::string_view p = a->text("payload");
                    if (p.size() != 30000 || p.find_first_not_of(p[0]) != std::string_view::npos) status = 4;
                }
                cache.flush();
            }
            _exit(status);
        }
        children.push_back(pid);
    }
    for (pid_t pid : children) {
        int status = -1;
        REQUIRE_EQ(waitpid(pid, &status, 0), pid);
        CHECK(WIFEXITED(status));
        CHECK_EQ(WEXITSTATUS(status), 0);
    }
    ArtifactCache cache(dir, 0);
    const std::optional<Artifact> a = cache.get(key);
    REQUIRE(a.has_value());
    CHECK_EQ(a->text("payload").size(), 30000u);
}
#endif

#ifdef BRASS_STANDALONE_TEST
BRASS_TEST_MAIN()
#endif
