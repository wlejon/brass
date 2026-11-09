#pragma once
// MIR -> cached loadable code in one call (docs/artifact_cache.md "Native
// modules"). On a miss the client's callback builds a MIR module (or an
// object file); brass compiles it for the host, links it into a shared
// library image (target::AotLinker), stores the image in an ArtifactCache and
// loads it (target::DynamicLibrary). On a hit — in this process or any later
// one — the stored image is loaded and nothing is built or compiled.

#include <brass/cache/artifact_cache.hpp>
#include <brass/target/image_imports.hpp>

#include <condition_variable>
#include <cstdint>
#include <filesystem>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace brass {
class Module;
namespace object {
struct ObjectFile;
}
namespace target {
class DynamicLibrary;
}
} // namespace brass

namespace brass::cache {

// How the image is linked. Part of the stored key: two option sets are two
// images.
struct NativeOptions {
    // Where undefined symbols come from (target::LinkerOptions::imports):
    // every function the module calls but does not define must be listed
    // under the library that exports it, or the link fails naming it.
    std::vector<target::ImportLibrary> imports;
    // Run-time search paths for those libraries (ELF / Mach-O).
    std::vector<std::string> rpaths;
    // The symbols to export; empty exports every function.
    std::vector<std::string> exports;
};

// A loaded image. Its code stays valid for as long as the NativeModule lives,
// whatever happens to the cache entry it came from: it is loaded from a
// private copy of the image, not from the entry, so an eviction (or another
// process replacing the entry) never touches it. Keep the shared_ptr for as
// long as any function pointer from it may be called.
class NativeModule {
public:
    ~NativeModule();
    NativeModule(const NativeModule&) = delete;
    NativeModule& operator=(const NativeModule&) = delete;

    void* symbol(std::string_view name) const;
    template <typename Fn>
    Fn function(std::string_view name) const {
        return reinterpret_cast<Fn>(symbol(name));
    }

    // True when the image came from the store (nothing was built or compiled
    // for it); false when this call built it.
    bool from_cache() const noexcept { return from_cache_; }
    // The key the image is stored under (NativeModuleCache::storage_key).
    const CacheKey& storage_key() const noexcept { return key_; }
    uint64_t image_bytes() const noexcept { return image_bytes_; }

private:
    friend class NativeModuleCache;
    NativeModule() = default;

    std::unique_ptr<target::DynamicLibrary> library_;
    std::filesystem::path load_path_;   // removed after unloading where it could not be before
    CacheKey key_;
    uint64_t image_bytes_ = 0;
    bool from_cache_ = false;
};

class NativeModuleCache {
public:
    struct Stats {
        uint64_t hits = 0;            // loaded from the store
        uint64_t reuses = 0;          // already loaded in this process by this instance
        uint64_t builds = 0;          // built, compiled and linked
        uint64_t build_failures = 0;  // the callback, compile or link failed
        uint64_t load_failures = 0;   // a stored image would not load (rebuilt)
    };

    // Fills `mod` (named after the key) with the functions to compile;
    // false with `error` set when it cannot. Run only on a miss. Whatever the
    // module is built from must be in the client's key.
    using BuildModule = std::function<bool(Module& mod, std::string& error)>;
    // The same for a client that compiles to an object itself.
    using BuildObject = std::function<bool(object::ObjectFile& obj, std::string& error)>;

    // Images are stored in `store` (which must outlive this) and loaded
    // from private copies under `<store dir>/load`. Copies a crashed
    // process left there are deleted here.
    explicit NativeModuleCache(ArtifactCache& store);
    ~NativeModuleCache();
    NativeModuleCache(const NativeModuleCache&) = delete;
    NativeModuleCache& operator=(const NativeModuleCache&) = delete;

    // The loaded module for `key`: one this instance already has loaded, the
    // stored image, or — on a miss — the one `build` describes, compiled
    // for the host (with the CPU features KeyBuilder::add_brass_build puts
    // in the key), linked, stored and loaded. Null with `error` set when it
    // cannot be had (build, compile, link or load failure, or no code
    // generator for this host). Concurrent calls for one key in this
    // instance build it once; the others wait for that build.
    //
    // `key` is the client's: what the module is built from. The image is
    // stored under storage_key(key, options), which adds brass's own build
    // identity and the link options, so a rebuilt brass never loads an old
    // image.
    std::shared_ptr<const NativeModule> get_or_build(const CacheKey& key, const BuildModule& build,
                                                     const NativeOptions& options = {},
                                                     std::string* error = nullptr);
    std::shared_ptr<const NativeModule> get_or_build_object(const CacheKey& key, const BuildObject& build,
                                                            const NativeOptions& options = {},
                                                            std::string* error = nullptr);

    // A hit only: the loaded or stored module, else null. Never builds.
    std::shared_ptr<const NativeModule> find(const CacheKey& key, const NativeOptions& options = {});

    CacheKey storage_key(const CacheKey& key, const NativeOptions& options) const;
    std::filesystem::path load_dir() const;
    Stats stats() const;

private:
    struct Slot {
        std::weak_ptr<const NativeModule> module;
        bool building = false;
    };
    // Produces the linked image for a storage key.
    using Builder = std::function<bool(const CacheKey& skey, std::vector<uint8_t>& image, std::string& error)>;

    std::shared_ptr<const NativeModule> obtain(const CacheKey& key, const NativeOptions& options,
                                               const Builder* build, std::string* error);
    std::shared_ptr<const NativeModule> load_stored(const CacheKey& skey, std::string* error);
    std::shared_ptr<const NativeModule> load_image(const CacheKey& skey, std::span<const uint8_t> image,
                                                   bool from_cache, std::string* error);
    bool link_object(const object::ObjectFile& obj, const CacheKey& skey, const NativeOptions& options,
                     std::vector<uint8_t>& image, std::string& error);
    void sweep_load_dir();

    ArtifactCache& store_;
    mutable std::mutex mutex_;
    std::condition_variable cv_;
    std::map<Digest128, Slot> slots_;   // by storage key id
    Stats stats_;
};

} // namespace brass::cache
