// Native modules through the artifact cache: build, link and store on a
// miss, load a private copy of the stored image on a hit
// (native_module_cache.hpp, docs/artifact_cache.md "Native modules").
#include <brass/cache/native_module_cache.hpp>

#include <brass/mir/module.hpp>
#include <brass/object/object_writer.hpp>
#include <brass/target/aot_linker.hpp>
#include <brass/target/dynamic_library.hpp>
#include <brass/target/target.hpp>

#include "cache/artifact_format.hpp"

#include <chrono>
#include <cstdlib>
#include <exception>
#include <system_error>

namespace brass::cache {

namespace {

// A load copy older than this is one a crashed process left (on Windows,
// one still loaded cannot be deleted, and the delete simply fails).
constexpr auto kAbandonedAfter = std::chrono::minutes(10);

const char* image_extension() {
    const Target host = Target::host();
    if (host.is_windows()) return ".dll";
    if (host.is_macos()) return ".dylib";
    return ".so";
}

void set_error(std::string* error, const std::string& text) {
    if (error) *error = text;
}

} // namespace

// ---------------------------------------------------------------------------
// NativeModule
// ---------------------------------------------------------------------------

NativeModule::~NativeModule() {
    library_.reset();
    if (!load_path_.empty()) {
        std::error_code ec;
        std::filesystem::remove(load_path_, ec);
    }
}

void* NativeModule::symbol(std::string_view name) const {
    return library_ ? library_->get_symbol(std::string(name)) : nullptr;
}

// ---------------------------------------------------------------------------
// NativeModuleCache
// ---------------------------------------------------------------------------

NativeModuleCache::NativeModuleCache(ArtifactCache& store) : store_(store) { sweep_load_dir(); }

NativeModuleCache::~NativeModuleCache() = default;

std::filesystem::path NativeModuleCache::load_dir() const { return store_.dir() / "load"; }

NativeModuleCache::Stats NativeModuleCache::stats() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return stats_;
}

CacheKey NativeModuleCache::storage_key(const CacheKey& key, const NativeOptions& options) const {
    KeyBuilder kb("brass.native/1");
    kb.add_key("client", key);
    kb.add_brass_build(Target::host());
    kb.add_u64("imports", options.imports.size());
    for (const target::ImportLibrary& lib : options.imports) {
        kb.add("import.library", lib.library);
        kb.add_u64("import.symbols", lib.symbols.size());
        for (const std::string& s : lib.symbols) kb.add("import.symbol", s);
    }
    kb.add_u64("rpaths", options.rpaths.size());
    for (const std::string& r : options.rpaths) kb.add("rpath", r);
    kb.add_u64("exports", options.exports.size());
    for (const std::string& e : options.exports) kb.add("export", e);
    return kb.finish();
}

void NativeModuleCache::sweep_load_dir() {
    const auto now = std::filesystem::file_time_type::clock::now();
    std::error_code ec;
    std::filesystem::directory_iterator it(load_dir(), ec);
    const std::filesystem::directory_iterator end;
    for (; !ec && it != end; it.increment(ec)) {
        std::error_code e2;
        if (!it->is_regular_file(e2) || e2) continue;
        const auto mtime = it->last_write_time(e2);
        if (e2 || now - mtime <= kAbandonedAfter) continue;
        std::filesystem::remove(it->path(), e2);
    }
}

std::shared_ptr<const NativeModule> NativeModuleCache::load_image(const CacheKey& skey,
                                                                  std::span<const uint8_t> image, bool from_cache,
                                                                  std::string* error) {
    // A private copy per load: the entry can then be evicted or replaced
    // while the code runs, and on Windows (where a loaded file cannot be
    // deleted) the entry itself is never held.
    const std::filesystem::path dir = load_dir();
    std::error_code ec;
    std::filesystem::create_directories(dir, ec);
    const std::filesystem::path path =
        dir / (skey.hex() + "-" + std::to_string(detail::process_id()) + "-" +
               std::to_string(detail::next_serial()) + image_extension());
    if (!detail::write_new_file(path, image, error, /*executable=*/true)) return nullptr;

    std::string why;
    std::unique_ptr<target::DynamicLibrary> lib = target::DynamicLibrary::open(path.string(), &why);
    bool keep_file = false;
#if defined(_WIN32)
    // A loaded DLL's file cannot be deleted: it goes when the module does.
    keep_file = lib != nullptr;
#endif
    if (!keep_file) std::filesystem::remove(path, ec);
    if (!lib || !lib->is_valid()) {
        set_error(error, "cannot load the image: " + why);
        return nullptr;
    }
    std::shared_ptr<NativeModule> m(new NativeModule());
    m->library_ = std::move(lib);
    if (keep_file) m->load_path_ = path;
    m->key_ = skey;
    m->image_bytes_ = image.size();
    m->from_cache_ = from_cache;
    return m;
}

std::shared_ptr<const NativeModule> NativeModuleCache::load_stored(const CacheKey& skey, std::string* error) {
    Miss why = Miss::None;
    const std::optional<Artifact> art = store_.get(skey, &why);
    if (!art) {
        set_error(error, to_string(why));
        return nullptr;
    }
    const std::span<const uint8_t> image = art->bytes("image");
    std::shared_ptr<const NativeModule> m;
    if (!image.empty()) m = load_image(skey, image, /*from_cache=*/true, error);
    if (!m) {
        // Rebuilt (when the caller can) and stored over it.
        std::lock_guard<std::mutex> lock(mutex_);
        ++stats_.load_failures;
        return nullptr;
    }
    std::lock_guard<std::mutex> lock(mutex_);
    ++stats_.hits;
    return m;
}

bool NativeModuleCache::link_object(const object::ObjectFile& obj, const CacheKey& skey,
                                    const NativeOptions& options, std::vector<uint8_t>& image, std::string& error) {
    if (obj.target != Target::host()) {
        error = "the object is not for this host";
        return false;
    }
    const std::string name = "brass-native-" + skey.hex() + image_extension();
    target::LinkerOptions lo;
    lo.module_name = name;
    lo.soname = name;
    lo.export_all_functions = options.exports.empty();
    lo.explicit_exports = options.exports;
    lo.imports = options.imports;
    lo.rpaths = options.rpaths;
    image = target::AotLinker::link(obj, lo, &error);
    if (image.empty()) {
        if (error.empty()) error = "the link failed";
        return false;
    }
#if defined(__APPLE__)
    // Apple silicon loads only signed code. Signed once here, so the stored
    // image carries its (ad hoc) signature and a hit loads it as it is.
    const std::filesystem::path dir = load_dir();
    std::error_code ec;
    std::filesystem::create_directories(dir, ec);
    const std::filesystem::path tmp = dir / (skey.hex() + "-sign-" + std::to_string(detail::process_id()) + "-" +
                                             std::to_string(detail::next_serial()) + image_extension());
    if (!detail::write_new_file(tmp, image, &error, /*executable=*/true)) return false;
    const std::string cmd = "codesign -s - -f \"" + tmp.string() + "\" > /dev/null 2>&1";
    const int rc = std::system(cmd.c_str());
    const bool read = rc == 0 && detail::read_whole_file(tmp, image);
    std::filesystem::remove(tmp, ec);
    if (!read) {
        error = "cannot sign the image";
        return false;
    }
#endif
    return true;
}

std::shared_ptr<const NativeModule> NativeModuleCache::obtain(const CacheKey& key, const NativeOptions& options,
                                                              const Builder* build, std::string* error) {
    if (!key.valid) {
        set_error(error, "the key is not valid");
        return nullptr;
    }
    const CacheKey skey = storage_key(key, options);
    if (!skey.valid) {
        set_error(error, "brass's binary cannot be identified, so nothing is cached");
        return nullptr;
    }

    // One load or build per key at a time in this instance: the others
    // wait for it and take its module.
    {
        std::unique_lock<std::mutex> lock(mutex_);
        for (;;) {
            Slot& slot = slots_[skey.id];
            if (std::shared_ptr<const NativeModule> m = slot.module.lock()) {
                ++stats_.reuses;
                return m;
            }
            if (!slot.building) {
                slot.building = true;
                break;
            }
            cv_.wait(lock);
        }
    }
    struct Release {
        NativeModuleCache& self;
        const Digest128& id;
        std::shared_ptr<const NativeModule> result;
        ~Release() {
            std::lock_guard<std::mutex> lock(self.mutex_);
            Slot& slot = self.slots_[id];
            slot.building = false;
            if (result) {
                slot.module = result;
            } else if (slot.module.expired()) {
                self.slots_.erase(id);
            }
            self.cv_.notify_all();
        }
    } release{*this, skey.id, nullptr};

    std::string why;
    release.result = load_stored(skey, &why);
    if (release.result || !build) {
        if (!release.result) set_error(error, why);
        return release.result;
    }

    std::vector<uint8_t> image;
    std::string build_error;
    bool built = false;
    try {
        built = (*build)(skey, image, build_error);
    } catch (const std::exception& ex) {
        build_error = std::string("the build threw: ") + ex.what();
    }
    if (!built) {
        std::lock_guard<std::mutex> lock(mutex_);
        ++stats_.build_failures;
        set_error(error, build_error.empty() ? std::string("the build failed") : build_error);
        return nullptr;
    }
    {
        std::lock_guard<std::mutex> lock(mutex_);
        ++stats_.builds;
    }
    // Stored before it is loaded: a store that fails costs the next process
    // a build, not this one its module.
    store_.put(skey, {BlobRef("image", std::span<const uint8_t>(image.data(), image.size()))});
    release.result = load_image(skey, image, /*from_cache=*/false, error);
    return release.result;
}

std::shared_ptr<const NativeModule> NativeModuleCache::get_or_build(const CacheKey& key, const BuildModule& build,
                                                                    const NativeOptions& options,
                                                                    std::string* error) {
#if !BRASS_HOST_BACKEND
    (void)build;
    if (std::shared_ptr<const NativeModule> m = find(key, options)) return m;
    set_error(error, "brass has no code generator for this host");
    return nullptr;
#else
    const Builder builder = [&](const CacheKey& skey, std::vector<uint8_t>& image, std::string& err) {
        Module mod("brass_native_" + skey.hex());
        if (!build(mod, err)) return false;
        const Target host = Target::host();
        object::ModuleCompiler compiler(host);
        // As the JIT does for the host: AVX scalar forms where the CPU has
        // AVX (in the key through add_brass_build).
        compiler.set_x64_avx(host.is_x64() && Target::host_has_avx());
        const object::ObjectFile obj = compiler.compile(mod);
        return link_object(obj, skey, options, image, err);
    };
    return obtain(key, options, &builder, error);
#endif
}

std::shared_ptr<const NativeModule> NativeModuleCache::get_or_build_object(const CacheKey& key,
                                                                           const BuildObject& build,
                                                                           const NativeOptions& options,
                                                                           std::string* error) {
    const Builder builder = [&](const CacheKey& skey, std::vector<uint8_t>& image, std::string& err) {
        object::ObjectFile obj;
        if (!build(obj, err)) return false;
        return link_object(obj, skey, options, image, err);
    };
    return obtain(key, options, &builder, error);
}

std::shared_ptr<const NativeModule> NativeModuleCache::find(const CacheKey& key, const NativeOptions& options) {
    return obtain(key, options, nullptr, nullptr);
}

} // namespace brass::cache
