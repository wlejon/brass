# The Artifact Cache

brass has one on-disk compile cache, `brass::cache::ArtifactCache` (`include/brass/cache/artifact_cache.hpp`), and one layer on top of it for native code, `brass::cache::NativeModuleCache` (`include/brass/cache/native_module_cache.hpp`). Every front end that compiles through brass keeps what it compiled there: a JavaScript compiler its lowered IL, a kernel language its SPIR-V and CPU code, a tensor library its fused kernels. The cache knows nothing about what it stores. It stores named byte blobs under a key the client builds, and it guarantees that what comes back is exactly what was stored under that key, or nothing.

This document covers the key, the file format, the guarantees, eviction, the native-module layer, and how a front end uses it.

---

## 1. Keys

A key is the digest of everything a compile read. The client builds it from named fields with `KeyBuilder`:

```cpp
using namespace brass::cache;
CacheKey key = KeyBuilder("bronze.il/3")          // the client and its key schema
                   .add_brass_build()             // what brass contributes (below)
                   .add_u64("abi", BRONZE_ABI_FINGERPRINT)
                   .add("entry.name", name)
                   .add("entry.text", text)
                   .add_bool("retain_source", retain)
                   .finish();
```

Each field is fed as its name and its value, each preceded by its length, so `("ab","c")` and `("a","bc")` are different keys and renaming a field changes every key. Fields are hashed in the order given. The domain string (the `KeyBuilder` argument) separates clients that share a directory. The material is not buffered: the builder hashes as it goes, so a key over megabytes of source costs one pass over it.

`CacheKey` carries two 128-bit digests of the same material under two seeds. `id` names the file. `check` is stored in the file's header and compared on every read, so two keys whose ids collide are a miss, not each other's artifact.

The hash is MurmurHash3 x64/128 (`brass/core/hash128.hpp`: `hash128`, the incremental `Hasher128`). It is stable across platforms, builds and runs, and is not cryptographic. It separates inputs that differ by accident, not inputs an adversary chose to collide. A cache directory is trusted local state, like the compiler binary.

### 1.1. What brass contributes

Invalidation is the client's job: an input the key leaves out is an input whose change the cache cannot see. Two of those inputs are brass's, and every client needs them, so the builder supplies them.

- `add_brass_build(target)` adds the cache format version, brass's version, the target (architecture, OS, object format, pointer size, stack alignment), the CPU features codegen for the host reads (AVX, AVX2, FMA, all false for a target that is not the host), and the identity of the binary holding brass's code: its path, size and modification time. A rebuild of brass, or of the library or executable it is linked into, is therefore a new key space. When the binary cannot be identified the key comes back with `valid == false`. Such a key never hits and is never stored, so nothing gets cached that could not be invalidated.
- `add_brass_codegen(target)` adds the same without the binary's identity. It is for an artifact that must outlive a rebuild of the compiler as long as its ABI holds, such as a module compiled ahead of time and shipped with an app. The client then owns the ABI half of the key: its runtime's ABI fingerprint and its own format versions.

`brass_binary_identity()` returns the identity string, for a client that wants to log it.

---

## 2. On Disk

A cache is a directory. Each key has one file, `<id as 32 hex digits>.bac`:

| offset | size | field |
|---|---|---|
| 0 | 4 | magic `BRAC` |
| 4 | 4 | format version (1) |
| 8 | 16 | key `id` (lo, hi) |
| 24 | 16 | key `check` (lo, hi) |
| 40 | 8 | payload bytes (everything after the 80-byte header) |
| 48 | 16 | checksum: Murmur3-128 of header bytes 0–48 and 64–80, then the payload |
| 64 | 4 | blob count |
| 68 | 12 | zero |
| 80 | | the blobs |

Each blob is its name length (u32), a zero u32, its data length (u64), the name, zero padding to a 16-byte offset, the data, and zero padding to a 16-byte offset. All integers are little-endian. Every blob starts 16-byte aligned in the file, and `Artifact` keeps the file in a 16-byte-aligned buffer, so `Artifact::as<uint32_t>("spirv")` is a view, not a copy.

A read checks the magic and version (`Miss::Foreign`), that the size is what the header says (`Miss::Truncated`), both key digests (`Miss::KeyMismatch`), the checksum over the header and payload (`Miss::Corrupt`), and that the blob table covers the payload exactly (`Miss::Corrupt`). A flipped bit anywhere in the file is a miss.

A store encodes the whole file, writes it to `<name>.tmp-<pid>-<serial>` beside the entry (created exclusively, with a handle no child process inherits), and renames it over the entry. A reader sees the old file or the new one, never part of one.

Other files in the directory are left alone and are not counted. The native layer keeps its load copies in the `load/` subdirectory (section 5).

`write_artifact_file(path, key, blobs)` and `read_artifact_file(path, key)` write and read the same format at a path the caller chooses, such as a record kept beside a module rather than in a cache directory.

---

## 3. Guarantees

- **No wrong hit.** A hit returns exactly the blobs stored under that key. A missing, foreign, torn, damaged or colliding file is a miss. A key whose material collides in both 128-bit digests is not handled; for accidental collisions this does not happen.
- **Atomic replacement.** Concurrent stores of one key, from any threads of any processes, each write a private temporary file and rename it into place. The last rename wins, and every reader sees one complete version.
- **No cross-process lock.** Nothing waits on another process. A miss returns at once and the caller compiles. Two processes that miss on the same key both compile and both store; the results are equivalent and either one may win.
- **Failures are misses.** A store that cannot be written returns false (with a reason) and changes nothing. Callers usually ignore it: a cache that cannot be written is a cache that misses.
- **Files may vanish.** Every file operation tolerates the file or the whole directory being deleted by another process or by hand. A store recreates the directory.
- **Threads.** Every `ArtifactCache` method is safe from any number of threads. An `Artifact` owns its bytes and is independent of the cache. `NativeModuleCache` is safe from any number of threads, and in one instance concurrent callers for one key share one build.
- **Isolation.** There are no globals and no environment switches. The directory and the budget are constructor arguments. Two instances on one directory share its contents and nothing else.

The limits:

- There is no `fsync`. After a power loss an entry may be torn or empty; the checksum turns that into a miss.
- The budget is soft. Each instance estimates the directory from its own last walk plus its own stores since, so other processes' stores show up only at its next walk. Between walks the directory can exceed the budget by what the other processes stored.
- Last use is the file's modification time, set on every hit. On a filesystem mounted read-only, or one with coarse timestamps, the LRU order is approximate.

---

## 4. Size and Eviction

`ArtifactCache(dir, max_bytes)` holds the directory's entries to `max_bytes` on disk, counted in whole 4 KiB units: entries are often a few kilobytes, and counting bytes would under-count a full directory by a good fraction. `max_bytes == 0` means no bound.

A store adds what it wrote to the instance's estimate. When the estimate passes the budget, or when the instance has not walked the directory yet, the store wakes the instance's trim thread. The storing thread never walks the directory. The walk:

1. lists the directory, deletes temporary files more than ten minutes old (a writer that died between write and rename), and counts younger ones (writes in progress);
2. if the entries take more than the budget, sorts them by last use and deletes the oldest until they take at most 7/8 of the budget, so the next few stores do not each trigger another walk;
3. never deletes the entry this instance stored last, nor one it stores while the walk runs, so a budget smaller than one entry still keeps the entry just made.

A delete that fails is skipped. On Windows, a file another process holds open without delete sharing cannot be deleted. Readers here open entries with delete sharing, so only foreign readers block it. A delete that finds the file already gone counts it as gone. `trim()` runs a walk on the calling thread, and `flush()` waits for the background one; tests and maintenance tools use them. The destructor waits for a walk in progress and drops one that has not started.

---

## 5. Native Modules

`NativeModuleCache` turns MIR into loadable code in one call:

```cpp
ArtifactCache store(dir, 512ull << 20);
NativeModuleCache natives(store);

std::shared_ptr<const NativeModule> m = natives.get_or_build(
    key,                                       // the client's: what the module is built from
    [&](brass::Module& mod, std::string& err) {
        return build_kernel(mod, spec, err);   // runs only on a miss
    },
    options, &error);
auto fn = m->function<KernelFn>("kernel");
```

On a miss the callback fills a MIR module. brass compiles it for the host, with AVX scalar forms where the CPU has AVX, as the JIT does. `target::AotLinker` links it into a shared library image: an ELF `.so`, a PE `.dll` or a Mach-O `.dylib`. The image is stored as the blob `image`, then loaded through `target::DynamicLibrary`. On a hit, in this process or any later one, the stored image is loaded and nothing is built or compiled; `NativeModule::from_cache()` says which happened. `get_or_build_object` takes a callback that produces a `object::ObjectFile` instead, for a client that runs its own pipeline up to the object. `find` loads a stored module and never builds one.

The image is stored under `storage_key(key, options)`, not under the client's key. The storage key adds `add_brass_build` and the link options (`NativeOptions`: imports, rpaths, exports), so a rebuilt brass never loads an image an older brass compiled, even when the client forgets brass in its own key.

**Imports.** A cached image cannot point at addresses in the process that built it. Every function the module calls but does not define must come from a library named in `NativeOptions::imports`, as with any `AotLinker` image, or the link fails naming the symbol. Code that needs a host's in-process helpers by address belongs in the JIT, not here.

**Lifetime.** Each load writes a private copy of the image under `<dir>/load/` and loads that copy. On POSIX the copy is unlinked as soon as it is loaded. On Windows a loaded DLL cannot be deleted, so the copy is deleted when its `NativeModule` is destroyed. Either way the cache entry itself is never mapped, so evicting or replacing it never touches running code. A `NativeModule` stays valid for as long as the `shared_ptr` lives. Keep it for as long as any function pointer from it may be called. Copies a crashed process left behind are deleted when a `NativeModuleCache` is constructed, once they are ten minutes old.

**Damage.** A stored image that does not load is counted (`Stats::load_failures`), rebuilt and stored over. On macOS the image is signed ad hoc (`codesign -s -`) once, at build time, and the signed bytes are what is stored, so a hit loads without signing again.

Within one instance, concurrent `get_or_build` calls for one key run one load or build, and the others wait for it and get its module. A module this instance already holds is handed out again (`Stats::reuses`). Across processes, two misses build twice.

---

## 6. Using It From a Front End

1. Give the cache a directory and a budget at construction. The host decides where they come from, such as a setting or the user cache directory. The library reads no environment.
2. Build the key from everything the compile reads that can change its output: source text or digests, options, versions, `add_brass_build()`. If anything that could change the output is not in the key, a stale hit is possible. Inputs found during the compile (imported files, module resolutions) can't be keyed in advance: store their digests as a blob and check them on a hit, and treat a mismatch as a miss.
3. `get`. On a hit, decode the blobs. If the decoded content does not hold up, treat it as a miss.
4. On a miss, compile, then `put` the blobs. Ignore a failed store.
5. For native code, use `NativeModuleCache` rather than storing images by hand: it adds brass to the key, signs on macOS, loads a private copy, and keeps the module valid across eviction.

One directory can serve several clients and both layers, each keyed under its own domain. All of them then share one budget.
