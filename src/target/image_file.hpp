#pragma once

// Writing a finished image to disk. One function for the three writers and
// the AotLinker, and a function of its own because `std::ofstream` is the
// wrong tool for a file that is about to be LOADED: the handle it opens is
// inheritable on Windows (the CRT marks every `_open` that way unless told
// not to) and lacks close-on-exec on POSIX, so a child another thread of the
// same process spawns while the write is in flight — a test harness that
// builds on N threads and runs each result — carries the open write handle
// for as long as it lives. On Windows that child's handle makes the next
// LoadLibrary of the file fail with a sharing violation; on Linux the next
// exec of it fails with ETXTBSY. Both are "the file is still being written"
// verdicts on a file that was completely written a moment ago.
//
// The descriptor opened here is not inherited by anyone: CreateFileW with no
// inheritable security attributes, open(2) with O_CLOEXEC. The file is also
// truncated and fully written before this returns, so a caller that goes on
// to load it sees the whole image.

#include <cstdint>
#include <string>
#include <vector>

namespace brass::target::image {

// True on success. On failure `error_out` (when given) names the path and
// the operating system's reason.
bool write_image_file(const std::string& path, const std::vector<uint8_t>& bytes,
                      std::string* error_out);

}  // namespace brass::target::image
