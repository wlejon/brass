#include "image_file.hpp"

#include <cerrno>
#include <cstring>
#include <system_error>

#if defined(_WIN32)
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <fcntl.h>
#include <unistd.h>
#endif

namespace brass::target::image {

namespace {

void set_error(std::string* error_out, const std::string& path, const char* what,
               const std::string& reason) {
    if (!error_out) return;
    *error_out = std::string("cannot ") + what + " " + path + ": " + reason;
}

#if defined(_WIN32)
std::string system_message(DWORD code) {
    char* text = nullptr;
    const DWORD len = FormatMessageA(
        FORMAT_MESSAGE_ALLOCATE_BUFFER | FORMAT_MESSAGE_FROM_SYSTEM | FORMAT_MESSAGE_IGNORE_INSERTS,
        nullptr, code, 0, reinterpret_cast<char*>(&text), 0, nullptr);
    std::string message = (len && text) ? std::string(text, len) : ("error " + std::to_string(code));
    if (text) LocalFree(text);
    while (!message.empty() && (message.back() == '\n' || message.back() == '\r' || message.back() == ' ')) {
        message.pop_back();
    }
    return message;
}

std::wstring to_wide(const std::string& utf8) {
    if (utf8.empty()) return std::wstring();
    const int len = MultiByteToWideChar(CP_UTF8, 0, utf8.data(), static_cast<int>(utf8.size()), nullptr, 0);
    if (len <= 0) return std::wstring(utf8.begin(), utf8.end());
    std::wstring wide(static_cast<size_t>(len), L'\0');
    MultiByteToWideChar(CP_UTF8, 0, utf8.data(), static_cast<int>(utf8.size()), wide.data(), len);
    return wide;
}
#endif

}  // namespace

bool write_image_file(const std::string& path, const std::vector<uint8_t>& bytes,
                      std::string* error_out) {
#if defined(_WIN32)
    // No SECURITY_ATTRIBUTES: the handle is not inheritable. FILE_SHARE_READ
    // so a concurrent reader of an OLD image at this path is not refused,
    // which CreateFile would otherwise do for the whole write.
    HANDLE file = CreateFileW(to_wide(path).c_str(), GENERIC_WRITE, FILE_SHARE_READ, nullptr,
                              CREATE_ALWAYS, FILE_ATTRIBUTE_NORMAL, nullptr);
    if (file == INVALID_HANDLE_VALUE) {
        set_error(error_out, path, "open", system_message(GetLastError()));
        return false;
    }
    size_t done = 0;
    while (done < bytes.size()) {
        const size_t remaining = bytes.size() - done;
        const DWORD chunk = remaining > 0x40000000u ? 0x40000000u : static_cast<DWORD>(remaining);
        DWORD written = 0;
        if (!WriteFile(file, bytes.data() + done, chunk, &written, nullptr)) {
            set_error(error_out, path, "write", system_message(GetLastError()));
            CloseHandle(file);
            return false;
        }
        done += written;
    }
    CloseHandle(file);
    return true;
#else
    const int fd = ::open(path.c_str(), O_WRONLY | O_CREAT | O_TRUNC | O_CLOEXEC, 0755);
    if (fd < 0) {
        set_error(error_out, path, "open", std::strerror(errno));
        return false;
    }
    size_t done = 0;
    while (done < bytes.size()) {
        const ssize_t put = ::write(fd, bytes.data() + done, bytes.size() - done);
        if (put < 0) {
            if (errno == EINTR) continue;
            set_error(error_out, path, "write", std::strerror(errno));
            ::close(fd);
            return false;
        }
        done += static_cast<size_t>(put);
    }
    if (::close(fd) != 0) {
        set_error(error_out, path, "write", std::strerror(errno));
        return false;
    }
    return true;
#endif
}

}  // namespace brass::target::image
