# Cross-compiles brass for aarch64 Linux with Debian's gcc cross toolchain and
# runs the result under qemu-user (CMAKE_CROSSCOMPILING_EMULATOR), so ctest and
# test discovery work on an x86_64 host without binfmt registration.
#
# BRASS_XROOT is the directory the toolchain packages were extracted into
# (scripts/linux-tests.sh setup); leave it empty for a system-wide install.
set(CMAKE_SYSTEM_NAME Linux)
set(CMAKE_SYSTEM_PROCESSOR aarch64)

set(BRASS_XROOT "$ENV{BRASS_XROOT}" CACHE PATH "Root the aarch64 cross toolchain and qemu were extracted into")
# try_compile projects re-read this file without the cache.
list(APPEND CMAKE_TRY_COMPILE_PLATFORM_VARIABLES BRASS_XROOT)

set(CMAKE_C_COMPILER "${BRASS_XROOT}/usr/bin/aarch64-linux-gnu-gcc")
set(CMAKE_CXX_COMPILER "${BRASS_XROOT}/usr/bin/aarch64-linux-gnu-g++")
set(CMAKE_AR "${BRASS_XROOT}/usr/bin/aarch64-linux-gnu-ar" CACHE FILEPATH "")
set(CMAKE_RANLIB "${BRASS_XROOT}/usr/bin/aarch64-linux-gnu-ranlib" CACHE FILEPATH "")

# Debian's libc.so linker script names /usr/aarch64-linux-gnu/lib/... by
# absolute path; a sysroot at the extraction root makes ld resolve those
# inside it. Headers and libraries are still found relative to the compiler.
if(BRASS_XROOT)
    set(CMAKE_SYSROOT "${BRASS_XROOT}")
endif()

set(CMAKE_FIND_ROOT_PATH "${BRASS_XROOT}/usr/aarch64-linux-gnu")
set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)

set(CMAKE_CROSSCOMPILING_EMULATOR
    "${BRASS_XROOT}/usr/bin/qemu-aarch64-static;-L;${BRASS_XROOT}/usr/aarch64-linux-gnu")
