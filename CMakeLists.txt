cmake_minimum_required(VERSION 3.20)

# Is brass the build, or a piece of someone else's?
if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    set(BRASS_IS_TOP_LEVEL ON)
else()
    set(BRASS_IS_TOP_LEVEL OFF)
endif()

if(BRASS_IS_TOP_LEVEL AND APPLE AND NOT CMAKE_OSX_ARCHITECTURES)
    # Default to host architecture on Apple Silicon (native arm64) or Intel (x86_64)
    # Users can pass -DCMAKE_OSX_ARCHITECTURES=x86_64 to test under Rosetta
endif()

project(brass VERSION 0.1.0 LANGUAGES CXX C)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Export compile commands for language servers
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Default to Release if not specified (top-level only)
if(BRASS_IS_TOP_LEVEL AND NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Choose build type: Debug Release RelWithDebInfo MinSizeRel." FORCE)
endif()

# Compiler flags and warnings
if(MSVC)
    if(BRASS_IS_TOP_LEVEL)
        string(REPLACE "/EHsc" "/EHs" CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
        add_compile_options(
            $<$<COMPILE_LANGUAGE:CXX>:/EHs>
            $<$<COMPILE_LANGUAGE:CXX>:/W4>
            $<$<COMPILE_LANGUAGE:CXX>:/WX>
            $<$<COMPILE_LANGUAGE:CXX>:/wd4324>
            $<$<COMPILE_LANGUAGE:CXX>:/wd4100>
            $<$<COMPILE_LANGUAGE:CXX>:/wd4146>
            $<$<COMPILE_LANGUAGE:CXX>:/wd4456>
            $<$<COMPILE_LANGUAGE:CXX>:/wd4457>
            $<$<COMPILE_LANGUAGE:CXX>:/wd4505>
            $<$<COMPILE_LANGUAGE:CXX>:/permissive->
            $<$<COMPILE_LANGUAGE:CXX>:/Zc:__cplusplus>
            $<$<COMPILE_LANGUAGE:CXX>:/utf-8>
            $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<CONFIG:Release>>:/O2>
            $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<CONFIG:Release>>:/Oy->
        )
    endif()
else()
    if(BRASS_IS_TOP_LEVEL)
        add_compile_options(
            -Wall
            -Wextra
            -Wpedantic
            -Wconversion
            -Wno-unused-parameter
            -fno-omit-frame-pointer
            $<$<CONFIG:Release>:-O3>
        )
        if(WIN32 OR MINGW)
            add_link_options(-static -static-libgcc -static-libstdc++)
        elseif(NOT APPLE)
            add_link_options(-static-libgcc -static-libstdc++)
        endif()
    endif()
endif()

# Does brass emit machine code for the target this build is for?
#
# Brass supports native machine code generation for x86_64 and AArch64.
# BRASS_HOST_BACKEND is a CACHE INTERNAL so it is visible from any directory
# of a build that add_subdirectory()s brass, however deep (an app -> its JS engine -> brass),
# and the macro of the same name is on the target so C++ can ask too.
if(APPLE AND CMAKE_OSX_ARCHITECTURES)
    set(_brass_target_arch "${CMAKE_OSX_ARCHITECTURES}")
else()
    set(_brass_target_arch "${CMAKE_SYSTEM_PROCESSOR}")
endif()
if(_brass_target_arch MATCHES "^(x86_64|AMD64|amd64|x64|aarch64|arm64|ARM64)$")
    set(_brass_host_backend ON)
else()
    set(_brass_host_backend OFF)
endif()
set(BRASS_HOST_BACKEND ${_brass_host_backend} CACHE INTERNAL
    "brass emits machine code for this build's target architecture")
if(NOT BRASS_HOST_BACKEND)
    message(STATUS "brass: no code generator for ${_brass_target_arch}; "
        "JIT-executing paths are compiled but cannot run on this target")
endif()

# brass library include directory
set(BRASS_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/include)

# brass core library
file(GLOB_RECURSE BRASS_SOURCES CONFIGURE_DEPENDS
    "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp"
)
list(APPEND BRASS_SOURCES
    "${CMAKE_CURRENT_SOURCE_DIR}/src/object/macho_writer.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/target/macho_dylib_writer.cpp"
)
list(REMOVE_DUPLICATES BRASS_SOURCES)

# The entry, throw and invoke stubs of an MSVC build (cl or clang-cl) are
# assembly; GCC/Clang builds define them as file-scope asm in C++.
if(MSVC)
    if(CMAKE_CXX_COMPILER_ARCHITECTURE_ID MATCHES "^(ARM64|arm64)$")
        enable_language(ASM_MARMASM)
        list(APPEND BRASS_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/src/gc/gc_msvc_arm64.asm")
    else()
        enable_language(ASM_MASM)
        list(APPEND BRASS_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/src/gc/gc_msvc_x64.asm")
    endif()
endif()

add_library(brass STATIC ${BRASS_SOURCES})
add_library(brass::brass ALIAS brass)

target_include_directories(brass
    PUBLIC
        $<BUILD_INTERFACE:${BRASS_INCLUDE_DIR}>
        $<INSTALL_INTERFACE:include>
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
)

target_compile_features(brass PUBLIC cxx_std_20)
target_compile_definitions(brass PUBLIC BRASS_HOST_BACKEND=$<BOOL:${BRASS_HOST_BACKEND}>)

if(MSVC)
    target_compile_options(brass PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:/EHs>
        $<$<COMPILE_LANGUAGE:CXX>:/W4>
        $<$<COMPILE_LANGUAGE:CXX>:/wd4324>
        $<$<COMPILE_LANGUAGE:CXX>:/wd4100>
        $<$<COMPILE_LANGUAGE:CXX>:/wd4146>
        $<$<COMPILE_LANGUAGE:CXX>:/wd4456>
        $<$<COMPILE_LANGUAGE:CXX>:/wd4457>
        $<$<COMPILE_LANGUAGE:CXX>:/wd4505>
        $<$<COMPILE_LANGUAGE:CXX>:/permissive->
        $<$<COMPILE_LANGUAGE:CXX>:/Zc:__cplusplus>
        $<$<COMPILE_LANGUAGE:CXX>:/utf-8>
    )
else()
    target_compile_options(brass PRIVATE
        -Wall
        -Wextra
        -Wpedantic
        -Wno-unused-parameter
        -fno-omit-frame-pointer
    )
endif()

option(BRASS_BUILD_SHARED "Build brass shared library" ${BRASS_IS_TOP_LEVEL})
option(BRASS_BUILD_TOOLS "Build brass CLI tools" ${BRASS_IS_TOP_LEVEL})
option(BRASS_BUILD_TESTS "Build brass tests" ${BRASS_IS_TOP_LEVEL})

if(BRASS_BUILD_SHARED)
    # brass shared library (libbrass)
    add_library(brass_shared SHARED ${BRASS_SOURCES})
    add_library(brass::shared ALIAS brass_shared)

    target_include_directories(brass_shared
        PUBLIC
            $<BUILD_INTERFACE:${BRASS_INCLUDE_DIR}>
            $<INSTALL_INTERFACE:include>
        PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}/src
    )

    target_compile_features(brass_shared PUBLIC cxx_std_20)
    target_compile_definitions(brass_shared PUBLIC BRASS_HOST_BACKEND=$<BOOL:${BRASS_HOST_BACKEND}>)

    target_compile_definitions(brass_shared
        PRIVATE
            BRASS_BUILD_SHARED
            BRASS_EXPORTS
    )

    set_target_properties(brass_shared PROPERTIES
        OUTPUT_NAME "brass"
        ARCHIVE_OUTPUT_NAME "brass_shared"
        SOVERSION 1
        VERSION 1.0.0
    )
endif()

# Installation
if(BRASS_IS_TOP_LEVEL)
    install(TARGETS brass
        ARCHIVE DESTINATION lib
        LIBRARY DESTINATION lib
        RUNTIME DESTINATION bin
    )
    if(BRASS_BUILD_SHARED)
        install(TARGETS brass_shared
            ARCHIVE DESTINATION lib
            LIBRARY DESTINATION lib
            RUNTIME DESTINATION bin
        )
    endif()
    install(FILES
        ${BRASS_INCLUDE_DIR}/brass/brass_c_api.h
        ${BRASS_INCLUDE_DIR}/brass/brass_export.h
        DESTINATION include/brass)
    install(FILES ${BRASS_INCLUDE_DIR}/brass/embedding/brass_c_api.h DESTINATION include/brass/embedding)
endif()

# Tools
if(BRASS_BUILD_TOOLS)
    add_subdirectory(tools)
endif()

# Testing
if(BRASS_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()
