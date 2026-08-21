cmake_minimum_required(VERSION 3.20)
project(brass VERSION 0.1.0 LANGUAGES CXX C)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Export compile commands for language servers
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Default to Release if not specified
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Choose build type: Debug Release RelWithDebInfo MinSizeRel." FORCE)
endif()

# Compiler flags and warnings
if(MSVC)
    add_compile_options(
        $<$<COMPILE_LANGUAGE:CXX>:/W4>
        $<$<COMPILE_LANGUAGE:CXX>:/WX>
        $<$<COMPILE_LANGUAGE:CXX>:/permissive->
        $<$<COMPILE_LANGUAGE:CXX>:/Zc:__cplusplus>
        $<$<COMPILE_LANGUAGE:CXX>:/utf-8>
        $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<CONFIG:Release>>:/O2>
        $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<CONFIG:Release>>:/Oy->
    )
else()
    add_compile_options(
        -Wall
        -Wextra
        -Wpedantic
        -Wconversion
        -Wno-unused-parameter
        -fno-omit-frame-pointer
        $<$<CONFIG:Release>:-O3>
    )
endif()

# brass library include directory
set(BRASS_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/include)

# brass core library
file(GLOB_RECURSE BRASS_SOURCES
    "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp"
)

if(MSVC)
    enable_language(ASM_MASM)
    list(APPEND BRASS_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/src/gc/gc_msvc_x64.asm")
endif()

add_library(brass STATIC ${BRASS_SOURCES})

target_include_directories(brass
    PUBLIC
        $<BUILD_INTERFACE:${BRASS_INCLUDE_DIR}>
        $<INSTALL_INTERFACE:include>
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
)

target_compile_features(brass PUBLIC cxx_std_20)

# Tools
add_subdirectory(tools)

# Testing
enable_testing()
add_subdirectory(tests)
