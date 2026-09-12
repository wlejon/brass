cmake_minimum_required(VERSION 3.20)


project(brass VERSION 0.1.0 LANGUAGES CXX C)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Export compile commands for language servers
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Default to Release if not specified
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Choose build type: Debug Release RelWithDebInfo MinSizeRel." FORCE)
endif()

# Compiler flags and warnings
if(MSVC)
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
    if(WIN32 OR MINGW)
        add_link_options(-static -static-libgcc -static-libstdc++)
    elseif(NOT APPLE)
        add_link_options(-static-libgcc -static-libstdc++)
    endif()
endif()

# brass library include directory
set(BRASS_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/include)

# brass core library
file(GLOB_RECURSE BRASS_SOURCES
    "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp"
)
list(APPEND BRASS_SOURCES
    "${CMAKE_CURRENT_SOURCE_DIR}/src/object/macho_writer.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/target/macho_dylib_writer.cpp"
)
list(REMOVE_DUPLICATES BRASS_SOURCES)

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

# brass shared library (libbrass)
add_library(brass_shared SHARED ${BRASS_SOURCES})

target_include_directories(brass_shared
    PUBLIC
        $<BUILD_INTERFACE:${BRASS_INCLUDE_DIR}>
        $<INSTALL_INTERFACE:include>
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
)

target_compile_features(brass_shared PUBLIC cxx_std_20)

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

# Installation
install(TARGETS brass brass_shared
    ARCHIVE DESTINATION lib
    LIBRARY DESTINATION lib
    RUNTIME DESTINATION bin
)
install(FILES ${BRASS_INCLUDE_DIR}/brass/brass_c_api.h DESTINATION include/brass)

# Tools
add_subdirectory(tools)

# Testing
enable_testing()
add_subdirectory(tests)
