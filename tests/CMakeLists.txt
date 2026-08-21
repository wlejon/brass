# Tests CMakeLists.txt

add_executable(brass_unit_tests
    unit/test_sanity.cpp
    unit/test_arena.cpp
    unit/test_core_utils.cpp
    unit/test_mir_builder.cpp
    unit/test_mir_verifier.cpp
    unit/test_mir_printer_parser.cpp
    unit/test_mir_loop_opt.cpp
    unit/test_mini_cheney.cpp
    unit/test_interpreter.cpp
    unit/test_target_calling_conv.cpp
    unit/test_x64_encoder.cpp
    unit/test_lir_isel.cpp
    unit/test_linear_scan.cpp
    unit/test_peephole.cpp
    unit/test_x64_codegen.cpp
    unit/test_coff_writer.cpp
    unit/test_elf_writer.cpp
    unit/test_stack_map.cpp
    unit/test_stack_walker.cpp
    unit/test_deopt_resume.cpp
    unit/test_patching_concurrency.cpp
    unit/test_determinism_ratchet.cpp
    differential/test_differential_exec.cpp
    differential/test_moving_gc_native.cpp
    differential/test_speculation_differential.cpp
    differential/test_fuzz_loops.cpp
    differential/test_fuzz_gc.cpp
    differential/test_fuzz_speculation.cpp
    differential/test_fuzz_memory.cpp
    differential/test_fuzz_patching.cpp
    differential/test_fuzz_matrix.cpp
    differential/test_fuzz_js_shapes.cpp
    differential/test_fuzz_optimizations.cpp
    differential/fuzz_generator.cpp
    benchmarks/bench_numeric_modules.cpp
    unit/test_msvc_milestone.cpp
)

target_include_directories(brass_unit_tests
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/harness
        ${CMAKE_CURRENT_SOURCE_DIR}/differential
        ${BRASS_INCLUDE_DIR}
)

target_link_libraries(brass_unit_tests
    PRIVATE
        brass
)

add_test(NAME unit_tests COMMAND brass_unit_tests)

# Retrieve Git Commit SHA
execute_process(
    COMMAND git rev-parse --short HEAD
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    OUTPUT_VARIABLE BRASS_GIT_COMMIT_HASH
    OUTPUT_STRIP_TRAILING_WHITESPACE
    ERROR_QUIET
)
if(NOT BRASS_GIT_COMMIT_HASH)
    set(BRASS_GIT_COMMIT_HASH "unknown")
endif()

# Benchmarks
add_executable(brass_benchmarks
    benchmarks/benchmark_suite.cpp
    benchmarks/bench_numeric.cpp
    benchmarks/bench_numeric_modules.cpp
    benchmarks/bench_js_shapes.cpp
    benchmarks/bench_js_shapes_modules.cpp
    benchmarks/bench_gc.cpp
    benchmarks/bench_compile_speed.cpp
)

target_include_directories(brass_benchmarks
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/benchmarks
        ${BRASS_INCLUDE_DIR}
)

target_compile_definitions(brass_benchmarks
    PRIVATE
        BRASS_GIT_COMMIT="${BRASS_GIT_COMMIT_HASH}"
        BRASS_BUILD_TYPE="${CMAKE_BUILD_TYPE}"
        BRASS_BENCH_RATCHET_PATH="${CMAKE_CURRENT_SOURCE_DIR}/../bench/ratchet.json"
)

target_link_libraries(brass_benchmarks
    PRIVATE
        brass
)

add_test(NAME benchmarks COMMAND brass_benchmarks)
add_test(NAME performance_ratchet COMMAND brass_benchmarks --check-ratchet)

