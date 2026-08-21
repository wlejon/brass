# Tests CMakeLists.txt

add_executable(brass_unit_tests
    unit/test_sanity.cpp
    unit/test_arena.cpp
    unit/test_core_utils.cpp
    unit/test_mir_builder.cpp
    unit/test_mir_verifier.cpp
    unit/test_mir_printer_parser.cpp
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
    differential/fuzz_generator.cpp
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

# Benchmarks
add_executable(brass_benchmarks
    benchmarks/benchmark_suite.cpp
    benchmarks/bench_numeric.cpp
    benchmarks/bench_gc.cpp
    benchmarks/bench_compile_speed.cpp
)

target_include_directories(brass_benchmarks
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/benchmarks
        ${BRASS_INCLUDE_DIR}
)

target_link_libraries(brass_benchmarks
    PRIVATE
        brass
)

add_test(NAME benchmarks COMMAND brass_benchmarks)
