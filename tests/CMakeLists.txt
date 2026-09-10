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
    unit/test_embedding_api.cpp
    unit/test_aot_embedding.cpp
    unit/test_mir_unroll_select.cpp
    unit/test_il_expressibility.cpp
    differential/test_embedding_differential.cpp
    unit/test_bronze_il_translator.cpp
    differential/test_fuzz_closures_gc.cpp
    differential/test_fuzz_js_loops.cpp
    unit/test_f64_demote.cpp
    unit/test_inliner.cpp
    differential/test_inlining_differential.cpp
    unit/test_simd_mir.cpp
    differential/test_simd_differential.cpp
    unit/test_vectorizer.cpp
    differential/test_vectorizer_differential.cpp
    unit/test_sroa.cpp
    differential/test_sroa_differential.cpp
    unit/test_gvn.cpp
    differential/test_gvn_differential.cpp
    unit/test_pe_dll_writer.cpp
    unit/test_elf_so_writer.cpp
    unit/test_aot_shared_library.cpp
    unit/test_loop_tile.cpp
    differential/test_loop_tile_differential.cpp
    unit/test_sccp.cpp
    differential/test_sccp_differential.cpp
    unit/test_loop_unswitch.cpp
    unit/test_jump_threading.cpp
    unit/test_block_layout.cpp
    differential/test_unswitch_differential.cpp
    unit/test_profile_serialization.cpp
    unit/test_pgo_instrument.cpp
    unit/test_branch_probability.cpp
    differential/test_pgo_differential.cpp
    unit/test_sched_dag.cpp
    unit/test_instruction_scheduler.cpp
    unit/test_software_pipeline.cpp
    differential/test_sched_differential.cpp
    unit/test_source_loc.cpp
    unit/test_source_map.cpp
    unit/test_symbolicator.cpp
    differential/test_debug_differential.cpp
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
else()
    execute_process(
        COMMAND git status --porcelain
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
        OUTPUT_VARIABLE BRASS_GIT_STATUS
        OUTPUT_STRIP_TRAILING_WHITESPACE
        ERROR_QUIET
    )
    if(BRASS_GIT_STATUS)
        set(BRASS_GIT_COMMIT_HASH "${BRASS_GIT_COMMIT_HASH}-dirty")
    endif()
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
    benchmarks/bench_simd_math.cpp
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

