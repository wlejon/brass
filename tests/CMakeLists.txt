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
    unit/test_x64_codegen.cpp
)

target_include_directories(brass_unit_tests
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/harness
        ${BRASS_INCLUDE_DIR}
)

target_link_libraries(brass_unit_tests
    PRIVATE
        brass
)

add_test(NAME unit_tests COMMAND brass_unit_tests)
