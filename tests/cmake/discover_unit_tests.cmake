# Runs inside ctest (via TEST_INCLUDE_FILES) and registers one ctest per
# TEST_CASE in the unit test binary, so a crash or hang fails that one test
# instead of hiding everything registered after it.
#
# Inputs (set by the per-config file tests/CMakeLists.txt generates):
#   BRASS_TEST_EXE      the brass_unit_tests binary
#   BRASS_TEST_WORKDIR  working directory the tests expect (the source root)
#   BRASS_TEST_TIMEOUT  per-test timeout in seconds

if(NOT EXISTS "${BRASS_TEST_EXE}")
    # ctest reports a missing executable as a failed test, which is the
    # honest answer when the suite was never built.
    add_test(unit_tests.not_built "${BRASS_TEST_EXE}")
    return()
endif()

execute_process(
    COMMAND "${BRASS_TEST_EXE}" --list
    WORKING_DIRECTORY "${BRASS_TEST_WORKDIR}"
    OUTPUT_VARIABLE _brass_list
    ERROR_VARIABLE _brass_list_err
    RESULT_VARIABLE _brass_list_rc
)

if(NOT _brass_list_rc EQUAL 0)
    # Re-run the listing as a test so the failure and its stderr show up in
    # the ctest report instead of the suite silently shrinking.
    message(WARNING "brass unit test discovery failed (${_brass_list_rc}): ${_brass_list_err}")
    add_test(unit_tests.discovery "${BRASS_TEST_EXE}" --list)
    set_tests_properties(unit_tests.discovery PROPERTIES
        WORKING_DIRECTORY "${BRASS_TEST_WORKDIR}"
        LABELS "correctness")
    return()
endif()

string(REPLACE "\r" "" _brass_list "${_brass_list}")
string(REPLACE "\n" ";" _brass_names "${_brass_list}")

foreach(_brass_name IN LISTS _brass_names)
    if(_brass_name STREQUAL "")
        continue()
    endif()
    add_test("${_brass_name}" "${BRASS_TEST_EXE}" "--exact=${_brass_name}")
    set_tests_properties("${_brass_name}" PROPERTIES
        WORKING_DIRECTORY "${BRASS_TEST_WORKDIR}"
        LABELS "correctness"
        TIMEOUT "${BRASS_TEST_TIMEOUT}")
endforeach()
