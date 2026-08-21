# CMake generated Testfile for 
# Source directory: D:/projects/brass/tests
# Build directory: D:/projects/brass/build-o2/tests
# 
# This file includes the relevant testing commands required for 
# testing this directory and lists subdirectories to be tested as well.
add_test([=[unit_tests]=] "D:/projects/brass/build-o2/tests/brass_unit_tests.exe")
set_tests_properties([=[unit_tests]=] PROPERTIES  _BACKTRACE_TRIPLES "D:/projects/brass/tests/CMakeLists.txt;41;add_test;D:/projects/brass/tests/CMakeLists.txt;0;")
add_test([=[benchmarks]=] "D:/projects/brass/build-o2/tests/brass_benchmarks.exe")
set_tests_properties([=[benchmarks]=] PROPERTIES  _BACKTRACE_TRIPLES "D:/projects/brass/tests/CMakeLists.txt;59;add_test;D:/projects/brass/tests/CMakeLists.txt;0;")
