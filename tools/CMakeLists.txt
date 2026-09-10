# Tools CMakeLists.txt

add_executable(brass-opt
    brass-opt/main.cpp
    brass-opt/opt_actions.cpp
)

target_include_directories(brass-opt
    PRIVATE
        ${BRASS_INCLUDE_DIR}
)

target_link_libraries(brass-opt
    PRIVATE
        brass
)

add_executable(brass-il
    brass-il/main.cpp
)

target_include_directories(brass-il
    PRIVATE
        ${BRASS_INCLUDE_DIR}
)

target_link_libraries(brass-il
    PRIVATE
        brass
)

add_executable(brass-fuzz
    brass-fuzz/main.cpp
)

target_include_directories(brass-fuzz
    PRIVATE
        ${BRASS_INCLUDE_DIR}
)

target_link_libraries(brass-fuzz
    PRIVATE
        brass
)
