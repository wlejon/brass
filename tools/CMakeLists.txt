# Tools CMakeLists.txt

add_executable(brass-opt
    brass-opt/main.cpp
)

target_include_directories(brass-opt
    PRIVATE
        ${BRASS_INCLUDE_DIR}
)

target_link_libraries(brass-opt
    PRIVATE
        brass
)
