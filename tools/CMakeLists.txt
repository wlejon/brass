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

if(APPLE)
    add_custom_command(TARGET brass-opt POST_BUILD
        COMMAND codesign -s - -f $<TARGET_FILE:brass-opt>
        COMMENT "Ad-hoc code signing brass-opt"
    )
    add_custom_command(TARGET brass-il POST_BUILD
        COMMAND codesign -s - -f $<TARGET_FILE:brass-il>
        COMMENT "Ad-hoc code signing brass-il"
    )
    add_custom_command(TARGET brass-fuzz POST_BUILD
        COMMAND codesign -s - -f $<TARGET_FILE:brass-fuzz>
        COMMENT "Ad-hoc code signing brass-fuzz"
    )
endif()
