project(Editor VERSION 0.1)

file(GLOB_RECURSE SOURCES CONFIGURE_DEPENDS ${PROJECT_SOURCE_DIR}/Source/*.cpp)

add_executable(editor ${SOURCES})

target_compile_definitions(editor PRIVATE 
    CURRENT_WORKING_DIRECTORY="${PROJECT_SOURCE_DIR}"
)

target_include_directories(editor PUBLIC 
    ${PROJECT_SOURCE_DIR}/Source
)

target_link_libraries(editor PRIVATE 
    core
    mathlib
)
