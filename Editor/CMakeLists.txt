project(Editor VERSION 0.1)

file(GLOB_RECURSE SOURCES CONFIGURE_DEPENDS ${PROJECT_SOURCE_DIR}/Source/*.cpp)

add_executable(editor ${SOURCES})

target_compile_definitions(editor PRIVATE 
    CURRENT_WORKING_DIRECTORY="${PROJECT_SOURCE_DIR}"
)

target_include_directories(editor PUBLIC 
    ${PROJECT_SOURCE_DIR}/Source
)

FetchContent_Declare(
    imgui
    GIT_REPOSITORY https://github.com/ocornut/imgui.git
    GIT_TAG 8936b58fe26e8c3da834b8f60b06511d537b4c63
)
FetchContent_MakeAvailable(imgui)

file(GLOB IMGUI_SOURCES
    ${imgui_SOURCE_DIR}/*.cpp
    ${imgui_SOURCE_DIR}/backends/imgui_impl_glfw.cpp
    ${imgui_SOURCE_DIR}/backends/imgui_impl_vulkan.cpp
)

add_library(imgui STATIC ${IMGUI_SOURCES})

target_include_directories(imgui PUBLIC
    ${imgui_SOURCE_DIR}
    ${imgui_SOURCE_DIR}/backends
    ${Vulkan_INCLUDE_DIR}
)

target_link_libraries(imgui PUBLIC glfw volk::volk)
target_compile_definitions(imgui PRIVATE IMGUI_IMPL_VULKAN_USE_VOLK)

target_link_libraries(editor PRIVATE 
    core
    mathlib
    imgui
)
