project(Core VERSION 0.1)

FetchContent_Declare(
    vk-bootstrap
    GIT_REPOSITORY https://github.com/charles-lunarg/vk-bootstrap.git
    GIT_TAG        b934b6c76cf7c3b7bef76c0ddd43a0996f8b9f92
)
FetchContent_MakeAvailable(vk-bootstrap)

FetchContent_Declare(
    glfw
    GIT_REPOSITORY https://github.com/glfw/glfw.git
    GIT_TAG        3.4
)
FetchContent_MakeAvailable(glfw)

FetchContent_Declare(
    volk
    GIT_REPOSITORY https://github.com/zeux/volk.git
    GIT_TAG 0b17a763ba5643e32da1b2152f8140461b3b7345
)
FetchContent_MakeAvailable(volk)

find_package(Vulkan REQUIRED)

file(GLOB_RECURSE SOURCES CONFIGURE_DEPENDS ${PROJECT_SOURCE_DIR}/Source/*.cpp)

add_library(core STATIC ${SOURCES})

target_precompile_headers(core PRIVATE
  ${PROJECT_SOURCE_DIR}/Include/Core/pch.h
)

target_compile_definitions(core PRIVATE 
  SHADER_DIR="${PROJECT_SOURCE_DIR}/Source/Render/Shaders"
)

target_compile_features(core PUBLIC cxx_std_20)

add_subdirectory(ThirdParty)

target_include_directories(core PUBLIC 
    ${PROJECT_SOURCE_DIR}/Include
    ${Vulkan_INCLUDE_DIR}
)

target_link_libraries(core PUBLIC  
    ${Vulkan_LIBRARY} 
    volk::volk 
    glfw
    vk-bootstrap::vk-bootstrap
    mathlib
)
