cmake_minimum_required(VERSION 4.0)

project(Engine VERSION 0.1)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

include(FetchContent)

FetchContent_Declare(
    tracy
    GIT_REPOSITORY https://github.com/wolfpld/tracy.git
    GIT_TAG 05cceee0df3b8d7c6fa87e9638af311dbabc63cb
    GIT_SHALLOW TRUE
)

set(TRACY_ENABLE ON CACHE BOOL "" FORCE)
set(TRACY_ON_DEMAND ON CACHE BOOL "" FORCE)

FetchContent_MakeAvailable(tracy)

add_subdirectory(MathLib)
add_subdirectory(Core)
add_subdirectory(Editor)
