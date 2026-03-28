cmake_minimum_required(VERSION 4.0)

project(Engine VERSION 0.1)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

include(FetchContent)

add_subdirectory(MathLib)
add_subdirectory(Core)
add_subdirectory(Editor)
