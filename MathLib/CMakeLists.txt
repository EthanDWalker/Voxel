if (DEFINED TEST) 
cmake_minimum_required(VERSION 3.20)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
endif()

project(MathLib VERSION 0.1)

add_library(mathlib INTERFACE)

target_compile_features(mathlib INTERFACE cxx_std_20)

target_include_directories(mathlib INTERFACE
    $<BUILD_INTERFACE:${${PROJECT_SOURCE_DIR}/Include}>
    $<INSTALL_INTERFACE:${${PROJECT_SOURCE_DIR}/Include}>
)

target_precompile_headers(mathlib INTERFACE
  <cstdint>
  "${PROJECT_SOURCE_DIR}/Include/Math/pch.h"
)

if (DEFINED TEST) 
add_executable(
  test ${PROJECT_SOURCE_DIR}/test.cpp
)
target_include_directories(test PUBLIC mathlib)
endif()
