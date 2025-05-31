cmake_minimum_required(VERSION 3.1.0)
project(zmap)

set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin)
set(CMAKE_BUILD_TYPE "Release")
set(CMAKE_CXX_FLAGS "-O3")

include_directories("/usr/include/eigen3")

add_subdirectory(rerun_cpp_sdk) # using 0.22.1
add_subdirectory(src)