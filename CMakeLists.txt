cmake_minimum_required(VERSION 3.1.0)
project(zmap)

set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin)
set(CMAKE_BUILD_TYPE "Release")
set(CMAKE_CXX_FLAGS "-O3")

include_directories("/usr/include/eigen3")

find_package(realsense2 REQUIRED)
include_directories(${realsense2_INCLUDE_DIR})

# add_subdirectory(rerun_cpp_sdk) # using 0.22.1
include(FetchContent)
FetchContent_Declare(rerun_sdk URL
    https://github.com/rerun-io/rerun/releases/download/0.22.1/rerun_cpp_sdk.zip)
FetchContent_MakeAvailable(rerun_sdk)
add_subdirectory(src)