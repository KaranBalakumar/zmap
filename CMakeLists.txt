cmake_minimum_required(VERSION 3.1.0)
project(zmap)

find_package(OpenGL REQUIRED)
find_package(Sophus REQUIRED)
find_package(Pangolin REQUIRED)
find_package(Python3 COMPONENTS Interpreter Development)
find_package(NumPy)

find_package(Matplot++ REQUIRED)

set(CMAKE_BUILD_TYPE "Release")
set(CMAKE_CXX_FLAGS "-O3")

include_directories("/usr/include/eigen3")
include_directories(${Pangolin_INCLUDE_DIRS} ${OpenGL_INCLUDE_DIRS} ${epoxy_INCLUDE_DIRS})

add_executable(zmap zmap.cpp render.hpp)
set_property(TARGET zmap PROPERTY CXX_STANDARD 11)
target_link_libraries(zmap realsense2 glfw ${OPENGL_LIBRARIES})

add_executable(zmap_test zmap_test.cpp /home/karan/librealsense-2.51.1/examples/example.hpp)
set_property(TARGET zmap_test PROPERTY CXX_STANDARD 20)
target_link_libraries(zmap_test realsense2 glfw ${OPENGL_LIBRARIES} ${Pangolin_LIBRARIES} Sophus::Sophus)

add_executable(zmap_pango zmap_pango.cpp /home/karan/librealsense-2.51.1/examples/example.hpp)
set_property(TARGET zmap_pango PROPERTY CXX_STANDARD 20)
target_link_libraries(zmap_pango realsense2 glfw ${OPENGL_LIBRARIES} ${Pangolin_LIBRARIES} Sophus::Sophus)

add_executable(test test.cpp render.hpp)
target_link_libraries(test Matplot++::matplot realsense2 glfw ${OPENGL_LIBRARIES})