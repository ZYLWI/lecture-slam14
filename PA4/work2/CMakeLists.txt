cmake_minimum_required(VERSION 3.12)

project(code)

set(CMAKE_CXX_FLAGS "-std=c++11")
set(CMAKE_BUILD_TYPE "Release")
set(CMAKE_CXX_STANDARD 14)

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

find_package(Pangolin)
include_directories(${Pangolin_INCLUDE_DIRS})

include_directories("/usr/include/eigen3")

add_executable(code main.cpp)

target_link_libraries(code ${OpenCV_LIBS} ${Pangolin_LIBRARIES})