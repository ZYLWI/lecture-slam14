cmake_minimum_required(VERSION 3.12)
project(code)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_BUILD_TYPE "Release")
set(CMAKE_CXX_FLAGS "-std=c++11")

include_directories("/usr/include/eigen3")

find_package(Pangolin REQUIRED)
include_directories(${Pangolin_INCLUDE_DIRS})

find_package(Sophus REQUIRED)
include_directories(${Sophus_INCLUDE_DIRS})

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(code main.cpp)

target_link_libraries(code ${Pangolin_LIBRARIES} ${Sophus_LIBRARIES} ${OpenCV_LIBRARIES})