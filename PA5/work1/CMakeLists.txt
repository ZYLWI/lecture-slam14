cmake_minimum_required(VERSION 3.12)
project(work1)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_BUILD_TYPE "Release")
set(CMAKE_CXX_FLAGS "-std=c++11")

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(work1 main.cpp)

target_link_libraries(work1 ${OpenCV_LIBS})