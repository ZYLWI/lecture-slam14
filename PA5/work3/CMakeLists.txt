cmake_minimum_required(VERSION 3.12)
project(code)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_BUILD_TYPE "Realease")
set(CMAKE_CXX_FLAGS "-std=c++11")

find_package("/usr/include/eigen3")

find_package(Sophus REQUIRED)
include_directories(${Sophus_INCLUDE_DIRS})

add_executable(code main.cpp)

target_link_libraries(code ${Sophus_LIBRARIES})