cmake_minimum_required(VERSION 3.16)

add_executable(benchmark_undistort benchmark_undistort.cpp)
target_link_libraries(benchmark_undistort PRIVATE ${OpenCV_LIBS})
target_include_directories(benchmark_undistort PRIVATE ${OpenCV_INCLUDE_DIRS})
