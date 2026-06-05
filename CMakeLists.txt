cmake_minimum_required(VERSION 3.16)
project(EmbodiedVisionToolkit LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# ──────────────────────────────────────────────
# Dependencies
# ──────────────────────────────────────────────
find_package(OpenCV REQUIRED)
find_package(Eigen3 REQUIRED)

# ──────────────────────────────────────────────
# Compiler warnings
# ──────────────────────────────────────────────
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    add_compile_options(-Wall -Wextra -Wpedantic)
elseif(MSVC)
    add_compile_options(/W4)
endif()

# ──────────────────────────────────────────────
# Module 1: Calibration & Stereo
# ──────────────────────────────────────────────

# fisheye_undistort — KB 等距模型去畸变
add_executable(fisheye_undistort
    01_Calibration_and_Stereo/fisheye_undistort.cpp
)
target_link_libraries(fisheye_undistort PRIVATE ${OpenCV_LIBS})
target_include_directories(fisheye_undistort PRIVATE ${OpenCV_INCLUDE_DIRS})

# baseline_eigen — 齐次变换基线解算
add_executable(baseline_eigen
    01_Calibration_and_Stereo/baseline_eigen.cpp
)
target_link_libraries(baseline_eigen PRIVATE Eigen3::Eigen)

# stereo_pointcloud — SGBM 立体匹配点云
add_executable(stereo_pointcloud
    01_Calibration_and_Stereo/stereo_pointcloud.cpp
)
target_link_libraries(stereo_pointcloud PRIVATE ${OpenCV_LIBS})
target_include_directories(stereo_pointcloud PRIVATE ${OpenCV_INCLUDE_DIRS})

# double_sphere_undistort — Double Sphere 模型去畸变
add_executable(double_sphere_undistort
    01_Calibration_and_Stereo/double_sphere_undistort.cpp
)
target_link_libraries(double_sphere_undistort PRIVATE ${OpenCV_LIBS} Eigen3::Eigen)
target_include_directories(double_sphere_undistort PRIVATE ${OpenCV_INCLUDE_DIRS})

# ──────────────────────────────────────────────
# Benchmark (optional)
# ──────────────────────────────────────────────
option(BUILD_BENCHMARKS "Build benchmark executables" OFF)
if(BUILD_BENCHMARKS)
    add_subdirectory(benchmarks)
endif()
