cmake_minimum_required(VERSION 3.18)
project(CudaPanorama VERSION 1.0 LANGUAGES CXX CUDA)

# Require C++17
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Find OpenCV (Required for image I/O and CPU side validation)
find_package(OpenCV REQUIRED)

# Include directories (where to look for .h / .cuh files)
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${OpenCV_INCLUDE_DIRS}
)

# Collect all source files
file(GLOB_RECURSE SOURCES "src/*.cpp" "src/*.cu")

# Create the executable
add_executable(panorama_engine ${SOURCES})

# Link OpenCV and CUDA libraries
target_link_libraries(panorama_engine
    ${OpenCV_LIBS}
)

# Optional: Set CUDA architecture for Colab GPUs (T4 is sm_75, V100 is sm_70)
set_target_properties(panorama_engine PROPERTIES CUDA_ARCHITECTURES "70;75")
