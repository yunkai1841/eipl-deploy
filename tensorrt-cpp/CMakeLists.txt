cmake_minimum_required(VERSION 3.18)
project(tensorrt-cpp)

# use cuda opencv and tensorrt
find_package(OpenCV REQUIRED)
find_package(CUDA REQUIRED)
find_package(gflags REQUIRED)

set(CUDASM 89)
set(CMAKE_CXX_STANDARD 14)

set(CMAKE_CXX_FLAGS_RELEASE "-std=c++14 -Wextra -Wall -Wno-missing-field-initializers -Wno-deprecated-declarations -O3")
set(CMAKE_CXX_FLAGS_DEBUG   "-std=c++14 -O0 -g")

set(CUDA_NVCC_FLAGS_RELEASE "-Werror=all-warnings -Xcompiler -std=c++14,-Wextra,-Wall,-Wno-deprecated-declarations,-O3")
set(CUDA_NVCC_FLAGS_DEBUG   "-std=c++14 -O0 -g")

set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} 
    -gencode arch=compute_${CUDASM},code=compute_${CUDASM}
)

include_directories(
    ${OpenCV_INCLUDE_DIRS}
    ${CUDA_INCLUDE_DIRS}
)

# file (GLOB_RECURSE SRC_FILES src/**/*.cpp src/**/*.cu src/**/*.h)
file (GLOB_RECURSE SRC_FILES src/*.cpp src/*.cu src/*.h src/common/*.h src/common/*.cpp)
add_executable(tensorrt-cpp ${SRC_FILES})

target_link_libraries(tensorrt-cpp
    ${OpenCV_LIBS}
    ${CUDA_LIBRARIES}
    gflags
    nvinfer
    nvinfer_plugin
    nvparsers
)

