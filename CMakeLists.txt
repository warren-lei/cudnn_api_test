project(cudnn_test)

cmake_minimum_required(VERSION 3.10)

set(CMAKE_CXX_STANDARD 14)

set(CUDA_LIB_PATH /usr/local/cuda-10.0/lib64)
set(CUDNN_LIB_PATH /usr/lib/x86_64-linux-gnu)

# add the executable
add_executable(cudnn_test test.cc)

target_link_libraries(cudnn_test PUBLIC ${CUDA_LIB_PATH}/libcudart.so)
target_link_libraries(cudnn_test PUBLIC ${CUDNN_LIB_PATH}/libcudnn.so.8)
target_include_directories(cudnn_test PUBLIC /usr/local/cuda-10.0/include)
