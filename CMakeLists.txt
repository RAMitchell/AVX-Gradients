cmake_minimum_required (VERSION 2.6)
project (AVX-Gradients)
if(MSVC)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /arch:AVX /openmp")
else()
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mavx -mfma -std=c++11 -fopenmp")
endif()

add_executable(AVX-Gradients src/avx_gradients.cc src/avx_helpers.h)

