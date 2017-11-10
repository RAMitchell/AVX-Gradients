cmake_minimum_required (VERSION 2.6)
project (AVX-Gradients)
if(MSVC)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /arch:AVX2")
else()
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mavx2 -mfma -std=c++11")
endif()

add_executable(AVX-Gradients src/avx_gradients.cc)

