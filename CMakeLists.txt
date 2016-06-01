
# littleCNN
# A Convolutinal Neural Network library.
#
# Usage:
#  cmake .

cmake_minimum_required(VERSION 2.8)

set(CMAKE_BUILD_TYPE Debug)

set(sources network/network.cpp)
set(sources ${sources} network/util.cpp)
set(sources ${sources} network/activation/activation.cpp)
set(sources ${sources} network/layer/fully_connected.cpp)
set(sources ${sources} network/layer/convolution.cpp)
set(sources ${sources} network/layer/max_pooling.cpp)

include_directories(network network/activation network/layer)

set(sources ${sources} samples/logic/main_logic.cpp)

add_executable(myapp ${sources})
