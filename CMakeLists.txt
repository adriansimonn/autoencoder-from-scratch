cmake_minimum_required(VERSION 3.16)
project(autoencoder-from-scratch LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Include paths
include_directories(${CMAKE_SOURCE_DIR}/src)
include_directories(${CMAKE_SOURCE_DIR}/third_party)

# Tensor library
add_library(tensor src/math/tensor.cpp)

# Neural network layers
add_library(nn
    src/nn/dense.cpp
    src/nn/relu.cpp
    src/nn/sigmoid.cpp
    src/nn/mse_loss.cpp
    src/nn/network.cpp
)
target_link_libraries(nn tensor)

# Optimizer
add_library(optim src/optim/adam.cpp)
target_link_libraries(optim nn)

# I/O
add_library(io
    src/io/image_io.cpp
    src/io/model_io.cpp
)
target_link_libraries(io nn)

# Autoencoder model
add_library(autoencoder src/models/autoencoder.cpp)
target_link_libraries(autoencoder nn)

# Executables
add_executable(train src/train_main.cpp)
target_link_libraries(train autoencoder optim io)

add_executable(reconstruct src/reconstruct_main.cpp)
target_link_libraries(reconstruct autoencoder nn io)

# Testing
enable_testing()

add_executable(test_tensor test/test_tensor.cpp)
target_link_libraries(test_tensor tensor)
add_test(NAME test_tensor COMMAND test_tensor)

add_executable(test_dense test/test_dense.cpp)
target_link_libraries(test_dense nn)
add_test(NAME test_dense COMMAND test_dense)

add_executable(test_activations test/test_activations.cpp)
target_link_libraries(test_activations nn)
add_test(NAME test_activations COMMAND test_activations)

add_executable(test_network test/test_network.cpp)
target_link_libraries(test_network nn optim io)
add_test(NAME test_network COMMAND test_network)
