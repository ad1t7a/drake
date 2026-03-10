cmake_minimum_required(VERSION 3.18)
project(gpu_parallel_environments CXX)

# ---------------------------------------------------------------------------
# C++ standard
# ---------------------------------------------------------------------------
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ---------------------------------------------------------------------------
# Compiler flags
# ---------------------------------------------------------------------------
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  add_compile_options(
    -O3
    -Wall
    -Wextra
    -Wno-unused-parameter
    -pthread
  )
endif()

# ---------------------------------------------------------------------------
# Optional: CUDA support
# ---------------------------------------------------------------------------
option(DRAKE_GPU_AVAILABLE "Build with CUDA GPU support" OFF)

if(DRAKE_GPU_AVAILABLE)
  enable_language(CUDA)
  find_package(CUDAToolkit REQUIRED)
  set(CMAKE_CUDA_STANDARD 17)
  set(CMAKE_CUDA_ARCHITECTURES "70;80;86" CACHE STRING "CUDA architectures")
  message(STATUS "GPU build: CUDA architectures = ${CMAKE_CUDA_ARCHITECTURES}")
endif()

# ---------------------------------------------------------------------------
# Library: gpu_parallel_cartpole_env
# ---------------------------------------------------------------------------

set(ENV_SOURCES
    gpu_parallel_cartpole_env.cc
)

if(DRAKE_GPU_AVAILABLE)
  # Include the CUDA kernel source when building for GPU.
  list(APPEND ENV_SOURCES cartpole_step.cu)

  add_library(gpu_parallel_cartpole_env STATIC ${ENV_SOURCES})
  target_compile_definitions(gpu_parallel_cartpole_env
    PUBLIC DRAKE_GPU_AVAILABLE)
  target_link_libraries(gpu_parallel_cartpole_env
    PUBLIC CUDA::cudart)
  set_target_properties(gpu_parallel_cartpole_env PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON)
else()
  add_library(gpu_parallel_cartpole_env STATIC ${ENV_SOURCES})
  target_link_libraries(gpu_parallel_cartpole_env
    PUBLIC pthread)
endif()

# The include root is the parent of the drake source tree, so that
#   #include "drake/gpu_environments/..."  resolves correctly.
# CMAKE_CURRENT_SOURCE_DIR = <drake_root>/gpu_environments
# We need <drake_root>/..  so that "drake/..." includes work.
target_include_directories(gpu_parallel_cartpole_env
  PUBLIC
    "${CMAKE_CURRENT_SOURCE_DIR}/../.."
)

# ---------------------------------------------------------------------------
# Test executable
# ---------------------------------------------------------------------------

add_executable(gpu_parallel_128_test
    test/gpu_parallel_128_test.cc
)

target_link_libraries(gpu_parallel_128_test
  PRIVATE
    gpu_parallel_cartpole_env
    pthread
)

# Enable CTest.
enable_testing()
add_test(NAME gpu_parallel_128_test COMMAND gpu_parallel_128_test)

# ---------------------------------------------------------------------------
# Install (optional)
# ---------------------------------------------------------------------------
install(TARGETS gpu_parallel_cartpole_env
    ARCHIVE DESTINATION lib
)
install(FILES gpu_parallel_cartpole_env.h
    DESTINATION include/drake/gpu_environments
)

# ---------------------------------------------------------------------------
# Status summary
# ---------------------------------------------------------------------------
message(STATUS "")
message(STATUS "=== gpu_parallel_environments build configuration ===")
message(STATUS "  C++ standard : ${CMAKE_CXX_STANDARD}")
message(STATUS "  GPU (CUDA)   : ${DRAKE_GPU_AVAILABLE}")
if(DRAKE_GPU_AVAILABLE)
  message(STATUS "  CUDA archs   : ${CMAKE_CUDA_ARCHITECTURES}")
endif()
message(STATUS "=====================================================")
