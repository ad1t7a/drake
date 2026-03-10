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

# Common include root: parent of drake/ so that
#   #include "drake/gpu_environments/..."  resolves correctly.
set(ENV_INCLUDE_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../..")

# ---------------------------------------------------------------------------
# Library: gpu_environments  (CartPole + Pendulum + shared infrastructure)
# ---------------------------------------------------------------------------

set(ENV_SOURCES
    gpu_parallel_cartpole_env.cc
    gpu_parallel_pendulum_env.cc
)

if(DRAKE_GPU_AVAILABLE)
  list(APPEND ENV_SOURCES cartpole_step.cu pendulum_step.cu)

  add_library(gpu_environments STATIC ${ENV_SOURCES})
  target_compile_definitions(gpu_environments PUBLIC DRAKE_GPU_AVAILABLE)
  target_link_libraries(gpu_environments PUBLIC CUDA::cudart)
  set_target_properties(gpu_environments PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON)
else()
  add_library(gpu_environments STATIC ${ENV_SOURCES})
  target_link_libraries(gpu_environments PUBLIC pthread)
endif()

target_include_directories(gpu_environments PUBLIC "${ENV_INCLUDE_ROOT}")

# Backwards-compat alias so existing CMake targets linking
# gpu_parallel_cartpole_env still work.
add_library(gpu_parallel_cartpole_env ALIAS gpu_environments)

# ---------------------------------------------------------------------------
# Test executables
# ---------------------------------------------------------------------------

# CartPole 128-environment test.
add_executable(gpu_parallel_128_test
    test/gpu_parallel_128_test.cc)
target_link_libraries(gpu_parallel_128_test PRIVATE gpu_environments pthread)

# Pendulum 128-environment test.
add_executable(gpu_parallel_pendulum_128_test
    test/gpu_parallel_pendulum_128_test.cc)
target_link_libraries(gpu_parallel_pendulum_128_test
    PRIVATE gpu_environments pthread)

# CTest registration.
enable_testing()
add_test(NAME gpu_parallel_128_test         COMMAND gpu_parallel_128_test)
add_test(NAME gpu_parallel_pendulum_128_test COMMAND gpu_parallel_pendulum_128_test)

# ---------------------------------------------------------------------------
# Install (optional)
# ---------------------------------------------------------------------------
install(TARGETS gpu_environments ARCHIVE DESTINATION lib)
install(FILES
    gpu_env_common.h
    gpu_parallel_cartpole_env.h
    gpu_parallel_pendulum_env.h
    DESTINATION include/drake/gpu_environments
)

# ---------------------------------------------------------------------------
# Status summary
# ---------------------------------------------------------------------------
message(STATUS "")
message(STATUS "=== gpu_parallel_environments build configuration ===")
message(STATUS "  C++ standard    : ${CMAKE_CXX_STANDARD}")
message(STATUS "  GPU (CUDA)      : ${DRAKE_GPU_AVAILABLE}")
message(STATUS "  Environments    : CartPole, Pendulum")
if(DRAKE_GPU_AVAILABLE)
  message(STATUS "  CUDA archs      : ${CMAKE_CUDA_ARCHITECTURES}")
endif()
message(STATUS "=====================================================")
