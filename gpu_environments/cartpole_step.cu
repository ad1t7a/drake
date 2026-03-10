/// @file cartpole_step.cu
///
/// CUDA kernels for the GPU parallel CartPole environment.
///
/// Kernel design
/// =============
/// Each thread handles exactly one environment.  For N=128 we dispatch a
/// single block of 128 threads.  For larger N we use ceil(N/256) blocks of
/// 256 threads each (tunable via kBlockSize below).
///
/// The state is stored in Structure-of-Arrays (SoA) format in device memory.
/// All threads read/write independent array elements → no shared-memory
/// conflicts; coalesced global-memory access for best bandwidth.
///
/// Compilation
/// ===========
///   nvcc -O3 -arch=sm_70 -std=c++17 \
///        -I$(DRAKE_ROOT) \
///        cartpole_step.cu -c -o cartpole_step.cu.o

#include <cuda_runtime.h>
#include <stdint.h>

// Pull in the physics constants and CartpoleDynamicsStep.
// NOTE: CARTPOLE_CALLABLE expands to __device__ __host__ when compiled by
// nvcc because DRAKE_GPU_AVAILABLE is set during GPU builds.
#define DRAKE_GPU_AVAILABLE
#include "drake/gpu_environments/gpu_parallel_cartpole_env.h"

namespace {

// ---------------------------------------------------------------------------
// Tuning constant
// ---------------------------------------------------------------------------
constexpr int kBlockSize = 256;

// ---------------------------------------------------------------------------
// xorshift32 on the device
// ---------------------------------------------------------------------------
__device__ inline float dev_xorshift_float(uint32_t& state, float scale) {
  state ^= state << 13;
  state ^= state >> 17;
  state ^= state << 5;
  float r = static_cast<float>(state) /
            static_cast<float>(0xFFFFFFFFu);
  return (r * 2.0f - 1.0f) * scale;
}

// ---------------------------------------------------------------------------
// cartpole_reset_kernel
/// Each thread (re-)initialises one environment using xorshift32.
// ---------------------------------------------------------------------------
__global__ void cartpole_reset_kernel(float* __restrict__ x,
                                      float* __restrict__ xdot,
                                      float* __restrict__ theta,
                                      float* __restrict__ tdot,
                                      uint32_t* __restrict__ seeds,
                                      int num_envs) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= num_envs) return;

  uint32_t s = seeds[i];
  if (s == 0) s = 0xBEEFu + static_cast<uint32_t>(i);

  using namespace drake::gpu_environments;
  x[i]     = dev_xorshift_float(s, kInitRange);
  xdot[i]  = dev_xorshift_float(s, kInitRange);
  theta[i] = dev_xorshift_float(s, kInitRange);
  tdot[i]  = dev_xorshift_float(s, kInitRange);

  seeds[i] = s;  // Write back advanced seed for auto-reset.
}

// ---------------------------------------------------------------------------
// cartpole_step_kernel
/// Each thread advances one environment by one time step.
///
/// Auto-reset: if the environment terminates (done=1) the thread immediately
/// re-initialises it in-place using the per-environment xorshift seed.
// ---------------------------------------------------------------------------
__global__ void cartpole_step_kernel(float* __restrict__ x,
                                     float* __restrict__ xdot,
                                     float* __restrict__ theta,
                                     float* __restrict__ tdot,
                                     const float* __restrict__ actions,
                                     float* __restrict__ rewards,
                                     uint8_t* __restrict__ done,
                                     uint32_t* __restrict__ seeds,
                                     int num_envs) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= num_envs) return;

  using namespace drake::gpu_environments;

  // Load state into registers.
  float xi     = x[i];
  float xdoti  = xdot[i];
  float thetai = theta[i];
  float tdoti  = tdot[i];
  float ai     = actions[i];

  // Apply dynamics.
  CartpoleDynamicsStep(xi, xdoti, thetai, tdoti, ai);

  const bool env_done = (xi < -kMaxCartX     || xi > kMaxCartX) ||
                        (thetai < -kMaxPoleAngle || thetai > kMaxPoleAngle);

  rewards[i] = env_done ? 0.0f : 1.0f;
  done[i]    = env_done ? 1u : 0u;

  if (env_done) {
    // In-place auto-reset.
    uint32_t s = seeds[i];
    if (s == 0) s = 0xBEEFu + static_cast<uint32_t>(i);
    xi     = dev_xorshift_float(s, kInitRange);
    xdoti  = dev_xorshift_float(s, kInitRange);
    thetai = dev_xorshift_float(s, kInitRange);
    tdoti  = dev_xorshift_float(s, kInitRange);
    seeds[i] = s;
  }

  // Write updated state.
  x[i]     = xi;
  xdot[i]  = xdoti;
  theta[i] = thetai;
  tdot[i]  = tdoti;
}

}  // anonymous namespace

// ---------------------------------------------------------------------------
// Launcher functions (called from gpu_parallel_cartpole_env.cc)
// ---------------------------------------------------------------------------

void LaunchCartpoleResetKernel(float* d_x, float* d_xdot,
                               float* d_theta, float* d_tdot,
                               uint32_t* d_seeds,
                               int num_envs,
                               cudaStream_t stream) {
  const int blocks = (num_envs + kBlockSize - 1) / kBlockSize;
  cartpole_reset_kernel<<<blocks, kBlockSize, 0, stream>>>(
      d_x, d_xdot, d_theta, d_tdot, d_seeds, num_envs);
}

void LaunchCartpoleStepKernel(float* d_x, float* d_xdot,
                              float* d_theta, float* d_tdot,
                              const float* d_actions,
                              float* d_rewards, uint8_t* d_done,
                              uint32_t* d_seeds,
                              int num_envs,
                              cudaStream_t stream) {
  const int blocks = (num_envs + kBlockSize - 1) / kBlockSize;
  cartpole_step_kernel<<<blocks, kBlockSize, 0, stream>>>(
      d_x, d_xdot, d_theta, d_tdot,
      d_actions, d_rewards, d_done, d_seeds, num_envs);
}
