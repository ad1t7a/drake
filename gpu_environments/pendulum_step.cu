/// @file pendulum_step.cu
///
/// CUDA kernels for the GPU parallel Pendulum environment.
///
/// Kernel design: same as cartpole_step.cu — one thread per environment.
/// For N=128: one block of 128 threads (or ceil(N/256) blocks of 256 each).

#include <cuda_runtime.h>
#include <stdint.h>
#include <math.h>

#define DRAKE_GPU_AVAILABLE
#include "drake/gpu_environments/gpu_parallel_pendulum_env.h"

namespace {

constexpr int kPendBlockSize = 256;

__device__ inline float pend_xf(uint32_t& state, float scale) {
  state ^= state << 13;
  state ^= state >> 17;
  state ^= state << 5;
  float r = static_cast<float>(state) / static_cast<float>(0xFFFFFFFFu);
  return (r * 2.0f - 1.0f) * scale;
}

// ---------------------------------------------------------------------------
// pendulum_reset_kernel
// ---------------------------------------------------------------------------
__global__ void pendulum_reset_kernel(float* __restrict__ theta,
                                      float* __restrict__ tdot,
                                      uint32_t* __restrict__ seeds,
                                      int* __restrict__ steps,
                                      int num_envs) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= num_envs) return;

  uint32_t s = seeds[i];
  if (s == 0) s = 0xBEEFu + static_cast<uint32_t>(i);

  theta[i] = pend_xf(s, static_cast<float>(M_PI));
  tdot[i]  = pend_xf(s, 1.0f);
  steps[i] = 0;
  seeds[i] = s;
}

// ---------------------------------------------------------------------------
// pendulum_step_kernel
// ---------------------------------------------------------------------------
__global__ void pendulum_step_kernel(float* __restrict__ theta,
                                     float* __restrict__ tdot,
                                     const float* __restrict__ actions,
                                     float* __restrict__ rewards,
                                     uint8_t* __restrict__ done,
                                     int* __restrict__ steps,
                                     uint32_t* __restrict__ seeds,
                                     int num_envs) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= num_envs) return;

  using namespace drake::gpu_environments;

  float th  = theta[i];
  float td  = tdot[i];
  float rew;
  PendulumDynamicsStep(th, td, actions[i], rew);

  rewards[i] = rew;
  int st = steps[i] + 1;
  steps[i] = st;

  const bool env_done = (st >= kPendMaxSteps);
  done[i] = env_done ? 1u : 0u;

  if (env_done) {
    uint32_t s = seeds[i];
    if (s == 0) s = 0xBEEFu + static_cast<uint32_t>(i);
    th = pend_xf(s, static_cast<float>(M_PI));
    td = pend_xf(s, 1.0f);
    steps[i] = 0;
    seeds[i] = s;
  }

  theta[i] = th;
  tdot[i]  = td;
}

}  // anonymous namespace

// ---------------------------------------------------------------------------
// Launcher functions
// ---------------------------------------------------------------------------

void LaunchPendulumResetKernel(float* d_theta, float* d_tdot,
                               uint32_t* d_seeds, int* d_steps,
                               int num_envs, cudaStream_t stream) {
  const int blocks = (num_envs + kPendBlockSize - 1) / kPendBlockSize;
  pendulum_reset_kernel<<<blocks, kPendBlockSize, 0, stream>>>(
      d_theta, d_tdot, d_seeds, d_steps, num_envs);
}

void LaunchPendulumStepKernel(float* d_theta, float* d_tdot,
                              const float* d_actions,
                              float* d_rewards, uint8_t* d_done,
                              int* d_steps, uint32_t* d_seeds,
                              int num_envs, cudaStream_t stream) {
  const int blocks = (num_envs + kPendBlockSize - 1) / kPendBlockSize;
  pendulum_step_kernel<<<blocks, kPendBlockSize, 0, stream>>>(
      d_theta, d_tdot, d_actions, d_rewards, d_done, d_steps, d_seeds,
      num_envs);
}
