/// @file mountaincar_step.cu
///
/// CUDA kernels for the GPU parallel MountainCar environment.

#include <cuda_runtime.h>
#include <stdint.h>
#include <math.h>

#define DRAKE_GPU_AVAILABLE
#include "drake/gpu_environments/gpu_parallel_mountaincar_env.h"

namespace {

constexpr int kMCBlockSize = 256;

__device__ inline float mc_xf(uint32_t& s, float scale) {
  s ^= s << 13; s ^= s >> 17; s ^= s << 5;
  return (static_cast<float>(s) / static_cast<float>(0xFFFFFFFFu) * 2.0f - 1.0f)
         * scale;
}

__global__ void mc_reset_kernel(float* __restrict__ pos,
                                float* __restrict__ vel,
                                uint32_t* __restrict__ seeds,
                                int* __restrict__ steps,
                                int num_envs) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= num_envs) return;
  uint32_t s = seeds[i];
  if (s == 0) s = 0xBEEFu + static_cast<uint32_t>(i);
  pos[i]   = -0.5f + mc_xf(s, 1.0f) * 0.1f;
  vel[i]   = 0.0f;
  steps[i] = 0;
  seeds[i] = s;
}

__global__ void mc_step_kernel(float* __restrict__ pos,
                               float* __restrict__ vel,
                               const float* __restrict__ actions,
                               float* __restrict__ rewards,
                               uint8_t* __restrict__ done,
                               int* __restrict__ steps,
                               uint32_t* __restrict__ seeds,
                               int num_envs) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= num_envs) return;

  using namespace drake::gpu_environments;

  float p = pos[i], v = vel[i], rew;
  bool env_done;
  MountainCarDynamicsStep(p, v, actions[i], rew, env_done);

  int st = steps[i] + 1;
  if (!env_done && st >= kMCMaxSteps) env_done = true;

  rewards[i] = rew;
  done[i]    = env_done ? 1u : 0u;
  steps[i]   = st;

  if (env_done) {
    uint32_t s = seeds[i];
    if (s == 0) s = 0xBEEFu + static_cast<uint32_t>(i);
    p       = -0.5f + mc_xf(s, 1.0f) * 0.1f;
    v       = 0.0f;
    steps[i] = 0;
    seeds[i] = s;
  }

  pos[i] = p;
  vel[i] = v;
}

}  // namespace

void LaunchMCResetKernel(float* d_pos, float* d_vel,
                         uint32_t* d_seeds, int* d_steps,
                         int num_envs, cudaStream_t stream) {
  const int blocks = (num_envs + kMCBlockSize - 1) / kMCBlockSize;
  mc_reset_kernel<<<blocks, kMCBlockSize, 0, stream>>>(
      d_pos, d_vel, d_seeds, d_steps, num_envs);
}

void LaunchMCStepKernel(float* d_pos, float* d_vel,
                        const float* d_actions,
                        float* d_rewards, uint8_t* d_done,
                        int* d_steps, uint32_t* d_seeds,
                        int num_envs, cudaStream_t stream) {
  const int blocks = (num_envs + kMCBlockSize - 1) / kMCBlockSize;
  mc_step_kernel<<<blocks, kMCBlockSize, 0, stream>>>(
      d_pos, d_vel, d_actions, d_rewards, d_done, d_steps, d_seeds, num_envs);
}
