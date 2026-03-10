/// @file gpu_parallel_pendulum_env.cc
///
/// CPU parallel implementation of GpuParallelPendulumEnv.

#include "drake/gpu_environments/gpu_parallel_pendulum_env.h"

#include <cmath>
#include <cstdio>
#include <stdexcept>

#ifdef DRAKE_GPU_AVAILABLE
extern void LaunchPendulumResetKernel(float* d_theta, float* d_tdot,
                                      uint32_t* d_seeds, int* d_steps,
                                      int num_envs, cudaStream_t stream);

extern void LaunchPendulumStepKernel(float* d_theta, float* d_tdot,
                                     const float* d_actions,
                                     float* d_rewards, uint8_t* d_done,
                                     int* d_steps, uint32_t* d_seeds,
                                     int num_envs, cudaStream_t stream);
#endif

namespace drake {
namespace gpu_environments {

// ===========================================================================
// Construction / destruction
// ===========================================================================

GpuParallelPendulumEnv::GpuParallelPendulumEnv(int num_envs)
    : num_envs_(num_envs),
      h_theta_(num_envs, 0.0f),
      h_tdot_(num_envs, 0.0f),
      h_rewards_(num_envs, 0.0f),
      h_done_(num_envs, 0),
      h_steps_(num_envs, 0),
      seeds_(num_envs, 0) {
  if (num_envs <= 0)
    throw std::invalid_argument("num_envs must be positive");
#ifdef DRAKE_GPU_AVAILABLE
  GpuAllocate();
#endif
}

GpuParallelPendulumEnv::~GpuParallelPendulumEnv() {
#ifdef DRAKE_GPU_AVAILABLE
  GpuFree();
#endif
}

GpuParallelPendulumEnv::GpuParallelPendulumEnv(
    GpuParallelPendulumEnv&& o) noexcept
    : num_envs_(o.num_envs_),
      h_theta_(std::move(o.h_theta_)),
      h_tdot_(std::move(o.h_tdot_)),
      h_rewards_(std::move(o.h_rewards_)),
      h_done_(std::move(o.h_done_)),
      h_steps_(std::move(o.h_steps_)),
      seeds_(std::move(o.seeds_)) {
#ifdef DRAKE_GPU_AVAILABLE
  d_theta_ = o.d_theta_; o.d_theta_ = nullptr;
  d_tdot_  = o.d_tdot_;  o.d_tdot_  = nullptr;
  d_actions_ = o.d_actions_; o.d_actions_ = nullptr;
  d_rewards_ = o.d_rewards_; o.d_rewards_ = nullptr;
  d_done_    = o.d_done_;    o.d_done_    = nullptr;
  d_steps_   = o.d_steps_;   o.d_steps_   = nullptr;
  d_seeds_   = o.d_seeds_;   o.d_seeds_   = nullptr;
  stream_    = o.stream_;    o.stream_    = nullptr;
#endif
  o.num_envs_ = 0;
}

GpuParallelPendulumEnv& GpuParallelPendulumEnv::operator=(
    GpuParallelPendulumEnv&& o) noexcept {
  if (this == &o) return *this;
#ifdef DRAKE_GPU_AVAILABLE
  GpuFree();
  d_theta_ = o.d_theta_; o.d_theta_ = nullptr;
  d_tdot_  = o.d_tdot_;  o.d_tdot_  = nullptr;
  d_actions_ = o.d_actions_; o.d_actions_ = nullptr;
  d_rewards_ = o.d_rewards_; o.d_rewards_ = nullptr;
  d_done_    = o.d_done_;    o.d_done_    = nullptr;
  d_steps_   = o.d_steps_;   o.d_steps_   = nullptr;
  d_seeds_   = o.d_seeds_;   o.d_seeds_   = nullptr;
  stream_    = o.stream_;    o.stream_    = nullptr;
#endif
  num_envs_  = o.num_envs_; o.num_envs_ = 0;
  h_theta_   = std::move(o.h_theta_);
  h_tdot_    = std::move(o.h_tdot_);
  h_rewards_ = std::move(o.h_rewards_);
  h_done_    = std::move(o.h_done_);
  h_steps_   = std::move(o.h_steps_);
  seeds_     = std::move(o.seeds_);
  return *this;
}

// ===========================================================================
// IsGpuEnabled
// ===========================================================================

bool GpuParallelPendulumEnv::IsGpuEnabled() const {
#ifdef DRAKE_GPU_AVAILABLE
  return true;
#else
  return false;
#endif
}

// ===========================================================================
// Reset
// ===========================================================================

void GpuParallelPendulumEnv::Reset(const std::vector<uint32_t>& seeds) {
  if (seeds.empty()) {
    for (int i = 0; i < num_envs_; ++i) seeds_[i] = static_cast<uint32_t>(i);
  } else if (static_cast<int>(seeds.size()) != num_envs_) {
    throw std::invalid_argument("seeds.size() must equal num_envs or be empty");
  } else {
    seeds_ = seeds;
  }
#ifdef DRAKE_GPU_AVAILABLE
  GpuReset(seeds_.data());
#else
  CpuReset(seeds_.data());
#endif
}

// ===========================================================================
// Step
// ===========================================================================

StepResult GpuParallelPendulumEnv::Step(const EnvMatrix& actions) {
  if (actions.rows() != kActionDim || actions.cols() != num_envs_)
    throw std::invalid_argument("actions must have shape [kActionDim x num_envs]");
#ifdef DRAKE_GPU_AVAILABLE
  GpuStep(actions.data());
  GpuCopyStateToHost();
#else
  CpuStep(actions.data());
#endif
  return GetStepResult();
}

// ===========================================================================
// GetObservations
// ===========================================================================

EnvMatrix GpuParallelPendulumEnv::GetObservations() const {
  EnvMatrix obs(kObsDim, num_envs_);
  for (int i = 0; i < num_envs_; ++i) {
    obs(0, i) = cosf(h_theta_[i]);
    obs(1, i) = sinf(h_theta_[i]);
    obs(2, i) = h_tdot_[i];
  }
  return obs;
}

// ===========================================================================
// GetStepResult (private)
// ===========================================================================

StepResult GpuParallelPendulumEnv::GetStepResult() const {
  StepResult r;
  r.observations.resize(kObsDim, num_envs_);
  r.rewards.resize(num_envs_);
  r.done.resize(num_envs_);
  for (int i = 0; i < num_envs_; ++i) {
    r.observations(0, i) = cosf(h_theta_[i]);
    r.observations(1, i) = sinf(h_theta_[i]);
    r.observations(2, i) = h_tdot_[i];
    r.rewards[i]         = h_rewards_[i];
    r.done[i]            = (h_done_[i] != 0);
  }
  return r;
}

// ===========================================================================
// CPU parallel implementation
// ===========================================================================

void GpuParallelPendulumEnv::CpuReset(const uint32_t* seed_ptr) {
  CpuParallelFor(num_envs_, [&](int start, int end) {
    for (int i = start; i < end; ++i) {
      uint32_t s = seed_ptr[i];
      if (s == 0) s = 0xBEEFu;
      // θ uniform in [-π, π], θ̇ uniform in [-1, 1].
      h_theta_[i]   = XorshiftFloat(s, static_cast<float>(M_PI));
      h_tdot_[i]    = XorshiftFloat(s, 1.0f);
      h_rewards_[i] = 0.0f;
      h_done_[i]    = 0;
      h_steps_[i]   = 0;
      seeds_[i]     = s;
    }
  });
}

void GpuParallelPendulumEnv::CpuStep(const float* actions_ptr) {
  CpuParallelFor(num_envs_, [&](int start, int end) {
    for (int i = start; i < end; ++i) {
      float reward;
      PendulumDynamicsStep(h_theta_[i], h_tdot_[i], actions_ptr[i], reward);
      h_rewards_[i] = reward;
      ++h_steps_[i];

      const bool done = (h_steps_[i] >= kPendMaxSteps);
      h_done_[i] = done ? 1 : 0;

      if (done) {
        // Auto-reset.
        uint32_t s = seeds_[i];
        if (s == 0) s = 0xBEEFu;
        h_theta_[i] = XorshiftFloat(s, static_cast<float>(M_PI));
        h_tdot_[i]  = XorshiftFloat(s, 1.0f);
        h_steps_[i] = 0;
        seeds_[i]   = s;
      }
    }
  });
}

// ===========================================================================
// GPU implementation
// ===========================================================================

#ifdef DRAKE_GPU_AVAILABLE

static void PendCudaCheck(cudaError_t err, const char* file, int line) {
  if (err != cudaSuccess) {
    char buf[256];
    std::snprintf(buf, sizeof(buf), "CUDA error %s at %s:%d",
                  cudaGetErrorString(err), file, line);
    throw std::runtime_error(buf);
  }
}
#define PEND_CUDA_CHECK(x) PendCudaCheck((x), __FILE__, __LINE__)

void GpuParallelPendulumEnv::GpuAllocate() {
  const size_t nb = num_envs_ * sizeof(float);
  PEND_CUDA_CHECK(cudaMalloc(&d_theta_,   nb));
  PEND_CUDA_CHECK(cudaMalloc(&d_tdot_,    nb));
  PEND_CUDA_CHECK(cudaMalloc(&d_actions_, nb));
  PEND_CUDA_CHECK(cudaMalloc(&d_rewards_, nb));
  PEND_CUDA_CHECK(cudaMalloc(&d_done_,    num_envs_ * sizeof(uint8_t)));
  PEND_CUDA_CHECK(cudaMalloc(&d_steps_,   num_envs_ * sizeof(int)));
  PEND_CUDA_CHECK(cudaMalloc(&d_seeds_,   num_envs_ * sizeof(uint32_t)));
  PEND_CUDA_CHECK(cudaStreamCreate(&stream_));
}

void GpuParallelPendulumEnv::GpuFree() {
  if (d_theta_)   { cudaFree(d_theta_);   d_theta_   = nullptr; }
  if (d_tdot_)    { cudaFree(d_tdot_);    d_tdot_    = nullptr; }
  if (d_actions_) { cudaFree(d_actions_); d_actions_ = nullptr; }
  if (d_rewards_) { cudaFree(d_rewards_); d_rewards_ = nullptr; }
  if (d_done_)    { cudaFree(d_done_);    d_done_    = nullptr; }
  if (d_steps_)   { cudaFree(d_steps_);   d_steps_   = nullptr; }
  if (d_seeds_)   { cudaFree(d_seeds_);   d_seeds_   = nullptr; }
  if (stream_)    { cudaStreamDestroy(stream_); stream_ = nullptr; }
}

void GpuParallelPendulumEnv::GpuReset(const uint32_t* h_seeds) {
  PEND_CUDA_CHECK(cudaMemcpyAsync(d_seeds_, h_seeds,
                                  num_envs_ * sizeof(uint32_t),
                                  cudaMemcpyHostToDevice, stream_));
  LaunchPendulumResetKernel(d_theta_, d_tdot_, d_seeds_, d_steps_,
                             num_envs_, stream_);
  GpuCopyStateToHost();
}

void GpuParallelPendulumEnv::GpuStep(const float* h_actions) {
  PEND_CUDA_CHECK(cudaMemcpyAsync(d_actions_, h_actions,
                                  num_envs_ * sizeof(float),
                                  cudaMemcpyHostToDevice, stream_));
  LaunchPendulumStepKernel(d_theta_, d_tdot_, d_actions_,
                            d_rewards_, d_done_, d_steps_, d_seeds_,
                            num_envs_, stream_);
}

void GpuParallelPendulumEnv::GpuCopyStateToHost() {
  const size_t n = static_cast<size_t>(num_envs_);
  PEND_CUDA_CHECK(cudaMemcpyAsync(h_theta_.data(),   d_theta_,
                                  n * sizeof(float), cudaMemcpyDeviceToHost,
                                  stream_));
  PEND_CUDA_CHECK(cudaMemcpyAsync(h_tdot_.data(),    d_tdot_,
                                  n * sizeof(float), cudaMemcpyDeviceToHost,
                                  stream_));
  PEND_CUDA_CHECK(cudaMemcpyAsync(h_rewards_.data(), d_rewards_,
                                  n * sizeof(float), cudaMemcpyDeviceToHost,
                                  stream_));
  PEND_CUDA_CHECK(cudaMemcpyAsync(h_done_.data(),    d_done_,
                                  n * sizeof(uint8_t), cudaMemcpyDeviceToHost,
                                  stream_));
  PEND_CUDA_CHECK(cudaMemcpyAsync(h_steps_.data(),   d_steps_,
                                  n * sizeof(int), cudaMemcpyDeviceToHost,
                                  stream_));
  PEND_CUDA_CHECK(cudaStreamSynchronize(stream_));
}

#endif  // DRAKE_GPU_AVAILABLE

}  // namespace gpu_environments
}  // namespace drake
