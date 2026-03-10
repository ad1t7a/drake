/// @file gpu_parallel_mountaincar_env.cc
///
/// CPU parallel implementation of GpuParallelMountainCarEnv.

#include "drake/gpu_environments/gpu_parallel_mountaincar_env.h"

#include <cmath>
#include <stdexcept>

#ifdef DRAKE_GPU_AVAILABLE
extern void LaunchMCResetKernel(float* d_pos, float* d_vel,
                                uint32_t* d_seeds, int* d_steps,
                                int num_envs, cudaStream_t stream);
extern void LaunchMCStepKernel(float* d_pos, float* d_vel,
                               const float* d_actions,
                               float* d_rewards, uint8_t* d_done,
                               int* d_steps, uint32_t* d_seeds,
                               int num_envs, cudaStream_t stream);
#endif

namespace drake {
namespace gpu_environments {

GpuParallelMountainCarEnv::GpuParallelMountainCarEnv(int num_envs)
    : num_envs_(num_envs),
      h_pos_(num_envs, 0.0f),
      h_vel_(num_envs, 0.0f),
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

GpuParallelMountainCarEnv::~GpuParallelMountainCarEnv() {
#ifdef DRAKE_GPU_AVAILABLE
  GpuFree();
#endif
}

GpuParallelMountainCarEnv::GpuParallelMountainCarEnv(
    GpuParallelMountainCarEnv&& o) noexcept
    : num_envs_(o.num_envs_),
      h_pos_(std::move(o.h_pos_)),
      h_vel_(std::move(o.h_vel_)),
      h_rewards_(std::move(o.h_rewards_)),
      h_done_(std::move(o.h_done_)),
      h_steps_(std::move(o.h_steps_)),
      seeds_(std::move(o.seeds_)) {
#ifdef DRAKE_GPU_AVAILABLE
  d_pos_ = o.d_pos_; o.d_pos_ = nullptr;
  d_vel_ = o.d_vel_; o.d_vel_ = nullptr;
  d_actions_ = o.d_actions_; o.d_actions_ = nullptr;
  d_rewards_ = o.d_rewards_; o.d_rewards_ = nullptr;
  d_done_    = o.d_done_;    o.d_done_    = nullptr;
  d_steps_   = o.d_steps_;   o.d_steps_   = nullptr;
  d_seeds_   = o.d_seeds_;   o.d_seeds_   = nullptr;
  stream_    = o.stream_;    o.stream_    = nullptr;
#endif
  o.num_envs_ = 0;
}

GpuParallelMountainCarEnv& GpuParallelMountainCarEnv::operator=(
    GpuParallelMountainCarEnv&& o) noexcept {
  if (this == &o) return *this;
#ifdef DRAKE_GPU_AVAILABLE
  GpuFree();
  d_pos_     = o.d_pos_;     o.d_pos_     = nullptr;
  d_vel_     = o.d_vel_;     o.d_vel_     = nullptr;
  d_actions_ = o.d_actions_; o.d_actions_ = nullptr;
  d_rewards_ = o.d_rewards_; o.d_rewards_ = nullptr;
  d_done_    = o.d_done_;    o.d_done_    = nullptr;
  d_steps_   = o.d_steps_;   o.d_steps_   = nullptr;
  d_seeds_   = o.d_seeds_;   o.d_seeds_   = nullptr;
  stream_    = o.stream_;    o.stream_    = nullptr;
#endif
  num_envs_  = o.num_envs_; o.num_envs_ = 0;
  h_pos_     = std::move(o.h_pos_);
  h_vel_     = std::move(o.h_vel_);
  h_rewards_ = std::move(o.h_rewards_);
  h_done_    = std::move(o.h_done_);
  h_steps_   = std::move(o.h_steps_);
  seeds_     = std::move(o.seeds_);
  return *this;
}

bool GpuParallelMountainCarEnv::IsGpuEnabled() const {
#ifdef DRAKE_GPU_AVAILABLE
  return true;
#else
  return false;
#endif
}

void GpuParallelMountainCarEnv::Reset(const std::vector<uint32_t>& seeds) {
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

StepResult GpuParallelMountainCarEnv::Step(const EnvMatrix& actions) {
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

EnvMatrix GpuParallelMountainCarEnv::GetObservations() const {
  EnvMatrix obs(kObsDim, num_envs_);
  for (int i = 0; i < num_envs_; ++i) {
    obs(0, i) = h_pos_[i];
    obs(1, i) = h_vel_[i];
  }
  return obs;
}

StepResult GpuParallelMountainCarEnv::GetStepResult() const {
  StepResult r;
  r.observations.resize(kObsDim, num_envs_);
  r.rewards.resize(num_envs_);
  r.done.resize(num_envs_);
  for (int i = 0; i < num_envs_; ++i) {
    r.observations(0, i) = h_pos_[i];
    r.observations(1, i) = h_vel_[i];
    r.rewards[i]         = h_rewards_[i];
    r.done[i]            = (h_done_[i] != 0);
  }
  return r;
}

// ---------------------------------------------------------------------------
// CPU parallel implementation
// ---------------------------------------------------------------------------

void GpuParallelMountainCarEnv::CpuReset(const uint32_t* seed_ptr) {
  CpuParallelFor(num_envs_, [&](int start, int end) {
    for (int i = start; i < end; ++i) {
      uint32_t s = seed_ptr[i];
      if (s == 0) s = 0xBEEFu;
      // Position uniform in [-0.6, -0.4]; velocity 0.
      const float raw = XorshiftFloat(s, 1.0f);  // in [-1,1]
      h_pos_[i]     = -0.5f + raw * 0.1f;  // [-0.6, -0.4]
      h_vel_[i]     = 0.0f;
      h_rewards_[i] = 0.0f;
      h_done_[i]    = 0;
      h_steps_[i]   = 0;
      seeds_[i]     = s;
    }
  });
}

void GpuParallelMountainCarEnv::CpuStep(const float* actions_ptr) {
  CpuParallelFor(num_envs_, [&](int start, int end) {
    for (int i = start; i < end; ++i) {
      float reward;
      bool  done;
      MountainCarDynamicsStep(h_pos_[i], h_vel_[i], actions_ptr[i],
                              reward, done);
      h_rewards_[i] = reward;
      ++h_steps_[i];

      // Also terminate on max steps.
      if (!done && h_steps_[i] >= kMCMaxSteps) done = true;
      h_done_[i] = done ? 1 : 0;

      if (done) {
        uint32_t s = seeds_[i];
        if (s == 0) s = 0xBEEFu;
        const float raw = XorshiftFloat(s, 1.0f);
        h_pos_[i]   = -0.5f + raw * 0.1f;
        h_vel_[i]   = 0.0f;
        h_steps_[i] = 0;
        seeds_[i]   = s;
      }
    }
  });
}

// ---------------------------------------------------------------------------
// GPU implementation
// ---------------------------------------------------------------------------

#ifdef DRAKE_GPU_AVAILABLE

static void MCCudaCheck(cudaError_t err, const char* file, int line) {
  if (err != cudaSuccess) {
    char buf[256];
    std::snprintf(buf, sizeof(buf), "CUDA error %s at %s:%d",
                  cudaGetErrorString(err), file, line);
    throw std::runtime_error(buf);
  }
}
#define MC_CUDA_CHECK(x) MCCudaCheck((x), __FILE__, __LINE__)

void GpuParallelMountainCarEnv::GpuAllocate() {
  const size_t nb = num_envs_ * sizeof(float);
  MC_CUDA_CHECK(cudaMalloc(&d_pos_,     nb));
  MC_CUDA_CHECK(cudaMalloc(&d_vel_,     nb));
  MC_CUDA_CHECK(cudaMalloc(&d_actions_, nb));
  MC_CUDA_CHECK(cudaMalloc(&d_rewards_, nb));
  MC_CUDA_CHECK(cudaMalloc(&d_done_,    num_envs_ * sizeof(uint8_t)));
  MC_CUDA_CHECK(cudaMalloc(&d_steps_,   num_envs_ * sizeof(int)));
  MC_CUDA_CHECK(cudaMalloc(&d_seeds_,   num_envs_ * sizeof(uint32_t)));
  MC_CUDA_CHECK(cudaStreamCreate(&stream_));
}

void GpuParallelMountainCarEnv::GpuFree() {
  if (d_pos_)     { cudaFree(d_pos_);     d_pos_     = nullptr; }
  if (d_vel_)     { cudaFree(d_vel_);     d_vel_     = nullptr; }
  if (d_actions_) { cudaFree(d_actions_); d_actions_ = nullptr; }
  if (d_rewards_) { cudaFree(d_rewards_); d_rewards_ = nullptr; }
  if (d_done_)    { cudaFree(d_done_);    d_done_    = nullptr; }
  if (d_steps_)   { cudaFree(d_steps_);   d_steps_   = nullptr; }
  if (d_seeds_)   { cudaFree(d_seeds_);   d_seeds_   = nullptr; }
  if (stream_)    { cudaStreamDestroy(stream_); stream_ = nullptr; }
}

void GpuParallelMountainCarEnv::GpuReset(const uint32_t* h_seeds) {
  MC_CUDA_CHECK(cudaMemcpyAsync(d_seeds_, h_seeds,
                                num_envs_ * sizeof(uint32_t),
                                cudaMemcpyHostToDevice, stream_));
  LaunchMCResetKernel(d_pos_, d_vel_, d_seeds_, d_steps_, num_envs_, stream_);
  GpuCopyStateToHost();
}

void GpuParallelMountainCarEnv::GpuStep(const float* h_actions) {
  MC_CUDA_CHECK(cudaMemcpyAsync(d_actions_, h_actions,
                                num_envs_ * sizeof(float),
                                cudaMemcpyHostToDevice, stream_));
  LaunchMCStepKernel(d_pos_, d_vel_, d_actions_, d_rewards_, d_done_,
                     d_steps_, d_seeds_, num_envs_, stream_);
}

void GpuParallelMountainCarEnv::GpuCopyStateToHost() {
  const size_t n = static_cast<size_t>(num_envs_);
  MC_CUDA_CHECK(cudaMemcpyAsync(h_pos_.data(),     d_pos_,
                                n * sizeof(float), cudaMemcpyDeviceToHost,
                                stream_));
  MC_CUDA_CHECK(cudaMemcpyAsync(h_vel_.data(),     d_vel_,
                                n * sizeof(float), cudaMemcpyDeviceToHost,
                                stream_));
  MC_CUDA_CHECK(cudaMemcpyAsync(h_rewards_.data(), d_rewards_,
                                n * sizeof(float), cudaMemcpyDeviceToHost,
                                stream_));
  MC_CUDA_CHECK(cudaMemcpyAsync(h_done_.data(),    d_done_,
                                n * sizeof(uint8_t), cudaMemcpyDeviceToHost,
                                stream_));
  MC_CUDA_CHECK(cudaMemcpyAsync(h_steps_.data(),   d_steps_,
                                n * sizeof(int), cudaMemcpyDeviceToHost,
                                stream_));
  MC_CUDA_CHECK(cudaStreamSynchronize(stream_));
}

#endif  // DRAKE_GPU_AVAILABLE

}  // namespace gpu_environments
}  // namespace drake
