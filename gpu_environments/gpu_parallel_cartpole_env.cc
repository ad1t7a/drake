/// @file gpu_parallel_cartpole_env.cc
///
/// CPU parallel implementation of GpuParallelCartpoleEnv.
///
/// When compiled without DRAKE_GPU_AVAILABLE this file provides the complete
/// implementation.  When CUDA is available this file still provides the host-
/// side logic (memory management, data marshalling) while the actual physics
/// kernel lives in cartpole_step.cu.

#include "drake/gpu_environments/gpu_parallel_cartpole_env.h"

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <thread>
#include <vector>

// ---------------------------------------------------------------------------
// xorshift32 — fast, reproducible, dependency-free PRNG
// ---------------------------------------------------------------------------

static inline float xorshift_float(uint32_t& state, float scale) {
  state ^= state << 13;
  state ^= state >> 17;
  state ^= state << 5;
  // Map to [-scale, +scale].
  float r = static_cast<float>(state) /
            static_cast<float>(std::numeric_limits<uint32_t>::max());
  return (r * 2.0f - 1.0f) * scale;
}

// ---------------------------------------------------------------------------
// CUDA declarations (only when CUDA is available)
// ---------------------------------------------------------------------------

#ifdef DRAKE_GPU_AVAILABLE
// Forward-declared; implemented in cartpole_step.cu.
extern void LaunchCartpoleStepKernel(float* d_x, float* d_xdot,
                                     float* d_theta, float* d_tdot,
                                     const float* d_actions,
                                     float* d_rewards, uint8_t* d_done,
                                     uint32_t* d_seeds,
                                     int num_envs,
                                     cudaStream_t stream);

extern void LaunchCartpoleResetKernel(float* d_x, float* d_xdot,
                                      float* d_theta, float* d_tdot,
                                      uint32_t* d_seeds,
                                      int num_envs,
                                      cudaStream_t stream);
#endif

namespace drake {
namespace gpu_environments {

// ===========================================================================
// Construction / destruction
// ===========================================================================

GpuParallelCartpoleEnv::GpuParallelCartpoleEnv(int num_envs)
    : num_envs_(num_envs),
      h_x_(num_envs, 0.0f),
      h_xdot_(num_envs, 0.0f),
      h_theta_(num_envs, 0.0f),
      h_tdot_(num_envs, 0.0f),
      h_rewards_(num_envs, 0.0f),
      h_done_(num_envs, 0),
      seeds_(num_envs, 0) {
  if (num_envs <= 0) {
    throw std::invalid_argument("num_envs must be positive");
  }

#ifdef DRAKE_GPU_AVAILABLE
  GpuAllocate();
#endif
}

GpuParallelCartpoleEnv::~GpuParallelCartpoleEnv() {
#ifdef DRAKE_GPU_AVAILABLE
  GpuFree();
#endif
}

GpuParallelCartpoleEnv::GpuParallelCartpoleEnv(
    GpuParallelCartpoleEnv&& other) noexcept
    : num_envs_(other.num_envs_),
      h_x_(std::move(other.h_x_)),
      h_xdot_(std::move(other.h_xdot_)),
      h_theta_(std::move(other.h_theta_)),
      h_tdot_(std::move(other.h_tdot_)),
      h_rewards_(std::move(other.h_rewards_)),
      h_done_(std::move(other.h_done_)),
      seeds_(std::move(other.seeds_)) {
#ifdef DRAKE_GPU_AVAILABLE
  d_x_     = other.d_x_;     other.d_x_     = nullptr;
  d_xdot_  = other.d_xdot_;  other.d_xdot_  = nullptr;
  d_theta_ = other.d_theta_; other.d_theta_ = nullptr;
  d_tdot_  = other.d_tdot_;  other.d_tdot_  = nullptr;
  d_actions_ = other.d_actions_; other.d_actions_ = nullptr;
  d_rewards_ = other.d_rewards_; other.d_rewards_ = nullptr;
  d_done_    = other.d_done_;    other.d_done_    = nullptr;
  d_seeds_   = other.d_seeds_;   other.d_seeds_   = nullptr;
  stream_    = other.stream_;    other.stream_    = nullptr;
#endif
  other.num_envs_ = 0;
}

GpuParallelCartpoleEnv& GpuParallelCartpoleEnv::operator=(
    GpuParallelCartpoleEnv&& other) noexcept {
  if (this == &other) return *this;
#ifdef DRAKE_GPU_AVAILABLE
  GpuFree();
  d_x_     = other.d_x_;     other.d_x_     = nullptr;
  d_xdot_  = other.d_xdot_;  other.d_xdot_  = nullptr;
  d_theta_ = other.d_theta_; other.d_theta_ = nullptr;
  d_tdot_  = other.d_tdot_;  other.d_tdot_  = nullptr;
  d_actions_ = other.d_actions_; other.d_actions_ = nullptr;
  d_rewards_ = other.d_rewards_; other.d_rewards_ = nullptr;
  d_done_    = other.d_done_;    other.d_done_    = nullptr;
  d_seeds_   = other.d_seeds_;   other.d_seeds_   = nullptr;
  stream_    = other.stream_;    other.stream_    = nullptr;
#endif
  num_envs_ = other.num_envs_; other.num_envs_ = 0;
  h_x_     = std::move(other.h_x_);
  h_xdot_  = std::move(other.h_xdot_);
  h_theta_ = std::move(other.h_theta_);
  h_tdot_  = std::move(other.h_tdot_);
  h_rewards_ = std::move(other.h_rewards_);
  h_done_    = std::move(other.h_done_);
  seeds_     = std::move(other.seeds_);
  return *this;
}

// ===========================================================================
// IsGpuEnabled
// ===========================================================================

bool GpuParallelCartpoleEnv::IsGpuEnabled() const {
#ifdef DRAKE_GPU_AVAILABLE
  return true;
#else
  return false;
#endif
}

// ===========================================================================
// Reset
// ===========================================================================

void GpuParallelCartpoleEnv::Reset(const std::vector<uint32_t>& seeds) {
  if (seeds.empty()) {
    for (int i = 0; i < num_envs_; ++i) seeds_[i] = static_cast<uint32_t>(i);
  } else if (static_cast<int>(seeds.size()) != num_envs_) {
    throw std::invalid_argument(
        "seeds.size() must equal num_envs() or be empty");
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

StepResult GpuParallelCartpoleEnv::Step(const EnvMatrix& actions) {
  if (actions.rows() != kActionDim || actions.cols() != num_envs_) {
    throw std::invalid_argument(
        "actions must have shape [kActionDim x num_envs]");
  }

  // Eigen column-major: for kActionDim==1 actions.data() is a flat array [N].
  const float* actions_ptr = actions.data();

#ifdef DRAKE_GPU_AVAILABLE
  GpuStep(actions_ptr);
  GpuCopyStateToHost();
#else
  CpuStep(actions_ptr);
#endif

  return GetStepResult();
}

// ===========================================================================
// GetStepResult (private)
// ===========================================================================

StepResult GpuParallelCartpoleEnv::GetStepResult() const {
  StepResult result;
  result.observations.resize(kObsDim, num_envs_);
  result.rewards.resize(num_envs_);
  result.done.resize(num_envs_);

  for (int i = 0; i < num_envs_; ++i) {
    result.observations(0, i) = h_x_[i];
    result.observations(1, i) = h_xdot_[i];
    result.observations(2, i) = h_theta_[i];
    result.observations(3, i) = h_tdot_[i];
    result.rewards[i]         = h_rewards_[i];
    result.done[i]            = (h_done_[i] != 0);
  }
  return result;
}

// ===========================================================================
// GetObservations
// ===========================================================================

EnvMatrix GpuParallelCartpoleEnv::GetObservations() const {
  EnvMatrix obs(kObsDim, num_envs_);
  for (int i = 0; i < num_envs_; ++i) {
    obs(0, i) = h_x_[i];
    obs(1, i) = h_xdot_[i];
    obs(2, i) = h_theta_[i];
    obs(3, i) = h_tdot_[i];
  }
  return obs;
}

// ===========================================================================
// CPU parallel implementation
// ===========================================================================

void GpuParallelCartpoleEnv::CpuReset(const uint32_t* seed_ptr) {
  const int N = num_envs_;
  const int hw_threads =
      static_cast<int>(std::thread::hardware_concurrency());
  const int num_threads = std::max(1, std::min(hw_threads, N));

  auto worker = [&](int start, int end) {
    for (int i = start; i < end; ++i) {
      uint32_t s = seed_ptr[i];
      // Ensure seed != 0 (xorshift32 fails for 0).
      if (s == 0) s = 0xBEEFu;
      h_x_[i]     = xorshift_float(s, kInitRange);
      h_xdot_[i]  = xorshift_float(s, kInitRange);
      h_theta_[i] = xorshift_float(s, kInitRange);
      h_tdot_[i]  = xorshift_float(s, kInitRange);
      seeds_[i]   = s;  // advance seed for auto-reset
      h_rewards_[i] = 0.0f;
      h_done_[i]    = 0;
    }
  };

  if (num_threads == 1) {
    worker(0, N);
    return;
  }

  std::vector<std::thread> threads;
  threads.reserve(num_threads);
  const int chunk = (N + num_threads - 1) / num_threads;
  for (int t = 0; t < num_threads; ++t) {
    int start = t * chunk;
    int end   = std::min(start + chunk, N);
    if (start >= end) break;
    threads.emplace_back(worker, start, end);
  }
  for (auto& th : threads) th.join();
}

void GpuParallelCartpoleEnv::CpuStep(const float* actions_ptr) {
  const int N = num_envs_;
  const int hw_threads =
      static_cast<int>(std::thread::hardware_concurrency());
  const int num_threads = std::max(1, std::min(hw_threads, N));

  auto worker = [&](int start, int end) {
    for (int i = start; i < end; ++i) {
      CartpoleDynamicsStep(h_x_[i], h_xdot_[i], h_theta_[i], h_tdot_[i],
                           actions_ptr[i]);

      const bool done = (h_x_[i] < -kMaxCartX || h_x_[i] > kMaxCartX) ||
                        (h_theta_[i] < -kMaxPoleAngle ||
                         h_theta_[i] > kMaxPoleAngle);
      h_rewards_[i] = done ? 0.0f : 1.0f;
      h_done_[i]    = done ? 1 : 0;

      if (done) {
        // Auto-reset: reinitialise this environment with its updated seed.
        uint32_t s = seeds_[i];
        if (s == 0) s = 0xBEEFu;
        h_x_[i]     = xorshift_float(s, kInitRange);
        h_xdot_[i]  = xorshift_float(s, kInitRange);
        h_theta_[i] = xorshift_float(s, kInitRange);
        h_tdot_[i]  = xorshift_float(s, kInitRange);
        seeds_[i]   = s;
      }
    }
  };

  if (num_threads == 1) {
    worker(0, N);
    return;
  }

  std::vector<std::thread> threads;
  threads.reserve(num_threads);
  const int chunk = (N + num_threads - 1) / num_threads;
  for (int t = 0; t < num_threads; ++t) {
    int start = t * chunk;
    int end   = std::min(start + chunk, N);
    if (start >= end) break;
    threads.emplace_back(worker, start, end);
  }
  for (auto& th : threads) th.join();
}

// ===========================================================================
// GPU implementation (only when CUDA available)
// ===========================================================================

#ifdef DRAKE_GPU_AVAILABLE

static void CudaCheck(cudaError_t err, const char* file, int line) {
  if (err != cudaSuccess) {
    char buf[256];
    std::snprintf(buf, sizeof(buf), "CUDA error %s at %s:%d",
                  cudaGetErrorString(err), file, line);
    throw std::runtime_error(buf);
  }
}
#define CUDA_CHECK(x) CudaCheck((x), __FILE__, __LINE__)

void GpuParallelCartpoleEnv::GpuAllocate() {
  const size_t n_bytes = num_envs_ * sizeof(float);
  CUDA_CHECK(cudaMalloc(&d_x_,       n_bytes));
  CUDA_CHECK(cudaMalloc(&d_xdot_,    n_bytes));
  CUDA_CHECK(cudaMalloc(&d_theta_,   n_bytes));
  CUDA_CHECK(cudaMalloc(&d_tdot_,    n_bytes));
  CUDA_CHECK(cudaMalloc(&d_actions_, n_bytes));
  CUDA_CHECK(cudaMalloc(&d_rewards_, n_bytes));
  CUDA_CHECK(cudaMalloc(&d_done_,    num_envs_ * sizeof(uint8_t)));
  CUDA_CHECK(cudaMalloc(&d_seeds_,   num_envs_ * sizeof(uint32_t)));
  CUDA_CHECK(cudaStreamCreate(&stream_));
}

void GpuParallelCartpoleEnv::GpuFree() {
  if (d_x_)       { cudaFree(d_x_);       d_x_       = nullptr; }
  if (d_xdot_)    { cudaFree(d_xdot_);    d_xdot_    = nullptr; }
  if (d_theta_)   { cudaFree(d_theta_);   d_theta_   = nullptr; }
  if (d_tdot_)    { cudaFree(d_tdot_);    d_tdot_    = nullptr; }
  if (d_actions_) { cudaFree(d_actions_); d_actions_ = nullptr; }
  if (d_rewards_) { cudaFree(d_rewards_); d_rewards_ = nullptr; }
  if (d_done_)    { cudaFree(d_done_);    d_done_    = nullptr; }
  if (d_seeds_)   { cudaFree(d_seeds_);   d_seeds_   = nullptr; }
  if (stream_)    { cudaStreamDestroy(stream_); stream_ = nullptr; }
}

void GpuParallelCartpoleEnv::GpuReset(const uint32_t* h_seeds) {
  CUDA_CHECK(cudaMemcpyAsync(d_seeds_, h_seeds,
                             num_envs_ * sizeof(uint32_t),
                             cudaMemcpyHostToDevice, stream_));
  LaunchCartpoleResetKernel(d_x_, d_xdot_, d_theta_, d_tdot_,
                            d_seeds_, num_envs_, stream_);
  GpuCopyStateToHost();
}

void GpuParallelCartpoleEnv::GpuStep(const float* h_actions) {
  CUDA_CHECK(cudaMemcpyAsync(d_actions_, h_actions,
                             num_envs_ * sizeof(float),
                             cudaMemcpyHostToDevice, stream_));
  LaunchCartpoleStepKernel(d_x_, d_xdot_, d_theta_, d_tdot_,
                           d_actions_, d_rewards_, d_done_,
                           d_seeds_, num_envs_, stream_);
}

void GpuParallelCartpoleEnv::GpuCopyStateToHost() {
  const size_t n = static_cast<size_t>(num_envs_);
  CUDA_CHECK(cudaMemcpyAsync(h_x_.data(),       d_x_,
                             n * sizeof(float), cudaMemcpyDeviceToHost,
                             stream_));
  CUDA_CHECK(cudaMemcpyAsync(h_xdot_.data(),    d_xdot_,
                             n * sizeof(float), cudaMemcpyDeviceToHost,
                             stream_));
  CUDA_CHECK(cudaMemcpyAsync(h_theta_.data(),   d_theta_,
                             n * sizeof(float), cudaMemcpyDeviceToHost,
                             stream_));
  CUDA_CHECK(cudaMemcpyAsync(h_tdot_.data(),    d_tdot_,
                             n * sizeof(float), cudaMemcpyDeviceToHost,
                             stream_));
  CUDA_CHECK(cudaMemcpyAsync(h_rewards_.data(), d_rewards_,
                             n * sizeof(float), cudaMemcpyDeviceToHost,
                             stream_));
  CUDA_CHECK(cudaMemcpyAsync(h_done_.data(),    d_done_,
                             n * sizeof(uint8_t), cudaMemcpyDeviceToHost,
                             stream_));
  CUDA_CHECK(cudaStreamSynchronize(stream_));
}

#endif  // DRAKE_GPU_AVAILABLE

}  // namespace gpu_environments
}  // namespace drake
