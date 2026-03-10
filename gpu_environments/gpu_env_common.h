/// @file gpu_env_common.h
///
/// Shared infrastructure for GPU-parallel environments:
/// - EnvMatrix and EnvVector types
/// - xorshift32 PRNG (CPU + GPU callable)
/// - Thread-pool parallel-for helper
/// - GPU/CPU dispatch macros
///
/// All environment implementations include this header.

#pragma once

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <thread>
#include <vector>

// ---------------------------------------------------------------------------
// GPU / CPU dispatch macros
// ---------------------------------------------------------------------------

#ifdef DRAKE_GPU_AVAILABLE
#include <cuda_runtime.h>
#define GPU_CALLABLE __device__ __host__
#define GPU_KERNEL   __global__
#else
#define GPU_CALLABLE
#define GPU_KERNEL
#endif

namespace drake {
namespace gpu_environments {

// ---------------------------------------------------------------------------
// EnvMatrix — column-major float matrix
//
// Layout: data[col * rows + row]  (matches Eigen ColMajor default).
// One column per environment.
// ---------------------------------------------------------------------------

class EnvMatrix {
 public:
  EnvMatrix() : rows_(0), cols_(0) {}
  EnvMatrix(int rows, int cols)
      : rows_(rows), cols_(cols), data_(rows * cols, 0.0f) {}

  float& operator()(int r, int c) { return data_[c * rows_ + r]; }
  float  operator()(int r, int c) const { return data_[c * rows_ + r]; }

  int    rows()  const { return rows_; }
  int    cols()  const { return cols_; }
  float* data()        { return data_.data(); }
  const float* data() const { return data_.data(); }

  void setZero() { std::fill(data_.begin(), data_.end(), 0.0f); }
  void resize(int r, int c) { rows_ = r; cols_ = c; data_.assign(r * c, 0.0f); }

 private:
  int rows_, cols_;
  std::vector<float> data_;
};

/// 1-D float vector (one entry per environment).
class EnvVector {
 public:
  EnvVector() : size_(0) {}
  explicit EnvVector(int n) : size_(n), data_(n, 0.0f) {}

  float& operator[](int i)       { return data_[i]; }
  float  operator[](int i) const { return data_[i]; }
  int    size()            const { return size_; }

  void resize(int n) { size_ = n; data_.assign(n, 0.0f); }

 private:
  int size_;
  std::vector<float> data_;
};

// ---------------------------------------------------------------------------
// StepResult — returned by every environment's Step() call
// ---------------------------------------------------------------------------

struct StepResult {
  EnvMatrix observations;   ///< [obs_dim x N]
  EnvVector rewards;        ///< [N]
  std::vector<bool> done;  ///< [N]
};

// ---------------------------------------------------------------------------
// xorshift32 PRNG (CPU + GPU callable)
// ---------------------------------------------------------------------------

/// Generate the next float in [-scale, +scale] from an xorshift32 state.
GPU_CALLABLE inline float XorshiftFloat(uint32_t& state, float scale) {
  state ^= state << 13;
  state ^= state >> 17;
  state ^= state << 5;
  float r = static_cast<float>(state) /
            static_cast<float>(0xFFFFFFFFu);
  return (r * 2.0f - 1.0f) * scale;
}

// ---------------------------------------------------------------------------
// CpuParallelFor — thread-pool parallel-for over [0, N)
// ---------------------------------------------------------------------------

/// Call fn(start, end) for each non-overlapping subrange of [0, N).
/// Uses at most max_threads hardware threads.
template <typename Fn>
void CpuParallelFor(int N, Fn fn) {
  if (N <= 0) return;
  const int hw = static_cast<int>(std::thread::hardware_concurrency());
  const int T  = std::max(1, std::min(hw, N));

  if (T == 1) {
    fn(0, N);
    return;
  }

  std::vector<std::thread> threads;
  threads.reserve(T);
  const int chunk = (N + T - 1) / T;
  for (int t = 0; t < T; ++t) {
    int start = t * chunk;
    int end   = std::min(start + chunk, N);
    if (start >= end) break;
    threads.emplace_back(fn, start, end);
  }
  for (auto& th : threads) th.join();
}

}  // namespace gpu_environments
}  // namespace drake
