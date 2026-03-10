/// @file gpu_parallel_cartpole_env.h
///
/// GpuParallelCartpoleEnv — a gym-styled C++ environment that runs N CartPole
/// instances in parallel, using CUDA when available and CPU threads otherwise.
///
/// ## Quick start
///
///   GpuParallelCartpoleEnv env(128);
///   env.Reset();
///
///   EnvMatrix actions(env.kActionDim, env.num_envs());
///   actions.setZero();
///
///   while (true) {
///     StepResult result = env.Step(actions);
///     // result.observations : [kObsDim  x N]
///     // result.rewards       : [N]
///     // result.done          : [N]
///   }
///
/// ## GPU vs CPU path
///
/// Compiled with -DDRAKE_GPU_AVAILABLE the class allocates CUDA device memory
/// and dispatches a CUDA kernel to advance all environments simultaneously.
/// Without CUDA it uses std::thread to parallelize the same cartpole dynamics
/// using float32 arithmetic.
///
/// ## Physics model — CartPole-v1 (OpenAI Gymnasium compatible)
///
///  State x = [cart_pos, cart_vel, pole_angle, pole_ang_vel]
///  Action a = [force_on_cart]   (continuous, typically ±10 N)
///
///  Parameters:
///    m_c = 1.0 kg    (cart mass)
///    m_p = 0.1 kg    (pole mass)
///    l   = 0.5 m     (half pole length)
///    g   = 9.81 m/s²
///    dt  = 0.02 s    (50 Hz)
///
///  Termination: |cart_pos| > 2.4 m  OR  |pole_angle| > 12°

#pragma once

// Shared infrastructure: EnvMatrix, EnvVector, StepResult, XorshiftFloat,
// CpuParallelFor, GPU macros.
#include "drake/gpu_environments/gpu_env_common.h"

#include <cmath>

// CartPole uses GPU_CALLABLE from gpu_env_common.h.
#define CARTPOLE_CALLABLE GPU_CALLABLE

namespace drake {
namespace gpu_environments {

// ---------------------------------------------------------------------------
// CartPole physical constants (float32 for GPU performance)
// ---------------------------------------------------------------------------

inline constexpr float kCartMass     = 1.0f;   ///< kg
inline constexpr float kPoleMass     = 0.1f;   ///< kg
inline constexpr float kHalfPoleLen  = 0.5f;   ///< m
inline constexpr float kGravity      = 9.81f;  ///< m/s²
inline constexpr float kDt           = 0.02f;  ///< s  (50 Hz)
inline constexpr float kMaxCartX     = 2.4f;   ///< m
inline constexpr float kMaxPoleAngle =
    static_cast<float>(12.0 * 3.14159265358979323846 / 180.0);  ///< rad ≈ 0.209
inline constexpr float kInitRange    = 0.05f;  ///< uniform init range

// StepResult is defined in gpu_env_common.h.

// ---------------------------------------------------------------------------
// Single-environment CartPole step (CPU + GPU callable)
// ---------------------------------------------------------------------------

/// Advance one CartPole environment by dt using semi-implicit Euler.
///
/// @param[in,out] x       Cart position (m)
/// @param[in,out] xdot    Cart velocity (m/s)
/// @param[in,out] theta   Pole angle (rad)
/// @param[in,out] tdot    Pole angular velocity (rad/s)
/// @param[in]    action   Force applied to cart (N)
CARTPOLE_CALLABLE inline void CartpoleDynamicsStep(float& x, float& xdot,
                                                   float& theta, float& tdot,
                                                   float action) {
  const float sin_t   = sinf(theta);
  const float cos_t   = cosf(theta);
  const float total_m = kCartMass + kPoleMass;
  const float pl      = kPoleMass * kHalfPoleLen;

  const float temp =
      (action + pl * tdot * tdot * sin_t) / total_m;

  const float theta_ddot =
      (kGravity * sin_t - cos_t * temp) /
      (kHalfPoleLen *
       (4.0f / 3.0f - kPoleMass * cos_t * cos_t / total_m));

  const float x_ddot =
      temp - pl * theta_ddot * cos_t / total_m;

  // Semi-implicit Euler: update velocities first, then positions.
  xdot  += kDt * x_ddot;
  tdot  += kDt * theta_ddot;
  x     += kDt * xdot;
  theta += kDt * tdot;
}

// ---------------------------------------------------------------------------
// GpuParallelCartpoleEnv
// ---------------------------------------------------------------------------

class GpuParallelCartpoleEnv {
 public:
  static constexpr int kObsDim    = 4;  ///< [x, ẋ, θ, θ̇]
  static constexpr int kActionDim = 1;  ///< [force]
  static constexpr int kStateDim  = kObsDim;

  // ---- Construction / destruction ----------------------------------------

  /// @param num_envs  Number of parallel environments (e.g. 128).
  explicit GpuParallelCartpoleEnv(int num_envs);
  ~GpuParallelCartpoleEnv();

  // Non-copyable; movable.
  GpuParallelCartpoleEnv(const GpuParallelCartpoleEnv&) = delete;
  GpuParallelCartpoleEnv& operator=(const GpuParallelCartpoleEnv&) = delete;
  GpuParallelCartpoleEnv(GpuParallelCartpoleEnv&&) noexcept;
  GpuParallelCartpoleEnv& operator=(GpuParallelCartpoleEnv&&) noexcept;

  // ---- Gymnasium interface ------------------------------------------------

  /// Reset all environments.
  ///
  /// @param seeds  Optional per-environment seeds.  If empty, uses 0..N-1.
  ///               Each seed drives a uniform(-0.05, 0.05) init for all
  ///               four state variables.
  void Reset(const std::vector<uint32_t>& seeds = {});

  /// Step all environments simultaneously.
  ///
  /// @param actions  [kActionDim x num_envs] EnvMatrix of actions.
  /// @return StepResult with observations, rewards, done flags.
  ///
  /// Terminated environments are automatically reset in-place (their
  /// `done` flag is set true for this step, then the state is reinitialised
  /// for the next step, matching the standard vectorised-env convention).
  [[nodiscard]] StepResult Step(const EnvMatrix& actions);

  // ---- Accessors ----------------------------------------------------------

  /// Current state observations as [kObsDim x num_envs] EnvMatrix.
  [[nodiscard]] EnvMatrix GetObservations() const;

  /// Number of parallel environments.
  [[nodiscard]] int num_envs() const { return num_envs_; }

  /// True when CUDA is enabled and data lives on device memory.
  [[nodiscard]] bool IsGpuEnabled() const;

 private:
  int num_envs_{0};

  // ---- CPU state storage (SoA — Structure of Arrays) --------------------
  // All arrays have length num_envs_.
  // SoA layout matches GPU CUDA memory layout exactly (coalesced access).

  std::vector<float>   h_x_;       ///< cart position
  std::vector<float>   h_xdot_;    ///< cart velocity
  std::vector<float>   h_theta_;   ///< pole angle
  std::vector<float>   h_tdot_;    ///< pole angular velocity

  std::vector<float>   h_rewards_; ///< last rewards
  std::vector<uint8_t> h_done_;    ///< last done flags (0=alive, 1=done)

  // Seeds for auto-reset (per environment).
  std::vector<uint32_t> seeds_;

  // ---- Private helpers ---------------------------------------------------

  void CpuStep(const float* actions_ptr);
  void CpuReset(const uint32_t* seed_ptr);
  [[nodiscard]] StepResult GetStepResult() const;

#ifdef DRAKE_GPU_AVAILABLE
  // ---- GPU state storage (device memory, SoA) ---------------------------

  float*    d_x_       {nullptr};
  float*    d_xdot_    {nullptr};
  float*    d_theta_   {nullptr};
  float*    d_tdot_    {nullptr};

  float*    d_actions_ {nullptr};
  float*    d_rewards_ {nullptr};
  uint8_t*  d_done_    {nullptr};
  uint32_t* d_seeds_   {nullptr};

  cudaStream_t stream_ {nullptr};

  void GpuAllocate();
  void GpuFree();
  void GpuStep(const float* h_actions);
  void GpuReset(const uint32_t* h_seeds);
  void GpuCopyStateToHost();
#endif
};

}  // namespace gpu_environments
}  // namespace drake
