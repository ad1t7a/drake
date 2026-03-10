/// @file gpu_parallel_pendulum_env.h
///
/// GpuParallelPendulumEnv — gym-styled C++ parallel Pendulum-v1 environment.
///
/// Runs N pendulum instances simultaneously on GPU (CUDA) or CPU threads.
///
/// ## Physics — Pendulum-v1 (OpenAI Gymnasium compatible)
///
///  State: [θ, θ̇]          (2-dimensional; angle from upright, angular vel)
///  Observation: [cos θ, sin θ, θ̇]  (3-dimensional, matching Gym Pendulum-v1)
///  Action: τ ∈ [-2, 2] N⋅m  (torque applied to joint)
///
///  Parameters:
///    m = 1.0 kg   (mass)
///    l = 1.0 m    (length)
///    g = 9.81 m/s²
///    dt = 0.05 s  (20 Hz)
///    max_torque = 2.0 N⋅m
///    max_speed  = 8.0 rad/s
///
///  Dynamics: θ̈ = -3g/(2l) * sin(θ) + 3/(ml²) * τ
///
///  Reward: -(θ_norm² + 0.1 * θ̇² + 0.001 * τ²)
///    where θ_norm = angle wrapped to [-π, π]
///
///  Episode length: 200 steps (no termination, only time limit).
///  Auto-reset: after 200 steps the environment resets.
///
/// ## Memory layout (SoA for GPU coalescing)
///
///  h_theta[N], h_tdot[N]   — internal state
///  Observations: [cos(θ), sin(θ), θ̇] returned per step

#pragma once

#include "drake/gpu_environments/gpu_env_common.h"

#include <cmath>
#include <vector>

namespace drake {
namespace gpu_environments {

// ---------------------------------------------------------------------------
// Pendulum physical constants
// ---------------------------------------------------------------------------

inline constexpr float kPendMass    = 1.0f;   ///< kg
inline constexpr float kPendLength  = 1.0f;   ///< m
inline constexpr float kPendGravity = 9.81f;  ///< m/s²
inline constexpr float kPendDt      = 0.05f;  ///< s (20 Hz)
inline constexpr float kPendMaxTorque = 2.0f; ///< N⋅m
inline constexpr float kPendMaxSpeed  = 8.0f; ///< rad/s
inline constexpr int   kPendMaxSteps  = 200;  ///< steps per episode

// ---------------------------------------------------------------------------
// Angle normalisation — wraps angle to [-π, π]
// ---------------------------------------------------------------------------

GPU_CALLABLE inline float NormalizeAngle(float angle) {
  // Use fmodf to wrap to (-2π, 2π), then shift to (-π, π).
  angle = fmodf(angle + static_cast<float>(M_PI),
                static_cast<float>(2.0 * M_PI));
  if (angle < 0.0f) angle += static_cast<float>(2.0 * M_PI);
  return angle - static_cast<float>(M_PI);
}

// ---------------------------------------------------------------------------
// Single-step pendulum dynamics (CPU + GPU callable)
// ---------------------------------------------------------------------------

/// Advance one pendulum environment by dt.
/// Clips torque to [-max_torque, max_torque] and angular speed to ±max_speed.
///
/// @param[in,out] theta   Pole angle (rad)
/// @param[in,out] tdot    Angular velocity (rad/s)
/// @param[in]    tau      Applied torque (N⋅m)
/// @param[out]   reward   Step reward
GPU_CALLABLE inline void PendulumDynamicsStep(float& theta, float& tdot,
                                              float tau, float& reward) {
  // Clip torque.
  const float t_clamp = fmaxf(-kPendMaxTorque, fminf(kPendMaxTorque, tau));

  const float theta_ddot =
      -3.0f * kPendGravity / (2.0f * kPendLength) * sinf(theta) +
      3.0f / (kPendMass * kPendLength * kPendLength) * t_clamp;

  // Semi-implicit Euler.
  tdot  += kPendDt * theta_ddot;
  tdot   = fmaxf(-kPendMaxSpeed, fminf(kPendMaxSpeed, tdot));
  theta += kPendDt * tdot;

  // Reward.
  const float theta_n = NormalizeAngle(theta);
  reward = -(theta_n * theta_n +
             0.1f * tdot * tdot +
             0.001f * t_clamp * t_clamp);
}

// ---------------------------------------------------------------------------
// GpuParallelPendulumEnv
// ---------------------------------------------------------------------------

class GpuParallelPendulumEnv {
 public:
  static constexpr int kObsDim    = 3;  ///< [cos θ, sin θ, θ̇]
  static constexpr int kActionDim = 1;  ///< [torque]
  static constexpr int kStateDim  = 2;  ///< [θ, θ̇]

  explicit GpuParallelPendulumEnv(int num_envs);
  ~GpuParallelPendulumEnv();

  GpuParallelPendulumEnv(const GpuParallelPendulumEnv&) = delete;
  GpuParallelPendulumEnv& operator=(const GpuParallelPendulumEnv&) = delete;
  GpuParallelPendulumEnv(GpuParallelPendulumEnv&&) noexcept;
  GpuParallelPendulumEnv& operator=(GpuParallelPendulumEnv&&) noexcept;

  // ---- Gymnasium interface ------------------------------------------------

  void Reset(const std::vector<uint32_t>& seeds = {});

  /// Step all environments.
  /// @param actions [kActionDim x num_envs] torques (clipped to ±2 N⋅m)
  [[nodiscard]] StepResult Step(const EnvMatrix& actions);

  [[nodiscard]] EnvMatrix GetObservations() const;
  [[nodiscard]] int  num_envs()     const { return num_envs_; }
  [[nodiscard]] bool IsGpuEnabled() const;

 private:
  int num_envs_{0};

  std::vector<float>   h_theta_;    ///< pole angle (rad)
  std::vector<float>   h_tdot_;     ///< angular velocity (rad/s)
  std::vector<float>   h_rewards_;
  std::vector<uint8_t> h_done_;
  std::vector<int>     h_steps_;    ///< steps since last reset
  std::vector<uint32_t> seeds_;

  void CpuReset(const uint32_t* seed_ptr);
  void CpuStep(const float* actions_ptr);
  [[nodiscard]] StepResult GetStepResult() const;

#ifdef DRAKE_GPU_AVAILABLE
  float*    d_theta_   {nullptr};
  float*    d_tdot_    {nullptr};
  float*    d_actions_ {nullptr};
  float*    d_rewards_ {nullptr};
  uint8_t*  d_done_    {nullptr};
  int*      d_steps_   {nullptr};
  uint32_t* d_seeds_   {nullptr};

  cudaStream_t stream_ {nullptr};

  void GpuAllocate();
  void GpuFree();
  void GpuReset(const uint32_t* h_seeds);
  void GpuStep(const float* h_actions);
  void GpuCopyStateToHost();
#endif
};

}  // namespace gpu_environments
}  // namespace drake
