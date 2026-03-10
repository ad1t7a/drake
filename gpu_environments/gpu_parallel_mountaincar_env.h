/// @file gpu_parallel_mountaincar_env.h
///
/// GpuParallelMountainCarEnv — continuous MountainCar-v0 environment.
///
/// ## Physics — MountainCarContinuous-v0 (OpenAI Gymnasium compatible)
///
///  State: [position, velocity]  (2-dimensional)
///  Observation = State
///  Action: force ∈ [-1, 1]  (clipped to ±1)
///
///  Parameters:
///    min_pos = -1.2,  max_pos = 0.6
///    min_vel = -0.07, max_vel = 0.07
///    goal_pos  = 0.45
///    goal_vel  = 0.0  (any velocity ≥ 0 at goal)
///    power     = 0.0015
///    gravity   = 0.0025
///    dt        = 1.0  (position/velocity already per-step)
///
///  Dynamics (per step):
///    vel  += force * power - gravity * cos(3 * pos)
///    vel   = clip(vel, min_vel, max_vel)
///    pos  += vel
///    pos   = clip(pos, min_pos, max_pos)
///    if pos == min_pos: vel = max(vel, 0)
///
///  Reward: -0.1 * action²  per step; +100 on reaching goal
///  Termination: pos ≥ goal_pos AND vel ≥ goal_vel
///  Max steps: 999 (auto-reset after)

#pragma once

#include "drake/gpu_environments/gpu_env_common.h"

#include <cmath>
#include <vector>

namespace drake {
namespace gpu_environments {

// ---------------------------------------------------------------------------
// MountainCar constants
// ---------------------------------------------------------------------------

inline constexpr float kMCMinPos   = -1.2f;
inline constexpr float kMCMaxPos   =  0.6f;
inline constexpr float kMCMinVel   = -0.07f;
inline constexpr float kMCMaxVel   =  0.07f;
inline constexpr float kMCGoalPos  =  0.45f;
inline constexpr float kMCPower    =  0.0015f;
inline constexpr float kMCGravity  =  0.0025f;
inline constexpr int   kMCMaxSteps =  999;

// ---------------------------------------------------------------------------
// Single-step dynamics (CPU + GPU callable)
// ---------------------------------------------------------------------------

/// Advance one MountainCar environment.
/// @param[in,out] pos     Cart position
/// @param[in,out] vel     Cart velocity
/// @param[in]    action   Force (-1 to 1)
/// @param[out]   reward   Step reward
/// @param[out]   done     True if goal reached
GPU_CALLABLE inline void MountainCarDynamicsStep(float& pos, float& vel,
                                                 float action,
                                                 float& reward, bool& done) {
  const float f = fmaxf(-1.0f, fminf(1.0f, action));
  vel += f * kMCPower + cosf(3.0f * pos) * (-kMCGravity);
  vel  = fmaxf(kMCMinVel, fminf(kMCMaxVel, vel));
  pos += vel;
  pos  = fmaxf(kMCMinPos, fminf(kMCMaxPos, pos));
  if (pos == kMCMinPos && vel < 0.0f) vel = 0.0f;

  done   = (pos >= kMCGoalPos);
  reward = done ? 100.0f : -0.1f * f * f;
}

// ---------------------------------------------------------------------------
// GpuParallelMountainCarEnv
// ---------------------------------------------------------------------------

class GpuParallelMountainCarEnv {
 public:
  static constexpr int kObsDim    = 2;  ///< [position, velocity]
  static constexpr int kActionDim = 1;  ///< [force]
  static constexpr int kStateDim  = 2;

  explicit GpuParallelMountainCarEnv(int num_envs);
  ~GpuParallelMountainCarEnv();

  GpuParallelMountainCarEnv(const GpuParallelMountainCarEnv&) = delete;
  GpuParallelMountainCarEnv& operator=(const GpuParallelMountainCarEnv&) = delete;
  GpuParallelMountainCarEnv(GpuParallelMountainCarEnv&&) noexcept;
  GpuParallelMountainCarEnv& operator=(GpuParallelMountainCarEnv&&) noexcept;

  void Reset(const std::vector<uint32_t>& seeds = {});
  [[nodiscard]] StepResult Step(const EnvMatrix& actions);
  [[nodiscard]] EnvMatrix  GetObservations() const;
  [[nodiscard]] int  num_envs()     const { return num_envs_; }
  [[nodiscard]] bool IsGpuEnabled() const;

 private:
  int num_envs_{0};

  std::vector<float>   h_pos_;
  std::vector<float>   h_vel_;
  std::vector<float>   h_rewards_;
  std::vector<uint8_t> h_done_;
  std::vector<int>     h_steps_;
  std::vector<uint32_t> seeds_;

  void CpuReset(const uint32_t* seed_ptr);
  void CpuStep(const float* actions_ptr);
  [[nodiscard]] StepResult GetStepResult() const;

#ifdef DRAKE_GPU_AVAILABLE
  float*    d_pos_     {nullptr};
  float*    d_vel_     {nullptr};
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
