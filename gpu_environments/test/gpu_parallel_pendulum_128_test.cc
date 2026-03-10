/// @file gpu_parallel_pendulum_128_test.cc
///
/// Tests 128 Pendulum environments in parallel (GPU or CPU fallback).
///
/// Test strategy
/// =============
/// 1. Construct 128 pendulum environments.
/// 2. Reset with deterministic seeds.
/// 3. Step with constant torque and verify observation shapes/values.
/// 4. Verify observations are in valid range ([−1,1] for cos/sin, [−8,8] for ω).
/// 5. Verify rewards are negative (by design: reward = −(θ²+...)).
/// 6. Verify episode auto-reset after 200 steps.
/// 7. Verify step-by-step physics vs serial reference.
/// 8. Benchmark throughput.

#include "drake/gpu_environments/gpu_parallel_pendulum_env.h"

#include <cassert>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <stdexcept>
#include <vector>

namespace drake {
namespace gpu_environments {
namespace test {
namespace {

constexpr int kNumEnvs  = 128;
constexpr int kNumSteps = 500;

// ---------------------------------------------------------------------------
// Test 1: Construction and Reset
// ---------------------------------------------------------------------------

void TestConstructionAndReset() {
  std::printf("[TEST 1] Pendulum: Construction and Reset\n");

  GpuParallelPendulumEnv env(kNumEnvs);
  assert(env.num_envs() == kNumEnvs);

  std::vector<uint32_t> seeds(kNumEnvs);
  for (int i = 0; i < kNumEnvs; ++i) seeds[i] = static_cast<uint32_t>(i + 1);
  env.Reset(seeds);

  EnvMatrix obs = env.GetObservations();
  assert(obs.rows() == GpuParallelPendulumEnv::kObsDim);
  assert(obs.cols() == kNumEnvs);

  for (int j = 0; j < kNumEnvs; ++j) {
    const float cos_t = obs(0, j);
    const float sin_t = obs(1, j);
    const float tdot  = obs(2, j);

    // cos and sin must be in [-1, 1].
    if (cos_t < -1.001f || cos_t > 1.001f) {
      std::fprintf(stderr, "FAIL: cos_theta[%d]=%.4f out of [-1,1]\n", j, cos_t);
      throw std::runtime_error("Initial cos_theta out of range");
    }
    if (sin_t < -1.001f || sin_t > 1.001f) {
      std::fprintf(stderr, "FAIL: sin_theta[%d]=%.4f out of [-1,1]\n", j, sin_t);
      throw std::runtime_error("Initial sin_theta out of range");
    }
    // θ̇ initialised in [-1, 1].
    if (std::abs(tdot) > 1.1f) {
      std::fprintf(stderr, "FAIL: tdot[%d]=%.4f exceeds init range\n", j, tdot);
      throw std::runtime_error("Initial tdot out of range");
    }
  }

  std::printf("  PASS: %d pendulum environments reset; obs in valid range.\n",
              kNumEnvs);
}

// ---------------------------------------------------------------------------
// Test 2: Single Step
// ---------------------------------------------------------------------------

void TestSingleStep() {
  std::printf("[TEST 2] Pendulum: Single Step\n");

  GpuParallelPendulumEnv env(kNumEnvs);
  std::vector<uint32_t> seeds(kNumEnvs);
  for (int i = 0; i < kNumEnvs; ++i) seeds[i] = static_cast<uint32_t>(i + 5);
  env.Reset(seeds);

  EnvMatrix actions(GpuParallelPendulumEnv::kActionDim, kNumEnvs);
  actions.setZero();

  StepResult res = env.Step(actions);

  assert(res.observations.rows() == GpuParallelPendulumEnv::kObsDim);
  assert(res.observations.cols() == kNumEnvs);
  assert(res.rewards.size() == kNumEnvs);
  assert(static_cast<int>(res.done.size()) == kNumEnvs);

  // No episode should be done after 1 step (max_steps=200).
  for (int i = 0; i < kNumEnvs; ++i) {
    if (res.done[i]) {
      std::fprintf(stderr,
                   "FAIL: env %d done after step 1 (max_steps=200)\n", i);
      throw std::runtime_error("Unexpected done after 1 step");
    }
    // All rewards should be negative (pendulum cost always ≤ 0).
    if (res.rewards[i] > 0.0f) {
      std::fprintf(stderr, "FAIL: env %d reward=%.4f > 0\n",
                   i, res.rewards[i]);
      throw std::runtime_error("Positive reward unexpected for pendulum");
    }
  }

  std::printf("  PASS: single step; all rewards ≤ 0; no premature dones.\n");
}

// ---------------------------------------------------------------------------
// Test 3: Physics correctness — parallel env-0 vs serial reference
// ---------------------------------------------------------------------------

void TestPhysicsCorrectness() {
  std::printf("[TEST 3] Pendulum: Physics correctness\n");

  constexpr int kCompare = 150;
  constexpr float kTorque = 1.5f;

  std::vector<uint32_t> seeds(kNumEnvs);
  for (int i = 0; i < kNumEnvs; ++i) seeds[i] = static_cast<uint32_t>(i + 1);

  // Replicate initial state for env-0.
  auto xf = [](uint32_t& s, float scale) -> float {
    s ^= s << 13; s ^= s >> 17; s ^= s << 5;
    return (static_cast<float>(s) / static_cast<float>(0xFFFFFFFFu) * 2.0f - 1.0f) * scale;
  };

  float s_theta, s_tdot;
  {
    uint32_t s = seeds[0];
    if (s == 0) s = 0xBEEFu;
    s_theta = xf(s, static_cast<float>(M_PI));
    s_tdot  = xf(s, 1.0f);
  }

  // Serial reference (no auto-reset within kCompare steps since max=200).
  std::vector<float> ref_cos(kCompare), ref_sin(kCompare), ref_td(kCompare);
  {
    float th = s_theta, td = s_tdot, rew;
    for (int t = 0; t < kCompare; ++t) {
      PendulumDynamicsStep(th, td, kTorque, rew);
      ref_cos[t] = cosf(th);
      ref_sin[t] = sinf(th);
      ref_td[t]  = td;
    }
  }

  // Parallel run.
  GpuParallelPendulumEnv env(kNumEnvs);
  env.Reset(seeds);

  EnvMatrix actions(GpuParallelPendulumEnv::kActionDim, kNumEnvs);
  for (int i = 0; i < kNumEnvs; ++i) actions(0, i) = kTorque;

  constexpr float kTol = 1e-5f;
  int mismatches = 0;
  for (int t = 0; t < kCompare; ++t) {
    StepResult res = env.Step(actions);
    const float pc = res.observations(0, 0);
    const float ps = res.observations(1, 0);
    const float pt = res.observations(2, 0);
    if (std::abs(pc - ref_cos[t]) > kTol ||
        std::abs(ps - ref_sin[t]) > kTol ||
        std::abs(pt - ref_td[t])  > kTol) {
      if (mismatches < 3) {
        std::fprintf(stderr,
            "  Mismatch t=%d: par(%.6f,%.6f,%.6f) ref(%.6f,%.6f,%.6f)\n",
            t, pc, ps, pt, ref_cos[t], ref_sin[t], ref_td[t]);
      }
      ++mismatches;
    }
  }

  if (mismatches > 0) {
    std::fprintf(stderr,
                 "FAIL: %d mismatches over %d steps\n", mismatches, kCompare);
    throw std::runtime_error("Pendulum parallel/serial mismatch");
  }

  std::printf("  PASS: env-0 matches serial for %d steps (tol=%.0e).\n",
              kCompare, static_cast<double>(kTol));
}

// ---------------------------------------------------------------------------
// Test 4: Episode auto-reset at step 200
// ---------------------------------------------------------------------------

void TestEpisodeAutoReset() {
  std::printf("[TEST 4] Pendulum: Episode auto-reset at %d steps\n",
              kPendMaxSteps);

  GpuParallelPendulumEnv env(kNumEnvs);
  std::vector<uint32_t> seeds(kNumEnvs);
  for (int i = 0; i < kNumEnvs; ++i) seeds[i] = static_cast<uint32_t>(i + 3);
  env.Reset(seeds);

  EnvMatrix actions(GpuParallelPendulumEnv::kActionDim, kNumEnvs);
  actions.setZero();

  int done_count = 0;
  // Run slightly past one episode to catch auto-resets.
  for (int t = 0; t < kPendMaxSteps + 5; ++t) {
    StepResult res = env.Step(actions);
    for (int i = 0; i < kNumEnvs; ++i) {
      if (res.done[i]) ++done_count;
    }
  }

  // All 128 environments should have fired done at step 200.
  if (done_count < kNumEnvs) {
    std::fprintf(stderr,
                 "FAIL: only %d/%d done events in %d steps\n",
                 done_count, kNumEnvs, kPendMaxSteps + 5);
    throw std::runtime_error("Missing episode terminations");
  }

  std::printf("  PASS: %d episode terminations observed (%d expected).\n",
              done_count, kNumEnvs);
}

// ---------------------------------------------------------------------------
// Test 5: Observations always in valid range
// ---------------------------------------------------------------------------

void TestObservationRange() {
  std::printf("[TEST 5] Pendulum: Observation range validation\n");

  GpuParallelPendulumEnv env(kNumEnvs);
  std::vector<uint32_t> seeds(kNumEnvs);
  for (int i = 0; i < kNumEnvs; ++i) seeds[i] = static_cast<uint32_t>(i + 99);
  env.Reset(seeds);

  EnvMatrix actions(GpuParallelPendulumEnv::kActionDim, kNumEnvs);

  for (int t = 0; t < kNumSteps; ++t) {
    // Random-ish actions.
    for (int i = 0; i < kNumEnvs; ++i) {
      actions(0, i) = (t % 2 == 0) ? kPendMaxTorque : -kPendMaxTorque;
    }
    StepResult res = env.Step(actions);
    for (int j = 0; j < kNumEnvs; ++j) {
      const float c = res.observations(0, j);
      const float s = res.observations(1, j);
      const float w = res.observations(2, j);
      if (!std::isfinite(c) || !std::isfinite(s) || !std::isfinite(w)) {
        std::fprintf(stderr,
                     "FAIL: non-finite obs at step %d env %d\n", t, j);
        throw std::runtime_error("Non-finite pendulum observation");
      }
      if (std::abs(c) > 1.001f || std::abs(s) > 1.001f) {
        std::fprintf(stderr,
                     "FAIL: cos/sin out of [-1,1] at step %d env %d\n", t, j);
        throw std::runtime_error("cos/sin out of valid range");
      }
      if (std::abs(w) > kPendMaxSpeed + 0.01f) {
        std::fprintf(stderr,
                     "FAIL: tdot=%.3f exceeds max_speed=%.1f\n", w, kPendMaxSpeed);
        throw std::runtime_error("Angular velocity exceeds max_speed");
      }
    }
  }

  std::printf("  PASS: %d x %d steps — all observations finite and in range.\n",
              kNumEnvs, kNumSteps);
}

// ---------------------------------------------------------------------------
// Benchmark
// ---------------------------------------------------------------------------

void BenchmarkPendulum() {
  std::printf("[BENCH ] Pendulum: %d envs x %d steps\n", kNumEnvs, kNumSteps);

  GpuParallelPendulumEnv env(kNumEnvs);
  std::vector<uint32_t> seeds(kNumEnvs);
  for (int i = 0; i < kNumEnvs; ++i) seeds[i] = static_cast<uint32_t>(i + 1);
  env.Reset(seeds);

  EnvMatrix actions(GpuParallelPendulumEnv::kActionDim, kNumEnvs);
  actions.setZero();

  for (int t = 0; t < 10; ++t) { [[maybe_unused]] auto r = env.Step(actions); }

  const auto t0 = std::chrono::high_resolution_clock::now();
  for (int t = 0; t < kNumSteps; ++t) { [[maybe_unused]] auto r = env.Step(actions); }
  const auto t1 = std::chrono::high_resolution_clock::now();

  const double elapsed = std::chrono::duration<double>(t1 - t0).count();
  const double total   = static_cast<double>(kNumEnvs) * kNumSteps;
  std::printf("  Elapsed: %.4f s  |  %.0f env-steps/s  |  %.2f us/env-step\n",
              elapsed, total / elapsed, elapsed * 1e6 / total);
}

}  // namespace
}  // namespace test
}  // namespace gpu_environments
}  // namespace drake

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------

int main() {
  using namespace drake::gpu_environments::test;

  std::printf("==========================================================\n");
  std::printf("  GPU Parallel Pendulum — 128 Environment Test Suite\n");
#ifdef DRAKE_GPU_AVAILABLE
  std::printf("  Build mode: GPU (CUDA)\n");
#else
  std::printf("  Build mode: CPU (CUDA not available)\n");
#endif
  std::printf("==========================================================\n\n");

  try {
    TestConstructionAndReset();
    TestSingleStep();
    TestPhysicsCorrectness();
    TestEpisodeAutoReset();
    TestObservationRange();
    BenchmarkPendulum();
  } catch (const std::exception& e) {
    std::fprintf(stderr, "\n[FATAL] %s\n", e.what());
    return 1;
  }

  std::printf("\n==========================================================\n");
  std::printf("  ALL PENDULUM TESTS PASSED\n");
  std::printf("==========================================================\n");
  return 0;
}
