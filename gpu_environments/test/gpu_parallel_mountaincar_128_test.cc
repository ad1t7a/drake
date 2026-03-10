/// @file gpu_parallel_mountaincar_128_test.cc
///
/// Tests 128 MountainCar environments in parallel.

#include "drake/gpu_environments/gpu_parallel_mountaincar_env.h"

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
  std::printf("[TEST 1] MountainCar: Construction and Reset\n");

  GpuParallelMountainCarEnv env(kNumEnvs);
  assert(env.num_envs() == kNumEnvs);

  std::vector<uint32_t> seeds(kNumEnvs);
  for (int i = 0; i < kNumEnvs; ++i) seeds[i] = static_cast<uint32_t>(i + 1);
  env.Reset(seeds);

  EnvMatrix obs = env.GetObservations();
  assert(obs.rows() == GpuParallelMountainCarEnv::kObsDim);
  assert(obs.cols() == kNumEnvs);

  for (int j = 0; j < kNumEnvs; ++j) {
    const float pos = obs(0, j);
    const float vel = obs(1, j);
    if (pos < kMCMinPos - 0.01f || pos > kMCMaxPos + 0.01f) {
      std::fprintf(stderr, "FAIL: pos[%d]=%.4f out of [%.2f,%.2f]\n",
                   j, pos, kMCMinPos, kMCMaxPos);
      throw std::runtime_error("Initial position out of bounds");
    }
    if (vel != 0.0f) {
      std::fprintf(stderr, "FAIL: initial vel[%d]=%.4f != 0\n", j, vel);
      throw std::runtime_error("Initial velocity not zero");
    }
    // Init position in [-0.6, -0.4].
    if (pos < -0.61f || pos > -0.39f) {
      std::fprintf(stderr,
                   "FAIL: init pos[%d]=%.4f not in [-0.6,-0.4]\n", j, pos);
      throw std::runtime_error("Initial position outside expected range");
    }
  }

  std::printf("  PASS: %d MC environments reset; pos in [-0.6,-0.4], vel=0.\n",
              kNumEnvs);
}

// ---------------------------------------------------------------------------
// Test 2: Single Step
// ---------------------------------------------------------------------------

void TestSingleStep() {
  std::printf("[TEST 2] MountainCar: Single Step\n");

  GpuParallelMountainCarEnv env(kNumEnvs);
  std::vector<uint32_t> seeds(kNumEnvs);
  for (int i = 0; i < kNumEnvs; ++i) seeds[i] = static_cast<uint32_t>(i + 7);
  env.Reset(seeds);

  EnvMatrix actions(GpuParallelMountainCarEnv::kActionDim, kNumEnvs);
  actions.setZero();
  StepResult res = env.Step(actions);

  assert(res.observations.rows() == GpuParallelMountainCarEnv::kObsDim);
  assert(res.rewards.size() == kNumEnvs);

  for (int i = 0; i < kNumEnvs; ++i) {
    // Zero action → -0.1*0² = 0 reward (not done so reward = -0).
    // Actually reward = -0.1 * 0^2 = 0.0
    if (res.done[i]) {
      std::fprintf(stderr, "FAIL: env %d done after 1 step\n", i);
      throw std::runtime_error("Unexpected done after 1 step");
    }
  }

  std::printf("  PASS: single step; no dones.\n");
}

// ---------------------------------------------------------------------------
// Test 3: Physics correctness vs serial reference
// ---------------------------------------------------------------------------

void TestPhysicsCorrectness() {
  std::printf("[TEST 3] MountainCar: Physics correctness\n");

  constexpr int   kCompare = 100;
  constexpr float kAction  = 0.5f;

  std::vector<uint32_t> seeds(kNumEnvs);
  for (int i = 0; i < kNumEnvs; ++i) seeds[i] = static_cast<uint32_t>(i + 1);

  auto xf = [](uint32_t& s, float scale) -> float {
    s ^= s << 13; s ^= s >> 17; s ^= s << 5;
    return (static_cast<float>(s) / static_cast<float>(0xFFFFFFFFu) * 2.0f - 1.0f) * scale;
  };

  float s_pos, s_vel;
  {
    uint32_t s = seeds[0];
    if (s == 0) s = 0xBEEFu;
    s_pos = -0.5f + xf(s, 1.0f) * 0.1f;
    s_vel = 0.0f;
  }

  std::vector<float> ref_pos(kCompare), ref_vel(kCompare);
  {
    float p = s_pos, v = s_vel, rew; bool done;
    for (int t = 0; t < kCompare; ++t) {
      MountainCarDynamicsStep(p, v, kAction, rew, done);
      ref_pos[t] = p;
      ref_vel[t] = v;
      if (done) break;  // stop if goal reached (unlikely with 0.5 force)
    }
  }

  GpuParallelMountainCarEnv env(kNumEnvs);
  env.Reset(seeds);

  EnvMatrix actions(GpuParallelMountainCarEnv::kActionDim, kNumEnvs);
  for (int i = 0; i < kNumEnvs; ++i) actions(0, i) = kAction;

  constexpr float kTol = 1e-5f;
  int mismatches = 0;
  for (int t = 0; t < kCompare; ++t) {
    StepResult res = env.Step(actions);
    const float pp = res.observations(0, 0);
    const float vp = res.observations(1, 0);
    if (std::abs(pp - ref_pos[t]) > kTol ||
        std::abs(vp - ref_vel[t]) > kTol) {
      if (mismatches < 3)
        std::fprintf(stderr,
            "  Mismatch t=%d: par(%.6f,%.6f) ref(%.6f,%.6f)\n",
            t, pp, vp, ref_pos[t], ref_vel[t]);
      ++mismatches;
      if (res.done[0]) break;
    }
  }

  if (mismatches > 0) {
    std::fprintf(stderr, "FAIL: %d mismatches\n", mismatches);
    throw std::runtime_error("MountainCar physics mismatch");
  }
  std::printf("  PASS: env-0 matches serial for %d steps (tol=%.0e).\n",
              kCompare, static_cast<double>(kTol));
}

// ---------------------------------------------------------------------------
// Test 4: State bounds are always respected
// ---------------------------------------------------------------------------

void TestStateBounds() {
  std::printf("[TEST 4] MountainCar: State bounds validation\n");

  GpuParallelMountainCarEnv env(kNumEnvs);
  std::vector<uint32_t> seeds(kNumEnvs);
  for (int i = 0; i < kNumEnvs; ++i) seeds[i] = static_cast<uint32_t>(i + 55);
  env.Reset(seeds);

  EnvMatrix actions(GpuParallelMountainCarEnv::kActionDim, kNumEnvs);

  for (int t = 0; t < kNumSteps; ++t) {
    for (int i = 0; i < kNumEnvs; ++i) {
      actions(0, i) = (t % 3 == 0) ? 1.0f : ((t % 3 == 1) ? -1.0f : 0.0f);
    }
    StepResult res = env.Step(actions);
    for (int j = 0; j < kNumEnvs; ++j) {
      const float p = res.observations(0, j);
      const float v = res.observations(1, j);
      if (!std::isfinite(p) || !std::isfinite(v)) {
        std::fprintf(stderr,
                     "FAIL: non-finite state at step %d env %d\n", t, j);
        throw std::runtime_error("Non-finite MC state");
      }
      if (p < kMCMinPos - 0.001f || p > kMCMaxPos + 0.001f) {
        std::fprintf(stderr,
                     "FAIL: pos=%.4f out of bounds at step %d\n", p, t);
        throw std::runtime_error("MC position out of bounds");
      }
      if (v < kMCMinVel - 0.001f || v > kMCMaxVel + 0.001f) {
        std::fprintf(stderr,
                     "FAIL: vel=%.5f out of bounds at step %d\n", v, t);
        throw std::runtime_error("MC velocity out of bounds");
      }
    }
  }

  std::printf("  PASS: %d x %d steps — pos in [%.2f,%.2f] vel in [%.3f,%.3f].\n",
              kNumEnvs, kNumSteps, kMCMinPos, kMCMaxPos, kMCMinVel, kMCMaxVel);
}

// ---------------------------------------------------------------------------
// Benchmark
// ---------------------------------------------------------------------------

void BenchmarkMC() {
  std::printf("[BENCH ] MountainCar: %d envs x %d steps\n", kNumEnvs, kNumSteps);

  GpuParallelMountainCarEnv env(kNumEnvs);
  std::vector<uint32_t> seeds(kNumEnvs);
  for (int i = 0; i < kNumEnvs; ++i) seeds[i] = static_cast<uint32_t>(i + 1);
  env.Reset(seeds);

  EnvMatrix actions(GpuParallelMountainCarEnv::kActionDim, kNumEnvs);
  for (int i = 0; i < kNumEnvs; ++i) actions(0, i) = 1.0f;

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

int main() {
  using namespace drake::gpu_environments::test;
  std::printf("==========================================================\n");
  std::printf("  GPU Parallel MountainCar — 128 Environment Test Suite\n");
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
    TestStateBounds();
    BenchmarkMC();
  } catch (const std::exception& e) {
    std::fprintf(stderr, "\n[FATAL] %s\n", e.what());
    return 1;
  }

  std::printf("\n==========================================================\n");
  std::printf("  ALL MOUNTAINCAR TESTS PASSED\n");
  std::printf("==========================================================\n");
  return 0;
}
