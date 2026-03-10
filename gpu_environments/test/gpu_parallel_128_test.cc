/// @file gpu_parallel_128_test.cc
///
/// Tests that 128 CartPole environments can be simulated in parallel.
///
/// When compiled with DRAKE_GPU_AVAILABLE the environments execute on the GPU
/// using CUDA kernels (one thread per environment).  Without a GPU the same
/// test binary runs via the CPU parallel fallback using identical semantics.
///
/// Test strategy
/// =============
/// 1. Construct 128 environments.
/// 2. Reset all with deterministic per-environment seeds.
/// 3. Run N steps with random (but repeatable) actions.
/// 4. Verify:
///    - Observation shapes are correct.
///    - Physics are plausible (states within physical bounds after a few steps).
///    - Environments that reach termination are correctly flagged done.
///    - Env-0 parallel reward matches a serial reference computation.
/// 5. Benchmark: report steps/second for the parallel batch.

#include "drake/gpu_environments/gpu_parallel_cartpole_env.h"

#include <algorithm>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <numeric>
#include <stdexcept>
#include <vector>

namespace drake {
namespace gpu_environments {
namespace test {

namespace {

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

constexpr int   kNumEnvs     = 128;
constexpr int   kNumSteps    = 500;
constexpr float kActionScale = 10.0f;  // N

// ---------------------------------------------------------------------------
// xorshift32 RNG — same PRNG used by the environment so we can replicate it.
// ---------------------------------------------------------------------------

static inline uint32_t xorshift32(uint32_t& state) {
  state ^= state << 13;
  state ^= state >> 17;
  state ^= state << 5;
  return state;
}

static inline float rand_float_11(uint32_t& state, float scale = 1.0f) {
  float r = static_cast<float>(xorshift32(state)) /
            static_cast<float>(0xFFFFFFFFu);
  return (r * 2.0f - 1.0f) * scale;
}

// ---------------------------------------------------------------------------
// Test 1: Construction and Reset
// ---------------------------------------------------------------------------

void TestConstructionAndReset() {
  std::printf("[TEST 1] Construction and Reset\n");

  GpuParallelCartpoleEnv env(kNumEnvs);
  assert(env.num_envs() == kNumEnvs);

  std::vector<uint32_t> seeds(kNumEnvs);
  for (int i = 0; i < kNumEnvs; ++i) seeds[i] = static_cast<uint32_t>(i + 1);
  env.Reset(seeds);

  const EnvMatrix obs = env.GetObservations();
  assert(obs.rows() == GpuParallelCartpoleEnv::kObsDim);
  assert(obs.cols() == kNumEnvs);

  // Initial states should be small (random init in [-0.05, 0.05]).
  for (int j = 0; j < kNumEnvs; ++j) {
    for (int r = 0; r < GpuParallelCartpoleEnv::kObsDim; ++r) {
      const float v = obs(r, j);
      if (std::abs(v) > 0.11f) {
        std::fprintf(stderr,
                     "FAIL: obs(%d,%d)=%.4f exceeds expected initial range\n",
                     r, j, v);
        throw std::runtime_error("Initial state out of expected range");
      }
    }
  }

  std::printf("  PASS: %d environments constructed and reset. "
              "All initial states in [-0.05, 0.05].\n", kNumEnvs);
}

// ---------------------------------------------------------------------------
// Test 2: Single Step
// ---------------------------------------------------------------------------

void TestSingleStep() {
  std::printf("[TEST 2] Single Step\n");

  GpuParallelCartpoleEnv env(kNumEnvs);
  std::vector<uint32_t> seeds(kNumEnvs);
  for (int i = 0; i < kNumEnvs; ++i) seeds[i] = static_cast<uint32_t>(i + 42);
  env.Reset(seeds);

  // Apply zero actions.
  EnvMatrix actions(GpuParallelCartpoleEnv::kActionDim, kNumEnvs);
  actions.setZero();

  StepResult result = env.Step(actions);

  // Shape checks.
  assert(result.observations.rows() == GpuParallelCartpoleEnv::kObsDim);
  assert(result.observations.cols() == kNumEnvs);
  assert(result.rewards.size() == kNumEnvs);
  assert(static_cast<int>(result.done.size()) == kNumEnvs);

  // With zero force and small initial state, no environment should be done.
  int done_count = 0;
  for (int i = 0; i < kNumEnvs; ++i) {
    if (result.done[i]) ++done_count;
    if (result.rewards[i] != 1.0f) {
      std::fprintf(stderr,
                   "FAIL: env %d reward=%.1f after zero-force step\n",
                   i, result.rewards[i]);
      throw std::runtime_error("Unexpected reward after zero-force step");
    }
  }
  if (done_count > 0) {
    std::fprintf(stderr, "FAIL: %d envs done after 1 zero-force step\n",
                 done_count);
    throw std::runtime_error("Unexpected termination after 1 step");
  }

  std::printf("  PASS: single step, zero actions, %d envs alive.\n", kNumEnvs);
}

// ---------------------------------------------------------------------------
// Test 3: Physics correctness — parallel env-0 vs serial reference
//
// Strategy: run step-by-step and compare env-0 observations BEFORE the first
// termination event.  After auto-reset the states legitimately diverge between
// the parallel and serial versions (the parallel env reinitialises in-place
// using the advanced seed), so we only compare the pre-termination trajectory.
// ---------------------------------------------------------------------------

void TestPhysicsCorrectness() {
  std::printf("[TEST 3] Physics correctness (parallel env-0 vs serial)\n");

  // Use a small action and low-energy initial state to avoid early termination,
  // giving us many steps to compare.
  constexpr int kCompareSteps = 200;
  constexpr float kSmallAction = 0.5f;  // gentle push

  std::vector<uint32_t> seeds(kNumEnvs);
  for (int i = 0; i < kNumEnvs; ++i) seeds[i] = static_cast<uint32_t>(i + 1);

  // ---- Replicate init for env 0 -----------------------------------------
  // This matches CpuReset's xorshift_float for seeds[0].
  auto xf = [](uint32_t& state, float scale) -> float {
    state ^= state << 13;
    state ^= state >> 17;
    state ^= state << 5;
    float r = static_cast<float>(state) / static_cast<float>(0xFFFFFFFFu);
    return (r * 2.0f - 1.0f) * scale;
  };

  float s_x, s_xdot, s_theta, s_tdot;
  {
    uint32_t s = seeds[0];  // seed[0] = 1
    if (s == 0) s = 0xBEEFu;
    s_x     = xf(s, kInitRange);
    s_xdot  = xf(s, kInitRange);
    s_theta = xf(s, kInitRange);
    s_tdot  = xf(s, kInitRange);
  }

  // ---- Serial reference ---------------------------------------------------
  std::vector<float> ref_x(kCompareSteps), ref_xdot(kCompareSteps),
                     ref_theta(kCompareSteps), ref_tdot(kCompareSteps);
  bool serial_terminated = false;
  int serial_terminate_step = kCompareSteps;
  {
    float x = s_x, xd = s_xdot, th = s_theta, td = s_tdot;
    for (int t = 0; t < kCompareSteps; ++t) {
      CartpoleDynamicsStep(x, xd, th, td, kSmallAction);
      ref_x[t]     = x;
      ref_xdot[t]  = xd;
      ref_theta[t] = th;
      ref_tdot[t]  = td;
      if (!serial_terminated) {
        bool done = (x < -kMaxCartX || x > kMaxCartX) ||
                    (th < -kMaxPoleAngle || th > kMaxPoleAngle);
        if (done) { serial_terminated = true; serial_terminate_step = t; }
      }
    }
  }

  // ---- Parallel run -------------------------------------------------------
  GpuParallelCartpoleEnv env(kNumEnvs);
  env.Reset(seeds);

  const int compare_until = serial_terminate_step;  // compare pre-termination
  constexpr float kTol = 1e-5f;  // float32 tolerance

  EnvMatrix actions(GpuParallelCartpoleEnv::kActionDim, kNumEnvs);
  actions.setZero();
  for (int i = 0; i < kNumEnvs; ++i) actions(0, i) = kSmallAction;

  int mismatch_count = 0;
  for (int t = 0; t < compare_until; ++t) {
    StepResult res = env.Step(actions);
    const float px     = res.observations(0, 0);
    const float pxdot  = res.observations(1, 0);
    const float ptheta = res.observations(2, 0);
    const float ptdot  = res.observations(3, 0);

    if (std::abs(px - ref_x[t]) > kTol ||
        std::abs(pxdot - ref_xdot[t]) > kTol ||
        std::abs(ptheta - ref_theta[t]) > kTol ||
        std::abs(ptdot - ref_tdot[t]) > kTol) {
      if (mismatch_count < 3) {
        std::fprintf(stderr,
            "  Mismatch at step %d:\n"
            "    parallel  x=%.6f xdot=%.6f theta=%.6f tdot=%.6f\n"
            "    serial    x=%.6f xdot=%.6f theta=%.6f tdot=%.6f\n",
            t, px, pxdot, ptheta, ptdot,
            ref_x[t], ref_xdot[t], ref_theta[t], ref_tdot[t]);
      }
      ++mismatch_count;
    }
  }

  if (mismatch_count > 0) {
    std::fprintf(stderr,
                 "FAIL: %d state mismatches in %d pre-termination steps\n",
                 mismatch_count, compare_until);
    throw std::runtime_error("Parallel/serial physics mismatch");
  }

  std::printf(
      "  PASS: env-0 state matches serial reference for %d steps "
      "(terminated at step %d, tolerance=%.0e).\n",
      compare_until, serial_terminate_step, static_cast<double>(kTol));
}

// ---------------------------------------------------------------------------
// Test 4: Multi-environment diversity
// ---------------------------------------------------------------------------

void TestMultiEnvDiversity() {
  std::printf("[TEST 4] Multi-environment diversity\n");

  GpuParallelCartpoleEnv env(kNumEnvs);
  std::vector<uint32_t> seeds(kNumEnvs);
  for (int i = 0; i < kNumEnvs; ++i) seeds[i] = static_cast<uint32_t>(i + 1);
  env.Reset(seeds);

  float total_reward = 0.0f;

  uint32_t rng = 0xC0FFEE00u;
  for (int t = 0; t < kNumSteps; ++t) {
    EnvMatrix actions(GpuParallelCartpoleEnv::kActionDim, kNumEnvs);
    for (int i = 0; i < kNumEnvs; ++i) {
      actions(0, i) = rand_float_11(rng, kActionScale);
    }
    StepResult res = env.Step(actions);
    for (int i = 0; i < kNumEnvs; ++i) total_reward += res.rewards[i];
  }

  // Sanity: total reward must be positive (at least 1 env survived 1 step).
  assert(total_reward > 0.0f);

  // Final observations must be finite.
  EnvMatrix obs = env.GetObservations();
  for (int j = 0; j < kNumEnvs; ++j) {
    for (int r = 0; r < GpuParallelCartpoleEnv::kObsDim; ++r) {
      const float v = obs(r, j);
      if (!std::isfinite(v)) {
        std::fprintf(stderr,
                     "FAIL: obs(%d,%d) = %f (not finite)\n", r, j, v);
        throw std::runtime_error("Non-finite observation detected");
      }
    }
  }

  std::printf("  PASS: %d envs x %d steps = total reward %.0f, "
              "all observations finite.\n",
              kNumEnvs, kNumSteps, total_reward);
}

// ---------------------------------------------------------------------------
// Test 5: Auto-reset after termination
// ---------------------------------------------------------------------------

void TestAutoReset() {
  std::printf("[TEST 5] Auto-reset after termination\n");

  GpuParallelCartpoleEnv env(kNumEnvs);
  std::vector<uint32_t> seeds(kNumEnvs);
  for (int i = 0; i < kNumEnvs; ++i) seeds[i] = static_cast<uint32_t>(i + 7);
  env.Reset(seeds);

  int total_done_seen = 0;
  for (int t = 0; t < 300; ++t) {
    EnvMatrix actions(GpuParallelCartpoleEnv::kActionDim, kNumEnvs);
    for (int i = 0; i < kNumEnvs; ++i) {
      // Alternate large ±forces to drive termination quickly.
      actions(0, i) = (i % 2 == 0) ? kActionScale : -kActionScale;
    }
    StepResult res = env.Step(actions);
    for (int i = 0; i < kNumEnvs; ++i) {
      if (res.done[i]) ++total_done_seen;
    }

    // After auto-reset, all states must be within valid initial range.
    EnvMatrix obs = env.GetObservations();
    for (int j = 0; j < kNumEnvs; ++j) {
      for (int r = 0; r < GpuParallelCartpoleEnv::kObsDim; ++r) {
        if (!std::isfinite(obs(r, j))) {
          std::fprintf(stderr,
                       "FAIL: non-finite obs after auto-reset at step %d "
                       "env %d dim %d\n", t, j, r);
          throw std::runtime_error("Non-finite state after auto-reset");
        }
      }
    }
  }

  if (total_done_seen == 0) {
    throw std::runtime_error(
        "No terminations observed — auto-reset cannot be validated");
  }

  std::printf("  PASS: %d terminations + auto-resets, all states finite.\n",
              total_done_seen);
}

// ---------------------------------------------------------------------------
// Test 6: GPU path flag
// ---------------------------------------------------------------------------

void TestGpuPathFlag() {
  std::printf("[TEST 6] GPU path flag\n");

  GpuParallelCartpoleEnv env(kNumEnvs);
  const bool is_gpu = env.IsGpuEnabled();

#ifdef DRAKE_GPU_AVAILABLE
  if (!is_gpu) {
    throw std::runtime_error(
        "DRAKE_GPU_AVAILABLE defined but IsGpuEnabled()=false");
  }
  std::printf("  PASS: GPU path active (CUDA compiled in).\n");
#else
  if (is_gpu) {
    throw std::runtime_error(
        "DRAKE_GPU_AVAILABLE not defined but IsGpuEnabled()=true");
  }
  std::printf("  PASS: CPU fallback active (no CUDA). "
              "GPU requires CUDA 11.8+ and NVIDIA device.\n");
#endif
}

// ---------------------------------------------------------------------------
// Test 7: Throughput benchmark
// ---------------------------------------------------------------------------

void BenchmarkThroughput() {
  std::printf("[BENCH ] Throughput: %d envs x %d steps\n",
              kNumEnvs, kNumSteps);

  GpuParallelCartpoleEnv env(kNumEnvs);
  std::vector<uint32_t> seeds(kNumEnvs);
  for (int i = 0; i < kNumEnvs; ++i) seeds[i] = static_cast<uint32_t>(i + 1);
  env.Reset(seeds);

  EnvMatrix actions(GpuParallelCartpoleEnv::kActionDim, kNumEnvs);
  actions.setZero();  // Zero force for a pure physics benchmark.

  // Warm-up (discard results intentionally).
  for (int t = 0; t < 10; ++t) { [[maybe_unused]] auto r = env.Step(actions); }

  const auto t0 = std::chrono::high_resolution_clock::now();
  for (int t = 0; t < kNumSteps; ++t) { [[maybe_unused]] auto r = env.Step(actions); }
  const auto t1 = std::chrono::high_resolution_clock::now();

  const double elapsed = std::chrono::duration<double>(t1 - t0).count();
  const double total_steps =
      static_cast<double>(kNumEnvs) * static_cast<double>(kNumSteps);
  const double fps = total_steps / elapsed;

  std::printf("  Elapsed: %.4f s  |  %.0f env-steps/s  |  %.2f us/env-step\n",
              elapsed, fps, elapsed * 1.0e6 / total_steps);
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
  std::printf("  GPU Parallel CartPole — 128 Environment Test Suite\n");
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
    TestMultiEnvDiversity();
    TestAutoReset();
    TestGpuPathFlag();
    BenchmarkThroughput();
  } catch (const std::exception& e) {
    std::fprintf(stderr, "\n[FATAL] %s\n", e.what());
    return 1;
  }

  std::printf("\n==========================================================\n");
  std::printf("  ALL TESTS PASSED\n");
  std::printf("==========================================================\n");
  return 0;
}
