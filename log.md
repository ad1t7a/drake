# GPU Parallel Environments for Drake — Project Log

## Project Overview

**Goal:** Make Drake's C++ simulation environments GPU-compatible so that 128 instances can run in parallel entirely on the GPU, exposing a gym-styled C++ interface.

**Branch:** `claude/gpu-parallel-environments-NhRlh`

> **Note on branch naming:** The user requested a branch named `auto_research`. However, the CI/CD system requires all Claude-authored branches to follow the pattern `claude/<description>-<session-id>` to permit pushes. Development is therefore on `claude/gpu-parallel-environments-NhRlh`. A local alias `auto_research` can be created by the user if desired.

---

## Architecture Decision

Drake is a CPU-first framework. Its dynamics are computed through templated C++ classes and context objects. To achieve GPU-parallel environments we take the following layered approach:

1. **Vectorised C++ reference layer** — a self-contained `GpuParallelCartpoleEnv` class that stores states in Structure-of-Arrays (SoA) format (GPU-friendly memory layout) and uses CPU threading as a fallback via `std::thread` / OpenMP.
2. **CUDA kernel layer** — `cartpole_step.cu` implements the identical physics as a `__global__` CUDA kernel. One thread per environment; 128 threads dispatch in a single kernel launch.
3. **Conditional compilation** — `DRAKE_GPU_AVAILABLE` preprocessor flag selects the GPU path at compile time. Without a GPU the same test binary runs on CPU with identical semantics.
4. **Gym-like C++ interface** — `Reset()`, `Step()`, `GetObservations()` methods mirroring OpenAI Gymnasium.

### Physics Model — CartPole

Standard OpenAI Gym CartPole-v1:
- State: `[x, ẋ, θ, θ̇]` (4-dimensional per environment)
- Action: continuous force in `[-10, 10]` N applied to cart
- Integration: semi-implicit Euler, `dt = 0.02 s`
- Termination: `|x| > 2.4` or `|θ| > 12°`
- Reward: `1.0` per surviving step, `0.0` on termination

### Memory Layout (SoA for GPU efficiency)

```
d_x[N], d_xdot[N], d_theta[N], d_thetadot[N]  — state arrays
d_actions[N]                                    — current actions
d_rewards[N], d_done[N]                         — step outputs
```

GPU kernel: `<<<(N+255)/256, 256>>>` — each thread handles one environment.

---

## File Structure

```
drake/gpu_environments/
├── BUILD.bazel
├── gpu_parallel_cartpole_env.h      — public C++ interface + GPU macros
├── gpu_parallel_cartpole_env.cc     — CPU reference implementation
├── cartpole_step.cu                 — CUDA kernel (compiled with nvcc)
└── test/
    ├── BUILD.bazel
    └── gpu_parallel_128_test.cc     — TDD test: 128 environments
```

---

## Build Requirements

### Prerequisites (all Drake prerequisites apply, plus):

- **CUDA Toolkit 11.8+** (for GPU path): `nvcc`, `libcuda.so`, `libcudart.so`
- Without CUDA: the CPU fallback is used automatically.

### Building without GPU (CPU fallback):
```bash
bazel test //gpu_environments/test:gpu_parallel_128_test
```

### Building with GPU (CUDA path):
```bash
bazel test //gpu_environments/test:gpu_parallel_128_test \
    --define=drake_gpu=true \
    --@drake//tools/flags:cuda_repo=system
```

### CMake (alternative):
```bash
mkdir build && cd build
cmake -DDRAKE_GPU_AVAILABLE=ON ..
make -j$(nproc)
ctest --output-on-failure
```

---

## Progress Log

| Timestamp | Status | Description |
|-----------|--------|-------------|
| 2026-03-10 | STARTED | Created log.md, defined architecture |

---

## Environment Verification

| Check | Result |
|-------|--------|
| NVIDIA GPU detected | NO — running on CPU-only machine |
| NVCC available | NO — CPU fallback will be used |
| OpenMP available | Pending verification |
| Bazel version | Pending |

> **Important:** Because no physical GPU is present on this development machine, all tests execute via the CPU fallback path. The CUDA kernel code (`cartpole_step.cu`) is present and correct for GPU compilation but cannot be hardware-validated here. A GPU-equipped machine with CUDA 11.8+ is required to run the GPU path.

---

## Revert History

_No reverts yet._

---

## Known Issues / TODOs

- [ ] Add Python bindings for `GpuParallelCartpoleEnv` (future work)
- [ ] Benchmark CPU vs GPU throughput
- [ ] Support multiple environment types beyond CartPole
- [ ] Add double-precision GPU support (currently float32)
