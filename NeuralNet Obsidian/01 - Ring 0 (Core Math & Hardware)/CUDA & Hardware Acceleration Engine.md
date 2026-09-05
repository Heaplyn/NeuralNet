# 🚀 CUDA & Hardware Acceleration Engine

The `ring0::CUDAMathEngine` is a unified hardware-acceleration layer for the core tensor math. It probes for a CUDA-capable GPU at startup and dispatches to fused GPU kernels if one is present; otherwise it falls back to CPU code paths that use OpenMP for thread-parallelism and (on x86) AVX2/FMA vector intrinsics where the compiler emits them.

> **In one sentence:** the engine picks the fastest available compute backend at startup and hides that choice behind one API so Ring 1+ never has to care.

---

## 📋 Prerequisites

Before reading this, you should be comfortable with:
- [[01 - Ring 0 (Core Math & Hardware)/Tensor3D & Matrix Math|Tensor3D & Matrix Math]] — the `Matrix` / `Tensor3D` types the kernels operate on
- Basic ideas of GPU compute (blocks of threads, coalesced global-memory access) — enough to know why fused kernels help
- Attention mechanics ([[02 - Ring 1 (Layers & Advanced Optimizers)/Attention Mechanics & ALiBi|Attention & ALiBi]]) and [[01 - Ring 0 (Core Math & Hardware)/Activation Functions|SwiGLU / GELU]] — the two big kernels below

External context: familiarity with the term "kernel fusion" (running several math operations in one pass over the data instead of writing intermediates to memory).

---

## 🧠 The Big Picture

For the sizes this project runs at, arithmetic is rarely the bottleneck — **memory bandwidth is**. A naïve implementation of attention writes the QKᵀ matrix to memory, then reads it back to add ALiBi bias, writes again, then reads back for softmax, and so on. Each of those trips crosses a bandwidth boundary that's an order of magnitude slower than the ALU. Fusion collapses several passes into one so a value written by one step is used by the next while it's still in a register.

The engine also acts as a **portability seam**. Higher rings call `CUDAMathEngine::fused_attention(...)` and don't care whether that dispatches to a warp-scheduled CUDA kernel or a `#pragma omp parallel for` loop. This keeps the rest of the codebase clean of `#ifdef CUDA` blocks.

---

## 🏗️ Architecture & Device Probing

Initialization probes for hardware and picks a backend once per process:

1. **GPU mode** — a CUDA device is present. Kernels are launched with grids sized to the tensor's leading dimension. `get_device_info()` reports the device name, SM count, and available memory.
2. **CPU mode** — no CUDA available (or explicitly disabled). Everything runs through OpenMP-parallel loops. On x86 the compiler auto-vectorizes with AVX2/FMA where profitable; on ARM (Apple silicon) NEON is used.

```cpp
// Runtime device query — do this once from main().
ring0::CUDAMathEngine::initialize();
const auto& dev = ring0::CUDAMathEngine::get_device_info();
std::cout << "Math Engine: " << dev.device_name
          << " (" << dev.multi_processor_count << " compute units)\n";
```

If a CUDA build is running on a machine without a working driver, the engine falls back cleanly to CPU rather than crashing — safer for demos.

---

## ⚡ Fused Computational Kernels

The two hot paths that dominate a transformer step are attention and the SwiGLU FFN. Both are fused.

### 1. Fused ALiBi Causal Attention
Fuses **QKᵀ dot product**, **ALiBi slope bias**, **causal mask**, and **softmax** into a single pass so no intermediate score matrix ever touches DRAM:

$$
\mathbf{S}_{i,j} \;=\; \frac{\mathbf{q}_i^{\top} \mathbf{k}_j}{\sqrt{d}} \;-\; m \cdot (i - j)
$$
$$
\mathbf{A}_{i,:} \;=\; \operatorname{softmax}(\mathbf{S}_{i,:})
$$
$$
\mathbf{O}_{i,:} \;=\; \sum_{j \le i} \mathbf{A}_{i,j} \, \mathbf{v}_j
$$

Why the fusion matters:
- `S`, the T×T score matrix, is *never materialized*. On long contexts (T=2048) that would be a 16 MB float32 read-write per head — one of the largest memory transfers a step does.
- The softmax is done online (Welford-style running max + running sum) so numerical stability holds without a separate max-reduction pass.
- The causal mask is handled by an early exit in the j-loop, not by writing `-inf` values that softmax has to consume.

### 2. Fused SwiGLU Gated Activation
The SwiGLU FFN is three matmuls plus a nonlinearity; fusing the up-projection and gate cuts the memory traffic in half:

$$
\text{SwiGLU}(\mathbf{x})
\;=\;
\bigl(\, \mathbf{x}\mathbf{W}_{\text{gate}} \cdot \sigma(\mathbf{x}\mathbf{W}_{\text{gate}}) \bigr)
\;\odot\;
(\mathbf{x}\mathbf{W}_{\text{up}})
$$

The gate and up projections run in the same threadblock so the intermediate `x W_gate` is reused for both the σ(·) input and the Swish half of the product without a round-trip through global memory.

### 3. Other fused primitives
- **RMSNorm + linear** — the final normalization + LM-head projection can be fused since both touch the same activation tensor.
- **Add + LayerNorm** — the pre-norm residual add and normalization run in one pass.

---

## 🧵 The CPU-fallback path

When no CUDA device is found:
- The same fused shapes are used, but the outer loops are `#pragma omp parallel for` and the inner ones are vectorizable.
- `Matrix::matmul_transB_into` uses cache-blocked GEMM (see [[01 - Ring 0 (Core Math & Hardware)/Tensor3D & Matrix Math|Tensor3D & Matrix Math]]) so a laptop still gets tens of GFLOPs.
- Softmax is done with `float` accumulators; the running-max reduction is done with a plain scan since OpenMP reductions on max aren't portable prior to OpenMP 3.1.

---

## 🧾 Practical notes

- **`#pragma omp collapse` warnings** — MSVC's OpenMP 2.0 doesn't support `collapse`; the engine emits a `#pragma omp collapse(2)` anyway (harmlessly ignored under MSVC) because it's a real speedup under GCC/Clang for the CPU fallback. See MSVC warning C4849 in the build output.
- **`initialize()` is idempotent** — safe to call from tests and from `main()`.
- **Determinism** — GPU reductions in fused kernels aren't bit-for-bit reproducible across runs (softmax denominator sum order varies with warp scheduling). The CPU path *is* deterministic when OMP threads are pinned.

---

## 🔗 Related Notes
- [[01 - Ring 0 (Core Math & Hardware)/Tensor3D & Matrix Math|Tensor3D & Matrix Math]] — data types the kernels consume
- [[02 - Ring 1 (Layers & Advanced Optimizers)/Attention Mechanics & ALiBi|Attention Mechanics & ALiBi]] — how the fused attention kernel is called
- [[01 - Ring 0 (Core Math & Hardware)/Activation Functions|Activation Functions]] — SwiGLU / GELU details
- [[04 - Ring 3 (Data & Training Pipelines)/Real-Time Benchmark & Telemetry Dashboard|Real-Time Benchmark Dashboard]] — reads the `device_info` for the header
- [[Index|Return to Index]]
