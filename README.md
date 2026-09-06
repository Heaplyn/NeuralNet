# 🧠 NeuralNet: RingWrapper Causal Transformer & Meta-Learning Optimization Engine

A high-performance, modular C++17 causal transformer language model and deep learning engine built from scratch with strict Ring Architectural Layering, CUDA hardware acceleration with OpenMP fallback, an online **Meta-Neural Loss & Step Optimizer**, **4-Formula Dynamic Weight Physics**, and hierarchical recursive thought chains.

---

## 🏛️ System Architecture & Layered Ring Topology

The codebase adheres strictly to the **Layered Ring Architectural Hierarchy**:
A module in **Ring $N$** can depend on modules in **Ring $M$** if and only if **$M \le N$**.

```
Ring 0 (Foundation) ---> Ring 1 (Layers & Physics) ---> Ring 2 (Models) ---> Ring 3 (Training & Data) ---> Ring 4 (Application)
```

| Layer | Namespace | Key Components |
| :--- | :--- | :--- |
| **Ring 0** | `ring0::` | Contiguous `Matrix` & `Tensor3D` math, cache-blocked matrix multiplication, SIMD activations (GELU, SwiGLU, RMSNorm, Softmax), `Loss` calculus, `LossDerivativePyramid`, `TaylorTrajectoryPredictor` (nth-order loss foresight), `CUDAMathEngine`, and `RuntimeConfig`. |
| **Ring 1** | `ring1::` | `Attention` (Grouped-Query Attention + ALiBi), `TransformerBlock`, `RecursiveLayer` (Hierarchical Thought Trees), `MetaLossOptimizer` (3-layer online policy MLP), `MultiFormulaKernel` (4-Formula Weight Physics), and `AdamW` (Fisher Natural Gradient + Nesterov). |
| **Ring 2** | `ring2::` | `TransformerLM` (10-Layer Causal Transformer Decoder), `BPETokenizer` (512 Subword Vocabulary), `VocabManager` (16 Semantic Concept Clusters & Neurogenesis), and `KVCache` streaming generation. |
| **Ring 3** | `ring3::` | `TextDataset` (Token Relevancy & Interpolated Parsing), `LLMTrainer` (Dynamic LR schedules, Loss Derivative Pyramid, 3D Progressive Curriculum), and multi-file checkpoint bundle manager. |
| **Ring 4** | `ring4::` / CLI | Interactive command-line interface, real-time code streaming, and thought trace diagnostics (`src/main.cpp`). |

---

## ⚡ Features

### 1. 🧠 Meta-Neural Loss & Step Optimizer Network (`Ring 1`)
- Replaces static learning rate schedulers with an online auxiliary neural network ($12 \to 32 \to 16 \to 4$).
- Observes 8 continuous optimization telemetry signals ($\mathcal{L}_t$, $\Delta\mathcal{L}$, $\Delta^2\mathcal{L}$, $\frac{d\mathcal{L}}{d\text{Pen}}$, gradient variance, alignment, entropy, learning rate) **plus 4 Taylor-forecast signals** (predicted $\Delta\mathcal{L}$, predicted net change, trajectory reward, forecast confidence).
- Predicts dynamic continuous scaling factors with tightly bounded safety envelopes:
  - $\mathcal{M}_{\mathcal{L}} \in [0.85, 1.25]$: Dynamic Loss Magnitude Multiplier
  - $\gamma_t \in [0.0, 0.35]$: Adaptive Focal Loss exponent for unigram plateau breakout
  - $\Delta \alpha_t \in [0.85, 1.20]$: Real-time learning rate step modulator
  - $\kappa_t \in [0.80, 1.20]$: Rayleigh curvature preconditioning factor
- Features **strided policy gradient updates** (every 4 steps), velocity-based step deceleration ($\text{meta\_step\_scale}$ dampening), self-lowering upon adverse loss deltas ($\Delta \mathcal{L} > 0.01$), and a rolling variance health monitor over 30 steps.
- Learns online via a policy-gradient update across **all** layers ($W_1, W_2, W_3$) with annealed output exploration noise, trained against a blend of the realized reward $-(\mathcal{L}_t - \mathcal{L}_{t-1})$ and the Taylor **trajectory reward** $R_t$.

### 2. 🔮 Taylor Loss-Trajectory Predictor & nth-Order Foresight (`Ring 0`)
Turns the engine from *reactive* into *anticipatory* by forecasting where the loss is going, purely from its step-to-step history (a few dozen FLOPs/step — no tensor work):
- Builds a backward finite-difference ladder $\nabla^{j}\mathcal{L}_t$ up to order $n$ (the discrete derivatives) and extrapolates the next $K$ losses via damped **Newton–Gregory** expansion: $\hat{\mathcal{L}}_{t+k} = \sum_j \binom{k+j-1}{j}\lambda_j \nabla^{j}\mathcal{L}_t$.
- **Recursive parent↔child coupling**: each derivative order self-tunes its trust $\lambda_j$ from prediction error, gated top-down by its parent and corrected bottom-up by its child — orders interact rather than acting alone.
- Emits a discounted **trajectory reward** plus anticipatory **penalty**, **learning-rate** (a forward-looking Armijo), and **curvature** signals, and drives **forecast-sized neurogenesis** with a self-determining width ceiling.
- Anti-explosion guards (per-order trust + tanh soft-clamp) tame the Runge phenomenon of high-order extrapolation.

### 3. 🔬 4-Formula Dynamic Weight Physics (`Ring 1`)
Rather than treating all network parameters uniformly, the optimizer evaluates each weight's importance via **First-Order Taylor Salience $\times$ Empirical Fisher Information Metric** $\mathcal{I}(w_i) \in [0, 1]$:
1. **Formula 1: Riemannian Geodesic Natural Gradient** ($\mathcal{I} > 0.65$): Updates critical routing hubs along the true probability manifold.
2. **Formula 2: Curvature-Scaled Nesterov Acceleration** ($0.45 < \mathcal{I} \le 0.65$): Applies Hamiltonian phase-space lookahead and curvature dampening.
3. **Formula 3: Variance-Bounded Decoupled AdamW** ($0.20 < \mathcal{I} \le 0.45$): Standard moment updates with numerical variance clipping.
4. **Formula 4: Inertial Sparse Decay & Pruning** ($\mathcal{I} \le 0.20$): Halves gradient steps and doubles weight decay to actively prune noise.

### 4. 🎯 Token Relevancy & Interpolated Neighborhood Context Parsing (`Ring 3`)
- Evaluates token information entropy and transition surprise $R(t) \in [0.0, 1.0]$.
- Non-linearly interpolates parsing radius: $W(R) = \text{round}(W_{\min} + R^\alpha \cdot (W_{\max} - W_{\min}))$.
- Applies distance-decayed cosine weighting: $r(d) = R_{\text{anchor}} \cdot \max(0, \cos(\frac{\pi d}{2 (W + 1)}))$.

### 5. 🌲 Hierarchical Recursive Thought Layer & Reflection Loops (`Ring 1`)
- Multi-depth tree of internal cognitive sub-reasoners allowing latent deliberation before token generation.
- Supports multi-pass self-reflection cycles ($C \ge 1$) with residual momentum damping.

### 6. 🚀 Fast-Start & Training-Stability Engine (`Ring 1`/`Ring 2`/`Ring 3`)
Getting cross-entropy loss to *start* low and *descend* without detonating, via orthogonal, scale-aware mechanisms:
- **🎯 Log-Unigram Head-Bias Init:** seed the output bias with $b_{\text{head}}[c] = \log P(c)$ so step-0 output *is* the corpus word-frequency distribution. Step-0 loss drops from $\ln V = 9.21$ to unigram entropy $H(p) \approx 7.42$.
- **📉 Loss-Adaptive Trust Region:** per-element step cap $|\Delta w|_{\max}$ moves inversely with loss — tight ($0.12$) when loss is high and unstable, loosening to $0.60$ as loss converges.
- **📐 Dimension-Aware Damping:** scale steps by $\sqrt{d_{\text{ref}}/\max(d_{\text{ref}}, \dim)}$ so wide tied vocabulary embeddings ($10000\times128$) are throttled ~9× without slowing down 128-wide layers.
- **🛡️ Tri-Level Mistake Checkpoint Repulsion Engine:** 30-slot negative-memory FIFO queue that captures normalized triple-signatures (gradient direction, latent representation distribution, and weight fingerprint). Penalizes proximity to past failure states using a sub-linear square-rooted inverted difference barrier $\mathcal{P} = \sqrt{\max(0, S)}$:
  - *Level A (Gradient Space - Every Step):* Orthogonalizes gradient updates away from bad gradient directions $\hat{g}_{\text{bad}}$.
  - *Level B (Representation Space - Every Step):* Penalizes entropy collapse directly within token prediction logits.
  - *Level C (Parameter Space - Every 10 Steps):* Applies gentle geometric parameter displacement away from divergence basins.
- **📈 Dynamic LR Slew-Rate Limiter:** caps maximum single-step learning rate growth to $+10\%$ per step ($\text{LR}_{t+1} \le \text{LR}_t \times 1.10$), preventing runaway surge spikes during rapid loss drops.
- **🌱 Soft Residual Scaling on Depth Growth:** when unlocking deeper transformer layers (4 $\to$ 6 $\to$ 8 $\to$ 10), newly activated block projections ($W_o, W_{\text{down}}$) are scaled by $0.10\times$ so new layers initially act as near-identity mappings without destabilizing converged representations.
- **⏱️ Multi-Phase Recovery Cooldowns:** dedicated cooldown timers (`bad_batch_cooldown`, `depth_jump_cooldown`, `context_jump_cooldown`) apply temporary 40–50% LR damping following weight rollbacks or structural expansions to settle network activations.

### 7. 📡 Asynchronous Streaming & Chrono Engine (`Ring 3`)
- **Background Data Streamer:** dedicated background worker thread ingests, tokenizes, and appends slices of large corpus files directly from disk into the active token buffer while forward/backward tensor passes execute without I/O stutter.
- **Chrono Async Engine:** concurrent multi-threaded engine orchestrating 5 periodic asynchronous subsystems:
  1. *Online Meta-Optimizer Policy Stepping*
  2. *Taylor Trajectory Horizon Extrapolation*
  3. *Calculus of Constructions (CoC) Proof Verification*
  4. *Semantic Vocab Concept Cluster Mining*
  5. *Stability Watchdog & Gradient Norm Auditing*

---

## 🔢 Recognition Benchmark: MNIST

The dense recognition path trains a `784 -> 128 -> 10` classifier on real MNIST IDX images. The reported accuracy is measured against held-out labeled test images, not the training set. The same recognition trainer also supports the harder Fashion-MNIST dataset and the seeded noisy A-Z benchmark.

### Representative MNIST Run

This run used 20,000 training images and 10,000 held-out test images. Meta-LR and Taylor-LR are the live adaptive multipliers reported by the trainer:

```text
[MNIST] Training on 20000 real images and evaluating on 10000 held-out images.
  MNIST epoch 1 | Loss: 1.5889 | Held-out accuracy: 80.0% | Meta LR: 1.006 | Taylor LR: 1.184
  MNIST epoch 2 | Loss: 0.8021 | Held-out accuracy: 83.7% | Meta LR: 1.005 | Taylor LR: 0.746
  MNIST epoch 4 | Loss: 0.4507 | Held-out accuracy: 89.2% | Meta LR: 0.986 | Taylor LR: 0.820
  MNIST epoch 6 | Loss: 0.3699 | Held-out accuracy: 89.8% | Meta LR: 0.992 | Taylor LR: 0.975
  MNIST epoch 8 | Loss: 0.2971 | Held-out accuracy: 91.8% | Meta LR: 0.969 | Taylor LR: 0.869
  MNIST epoch 10 | Loss: 0.3004 | Held-out accuracy: 92.4% | Meta LR: 0.942 | Taylor LR: 0.897
  MNIST epoch 12 | Loss: 0.2578 | Held-out accuracy: 92.2% | Meta LR: 0.988 | Taylor LR: 1.052
  MNIST epoch 14 | Loss: 0.2121 | Held-out accuracy: 93.2% | Meta LR: 0.947 | Taylor LR: 1.041
  MNIST epoch 16 | Loss: 0.2047 | Held-out accuracy: 93.7% | Meta LR: 0.948 | Taylor LR: 1.038
  MNIST epoch 18 | Loss: 0.1754 | Held-out accuracy: 94.0% | Meta LR: 0.958 | Taylor LR: 0.905
  MNIST epoch 20 | Loss: 0.1646 | Held-out accuracy: 93.9% | Meta LR: 0.990 | Taylor LR: 0.953
  MNIST epoch 22 | Loss: 0.1531 | Held-out accuracy: 94.0% | Meta LR: 0.985 | Taylor LR: 0.855
  MNIST epoch 24 | Loss: 0.1335 | Held-out accuracy: 93.6% | Meta LR: 0.982 | Taylor LR: 0.858
  MNIST epoch 26 | Loss: 0.1260 | Held-out accuracy: 93.7% | Meta LR: 0.925 | Taylor LR: 0.964
  MNIST epoch 28 | Loss: 0.1208 | Held-out accuracy: 95.2% | Meta LR: 0.942 | Taylor LR: 0.921
  MNIST epoch 30 | Loss: 0.0988 | Held-out accuracy: 95.7% | Meta LR: 0.989 | Taylor LR: 1.021
  MNIST epoch 32 | Loss: 0.0806 | Held-out accuracy: 95.5% | Meta LR: 0.941 | Taylor LR: 1.017
  MNIST epoch 34 | Loss: 0.1150 | Held-out accuracy: 94.6% | Meta LR: 0.921 | Taylor LR: 1.027
  MNIST epoch 36 | Loss: 0.0602 | Held-out accuracy: 96.2% | Meta LR: 0.998 | Taylor LR: 1.012
  MNIST epoch 38 | Loss: 0.0617 | Held-out accuracy: 96.4% | Meta LR: 0.980 | Taylor LR: 0.887
  MNIST epoch 40 | Loss: 0.0435 | Held-out accuracy: 96.5% | Meta LR: 1.003 | Taylor LR: 0.962
  MNIST epoch 42 | Loss: 0.0431 | Held-out accuracy: 96.6% | Meta LR: 1.004 | Taylor LR: 1.008
  MNIST epoch 44 | Loss: 0.0341 | Held-out accuracy: 96.7% | Meta LR: 1.005 | Taylor LR: 0.893
  MNIST epoch 46 | Loss: 0.0470 | Held-out accuracy: 96.7% | Meta LR: 0.936 | Taylor LR: 1.009
  MNIST epoch 48 | Loss: 0.0246 | Held-out accuracy: 96.8% | Meta LR: 0.999 | Taylor LR: 1.005
  MNIST epoch 50 | Loss: 0.0354 | Held-out accuracy: 96.8% | Meta LR: 1.004 | Taylor LR: 1.005
  MNIST epoch 52 | Loss: 0.0187 | Held-out accuracy: 97.0% | Meta LR: 0.998 | Taylor LR: 1.004
```

### Benchmark Parameters

| Parameter | Value |
| :--- | :--- |
| Input features | 784 normalized grayscale pixels (28 x 28) |
| Hidden layer | 128 neurons, ReLU |
| Output layer | 10 linear logits, classes 0-9 |
| Training samples | 20,000 |
| Held-out samples | 10,000 |
| Batch size | 128 |
| Base learning rate | 0.01 |
| Weight decay | 0.005 |
| Gradient clipping | Global norm 1.0 |
| Optimizer | AdamW with Fisher/Nesterov and multi-formula routing |
| Meta optimizer | Enabled; dynamically modulates loss, LR, and curvature |
| Taylor foresight | Enabled; direct predictive LR and curvature modulation |
| Taylor/meta blend | 0.5 Taylor forecast weight |
| Growth controller | Enabled with width safety bounds |
| Evaluation metric | Top-1 accuracy on held-out labels |

The best displayed result in this run is **97.0% held-out accuracy** at epoch 52. The Meta LR and Taylor LR values are multipliers around the configured base learning rate; they are not replacement learning-rate values.

---

## 📚 Complete Obsidian Knowledge Vault

A full 41-document **Obsidian Knowledge Vault** is included in [`NeuralNet Obsidian/`](./NeuralNet%20Obsidian/):
- **Overview & Architecture**: Beginner roadmap, system map, Ring hierarchy, and 5-phase roadmap.
- **Ring 0**: Tensor memory layouts, cache-blocked matrix multiplication, SIMD activations, CUDA engines, and the **Taylor Loss-Trajectory Predictor**.
- **Ring 1**: Attention mechanics, recursive thought chains, meta-loss optimization, 4-formula weight physics, and **training-stability / fast-start descent** (bias init, trust region, dimension damping).
- **Ring 2**: TransformerLM decoder, BPE subword tokenizer, and semantic VocabManager.
- **Ring 3**: Token relevancy parsing, 3D progressive curriculum, **Mistake Checkpoint Memory**, **Chrono Async Engine**, and background data streaming.
- **Theoretical Foundations**: Information geometry, Riemannian manifolds, Fisher metrics, Calculus of Constructions, and Adaptive Focal Loss theory.

---

## 🛠️ Building & Running

### Requirements
- C++17 compliant compiler (MSVC 19.50+, GCC 9+, Clang 10+)
- CMake 3.20+
- OpenMP 2.0+ (Optional for parallel multicore acceleration)

### Build Instructions
```bash
# Configure build with CMake
cmake -B build -S . -G "Visual Studio 18 2026" -A x64

# Compile Release binary
cmake --build build --config Release

# Run training benchmark with 100 steps and real-time dashboard
./build/Release/nn_demo.exe --steps 100
```

---

## 📜 License
MIT License. Created by Heaplyn.
