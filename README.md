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
- Predicts dynamic continuous scaling factors:
  - $\mathcal{M}_{\mathcal{L}} \in [0.2, 4.0]$: Dynamic Loss Magnitude Multiplier
  - $\gamma_t \in [0.0, 3.0]$: Adaptive Focal Loss exponent for unigram plateau breakout
  - $\Delta \alpha_t \in [0.5, 3.0]$: Real-time learning rate step modulator
  - $\kappa_t \in [0.2, 2.5]$: Rayleigh curvature preconditioning factor
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

### 6. 🚀 Fast-Start & Training-Stability Engine (`Ring 1`/`Ring 2`)
Getting cross-entropy loss to *start* low and *descend* without detonating, via three orthogonal, scale-aware mechanisms. See [`Training Stability & Fast-Start Descent`](./NeuralNet%20Obsidian/02%20-%20Ring%201%20(Layers%20%26%20Advanced%20Optimizers)/Training%20Stability%20%26%20Fast-Start%20Descent.md) for full intuition and worked numbers.

**The problem, intuitively:** an untrained model over a 10k-word vocabulary starts by guessing *every word equally likely* — the worst possible strategy, costing $\ln V \approx 9.2$ nats. And the moment it *does* start learning, one optimizer step on its biggest weight can over-correct so violently that loss rockets **8.7 → 47** in a single update.

- **🎯 Log-Unigram Head-Bias Init:** seed the output bias with $b_{\text{head}}[c] = \log P(c)$ so step-0 output *is* the corpus word-frequency distribution. Step-0 loss drops from the uniform floor $\ln V = 9.21$ to the **unigram entropy** $H(p) \approx 7.42$ — a free −1.8 nats (perplexity $10{,}000 \to 1{,}770$) before a single gradient — and the model then only learns the *contextual deviation* from the marginal.
- **📉 Loss-Adaptive Trust Region ("move inversely to loss"):** the per-element step cap $|\Delta w|_{\max}$ moves *opposite* to the loss — tight ($0.12$) when loss is high and unstable, loosening to $0.60$ as loss converges. Timid when lost, confident when oriented. This alone eliminated the $8.7\to47$ blow-up (worst excursion fell to ~16).
- **📐 Dimension-Aware Damping ("higher dimension → less affected"):** scale the step by $\sqrt{d_{\text{ref}}/\max(d_{\text{ref}}, \dim)}$ so wider tensors move less per element — the same $1/\sqrt{d}$ logic as Xavier/He init and attention's $1/\sqrt{d_k}$, applied to the *update*. The tied vocab weight ($10000\times128$, used as **both** embedding and output head) is throttled ~9× while 128-wide layers run at full speed — surgically taming the one tensor that caused the instability.

> These attack three different axes — *where you start* (altitude), *when* steps are dangerous (high-loss regime), and *which weight* is dangerous (the biggest one) — so they stack cleanly and cost only a handful of scalar ops per step.

---

## 📚 Complete Obsidian Knowledge Vault

A full 28-document **Obsidian Knowledge Vault** is included in [`NeuralNet Obsidian/`](./NeuralNet%20Obsidian/):
- **Overview & Architecture**: System map, Ring hierarchy, and 5-phase roadmap.
- **Ring 0**: Tensor memory layouts, cache-blocked matrix multiplication, SIMD activations, CUDA engines, and the **Taylor Loss-Trajectory Predictor**.
- **Ring 1**: Attention mechanics, recursive thought chains, meta-loss optimization, 4-formula weight physics, and **training-stability / fast-start descent** (bias init, trust region, dimension damping).
- **Ring 2**: TransformerLM decoder, BPE subword tokenizer, and semantic VocabManager.
- **Ring 3**: Token relevancy parsing, 3D progressive curriculum, and training workflows.
- **Theoretical Foundations**: Information geometry, Riemannian manifolds, Fisher metrics, and Adaptive Focal Loss theory.

---

## 🛠️ Building & Running

### Requirements
- C++17 compliant compiler (MSVC 19.50+, GCC 9+, Clang 10+)
- CMake 3.20+
- OpenMP 2.0+ (Optional for parallel multicore acceleration)

### Build Instructions
```bash
# Configure build with CMake (use the Visual Studio generator installed on your machine)
cmake -B build -S . -G "Visual Studio 17 2022" -A x64

# Compile Release binary
cmake --build build --config Release

# Run with interactive training and debug diagnostics
./build/Release/nn_demo.exe --steps 100 --lr 0.3 --debug
```

---

## 📜 License
MIT License. Created by Heaplyn.
