# 🏎️ AdamW, Fisher Metric & Nesterov Acceleration

The `ring1::AdamW` optimizer combines modern decoupled weight decay with second-order Riemannian Fisher information preconditioning, Hamiltonian phase space lookahead, and layer-wise credit attribution.

---

## 📋 Prerequisites

Before reading this, you should be comfortable with:
- Vanilla SGD and momentum-SGD (the "accumulate a velocity" idea)
- Basic Adam — running mean and running variance of the gradient
- Weight decay vs. L2 regularization — knowing why AdamW *decouples* them
- [[01 - Ring 0 (Core Math & Hardware)/Tensor3D & Matrix Math|Tensor3D & Matrix Math]] — the per-element storage layout
- [[05 - Theoretical Foundations & Physics/Riemannian Manifolds & Fisher Information|Riemannian Manifolds & Fisher Information]] — what the Fisher term actually means geometrically
- Optional: [[02 - Ring 1 (Layers & Advanced Optimizers)/4-Formula Dynamic Weight Physics|4-Formula Dynamic Weight Physics]] — the router that this optimizer dispatches to per-parameter

---

## ⚡ Mathematical Formulations

### 1. Moment Accumulation & Bias Correction
$$m_t = \beta_1 m_{t-1} + (1 - \beta_1) g_t, \quad \hat{m}_t = \frac{m_t}{1 - \beta_1^t}$$
$$v_t = \beta_2 v_{t-1} + (1 - \beta_2) g_t^2, \quad \hat{v}_t = \frac{v_t}{1 - \beta_2^t}$$

### 2. Empirical Diagonal Fisher Information Metric
$$F_{ii}^{(t)} = 0.95 \cdot F_{ii}^{(t-1)} + 0.05 \cdot (g_i^2), \quad \hat{F}_{ii}^{(t)} = \frac{F_{ii}^{(t)}}{1 - \beta_2^t}$$

### 3. Nesterov Lookahead Momentum in Phase Space
$$\tilde{m}_t = \beta_1 \hat{m}_t + (1 - \beta_1) \frac{g_t}{1 - \beta_1^t}$$

### 4. Decoupled Weight Decay Coupled to Penalty Sensitivity
$$w_{t+1} = w_t - \alpha \cdot (\lambda \cdot \mathcal{P}_t) \cdot w_t - \frac{\alpha}{\sqrt{\hat{v}_t} + \epsilon} \cdot \tilde{m}_t$$

---

## 🎯 Layer-Wise Credit Attribution

The optimizer tracks parameter shift vectors $\Delta \mathbf{W}_l$ from step $t-1$. When step $t$ observes a loss change $\Delta \mathcal{L} = \mathcal{L}_t - \mathcal{L}_{t-1}$:

$$\text{Attribution}_l = -\Delta \mathcal{L} \cdot \frac{\|\Delta \mathbf{W}_l\|_2}{\sum_k \|\Delta \mathbf{W}_k\|_2}$$
$$\text{Scale}_l \leftarrow \text{clamp}\left(\text{Scale}_l \cdot (1.0 + 0.1 \cdot \text{Attribution}_l), 0.5, 2.0\right)$$

Layers that directly contributed to loss reduction receive higher learning rates on subsequent iterations.

---

---

## 🛡️ Step Safety: Trust Region & Dimension Damping

The raw step magnitude needs two guardrails. Dense image and letter classifiers can still produce sharp gradient changes, so both guardrails live inside `update_param`, right before the weight is written.

### 1. Loss-adaptive trust region (`config.max_step`)
The per-element step is clamped to `±max_step`, and `max_step` is **inverse to the loss** — tight when loss is high (unstable), loose when low (converging):

$$|\Delta w|_{\max}(\mathcal{L}) \in [0.12,\ 0.60], \quad \text{smaller when } \mathcal{L} \text{ larger}$$

**Intuition:** big error means you're probably somewhere strange, so step timidly; small error means you're near a known-good spot, so you can stride. The trainer sets it each step via `trust_region_for_loss(ema_loss)`. Unlike a global LR cut, this is *state-dependent* — strict only while it needs to be.

### 2. Dimension-aware damping (fan-in scaling of the step)
The effective LR is scaled by $\sqrt{d_{\text{ref}}/\max(d_{\text{ref}}, \dim)}$ with $d_{\text{ref}}=128$, so **wider tensors move less per element**:

```cpp
float dim_damp = sqrt(128.0f / max(128.0f, (float)max(param.rows, param.cols)));
effective_lr  *= dim_damp;
```

**Intuition:** a wide image weight matrix accumulates more per-element update pressure than a small letter matrix. Dimension-aware damping reduces that pressure while preserving the learning signal. This is the same $1/\sqrt{d}$ logic as Xavier/He initialization, applied to the update instead of the initialization.


---

## 🔗 Related Notes
- [[04 - Ring 3 (Data & Training Pipelines)/Recognition Benchmarks - Letters MNIST Fashion-MNIST|Recognition Benchmarks]]
- [[02 - Ring 1 (Layers & Advanced Optimizers)/4-Formula Dynamic Weight Physics|4-Formula Dynamic Weight Physics]]
- [[05 - Theoretical Foundations & Physics/Riemannian Manifolds & Fisher Information|Riemannian Manifolds]]
- [[Index|Return to Index]]
