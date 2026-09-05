# 🔬 4-Formula Dynamic Weight Physics

Rather than treating all network parameters uniformly, the **Multi-Formula Dynamic Weight Physics Engine** (`ring1::MultiFormulaKernel`) classifies every individual weight element into functional importance tiers in real time and applies dedicated mathematical physics equations.

---

## 📋 Prerequisites

Before reading this, you should be comfortable with:
- [[02 - Ring 1 (Layers & Advanced Optimizers)/AdamW, Fisher Metric & Nesterov|AdamW]] — this engine dispatches from inside the AdamW step
- **First-order Taylor salience** — `|g · w|` as a proxy for "how much removing this weight would change the loss" (standard in pruning literature)
- **Diagonal Fisher information** — running average of squared gradients; used here as a cheap curvature proxy
- **Nesterov accelerated momentum** and **natural gradient** (see [[05 - Theoretical Foundations & Physics/Riemannian Manifolds & Fisher Information|Riemannian Manifolds & Fisher Information]])
- Optional: [[04 - Ring 3 (Data & Training Pipelines)/Debug Log Format & Reading Guide|Debug Log Format]] — the `FORM` line shows this router's live F1/F2/F3/F4 split

> **Honest framing:** per-parameter routing between different update rules is a reasonable adaptive idea (related to layer-wise adaptive optimizers, sparse training, and hybrid second-order methods). The "physics" branding is stylistic — mathematically this is a Taylor/Fisher importance score gating four update variants.

---

## 🎯 Practical Explanation: What is this and Why Does it Exist?

### The Uniform Weight Assumption Flaw
In standard deep learning optimizers (SGD, AdamW, RMSProp), every weight in the network is updated using the exact same equation:
$$\theta_{t+1} = \theta_t - \alpha \cdot \frac{m_t}{\sqrt{v_t} + \epsilon} - \alpha \lambda \theta_t$$

However, in real neural networks:
1. **Critical Semantic Hubs (Top ~5%)**: Represent pivotal routing decisions, attention projection bases, and core syntax keys. If updated recklessly with noisy stochastic gradients, catastrophic forgetting occurs.
2. **Feature Extractors (Middle ~40%)**: Benefit from second-order momentum and curvature dampening.
3. **Noisy Background Weights (Bottom ~30%)**: Are near zero, noisy, or redundant. Applying standard momentum keeps them alive unnecessarily, preventing sparse representation and wasting capacity.

### How 4-Formula Physics Solves This
By measuring **First-Order Taylor Salience** and **Empirical Fisher Curvature**, the optimizer computes an instantaneous importance factor $\mathcal{I}(w_i) \in [0, 1]$ for every weight element and routes it to one of **4 specialized physics equations**:

```mermaid
graph TD
    Weight["Parameter Element w_i"] --> Calc["Compute Importance I(w_i) in [0, 1]"]
    Calc -->|I > 0.70 (Critical Hubs)| F1["Formula 1: Riemannian Geodesic Natural Gradient"]
    Calc -->|0.40 < I <= 0.70 (High Impact)| F2["Formula 2: Curvature-Scaled Nesterov Acceleration"]
    Calc -->|0.15 < I <= 0.40 (Standard Mid)| F3["Formula 3: Variance-Bounded Decoupled AdamW"]
    Calc -->|I <= 0.15 (Noise / Redundant)| F4["Formula 4: Inertial Sparse Decay & Pruning"]
```

---

## 💻 Deep Code Breakdown

### 1. The Importance Metric Calculation $\mathcal{I}(w_i)$
Located in `src/ring1/multi_formula_optimizer.cpp`:

```cpp
float MultiFormulaKernel::compute_importance(
    float w, 
    float g, 
    float fisher, 
    float norm_g, 
    float norm_w,
    size_t num_elements
) {
    float n_f = static_cast<float>(std::max<size_t>(1, num_elements));
    
    // 1. Normalized Taylor salience relative to mean element magnitude in tensor:
    // mean_energy = (|g| * |w|) / N
    float mean_energy = (norm_g * norm_w) / n_f + 1e-8f;
    float relative_salience = std::abs(g * w) / mean_energy;

    // 2. Normalized Fisher curvature relative to mean gradient variance:
    // mean_fisher = |g|^2 / N
    float mean_fisher = (norm_g * norm_g) / n_f + 1e-8f;
    float relative_fisher = std::sqrt(std::max(0.0f, fisher) / mean_fisher);

    // 3. Dynamic non-linear blend
    float raw_score = 0.6f * relative_salience + 0.4f * relative_fisher;

    // 4. Smooth sigmoidal importance mapping into [0.0, 1.0]
    // Average element (raw_score ~ 1.0) -> importance ~ 0.50
    return raw_score / (1.0f + raw_score);
}
```

---

### 2. The 4 Mathematical Update Formulas
Located in `src/ring1/multi_formula_optimizer.cpp`:

```cpp
float MultiFormulaKernel::execute_update_formula(
    WeightFormulaType formula,
    float w,
    float g,
    float m_hat,
    float v_hat,
    float f_hat,
    float effective_lr,
    float beta1,
    float beta1_corr,
    float eps,
    float effective_wd,
    float curvature_scale
) {
    float delta_w = 0.0f;

    switch (formula) {
        // =========================================================================
        // FORMULA 1: Riemannian Geodesic Natural Gradient (Ultra-High Importance, I > 0.70)
        // Used for critical semantic routing hubs.
        // Preconditions step using the true Fisher information metric on the Riemannian manifold.
        // =========================================================================
        case WeightFormulaType::FORMULA_1_GEODESIC_NATURAL_GRAD: {
            float g_corr = g / beta1_corr;
            float nesterov_m = beta1 * m_hat + (1.0f - beta1) * g_corr;
            float fisher_val = std::sqrt(std::max(0.0f, f_hat)) + eps;
            float v_val = std::sqrt(std::max(0.0f, v_hat)) + eps;
            
            // Blend Riemannian Fisher metric with gradient variance
            float natural_precond = 0.5f * v_val + 0.5f * fisher_val;
            delta_w = (effective_lr / natural_precond) * nesterov_m;
            
            // Soft regularization: Protect high-importance parameters from aggressive decay
            if (effective_wd > 0.0f) {
                delta_w += effective_lr * (effective_wd * 0.5f) * w;
            }
            break;
        }

        // =========================================================================
        // FORMULA 2: Curvature-Scaled Nesterov Acceleration (High Importance, 0.40 < I <= 0.70)
        // Used for active feature representation channels.
        // Incorporates Hamiltonian phase-space lookahead and Rayleigh quotient curvature scale.
        // =========================================================================
        case WeightFormulaType::FORMULA_2_CURVATURE_NESTEROV: {
            float g_corr = g / beta1_corr;
            float nesterov_m = beta1 * m_hat + (1.0f - beta1) * g_corr;
            float precond = (std::sqrt(std::max(0.0f, v_hat)) + eps) * std::max(0.2f, curvature_scale);
            delta_w = (effective_lr / precond) * nesterov_m;
            if (effective_wd > 0.0f) {
                delta_w += effective_lr * effective_wd * w;
            }
            break;
        }

        // =========================================================================
        // FORMULA 3: Variance-Bounded Decoupled AdamW (Medium Importance, 0.15 < I <= 0.40)
        // Standard first-order AdamW with numerical variance bounds.
        // =========================================================================
        case WeightFormulaType::FORMULA_3_VARIANCE_BOUNDED_ADAMW: {
            float bounded_v = std::max(1e-8f, std::min(100.0f, v_hat));
            float precond = std::sqrt(bounded_v) + eps;
            delta_w = (effective_lr / precond) * m_hat;
            if (effective_wd > 0.0f) {
                delta_w += effective_lr * effective_wd * w;
            }
            break;
        }

        // =========================================================================
        // FORMULA 4: Inertial Sparse Decay & Representation Compression (Low Importance, I <= 0.15)
        // Applied to noisy and redundant weights.
        // Halves gradient step and doubles weight decay to actively prune noise and accelerate sparsity.
        // =========================================================================
        case WeightFormulaType::FORMULA_4_INERTIAL_SPARSE_DECAY: {
            float precond = std::sqrt(std::max(0.0f, v_hat)) + eps;
            float standard_step = (effective_lr / precond) * m_hat;
            float decay_step = effective_lr * (effective_wd * 2.0f) * w;
            delta_w = 0.5f * standard_step + decay_step;
            break;
        }
    }

    return delta_w;
}
```

---

### 3. OpenMP Threaded Dispatch Loop in `AdamW::update_param`
Located in `src/ring1/adamw.cpp`:

```cpp
size_t local_f1 = 0, local_f2 = 0, local_f3 = 0, local_f4 = 0;

#pragma omp parallel for schedule(static) if (param.data.size() > 1024) \
    reduction(+:local_f1, local_f2, local_f3, local_f4)
for (int i_idx = 0; i_idx < static_cast<int>(param.data.size()); ++i_idx) {
    size_t i = static_cast<size_t>(i_idx);
    float g = grad.data[i];
    float w = param.data[i];

    // Moments update
    m.data[i] = beta1 * m.data[i] + (1.0f - beta1) * g;
    v.data[i] = beta2 * v.data[i] + (1.0f - beta2) * g * g;
    f.data[i] = 0.95f * f.data[i] + 0.05f * (g * g); // Fisher

    // Bias correction
    float m_hat = m.data[i] / beta1_corr;
    float v_hat = v.data[i] / beta2_corr;
    float f_hat = f.data[i] / beta2_corr;

    // Importance & Formula dispatch
    float importance = MultiFormulaKernel::compute_importance(w, g, f_hat, norm_g, norm_w);
    WeightFormulaType formula;
    if (importance > 0.70f) {
        formula = WeightFormulaType::FORMULA_1_GEODESIC_NATURAL_GRAD;
        local_f1++;
    } else if (importance > 0.40f) {
        formula = WeightFormulaType::FORMULA_2_CURVATURE_NESTEROV;
        local_f2++;
    } else if (importance > 0.15f) {
        formula = WeightFormulaType::FORMULA_3_VARIANCE_BOUNDED_ADAMW;
        local_f3++;
    } else {
        formula = WeightFormulaType::FORMULA_4_INERTIAL_SPARSE_DECAY;
        local_f4++;
    }

    float delta_w = MultiFormulaKernel::execute_update_formula(
        formula, w, g, m_hat, v_hat, f_hat,
        effective_lr, beta1, beta1_corr, eps, effective_wd, curvature_scale
    );

    param.data[i] -= delta_w;
}
```

---

## 🔗 Related Notes
- [[02 - Ring 1 (Layers & Advanced Optimizers)/AdamW, Fisher Metric & Nesterov|AdamW, Fisher Metric & Nesterov]]
- [[02 - Ring 1 (Layers & Advanced Optimizers)/Meta-Neural Loss & Step Optimizer|Meta-Neural Loss Optimizer]]
- [[05 - Theoretical Foundations & Physics/Riemannian Manifolds & Fisher Information|Riemannian Manifolds & Information Geometry]]
- [[Index|Return to Master Index]]
