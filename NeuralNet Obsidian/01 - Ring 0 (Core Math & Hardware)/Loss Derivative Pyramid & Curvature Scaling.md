# 🔺 Loss Derivative Pyramid & Rayleigh Curvature Scaling

> **Ring Level**: Ring 0 (`ring0::LossDerivativePyramid`)
> **Prerequisites**: [[01 - Ring 0 (Core Math & Hardware)/Loss Formulations & Calculus|Loss Formulations & Calculus]], [[05 - Theoretical Foundations & Physics/Information Geometry & Loss Dynamics|Information Geometry & Loss Dynamics]]
> **Source Files**: `include/ring0/loss.hpp`, `src/ring0/loss.cpp`, `src/ring3/llm_trainer.cpp`

---

## 🏛️ Motivation: Beyond First-Order Gradients

Standard backpropagation computes only first derivatives $\frac{\partial \mathcal{L}}{\partial \theta}$. However, optimal step length in non-convex landscapes depends heavily on the local curvature (the Hessian $H = \nabla^2 \mathcal{L}$).

Computing and inverting the full Hessian is $O(D^3)$ in time and $O(D^2)$ in space — completely intractable for modern neural networks.

The **Loss Derivative Pyramid** provides an $O(N)$ hierarchical pyramid across batch token losses that computes continuous multi-order finite differences and Rayleigh curvature approximations.

---

## 📐 Multi-Order Pyramid Construction

Given an array of token losses $\mathbf{L} = [L_1, L_2, \dots, L_N]$ sorted by magnitude:

1. **Level 0 (Base)**: Raw token losses $L_i$.
2. **Level 1 (First Order)**: Finite difference velocity $\Delta L_i = L_{i+1} - L_i$.
3. **Level 2 (Second Order / Acceleration)**: Difference curvature $\Delta^2 L_i = \Delta L_{i+1} - \Delta L_i$.
4. **Level 3 (Third Order / Jerk)**: Rate of change of curvature $\Delta^3 L_i$.

```
Level 3:      [ Δ³L₁ ]        [ Δ³L₂ ]
Level 2:    [ Δ²L₁ ]   [ Δ²L₂ ]   [ Δ²L₃ ]
Level 1:   [ ΔL₁ ]  [ ΔL₂ ]  [ ΔL₃ ]  [ ΔL₄ ]
Level 0:  [ L₁ ] [ L₂ ] [ L₃ ] [ L₄ ] [ L₅ ]
```

---

## 🧮 Rayleigh Quotient Curvature Modulation

From Level 2 curvature statistics, the pyramid extracts a dimensionless curvature quotient $\kappa$:

$$\kappa = \frac{\text{Var}(\Delta^2 L)}{\text{Mean}(|\Delta L|) + \epsilon}$$

In `LLMTrainer::train_step()`, when `config.use_curvature_scaling` is enabled:
$$\eta_{\text{eff}} = \eta \cdot \text{clamp}\left(\frac{1.0}{\sqrt{1.0 + \kappa}}, 0.50, 1.50\right)$$

- When traversing **flat plateaus** ($\kappa \approx 0$), $\eta_{\text{eff}}$ accelerates up to $1.5\times$ to escape saddles quickly.
- When entering **sharp ravines** ($\kappa \gg 1$), $\eta_{\text{eff}}$ automatically brakes down to $0.5\times$ to prevent overshooting.
