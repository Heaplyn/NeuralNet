# ⚡ Activation Functions & Non-Linear Transformations

In `ring0::Activations`, all forward activations and their exact analytical Jacobian derivatives are implemented with SIMD and OpenMP vectorization.

---

## 📊 Catalog of Mathematical Activations

### 1. Gaussian Error Linear Unit (GELU)
Approximates the expectation of standard normal distribution gating:

$$\text{GELU}(x) = x \cdot \Phi(x) \approx 0.5 \cdot x \cdot \left(1 + \tanh\left(\sqrt{\frac{2}{\pi}} \cdot (x + 0.044715 \cdot x^3)\right)\right)$$

**Derivative $\frac{d\text{GELU}}{dx}$**:
$$\frac{d}{dx}\text{GELU}(x) = 0.5 \cdot \left(1 + \tanh(u)\right) + 0.5 \cdot x \cdot (1 - \tanh^2(u)) \cdot \sqrt{\frac{2}{\pi}} \cdot (1 + 3 \cdot 0.044715 \cdot x^2)$$

### 2. Swish / SiLU (Sigmoid Linear Unit)
Used in SwiGLU gating channels:
$$\text{Swish}(x) = x \cdot \sigma(x) = \frac{x}{1 + e^{-x}}$$
$$\frac{d}{dx}\text{Swish}(x) = \sigma(x) + x \cdot \sigma(x) \cdot (1 - \sigma(x)) = \sigma(x) \cdot (1 + x \cdot (1 - \sigma(x)))$$

### 3. Numerically Stable Softmax
Prevents exponential underflow/overflow via max-subtraction:

$$m = \max_{j} x_j, \quad s_i = \frac{e^{x_i - m}}{\sum_{j} e^{x_j - m}}$$

**Vector-Jacobian Product**:
$$\nabla x_i = s_i \cdot \left( \nabla s_i - \sum_j \nabla s_j \cdot s_j \right)$$

### 4. Root Mean Square Normalization (RMSNorm)
Standard pre-normalization layer (30% faster than LayerNorm by omitting mean-centering):

$$\text{RMSNorm}(\mathbf{x}) = \frac{\mathbf{x}}{\sqrt{\frac{1}{d} \sum_{i=1}^d x_i^2 + \epsilon}} \odot \boldsymbol{\gamma}$$

---

## 🔗 Related Notes
- [[01 - Ring 0 (Core Math & Hardware)/Tensor3D & Matrix Math|Tensor3D & Matrix Math]]
- [[02 - Ring 1 (Layers & Advanced Optimizers)/Attention Mechanics & ALiBi|Attention Mechanics]]
- [[Index|Return to Index]]
