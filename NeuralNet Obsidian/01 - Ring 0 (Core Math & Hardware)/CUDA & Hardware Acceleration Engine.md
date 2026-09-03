# 🚀 CUDA & Hardware Acceleration Engine

The `ring0::CUDAMathEngine` provides a unified hardware acceleration layer designed for GPU tensor math with automatic CPU fallback.

---

## 🏗️ Architecture & Device Probing

The engine automatically probes for available hardware at initialization:
1. **GPU Mode**: Dispatches compute to NVIDIA CUDA cores.
2. **CPU Mode**: Dispatches parallel OpenMP kernels with AVX2/FMA vector instructions.

```cpp
// Runtime Device Query
ring0::CUDAMathEngine::initialize();
const auto& dev_info = ring0::CUDAMathEngine::get_device_info();
std::cout << "Math Engine: " << dev_info.device_name 
          << " (" << dev_info.multi_processor_count << " compute units active)\n";
```

---

## ⚡ Fused Computational Kernels

### 1. Fused ALiBi Causal Attention
Fuses QK dot-product, ALiBi slope bias, and Softmax into a single pass:

$$\mathbf{S}_{i,j} = \frac{\mathbf{q}_i^\top \mathbf{k}_j}{\sqrt{d}} - m \cdot (i - j)$$
$$\mathbf{A}_{i,:} = \text{Softmax}(\mathbf{S}_{i,:})$$
$$\mathbf{O}_{i,:} = \sum_{j \le i} \mathbf{A}_{i,j} \cdot \mathbf{v}_j$$

### 2. Fused SwiGLU Gated Activation
Fuses matrix projection and gating to eliminate memory round-trips:

$$\text{SwiGLU}(\mathbf{x}) = \left(\mathbf{x} \mathbf{W}_{\text{gate}} \cdot \sigma(\mathbf{x} \mathbf{W}_{\text{gate}})\right) \odot (\mathbf{x} \mathbf{W}_{\text{up}})$$

---

## 🔗 Related Notes
- [[01 - Ring 0 (Core Math & Hardware)/Tensor3D & Matrix Math|Tensor3D & Matrix Math]]
- [[02 - Ring 1 (Layers & Advanced Optimizers)/Attention Mechanics & ALiBi|Attention Mechanics & ALiBi]]
- [[Index|Return to Index]]
