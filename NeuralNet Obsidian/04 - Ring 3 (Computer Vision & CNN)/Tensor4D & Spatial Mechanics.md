# Plain English Summary

In traditional neural networks, data is represented as simple flat lists of numbers (2D matrices: rows for samples, columns for features). However, images have two-dimensional spatial structures (height and width) and multiple color channels (like Red, Green, Blue, or grayscale). `Tensor4D` is a 4-dimensional data container shaped as $(B, C, H, W)$—representing Batch size, Channels, Height, and Width. It provides zero-padding, He/Xavier initialization, matrix conversion bridging, and fast vectorized index mapping.

---

# Tensor4D & Spatial Mechanics

## 1. Mathematical Formulation

A 4D Tensor $\mathcal{T} \in \mathbb{R}^{B \times C \times H \times W}$ represents a mini-batch of spatial feature maps:
- $B$: Batch size (number of images/samples in the parallel forward pass).
- $C$: Number of feature channels (e.g., $1$ for grayscale bitmaps, $3$ for RGB, or $K$ intermediate convolutional filter maps).
- $H$: Spatial height in pixels / feature cells.
- $W$: Spatial width in pixels / feature cells.

### Memory Layout & Flattening Index Formula

Elements are stored contiguously in row-major order within a single heap-allocated buffer:

$$\text{Index}(b, c, h, w) = ((b \cdot C + c) \cdot H + h) \cdot W + w$$

$$\text{Total Elements} = B \cdot C \cdot H \cdot W$$

---

## 2. Spatial Zero-Padding

Convolutional kernels reduce feature dimensions unless padded. `Tensor4D::pad(pad_h, pad_w)` creates an expanded tensor $\mathcal{T}_{\text{pad}} \in \mathbb{R}^{B \times C \times (H + 2P_h) \times (W + 2P_w)}$:

$$\mathcal{T}_{\text{pad}}(b, c, h', w') = \begin{cases} \mathcal{T}(b, c, h' - P_h, w' - P_w) & \text{if } P_h \le h' < H + P_h \text{ and } P_w \le w' < W + P_w \\ 0 & \text{otherwise} \end{cases}$$

---

## 3. Matrix Bridging & Inter-Ring Compatibility

To feed convolutional activations into Ring 1 Dense layers, `Tensor4D` converts seamlessly to and from `ring0::Matrix`:
- **`to_matrix()`**: Reshapes $(B, C, H, W) \to (B, C \cdot H \cdot W)$ preserving batch rows.
- **`from_matrix(m, C, H, W)`**: Unflattens $(B, C \cdot H \cdot W) \to (B, C, H, W)$ during forward propagation and gradient backpropagation.

---

## 4. Initialization Schemes

1. **He (Kaiming) Normal Initialization**:
   $$\sigma = \sqrt{\frac{2}{\text{fan\_in}}} = \sqrt{\frac{2}{C_{\text{in}} \cdot K_h \cdot K_w}}, \quad W \sim \mathcal{N}(0, \sigma^2)$$
   Essential for ReLU and LeakyReLU activations to prevent signal vanishing.

2. **Xavier (Glorot) Uniform Initialization**:
   $$\text{limit} = \sqrt{\frac{6}{\text{fan\_in} + \text{fan\_out}}} = \sqrt{\frac{6}{(C_{\text{in}} + C_{\text{out}}) \cdot K_h \cdot K_w}}, \quad W \sim \mathcal{U}(-\text{limit}, \text{limit})$$
