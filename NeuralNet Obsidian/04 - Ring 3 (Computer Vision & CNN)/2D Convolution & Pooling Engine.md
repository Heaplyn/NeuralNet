# Plain English Summary

Convolutional neural networks work by sliding tiny learned filter windows (like 3x3 pixel magnifying glasses) across an image. Each filter looks for local visual features such as edges, curves, or textures, regardless of where they appear in the image. Pooling layers (like MaxPool) then shrink the feature maps by picking the strongest response in each 2x2 patch. This reduces data size and makes the network resistant to small shifts or distortions. This document details the exact mathematics of 2D Convolution, analytical backward passes, and MaxPool argmax routing.

---

# 2D Convolution & Pooling Engine

## 1. Conv2D Forward Mechanics

Given an input feature map $X \in \mathbb{R}^{B \times C_{\text{in}} \times H_{\text{in}} \times W_{\text{in}}}$ and learnable filter bank $W \in \mathbb{R}^{C_{\text{out}} \times C_{\text{in}} \times K_h \times K_w}$ with bias $b \in \mathbb{R}^{C_{\text{out}}}$:

### Output Dimensions
$$H_{\text{out}} = \left\lfloor \frac{H_{\text{in}} + 2P_h - K_h}{S_h} \right\rfloor + 1, \quad W_{\text{out}} = \left\lfloor \frac{W_{\text{in}} + 2P_w - K_w}{S_w} \right\rfloor + 1$$

### Pre-activation Convolution Equation
$$Z(b, c_{\text{out}}, h, w) = b(c_{\text{out}}) + \sum_{c_{\text{in}}=0}^{C_{\text{in}}-1} \sum_{k_h=0}^{K_h-1} \sum_{k_w=0}^{K_w-1} X_{\text{pad}}(b, c_{\text{in}}, h \cdot S_h + k_h, w \cdot S_w + k_w) \cdot W(c_{\text{out}}, c_{\text{in}}, k_h, k_w)$$

$$A(b, c_{\text{out}}, h, w) = \sigma(Z(b, c_{\text{out}}, h, w))$$

---

## 2. Conv2D Analytical Backward Pass

Let $\delta_{\text{out}} = \frac{\partial \mathcal{L}}{\partial A}$ be the incoming gradient from the next layer.

### 1. Pre-activation Gradient
$$\delta_{\text{pre}} = \delta_{\text{out}} \odot \sigma'(Z)$$

### 2. Bias Gradient
$$\frac{\partial \mathcal{L}}{\partial b(c_{\text{out}})} = \sum_{b=0}^{B-1} \sum_{h=0}^{H_{\text{out}}-1} \sum_{w=0}^{W_{\text{out}}-1} \delta_{\text{pre}}(b, c_{\text{out}}, h, w)$$

### 3. Filter Weight Gradients
$$\frac{\partial \mathcal{L}}{\partial W(c_{\text{out}}, c_{\text{in}}, k_h, k_w)} = \sum_{b=0}^{B-1} \sum_{h=0}^{H_{\text{out}}-1} \sum_{w=0}^{W_{\text{out}}-1} \delta_{\text{pre}}(b, c_{\text{out}}, h, w) \cdot X_{\text{pad}}(b, c_{\text{in}}, h \cdot S_h + k_h, w \cdot S_w + k_w)$$

### 4. Input Gradient (for Lower Layer Backpropagation)
$$\frac{\partial \mathcal{L}}{\partial X_{\text{pad}}(b, c_{\text{in}}, i_h, i_w)} = \sum_{c_{\text{out}}} \sum_{h, w \text{ s.t. } h \cdot S_h + k_h = i_h} \delta_{\text{pre}}(b, c_{\text{out}}, h, w) \cdot W(c_{\text{out}}, c_{\text{in}}, k_h, k_w)$$

The input gradient is then unpadded to recover $\frac{\partial \mathcal{L}}{\partial X} \in \mathbb{R}^{B \times C_{\text{in}} \times H_{\text{in}} \times W_{\text{in}}}$.

---

## 3. MaxPool2D & Argmax Gradient Routing

`MaxPool2D` downsamples spatial activations using non-overlapping or strided windows.

### Forward Argmax Caching
$$Y(b, c, h, w) = \max_{(k_h, k_w) \in [0, K_h) \times [0, K_w)} X(b, c, h \cdot S_h + k_h, w \cdot S_w + k_w)$$

$$\text{argmax}(b, c, h, w) = \arg\max_{(k_h, k_w)} X(b, c, h \cdot S_h + k_h, w \cdot S_w + k_w)$$

### Backward Routing
Gradient flows exclusively to the single spatial cell that generated the forward maximum value:

$$\frac{\partial \mathcal{L}}{\partial X(b, c, i_h, i_w)} = \sum_{h, w} \delta_{\text{out}}(b, c, h, w) \cdot \mathbb{I}\left[ \text{argmax}(b, c, h, w) = (i_h, i_w) \right]$$
