# Plain English Summary

A Convolutional Neural Network (CNN) combines two main parts:
1. **Feature Extraction Backbone**: A sequence of convolutional and pooling layers that scan raw images and turn raw pixels into abstract visual features (like lines, loops, corners, and shapes).
2. **Dense Classification Head**: Fully connected layers (from Ring 1) that take those abstract visual features and decide which class (like letter 'A' or digit '7') the image belongs to.

This document describes how the `CNN` container connects these pieces, tracks layer dimensions automatically, and supports dynamic filter expansion.

---

# Modular CNN Architecture

## 1. Architectural Pipeline

```mermaid
graph TD
    IN["Raw Image Input (B, C_in, H, W)"] --> C0["Conv2D (1 -> C1, K=3x3, P=1)"]
    C0 --> P0["MaxPool2D (2x2, S=2)"]
    P0 --> C1["Conv2D (C1 -> C2, K=3x3, P=1)"]
    C1 --> P1["MaxPool2D (2x2, S=2)"]
    P1 --> FLAT["Flatten Bridge (B, C2 * H/4 * W/4)"]
    FLAT --> D0["Dense Layer 0 (Linear + ReLU)"]
    D0 --> D1["Dense Layer 1 (Classification Logits)"]
    D1 --> SOFT["Softmax Cross-Entropy Loss"]
```

---

## 2. Dynamic Filter & Capacity Growth

When the network encounters a plateau or repeats mistake patterns, Ring 3 can expand filter capacity on-the-fly:

### Conv Filter Expansion
- Layer $L$: Filters expand from $C_{\text{out}} \to C_{\text{out}} + \Delta C$.
- Layer $L+1$: Input channels expand from $C_{\text{in}} \to C_{\text{in}} + \Delta C$.
- If Layer $L$ is the final conv layer before flattening, Dense Layer 0's incoming weight matrix expands its row dimension by $(\Delta C \cdot H_{\text{final}} \cdot W_{\text{final}})$.

New filter weights are initialized with scaled random noise ($0.05 \times \text{std}$) to preserve existing learned representations while unlocking new feature capacity.
