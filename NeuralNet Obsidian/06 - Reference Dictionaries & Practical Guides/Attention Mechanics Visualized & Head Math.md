# 👁️ Attention Mechanics Visualized & Head Math Walkthrough

This guide provides a step-by-step mathematical and tensor dimension walkthrough of the **Grouped-Query Attention (GQA)**, **Rotary Position Embedding (RoPE)**, and **Attention with Linear Biases (ALiBi)** engine implemented in `include/ring1/attention.hpp` and `src/ring1/attention.cpp`.

---

## 💡 In Plain English

Attention is the mechanism that lets each token look at every earlier token and decide *how much each one matters* for predicting what comes next. For every position, we compute three vectors — a **query** ("what am I looking for?"), a **key** ("what do I offer?"), and a **value** ("what's my payload?") — then score every (query, key) pair, softmax the scores, and use those scores as weights to average up the values.

Three refinements the modern version adds:
- **GQA** — instead of every head having its own K/V, groups of query heads share one K/V. Cuts memory ~4× at almost no quality cost.
- **RoPE** — rotates the query and key vectors by an angle proportional to their position. Encodes *relative* position naturally into the dot product.
- **ALiBi** — adds a linear "the farther apart, the smaller the score" bias so the model behaves sensibly on longer contexts than it was trained on.

**Real-world analogy:** every token holds up a sign asking a question ("who's talking about food?"). Every earlier token holds up a matching sign ("I mentioned food"). The dot-product is the match strength; softmax picks who to listen to most; the value vectors are what those tokens actually say.

---

---

## 🧭 Pipeline Overview

```mermaid
graph TD
    A["Input Sequence x (Batch B, SeqLen T, Dim d_model)"] --> B["RMSNorm(x)"]
    B --> C["Linear Projections: W_q, W_k, W_v"]
    C --> D["Split into Query Heads (H_Q) and KV Heads (H_KV)"]
    D --> E["Apply RoPE Rotations to Q and K (q_rot, k_rot)"]
    E --> F["Repeat KV Heads to match Query Heads (GQA Expansion)"]
    F --> G["Compute Scaled Dot Product: (Q K^T) / sqrt(d_k)"]
    G --> H["Apply Causal Mask + ALiBi Linear Slopes (m_h * |i - j|)"]
    H --> I["Logit Soft-Cap: 20.0 * tanh(logits / 20.0)"]
    I --> J["Softmax(logits) -> Attention Weights A"]
    J --> K["Weighted Sum: A * V"]
    K --> L["Output Linear Projection W_o + Residual Add (x + Attn)"]
```

---

## 1. Concrete Dimension Example

Let us trace a batch through the exact default training dimensions:
- **Batch Size ($B$)**: $32$
- **Sequence Length ($T$)**: $64$
- **Model Dimension ($d_{\text{model}}$)**: $256$
- **Query Heads ($H_Q$)**: $8$
- **Key/Value Heads ($H_{KV}$)**: $2$ (Grouped-Query Attention with group size $G = \frac{8}{2} = 4$)
- **Head Dimension ($d_k = d_v$)**: $\frac{256}{8} = 32$
- **Feed-Forward Hidden Dimension ($d_{\text{ffn}}$)**: $512$

---

## 2. Step-by-Step Mathematical Walkthrough

### Step 1: Pre-Attention RMSNorm
Before projection, inputs are normalized to prevent activation scale drift:

$$\text{RMS}(x) = \sqrt{\frac{1}{d} \sum_{i=1}^d x_i^2 + \epsilon}$$

$$x_{\text{norm}} = \frac{x}{\text{RMS}(x)} \odot \gamma \quad \in \mathbb{R}^{B \times T \times 256}$$

where $\gamma \in \mathbb{R}^{256}$ is a learnable scaling vector initialized to $1.0$.

---

### Step 2: Linear Projections & Head Splitting
The normalized hidden state is projected into Query, Key, and Value subspaces:

$$\mathbf{Q} = x_{\text{norm}} W_q \quad \in \mathbb{R}^{B \times T \times (8 \times 32)} = \mathbb{R}^{32 \times 64 \times 256}$$

$$\mathbf{K} = x_{\text{norm}} W_k \quad \in \mathbb{R}^{B \times T \times (2 \times 32)} = \mathbb{R}^{32 \times 64 \times 64}$$

$$\mathbf{V} = x_{\text{norm}} W_v \quad \in \mathbb{R}^{B \times T \times (2 \times 32)} = \mathbb{R}^{32 \times 64 \times 64}$$

> [!TIP]
> **Why GQA Saves Memory**: Standard Multi-Head Attention (MHA) would allocate $8 \times 32 = 256$ dimensions for $K$ and $V$. By using $H_{KV} = 2$, the memory footprint of the autoregressive KV cache is **reduced by $75\%$ ($4\times$ smaller)**.

---

### Step 3: Rotary Position Embedding (RoPE)
For each query and key vector at token position $m \in [0, T-1]$, coordinate pairs are rotated in 2D planes:

$$\theta_i = 10000^{-2(i-1)/d_k} \quad \text{for } i \in \{1, 2, \dots, 16\}$$

$$\begin{pmatrix} q_{2i-1}' \\ q_{2i}' \end{pmatrix} = \begin{pmatrix} \cos(m \theta_i) & -\sin(m \theta_i) \\ \sin(m \theta_i) & \cos(m \theta_i) \end{pmatrix} \begin{pmatrix} q_{2i-1} \\ q_{2i} \end{pmatrix}$$

This endows the dot product $q_m^T k_n$ with relative distance awareness:

$$\langle \mathbf{R}_{\Theta, m} q_m, \mathbf{R}_{\Theta, n} k_n \rangle = g(q_m, k_n, m - n)$$

---

### Step 4: Grouped-Query Attention Expansion
Because there are $8$ Query heads but only $2$ KV heads, each KV head is shared across $4$ Query heads:
- Query Heads $0, 1, 2, 3 \longrightarrow$ attend to Key/Value Head $0$.
- Query Heads $4, 5, 6, 7 \longrightarrow$ attend to Key/Value Head $1$.

---

### Step 5: Scaled Dot-Product & ALiBi Falloff
For each query head $h \in [0, 7]$, the raw attention logits between token $i$ and token $j$ are:

$$\text{Logit}_{h}(i, j) = \frac{\mathbf{q}_{h, i} \cdot \mathbf{k}_{\text{group}(h), j}}{\sqrt{d_k}} - m_h \cdot |i - j| + \text{Mask}(i, j)$$

where:
- $\sqrt{d_k} = \sqrt{32} \approx 5.6568$.
- $m_h = 2^{-8 \cdot (h+1) / H}$ is the ALiBi geometric slope.
- $\text{Mask}(i, j) = 0$ if $j \le i$, and $-\infty$ if $j > i$ (strictly causal / autoregressive).

---

### Step 6: Logit Soft-Capping
To prevent extreme values from causing softmax saturation:

$$\text{Logit}_{\text{capped}} = 20.0 \cdot \tanh\left(\frac{\text{Logit}}{20.0}\right)$$

---

### Step 7: Softmax & Context Aggregation
$$\mathbf{A}_{h}(i, :) = \text{softmax}\left(\text{Logit}_{\text{capped}, h}(i, :)\right) \quad \in \mathbb{R}^{64}$$

$$\mathbf{O}_{h, i} = \sum_{j=1}^i \mathbf{A}_{h}(i, j) \cdot \mathbf{v}_{\text{group}(h), j} \quad \in \mathbb{R}^{32}$$

All 8 head outputs are concatenated into a single vector $\mathbf{O}_i \in \mathbb{R}^{256}$ and passed through the output projection:

$$\text{AttnOut} = \mathbf{O} \cdot W_o \quad \in \mathbb{R}^{32 \times 64 \times 256}$$

---

### Step 8: Residual Connection & SwiGLU FFN Block
$$\mathbf{x}_{\text{mid}} = \mathbf{x}_{\text{in}} + \text{AttnOut}$$

The intermediate state passes through a SwiGLU Feed-Forward Network:

$$\text{SwiGLU}(z) = \left( (z W_{\text{gate}}) \odot \text{SiLU}(z W_{\text{up}}) \right) W_{\text{down}}$$

$$\mathbf{x}_{\text{out}} = \mathbf{x}_{\text{mid}} + \text{SwiGLU}(\text{RMSNorm}(\mathbf{x}_{\text{mid}}))$$

---

## 3. Attention Summary Table

| Operation | Input Shape | Weights Shape | Output Shape | Primary Purpose |
| :--- | :--- | :--- | :--- | :--- |
| **RMSNorm** | $(32, 64, 256)$ | $\gamma \in \mathbb{R}^{256}$ | $(32, 64, 256)$ | Prevents activation scale explosion. |
| **$W_q$ Projection** | $(32, 64, 256)$ | $(256, 256)$ | $(32, 64, 8, 32)$ | Maps to 8 Query subspaces. |
| **$W_k, W_v$ Projection** | $(32, 64, 256)$ | $(256, 64)$ | $(32, 64, 2, 32)$ | Maps to 2 shared KV subspaces (GQA). |
| **RoPE Rotation** | $(32, 64, 8, 32)$ | Fixed Frequencies | $(32, 64, 8, 32)$ | Injects relative positional geometry. |
| **Attention Matrix** | $Q \in (8, 32), K \in (2, 32)$ | None | $(32, 8, 64, 64)$ | Computes all-pairs causal relevance. |
| **ALiBi Falloff** | $(32, 8, 64, 64)$ | Geometric slopes | $(32, 8, 64, 64)$ | Enables context length extrapolation. |
| **Output $W_o$** | $(32, 64, 256)$ | $(256, 256)$ | $(32, 64, 256)$ | Merges multi-head features into stream. |
| **SwiGLU FFN** | $(32, 64, 256)$ | $(256, 512) \times 3$ | $(32, 64, 256)$ | Non-linear knowledge storage & retrieval. |

---

## 4. Common Beginner Confusions

**"Why divide by √d in the score?"** — Without it, dot products of higher-dimensional vectors grow linearly with d, which pushes softmax toward one-hot (only the biggest score survives). The `1/√d` factor keeps the pre-softmax distribution reasonable so gradients flow to more than one key.

**"How is GQA different from MQA?"** — MQA (Multi-Query Attention) is the extreme case: one K/V shared across every Q head. GQA is the interpolation: groups of Q heads share a K/V. In this project, 8 Q heads share 2 KV heads (4:1), matching LLaMA-2/3.

**"Why RoPE *and* ALiBi?"** — RoPE encodes precise relative position via rotation; ALiBi adds a soft distance decay. They compose: RoPE gives the model the geometry, ALiBi gives it a length-generalization prior. Modern implementations sometimes pick one or the other; this engine keeps both for the extrapolation robustness.

**"Where does the causal mask happen?"** — The engine skips computing scores for j > i entirely in the fused kernel, rather than filling those positions with `-inf` and letting softmax consume them. Cheaper *and* numerically cleaner.

---

## 🔗 Related Notes
- [[02 - Ring 1 (Layers & Advanced Optimizers)/Attention Mechanics & ALiBi|Attention Mechanics & ALiBi]] — primary implementation note in the main vault
- [[01 - Ring 0 (Core Math & Hardware)/CUDA & Hardware Acceleration Engine|CUDA & Hardware Acceleration Engine]] — the fused attention kernel this walk-through corresponds to
- [[03 - Ring 2 (Models & Transformers)/TransformerLM Decoder (GQA + SwiGLU + RoPE)|TransformerLM Decoder]] — the block that wraps attention + SwiGLU
- [[03 - Ring 2 (Models & Transformers)/Autoregressive KV-Cache Generation|KV-Cache Generation]] — inference-time reuse of the K/V computed here
- [[Index|Return to Master Index]]
