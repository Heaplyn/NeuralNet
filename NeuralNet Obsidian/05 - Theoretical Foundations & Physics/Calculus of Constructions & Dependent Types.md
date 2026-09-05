# 🪢 Calculus of Constructions & Dependent-Typed Neural Attention

> **Ring Level**: Ring 0 (`ring0::CoCEngine`) & Ring 1 (`ring1::DependentTypeAttention`)
> **Prerequisites**: [[01 - Ring 0 (Core Math & Hardware)/Calculus of Constructions & Dependent-Typed Neural Reasoning|CoC Engine Implementation]], [[02 - Ring 1 (Layers & Advanced Optimizers)/Attention Mechanics & ALiBi|Attention Mechanics & ALiBi]]
> **Source Files**: `include/ring0/calculus_of_constructions.hpp`, `src/ring0/calculus_of_constructions.cpp`, `include/ring1/dependent_type_attention.hpp`

---

## 🧠 Plain-English Summary

Standard transformer attention decides *which tokens to look at* based purely on geometric similarity — how close are the query and key embedding vectors in space? Two tokens get high attention if their embeddings point in similar directions, regardless of whether they *logically belong together*.

**Dependent-Typed Attention adds a second signal**: a *logical compatibility score* computed by the CoC engine. The idea is:

> "A verb should attend to its subject (a noun), not to another verb — even if the verb embeddings are geometrically closer together."

The network learns a separate **type embedding space** where tokens are projected into abstract type signatures (Noun, Verb, Logical Connector, etc.). Attention scores are biased toward pairs that have *high type compatibility* — i.e., pairs that "make logical sense" according to the Calculus of Constructions proof engine.

**Why this matters during training**: The CoC proof score acts as a regularization signal. Rather than purely minimizing cross-entropy loss, the model is also penalized for attention patterns that are logically incoherent. This helps it generalize with fewer data and resist hallucination.

---

## 🏛️ What is Calculus of Constructions (CoC)?

The **Calculus of Constructions (CoC)** is a higher-order typed lambda calculus developed by Thierry Coquand and Gérard Huet in 1988. It forms the mathematical foundation of formal proof assistants like Coq and Lean.

### The Core Insight: Curry-Howard Isomorphism

$$\text{Propositions} \equiv \text{Types} \qquad \text{and} \qquad \text{Proofs} \equiv \text{Programs / Terms}$$

This means:
- **Every type is a proposition** — the type `Nat → Nat` is the proposition "there exists a function from natural numbers to natural numbers"
- **Every program is a proof** — if a program type-checks, it is itself a proof that its type/proposition holds
- **Type errors = logical fallacies** — a program that doesn't type-check corresponds to an invalid logical argument

### Dependent Types — the Key Concept

A **dependent type** is a type that depends on a *value*:

| Ordinary Type | Dependent Type |
|---|---|
| `List` — a list (of any length) | `Vector(n)` — a list of *exactly* `n` elements |
| `Int → Int` — any integer function | `Π(n:Int). Matrix(n,n)` — an n×n matrix *for that specific n* |

In our neural network: a token's logical type can depend on what other tokens it's connected to — it's not a fixed label but a *context-sensitive proposition*.

---

## 🔬 Integrating CoC with Transformer Attention

### Step 1: Standard Attention (No Logic)

Standard transformer self-attention scores are purely geometric:

$$A_{ij} = \text{softmax}\left(\frac{q_i k_j^T}{\sqrt{d_k}} + M_{ij}\right)$$

where:
- $q_i$ = query vector for position $i$ (what am I looking for?)
- $k_j$ = key vector for position $j$ (what do I offer?)
- $M_{ij}$ = causal mask (−∞ for future tokens, 0 otherwise)
- $d_k$ = key dimension (used for scaling to avoid saturation)

> **Problem**: This scores only geometric similarity. Two tokens with similar embeddings get high attention *regardless of whether they logically co-occur*.

### Step 2: Type Embedding Projection

Each query and key token is projected into a separate **type embedding space**:

```cpp
// From dependent_type_attention.cpp
// W_type_q and W_type_k are learnable matrices (embed_dim → type_dim)
// They project high-dimensional token embeddings into a compact type space

float scale = std::sqrt(2.0f / static_cast<float>(embed_dim + type_dim));
W_type_q = ring0::Matrix::random_normal(embed_dim, type_dim, 0.0f, scale);
W_type_k = ring0::Matrix::random_normal(embed_dim, type_dim, 0.0f, scale);
```

> **What this does**: Initializes two learnable matrices with He-style scaling. `W_type_q` projects *query* tokens into type space; `W_type_k` projects *key* tokens into type space. The scaling `√(2/(d_in + d_out))` keeps initial activations from being too large or small, helping early training stability. These matrices are learned during training — the network discovers what "type space" means on its own.

### Step 3: Type Compatibility Score

```cpp
// From dependent_type_attention.cpp — compute_type_compatibility()
for (int i = 0; i < T; ++i) {
    // Project query token i into type space: tau_q = Q[i] * W_type_q
    std::vector<float> tau_q(type_dim, 0.0f);
    for (size_t d = 0; d < type_dim; ++d) {
        float sum = 0.0f;
        for (size_t c = 0; c < embed_dim; ++c) {
            sum += Q(b, i, c) * W_type_q(c, d); // Matrix multiply row of Q by W_type_q column
        }
        tau_q[d] = sum; // tau_q[d] = d-th component of token i's type embedding
    }

    // For each past token j (causal — j <= i only):
    for (int j = 0; j <= i; ++j) {
        // Project key token j into type space: tau_k = K[j] * W_type_k
        std::vector<float> tau_k(type_dim, 0.0f);
        for (size_t d = 0; d < type_dim; ++d) {
            float sum = 0.0f;
            for (size_t c = 0; c < embed_dim; ++c) {
                sum += K(b, j, c) * W_type_k(c, d);
            }
            tau_k[d] = sum;
        }

        // Compute cosine similarity between tau_q and tau_k
        float dot_type = 0.0f, norm_q = 0.0f, norm_k = 0.0f;
        for (size_t d = 0; d < type_dim; ++d) {
            dot_type += tau_q[d] * tau_k[d]; // Dot product of type embeddings
            norm_q   += tau_q[d] * tau_q[d]; // ||tau_q||^2
            norm_k   += tau_k[d] * tau_k[d]; // ||tau_k||^2
        }

        float denom = (std::sqrt(norm_q) * std::sqrt(norm_k)) + 1e-4f; // Avoid division by zero
        float type_match = (dot_type / denom) * inv_sqrt_type;          // Normalized type score
        compatibility(b, i, j) = std::clamp(type_match, -2.0f, 2.0f);  // Clamp to safe range
    }
}
```

> **What this does (step by step)**:
> 1. For each query token `i`, compute its type embedding `tau_q` by multiplying its embedding through `W_type_q`
> 2. For each key token `j` that `i` is allowed to look at (j ≤ i, because this is a causal model), compute its type embedding `tau_k`
> 3. Compute **cosine similarity** between `tau_q` and `tau_k` — this gives a score from −1 to +1 representing how compatible the two tokens' types are
> 4. Scale by `inv_sqrt_type` (analogous to the `1/√d_k` scaling in normal attention) to keep magnitudes reasonable
> 5. Clamp to [−2, +2] to prevent extreme values from dominating the softmax

### Step 4: Type-Guided Attention Formula

In **Dependent-Typed Attention**, the CoC proof consistency score modifies attention logits:

$$A_{ij}^{\text{CoC}} = \text{softmax}\left(\frac{q_i k_j^T}{\sqrt{d_k}} + \alpha \cdot \log P(\tau_j \vdash \pi_i) + M_{ij}\right)$$

where:
- $\tau_j$ = the dependent type assigned to token $j$ (its type embedding)
- $\pi_i$ = the proof requirement of token $i$ (what type it "needs" to see)
- $P(\tau_j \vdash \pi_i) \in [0, 1]$ = proof consistency score from `ring0::CoCEngine`
- $\alpha$ = `coc_type_guidance_alpha` (default 0.25 in `config.hpp`) — how strongly logic biases attention

```cpp
// From dependent_type_attention.cpp — apply_type_guidance()
void DependentTypeAttention::apply_type_guidance(
    ring0::Tensor3D &attn_weights,          // The attention logit matrix (BEFORE softmax)
    const ring0::Tensor3D &type_compatibility) // The type scores computed above
{
    // Add alpha * type_compatibility to every attention logit
    // This biases the softmax toward logically compatible pairs
    for (int b = 0; b < B; ++b)
        for (int i = 0; i < T; ++i)
            for (int j = 0; j <= i; ++j)
                attn_weights(b, i, j) += type_guidance_alpha * type_compatibility(b, i, j);
    //                                    ^^^^^^^^^^^^^^^^^^   ^^^^^^^^^^^^^^^^^^^^^^^^^^
    //                                    strength of logic     computed CoC type score
    //                                    signal (0.25 default)  (+ve = compatible, -ve = incompatible)
}
```

> **What this does**: Before the softmax is applied, each attention logit gets a bonus (or penalty) based on type compatibility. If tokens `i` and `j` have highly compatible type signatures, their attention score goes *up* — making it more likely that `i` attends to `j`. If they're incompatible types (e.g., a logical connective attending to itself), the score goes *down*. The `alpha = 0.25` means type logic contributes 25% as strongly as the raw geometric attention score.

---

## 🛡️ Telemetry & Proof Consistency Monitoring

During training, the Chrono Async Engine periodically audits the model's proof consistency score:

```
[CoC Logic & Proof]   Proof Consistency: 100.0% | Type-Attention Prior: ACTIVE (alpha=0.25)
```

| Field | What It Means | Good Value |
|-------|---------------|------------|
| `Proof Consistency` | % of thought-chain steps that passed formal CoC type verification | ≥ 85% |
| `Type-Attention Prior` | Whether DependentTypeAttention is actively biasing scores | ACTIVE |
| `alpha` | Strength of the type-guidance term (from `coc_type_guidance_alpha` in config) | 0.25 |

### What Low Proof Consistency Means in Practice

If the proof consistency score drops below 85% (`coc_proof_consistency_threshold` in `config.hpp`):
- The model's reasoning steps are no longer logically coherent
- Attention patterns are likely focusing on semantically wrong token pairs
- Training instability may follow (loss spikes, gradient explosions)

This typically happens during **rapid early learning** when the type embeddings haven't yet organized into meaningful clusters, or after **bad batch rollbacks** disturb the type embedding matrices.

---

## 🔧 Config Parameters (from `config.hpp`)

| Parameter | Default | Plain-English Meaning |
|-----------|---------|----------------------|
| `enable_coc_verification` | `true` | Turn on/off the whole CoC proof checker |
| `coc_verification_interval` | `5` | Run proof checks every 5 steps (not every step — too expensive) |
| `coc_type_guidance_alpha` | `0.25f` | How much weight logic gives to attention (0 = none, 1 = equal to geometric) |
| `enable_coc_universe_stratification` | `true` | Enforce Prop < Type_0 < Type_1 hierarchy to prevent type paradoxes |
| `max_beta_reduction_steps` | `1000` | Stop normalizing after 1000 steps — catches infinite-loop edge cases |
| `coc_proof_consistency_threshold` | `0.85f` | Below this score, flag degraded logical reasoning |

---

## 📎 See Also

- [[01 - Ring 0 (Core Math & Hardware)/Calculus of Constructions & Dependent-Typed Neural Reasoning|CoC Implementation Details (Ring 0)]]
- [[02 - Ring 1 (Layers & Advanced Optimizers)/Attention Mechanics & ALiBi|Standard Attention + ALiBi]]
- [[06 - Reference Dictionaries & Practical Guides/Config Values Reference|Full Config Reference]]
