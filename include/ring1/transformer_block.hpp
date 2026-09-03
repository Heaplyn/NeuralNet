#pragma once

/**
 * @file transformer_block.hpp
 * @brief Pre-LayerNorm Transformer Decoder Block with Causal Self-Attention, SwiGLU Gated FFN, Dynamic Growth, and KV-Caching in Ring 1.
 */

#include "ring0/tensor.hpp"
#include "ring0/activations.hpp"
#include "ring1/attention.hpp"

using namespace std;

namespace ring1 {

/**
 * @class TransformerBlock
 * @brief Transformer Decoder Block with Pre-RMSNorm, SwiGLU Bilinear Gated MLP, and Dynamic Capacity Growth.
 * 
 * Pipeline:
 *   1. H = X + Attention(RMSNorm1(X))
 *   2. Y = H + SwiGLU(RMSNorm2(H))
 *   where SwiGLU(u) = (SiLU(u * W_gate + b_gate) .* (u * W_up + b_up)) * W_down + b_down
 */
class TransformerBlock {
public:
    size_t embed_dim;                ///< Embedding dimension (C)
    size_t ffn_dim;                  ///< Feed-Forward expansion dimension
    size_t num_heads;                ///< Number of attention heads

    MultiHeadAttention attention;   ///< Grouped-Query Causal Self-Attention sublayer

    // --- RMSNorm 1 & 2 Parameters ---
    ring0::Matrix ln1_gamma, ln1_beta;
    ring0::Matrix grad_ln1_gamma, grad_ln1_beta;

    ring0::Matrix ln2_gamma, ln2_beta;
    ring0::Matrix grad_ln2_gamma, grad_ln2_beta;

    // --- SwiGLU Gated MLP Parameters (LLaMA / Mistral Architecture) ---
    ring0::Matrix W_gate, b_gate;    ///< Gate projection (embed_dim -> ffn_dim)
    ring0::Matrix W_up, b_up;        ///< Up projection (embed_dim -> ffn_dim)
    ring0::Matrix W_down, b_down;    ///< Down projection (ffn_dim -> embed_dim)

    ring0::Matrix grad_W_gate, grad_b_gate;
    ring0::Matrix grad_W_up, grad_b_up;
    ring0::Matrix grad_W_down, grad_b_down;

    // --- Backpropagation Cache ---
    ring0::Tensor3D last_input;
    ring0::Tensor3D last_ln1;
    ring0::Tensor3D last_attn_res;
    ring0::Tensor3D last_ln2;
    ring0::Matrix last_gate_linear;
    ring0::Matrix last_up_linear;
    ring0::Matrix last_silu_gate;
    ring0::Matrix last_gated;

    TransformerBlock(size_t dim, size_t heads, size_t ff_dim, size_t kv_heads = 0);

    /// Forward pass through Pre-RMSNorm Attention and Pre-RMSNorm SwiGLU MLP for full sequence
    ring0::Tensor3D forward(const ring0::Tensor3D& input);

    /**
     * @brief O(1) single-step forward pass using KV-Cache for fast autoregressive generation.
     * @param x_token Input vector for single token (1 x embed_dim).
     * @param pos_idx Current token position index.
     * @param cache Layer KV-Cache.
     * @return Output vector for single token (1 x embed_dim).
     */
    ring0::Matrix forward_step(const ring0::Matrix& x_token, size_t pos_idx, LayerKVCache& cache);

    /// Backward pass computing gradients for all parameters and returning dInput
    ring0::Tensor3D backward(const ring0::Tensor3D& grad_output);

    /// Dynamically expands SwiGLU hidden feature dimension as the model masters lower loss levels
    void expand_ffn_dim(size_t additional_dim);

    /// Resets all gradient matrices to zero
    void reset_gradients();
};

} // namespace ring1
