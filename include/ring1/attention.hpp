#pragma once

/**
 * @file attention.hpp
 * @brief Grouped-Query Attention (GQA), Rotary Position Embeddings (RoPE), ALiBi distance-decay attention bias, and KV-Caching in Ring 1.
 */

#include "ring0/tensor.hpp"
#include <vector>

using namespace std;

namespace ring1 {

/**
 * @struct LayerKVCache
 * @brief Stores projected Key and Value matrices across sequential decoding steps.
 */
struct LayerKVCache {
    ring0::Matrix k_cache; ///< Cached Keys: shape (Cached_Seq_Len x kv_dim)
    ring0::Matrix v_cache; ///< Cached Values: shape (Cached_Seq_Len x kv_dim)

    /// Clears cached Key and Value vectors
    void clear() {
        k_cache = ring0::Matrix();
        v_cache = ring0::Matrix();
    }

    /// Appends a new 1xkv_dim row to Key and Value cache matrices
    void append(const ring0::Matrix& k_new, const ring0::Matrix& v_new) {
        if (k_cache.rows == 0) {
            k_cache = k_new;
            v_cache = v_new;
        } else {
            size_t old_r = k_cache.rows;
            size_t cols = k_cache.cols;
            k_cache.expand_rows(old_r + 1);
            for (size_t c = 0; c < cols; ++c) {
                k_cache(old_r, c) = k_new(0, c);
            }

            v_cache.expand_rows(old_r + 1);
            for (size_t c = 0; c < cols; ++c) {
                v_cache(old_r, c) = v_new(0, c);
            }
        }
    }
};

/**
 * @class MultiHeadAttention
 * @brief Grouped-Query Attention (GQA) with RoPE relative rotation, ALiBi distance falloff, and KV-cached decoding.
 */
class MultiHeadAttention {
public:
    size_t embed_dim;    ///< Total embedding dimension (e.g. 96)
    size_t num_heads;    ///< Number of Query heads (e.g. 4)
    size_t num_kv_heads; ///< Number of Key/Value heads (e.g. 2 for GQA)
    size_t head_dim;     ///< Dimension per head = embed_dim / num_heads (e.g. 24)
    size_t kv_dim;       ///< Dimension of Key/Value representations = num_kv_heads * head_dim
    size_t group_size;   ///< Number of Query heads per KV head = num_heads / num_kv_heads

    // --- Learnable Projection Matrices ---
    ring0::Matrix W_q, b_q; ///< Query projection weights & bias (embed_dim -> embed_dim)
    ring0::Matrix W_k, b_k; ///< Key projection weights & bias (embed_dim -> kv_dim)
    ring0::Matrix W_v, b_v; ///< Value projection weights & bias (embed_dim -> kv_dim)
    ring0::Matrix W_o, b_o; ///< Output linear projection weights & bias (embed_dim -> embed_dim)

    // --- Gradient Accumulators ---
    ring0::Matrix grad_W_q, grad_b_q;
    ring0::Matrix grad_W_k, grad_b_k;
    ring0::Matrix grad_W_v, grad_b_v;
    ring0::Matrix grad_W_o, grad_b_o;

    // --- Backpropagation Cache ---
    ring0::Tensor3D last_input;                    ///< Cached input X
    ring0::Tensor3D last_Q, last_K, last_V;        ///< Cached projected Q, K, V
    vector<ring0::Matrix> last_attn_weights;       ///< Attention probabilities
    ring0::Tensor3D last_attn_output;              ///< Concatenated head outputs

    MultiHeadAttention(size_t dim, size_t q_heads, size_t kv_heads = 0);

    /// Computes ALiBi head slope m_h = 2^(-8 * (h + 1) / num_heads) * 0.25f
    float get_alibi_slope(size_t head_idx) const;

    /// Forward pass of Grouped-Query Attention with RoPE and ALiBi distance falloff for training
    ring0::Tensor3D forward(const ring0::Tensor3D& input);

    /**
     * @brief O(1) single-step forward pass using GQA KV-Cache, RoPE, and ALiBi distance falloff.
     */
    ring0::Matrix forward_step(const ring0::Matrix& x_token, size_t pos_idx, LayerKVCache& cache);

    /// Backward pass computing parameter and input gradients for GQA
    ring0::Tensor3D backward(const ring0::Tensor3D& grad_output);

    /// Resets all gradient accumulators to zero
    void reset_gradients();
};

} // namespace ring1
