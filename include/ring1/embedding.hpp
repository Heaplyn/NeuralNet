#pragma once

/**
 * @file embedding.hpp
 * @brief Token and learnable positional embedding layers for Transformer models in Ring 1.
 */

#include "ring0/tensor.hpp"
#include <vector>

using namespace std;

namespace ring1
{

    /**
     * @class EmbeddingLayer
     * @brief Maps token IDs to continuous dense vectors and adds learned positional embeddings.
     *
     * Formula for token at position s in sequence:
     *   Embedding(token, s) = Token_Weights[token] + Positional_Weights[s]
     */
    class EmbeddingLayer
    {
    public:
        size_t vocab_size;  ///< Total number of distinct tokens in vocabulary (V)
        size_t max_seq_len; ///< Maximum supported sequence length (T_max)
        size_t embed_dim;   ///< Channel / Embedding dimension (C)

        ring0::Matrix token_weights; ///< Token embedding table (vocab_size x embed_dim)
        ring0::Matrix pos_weights;   ///< Positional embedding table (max_seq_len x embed_dim)

        ring0::Matrix grad_token_weights; ///< Gradient accumulator for token weights
        ring0::Matrix grad_pos_weights;   ///< Gradient accumulator for positional weights

        // --- Backpropagation Cache ---
        vector<int> last_token_indices; ///< Cached input token IDs (flat B*T)
        size_t last_batch_size;         ///< Cached batch size (B)
        size_t last_seq_len;            ///< Cached sequence length (T)

        /**
         * @brief Constructs embedding tables initialized with Gaussian weights N(0, 0.02).
         */
        EmbeddingLayer(size_t vocab_sz, size_t max_seq, size_t dim);

        /**
         * @brief Performs forward lookup and sums token + position embeddings.
         * @param token_ids Flattened vector of token IDs of size B * T.
         * @param batch_size Number of batch items (B).
         * @param seq_len Sequence length (T).
         * @return 3D Tensor of shape (Batch, Seq_Len, Embed_Dim).
         */
        ring0::Tensor3D forward(const vector<int> &token_ids, size_t batch_size, size_t seq_len);

        /**
         * @brief Backward pass accumulating gradients into token_weights and pos_weights.
         * @param grad_output Gradient of loss with respect to embedding output (shape: B x T x C).
         */
        void backward(const ring0::Tensor3D &grad_output);

        /// Dynamically expands vocabulary row dimension as training steps unlock new subwords
        void expand_vocab(size_t new_vocab_size);

        /// Dynamically expands maximum supported sequence length and positional embeddings
        void expand_max_seq_len(size_t new_max_seq);

        /// Resets gradient tables to zero
        void reset_gradients();
    };

} // namespace ring1
