#include "ring1/embedding.hpp"
#include <cmath>

using namespace std;

namespace ring1
{

    // Embedding constructor: Initializes weights with standard normal distribution (std = 0.02)
    EmbeddingLayer::EmbeddingLayer(size_t vocab_sz, size_t max_seq, size_t dim)
        : vocab_size(vocab_sz), max_seq_len(max_seq), embed_dim(dim),
          last_batch_size(0), last_seq_len(0)
    {

        token_weights = ring0::Matrix::random_normal(vocab_size, embed_dim, 0.0f, 0.02f);
        pos_weights = ring0::Matrix::random_normal(max_seq_len, embed_dim, 0.0f, 0.02f);

        grad_token_weights = ring0::Matrix::zeros(vocab_size, embed_dim);
        grad_pos_weights = ring0::Matrix::zeros(max_seq_len, embed_dim);
    }

    // Forward pass: token lookup + positional embedding addition
    ring0::Tensor3D EmbeddingLayer::forward(const vector<int> &token_ids, size_t batch_size, size_t seq_len)
    {
        last_token_indices = token_ids;
        last_batch_size = batch_size;
        last_seq_len = seq_len;

        ring0::Tensor3D output(batch_size, seq_len, embed_dim);

        for (size_t b = 0; b < batch_size; ++b)
        {
            for (size_t s = 0; s < seq_len; ++s)
            {
                int token = token_ids[b * seq_len + s];
                for (size_t d = 0; d < embed_dim; ++d)
                {
                    // Lookup token embedding
                    float tok_val = (token >= 0 && static_cast<size_t>(token) < vocab_size)
                                        ? token_weights(token, d)
                                        : 0.0f;
                    // Lookup position embedding
                    float pos_val = (s < max_seq_len) ? pos_weights(s, d) : 0.0f;
                    // Sum embeddings
                    output(b, s, d) = tok_val + pos_val;
                }
            }
        }

        return output;
    }

    // Backward pass: accumulates incoming gradients directly into token and position gradient tables
    void EmbeddingLayer::backward(const ring0::Tensor3D &grad_output)
    {
        for (size_t b = 0; b < last_batch_size; ++b)
        {
            for (size_t s = 0; s < last_seq_len; ++s)
            {
                int token = last_token_indices[b * last_seq_len + s];
                for (size_t d = 0; d < embed_dim; ++d)
                {
                    float g = grad_output(b, s, d);
                    if (token >= 0 && static_cast<size_t>(token) < vocab_size)
                    {
                        grad_token_weights(token, d) += g;
                    }
                    if (s < max_seq_len)
                    {
                        grad_pos_weights(s, d) += g;
                    }
                }
            }
        }
    }

    // Resets gradient tables to zero
    void EmbeddingLayer::reset_gradients()
    {
        grad_token_weights = ring0::Matrix::zeros(vocab_size, embed_dim);
        grad_pos_weights = ring0::Matrix::zeros(max_seq_len, embed_dim);
    }

    // Dynamically expands vocabulary row dimension as training steps unlock new subwords
    void EmbeddingLayer::expand_vocab(size_t new_vocab_size)
    {
        if (new_vocab_size <= vocab_size)
            return;
        size_t old_v = vocab_size;
        vocab_size = new_vocab_size;
        token_weights.expand_rows(new_vocab_size);
        grad_token_weights = ring0::Matrix::zeros(new_vocab_size, embed_dim);

        // Initialize newly added token rows with small Gaussian noise
        float stddev = 0.02f;
        for (size_t r = old_v; r < new_vocab_size; ++r)
        {
            for (size_t c = 0; c < embed_dim; ++c)
            {
                float u1 = (rand() + 1.0f) / (RAND_MAX + 1.0f);
                float u2 = (rand() + 1.0f) / (RAND_MAX + 1.0f);
                token_weights(r, c) = sqrt(-2.0f * log(u1)) * cos(2.0f * 3.14159265f * u2) * stddev;
            }
        }
    }

    // Dynamically expands maximum supported sequence length and positional embeddings
    void EmbeddingLayer::expand_max_seq_len(size_t new_max_seq)
    {
        if (new_max_seq <= max_seq_len)
            return;
        size_t old_max = max_seq_len;
        max_seq_len = new_max_seq;

        // Expand positional weights matrix and gradient matrix
        pos_weights.expand_rows(new_max_seq);
        grad_pos_weights = ring0::Matrix::zeros(new_max_seq, embed_dim);

        // Initialize newly added positional rows with small Gaussian noise
        float stddev = 0.02f;
        for (size_t r = old_max; r < new_max_seq; ++r)
        {
            for (size_t c = 0; c < embed_dim; ++c)
            {
                float u1 = (rand() + 1.0f) / (RAND_MAX + 1.0f);
                float u2 = (rand() + 1.0f) / (RAND_MAX + 1.0f);
                pos_weights(r, c) = sqrt(-2.0f * log(u1)) * cos(2.0f * 3.14159265f * u2) * stddev;
            }
        }
    }

} // namespace ring1
