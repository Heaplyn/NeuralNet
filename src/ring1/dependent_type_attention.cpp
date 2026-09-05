#include "ring1/dependent_type_attention.hpp"
#include <cmath>
#include <algorithm>

namespace ring1
{

    DependentTypeAttention::DependentTypeAttention(size_t dim, size_t t_dim, float alpha)
        : embed_dim(dim), type_dim(t_dim), type_guidance_alpha(alpha)
    {
        float scale = std::sqrt(2.0f / static_cast<float>(embed_dim + type_dim));
        W_type_q = ring0::Matrix::random_normal(embed_dim, type_dim, 0.0f, scale);
        W_type_k = ring0::Matrix::random_normal(embed_dim, type_dim, 0.0f, scale);

        grad_W_type_q = ring0::Matrix::zeros(embed_dim, type_dim);
        grad_W_type_k = ring0::Matrix::zeros(embed_dim, type_dim);
    }

    ring0::Tensor3D DependentTypeAttention::compute_type_compatibility(const ring0::Tensor3D &Q, const ring0::Tensor3D &K)
    {
        size_t B = Q.batch_size;
        size_t T = Q.seq_len;
        ring0::Tensor3D compatibility(B, T, T, 0.0f);

        // Project Q and K into type space
        // Q: (B, T, embed_dim) * W_type_q (embed_dim, type_dim) -> Tau_Q: (B, T, type_dim)
        float inv_sqrt_type = 1.0f / std::sqrt(static_cast<float>(type_dim) + 1e-6f);

#pragma omp parallel for schedule(static)
        for (int b = 0; b < static_cast<int>(B); ++b)
        {
            for (int i = 0; i < static_cast<int>(T); ++i)
            {
                std::vector<float> tau_q(type_dim, 0.0f);
                for (size_t d = 0; d < type_dim; ++d)
                {
                    float sum = 0.0f;
                    for (size_t c = 0; c < embed_dim; ++c)
                    {
                        sum += Q(b, i, c) * W_type_q(c, d);
                    }
                    tau_q[d] = sum;
                }

                // Compute cosine-like dot product with all keys j <= i
                for (int j = 0; j <= i; ++j)
                {
                    std::vector<float> tau_k(type_dim, 0.0f);
                    for (size_t d = 0; d < type_dim; ++d)
                    {
                        float sum = 0.0f;
                        for (size_t c = 0; c < embed_dim; ++c)
                        {
                            sum += K(b, j, c) * W_type_k(c, d);
                        }
                        tau_k[d] = sum;
                    }

                    float dot_type = 0.0f;
                    float norm_q = 0.0f;
                    float norm_k = 0.0f;
                    for (size_t d = 0; d < type_dim; ++d)
                    {
                        dot_type += tau_q[d] * tau_k[d];
                        norm_q += tau_q[d] * tau_q[d];
                        norm_k += tau_k[d] * tau_k[d];
                    }
                    float denom = (std::sqrt(norm_q) * std::sqrt(norm_k)) + 1e-4f;
                    float type_match = (dot_type / denom) * inv_sqrt_type;
                    compatibility(b, i, j) = std::clamp(type_match, -2.0f, 2.0f);
                }
            }
        }
        return compatibility;
    }

    void DependentTypeAttention::apply_type_guidance(ring0::Tensor3D &attn_weights, const ring0::Tensor3D &type_compatibility)
    {
        size_t B = attn_weights.batch_size;
        size_t T = attn_weights.seq_len;

#pragma omp parallel for schedule(static)
        for (int b = 0; b < static_cast<int>(B); ++b)
        {
            for (int i = 0; i < static_cast<int>(T); ++i)
            {
                for (int j = 0; j <= i; ++j)
                {
                    // Add constructive type compatibility prior
                    attn_weights(b, i, j) += type_guidance_alpha * type_compatibility(b, i, j);
                }
            }
        }
    }

    void DependentTypeAttention::reset_gradients()
    {
        grad_W_type_q = ring0::Matrix::zeros(embed_dim, type_dim);
        grad_W_type_k = ring0::Matrix::zeros(embed_dim, type_dim);
    }

    void DependentTypeAttention::update_parameters(float lr)
    {
        for (size_t r = 0; r < W_type_q.rows; ++r)
        {
            for (size_t c = 0; c < W_type_q.cols; ++c)
            {
                W_type_q(r, c) -= lr * std::clamp(grad_W_type_q(r, c), -0.1f, 0.1f);
            }
        }
        for (size_t r = 0; r < W_type_k.rows; ++r)
        {
            for (size_t c = 0; c < W_type_k.cols; ++c)
            {
                W_type_k(r, c) -= lr * std::clamp(grad_W_type_k(r, c), -0.8f, 0.8f);
            }
        }
    }

} // namespace ring1
