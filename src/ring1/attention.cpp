#include "ring1/attention.hpp"
#include <cmath>
#include <stdexcept>
#include <algorithm>

using namespace std;

namespace ring1
{

    // Constructor: Initializes Q, K, V, O projections for Grouped-Query Attention (GQA)
    MultiHeadAttention::MultiHeadAttention(size_t dim, size_t q_heads, size_t kv_heads)
        : embed_dim(dim),
          num_heads(q_heads),
          num_kv_heads(kv_heads == 0 ? q_heads : kv_heads),
          head_dim(dim / q_heads)
    {

        if (dim % q_heads != 0)
        {
            throw invalid_argument("embed_dim must be divisible by num_heads");
        }
        if (q_heads % num_kv_heads != 0)
        {
            throw invalid_argument("num_heads must be divisible by num_kv_heads");
        }

        group_size = num_heads / num_kv_heads;
        kv_dim = num_kv_heads * head_dim;

        float scale_q = sqrt(2.0f / static_cast<float>(dim));
        float scale_kv = sqrt(2.0f / static_cast<float>(dim));
        float scale_o = sqrt(2.0f / static_cast<float>(dim));

        W_q = ring0::Matrix::random_normal(dim, dim, 0.0f, scale_q);
        b_q = ring0::Matrix::zeros(1, dim);

        W_k = ring0::Matrix::random_normal(dim, kv_dim, 0.0f, scale_kv);
        b_k = ring0::Matrix::zeros(1, kv_dim);

        W_v = ring0::Matrix::random_normal(dim, kv_dim, 0.0f, scale_kv);
        b_v = ring0::Matrix::zeros(1, kv_dim);

        W_o = ring0::Matrix::random_normal(dim, dim, 0.0f, scale_o);
        b_o = ring0::Matrix::zeros(1, dim);

        reset_gradients();
    }

    // Resets gradient matrices to zero
    void MultiHeadAttention::reset_gradients()
    {
        grad_W_q = ring0::Matrix::zeros(embed_dim, embed_dim);
        grad_b_q = ring0::Matrix::zeros(1, embed_dim);

        grad_W_k = ring0::Matrix::zeros(embed_dim, kv_dim);
        grad_b_k = ring0::Matrix::zeros(1, kv_dim);

        grad_W_v = ring0::Matrix::zeros(embed_dim, kv_dim);
        grad_b_v = ring0::Matrix::zeros(1, kv_dim);

        grad_W_o = ring0::Matrix::zeros(embed_dim, embed_dim);
        grad_b_o = ring0::Matrix::zeros(1, embed_dim);
    }

    // Computes ALiBi geometric slope for head h
    float MultiHeadAttention::get_alibi_slope(size_t head_idx) const
    {
        float power = 8.0f * static_cast<float>(head_idx + 1) / static_cast<float>(num_heads);
        return pow(2.0f, -power * 1.15f) * 0.8f; // Softened distance falloff
    }

    // Forward pass of Grouped-Query Attention (GQA) with ALiBi distance falloff (Training phase)
    ring0::Tensor3D MultiHeadAttention::forward(const ring0::Tensor3D &input)
    {
        last_input = input;
        size_t B = input.batch_size;
        size_t T = input.seq_len;

        // 1. Flatten input (B, T, C) -> (B*T, C)
        ring0::Matrix X_flat = input.to_matrix();

        // 2. Project Q (B*T, embed_dim), K (B*T, kv_dim), V (B*T, kv_dim)
        ring0::Matrix Q_flat = X_flat.matmul(W_q).add_bias(b_q);
        ring0::Matrix K_flat = X_flat.matmul(W_k).add_bias(b_k);
        ring0::Matrix V_flat = X_flat.matmul(W_v).add_bias(b_v);

        // Apply Rotary Position Embeddings (RoPE) to Query and Key projections
        Q_flat.apply_rope(num_heads, head_dim, 0);
        K_flat.apply_rope(num_kv_heads, head_dim, 0);

        last_Q = ring0::Tensor3D::from_matrix(Q_flat, B, T);
        last_K = ring0::Tensor3D::from_matrix(K_flat, B, T);
        last_V = ring0::Tensor3D::from_matrix(V_flat, B, T);

        last_attn_weights.resize(B * num_heads);

        ring0::Tensor3D head_concat(B, T, embed_dim);
        float scale = 1.0f / sqrt(static_cast<float>(head_dim));

// 3. Multi-Threaded Grouped-Query Attention Computation across all batches and heads
#pragma omp parallel for collapse(2) schedule(static)
        for (int b = 0; b < static_cast<int>(B); ++b)
        {
            for (int h = 0; h < static_cast<int>(num_heads); ++h)
            {
                size_t q_head_start = static_cast<size_t>(h) * head_dim;
                size_t kv_head_idx = static_cast<size_t>(h) / group_size;
                size_t kv_head_start = kv_head_idx * head_dim;
                float slope = get_alibi_slope(static_cast<size_t>(h));

                // (a) Scaled dot product with causal mask and ALiBi distance decay
                ring0::Matrix scores(T, T, 0.0f);
                for (size_t i = 0; i < T; ++i)
                {
                    float row_max = -1e30f;
                    // Compute dot products up to causal diagonal j <= i
                    for (size_t j = 0; j <= i; ++j)
                    {
                        float dot = 0.0f;
                        for (size_t d = 0; d < head_dim; ++d)
                        {
                            float q_v = last_Q(static_cast<size_t>(b), i, q_head_start + d);
                            float k_v = last_K(static_cast<size_t>(b), j, kv_head_start + d);
                            if (!std::isnan(q_v) && !std::isnan(k_v) && !std::isinf(q_v) && !std::isinf(k_v))
                            {
                                dot += q_v * k_v;
                            }
                        }
                        float distance = static_cast<float>(i - j);
                        float score = (dot * scale) - sqrt(slope * slope * distance * distance);
                        if (std::isnan(score) || std::isinf(score))
                            score = -50.0f;
                        scores(i, j) = score;
                        if (score > row_max)
                            row_max = score;
                    }
                    if (row_max < -1e20f)
                        row_max = 0.0f;

                    for (size_t j = i + 1; j < T; ++j)
                    {
                        scores(i, j) = -1e9f;
                    }
                }

                // (b) Exact causal row softmax with numerical clipping
                ring0::Matrix attn(T, T, 0.0f);
                for (size_t i = 0; i < T; ++i)
                {
                    float row_max = scores(i, 0);
                    for (size_t j = 1; j <= i; ++j)
                    {
                        if (scores(i, j) > row_max)
                            row_max = scores(i, j);
                    }
                    float sum_exp = 0.0f;
                    for (size_t j = 0; j <= i; ++j)
                    {
                        float diff = scores(i, j) - row_max;
                        diff = std::clamp(diff, -50.0f, 50.0f);
                        float p = expf(diff);
                        attn(i, j) = p;
                        sum_exp += p;
                    }
                    float inv_sum = 1.0f / max(1e-8f, sum_exp);
                    for (size_t j = 0; j <= i; ++j)
                    {
                        float p = attn(i, j) * inv_sum;
                        attn(i, j) = (std::isnan(p) || std::isinf(p)) ? (1.0f / static_cast<float>(i + 1)) : p;
                    }
                }
                last_attn_weights[static_cast<size_t>(b) * num_heads + static_cast<size_t>(h)] = attn;

                // (c) Multiply attention weights by corresponding Value vectors
                for (size_t i = 0; i < T; ++i)
                {
                    for (size_t d = 0; d < head_dim; ++d)
                    {
                        float val = 0.0f;
                        for (size_t j = 0; j <= i; ++j)
                        {
                            float v_v = last_V(static_cast<size_t>(b), j, kv_head_start + d);
                            if (!std::isnan(v_v) && !std::isinf(v_v))
                            {
                                val += attn(i, j) * v_v;
                            }
                        }
                        if (std::isnan(val) || std::isinf(val))
                            val = 0.0f;
                        head_concat(static_cast<size_t>(b), i, q_head_start + d) = val;
                    }
                }
            }
        }

        last_attn_output = head_concat;

        // 4. Output linear projection
        ring0::Matrix out_flat = head_concat.to_matrix().matmul(W_o).add_bias(b_o);
        out_flat.sanitize_nan_inf(0.0f, -50.0f, 50.0f);
        return ring0::Tensor3D::from_matrix(out_flat, B, T);
    }

    // O(1) single-step forward pass using GQA KV-Cache and RoPE for fast generation
    ring0::Matrix MultiHeadAttention::forward_step(const ring0::Matrix &x_token, size_t pos_idx, LayerKVCache &cache)
    {
        // 1. Project Query (1 x embed_dim), Key (1 x kv_dim), and Value (1 x kv_dim)
        ring0::Matrix q_cur = x_token.matmul(W_q).add_bias(b_q);
        ring0::Matrix k_cur = x_token.matmul(W_k).add_bias(b_k);
        ring0::Matrix v_cur = x_token.matmul(W_v).add_bias(b_v);

        // Apply RoPE rotation at current token position
        q_cur.apply_rope(num_heads, head_dim, pos_idx);
        k_cur.apply_rope(num_kv_heads, head_dim, pos_idx);

        // 2. Append new Key and Value to layer cache (storing only kv_dim features)
        cache.append(k_cur, v_cur);

        size_t cache_len = cache.k_cache.rows;
        ring0::Matrix head_concat(1, embed_dim, 0.0f);
        float scale = 1.0f / sqrt(static_cast<float>(head_dim));

        // 3. Compute Grouped-Query Attention against cached tokens
        for (size_t h = 0; h < num_heads; ++h)
        {
            size_t q_head_start = h * head_dim;
            size_t kv_head_idx = h / group_size;
            size_t kv_head_start = kv_head_idx * head_dim;
            float slope = get_alibi_slope(h);

            ring0::Matrix scores(1, cache_len, 0.0f);
            size_t curr_pos = cache_len - 1;

            for (size_t j = 0; j < cache_len; ++j)
            {
                float dot = 0.0f;
                for (size_t d = 0; d < head_dim; ++d)
                {
                    dot += q_cur(0, q_head_start + d) * cache.k_cache(j, kv_head_start + d);
                }
                float distance = static_cast<float>(curr_pos - j);
                scores(0, j) = (dot * scale) - (slope * distance);
            }

            ring0::Matrix attn = scores.softmax_rows();

            for (size_t d = 0; d < head_dim; ++d)
            {
                float val = 0.0f;
                for (size_t j = 0; j < cache_len; ++j)
                {
                    val += attn(0, j) * cache.v_cache(j, kv_head_start + d);
                }
                head_concat(0, q_head_start + d) = val;
            }
        }

        // 4. Output projection
        return head_concat.matmul(W_o).add_bias(b_o);
    }

    // Backward pass computing parameter and input gradients for GQA
    ring0::Tensor3D MultiHeadAttention::backward(const ring0::Tensor3D &grad_output)
    {
        size_t B = grad_output.batch_size;
        size_t T = grad_output.seq_len;
        float scale = 1.0f / sqrt(static_cast<float>(head_dim));

        // 1. Output projection gradients
        ring0::Matrix grad_out_flat = grad_output.to_matrix();
        grad_W_o += last_attn_output.to_matrix().transpose().matmul(grad_out_flat);
        grad_b_o += grad_out_flat.sum_rows();

        ring0::Matrix dHeadConcat_flat = grad_out_flat.matmul(W_o.transpose());
        ring0::Tensor3D dHeadConcat = ring0::Tensor3D::from_matrix(dHeadConcat_flat, B, T);

        ring0::Tensor3D dQ(B, T, embed_dim, 0.0f);
        ring0::Tensor3D dK(B, T, kv_dim, 0.0f);
        ring0::Tensor3D dV(B, T, kv_dim, 0.0f);

#pragma omp parallel for schedule(static)
        for (int b_idx = 0; b_idx < static_cast<int>(B); ++b_idx)
        {
            size_t b = static_cast<size_t>(b_idx);
            for (size_t h = 0; h < num_heads; ++h)
            {
                size_t q_head_start = h * head_dim;
                size_t kv_head_idx = h / group_size;
                size_t kv_head_start = kv_head_idx * head_dim;

                const ring0::Matrix &attn = last_attn_weights[b * num_heads + h];

                // Gradient through Value multiplication: Out = Attn * V
                ring0::Matrix dAttn(T, T, 0.0f);
                for (size_t i = 0; i < T; ++i)
                {
                    for (size_t j = 0; j <= i; ++j)
                    {
                        float sum_v = 0.0f;
                        for (size_t d = 0; d < head_dim; ++d)
                        {
                            float dOut = dHeadConcat(b, i, q_head_start + d);
                            sum_v += dOut * last_V(b, j, kv_head_start + d);
                            dV(b, j, kv_head_start + d) += dOut * attn(i, j);
                        }
                        dAttn(i, j) = sum_v;
                    }
                }

                // Gradient through Softmax
                ring0::Matrix dScores(T, T, 0.0f);
                for (size_t i = 0; i < T; ++i)
                {
                    float sum_dAttn_Attn = 0.0f;
                    for (size_t j = 0; j <= i; ++j)
                    {
                        sum_dAttn_Attn += dAttn(i, j) * attn(i, j);
                    }
                    for (size_t j = 0; j <= i; ++j)
                    {
                        dScores(i, j) = attn(i, j) * (dAttn(i, j) - sum_dAttn_Attn);
                    }
                }

                // Gradient through Q * K^T dot product
                for (size_t i = 0; i < T; ++i)
                {
                    for (size_t j = 0; j <= i; ++j)
                    {
                        float dS = dScores(i, j) * scale;
                        for (size_t d = 0; d < head_dim; ++d)
                        {
                            dQ(b, i, q_head_start + d) += dS * last_K(b, j, kv_head_start + d);
                            dK(b, j, kv_head_start + d) += dS * last_Q(b, i, q_head_start + d);
                        }
                    }
                }
            }
        }

        // 2. Gradients through Q, K, V projections
        ring0::Matrix dQ_flat = dQ.to_matrix();
        ring0::Matrix dK_flat = dK.to_matrix();
        ring0::Matrix dV_flat = dV.to_matrix();

        ring0::Matrix X_flat = last_input.to_matrix();
        ring0::Matrix X_T = X_flat.transpose();

        grad_W_q += X_T.matmul(dQ_flat);
        grad_b_q += dQ_flat.sum_rows();

        grad_W_k += X_T.matmul(dK_flat);
        grad_b_k += dK_flat.sum_rows();

        grad_W_v += X_T.matmul(dV_flat);
        grad_b_v += dV_flat.sum_rows();

        // 3. Input gradient dX using zero-alloc pre-transposed matmul
        ring0::Matrix dX_flat(dQ_flat.rows, W_q.rows, 0.0f);
        ring0::Matrix dK_proj(dK_flat.rows, W_k.rows, 0.0f);
        ring0::Matrix dV_proj(dV_flat.rows, W_v.rows, 0.0f);

        dQ_flat.matmul_transB_into(W_q, dX_flat);
        dK_flat.matmul_transB_into(W_k, dK_proj);
        dV_flat.matmul_transB_into(W_v, dV_proj);

        dX_flat += dK_proj;
        dX_flat += dV_proj;

        return ring0::Tensor3D::from_matrix(dX_flat, B, T);
    }

} // namespace ring1
