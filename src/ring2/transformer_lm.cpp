#include "ring2/transformer_lm.hpp"
#include "ring2/tokenizer.hpp"
#include "ring0/loss.hpp"
#include <iostream>
#include <random>
#include <cmath>
#include <algorithm>
#include <fstream>
#include <unordered_map>
#include <filesystem>
#include <iomanip>
#include <sstream>

using namespace std;

namespace ring2
{

    // Global deterministic generator RNG engine for reproducible text generation
    static mt19937 &get_gen_rng()
    {
        static mt19937 rng(1337);
        return rng;
    }

    // Helper for sampling a token index using Min-P, Top-P (Nucleus), Top-K, and Repetition Penalty
    static int sample_token_from_logits(const ring0::Matrix &logits_row,
                                        size_t vocab_size,
                                        float temperature,
                                        size_t top_k,
                                        float top_p,
                                        float min_p,
                                        float repetition_penalty,
                                        const vector<int> &context_tokens)
    {

        // 1. Track recently generated tokens to apply repetition penalty
        unordered_map<int, int> recent_counts;
        size_t lookback = min(context_tokens.size(), static_cast<size_t>(32));
        for (size_t i = context_tokens.size() - lookback; i < context_tokens.size(); ++i)
        {
            recent_counts[context_tokens[i]]++;
        }

        vector<float> scaled_logits(vocab_size);
        float max_val = -1e9f;

        for (size_t v = 0; v < vocab_size; ++v)
        {
            float val = logits_row(0, v);
            // Apply repetition penalty
            if (repetition_penalty > 1.0f && recent_counts.find(static_cast<int>(v)) != recent_counts.end())
            {
                if (val > 0.0f)
                {
                    val /= repetition_penalty;
                }
                else
                {
                    val *= repetition_penalty;
                }
            }
            val /= max(0.01f, temperature);
            scaled_logits[v] = val;
            if (val > max_val)
                max_val = val;
        }

        // 2. Softmax normalization
        float sum_exp = 0.0f;
        vector<pair<float, int>> token_probs;
        token_probs.reserve(vocab_size);

        for (size_t v = 0; v < vocab_size; ++v)
        {
            float p = exp(scaled_logits[v] - max_val);
            token_probs.emplace_back(p, static_cast<int>(v));
            sum_exp += p;
        }

        for (auto &tp : token_probs)
        {
            tp.first /= max(1e-8f, sum_exp);
        }

        // Sort descending by probability
        sort(token_probs.begin(), token_probs.end(), [](const auto &a, const auto &b)
             { return a.first > b.first; });

        float p_max = token_probs.empty() ? 1.0f : token_probs[0].first;

        // 3. Min-P Filtering: discard tokens with p < min_p * p_max
        if (min_p > 0.0f && p_max > 0.0f)
        {
            float min_thresh = min_p * p_max;
            vector<pair<float, int>> filtered;
            for (const auto &tp : token_probs)
            {
                if (tp.first >= min_thresh)
                {
                    filtered.push_back(tp);
                }
            }
            if (!filtered.empty())
            {
                token_probs = move(filtered);
            }
        }

        // 4. Top-K Filtering
        if (top_k > 0 && top_k < token_probs.size())
        {
            token_probs.resize(top_k);
        }

        // 5. Top-P (Nucleus) Filtering: accumulate probability mass up to top_p
        if (top_p > 0.0f && top_p < 1.0f && token_probs.size() > 1)
        {
            float cum_p = 0.0f;
            size_t cutoff = 0;
            for (size_t i = 0; i < token_probs.size(); ++i)
            {
                cum_p += token_probs[i].first;
                cutoff = i + 1;
                if (cum_p >= top_p)
                    break;
            }
            token_probs.resize(max(size_t(1), cutoff));
        }

        // 6. Re-normalize surviving candidate probabilities
        float total_p = 0.0f;
        for (const auto &tp : token_probs)
        {
            total_p += tp.first;
        }
        for (auto &tp : token_probs)
        {
            tp.first /= max(1e-8f, total_p);
        }

        // 7. Stochastic sampling from surviving distribution
        uniform_real_distribution<float> dist(0.0f, 1.0f);
        float r = dist(get_gen_rng());
        float accum = 0.0f;
        int next_token = token_probs[0].second;

        for (const auto &tp : token_probs)
        {
            accum += tp.first;
            if (r <= accum)
            {
                next_token = tp.second;
                break;
            }
        }

        return next_token;
    }

    // Constructor: Builds embeddings, transformer blocks with GQA and SwiGLU, RMSNorm, and LM Head
    TransformerLM::TransformerLM(TransformerConfig cfg)
        : config(cfg),
          embedding(cfg.vocab_size, cfg.max_seq_len, cfg.embed_dim),
          type_attention(cfg.embed_dim, cfg.type_dim)
    {

        for (size_t i = 0; i < cfg.num_layers; ++i)
        {
            blocks.emplace_back(cfg.embed_dim, cfg.num_heads, cfg.ffn_dim, cfg.num_kv_heads);
        }

        ln_f_gamma = ring0::Matrix::ones(1, cfg.embed_dim);
        ln_f_beta = ring0::Matrix::zeros(1, cfg.embed_dim);

        float scale = sqrt(2.0f / static_cast<float>(cfg.embed_dim) / ring0::Loss::get_latest_loss() / .5f);
        W_head = ring0::Matrix::random_normal(cfg.embed_dim, cfg.vocab_size, 0.0f, scale);
        b_head = ring0::Matrix::zeros(1, cfg.vocab_size);

        reset_gradients();
    }

    // Resets gradient accumulators across all sublayers to zero
    void TransformerLM::reset_gradients()
    {
        embedding.reset_gradients();
        for (auto &b : blocks)
        {
            b.reset_gradients();
        }
        if (config.enable_coc_type_attention)
        {
            type_attention.reset_gradients();
        }
        grad_ln_f_gamma = ring0::Matrix::zeros(1, config.embed_dim);
        grad_ln_f_beta = ring0::Matrix::zeros(1, config.embed_dim);
        grad_W_head = ring0::Matrix::zeros(config.embed_dim, config.vocab_size);
        grad_b_head = ring0::Matrix::zeros(1, config.vocab_size);
    }

    // Saves rolling in-memory snapshot of healthy parameters
    void TransformerLM::save_safe_snapshot()
    {
        safe_snapshot.token_weights = embedding.token_weights;
        safe_snapshot.pos_weights = embedding.pos_weights;
        safe_snapshot.block_weights.clear();

        for (const auto &b : blocks)
        {
            safe_snapshot.block_weights.push_back(b.attention.W_q);
            safe_snapshot.block_weights.push_back(b.attention.b_q);
            safe_snapshot.block_weights.push_back(b.attention.W_k);
            safe_snapshot.block_weights.push_back(b.attention.b_k);
            safe_snapshot.block_weights.push_back(b.attention.W_v);
            safe_snapshot.block_weights.push_back(b.attention.b_v);
            safe_snapshot.block_weights.push_back(b.attention.W_o);
            safe_snapshot.block_weights.push_back(b.attention.b_o);

            safe_snapshot.block_weights.push_back(b.ln1_gamma);
            safe_snapshot.block_weights.push_back(b.ln1_beta);

            safe_snapshot.block_weights.push_back(b.W_gate);
            safe_snapshot.block_weights.push_back(b.b_gate);
            safe_snapshot.block_weights.push_back(b.W_up);
            safe_snapshot.block_weights.push_back(b.b_up);
            safe_snapshot.block_weights.push_back(b.W_down);
            safe_snapshot.block_weights.push_back(b.b_down);

            safe_snapshot.block_weights.push_back(b.ln2_gamma);
            safe_snapshot.block_weights.push_back(b.ln2_beta);
        }

        safe_snapshot.ln_f_gamma = ln_f_gamma;
        safe_snapshot.ln_f_beta = ln_f_beta;
        safe_snapshot.b_head = b_head;
        safe_snapshot.valid = true;
    }

    // Restores model weights to the last known healthy snapshot
    bool TransformerLM::restore_safe_snapshot()
    {
        if (!safe_snapshot.valid)
            return false;

        embedding.token_weights = safe_snapshot.token_weights;
        embedding.pos_weights = safe_snapshot.pos_weights;

        size_t mat_idx = 0;
        for (auto &b : blocks)
        {
            if (mat_idx + 18 > safe_snapshot.block_weights.size())
                break;

            b.attention.W_q = safe_snapshot.block_weights[mat_idx++];
            b.attention.b_q = safe_snapshot.block_weights[mat_idx++];
            b.attention.W_k = safe_snapshot.block_weights[mat_idx++];
            b.attention.b_k = safe_snapshot.block_weights[mat_idx++];
            b.attention.W_v = safe_snapshot.block_weights[mat_idx++];
            b.attention.b_v = safe_snapshot.block_weights[mat_idx++];
            b.attention.W_o = safe_snapshot.block_weights[mat_idx++];
            b.attention.b_o = safe_snapshot.block_weights[mat_idx++];

            b.ln1_gamma = safe_snapshot.block_weights[mat_idx++];
            b.ln1_beta = safe_snapshot.block_weights[mat_idx++];

            b.W_gate = safe_snapshot.block_weights[mat_idx++];
            b.b_gate = safe_snapshot.block_weights[mat_idx++];
            b.W_up = safe_snapshot.block_weights[mat_idx++];
            b.b_up = safe_snapshot.block_weights[mat_idx++];
            b.W_down = safe_snapshot.block_weights[mat_idx++];
            b.b_down = safe_snapshot.block_weights[mat_idx++];

            b.ln2_gamma = safe_snapshot.block_weights[mat_idx++];
            b.ln2_beta = safe_snapshot.block_weights[mat_idx++];
        }

        ln_f_gamma = safe_snapshot.ln_f_gamma;
        ln_f_beta = safe_snapshot.ln_f_beta;
        b_head = safe_snapshot.b_head;

        reset_gradients();
        return true;
    }

    // Seeds the LM-head bias with log-unigram frequencies for a fast loss start.
    void TransformerLM::init_head_bias_from_unigram(const vector<int> &token_stream)
    {
        const size_t V = config.vocab_size;
        if (V == 0 || token_stream.empty())
            return;

        // 1. Count token occurrences with Laplace (add-1) smoothing
        vector<double> counts(V, 1.0);
        double total = static_cast<double>(V);
        for (int tok : token_stream)
        {
            if (tok >= 0 && static_cast<size_t>(tok) < V)
            {
                counts[static_cast<size_t>(tok)] += 1.0;
                total += 1.0;
            }
        }

        // 2. b_head[c] = log P(c) - mean(log P)
        double mean_log = 0.0;
        vector<float> logp(V);
        for (size_t c = 0; c < V; ++c)
        {
            double p = counts[c] / total;
            float lp = static_cast<float>(std::log(p));
            logp[c] = lp;
            mean_log += lp;
        }
        mean_log /= static_cast<double>(V); // Proper mean!

        for (size_t c = 0; c < V; ++c)
        {
            b_head.data[c] = logp[c] - static_cast<float>(mean_log);
        }

        // Report the theoretical step-0 loss this bias yields (the unigram entropy).
        double entropy = 0.0;
        for (size_t c = 0; c < V; ++c)
        {
            double p = counts[c] / total;
            entropy -= p * std::log(p);
        }
        cout << "  >> [Fast-Start Init] LM-head bias seeded from log-unigram frequencies. "
             << "Expected step-0 loss ~ " << fixed << setprecision(3) << entropy
             << " (vs ln(V) = " << std::log(static_cast<double>(V)) << ").\n";
    }

    // Dynamically expands hidden SwiGLU feature dimension across all layers as loss drops
    void TransformerLM::expand_capacity(size_t additional_ffn_dim)
    {
        if (additional_ffn_dim == 0)
            return;

        cout << "\n[Neurogenesis Event] Loss dropped! Expanding SwiGLU FFN hidden dimension: "
             << config.ffn_dim << " -> " << (config.ffn_dim * additional_ffn_dim) << "...\n";

        for (auto &block : blocks)
        {
            block.expand_ffn_dim(additional_ffn_dim);
        }
        config.ffn_dim *= additional_ffn_dim;
        reset_gradients();

        cout << "  >> Model capacity expanded successfully! Total params now: " << get_total_parameters() << "\n\n";
    }

    // Sets the active number of layers for progressive depth ramping
    void TransformerLM::set_active_layers(size_t active_layers)
    {
        num_active_layers = max(size_t(1), min(active_layers, blocks.size()));
    }

    // Forward pass for full context (Training phase with GQA + SwiGLU + Pre-RMSNorm)
    ring0::Matrix TransformerLM::forward(const vector<int> &token_ids, size_t batch_size, size_t seq_len)
    {
        // 1. Embedding lookup
        ring0::Tensor3D hidden = embedding.forward(token_ids, batch_size, seq_len);

        // 2. Pass through active Transformer Blocks (Progressive Depth Ramping)
        size_t active_n = min(num_active_layers, blocks.size());
        for (size_t i = 0; i < active_n; ++i)
        {
            hidden = blocks[i].forward(hidden);
        }
        last_blocks_output = hidden;

        // 3. Final RMSNorm
        ring0::Matrix h_mat = hidden.to_matrix();
        ring0::Matrix ln_f_mat = h_mat.rms_norm(ln_f_gamma);
        last_ln_f = ring0::Tensor3D::from_matrix(ln_f_mat, batch_size, seq_len);

        // 4. Project through Language Model Head via tied embeddings with zero-alloc pre-transposed matmul
        ring0::Matrix logits_proj(ln_f_mat.rows, embedding.token_weights.rows, 0.0f);
        ln_f_mat.matmul_transB_into(embedding.token_weights, logits_proj);
        last_logits = logits_proj.add_bias(b_head);
        return last_logits;
    }

    // O(1) single-token forward step utilizing ModelKVCache and GQA
    ring0::Matrix TransformerLM::forward_step(int token_id, size_t pos_idx, ModelKVCache &cache)
    {
        // 1. Token & Positional embedding lookup for single token (shape: 1 x embed_dim)
        ring0::Matrix h(1, config.embed_dim);
        for (size_t d = 0; d < config.embed_dim; ++d)
        {
            float tok_val = (token_id >= 0 && static_cast<size_t>(token_id) < config.vocab_size)
                                ? embedding.token_weights(token_id, d)
                                : 0.0f;
            float pos_val = (pos_idx < config.max_seq_len)
                                ? embedding.pos_weights(pos_idx, d)
                                : 0.0f;
            h(0, d) = tok_val + pos_val;
        }

        // 2. Pass through active stacked decoder blocks with GQA KV-Cache
        size_t active_n = min(num_active_layers, blocks.size());
        for (size_t i = 0; i < active_n; ++i)
        {
            h = blocks[i].forward_step(h, pos_idx, cache.layer_caches[i]);
        }

        // 3. Final RMSNorm
        ring0::Matrix ln_f = h.rms_norm(ln_f_gamma);

        // 4. Project to vocabulary logits via tied embedding weights
        ring0::Matrix logits_proj(ln_f.rows, embedding.token_weights.rows, 0.0f);
        ln_f.matmul_transB_into(embedding.token_weights, logits_proj);
        return logits_proj.add_bias(b_head);
    }

    // Backward pass: Propagates loss gradients backward through all layers with analytical RMSNorm, SwiGLU, & GQA
    void TransformerLM::backward(const ring0::Matrix &grad_logits)
    {
        size_t B = last_ln_f.batch_size;
        size_t T = last_ln_f.seq_len;

        // 1. LM Head gradients (tied to embedding.token_weights)
        ring0::Matrix ln_f_mat = last_ln_f.to_matrix();
        embedding.grad_token_weights += grad_logits.transpose().matmul(ln_f_mat);
        grad_b_head += grad_logits.sum_rows();

        ring0::Matrix dLn_f_mat = grad_logits.matmul(embedding.token_weights);

        // Exact analytical RMSNorm final backward
        ring0::Matrix dHidden_mat = last_blocks_output.to_matrix().rms_norm_backward(dLn_f_mat, ln_f_gamma, grad_ln_f_gamma);

        ring0::Tensor3D dHidden = ring0::Tensor3D::from_matrix(dHidden_mat, B, T);

        // 2. Transformer active blocks backward pass
        int active_n = static_cast<int>(min(num_active_layers, blocks.size()));
        for (int i = active_n - 1; i >= 0; --i)
        {
            dHidden = blocks[i].backward(dHidden);
        }

        // 3. Embedding backward pass
        embedding.backward(dHidden);
    }

    // Applies AdamW optimizer updates to all trainable matrices including SwiGLU & GQA
    void TransformerLM::update_parameters(ring1::AdamW &optimizer)
    {
        size_t param_idx = 0;

        auto update_mat = [&](ring0::Matrix &param, const ring0::Matrix &grad, bool decay)
        {
            if (optimizer.m_list.size() <= param_idx)
            {
                optimizer.register_param(param, decay);
            }
            optimizer.update_param(param_idx++, param, grad);
        };
        auto update_w = [&](ring0::Matrix &p, const ring0::Matrix &g)
        { update_mat(p, g, true); };
        auto update_b = [&](ring0::Matrix &p, const ring0::Matrix &g)
        { update_mat(p, g, false); };

        update_w(embedding.token_weights, embedding.grad_token_weights);
        update_w(embedding.pos_weights, embedding.grad_pos_weights);

        size_t active_n = min(num_active_layers, blocks.size());
        for (size_t i = 0; i < active_n; ++i)
        {
            auto &b = blocks[i];
            update_w(b.attention.W_q, b.attention.grad_W_q);
            update_b(b.attention.b_q, b.attention.grad_b_q);
            update_w(b.attention.W_k, b.attention.grad_W_k);
            update_b(b.attention.b_k, b.attention.grad_b_k);
            update_w(b.attention.W_v, b.attention.grad_W_v);
            update_b(b.attention.b_v, b.attention.grad_b_v);
            update_w(b.attention.W_o, b.attention.grad_W_o);
            update_b(b.attention.b_o, b.attention.grad_b_o);

            update_b(b.ln1_gamma, b.grad_ln1_gamma);
            update_b(b.ln1_beta, b.grad_ln1_beta);

            // SwiGLU parameters
            update_w(b.W_gate, b.grad_W_gate);
            update_b(b.b_gate, b.grad_b_gate);
            update_w(b.W_up, b.grad_W_up);
            update_b(b.b_up, b.grad_b_up);
            update_w(b.W_down, b.grad_W_down);
            update_b(b.b_down, b.grad_b_down);

            update_b(b.ln2_gamma, b.grad_ln2_gamma);
            update_b(b.ln2_beta, b.grad_ln2_beta);
        }

        if (config.enable_coc_type_attention)
        {
            update_w(type_attention.W_type_q, type_attention.grad_W_type_q);
            update_w(type_attention.W_type_k, type_attention.grad_W_type_k);
        }

        update_b(ln_f_gamma, grad_ln_f_gamma);
        update_b(ln_f_beta, grad_ln_f_beta);
        update_b(b_head, grad_b_head);

        optimizer.timestep++;
        reset_gradients();
    }

    // Rescales parameter gradients so global L2 norm is <= max_norm
    float TransformerLM::clip_grad_norm(float max_norm)
    {
        vector<ring0::Matrix *> grads;
        grads.push_back(&embedding.grad_token_weights);
        grads.push_back(&embedding.grad_pos_weights);

        if (config.enable_coc_type_attention)
        {
            grads.push_back(&type_attention.grad_W_type_q);
            grads.push_back(&type_attention.grad_W_type_k);
        }

        size_t active_n = min(num_active_layers, blocks.size());
        for (size_t i = 0; i < active_n; ++i)
        {
            auto &b = blocks[i];
            grads.push_back(&b.attention.grad_W_q);
            grads.push_back(&b.attention.grad_b_q);
            grads.push_back(&b.attention.grad_W_k);
            grads.push_back(&b.attention.grad_b_k);
            grads.push_back(&b.attention.grad_W_v);
            grads.push_back(&b.attention.grad_b_v);
            grads.push_back(&b.attention.grad_W_o);
            grads.push_back(&b.attention.grad_b_o);
            grads.push_back(&b.grad_ln1_gamma);
            grads.push_back(&b.grad_ln1_beta);

            // SwiGLU grads
            grads.push_back(&b.grad_W_gate);
            grads.push_back(&b.grad_b_gate);
            grads.push_back(&b.grad_W_up);
            grads.push_back(&b.grad_b_up);
            grads.push_back(&b.grad_W_down);
            grads.push_back(&b.grad_b_down);

            grads.push_back(&b.grad_ln2_gamma);
            grads.push_back(&b.grad_ln2_beta);
        }
        grads.push_back(&grad_ln_f_gamma);
        grads.push_back(&grad_ln_f_beta);
        grads.push_back(&grad_b_head);

        float sum_sq = 0.0f;
        bool found_nan_inf = false;
        for (auto *g : grads)
        {
            for (float &v : g->data)
            {
                if (std::isnan(v) || std::isinf(v))
                {
                    v = 0.0f;
                    found_nan_inf = true;
                }
                else
                {
                    sum_sq += v * v;
                }
            }
        }
        if (found_nan_inf || std::isnan(sum_sq) || std::isinf(sum_sq))
        {
            sum_sq = 0.0f;
        }

        float norm = sqrt(sum_sq);
        if (max_norm > 0.0f && norm > max_norm)
        {
            float scale = max_norm / (norm + 1e-6f);
            for (auto *g : grads)
            {
                for (float &v : g->data)
                    v *= scale;
            }
        }
        return norm;
    }

    // Dynamically expands vocabulary size for embeddings and LM Head as steps progress
    void TransformerLM::expand_vocab(size_t new_vocab_size)
    {
        if (new_vocab_size <= config.vocab_size)
            return;
        size_t old_v = config.vocab_size;
        config.vocab_size = new_vocab_size;

        cout << "\n[Progressive Vocab Event] Unlocking vocabulary tier: " << old_v << " -> " << new_vocab_size << " subwords...\n";

        // 1. Expand input token embedding table
        embedding.expand_vocab(new_vocab_size);

        // 2. Expand LM Head projection matrix (cols) and bias vector
        float scale = sqrt(2.0f / static_cast<float>(config.embed_dim));
        W_head.expand_cols(new_vocab_size, scale);
        b_head.expand_cols(new_vocab_size, 0.0f);

        grad_W_head = ring0::Matrix::zeros(config.embed_dim, new_vocab_size);
        grad_b_head = ring0::Matrix::zeros(1, new_vocab_size);

        reset_gradients();
        cout << "  >> Vocabulary expanded successfully! Total params now: " << get_total_parameters() << "\n\n";
    }

    // Dynamically expands maximum sequence capacity across embeddings up to 2048
    void TransformerLM::expand_max_seq_len(size_t new_max_seq)
    {
        if (new_max_seq <= config.max_seq_len)
            return;
        config.max_seq_len = new_max_seq;
        embedding.expand_max_seq_len(new_max_seq);
        reset_gradients();
    }

    // Autoregressive generation with Min-P, Top-P, Top-K, Repetition Penalty, and KV-Caching
    vector<int> TransformerLM::generate(const vector<int> &prompt_tokens,
                                        size_t max_new_tokens,
                                        float temperature,
                                        size_t top_k,
                                        float top_p,
                                        float min_p,
                                        float repetition_penalty,
                                        const function<void(int)> &on_token_generated,
                                        bool use_kv_cache)
    {
        vector<int> current_tokens = prompt_tokens;
        if (current_tokens.empty())
        {
            current_tokens.push_back(0);
        }

        if (use_kv_cache)
        {
            // --- Fast O(1) Path using ModelKVCache ---
            ModelKVCache cache(blocks.size());

            // 1. Prefill Phase: Process initial prompt tokens into the KV-Cache
            ring0::Matrix last_logits(1, config.vocab_size);
            for (size_t i = 0; i < current_tokens.size(); ++i)
            {
                size_t pos = min(i, config.max_seq_len - 1);
                last_logits = forward_step(current_tokens[i], pos, cache);
            }

            // 2. Autoregressive Decoding Phase with Min-P & Top-P Sampling
            for (size_t step = 0; step < max_new_tokens; ++step)
            {
                int next_token = sample_token_from_logits(last_logits, config.vocab_size, temperature, top_k, top_p, min_p, repetition_penalty, current_tokens);
                current_tokens.push_back(next_token);

                if (on_token_generated)
                {
                    on_token_generated(next_token);
                }

                size_t next_pos = min(current_tokens.size() - 1, config.max_seq_len - 1);
                last_logits = forward_step(next_token, next_pos, cache);
            }
        }
        else
        {
            // --- Uncached O(T^2) Baseline Path ---
            for (size_t step = 0; step < max_new_tokens; ++step)
            {
                size_t seq_len = min(current_tokens.size(), config.max_seq_len);
                vector<int> context(current_tokens.end() - seq_len, current_tokens.end());

                ring0::Matrix logits = forward(context, 1, seq_len);
                size_t last_row = seq_len - 1;

                ring0::Matrix last_row_mat(1, config.vocab_size);
                for (size_t v = 0; v < config.vocab_size; ++v)
                {
                    last_row_mat(0, v) = logits(last_row, v);
                }

                int next_token = sample_token_from_logits(last_row_mat, config.vocab_size, temperature, top_k, top_p, min_p, repetition_penalty, current_tokens);
                current_tokens.push_back(next_token);

                if (on_token_generated)
                {
                    on_token_generated(next_token);
                }
            }
        }

        return current_tokens;
    }

    // Saves full model weights, recorded loss, and architecture to binary file
    bool TransformerLM::save_checkpoint(const string &filepath, float loss) const
    {
        ofstream out(filepath, ios::binary);
        if (!out.is_open())
            return false;

        uint32_t magic = 0x4C4C4D34; // "LLM4" (Includes loss header)
        out.write(reinterpret_cast<const char *>(&magic), sizeof(magic));
        out.write(reinterpret_cast<const char *>(&loss), sizeof(loss));
        out.write(reinterpret_cast<const char *>(&config), sizeof(config));

        auto write_mat = [&](const ring0::Matrix &m)
        {
            uint64_t r = m.rows;
            uint64_t c = m.cols;
            out.write(reinterpret_cast<const char *>(&r), sizeof(r));
            out.write(reinterpret_cast<const char *>(&c), sizeof(c));
            if (!m.data.empty())
            {
                out.write(reinterpret_cast<const char *>(m.data.data()), m.data.size() * sizeof(float));
            }
        };

        write_mat(embedding.token_weights);
        write_mat(embedding.pos_weights);

        for (const auto &b : blocks)
        {
            write_mat(b.attention.W_q);
            write_mat(b.attention.b_q);
            write_mat(b.attention.W_k);
            write_mat(b.attention.b_k);
            write_mat(b.attention.W_v);
            write_mat(b.attention.b_v);
            write_mat(b.attention.W_o);
            write_mat(b.attention.b_o);

            write_mat(b.ln1_gamma);
            write_mat(b.ln1_beta);

            // SwiGLU weights
            write_mat(b.W_gate);
            write_mat(b.b_gate);
            write_mat(b.W_up);
            write_mat(b.b_up);
            write_mat(b.W_down);
            write_mat(b.b_down);

            write_mat(b.ln2_gamma);
            write_mat(b.ln2_beta);
        }

        write_mat(ln_f_gamma);
        write_mat(ln_f_beta);
        write_mat(b_head);

        return true;
    }

    // Inspects binary checkpoint header returning loss and configuration without full weight load
    bool TransformerLM::inspect_checkpoint(const string &filepath, float &out_loss, TransformerConfig &out_cfg)
    {
        ifstream in(filepath, ios::binary);
        if (!in.is_open())
            return false;

        uint32_t magic = 0;
        in.read(reinterpret_cast<char *>(&magic), sizeof(magic));
        if (magic == 0x4C4C4D34)
        {
            in.read(reinterpret_cast<char *>(&out_loss), sizeof(out_loss));
            in.read(reinterpret_cast<char *>(&out_cfg), sizeof(out_cfg));
            return true;
        }
        else if (magic == 0x4C4C4D33)
        {
            out_loss = 999.0f;
            in.read(reinterpret_cast<char *>(&out_cfg), sizeof(out_cfg));
            return true;
        }
        return false;
    }

    // Loads model weights from binary file
    bool TransformerLM::load_checkpoint(const string &filepath)
    {
        ifstream in(filepath, ios::binary);
        if (!in.is_open())
            return false;

        uint32_t magic = 0;
        in.read(reinterpret_cast<char *>(&magic), sizeof(magic));
        if (magic != 0x4C4C4D34 && magic != 0x4C4C4D33)
            return false;

        float saved_loss = 0.0f;
        if (magic == 0x4C4C4D34)
        {
            in.read(reinterpret_cast<char *>(&saved_loss), sizeof(saved_loss));
        }

        TransformerConfig saved_cfg;
        in.read(reinterpret_cast<char *>(&saved_cfg), sizeof(saved_cfg));

        if (saved_cfg.vocab_size != config.vocab_size ||
            saved_cfg.embed_dim != config.embed_dim ||
            saved_cfg.num_layers != config.num_layers ||
            saved_cfg.num_heads != config.num_heads ||
            saved_cfg.num_kv_heads != config.num_kv_heads ||
            saved_cfg.ffn_dim != config.ffn_dim)
        {
            return false; // Shape mismatch
        }

        auto read_mat = [&](ring0::Matrix &m)
        {
            uint64_t r = 0, c = 0;
            in.read(reinterpret_cast<char *>(&r), sizeof(r));
            in.read(reinterpret_cast<char *>(&c), sizeof(c));
            m = ring0::Matrix(r, c);
            if (!m.data.empty())
            {
                in.read(reinterpret_cast<char *>(m.data.data()), m.data.size() * sizeof(float));
            }
        };

        read_mat(embedding.token_weights);
        read_mat(embedding.pos_weights);

        for (auto &b : blocks)
        {
            read_mat(b.attention.W_q);
            read_mat(b.attention.b_q);
            read_mat(b.attention.W_k);
            read_mat(b.attention.b_k);
            read_mat(b.attention.W_v);
            read_mat(b.attention.b_v);
            read_mat(b.attention.W_o);
            read_mat(b.attention.b_o);

            read_mat(b.ln1_gamma);
            read_mat(b.ln1_beta);

            // SwiGLU weights
            read_mat(b.W_gate);
            read_mat(b.b_gate);
            read_mat(b.W_up);
            read_mat(b.b_up);
            read_mat(b.W_down);
            read_mat(b.b_down);

            read_mat(b.ln2_gamma);
            read_mat(b.ln2_beta);
        }

        read_mat(ln_f_gamma);
        read_mat(ln_f_beta);
        read_mat(b_head);

        reset_gradients();
        return true;
    }

    // Conditionally loads checkpoint ONLY if its saved loss is around current loss or less
    bool TransformerLM::load_checkpoint_if_better(const string &filepath, float current_eval_loss, float tolerance_multiplier)
    {
        float saved_loss = 0.0f;
        TransformerConfig saved_cfg;
        if (!inspect_checkpoint(filepath, saved_loss, saved_cfg))
        {
            return false;
        }

        if (saved_cfg.vocab_size != config.vocab_size ||
            saved_cfg.embed_dim != config.embed_dim ||
            saved_cfg.num_layers != config.num_layers ||
            saved_cfg.num_heads != config.num_heads ||
            saved_cfg.num_kv_heads != config.num_kv_heads ||
            saved_cfg.ffn_dim != config.ffn_dim)
        {
            cout << "  >> [Checkpoint Reference] Skipped " << filepath << " (Architecture mismatch).\n";
            return false;
        }

        float max_acceptable_loss = current_eval_loss * tolerance_multiplier;
        if (saved_loss <= max_acceptable_loss)
        {
            cout << "  >> [Checkpoint Reference] Saved loss (" << fixed << setprecision(3) << saved_loss
                 << ") <= current baseline (" << fixed << setprecision(3) << current_eval_loss
                 << ")! Adopting saved reference weights from " << filepath << "...\n";
            return load_checkpoint(filepath);
        }
        else
        {
            cout << "  >> [Checkpoint Reference] Saved loss (" << fixed << setprecision(3) << saved_loss
                 << ") is worse than current baseline (" << fixed << setprecision(3) << current_eval_loss
                 << "). Retaining current weights!\n";
            return false;
        }
    }

    // Scans directory for best saved checkpoint and adopts it if loss is <= current_eval_loss
    bool TransformerLM::load_best_checkpoint_from_dir(const string &checkpoints_dir, float current_eval_loss)
    {
        if (!filesystem::exists(checkpoints_dir))
            return false;

        string best_file = "";
        float lowest_loss = 1e9f;

        for (const auto &entry : filesystem::recursive_directory_iterator(checkpoints_dir))
        {
            if (entry.is_regular_file() && entry.path().extension() == ".bin")
            {
                float chk_loss = 0.0f;
                TransformerConfig chk_cfg;
                if (inspect_checkpoint(entry.path().string(), chk_loss, chk_cfg))
                {
                    if (chk_loss < lowest_loss &&
                        chk_cfg.vocab_size == config.vocab_size &&
                        chk_cfg.embed_dim == config.embed_dim)
                    {
                        lowest_loss = chk_loss;
                        best_file = entry.path().string();
                    }
                }
            }
        }

        if (!best_file.empty())
        {
            return load_checkpoint_if_better(best_file, current_eval_loss);
        }
        return false;
    }

    // Calculates total trainable parameters
    size_t TransformerLM::get_total_parameters() const
    {
        size_t total = 0;
        total += embedding.token_weights.data.size();
        total += embedding.pos_weights.data.size();

        for (const auto &b : blocks)
        {
            total += b.attention.W_q.data.size() + b.attention.b_q.data.size();
            total += b.attention.W_k.data.size() + b.attention.b_k.data.size();
            total += b.attention.W_v.data.size() + b.attention.b_v.data.size();
            total += b.attention.W_o.data.size() + b.attention.b_o.data.size();

            total += b.ln1_gamma.data.size() + b.ln1_beta.data.size();
            total += b.W_gate.data.size() + b.b_gate.data.size();
            total += b.W_up.data.size() + b.b_up.data.size();
            total += b.W_down.data.size() + b.b_down.data.size();
            total += b.ln2_gamma.data.size() + b.ln2_beta.data.size();
        }

        total += ln_f_gamma.data.size() + ln_f_beta.data.size();
        total += b_head.data.size();

        return total;
    }

    // Saves a complete multi-file checkpoint bundle into a dedicated directory
    bool TransformerLM::save_checkpoint_bundle(const string &folder_path,
                                               size_t step,
                                               float loss,
                                               float top1_acc,
                                               float rank_score,
                                               const Tokenizer &tokenizer,
                                               const string &sample_text) const
    {
        try
        {
            filesystem::create_directories(folder_path);

            // 1. Save binary weights with recorded loss
            string weights_path = folder_path + "/model_weights.bin";
            if (!save_checkpoint(weights_path, loss))
                return false;

            // 2. Save metadata text report
            string meta_path = folder_path + "/metadata.txt";
            ofstream meta_out(meta_path);
            if (meta_out.is_open())
            {
                meta_out << "===========================================\n"
                         << "   RINGWRAPPER LLM CHECKPOINT METADATA     \n"
                         << "===========================================\n"
                         << "Training Step:       " << step << "\n"
                         << "Focal Loss:          " << fixed << setprecision(4) << loss << "\n"
                         << "Top-1 Accuracy:      " << fixed << setprecision(2) << top1_acc << "%\n"
                         << "Top-20 Rank Score:   " << fixed << setprecision(2) << rank_score << "%\n"
                         << "Perplexity (PPL):    " << fixed << setprecision(2) << exp(min(loss, 10.0f)) << "\n"
                         << "Total Parameters:    " << get_total_parameters() << "\n\n"
                         << "--- Architecture Parameters ---\n"
                         << "Vocab Size:          " << config.vocab_size << "\n"
                         << "Max Context Len:     " << config.max_seq_len << "\n"
                         << "Embedding Dimension: " << config.embed_dim << "\n"
                         << "Query Heads (Q):     " << config.num_heads << "\n"
                         << "KV Heads (GQA):      " << config.num_kv_heads << "\n"
                         << "Stacked Layers:      " << config.num_layers << "\n"
                         << "SwiGLU FFN Hidden:   " << config.ffn_dim << "\n"
                         << "Positional Bias:     ALiBi Geometric Slope Active\n";
                meta_out.close();
            }

            // 3. Save tokenizer vocabulary and merge rules
            string vocab_path = folder_path + "/vocab.txt";
            ofstream vocab_out(vocab_path);
            if (vocab_out.is_open())
            {
                vocab_out << "=== ACTIVE VOCABULARY TOKENS (" << tokenizer.id_to_token.size() << ") ===\n";
                for (size_t id = 0; id < tokenizer.vocab_size; ++id)
                {
                    auto it = tokenizer.id_to_token.find(static_cast<int>(id));
                    if (it != tokenizer.id_to_token.end())
                    {
                        vocab_out << id << "\t[" << it->second << "]\n";
                    }
                }
                vocab_out << "\n=== LEARNED BPE MERGES (" << tokenizer.merges.size() << ") ===\n";
                for (size_t i = 0; i < tokenizer.merges.size(); ++i)
                {
                    const auto &m = tokenizer.merges[i];
                    vocab_out << i << "\t(" << m.token_a << " + " << m.token_b << ") -> " << m.merged_id << "\n";
                }
                vocab_out.close();
            }

            // 4. Save sample text generation
            string sample_path = folder_path + "/sample_generation.txt";
            ofstream sample_out(sample_path);
            if (sample_out.is_open())
            {
                sample_out << "=== GENERATION SNAPSHOT AT MILESTONE (Step " << step
                           << " | Loss " << fixed << setprecision(4) << loss << ") ===\n\n"
                           << sample_text << "\n";
                sample_out.close();
            }

            return true;
        }
        catch (...)
        {
            return false;
        }
    }

    // Auto-purges checkpoints in a directory whose recorded loss is above (min_loss * threshold_multiplier)
    size_t TransformerLM::purge_stale_checkpoints(const string &checkpoints_dir, float min_loss, float threshold_multiplier)
    {
        if (min_loss <= 0.0f || !filesystem::exists(checkpoints_dir))
            return 0;
        float max_allowed_loss = min_loss * threshold_multiplier;
        size_t purged_count = 0;

        try
        {
            for (const auto &entry : filesystem::directory_iterator(checkpoints_dir))
            {
                string candidate_bin;
                bool is_bundle_dir = false;

                if (entry.is_directory())
                {
                    candidate_bin = entry.path().string() + "/model_weights.bin";
                    is_bundle_dir = true;
                }
                else if (entry.is_regular_file() && entry.path().extension() == ".bin")
                {
                    candidate_bin = entry.path().string();
                    is_bundle_dir = false;
                }

                if (!candidate_bin.empty() && filesystem::exists(candidate_bin))
                {
                    float chk_loss = 0.0f;
                    TransformerConfig chk_cfg;
                    if (inspect_checkpoint(candidate_bin, chk_loss, chk_cfg))
                    {
                        if (chk_loss > max_allowed_loss)
                        {
                            // Stale checkpoint exceeds threshold relative to min_loss
                            if (is_bundle_dir)
                            {
                                filesystem::remove_all(entry.path());
                            }
                            else
                            {
                                filesystem::remove(entry.path());
                            }
                            cout << "  🗑️  [Checkpoint Purge] Removed stale checkpoint: "
                                 << entry.path().filename().string()
                                 << " (Loss: " << fixed << setprecision(3) << chk_loss
                                 << " > Threshold: " << setprecision(3) << max_allowed_loss << " ["
                                 << fixed << setprecision(2) << min_loss << " * " << threshold_multiplier << "])\n";
                            purged_count++;
                        }
                    }
                }
            }
        }
        catch (...)
        {
            // Suppress filesystem iteration errors
        }

        return purged_count;
    }

    // Prints architecture
    void TransformerLM::print_architecture() const
    {
        cout << "--- Transformer Causal LLM Architecture ---\n"
             << "  Vocab Size:     " << config.vocab_size << "\n"
             << "  Max Seq Len:    " << config.max_seq_len << "\n"
             << "  Embedding Dim:  " << config.embed_dim << "\n"
             << "  Query Heads:    " << config.num_heads << " (Head Dim: " << (config.embed_dim / config.num_heads) << ")\n"
             << "  KV Heads (GQA): " << config.num_kv_heads << " (Group Size: " << (config.num_heads / config.num_kv_heads) << ")\n"
             << "  Num Layers:     " << config.num_layers << "\n"
             << "  FFN Hidden Dim: " << config.ffn_dim << " (SwiGLU Gated Activation with Dynamic Growth)\n"
             << "  Attention Bias: ALiBi Distance Falloff Active\n"
             << "  KV-Cache Engine: Active (GQA Accelerated)\n"
             << "  Total Model Params: " << get_total_parameters() << "\n"
             << "-------------------------------------------\n";
    }

} // namespace ring2
