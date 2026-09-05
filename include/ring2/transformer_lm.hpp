#pragma once

/**
 * @file transformer_lm.hpp
 * @brief Causal Transformer Large Language Model with GQA, Min-P/Top-P Adaptive Sampling, Dynamic Capacity Expansion, and Loss-Conditional Checkpoint Loading in Ring 2.
 */

#include "ring0/tensor.hpp"
#include "ring1/embedding.hpp"
#include "ring1/transformer_block.hpp"
#include "ring1/dependent_type_attention.hpp"
#include "ring1/adamw.hpp"
#include <vector>
#include <string>
#include <functional>

using namespace std;

namespace ring2
{

    // Forward declaration of Tokenizer
    class Tokenizer;

    /**
     * @struct ModelKVCache
     * @brief Holds layer-specific Key and Value caches for all Transformer decoder blocks.
     */
    struct ModelKVCache
    {
        vector<ring1::LayerKVCache> layer_caches;

        explicit ModelKVCache(size_t num_layers = 0) : layer_caches(num_layers) {}

        /// Clears all layer caches
        void clear()
        {
            for (auto &lc : layer_caches)
            {
                lc.clear();
            }
        }
    };

    /**
     * @struct TransformerConfig
     * @brief Architectural hyperparameters defining the Transformer LLM.
     */
    struct TransformerConfig
    {
        size_t vocab_size = 1024; ///< Number of distinct tokens in vocabulary (V)
        size_t max_seq_len = 256; ///< Maximum context length / sequence capacity (T)
        size_t embed_dim = 96;    ///< Token & positional embedding dimension (C)
        size_t num_heads = 8;     ///< Number of Query attention heads (N_q)
        size_t num_kv_heads = 4;  ///< Number of Key/Value attention heads (N_kv for GQA)
        size_t num_layers = 5;    ///< Number of stacked Transformer decoder blocks (N)
        size_t ffn_dim = 192;     ///< SwiGLU expansion hidden dimension
        bool enable_coc_type_attention = true; ///< Constructive type-directed attention prior
        size_t type_dim = 64;     ///< Dependent type embedding dimension
    };

    /**
     * @class TransformerLM
     * @brief Complete Causal Transformer Large Language Model with GQA, Min-P/Top-P Adaptive Sampling, Dynamic Growth, and Calculus of Constructions Type Attention.
     */
    class TransformerLM
    {
    public:
        TransformerConfig config;

        ring1::EmbeddingLayer embedding;        ///< Token & positional embedding layer
        vector<ring1::TransformerBlock> blocks; ///< Stack of Transformer decoder blocks
        ring1::DependentTypeAttention type_attention; ///< Calculus of Constructions dependent type attention engine

        // --- Final RMSNorm (ln_f) ---
        ring0::Matrix ln_f_gamma, ln_f_beta;
        ring0::Matrix grad_ln_f_gamma, grad_ln_f_beta;

        // --- Language Model Head (LM Head) ---
        ring0::Matrix W_head, b_head; ///< Projects embed_dim -> vocab_size
        ring0::Matrix grad_W_head, grad_b_head;

        // --- Backpropagation Cache ---
        ring0::Tensor3D last_blocks_output;
        ring0::Tensor3D last_ln_f;
        ring0::Matrix last_logits;
        ring0::Tensor3D last_type_compat;

        // --- Rolling Safe Snapshot for Instant Weight Rollback Recovery ---
        struct ModelSnapshot
        {
            bool valid = false;
            ring0::Matrix token_weights;
            ring0::Matrix pos_weights;
            vector<ring0::Matrix> block_weights;
            ring0::Matrix ln_f_gamma, ln_f_beta, b_head;
        };

        ModelSnapshot safe_snapshot;

        /// Saves rolling copy of healthy model parameters
        void save_safe_snapshot();

        /// Restores model weights to the last known healthy snapshot
        bool restore_safe_snapshot();

        explicit TransformerLM(TransformerConfig cfg = {});

        /// Computes logits matrix (shape: B*T x vocab_size) from input token IDs for training
        ring0::Matrix forward(const vector<int> &token_ids, size_t batch_size, size_t seq_len);

        /**
         * @brief Computes logits (1 x vocab_size) for a single token using KV-Cache.
         */
        ring0::Matrix forward_step(int token_id, size_t pos_idx, ModelKVCache &cache);

        /// Full backward gradient propagation
        void backward(const ring0::Matrix &grad_logits);

        /// Resets gradients across all layers
        void reset_gradients();

        /// Applies AdamW parameter updates to all weights and biases
        void update_parameters(ring1::AdamW &optimizer);

        /// Rescales all parameter gradients so their global L2 norm is <= max_norm.
        float clip_grad_norm(float max_norm);

        size_t num_active_layers = 10; ///< Number of currently active blocks for progressive depth ramping

        /// Dynamically adjusts the number of active layers (progressive depth ramping with soft residual scaling)
        void set_active_layers(size_t active_layers);

        /// Computes a fast, compact fingerprint of model state (norms of key matrices) for mistake detection
        std::vector<float> compute_lightweight_fingerprint() const;

        /**
         * @brief Seeds the LM-head output bias with the log-unigram token frequencies.
         *
         * At initialization the head bias is zero, so step-0 loss equals ln(V) (the
         * uniform-guess floor, ~9.2 for a 10k vocab). Setting b_head[c] = log P(c)
         * makes the model output the marginal token distribution on step 0, which
         * instantly lowers the loss to the unigram entropy H(p) (~6-7 for text) and
         * greatly accelerates early descent: the network then only has to learn the
         * CONTEXTUAL deviation from the marginal, not the marginal itself.
         * @param token_stream Full corpus token stream (unigram counts are taken from it).
         */
        void init_head_bias_from_unigram(const vector<int> &token_stream);

        /// Dynamically expands hidden SwiGLU feature dimension across all layers as loss drops
        void expand_capacity(size_t additional_ffn_dim);

        /// Dynamically expands vocabulary size for embeddings and LM Head as steps progress
        void expand_vocab(size_t new_vocab_size);

        /// Dynamically expands maximum sequence capacity across embeddings up to 2048
        void expand_max_seq_len(size_t new_max_seq);

        /**
         * @brief Autoregressively generates new tokens with Min-P, Top-P, Top-K, Repetition Penalty, and KV-Caching.
         */
        vector<int> generate(const vector<int> &prompt_tokens,
                             size_t max_new_tokens,
                             float temperature = 0.8f,
                             size_t top_k = 50,
                             float top_p = 0.90f,
                             float min_p = 0.05f,
                             float repetition_penalty = 1.2f,
                             const function<void(int)> &on_token_generated = nullptr,
                             bool use_kv_cache = true);

        /// Saves full model weights, recorded loss, and architecture to a binary checkpoint file
        bool save_checkpoint(const string &filepath, float loss = 0.0f) const;

        /// Loads model weights from a binary checkpoint file
        bool load_checkpoint(const string &filepath);

        /// Inspects a binary checkpoint header returning saved loss and config without full weight load
        static bool inspect_checkpoint(const string &filepath, float &out_loss, TransformerConfig &out_cfg);

        /**
         * @brief Conditionally loads checkpoint ONLY if its saved loss is around the current baseline loss or less.
         * @param filepath Path to the checkpoint file.
         * @param current_eval_loss Current evaluated loss of the active network.
         * @param tolerance_multiplier Margin allowing loading if saved_loss <= current_eval_loss * tolerance_multiplier.
         * @return true if checkpoint had lower/comparable loss and was adopted, false otherwise.
         */
        bool load_checkpoint_if_better(const string &filepath, float current_eval_loss, float tolerance_multiplier = 1.05f);

        /**
         * @brief Scans directories for saved checkpoints and loads the best one if its loss is <= current_eval_loss.
         */
        bool load_best_checkpoint_from_dir(const string &checkpoints_dir, float current_eval_loss);

        /**
         * @brief Saves a complete multi-file checkpoint bundle into a dedicated milestone directory.
         */
        bool save_checkpoint_bundle(const string &folder_path,
                                    size_t step,
                                    float loss,
                                    float top1_acc,
                                    float rank_score,
                                    const Tokenizer &tokenizer,
                                    const string &sample_text) const;

        /**
         * @brief Auto-purges checkpoints in a directory whose recorded loss is above (min_loss * threshold_multiplier).
         * @param checkpoints_dir Directory containing saved checkpoint files and bundle folders.
         * @param min_loss The best (lowest) loss observed so far.
         * @param threshold_multiplier Multiplier threshold (e.g. 1.15f). Any checkpoint with loss > min_loss * 1.15 is purged.
         * @return Number of purged checkpoint files/bundles.
         */
        static size_t purge_stale_checkpoints(const string &checkpoints_dir, float min_loss, float threshold_multiplier = 1.15f);

        /// Total number of trainable weights and biases
        size_t get_total_parameters() const;

        /// Prints layer-by-layer architectural summary
        void print_architecture() const;
    };

} // namespace ring2
