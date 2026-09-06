#pragma once

/**
 * @file text_dataset.hpp
 * @brief Next-token prediction dataset and batch generator for Transformer LLMs in Ring 4.
 */

#include "ring2/tokenizer.hpp"
#include <vector>
#include <string>

using namespace std;

namespace ring4 {

/**
 * @struct TextBatch
 * @brief Container for a batch of context tokens (input_ids) and target tokens (target_ids).
 */
struct TextBatch {
    vector<int> input_ids;  ///< Flattened matrix of input tokens (B * T)
    vector<int> target_ids; ///< Flattened matrix of ground-truth target tokens (B * T)
    size_t batch_size;      ///< Number of sequences (B)
    size_t seq_len;         ///< Sequence length (T)
};

/**
 * @class TextDataset
 * @brief Encodes continuous text into tokens and provides randomized/sequential training batches.
 */
class TextDataset {
public:
    vector<int> token_stream; ///< Full tokenized stream of the entire text corpus
    vector<bool> is_prompt_mask; ///< True for prompt comment tokens (masked out during SFT)
    size_t seq_len;           ///< Sequence context length (T)
    bool sft_masking_enabled = true; ///< When true, masks prompt comments with target = -1

    size_t active_tokens_limit = 0; ///< When > 0, restricts batch sampling to the first N tokens of the corpus

    /**
     * @brief Encodes text using the tokenizer and initializes sequence length.
     */
    TextDataset(const string& text, const ring2::Tokenizer& tokenizer, size_t seq_length);

    /// Total non-overlapping batches available
    size_t get_num_batches(size_t batch_size) const;

    /// Extracts sequential batch at batch_idx
    TextBatch get_batch(size_t batch_idx, size_t batch_size) const;

    /// Samples a randomized batch from anywhere in the token stream using default seq_len
    TextBatch get_random_batch(size_t batch_size) const;

    /// Samples a randomized batch from anywhere in the token stream using dynamic sequence length
    TextBatch get_random_batch(size_t batch_size, size_t dynamic_seq_len) const;

    /// Feature 6: Auto-Learning Data Filter (Calculus-based Information Entropy & Gradient Surprise)
    TextBatch get_information_filtered_batch(size_t batch_size, size_t dynamic_seq_len, size_t candidate_pool_multiplier = 3) const;

    /**
     * @struct TokenContextWindow
     * @brief Parsed context tokens surrounding a focal anchor token with interpolated relevancy weights.
     */
    struct TokenContextWindow {
        int anchor_token = 0;
        size_t anchor_position = 0;
        float anchor_relevance = 0.0f;
        size_t parsed_window_radius = 0;
        vector<int> tokens;
        vector<float> interpolated_relevancies; ///< Distance-decayed relevancy value for each token in window
    };

    /// Computes intrinsic informational/semantic relevancy score for a token in [0.0, 1.0]
    float compute_token_relevance(size_t token_pos) const;

    /// Computes interpolated context window radius based on token relevancy
    size_t get_interpolated_window_radius(float relevance) const;

    /// Extracts a context window surrounding a token with interpolated relevancy scores for each neighbor
    TokenContextWindow extract_relevance_parsed_window(size_t token_pos) const;

    /// Generates a training batch centered on high-relevance anchor tokens with interpolated context parsing
    TextBatch get_token_relevance_batch(size_t batch_size, size_t dynamic_seq_len) const;

    /// Sets active subset of dataset as a fraction in (0.0, 1.0]
    void set_active_ratio(float ratio);

    /// Sets active subset of dataset to explicit token count
    void set_active_tokens(size_t tokens);

    /// Retrieves current active token count available for sampling
    size_t get_active_tokens() const;

    /// Ingests and appends another text file (.txt, .md) to the active token stream
    size_t append_text_file(const string& filepath, const ring2::Tokenizer& tokenizer);

    /// Ingests and appends a binary file (.bin: raw text or token ids) to the active token stream
    size_t append_binary_file(const string& filepath, const ring2::Tokenizer& tokenizer);

    /// Directly appends a vector of pre-encoded token IDs (used for background streaming ingestion)
    size_t append_tokens(const vector<int>& new_tokens, const vector<bool>& prompt_mask = {});
};

} // namespace ring4
