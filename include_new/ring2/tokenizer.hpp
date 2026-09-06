#pragma once

/**
 * @file tokenizer.hpp
 * @brief Trie-based O(N) Subword Byte-Pair Encoding (BPE) Tokenizer with Word-Boundary Pre-segmentation in Ring 2.
 */

#include <string>
#include <vector>
#include <unordered_map>
#include <map>
#include <memory>

using namespace std;

namespace ring2 {

/**
 * @struct BPEMerge
 * @brief Represents a learned pair-merging rule: (token_a + token_b) -> merged_id.
 */
struct BPEMerge {
    int token_a;
    int token_b;
    int merged_id;
};

/**
 * @struct TrieNode
 * @brief Prefix tree node for O(N) single-pass longest-matching tokenization.
 */
struct TrieNode {
    unordered_map<char, unique_ptr<TrieNode>> children;
    int token_id = -1;
};

/**
 * @struct DynamicVocabFactors
 * @brief Telemetry factors used to mathematically compute the optimal vocabulary size.
 */
struct DynamicVocabFactors {
    size_t corpus_bytes = 0;              ///< Total bytes/chars in training corpus
    size_t unique_chars_count = 0;        ///< Unique alphabet/byte character count
    float shannon_entropy = 0.0f;         ///< Information entropy of corpus (bits/char)
    float compression_ratio = 1.0f;       ///< Subword characters-per-token ratio
    size_t optimal_vocab_size = 0;        ///< Computed optimal vocabulary limit (up to 10k+)
    size_t min_vocab_limit = 256;         ///< Lower bound floor
    size_t max_vocab_limit = 10000;       ///< Upper bound cap (10k tokens)
    size_t special_tokens_count = 10;     ///< Reserved special control tokens
};

/**
 * @class Tokenizer
 * @brief High-performance Radix-Trie BPE Tokenizer with Dynamic Factor-Based Token Sizing up to 10,000+ tokens.
 */
class Tokenizer {
public:
    unordered_map<string, int> token_to_id; ///< Map from token string to ID
    unordered_map<int, string> id_to_token; ///< Inverse map from token ID to token string
    vector<BPEMerge> merges;                ///< Ordered list of BPE merge rules
    size_t vocab_size;                     ///< Total active vocabulary size
    unique_ptr<TrieNode> trie_root;         ///< Root of fast lookup prefix tree
    DynamicVocabFactors last_factors;       ///< Metrics and factors calculated during adaptive fitting

    Tokenizer();

    /// Analyzes the corpus and computes optimal token size based on multiple mathematical factors
    DynamicVocabFactors analyze_corpus_factors(const string& corpus, size_t max_vocab_limit = 10000) const;

    /// Automatically fits the BPE vocabulary adjusting its token size based on various factors (entropy, volume, utility) up to 10k
    DynamicVocabFactors fit_adaptive(const string& corpus, size_t max_vocab_limit = 10000);

    /// Scans a text corpus and builds the subword vocabulary via BPE up to target_vocab_size
    void fit(const string& corpus, size_t target_vocab_size = 128);

    /// Builds prefix Trie from current vocabulary for fast O(N) streaming tokenization
    void build_trie();

    /// Encodes a human-readable text string into token IDs via single-pass Trie matching
    vector<int> encode(const string& text) const;

    /// Decodes a sequence of subword token IDs back into full text string
    string decode(const vector<int>& tokens) const;

    size_t get_vocab_size() const { return vocab_size; }
};

} // namespace ring2
