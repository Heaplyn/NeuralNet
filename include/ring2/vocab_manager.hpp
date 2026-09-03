#pragma once

/**
 * @file vocab_manager.hpp
 * @brief Multi-layer semantic vocabulary manager in Ring 2.
 * Tracks subwords (Layer 0), parsed words/chunks (Layer 1), and semantic categories /
 * "meaning of meanings" (Layer 2) with deterministic multi-probe hash vectorization.
 */

#include "ring0/tensor.hpp"
#include "ring2/tokenizer.hpp"
#include "ring2/transformer_lm.hpp"
#include <vector>
#include <string>
#include <unordered_map>
#include <memory>

namespace ring2 {

/**
 * @struct SemanticMeaning
 * @brief Semantic representation of a term with a meaning transformation matrix and concept vector.
 */
struct SemanticMeaning {
    std::string term;
    int token_id = -1;
    ring0::Matrix meaning_matrix;       ///< (embed_dim x embed_dim) Bilinear concept meaning matrix
    std::vector<float> concept_vector;  ///< (1 x embed_dim) Normalized semantic representation vector
    std::vector<float> hash_vector;     ///< (1 x embed_dim) Deterministic hash fingerprint vector
    int category_id = 0;                ///< Layer 2 semantic category index
    size_t frequency = 1;
};

/**
 * @struct SemanticCategory
 * @brief Layer 2 Meta-Meaning representing a data-agnostic vector index cluster ("meaning of meanings").
 *        Uses discrete vector coordinates and continuous latent centroids for arbitrary data types.
 */
struct SemanticCategory {
    std::string name;                   ///< Text identifier (e.g. "[VEC_0x04]")
    int category_id = 0;                ///< Integer category identifier
    std::vector<float> vector_index;    ///< Discrete vector coordinate address in latent codebook
    ring0::Matrix meta_meaning_matrix;  ///< (embed_dim x embed_dim) Higher-order category transformation
    std::vector<float> centroid_vector; ///< (1 x embed_dim) Center of semantic mass for this category
    size_t member_count = 0;
};

/**
 * @struct LexiconEntry
 * @brief Layer 1 Parsed word or chunk entry from the words database corpus.
 */
struct LexiconEntry {
    std::string text;
    size_t corpus_frequency = 0;
    float semantic_loss = 0.0f;
    int best_category_id = 0;
    bool is_registered = false;
};

/**
 * @struct VocabLayerMetrics
 * @brief Telemetry metrics tracking the health and convergence of the multi-layer vocabulary.
 */
struct VocabLayerMetrics {
    size_t layer0_subwords = 0;
    size_t layer1_words = 0;
    size_t layer2_categories = 0;
    float category_clustering_loss = 0.0f;
    float hash_alignment_score = 0.0f;
};

/**
 * @class VocabManager
 * @brief Manages hierarchical multi-layer vocabulary, data-agnostic vector indexed categories ("meaning of meanings"),
 *        continuous lexicon loop evaluation alongside training, and multi-probe hash vectorization.
 */
class VocabManager {
public:
    size_t embed_dim;
    size_t num_vector_categories = 16;  ///< Number of data-agnostic vector index clusters
    std::unordered_map<std::string, SemanticMeaning> semantic_registry;
    std::vector<std::string> discovered_terms;

    // Layer 2: Semantic Vector Index Categories ("Meaning of Meanings")
    std::vector<SemanticCategory> categories;

    // Layer 1: Continuous Parsed Corpus Lexicon & Chunk Queue
    std::vector<LexiconEntry> parsed_lexicon;
    size_t current_lexicon_cursor = 0;

    explicit VocabManager(size_t embedding_dimension = 96, size_t num_categories = 16);

    /// Initializes data-agnostic Layer 2 Vector Index Categories across latent unit sphere
    void initialize_categories(size_t num_categories = 16);

    /// Initializes base semantic matrices for all tokens currently in the tokenizer
    void initialize_base_meanings(const Tokenizer& tokenizer);

    /// Multi-probe semantic feature hashing: projects token/chunk into deterministic unit vector in R^D
    std::vector<float> compute_hash_vector(const std::string& text) const;

    /// Parses the entire corpus into Layer 1 unique words and multi-word chunks
    void parse_corpus_lexicon(const std::string& corpus);

    /// Finds closest semantic category for a given concept vector via dot-product
    int classify_category(const std::vector<float>& concept_vec) const;

    /// Registers a new semantic term, synthesizing meaning matrix, hash vector, and category meta-matrix
    int add_semantic_term(const std::string& term, Tokenizer& tokenizer, TransformerLM& model);

    /// Continuous Vocab Loop: Steps through parsed lexicon alongside training loop, running loss & expanding
    size_t step_vocab_evaluation(const std::string& corpus, Tokenizer& tokenizer, TransformerLM& model, float current_step_loss);

    /// Discovers frequent words/chunks in corpus and expands model vocabulary
    size_t scan_and_expand_vocabulary(const std::string& corpus, Tokenizer& tokenizer, TransformerLM& model, size_t max_new_terms = 16);

    /// Computes multi-layer semantic loss and alignment metrics across categories and hash projections
    VocabLayerMetrics evaluate_vocab_layers(const TransformerLM& model) const;

    /// Computes Frobenius inner-product similarity between two terms' meaning matrices in [-1.0, 1.0]
    float compute_meaning_similarity(const std::string& term_a, const std::string& term_b) const;

    /// Retrieves pointer to meaning matrix for a registered term (or nullptr)
    const ring0::Matrix* get_meaning_matrix(const std::string& term) const;
};

} // namespace ring2
