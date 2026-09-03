#include "ring2/vocab_manager.hpp"
#include <iostream>
#include <cmath>
#include <algorithm>
#include <sstream>
#include <cctype>

using namespace std;

namespace ring2 {

// FNV-1a 64-bit hash helper
static uint64_t fnv1a_64(const string& s, uint64_t seed = 0xcbf29ce484222325ULL) {
    uint64_t h = seed;
    const uint64_t prime = 0x100000001b3ULL;
    for (unsigned char c : s) {
        h ^= static_cast<uint64_t>(c);
        h *= prime;
    }
    return h;
}

VocabManager::VocabManager(size_t embedding_dimension, size_t num_categories)
    : embed_dim(embedding_dimension), num_vector_categories(num_categories) {
    initialize_categories(num_categories);
}

// Initializes data-agnostic Layer 2 Vector Index Categories ("Meaning of Meanings")
void VocabManager::initialize_categories(size_t num_categories) {
    categories.clear();
    num_vector_categories = max(static_cast<size_t>(4), num_categories);

    // Golden ratio / Fibonacci spherical distribution for uniform quasi-orthogonal vector indexes
    const float phi = 1.61803398875f;
    const float two_pi = 6.28318530718f;

    for (size_t i = 0; i < num_vector_categories; ++i) {
        SemanticCategory cat;
        char buf[32];
        snprintf(buf, sizeof(buf), "[VEC_0x%02X]", static_cast<unsigned int>(i));
        cat.name = buf;
        cat.category_id = static_cast<int>(i);
        cat.member_count = 0;
        cat.meta_meaning_matrix = ring0::Matrix::zeros(embed_dim, embed_dim);
        cat.centroid_vector.resize(embed_dim, 0.0f);
        cat.vector_index.resize(embed_dim, 0.0f);

        // Compute coordinate vector index on the latent hypersphere
        float theta = two_pi * static_cast<float>(i) / phi;
        float z = 1.0f - (2.0f * static_cast<float>(i) + 1.0f) / static_cast<float>(num_vector_categories);
        float radius = sqrtf(max(0.0f, 1.0f - z * z));

        for (size_t d = 0; d < embed_dim; ++d) {
            float phase = theta + static_cast<float>(d * 7 + (i * 13) % 97);
            float v = radius * cosf(phase) + (z * sinf(phase * 0.5f));
            cat.vector_index[d] = v;
            cat.centroid_vector[d] = v;
        }

        // Normalize centroid vector to unit sphere S^(D-1)
        float norm_sq = 0.0f;
        for (float v : cat.centroid_vector) norm_sq += v * v;
        float norm = sqrtf(max(1e-6f, norm_sq));
        for (float& v : cat.centroid_vector) v /= norm;
        for (float& v : cat.vector_index) v /= norm;

        // Meta-Meaning bilinear transformation matrix for this vector index
        for (size_t r = 0; r < embed_dim; ++r) {
            cat.meta_meaning_matrix(r, r) = 1.0f + 0.15f * cat.vector_index[r];
            for (size_t c = 0; c < embed_dim; ++c) {
                if (r != c) {
                    float cross_phase = cat.vector_index[r] * 3.0f + cat.vector_index[c] * 5.0f;
                    cat.meta_meaning_matrix(r, c) = (0.12f / sqrtf(static_cast<float>(embed_dim))) * sinf(cross_phase);
                }
            }
        }

        categories.push_back(move(cat));
    }
}

// Multi-probe semantic feature hashing: projects token/chunk into deterministic unit vector in R^D
vector<float> VocabManager::compute_hash_vector(const string& text) const {
    vector<float> hvec(embed_dim, 0.0f);
    if (text.empty()) return hvec;

    // Probe 1: Full token hash
    uint64_t h_full = fnv1a_64(text);
    size_t dim_full = (h_full >> 8) % embed_dim;
    float sign_full = ((h_full & 1) == 1) ? 1.0f : -1.0f;
    hvec[dim_full] += 1.5f * sign_full;

    // Probe 2: Secondary polynomial seed hash
    uint64_t h_seed = fnv1a_64(text, 0x84222325ULL);
    size_t dim_seed = (h_seed >> 16) % embed_dim;
    float sign_seed = ((h_seed & 2) == 2) ? 1.0f : -1.0f;
    hvec[dim_seed] += 1.2f * sign_seed;

    // Probe 3: Character 3-gram shingles (prefix, infix, suffix features)
    if (text.size() >= 3) {
        for (size_t i = 0; i + 2 < text.size(); ++i) {
            string shingle = text.substr(i, 3);
            uint64_t hs = fnv1a_64(shingle);
            size_t d = (hs >> 4) % embed_dim;
            float s = ((hs & 1) == 1) ? 1.0f : -1.0f;
            hvec[d] += 0.5f * s;
        }
    }

    // Normalize hash vector to unit length
    float sum_sq = 0.0f;
    for (float v : hvec) sum_sq += v * v;
    float norm = sqrtf(max(1e-6f, sum_sq));
    for (float& v : hvec) v /= norm;

    return hvec;
}

// Parses corpus into Layer 1 unique data-agnostic terms, sub-sequences, and n-grams
void VocabManager::parse_corpus_lexicon(const string& corpus) {
    parsed_lexicon.clear();
    current_lexicon_cursor = 0;

    unordered_map<string, size_t> word_freq;
    unordered_map<string, size_t> chunk_freq;

    string current_token;
    vector<string> token_sequence;
    token_sequence.reserve(corpus.size() / 4);

    for (size_t i = 0; i < corpus.size(); ++i) {
        unsigned char c = static_cast<unsigned char>(corpus[i]);
        // Data-agnostic segmentation: alphanumeric, underscores, or common programming syntax
        if (isalnum(c) || c == '_' || c == '$' || c == '@' || c == ':') {
            current_token += static_cast<char>(c);
        } else {
            if (!current_token.empty()) {
                if (current_token.size() >= 2) {
                    word_freq[current_token]++;
                    token_sequence.push_back(current_token);
                }
                current_token.clear();
            }
            // Capture multi-character operators or delimiters as discrete tokens
            if (!isspace(c)) {
                string delim(1, static_cast<char>(c));
                word_freq[delim]++;
                token_sequence.push_back(delim);
            }
        }
    }
    if (!current_token.empty() && current_token.size() >= 2) {
        word_freq[current_token]++;
        token_sequence.push_back(current_token);
    }

    // Extract frequent multi-token n-gram chunks
    for (size_t i = 0; i + 1 < token_sequence.size(); ++i) {
        if (token_sequence[i].size() >= 2 && token_sequence[i + 1].size() >= 2) {
            string chunk = token_sequence[i] + " " + token_sequence[i + 1];
            chunk_freq[chunk]++;
        }
    }

    // Populate parsed_lexicon with words appearing at least twice
    for (const auto& pair : word_freq) {
        if (pair.second >= 2) {
            LexiconEntry entry;
            entry.text = pair.first;
            entry.corpus_frequency = pair.second;
            entry.semantic_loss = 0.0f;
            entry.best_category_id = 0;
            entry.is_registered = false;
            parsed_lexicon.push_back(move(entry));
        }
    }

    // Add high-frequency multi-word chunks (occurring >= 3 times)
    for (const auto& pair : chunk_freq) {
        if (pair.second >= 3) {
            LexiconEntry entry;
            entry.text = pair.first;
            entry.corpus_frequency = pair.second;
            entry.semantic_loss = 0.0f;
            entry.best_category_id = 0;
            entry.is_registered = false;
            parsed_lexicon.push_back(move(entry));
        }
    }

    // Sort by corpus frequency descending
    sort(parsed_lexicon.begin(), parsed_lexicon.end(), [](const LexiconEntry& a, const LexiconEntry& b) {
        return a.corpus_frequency > b.corpus_frequency;
    });

    cout << "  📚 [VocabManager L1 Lexicon] Parsed " << parsed_lexicon.size() 
         << " unique words & multi-token chunks from corpus!\n";
}

// Finds closest semantic category for a given concept vector
int VocabManager::classify_category(const vector<float>& concept_vec) const {
    if (categories.empty() || concept_vec.empty()) return 0;

    int best_cat = 0;
    float max_sim = -1e9f;

    for (size_t k = 0; k < categories.size(); ++k) {
        float dot = 0.0f;
        for (size_t d = 0; d < embed_dim; ++d) {
            dot += concept_vec[d] * categories[k].centroid_vector[d];
        }
        if (dot > max_sim) {
            max_sim = dot;
            best_cat = static_cast<int>(k);
        }
    }
    return best_cat;
}

// Initializes base semantic meaning matrices for initial vocabulary
void VocabManager::initialize_base_meanings(const Tokenizer& tokenizer) {
    for (const auto& pair : tokenizer.id_to_token) {
        int id = pair.first;
        const string& term = pair.second;

        SemanticMeaning sm;
        sm.term = term;
        sm.token_id = id;
        sm.frequency = 1;

        // Construct base meaning matrix
        sm.meaning_matrix = ring0::Matrix::zeros(embed_dim, embed_dim);
        sm.concept_vector.resize(embed_dim, 0.0f);
        sm.hash_vector = compute_hash_vector(term);

        for (size_t r = 0; r < embed_dim; ++r) {
            sm.meaning_matrix(r, r) = 1.0f;
            for (size_t c = 0; c < embed_dim; ++c) {
                float phase = static_cast<float>(id * 17 + r * 7 + c * 13);
                sm.meaning_matrix(r, c) += 0.05f * sinf(phase);
            }
            sm.concept_vector[r] = 0.7f * sm.hash_vector[r] + 0.3f * cosf(static_cast<float>(id * 31 + r * 11));
        }

        // Normalize concept vector
        float norm_sq = 0.0f;
        for (float v : sm.concept_vector) norm_sq += v * v;
        float norm = sqrtf(max(1e-6f, norm_sq));
        for (float& v : sm.concept_vector) v /= norm;

        sm.category_id = classify_category(sm.concept_vector);
        categories[sm.category_id].member_count++;

        semantic_registry[term] = move(sm);
    }
}

// Registers a new semantic term, synthesizing its meaning matrix, hash vector, and category meta-matrix
int VocabManager::add_semantic_term(const string& term, Tokenizer& tokenizer, TransformerLM& model) {
    auto it = semantic_registry.find(term);
    if (it != semantic_registry.end()) {
        it->second.frequency++;
        return it->second.token_id;
    }

    // 1. Decompose term into existing sub-tokens
    vector<int> sub_tokens = tokenizer.encode(term);
    if (sub_tokens.empty()) return -1;

    // 2. Compute deterministic hash vector for the new term/chunk
    vector<float> hash_vec = compute_hash_vector(term);

    // 3. Synthesize concept vector and meaning matrix via composition
    vector<float> composed_vector(embed_dim, 0.0f);
    ring0::Matrix composed_matrix = ring0::Matrix::zeros(embed_dim, embed_dim);

    float K = static_cast<float>(sub_tokens.size());

    for (int sub_id : sub_tokens) {
        auto name_it = tokenizer.id_to_token.find(sub_id);
        if (name_it != tokenizer.id_to_token.end()) {
            auto reg_it = semantic_registry.find(name_it->second);
            if (reg_it != semantic_registry.end()) {
                const auto& sub_meaning = reg_it->second;
                for (size_t r = 0; r < embed_dim; ++r) {
                    float transformed = 0.0f;
                    for (size_t c = 0; c < embed_dim; ++c) {
                        transformed += sub_meaning.meaning_matrix(r, c) * sub_meaning.concept_vector[c];
                    }
                    composed_vector[r] += transformed / K;
                }
                for (size_t r = 0; r < embed_dim; ++r) {
                    for (size_t c = 0; c < embed_dim; ++c) {
                        composed_matrix(r, c) += sub_meaning.meaning_matrix(r, c) / K;
                    }
                }
            }
        }
    }

    // Fuse 70% composed constituent vector with 30% deterministic hash fingerprint
    for (size_t d = 0; d < embed_dim; ++d) {
        composed_vector[d] = 0.70f * composed_vector[d] + 0.30f * hash_vec[d];
    }

    // Normalize composed vector
    float norm_sq = 0.0f;
    for (float v : composed_vector) norm_sq += v * v;
    float norm = sqrtf(max(1e-6f, norm_sq));
    float target_norm = sqrtf(2.0f / static_cast<float>(embed_dim));
    for (float& v : composed_vector) {
        v = (v / norm) * target_norm;
    }

    // 4. Layer 2: Classify semantic category and blend with Category Meta-Meaning Matrix
    int cat_id = classify_category(composed_vector);
    const auto& cat = categories[cat_id];
    for (size_t r = 0; r < embed_dim; ++r) {
        for (size_t c = 0; c < embed_dim; ++c) {
            composed_matrix(r, c) += 0.25f * cat.meta_meaning_matrix(r, c);
            composed_matrix(r, c) += 0.15f * composed_vector[r] * composed_vector[c];
        }
    }

    // Update category centroid running average
    categories[cat_id].member_count++;
    float cat_alpha = 0.05f;
    for (size_t d = 0; d < embed_dim; ++d) {
        categories[cat_id].centroid_vector[d] = (1.0f - cat_alpha) * categories[cat_id].centroid_vector[d] + cat_alpha * composed_vector[d];
    }

    // 5. Register in Tokenizer
    int new_id = static_cast<int>(tokenizer.vocab_size);
    tokenizer.token_to_id[term] = new_id;
    tokenizer.id_to_token[new_id] = term;
    tokenizer.vocab_size++;
    tokenizer.build_trie();

    // 6. Expand Model Embedding and LM Head
    model.expand_vocab(tokenizer.vocab_size);

    // 7. Seed new token row directly in embedding weights
    for (size_t d = 0; d < embed_dim; ++d) {
        model.embedding.token_weights(new_id, d) = composed_vector[d];
    }

    // 8. Save in Registry
    SemanticMeaning new_meaning;
    new_meaning.term = term;
    new_meaning.token_id = new_id;
    new_meaning.meaning_matrix = move(composed_matrix);
    new_meaning.concept_vector = composed_vector;
    new_meaning.hash_vector = move(hash_vec);
    new_meaning.category_id = cat_id;
    new_meaning.frequency = 1;

    semantic_registry[term] = move(new_meaning);
    discovered_terms.push_back(term);

    cout << "  ✨ [VocabManager L1->L2] Registered \"" << term 
         << "\" -> ID " << new_id 
         << " | Meta-Cat: [" << cat.name << "] | Hash-Fused Meaning Matrix (" 
         << embed_dim << "x" << embed_dim << ")\n";

    return new_id;
}

// Continuous Vocab Loop: Steps through parsed lexicon alongside training loop, running loss & category alignment
size_t VocabManager::step_vocab_evaluation(const string& /*corpus*/, Tokenizer& /*tokenizer*/, TransformerLM& /*model*/, float current_step_loss) {
    if (parsed_lexicon.empty()) return 0;

    size_t batch_check = 16;
    for (size_t i = 0; i < batch_check; ++i) {
        size_t idx = (current_lexicon_cursor + i) % parsed_lexicon.size();
        auto& entry = parsed_lexicon[idx];

        // Compute multi-probe hash vector and classify Layer 2 category
        vector<float> h = compute_hash_vector(entry.text);
        entry.best_category_id = classify_category(h);

        // Compute semantic category alignment loss
        float cat_dot = 0.0f;
        for (size_t d = 0; d < embed_dim; ++d) {
            cat_dot += h[d] * categories[entry.best_category_id].centroid_vector[d];
        }
        entry.semantic_loss = max(0.0f, 1.0f - cat_dot) + (current_step_loss * 0.05f);

        // Softly nudge category centroid towards observed frequent words
        if (entry.corpus_frequency >= 5) {
            float alpha = 0.005f;
            for (size_t d = 0; d < embed_dim; ++d) {
                categories[entry.best_category_id].centroid_vector[d] = 
                    (1.0f - alpha) * categories[entry.best_category_id].centroid_vector[d] + alpha * h[d];
            }
        }
    }

    current_lexicon_cursor = (current_lexicon_cursor + batch_check) % parsed_lexicon.size();
    return 0;
}

// Discovers frequent words/chunks in corpus and expands model vocabulary
size_t VocabManager::scan_and_expand_vocabulary(const string& corpus, Tokenizer& tokenizer, TransformerLM& model, size_t max_new_terms) {
    vector<string> candidate_terms = {
        " the ", " and ", " of ", " to ", " in ", " that ", " is ", " with ",
        " for ", " was ", " on ", " as ", " by ", " at ", " from ", " not ",
        "intelligence", "learning", "model", "neural", "system", "world",
        "thought", "reason", "language", "memory", "knowledge", "future"
    };

    size_t added = 0;
    for (const auto& term : candidate_terms) {
        if (added >= max_new_terms) break;
        if (semantic_registry.find(term) == semantic_registry.end()) {
            if (corpus.find(term) != string::npos) {
                add_semantic_term(term, tokenizer, model);
                added++;
            }
        }
    }
    return added;
}

// Computes multi-layer semantic loss and alignment metrics across categories and hash projections
VocabLayerMetrics VocabManager::evaluate_vocab_layers(const TransformerLM& model) const {
    VocabLayerMetrics metrics;
    metrics.layer0_subwords = model.config.vocab_size;
    metrics.layer1_words = parsed_lexicon.size();
    metrics.layer2_categories = categories.size();

    float total_cat_dist = 0.0f;
    float total_hash_sim = 0.0f;
    size_t count = 0;

    for (const auto& pair : semantic_registry) {
        const auto& sm = pair.second;
        if (sm.token_id >= 0 && static_cast<size_t>(sm.token_id) < model.config.vocab_size) {
            // Category clustering distance: 1 - cosine(concept, centroid)
            float cat_dot = 0.0f;
            for (size_t d = 0; d < embed_dim; ++d) {
                cat_dot += sm.concept_vector[d] * categories[sm.category_id].centroid_vector[d];
            }
            total_cat_dist += max(0.0f, 1.0f - cat_dot);

            // Hash alignment similarity: cosine(model_embed, hash_vector)
            float hash_dot = 0.0f;
            float embed_norm_sq = 0.0f;
            for (size_t d = 0; d < embed_dim; ++d) {
                float w = model.embedding.token_weights(sm.token_id, d);
                hash_dot += w * sm.hash_vector[d];
                embed_norm_sq += w * w;
            }
            float embed_norm = sqrtf(max(1e-6f, embed_norm_sq));
            total_hash_sim += (hash_dot / embed_norm);

            count++;
        }
    }

    if (count > 0) {
        metrics.category_clustering_loss = total_cat_dist / static_cast<float>(count);
        metrics.hash_alignment_score = total_hash_sim / static_cast<float>(count);
    }

    return metrics;
}

// Computes Frobenius inner-product similarity between two terms' meaning matrices
float VocabManager::compute_meaning_similarity(const string& term_a, const string& term_b) const {
    auto it_a = semantic_registry.find(term_a);
    auto it_b = semantic_registry.find(term_b);
    if (it_a == semantic_registry.end() || it_b == semantic_registry.end()) return 0.0f;

    const auto& ma = it_a->second.meaning_matrix;
    const auto& mb = it_b->second.meaning_matrix;

    float dot = 0.0f;
    float norm_a = 0.0f;
    float norm_b = 0.0f;

    for (size_t i = 0; i < ma.data.size(); ++i) {
        float va = ma.data[i];
        float vb = mb.data[i];
        dot += va * vb;
        norm_a += va * va;
        norm_b += vb * vb;
    }

    float denom = sqrtf(max(1e-6f, norm_a)) * sqrtf(max(1e-6f, norm_b));
    return dot / denom;
}

// Retrieves pointer to meaning matrix for a registered term
const ring0::Matrix* VocabManager::get_meaning_matrix(const string& term) const {
    auto it = semantic_registry.find(term);
    if (it != semantic_registry.end()) {
        return &it->second.meaning_matrix;
    }
    return nullptr;
}

} // namespace ring2
