#include "ring2/tokenizer.hpp"
#include <set>
#include <iostream>
#include <algorithm>
#include <cstdint>
#include <cmath>

using namespace std;

namespace ring2 {

Tokenizer::Tokenizer() : vocab_size(0), trie_root(make_unique<TrieNode>()) {}

// Builds prefix Trie from active vocabulary dictionary for O(N) streaming tokenization
void Tokenizer::build_trie() {
    trie_root = make_unique<TrieNode>();

    for (const auto& kv : token_to_id) {
        const string& tok_str = kv.first;
        int tok_id = kv.second;

        TrieNode* curr = trie_root.get();
        for (char c : tok_str) {
            if (curr->children.find(c) == curr->children.end()) {
                curr->children[c] = make_unique<TrieNode>();
            }
            curr = curr->children[c].get();
        }
        curr->token_id = tok_id;
    }
}

// Analyzes the corpus and computes optimal token size based on multiple mathematical factors
DynamicVocabFactors Tokenizer::analyze_corpus_factors(const string& corpus, size_t max_vocab_limit) const {
    DynamicVocabFactors factors;
    factors.corpus_bytes = corpus.size();
    factors.max_vocab_limit = max_vocab_limit;

    if (corpus.empty()) {
        factors.optimal_vocab_size = factors.min_vocab_limit;
        return factors;
    }

    // 1. Character frequency and alphabet analysis
    unordered_map<char, size_t> char_counts;
    for (char c : corpus) {
        char_counts[c]++;
    }
    factors.unique_chars_count = char_counts.size();

    // 2. Shannon Information Entropy: H = -sum(p * log2(p))
    float total_chars = static_cast<float>(corpus.size());
    float entropy = 0.0f;
    for (const auto& kv : char_counts) {
        float p = static_cast<float>(kv.second) / total_chars;
        if (p > 0.0f) {
            entropy -= p * (logf(p) / logf(2.0f));
        }
    }
    factors.shannon_entropy = entropy;

    // 3. Mathematical Factor Scaling for Optimal Token Capacity
    // Curve: S-curve saturation based on volume, modulated by entropy diversity
    float volume_ratio = 1.0f - expf(-static_cast<float>(corpus.size()) / 120000.0f);
    float entropy_factor = std::clamp(entropy / 4.2f, 0.70f, 1.30f);
    
    float raw_vocab = static_cast<float>(factors.min_vocab_limit) + 
                      static_cast<float>(max_vocab_limit - factors.min_vocab_limit) * volume_ratio * entropy_factor;

    factors.optimal_vocab_size = std::clamp<size_t>(
        static_cast<size_t>(raw_vocab),
        factors.min_vocab_limit,
        max_vocab_limit
    );

    return factors;
}

// Automatically fits the BPE vocabulary adjusting its token size based on various factors up to 10k
DynamicVocabFactors Tokenizer::fit_adaptive(const string& corpus, size_t max_vocab_limit) {
    token_to_id.clear();
    id_to_token.clear();
    merges.clear();
    vocab_size = 0;

    DynamicVocabFactors factors = analyze_corpus_factors(corpus, max_vocab_limit);

    if (corpus.empty()) {
        last_factors = factors;
        return factors;
    }

    // 1. Reserve Special Control Tokens
    vector<string> special_tokens = {
        "<pad>", "<unk>", "<bos>", "<eos>", "<cls>", "<sep>", "<mask>", "[NUM]", "[CODE]", "[CSV_ROW]"
    };
    for (const auto& sp : special_tokens) {
        int id = static_cast<int>(vocab_size++);
        token_to_id[sp] = id;
        id_to_token[id] = sp;
    }
    factors.special_tokens_count = special_tokens.size();

    // 2. Base Character Vocabulary from full corpus
    set<char> unique_chars;
    for (char c : corpus) {
        unique_chars.insert(c);
    }

    for (char c : unique_chars) {
        string s(1, c);
        if (token_to_id.find(s) == token_to_id.end()) {
            int id = static_cast<int>(vocab_size++);
            token_to_id[s] = id;
            id_to_token[id] = s;
        }
    }

    // 3. Convert sample slice (up to 400k characters) to token IDs for high-speed multi-pair BPE extraction
    size_t sample_len = min<size_t>(corpus.size(), 400000);
    string sample = corpus.substr(0, sample_len);

    vector<int> stream;
    stream.reserve(sample.size());
    for (char c : sample) {
        string s(1, c);
        stream.push_back(token_to_id[s]);
    }

    // 4. Multi-Pair Accelerated BPE Merge Extraction up to factors.optimal_vocab_size
    size_t target_v = factors.optimal_vocab_size;

    while (vocab_size < target_v && stream.size() >= 2) {
        // Count pair frequencies
        unordered_map<uint64_t, int> pair_freqs;
        pair_freqs.reserve(stream.size() / 2);
        for (size_t i = 0; i < stream.size() - 1; ++i) {
            uint64_t key = (static_cast<uint64_t>(stream[i]) << 32) | (static_cast<uint64_t>(stream[i + 1]) & 0xFFFFFFFFULL);
            pair_freqs[key]++;
        }

        if (pair_freqs.empty()) break;

        // Extract top candidate pairs for batch merging in this pass
        vector<pair<uint64_t, int>> sorted_pairs(pair_freqs.begin(), pair_freqs.end());
        size_t batch_size = min<size_t>(sorted_pairs.size(), min<size_t>(32, target_v - vocab_size));
        
        std::partial_sort(
            sorted_pairs.begin(), 
            sorted_pairs.begin() + batch_size, 
            sorted_pairs.end(),
            [](const auto& a, const auto& b) { return a.second > b.second; }
        );

        if (sorted_pairs.front().second < 2) {
            // Factor-based early stopping: merge utility below threshold
            break;
        }

        // Map pair -> new token ID
        unordered_map<uint64_t, int> active_merges;
        for (size_t k = 0; k < batch_size; ++k) {
            if (sorted_pairs[k].second < 2 || vocab_size >= target_v) break;

            uint64_t key = sorted_pairs[k].first;
            int first_id = static_cast<int>(key >> 32);
            int second_id = static_cast<int>(key & 0xFFFFFFFFULL);

            if (id_to_token.find(first_id) == id_to_token.end() || id_to_token.find(second_id) == id_to_token.end()) {
                continue;
            }

            string merged_str = id_to_token[first_id] + id_to_token[second_id];
            if (token_to_id.find(merged_str) != token_to_id.end()) {
                continue;
            }

            int new_id = static_cast<int>(vocab_size++);
            token_to_id[merged_str] = new_id;
            id_to_token[new_id] = merged_str;
            merges.push_back(BPEMerge{first_id, second_id, new_id});
            active_merges[key] = new_id;
        }

        if (active_merges.empty()) break;

        // Apply batch substitutions across the stream
        vector<int> new_stream;
        new_stream.reserve(stream.size());
        for (size_t i = 0; i < stream.size(); ++i) {
            if (i < stream.size() - 1) {
                uint64_t key = (static_cast<uint64_t>(stream[i]) << 32) | (static_cast<uint64_t>(stream[i + 1]) & 0xFFFFFFFFULL);
                auto it = active_merges.find(key);
                if (it != active_merges.end()) {
                    new_stream.push_back(it->second);
                    i++;
                    continue;
                }
            }
            new_stream.push_back(stream[i]);
        }
        stream = move(new_stream);
    }

    // 5. Pre-compile Trie and calculate final compression ratio
    build_trie();

    size_t sample_tokens = stream.size();
    factors.compression_ratio = (sample_tokens > 0) ? (static_cast<float>(sample.size()) / static_cast<float>(sample_tokens)) : 1.0f;
    factors.optimal_vocab_size = vocab_size;
    last_factors = factors;

    return factors;
}

// Fits the tokenizer on a corpus using Byte-Pair Encoding (BPE)
void Tokenizer::fit(const string& corpus, size_t target_vocab_size) {
    fit_adaptive(corpus, target_vocab_size);
}

// Encodes raw text into subword token IDs using single-pass Trie matching (O(N) runtime)
vector<int> Tokenizer::encode(const string& text) const {
    if (text.empty()) return {};
    if (!trie_root) return {};

    vector<int> tokens;
    tokens.reserve(text.size());

    size_t i = 0;
    while (i < text.size()) {
        const TrieNode* curr = trie_root.get();
        size_t match_len = 0;
        int best_token = -1;

        // Longest matching subword token prefix search
        for (size_t j = i; j < text.size(); ++j) {
            char c = text[j];
            auto it = curr->children.find(c);
            if (it == curr->children.end()) {
                break;
            }
            curr = it->second.get();
            if (curr->token_id != -1) {
                best_token = curr->token_id;
                match_len = (j - i + 1);
            }
        }

        if (match_len > 0 && best_token != -1) {
            tokens.push_back(best_token);
            i += match_len;
        } else {
            // Fallback for unknown character
            tokens.push_back(0);
            i++;
        }
    }

    return tokens;
}

// Decodes subword token IDs back into full text string
string Tokenizer::decode(const vector<int>& tokens) const {
    string text;
    for (int t : tokens) {
        auto it = id_to_token.find(t);
        if (it != id_to_token.end()) {
            text += it->second;
        } else {
            text += "?";
        }
    }
    return text;
}

} // namespace ring2
