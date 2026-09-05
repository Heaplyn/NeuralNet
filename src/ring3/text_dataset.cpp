#include <random>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <unordered_set>
#include <vector>
#include <string>
#include <iostream>
#include <cmath>

#include "ring0/config.hpp"
#include "ring3/text_dataset.hpp"

using namespace std;

namespace ring3 {

// Local deterministic RNG engine for randomized batch generation
static mt19937& get_batch_rng() {
    static mt19937 rng(42);
    return rng;
}

// Encodes the corpus text into a contiguous sequence of integer token IDs and marks SFT prompt comment masks
TextDataset::TextDataset(const string& text, const ring2::Tokenizer& tokenizer, size_t seq_length)
    : seq_len(seq_length), sft_masking_enabled(true) {
    token_stream = tokenizer.encode(text);
    is_prompt_mask.resize(token_stream.size(), false);

    // Identify prompt comment tokens (lines starting with "//" up to newline '\n')
    bool in_comment = false;
    for (size_t i = 0; i < token_stream.size(); ++i) {
        int tid = token_stream[i];
        auto it = tokenizer.id_to_token.find(tid);
        if (it != tokenizer.id_to_token.end()) {
            const string& s = it->second;
            if (s.find("//") != string::npos) {
                in_comment = true;
            }
            if (in_comment) {
                is_prompt_mask[i] = true;
                if (s.find('\n') != string::npos) {
                    in_comment = false; // newline terminates the prompt comment
                }
            }
        }
    }
}

// Calculates maximum number of non-overlapping full batches in the stream
size_t TextDataset::get_num_batches(size_t batch_size) const {
    if (token_stream.size() <= seq_len + 1) return 0;
    size_t total_samples = (token_stream.size() - 1) / seq_len;
    return total_samples / batch_size;
}

// Slices a deterministic sequential batch
TextBatch TextDataset::get_batch(size_t batch_idx, size_t batch_size) const {
    TextBatch batch;
    batch.batch_size = batch_size;
    batch.seq_len = seq_len;
    batch.input_ids.resize(batch_size * seq_len);
    batch.target_ids.resize(batch_size * seq_len);

    size_t sample_offset = batch_idx * batch_size * seq_len;

    for (size_t b = 0; b < batch_size; ++b) {
        size_t start = sample_offset + b * seq_len;
        if (start + seq_len + 1 > token_stream.size()) {
            start = 0; // Wrap around if out of bounds
        }

        for (size_t s = 0; s < seq_len; ++s) {
            batch.input_ids[b * seq_len + s] = token_stream[start + s];
            int target = token_stream[start + s + 1];
            if (sft_masking_enabled && start + s + 1 < is_prompt_mask.size() && is_prompt_mask[start + s + 1]) {
                target = -1; // SFT Masked prompt comment!
            }
            batch.target_ids[b * seq_len + s] = target;
        }
    }

    return batch;
}

// Sets active subset of dataset as a fraction in (0.0, 1.0]
void TextDataset::set_active_ratio(float ratio) {
    float r = max(0.001f, min(1.0f, ratio));
    size_t target_tokens = static_cast<size_t>(static_cast<float>(token_stream.size()) * r);
    set_active_tokens(target_tokens);
}

// Sets active subset of dataset to explicit token count
void TextDataset::set_active_tokens(size_t tokens) {
    if (tokens == 0 || tokens >= token_stream.size()) {
        active_tokens_limit = token_stream.size();
    } else {
        active_tokens_limit = max(static_cast<size_t>(128), min(token_stream.size(), tokens));
    }
}

// Retrieves current active token count available for sampling
size_t TextDataset::get_active_tokens() const {
    if (active_tokens_limit > 0 && active_tokens_limit < token_stream.size()) {
        return active_tokens_limit;
    }
    return token_stream.size();
}

// Slices a batch starting at random offsets throughout the corpus
TextBatch TextDataset::get_random_batch(size_t batch_size) const {
    return get_random_batch(batch_size, seq_len);
}

// Slices a batch with dynamic sequence context length
TextBatch TextDataset::get_random_batch(size_t batch_size, size_t dynamic_seq_len) const {
    TextBatch batch;
    batch.batch_size = batch_size;
    batch.seq_len = dynamic_seq_len;
    batch.input_ids.resize(batch_size * dynamic_seq_len);
    batch.target_ids.resize(batch_size * dynamic_seq_len);

    size_t available_tokens = get_active_tokens();
    if (available_tokens <= dynamic_seq_len + 1) {
        available_tokens = token_stream.size();
    }
    if (available_tokens <= dynamic_seq_len + 1) return batch;

    size_t max_start = available_tokens - dynamic_seq_len - 1;
    uniform_int_distribution<size_t> dist(0, max_start);

    for (size_t b = 0; b < batch_size; ++b) {
        size_t start = dist(get_batch_rng());
        for (size_t s = 0; s < dynamic_seq_len; ++s) {
            batch.input_ids[b * dynamic_seq_len + s] = token_stream[start + s];
            int target = token_stream[start + s + 1];
            if (sft_masking_enabled && start + s + 1 < is_prompt_mask.size() && is_prompt_mask[start + s + 1]) {
                target = -1; // SFT Masked prompt comment!
            }
            batch.target_ids[b * dynamic_seq_len + s] = target;
        }
    }

    return batch;
}

// Feature 6: Auto-Learning Data Filter (Calculus-based Information Entropy & Gradient Surprise)
// Filters candidate sequence windows based on token transition surprise and diversity
TextBatch TextDataset::get_information_filtered_batch(size_t batch_size, size_t dynamic_seq_len, size_t candidate_pool_multiplier) const {
    size_t available_tokens = get_active_tokens();
    if (available_tokens <= dynamic_seq_len + 1) {
        return get_random_batch(batch_size, dynamic_seq_len);
    }

    size_t pool_size = max(batch_size, batch_size * candidate_pool_multiplier);
    size_t max_start = available_tokens - dynamic_seq_len - 1;
    uniform_int_distribution<size_t> dist(0, max_start);

    struct Candidate {
        size_t start_idx;
        float information_score;
    };
    vector<Candidate> candidates;
    candidates.reserve(pool_size);

    for (size_t c = 0; c < pool_size; ++c) {
        size_t start = dist(get_batch_rng());
        
        // Calculus: Discrete 1st difference / gradient of token transitions Delta t = |t_{i+1} - t_i|
        // Measures transition variability (information density) vs flat repetitive padding/whitespace
        float transition_variance = 0.0f;
        float unique_token_count = 0.0f;
        unordered_set<int> seen_tokens;

        for (size_t s = 0; s < dynamic_seq_len; ++s) {
            int t1 = token_stream[start + s];
            int t2 = token_stream[start + s + 1];
            float diff = static_cast<float>(abs(t2 - t1));
            transition_variance += diff;
            seen_tokens.insert(t1);
        }
        unique_token_count = static_cast<float>(seen_tokens.size());
        float diversity_ratio = unique_token_count / static_cast<float>(dynamic_seq_len);

        // Information score: penalizes repetitive trivial sequences (diversity_ratio < 0.2)
        // Rewards dense, highly-structured lexical phrases
        float score = (transition_variance / static_cast<float>(dynamic_seq_len)) * (diversity_ratio * diversity_ratio);
        candidates.push_back({start, score});
    }

    // Sort descending by information score (steepest information content)
    sort(candidates.begin(), candidates.end(), [](const Candidate& a, const Candidate& b) {
        return a.information_score > b.information_score;
    });

    TextBatch batch;
    batch.batch_size = batch_size;
    batch.seq_len = dynamic_seq_len;
    batch.input_ids.resize(batch_size * dynamic_seq_len);
    batch.target_ids.resize(batch_size * dynamic_seq_len);

    for (size_t b = 0; b < batch_size; ++b) {
        size_t start = candidates[b].start_idx;
        for (size_t s = 0; s < dynamic_seq_len; ++s) {
            batch.input_ids[b * dynamic_seq_len + s] = token_stream[start + s];
            int target = token_stream[start + s + 1];
            if (sft_masking_enabled && start + s + 1 < is_prompt_mask.size() && is_prompt_mask[start + s + 1]) {
                target = -1;
            }
            batch.target_ids[b * dynamic_seq_len + s] = target;
        }
    }

    return batch;
}

// Computes intrinsic informational/semantic relevancy score for a token in [0.0, 1.0]
float TextDataset::compute_token_relevance(size_t token_pos) const {
    if (token_stream.empty() || token_pos >= token_stream.size()) return 0.0f;

    int t = token_stream[token_pos];

    // 1. High-frequency / common token penalty (e.g., whitespace / punctuation typically < 32 in ASCII vocab)
    float base_salience = 0.5f;
    if (t > 32 && t < 127) {
        // Alphanumeric standard character
        base_salience = 0.70f;
    } else if (t >= 127) {
        // Multi-byte subword / merged BPE chunk
        base_salience = 0.95f;
    } else if (t == 32 || t == 10 || t == 9 || t == 13) {
        // Whitespace / newline
        base_salience = 0.20f;
    }

    // 2. Local Transition Novelty (Calculus of gradient surprise with neighbors)
    float transition_novelty = 0.0f;
    int count = 0;
    if (token_pos > 0) {
        transition_novelty += static_cast<float>(abs(t - token_stream[token_pos - 1]));
        count++;
    }
    if (token_pos + 1 < token_stream.size()) {
        transition_novelty += static_cast<float>(abs(t - token_stream[token_pos + 1]));
        count++;
    }
    if (count > 0) transition_novelty /= (count * 256.0f);
    transition_novelty = min(1.0f, transition_novelty);

    // 3. Combined normalized relevance in [0.0, 1.0]
    float raw_relevance = 0.6f * base_salience + 0.4f * transition_novelty;
    return max(0.0f, min(1.0f, raw_relevance));
}

// Computes interpolated context window radius based on token relevancy
size_t TextDataset::get_interpolated_window_radius(float relevance) const {
    const auto& cfg = ring0::get_config();
    float r_clamped = max(0.0f, min(1.0f, relevance));
    float interpolated_factor = pow(r_clamped, cfg.relevance_interpolated_alpha);

    float min_w = static_cast<float>(cfg.min_relevance_window);
    float max_w = static_cast<float>(cfg.max_relevance_window);
    float w = min_w + interpolated_factor * (max_w - min_w);

    return static_cast<size_t>(round(w));
}

// Extracts a context window surrounding a token with interpolated relevancy scores for each neighbor
TextDataset::TokenContextWindow TextDataset::extract_relevance_parsed_window(size_t token_pos) const {
    TokenContextWindow win;
    if (token_stream.empty() || token_pos >= token_stream.size()) return win;

    float r = compute_token_relevance(token_pos);
    size_t radius = get_interpolated_window_radius(r);

    win.anchor_token = token_stream[token_pos];
    win.anchor_position = token_pos;
    win.anchor_relevance = r;
    win.parsed_window_radius = radius;

    size_t start = (token_pos > radius) ? (token_pos - radius) : 0;
    size_t end = min(token_stream.size() - 1, token_pos + radius);

    win.tokens.reserve(end - start + 1);
    win.interpolated_relevancies.reserve(end - start + 1);

    const float PI = 3.14159265358979323846f;
    float denom = static_cast<float>(radius + 1);

    for (size_t i = start; i <= end; ++i) {
        win.tokens.push_back(token_stream[i]);

        // Interpolate relevancy amount based on distance d to anchor token
        float dist = static_cast<float>(abs(static_cast<long long>(i) - static_cast<long long>(token_pos)));
        float kernel_weight = cos((PI * 0.5f * dist) / denom);
        kernel_weight = max(0.0f, kernel_weight);

        float interpolated_amount = r * kernel_weight;
        win.interpolated_relevancies.push_back(interpolated_amount);
    }

    return win;
}

// Generates a training batch centered on high-relevance anchor tokens with interpolated context parsing
TextBatch TextDataset::get_token_relevance_batch(size_t batch_size, size_t dynamic_seq_len) const {
    size_t available_tokens = get_active_tokens();
    if (available_tokens <= dynamic_seq_len + 1) {
        return get_random_batch(batch_size, dynamic_seq_len);
    }

    // Candidate anchor sampling pool
    size_t pool_size = max(batch_size * 4, static_cast<size_t>(16));
    size_t max_anchor = available_tokens - 1;
    uniform_int_distribution<size_t> dist(0, max_anchor);

    struct AnchorCandidate {
        size_t anchor_pos;
        float relevance;
    };
    vector<AnchorCandidate> candidates;
    candidates.reserve(pool_size);

    for (size_t c = 0; c < pool_size; ++c) {
        size_t pos = dist(get_batch_rng());
        float r = compute_token_relevance(pos);
        candidates.push_back({pos, r});
    }

    // Sort descending by token relevancy
    sort(candidates.begin(), candidates.end(), [](const AnchorCandidate& a, const AnchorCandidate& b) {
        return a.relevance > b.relevance;
    });

    TextBatch batch;
    batch.batch_size = batch_size;
    batch.seq_len = dynamic_seq_len;
    batch.input_ids.resize(batch_size * dynamic_seq_len);
    batch.target_ids.resize(batch_size * dynamic_seq_len);

    for (size_t b = 0; b < batch_size; ++b) {
        size_t anchor = candidates[b].anchor_pos;
        float r = candidates[b].relevance;
        size_t radius = get_interpolated_window_radius(r);

        // Center window on anchor token with interpolated radius bounds clamping
        size_t half_seq = min(dynamic_seq_len / 2, radius);
        size_t start = 0;
        if (anchor > half_seq) {
            start = min(anchor - half_seq, available_tokens - dynamic_seq_len - 1);
        }

        for (size_t s = 0; s < dynamic_seq_len; ++s) {
            batch.input_ids[b * dynamic_seq_len + s] = token_stream[start + s];
            int target = token_stream[start + s + 1];
            if (sft_masking_enabled && start + s + 1 < is_prompt_mask.size() && is_prompt_mask[start + s + 1]) {
                target = -1;
            }
            batch.target_ids[b * dynamic_seq_len + s] = target;
        }
    }

    return batch;
}

// Ingests and appends another text file (.txt, .md) to the active token stream
size_t TextDataset::append_text_file(const string& filepath, const ring2::Tokenizer& tokenizer) {
    ifstream f(filepath, ios::binary);
    if (!f.is_open()) return 0;

    stringstream ss;
    ss << f.rdbuf();
    string content = ss.str();
    if (content.empty()) return 0;

    vector<int> new_tokens = tokenizer.encode(content);
    size_t added = new_tokens.size();
    token_stream.insert(token_stream.end(), new_tokens.begin(), new_tokens.end());
    is_prompt_mask.resize(token_stream.size(), false);
    return added;
}

// Ingests and appends a binary file (.bin: raw text or token ids) to the active token stream
size_t TextDataset::append_binary_file(const string& filepath, const ring2::Tokenizer& tokenizer) {
    ifstream f(filepath, ios::binary | ios::ate);
    if (!f.is_open()) return 0;

    streamsize file_size = f.tellg();
    if (file_size <= 0) return 0;
    f.seekg(0, ios::beg);

    vector<char> buffer(static_cast<size_t>(file_size));
    if (!f.read(buffer.data(), file_size)) return 0;

    // Check if buffer contains binary token IDs (e.g. uint16_t or int32_t) or raw binary text
    bool is_int32_tokens = (file_size % 4 == 0) && (file_size >= 8);
    bool is_uint16_tokens = (file_size % 2 == 0) && (file_size >= 4);

    size_t added = 0;
    size_t vocab_sz = tokenizer.vocab_size;

    // Heuristic 1: If file length is multiple of 4, check if values look like valid token IDs
    if (is_int32_tokens) {
        const int32_t* i32_ptr = reinterpret_cast<const int32_t*>(buffer.data());
        size_t count = static_cast<size_t>(file_size / 4);
        bool valid_i32 = true;
        size_t check_count = min(count, static_cast<size_t>(256));
        for (size_t i = 0; i < check_count; ++i) {
            if (i32_ptr[i] < 0 || static_cast<size_t>(i32_ptr[i]) >= vocab_sz) {
                valid_i32 = false;
                break;
            }
        }
        if (valid_i32) {
            token_stream.reserve(token_stream.size() + count);
            for (size_t i = 0; i < count; ++i) {
                token_stream.push_back(static_cast<int>(i32_ptr[i]));
            }
            is_prompt_mask.resize(token_stream.size(), false);
            return count;
        }
    }

    // Heuristic 2: If file length is multiple of 2, check if values look like uint16 token IDs
    if (is_uint16_tokens) {
        const uint16_t* u16_ptr = reinterpret_cast<const uint16_t*>(buffer.data());
        size_t count = static_cast<size_t>(file_size / 2);
        bool valid_u16 = true;
        size_t check_count = min(count, static_cast<size_t>(256));
        for (size_t i = 0; i < check_count; ++i) {
            if (static_cast<size_t>(u16_ptr[i]) >= vocab_sz) {
                valid_u16 = false;
                break;
            }
        }
        if (valid_u16) {
            token_stream.reserve(token_stream.size() + count);
            for (size_t i = 0; i < count; ++i) {
                token_stream.push_back(static_cast<int>(u16_ptr[i]));
            }
            is_prompt_mask.resize(token_stream.size(), false);
            return count;
        }
    }

    // Heuristic 3: Otherwise treat as raw binary text buffer and encode via Tokenizer
    string raw_str(buffer.begin(), buffer.end());
    vector<int> new_tokens = tokenizer.encode(raw_str);
    added = new_tokens.size();
    token_stream.insert(token_stream.end(), new_tokens.begin(), new_tokens.end());
    is_prompt_mask.resize(token_stream.size(), false);
    return added;
}

} // namespace ring3
