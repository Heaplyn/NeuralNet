# 🎯 Token Relevancy & Interpolated Context Parsing

The **Token Relevancy and Interpolated Parsing Algorithm** in `ring3::TextDataset` dynamically evaluates the semantic importance and information entropy of every token, extracting **variable, non-linearly sized context windows with distance-decayed relevancies**.

---

## 🎯 Practical Explanation: What is this and Why Does it Exist?

### The Fixed Context Window Inefficiency
In traditional LLM training, datasets are sliced into uniform, fixed-size chunks (e.g. exactly 128 or 2048 tokens per sequence). 

This approach has two major flaws:
1. **Low Information Density Slices**: Slicing through long stretches of repetitive whitespace, punctuation, or boilerplate wastes precious GPU/CPU compute cycles on trivial tokens.
2. **Context Fragmentation**: Splitting text blindly across fixed boundaries cuts critical function signatures, reasoning chains, and sentences in half.

### How Token Relevancy & Interpolated Parsing Solves This
1. **Relevance Scoring $R(t)$**: Every token is scored based on lexical value (keywords, identifiers vs spaces) and transition surprise.
2. **Non-Linear Interpolated Radius $W(R)$**: High-relevance anchor tokens (like a complex algorithm name or key variable) automatically expand their parsing window up to $\pm 64$ tokens to capture full surrounding dependencies, while low-relevance tokens shrink their window down to $\pm 8$ tokens.
3. **Distance-Decayed Gradient Weights**: Surrounding tokens within the window receive cosine-decayed importance weights so that optimization concentrates on the semantic anchor.

---

## 💻 Deep Code Breakdown

### 1. Interpolated Radius & Cosine Decay Mathematics
Located in `src/ring3/text_dataset.cpp`:

```cpp
RelevanceParsedWindow TextDataset::extract_relevance_parsed_window(size_t anchor_pos) const {
    const auto& cfg = ring0::get_config();
    const size_t N = token_stream.size();
    if (N == 0 || anchor_pos >= N) {
        return {};
    }

    // 1. Calculate anchor token relevancy R(t) in [0.0, 1.0]
    int anchor_token = token_stream[anchor_pos];
    float relevance = compute_token_relevance(anchor_pos);

    // 2. Non-linear interpolation for context radius:
    // W(R) = round( W_min + (R)^alpha * (W_max - W_min) )
    float alpha = cfg.relevance_power_alpha; // default 1.3
    float r_interp = std::pow(std::max(0.0f, std::min(1.0f, relevance)), alpha);
    size_t radius = static_cast<size_t>(std::round(
        static_cast<float>(cfg.relevance_min_window) + 
        r_interp * static_cast<float>(cfg.relevance_max_window - cfg.relevance_min_window)
    ));
    radius = std::max(cfg.relevance_min_window, std::min(cfg.relevance_max_window, radius));

    // 3. Extract surrounding window bounds [left, right]
    size_t left = (anchor_pos > radius) ? (anchor_pos - radius) : 0;
    size_t right = std::min(N - 1, anchor_pos + radius);

    RelevanceParsedWindow win;
    win.anchor_token = anchor_token;
    win.anchor_position = anchor_pos;
    win.anchor_relevance = relevance;
    win.parsed_window_radius = radius;

    // 4. Compute distance-decayed interpolated relevancies:
    // r(d) = R_anchor * cos( (pi * d) / (2 * (W + 1)) )
    for (size_t pos = left; pos <= right; ++pos) {
        win.tokens.push_back(token_stream[pos]);
        size_t dist = (pos >= anchor_pos) ? (pos - anchor_pos) : (anchor_pos - pos);
        float frac = static_cast<float>(dist) / static_cast<float>(radius + 1);
        float decayed_rel = relevance * std::max(0.0f, std::cos(frac * 1.5707963f));
        win.decayed_relevancies.push_back(decayed_rel);
    }

    return win;
}
```

---

### 2. Batch Construction via Relevancy Sampling
When the training engine requests a relevancy-parsed batch, it samples high-relevance anchor tokens and assembles continuous target sequences:

```cpp
TextBatch TextDataset::get_token_relevance_batch(size_t batch_size, size_t seq_len) const {
    TextBatch batch;
    batch.input_ids.resize(batch_size * seq_len);
    batch.target_ids.resize(batch_size * seq_len);

    for (size_t b = 0; b < batch_size; ++b) {
        // Sample candidate positions and pick the one with highest relevance
        size_t best_pos = 0;
        float max_r = -1.0f;
        for (size_t trial = 0; trial < 5; ++trial) {
            size_t cand = rand_dist(gen);
            float r = compute_token_relevance(cand);
            if (r > max_r) {
                max_r = r;
                best_pos = cand;
            }
        }

        // Align sequence starting at best_pos
        for (size_t t = 0; t < seq_len; ++t) {
            size_t idx = (best_pos + t) % token_stream.size();
            size_t target_idx = (idx + 1) % token_stream.size();
            batch.input_ids[b * seq_len + t] = token_stream[idx];
            batch.target_ids[b * seq_len + t] = token_stream[target_idx];
        }
    }
    return batch;
}
```

---

## 🔗 Related Notes
- [[04 - Ring 3 (Data & Training Pipelines)/Progressive Curriculum & Horizon Growth|Progressive Curriculum]]
- [[04 - Ring 3 (Data & Training Pipelines)/LLMTrainer Architecture|LLMTrainer Architecture]]
- [[01 - Ring 0 (Core Math & Hardware)/Loss Formulations & Calculus|Loss Formulations & Calculus]]
- [[Index|Return to Master Index]]
