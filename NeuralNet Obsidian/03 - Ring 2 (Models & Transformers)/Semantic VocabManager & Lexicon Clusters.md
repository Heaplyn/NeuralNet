# 🧠 Semantic VocabManager & Vector Indexed Clusters

The `ring2::VocabManager` organizes the model's vocabulary into **16 semantic concept clusters** and tracks layer-wise semantic meaning matrices to enable dynamic vocabulary expansion (neurogenesis).

---

## 🎓 Beginner-Friendly Learning Guide: How Words Live in Vector Space

### Word Vectors as Coordinates in Thought Space
Words are not just isolated IDs; their 128-dimensional embedding vectors are **points in space**:
- `"king"` and `"queen"` are located close together in space.
- The vector math $\mathbf{e}_{\text{king}} - \mathbf{e}_{\text{man}} + \mathbf{e}_{\text{woman}} \approx \mathbf{e}_{\text{queen}}$ actually works because the dimensions encode semantic concepts like gender, royalty, and tense!

### The Concept Cluster Hierarchy
Instead of scattering words randomly, `VocabManager` organizes the embedding space into **16 semantic clusters**:

```mermaid
mindmap
  root((16 Concept Clusters))
    Code Syntax & Keywords
      Cluster 0: Control Flow (if, while, return, break)
      Cluster 1: Types & Storage (int, float, auto, struct)
      Cluster 2: OOP & Methods (class, public, virtual)
      Cluster 3: Operators & Math (+, -, *, ->, ==)
    Language & Semantics
      Cluster 4: AI & ML Concepts (attention, loss, layer)
      Cluster 5: Data Structures (vector, tensor, array)
      Cluster 6: Action Verbs (compute, train, optimize)
      Cluster 7: Entities & Nouns (model, token, parameter)
    Grammar & Structure
      Cluster 8: Punctuation & Delimiters ({, }, ;, ,, .)
      Cluster 9: Common Subwords (ing, ed, tion, al)
      Cluster 10..15: Domain Lexicon & Extended Clusters
```

---

## ⚡ Dynamic Vocabulary Expansion (Neurogenesis)

### The Problem with Fixed Vocabularies
In standard models, if you want to add a new token (e.g. `"std::vector"` as a single token) after training has started, you have to resize the embedding table and randomly initialize the new vector, which produces gibberish and destroys pretrained alignments.

### The Semantic Neighbor Averaging Solution
When `VocabManager` identifies a new high-value token, it:
1. Identifies the semantic cluster it belongs to (e.g. Cluster 5: Data Structures).
2. Computes the **centroid of existing nearest neighbors** in that cluster.
3. Initializes the new token's embedding vector as the centroid plus a tiny Gaussian noise:

$$\mathbf{e}_{\text{new}} = \frac{1}{|K|} \sum_{k \in K} \mathbf{e}_k + \mathcal{N}(0, 0.02^2)$$

> [!TIP]
> **Why this helps**: The new token enters the network already carrying an approximate semantic meaning, so it needs only a few gradient steps to fine-tune its exact nuance.

---

## 💻 Deep Code Breakdown

Located in `src/ring2/vocab_manager.cpp`:

```cpp
void VocabManager::expand_with_semantic_centroid(
    int new_token_id, 
    const string& token_str, 
    size_t target_cluster,
    Matrix& embedding_table
) {
    size_t embed_dim = embedding_table.cols;
    const auto& cluster_tokens = clusters[target_cluster].token_ids;

    vector<float> centroid(embed_dim, 0.0f);
    if (!cluster_tokens.empty()) {
        // Average the embeddings of all existing tokens in this cluster
        for (int tok_id : cluster_tokens) {
            for (size_t d = 0; d < embed_dim; ++d) {
                centroid[d] += embedding_table(tok_id, d);
            }
        }
        float inv_count = 1.0f / static_cast<float>(cluster_tokens.size());
        for (size_t d = 0; d < embed_dim; ++d) {
            centroid[d] *= inv_count;
        }
    }

    // Assign centroid + tiny exploratory noise to the new token
    for (size_t d = 0; d < embed_dim; ++d) {
        float noise = ((rand() % 1000) / 1000.0f - 0.5f) * 0.02f;
        embedding_table(new_token_id, d) = centroid[d] + noise;
    }

    // Register token into the cluster
    clusters[target_cluster].token_ids.push_back(new_token_id);
}
```

---

## 🔗 Related Notes
- [[03 - Ring 2 (Models & Transformers)/BPE Tokenizer & Merging Engine|BPE Tokenizer]]
- [[03 - Ring 2 (Models & Transformers)/TransformerLM Decoder (GQA + SwiGLU + RoPE)|TransformerLM Decoder]]
- [[Index|Return to Master Index]]
