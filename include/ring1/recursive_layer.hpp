#pragma once

/**
 * @file recursive_layer.hpp
 * @brief Recursive thinking and hierarchical composite layer architecture in Ring 1.
 * 
 * Features:
 * - Tree structure: each layer has a `parent` pointer and multiple `children` sub-layers.
 * - Latent Thinking Loops: recursively refines its internal state over K thinking steps.
 * - Thought Chain History: tracks thought progression, vector norms, deltas, and semantic stages.
 * - Multi-Pass Thought Chain Looping: iterates through and self-reflects over its own thought chains.
 * - Debug Mode Diagnostics: verbose logging of thought chains and convergence behavior.
 */

#include "ring0/tensor.hpp"
#include "ring0/config.hpp"
#include "ring1/layer.hpp"
#include <vector>
#include <memory>
#include <string>

using namespace std;

namespace ring1 {

/**
 * @struct ThoughtStep
 * @brief Diagnostic telemetry recorded per step of a reasoning thought chain.
 */
struct ThoughtStep {
    size_t step_index;             ///< Thought iteration index (0..K-1)
    size_t loop_cycle;             ///< Reflection / chain loop pass
    string layer_name;             ///< Name of executing recursive reasoner
    float energy_norm;             ///< L2 norm of the thought representation
    float delta_magnitude;         ///< ||H_{t+1} - H_t|| residual shift
    float cosine_similarity;       ///< Directional alignment with previous thought state
    string stage_description;      ///< Semantic stage label (e.g., "Perceptual Encoding", "Synthesis")
};

/**
 * @class RecursiveLayer
 * @brief Composite AI layer supporting recursive self-reasoning loops and hierarchical tree navigation.
 */
class RecursiveLayer {
public:
    string name;
    size_t in_features;
    size_t out_features;
    size_t thinking_depth;               ///< Number of recursive reasoning loops (K)

    RecursiveLayer* parent;              ///< Pointer to parent layer (nullptr if root)
    vector<unique_ptr<RecursiveLayer>> children; ///< Child sub-layers

    // Internal transformation matrices for recursive thinking
    ring0::Matrix W_think;
    ring0::Matrix b_think;
    ring0::Matrix W_context;

    // Thought chain history and reflection buffers
    vector<ThoughtStep> thought_chain_history;

    RecursiveLayer(const string& layer_name, size_t in_dim, size_t out_dim, size_t depth = 2);

    /// Adds a child sub-layer and binds its parent pointer
    void add_child(unique_ptr<RecursiveLayer> child);

    /// Recursively processes input X through K internal thinking loops and child sub-layers
    ring0::Matrix forward(const ring0::Matrix& X);

    /**
     * @brief Loops through its own thought chains for multiple reflection cycles,
     *        re-evaluating and refining the latent representation with residual damping.
     * @param X Initial input matrix or thought stimulus.
     * @param num_reflection_cycles Number of self-reflective loops over the thought chain.
     * @param verbose_override Force verbose output if desired (defaults to config setting).
     */
    ring0::Matrix loop_thought_chain(const ring0::Matrix& X, size_t num_reflection_cycles = 3, bool verbose_override = false);

    /// Retrieves contextual state from ancestor/parent hierarchy
    ring0::Matrix get_parent_context() const;

    /// Checks if this layer is a root node
    bool is_root() const { return parent == nullptr; }

    /// Returns depth level in the layer tree
    size_t get_tree_depth() const;

    /// Prints a formatted ASCII tree & thought chain summary
    void print_thought_chain_summary() const;
};

} // namespace ring1
