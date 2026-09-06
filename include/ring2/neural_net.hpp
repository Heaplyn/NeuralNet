#pragma once

/**
 * @file neural_net.hpp
 * @brief Multi-layer feedforward neural network container with dynamic growth support in Ring 2.
 */

#include "ring0/tensor.hpp"
#include "ring1/layer.hpp"
#include <vector>
#include <string>

using namespace std;

namespace ring2 {

/**
 * @class NeuralNet
 * @brief Sequential multi-layer neural network with layer-by-layer forward/backward passes.
 */
class NeuralNet {
public:
    vector<ring1::DenseLayer> layers; ///< Stack of dense layers

    NeuralNet() = default;

    /// Adds an existing DenseLayer
    void add_layer(const ring1::DenseLayer& layer);

    /// Constructs and appends a DenseLayer directly
    void add_dense(size_t in_dim, size_t out_dim, ring0::ActivationType act = ring0::ActivationType::ReLU);

    /// Forward pass through all layers sequentially
    ring0::Matrix forward(const ring0::Matrix& input);

    /// Backward gradient backpropagation in reverse layer order returning input gradient dX
    ring0::Matrix backward(const ring0::Matrix& grad_output, float relevancy = 1.0f);

    /// Resets parameter gradients to zero across all layers
    void reset_gradients();

    /// Dynamically expands hidden layer width at layer_index
    bool expand_hidden_layer(size_t layer_index, size_t additional_neurons);

    size_t get_num_layers() const { return layers.size(); }
    size_t get_total_parameters() const;
    void print_architecture() const;
};

} // namespace ring2
