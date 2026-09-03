#pragma once

/**
 * @file layer.hpp
 * @brief Dense feedforward layer with dynamic neuron expansion in Ring 1.
 */

#include "ring0/tensor.hpp"
#include "ring0/activations.hpp"
#include <string>

using namespace std;

namespace ring1 {

/**
 * @class DenseLayer
 * @brief Fully connected layer with forward/backward passes and dynamic width resizing.
 */
class DenseLayer {
public:
    size_t in_features;                ///< Input dimension (number of inputs)
    size_t out_features;               ///< Output dimension (number of neurons)
    ring0::ActivationType activation;  ///< Activation function applied to linear output

    ring0::Matrix weights;             ///< Weight matrix (in_features x out_features)
    ring0::Matrix biases;              ///< Bias row vector (1 x out_features)

    ring0::Matrix grad_weights;        ///< Accumulated weight gradients
    ring0::Matrix grad_biases;         ///< Accumulated bias gradients

    // --- Backprop Cache ---
    ring0::Matrix last_input;          ///< Cached input X
    ring0::Matrix last_linear;         ///< Cached linear pre-activation Z = X*W + b
    ring0::Matrix last_output;         ///< Cached post-activation output A = act(Z)

    DenseLayer(size_t in_dim, size_t out_dim, ring0::ActivationType act = ring0::ActivationType::ReLU);

    /// Forward pass: computes A = activation(X * W + b)
    ring0::Matrix forward(const ring0::Matrix& input);

    /// Backward pass: computes gradients dW, db, and returns input gradient dX
    ring0::Matrix backward(const ring0::Matrix& grad_output, float relevancy = 1.0f);

    /// Expands output dimension (adds neurons dynamically)
    void expand_output_dim(size_t additional_neurons);

    /// Expands input dimension (connects new inputs dynamically)
    void expand_input_dim(size_t additional_inputs);

    /// Resets parameter gradients to zero
    void reset_gradients();
};

} // namespace ring1
