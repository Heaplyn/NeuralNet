#include "ring1/layer.hpp"
#include <cmath>
#include <cstdlib>

using namespace std;

namespace ring1 {

// Constructor: Initializes weights using He or Xavier depending on activation
DenseLayer::DenseLayer(size_t in_dim, size_t out_dim, ring0::ActivationType act)
    : in_features(in_dim), out_features(out_dim), activation(act) {
    if (activation == ring0::ActivationType::ReLU || activation == ring0::ActivationType::LeakyReLU) {
        weights = ring0::Matrix::he(in_features, out_features);
    } else {
        weights = ring0::Matrix::xavier(in_features, out_features);
    }
    biases = ring0::Matrix::zeros(1, out_features);

    grad_weights = ring0::Matrix::zeros(in_features, out_features);
    grad_biases = ring0::Matrix::zeros(1, out_features);
}

// Forward pass: Z = X * W + b, A = activation(Z)
ring0::Matrix DenseLayer::forward(const ring0::Matrix& input) {
    last_input = input;
    last_linear = input.matmul(weights).add_bias(biases);
    last_output = ring0::Activations::forward(activation, last_linear);
    return last_output;
}

// Backward pass: computes dW = X^T * dZ, db = sum(dZ), dX = dZ * W^T
ring0::Matrix DenseLayer::backward(const ring0::Matrix& grad_output,float relevancy) {
    relevancy /= 1.02f;
    ring0::Matrix grad_z = ring0::Activations::backward(activation, last_output, grad_output * relevancy);

    grad_weights = last_input.transpose().matmul(grad_z);
    grad_biases = grad_z.sum_rows();
    ring0::Matrix grad_input = grad_z.matmul(weights.transpose());
    return grad_input;
}

// Dynamically adds output neurons to the layer
void DenseLayer::expand_output_dim(size_t additional_neurons) {
    if (additional_neurons == 0) return;
    size_t new_out = out_features + additional_neurons;

    ring0::Matrix new_weights(in_features, new_out);
    for (size_t r = 0; r < in_features; ++r) {
        for (size_t c = 0; c < out_features; ++c) {
            new_weights(r, c) = weights(r, c);
        }
        for (size_t c = out_features; c < new_out; ++c) {
            new_weights(r, c) = ((rand() / (float)RAND_MAX) - 0.5f) * 0.05f;///(1+sqrt(additional_neurons));
        }
    }
    weights = move(new_weights);

    ring0::Matrix new_biases = ring0::Matrix::zeros(1, new_out);
    for (size_t c = 0; c < out_features; ++c) {
        new_biases(0, c) = biases(0, c);
    }
    biases = move(new_biases);

    grad_weights = ring0::Matrix::zeros(in_features, new_out);
    grad_biases = ring0::Matrix::zeros(1, new_out);
    out_features = new_out;
}

// Dynamically adds incoming inputs to the layer
void DenseLayer::expand_input_dim(size_t additional_inputs) {
    if (additional_inputs == 0) return;
    size_t new_in = in_features + additional_inputs;

    ring0::Matrix new_weights(new_in, out_features);
    for (size_t r = 0; r < in_features; ++r) {
        for (size_t c = 0; c < out_features; ++c) {
            new_weights(r, c) = weights(r, c);
        }
    }
    for (size_t r = in_features; r < new_in; ++r) {
        for (size_t c = 0; c < out_features; ++c) {
            new_weights(r, c) = ((rand() / (float)RAND_MAX) - 0.5f) * 0.05f;
        }
    }
    weights = move(new_weights);
    grad_weights = ring0::Matrix::zeros(new_in, out_features);
    in_features = new_in;
}

// Resets accumulated gradients to zero
void DenseLayer::reset_gradients() {
    grad_weights = ring0::Matrix::zeros(in_features, out_features);
    grad_biases = ring0::Matrix::zeros(1, out_features);
}

} // namespace ring1
