#pragma once

/**
 * @file activations.hpp
 * @brief Activation function registry and analytical derivatives for Ring 0.
 * 
 * Provides:
 * - GELU: Gaussian Error Linear Unit for Transformer LLMs.
 * - ReLU & LeakyReLU: Rectified Linear activations for dense networks.
 * - Sigmoid & Tanh: Smooth non-linear activations.
 * - Softmax: Normalized exponential probabilities.
 */

#include "ring0/tensor.hpp"

using namespace std;

namespace ring0 {

/// Enum of supported activation function types
enum class ActivationType {
    None,
    Sigmoid,
    ReLU,
    LeakyReLU,
    Tanh,
    Softmax,
    GELU,
    SiLU
};

/**
 * @class Activations
 * @brief Static registry for forward and backward passes of all activation functions.
 */
class Activations {
public:
    /// Dispatches forward activation based on ActivationType enum
    static Matrix forward(ActivationType type, const Matrix& input);

    /// Dispatches backward analytical gradient based on ActivationType enum
    static Matrix backward(ActivationType type, const Matrix& forward_output, const Matrix& grad_output);

    // --- Explicit Function Implementations ---

    /// Sigmoid: f(x) = 1 / (1 + exp(-x))
    static Matrix sigmoid(const Matrix& x);
    /// Sigmoid derivative: dL/dx = grad * (s * (1 - s))
    static Matrix sigmoid_derivative(const Matrix& sigmoid_out, const Matrix& grad_output);

    /// ReLU: f(x) = max(0, x)
    static Matrix relu(const Matrix& x);
    /// ReLU derivative: dL/dx = grad * (x > 0 ? 1 : 0)
    static Matrix relu_derivative(const Matrix& relu_out, const Matrix& grad_output);

    /// LeakyReLU: f(x) = x if x > 0 else alpha * x
    static Matrix leaky_relu(const Matrix& x, float alpha = 0.01f);
    /// LeakyReLU derivative: dL/dx = grad * (x > 0 ? 1 : alpha)
    static Matrix leaky_relu_derivative(const Matrix& leaky_out, const Matrix& grad_output, float alpha = 0.01f);

    /// Hyperbolic Tangent: f(x) = tanh(x)
    static Matrix tanh_act(const Matrix& x);
    /// Tanh derivative: dL/dx = grad * (1 - tanh(x)^2)
    static Matrix tanh_derivative(const Matrix& tanh_out, const Matrix& grad_output);

    /// Softmax: f(x_i) = exp(x_i) / sum(exp(x_j))
    static Matrix softmax(const Matrix& x);

    /// GELU: f(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    static Matrix gelu(const Matrix& x);
    /// GELU derivative: exact analytical derivative via product rule and sech^2
    static Matrix gelu_derivative(const Matrix& input, const Matrix& grad_output);

    /// SiLU / Swish: f(x) = x * sigmoid(x)
    static Matrix silu(const Matrix& x);
    /// SiLU derivative: dL/dx = grad * (sigmoid(x) * (1 + x * (1 - sigmoid(x))))
    static Matrix silu_derivative(const Matrix& input, const Matrix& grad_output);
};

} // namespace ring0
