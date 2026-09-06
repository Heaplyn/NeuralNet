#pragma once

/**
 * @file conv2d.hpp
 * @brief 2D Convolutional Layer with dynamic filter expansion and analytical backward propagation in Ring 3.
 */

#include "ring3/tensor4d.hpp"
#include "ring0/activations.hpp"
#include <vector>
#include <string>

namespace ring3 {

/**
 * @class Conv2D
 * @brief 2D Convolutional layer with weights (C_out x C_in x Kh x Kw), bias, padding, stride, and activations.
 */
class Conv2D {
public:
    size_t in_channels = 1;
    size_t out_channels = 16;
    size_t kernel_h = 3;
    size_t kernel_w = 3;
    size_t stride_h = 1;
    size_t stride_w = 1;
    size_t pad_h = 1;
    size_t pad_w = 1;

    ring0::ActivationType activation = ring0::ActivationType::ReLU;

    Tensor4D weights;                 ///< Kernel weights (out_channels, in_channels, kernel_h, kernel_w)
    std::vector<float> biases;        ///< Bias vector (out_channels)

    Tensor4D grad_weights;            ///< Accumulated weight gradients
    std::vector<float> grad_biases;   ///< Accumulated bias gradients

    // --- Backpropagation Cache ---
    Tensor4D last_input;              ///< Cached input tensor (B, C_in, H_in, W_in)
    Tensor4D last_pre_act;            ///< Cached pre-activation output tensor (B, C_out, H_out, W_out)
    Tensor4D last_output;             ///< Cached post-activation output tensor

    Conv2D(size_t in_c, size_t out_c, size_t k_h = 3, size_t k_w = 3,
           size_t s_h = 1, size_t s_w = 1, size_t p_h = 1, size_t p_w = 1,
           ring0::ActivationType act = ring0::ActivationType::ReLU);

    /// Computes forward 2D convolution and activation
    Tensor4D forward(const Tensor4D& input);

    /// Computes gradients dW, db, and backpropagates input gradient dX
    Tensor4D backward(const Tensor4D& grad_output, float relevancy = 1.0f);

    /// Dynamically expands the number of output filters (adds capacity)
    void expand_filters(size_t additional_out_channels);

    /// Dynamically expands the number of input channels
    void expand_input_channels(size_t additional_in_channels);

    /// Resets gradients to zero
    void reset_gradients();

    size_t get_parameter_count() const {
        return weights.size() + biases.size();
    }
};

} // namespace ring3
