#pragma once

/**
 * @file maxpool2d.hpp
 * @brief 2D Max Pooling and Average Pooling layers with analytical backpropagation in Ring 3.
 */

#include "ring3/tensor4d.hpp"
#include <vector>

namespace ring3 {

/**
 * @class MaxPool2D
 * @brief 2D Max Pooling layer with argmax index caching for exact gradient routing.
 */
class MaxPool2D {
public:
    size_t kernel_h = 2;
    size_t kernel_w = 2;
    size_t stride_h = 2;
    size_t stride_w = 2;
    size_t pad_h = 0;
    size_t pad_w = 0;

    // --- Backprop Cache ---
    size_t last_batch_size = 0;
    size_t last_channels = 0;
    size_t last_in_h = 0;
    size_t last_in_w = 0;
    std::vector<size_t> argmax_indices; ///< Cached flat indices of max elements

    MaxPool2D(size_t k_h = 2, size_t k_w = 2, size_t s_h = 2, size_t s_w = 2, size_t p_h = 0, size_t p_w = 0);

    /// Forward max pooling
    Tensor4D forward(const Tensor4D& input);

    /// Backward gradient routing to argmax locations
    Tensor4D backward(const Tensor4D& grad_output, float relevancy = 1.0f);
};

/**
 * @class AvgPool2D
 * @brief 2D Average Pooling layer.
 */
class AvgPool2D {
public:
    size_t kernel_h = 2;
    size_t kernel_w = 2;
    size_t stride_h = 2;
    size_t stride_w = 2;
    size_t pad_h = 0;
    size_t pad_w = 0;

    size_t last_batch_size = 0;
    size_t last_channels = 0;
    size_t last_in_h = 0;
    size_t last_in_w = 0;

    AvgPool2D(size_t k_h = 2, size_t k_w = 2, size_t s_h = 2, size_t s_w = 2, size_t p_h = 0, size_t p_w = 0);

    Tensor4D forward(const Tensor4D& input);
    Tensor4D backward(const Tensor4D& grad_output, float relevancy = 1.0f);
};

} // namespace ring3
