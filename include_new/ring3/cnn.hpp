#pragma once

/**
 * @file cnn.hpp
 * @brief Convolutional Neural Network (CNN) architecture with dynamic capacity expansion and multi-stage backpropagation in Ring 3.
 */

#include "ring3/tensor4d.hpp"
#include "ring3/conv2d.hpp"
#include "ring3/maxpool2d.hpp"
#include "ring1/layer.hpp"
#include <vector>
#include <memory>
#include <string>

namespace ring3 {

/**
 * @struct ConvBlockConfig
 * @brief Specification for a convolutional block (Conv2D followed by optional MaxPool2D).
 */
struct ConvBlockConfig {
    size_t in_channels = 1;
    size_t out_channels = 16;
    size_t kernel_size = 3;
    size_t stride = 1;
    size_t padding = 1;
    ring0::ActivationType activation = ring0::ActivationType::ReLU;
    bool use_maxpool = true;
    size_t pool_size = 2;
    size_t pool_stride = 2;
};

/**
 * @class CNN
 * @brief Convolutional Neural Network integrating Conv2D feature extractors with Ring 1 Dense classification heads.
 */
class CNN {
public:
    size_t input_channels = 1;
    size_t input_height = 28;
    size_t input_width = 28;

    std::vector<Conv2D> conv_layers;
    std::vector<MaxPool2D> pool_layers;
    std::vector<bool> has_pooling; // per conv layer
    std::vector<ring1::DenseLayer> dense_layers;

    // --- Backprop Cache ---
    std::vector<Tensor4D> conv_outputs;
    std::vector<Tensor4D> pool_outputs;
    Tensor4D last_conv_final_feature_map;
    ring0::Matrix last_flattened_features;

    CNN() = default;
    CNN(size_t in_c, size_t in_h, size_t in_w);

    /// Appends a Conv2D block with optional MaxPool2D
    void add_conv_block(const ConvBlockConfig& cfg);

    /// Appends a Conv2D layer directly
    void add_conv(size_t in_c, size_t out_c, size_t k = 3, size_t s = 1, size_t p = 1,
                  ring0::ActivationType act = ring0::ActivationType::ReLU, bool pool = true);

    /// Finalizes conv backbone and adds a dense classifier head
    void add_dense(size_t out_dim, ring0::ActivationType act = ring0::ActivationType::ReLU);

    /// Forward pass taking 4D input batch (B, C, H, W)
    ring0::Matrix forward(const Tensor4D& input);

    /// Forward pass taking flat 2D image matrix (B, C*H*W)
    ring0::Matrix forward_flat(const ring0::Matrix& flat_images);

    /// Backward backpropagation across dense classifier, bridge, and convolutional stages
    void backward(const ring0::Matrix& grad_output, float relevancy = 1.0f);

    /// Resets all gradients across conv and dense layers
    void reset_gradients();

    /// Dynamically expands output filters of a specific conv layer
    bool expand_conv_filters(size_t layer_idx, size_t additional_filters);

    /// Dynamically expands neurons of a dense layer
    bool expand_dense_neurons(size_t layer_idx, size_t additional_neurons);

    /// Proportional capacity growth across all feature channels
    void expand_capacity(float growth_factor = 1.25f);

    size_t get_total_parameters() const;
    void print_architecture() const;

    /// Computes spatial output dimensions after conv backbone
    void get_flattened_dim(size_t& out_c, size_t& out_h, size_t& out_w) const;
};

} // namespace ring3
