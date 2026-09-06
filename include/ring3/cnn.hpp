#pragma once

/**
 * @file cnn.hpp
 * @brief Convolutional Neural Network (CNN) architecture connected with Ring 1 (Dense, Attention, Transformer, Recursive Thought) and Ring 2 (NeuralNet).
 */

#include "ring0/tensor.hpp"
#include "ring1/layer.hpp"
#include "ring1/attention.hpp"
#include "ring1/transformer_block.hpp"
#include "ring1/recursive_layer.hpp"
#include "ring2/neural_net.hpp"
#include "ring3/tensor4d.hpp"
#include "ring3/conv2d.hpp"
#include "ring3/maxpool2d.hpp"
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
 * @brief Convolutional Neural Network integrating Conv2D spatial backbones with Ring 1 and Ring 2 layers:
 *        - Ring 1 Dense layers & Ring 2 NeuralNet classifier backbones
 *        - Ring 1 Recursive Thought layers (latent reasoning loops & CoC proof verification)
 *        - Ring 1 Multi-Head Attention & Transformer Blocks (Visual attention & SwiGLU MLP)
 */
class CNN {
public:
    size_t input_channels = 1;
    size_t input_height = 28;
    size_t input_width = 28;

    // --- Spatial Feature Extraction (Ring 3) ---
    std::vector<Conv2D> conv_layers;
    std::vector<MaxPool2D> pool_layers;
    std::vector<bool> has_pooling; // per conv layer

    // --- Visual Attention & Transformer Stages (Ring 1) ---
    std::vector<ring1::MultiHeadAttention> attention_blocks;
    std::vector<ring1::TransformerBlock> transformer_blocks;

    // --- Cognitive Recursive Thinking Layers (Ring 1) ---
    std::vector<ring1::RecursiveLayer> recursive_layers;

    // --- Dense Classification Stages (Ring 1 & Ring 2) ---
    std::vector<ring1::DenseLayer> dense_layers;
    ring2::NeuralNet dense_head;       ///< Attached Ring 2 sequential NeuralNet
    bool use_neural_net_head = false;  ///< Flag indicating if external NeuralNet is attached

    // --- Backprop Cache ---
    std::vector<Tensor4D> conv_outputs;
    std::vector<Tensor4D> pool_outputs;
    Tensor4D last_conv_final_feature_map;
    ring0::Matrix last_flattened_features;
    ring0::Matrix last_post_attention_features;
    ring0::Matrix last_post_transformer_features;
    ring0::Matrix last_post_recursive_features;

    CNN() = default;
    CNN(size_t in_c, size_t in_h, size_t in_w);

    /// Appends a Conv2D block with optional MaxPool2D
    void add_conv_block(const ConvBlockConfig& cfg);

    /// Appends a Conv2D layer directly
    void add_conv(size_t in_c, size_t out_c, size_t k = 3, size_t s = 1, size_t p = 1,
                  ring0::ActivationType act = ring0::ActivationType::ReLU, bool pool = true);

    /// Appends a Ring 1 Multi-Head Attention block
    void add_attention(size_t embed_dim, size_t num_heads, size_t num_kv_heads = 0);

    /// Appends a Ring 1 Transformer Decoder Block with SwiGLU
    void add_transformer_block(size_t embed_dim, size_t num_heads, size_t ffn_dim);

    /// Appends a Ring 1 Recursive Thought reasoning layer with CoC logic
    void add_recursive_thought(const std::string& name, size_t out_dim, size_t depth = 2);

    /// Appends an existing Ring 1 RecursiveLayer
    void add_recursive_layer(const ring1::RecursiveLayer& layer);

    /// Adds a dense layer to the classifier head
    void add_dense(size_t out_dim, ring0::ActivationType act = ring0::ActivationType::ReLU);

    /// Appends an existing Ring 1 DenseLayer
    void add_dense_layer(const ring1::DenseLayer& layer);

    /// Attaches an entire Ring 2 NeuralNet as the classifier stage
    void attach_neural_net(const ring2::NeuralNet& head_net);

    /// Retrieves reference to attached Ring 2 NeuralNet head
    ring2::NeuralNet& get_neural_net_head() { return dense_head; }
    const ring2::NeuralNet& get_neural_net_head() const { return dense_head; }

    /// Forward pass taking 4D input batch (B, C, H, W)
    ring0::Matrix forward(const Tensor4D& input);

    /// Forward pass taking flat 2D image matrix (B, C*H*W)
    ring0::Matrix forward_flat(const ring0::Matrix& flat_images);

    /// Backward backpropagation across dense classifier, recursive thinking, attention, and conv stages
    void backward(const ring0::Matrix& grad_output, float relevancy = 1.0f);

    /// Resets all gradients across conv, attention, transformer, recursive, and dense layers
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

    /// Computes current feature dimension going into the classifier
    size_t get_current_feature_dim() const;
};

} // namespace ring3
