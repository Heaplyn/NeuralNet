#include "ring3/cnn.hpp"
#include <iostream>
#include <iomanip>

namespace ring3 {

CNN::CNN(size_t in_c, size_t in_h, size_t in_w)
    : input_channels(in_c), input_height(in_h), input_width(in_w) {}

void CNN::get_flattened_dim(size_t& out_c, size_t& out_h, size_t& out_w) const {
    out_c = input_channels;
    out_h = input_height;
    out_w = input_width;

    for (size_t i = 0; i < conv_layers.size(); ++i) {
        const auto& conv = conv_layers[i];
        out_h = (out_h + 2 * conv.pad_h - conv.kernel_h) / conv.stride_h + 1;
        out_w = (out_w + 2 * conv.pad_w - conv.kernel_w) / conv.stride_w + 1;
        out_c = conv.out_channels;

        if (has_pooling[i] && i < pool_layers.size()) {
            const auto& pool = pool_layers[i];
            out_h = (out_h + 2 * pool.pad_h - pool.kernel_h) / pool.stride_h + 1;
            out_w = (out_w + 2 * pool.pad_w - pool.kernel_w) / pool.stride_w + 1;
        }
    }
}

void CNN::add_conv_block(const ConvBlockConfig& cfg) {
    size_t in_c = conv_layers.empty() ? input_channels : conv_layers.back().out_channels;
    conv_layers.emplace_back(in_c, cfg.out_channels, cfg.kernel_size, cfg.kernel_size,
                             cfg.stride, cfg.stride, cfg.padding, cfg.padding, cfg.activation);
    has_pooling.push_back(cfg.use_maxpool);
    if (cfg.use_maxpool) {
        pool_layers.emplace_back(cfg.pool_size, cfg.pool_size, cfg.pool_stride, cfg.pool_stride);
    }
}

void CNN::add_conv(size_t in_c, size_t out_c, size_t k, size_t s, size_t p,
                   ring0::ActivationType act, bool pool) {
    ConvBlockConfig cfg;
    cfg.in_channels = in_c;
    cfg.out_channels = out_c;
    cfg.kernel_size = k;
    cfg.stride = s;
    cfg.padding = p;
    cfg.activation = act;
    cfg.use_maxpool = pool;
    add_conv_block(cfg);
}

void CNN::add_dense(size_t out_dim, ring0::ActivationType act) {
    size_t in_dim = 0;
    if (dense_layers.empty()) {
        size_t fc, fh, fw;
        get_flattened_dim(fc, fh, fw);
        in_dim = fc * fh * fw;
    } else {
        in_dim = dense_layers.back().out_features;
    }

    dense_layers.emplace_back(in_dim, out_dim, act);
}

ring0::Matrix CNN::forward(const Tensor4D& input) {
    Tensor4D current = input;
    conv_outputs.clear();
    pool_outputs.clear();

    size_t pool_idx = 0;
    for (size_t i = 0; i < conv_layers.size(); ++i) {
        current = conv_layers[i].forward(current);
        conv_outputs.push_back(current);

        if (has_pooling[i] && pool_idx < pool_layers.size()) {
            current = pool_layers[pool_idx].forward(current);
            pool_outputs.push_back(current);
            pool_idx++;
        }
    }

    last_conv_final_feature_map = current;
    last_flattened_features = current.to_matrix();

    ring0::Matrix dense_act = last_flattened_features;
    for (auto& dense : dense_layers) {
        dense_act = dense.forward(dense_act);
    }

    return dense_act;
}

ring0::Matrix CNN::forward_flat(const ring0::Matrix& flat_images) {
    Tensor4D input_4d = Tensor4D::from_matrix(flat_images, input_channels, input_height, input_width);
    return forward(input_4d);
}

void CNN::backward(const ring0::Matrix& grad_output, float relevancy) {
    if (dense_layers.empty()) return;

    // 1. Backprop through Dense classification layers
    ring0::Matrix grad_dense = grad_output;
    for (int i = static_cast<int>(dense_layers.size()) - 1; i >= 0; --i) {
        grad_dense = dense_layers[i].backward(grad_dense, relevancy);
    }

    // 2. Un-flatten gradient to 4D tensor matching conv backbone output
    Tensor4D grad_conv = Tensor4D::from_matrix(grad_dense,
                                               last_conv_final_feature_map.channels,
                                               last_conv_final_feature_map.height,
                                               last_conv_final_feature_map.width);

    // 3. Backprop through Convolutional and Pooling layers in reverse
    int pool_idx = static_cast<int>(pool_layers.size()) - 1;
    for (int i = static_cast<int>(conv_layers.size()) - 1; i >= 0; --i) {
        if (has_pooling[i] && pool_idx >= 0) {
            grad_conv = pool_layers[pool_idx].backward(grad_conv, relevancy);
            pool_idx--;
        }
        grad_conv = conv_layers[i].backward(grad_conv, relevancy);
    }
}

void CNN::reset_gradients() {
    for (auto& conv : conv_layers) {
        conv.reset_gradients();
    }
    for (auto& dense : dense_layers) {
        dense.reset_gradients();
    }
}

bool CNN::expand_conv_filters(size_t layer_idx, size_t additional_filters) {
    if (layer_idx >= conv_layers.size() || additional_filters == 0) return false;

    conv_layers[layer_idx].expand_filters(additional_filters);

    if (layer_idx + 1 < conv_layers.size()) {
        conv_layers[layer_idx + 1].expand_input_channels(additional_filters);
    } else if (!dense_layers.empty()) {
        size_t fc, fh, fw;
        get_flattened_dim(fc, fh, fw);
        size_t new_in_dim = fc * fh * fw;
        size_t current_in = dense_layers[0].in_features;
        if (new_in_dim > current_in) {
            dense_layers[0].expand_input_dim(new_in_dim - current_in);
        }
    }
    return true;
}

bool CNN::expand_dense_neurons(size_t layer_idx, size_t additional_neurons) {
    if (layer_idx >= dense_layers.size() || additional_neurons == 0) return false;
    dense_layers[layer_idx].expand_output_dim(additional_neurons);
    if (layer_idx + 1 < dense_layers.size()) {
        dense_layers[layer_idx + 1].expand_input_dim(additional_neurons);
    }
    return true;
}

void CNN::expand_capacity(float growth_factor) {
    if (growth_factor <= 1.0f) return;

    for (size_t i = 0; i < conv_layers.size(); ++i) {
        size_t added = std::max<size_t>(1, static_cast<size_t>(conv_layers[i].out_channels * (growth_factor - 1.0f)));
        expand_conv_filters(i, added);
    }

    for (size_t j = 0; j + 1 < dense_layers.size(); ++j) {
        size_t added = std::max<size_t>(2, static_cast<size_t>(dense_layers[j].out_features * (growth_factor - 1.0f)));
        expand_dense_neurons(j, added);
    }
}

size_t CNN::get_total_parameters() const {
    size_t total = 0;
    for (const auto& conv : conv_layers) {
        total += conv.get_parameter_count();
    }
    for (const auto& dense : dense_layers) {
        total += dense.weights.rows * dense.weights.cols + dense.biases.cols;
    }
    return total;
}

void CNN::print_architecture() const {
    std::cout << "=========================================================\n";
    std::cout << "               CNN ARCHITECTURE SUMMARY                 \n";
    std::cout << "=========================================================\n";
    std::cout << "Input Dimensions: (" << input_channels << "x" << input_height << "x" << input_width << ")\n";
    std::cout << "---------------------------------------------------------\n";

    size_t pool_idx = 0;
    for (size_t i = 0; i < conv_layers.size(); ++i) {
        const auto& c = conv_layers[i];
        std::cout << " Layer " << i << " [Conv2D]    : " << c.in_channels << " -> " << c.out_channels
                  << " (" << c.kernel_h << "x" << c.kernel_w << ", s=" << c.stride_h << ", p=" << c.pad_h
                  << ") | Params: " << c.get_parameter_count() << "\n";
        if (has_pooling[i] && pool_idx < pool_layers.size()) {
            std::cout << "         [MaxPool2D] : 2x2, stride=2\n";
            pool_idx++;
        }
    }

    size_t fc, fh, fw;
    get_flattened_dim(fc, fh, fw);
    std::cout << " Bridge  [Flatten]   : (" << fc << "x" << fh << "x" << fw << ") -> " << (fc * fh * fw) << " features\n";

    for (size_t j = 0; j < dense_layers.size(); ++j) {
        const auto& d = dense_layers[j];
        std::cout << " Dense " << j << "  [DenseFC]   : " << d.in_features << " -> " << d.out_features
                  << " | Params: " << (d.weights.rows * d.weights.cols + d.biases.cols) << "\n";
    }

    std::cout << "---------------------------------------------------------\n";
    std::cout << "Total Parameters: " << get_total_parameters() << "\n";
    std::cout << "=========================================================\n\n";
}

} // namespace ring3
