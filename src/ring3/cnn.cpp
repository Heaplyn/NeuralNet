#include "ring3/cnn.hpp"
#include <iostream>
#include <iomanip>
#include <algorithm>

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

size_t CNN::get_current_feature_dim() const {
    if (!dense_layers.empty()) {
        return dense_layers.back().out_features;
    }
    if (use_neural_net_head && !dense_head.layers.empty()) {
        return dense_head.layers.back().out_features;
    }
    if (!recursive_layers.empty()) {
        return recursive_layers.back().out_features;
    }
    if (!transformer_blocks.empty()) {
        return transformer_blocks.back().embed_dim;
    }
    if (!attention_blocks.empty()) {
        return attention_blocks.back().embed_dim;
    }
    size_t fc, fh, fw;
    get_flattened_dim(fc, fh, fw);
    return fc * fh * fw;
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

void CNN::add_attention(size_t embed_dim, size_t num_heads, size_t num_kv_heads) {
    attention_blocks.emplace_back(embed_dim, num_heads, num_kv_heads);
}

void CNN::add_transformer_block(size_t embed_dim, size_t num_heads, size_t ffn_dim) {
    transformer_blocks.emplace_back(embed_dim, num_heads, ffn_dim);
}

void CNN::add_recursive_thought(const std::string& name, size_t out_dim, size_t depth) {
    size_t in_dim = get_current_feature_dim();
    recursive_layers.emplace_back(name, in_dim, out_dim, depth);
}

void CNN::add_recursive_layer(const ring1::RecursiveLayer& layer) {
    recursive_layers.push_back(layer);
}

void CNN::add_dense(size_t out_dim, ring0::ActivationType act) {
    size_t in_dim = get_current_feature_dim();
    dense_layers.emplace_back(in_dim, out_dim, act);
}

void CNN::add_dense_layer(const ring1::DenseLayer& layer) {
    dense_layers.push_back(layer);
}

void CNN::attach_neural_net(const ring2::NeuralNet& head_net) {
    dense_head = head_net;
    use_neural_net_head = true;
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
    ring0::Matrix feat = last_flattened_features;

    // 1. Ring 1 Attention blocks
    for (auto& attn : attention_blocks) {
        ring0::Tensor3D t_in(feat.rows, 1, feat.cols);
        for (size_t b = 0; b < feat.rows; ++b) {
            for (size_t c = 0; c < feat.cols; ++c) {
                t_in(b, 0, c) = feat(b, c);
            }
        }
        ring0::Tensor3D t_out = attn.forward(t_in);
        for (size_t b = 0; b < feat.rows; ++b) {
            for (size_t c = 0; c < feat.cols; ++c) {
                feat(b, c) = t_out(b, 0, c);
            }
        }
    }
    last_post_attention_features = feat;

    // 2. Ring 1 Transformer blocks
    for (auto& tb : transformer_blocks) {
        ring0::Tensor3D t_in(feat.rows, 1, feat.cols);
        for (size_t b = 0; b < feat.rows; ++b) {
            for (size_t c = 0; c < feat.cols; ++c) {
                t_in(b, 0, c) = feat(b, c);
            }
        }
        ring0::Tensor3D t_out = tb.forward(t_in);
        for (size_t b = 0; b < feat.rows; ++b) {
            for (size_t c = 0; c < feat.cols; ++c) {
                feat(b, c) = t_out(b, 0, c);
            }
        }
    }
    last_post_transformer_features = feat;

    // 3. Ring 1 Recursive Thought reasoning layers
    for (auto& rec : recursive_layers) {
        feat = rec.forward(feat);
    }
    last_post_recursive_features = feat;

    // 4. Ring 1 / Ring 2 Dense classification heads
    if (use_neural_net_head) {
        feat = dense_head.forward(feat);
    } else {
        for (auto& dense : dense_layers) {
            feat = dense.forward(feat);
        }
    }

    return feat;
}

ring0::Matrix CNN::forward_flat(const ring0::Matrix& flat_images) {
    Tensor4D input_4d = Tensor4D::from_matrix(flat_images, input_channels, input_height, input_width);
    return forward(input_4d);
}

void CNN::backward(const ring0::Matrix& grad_output, float relevancy) {
    ring0::Matrix grad = grad_output;

    // 1. Backprop through Dense classification layers / NeuralNet head
    if (use_neural_net_head) {
        grad = dense_head.backward(grad, relevancy);
    } else {
        for (int i = static_cast<int>(dense_layers.size()) - 1; i >= 0; --i) {
            grad = dense_layers[i].backward(grad, relevancy);
        }
    }

    // 2. Backprop through Recursive Thought reasoning layers
    for (int i = static_cast<int>(recursive_layers.size()) - 1; i >= 0; --i) {
        grad = recursive_layers[i].backward(grad, relevancy);
    }

    // 3. Backprop through Transformer Decoder blocks
    for (int i = static_cast<int>(transformer_blocks.size()) - 1; i >= 0; --i) {
        ring0::Tensor3D t_grad(grad.rows, 1, grad.cols);
        for (size_t b = 0; b < grad.rows; ++b) {
            for (size_t c = 0; c < grad.cols; ++c) {
                t_grad(b, 0, c) = grad(b, c);
            }
        }
        ring0::Tensor3D t_in_grad = transformer_blocks[i].backward(t_grad);
        for (size_t b = 0; b < grad.rows; ++b) {
            for (size_t c = 0; c < grad.cols; ++c) {
                grad(b, c) = t_in_grad(b, 0, c);
            }
        }
    }

    // 4. Backprop through Attention blocks
    for (int i = static_cast<int>(attention_blocks.size()) - 1; i >= 0; --i) {
        ring0::Tensor3D t_grad(grad.rows, 1, grad.cols);
        for (size_t b = 0; b < grad.rows; ++b) {
            for (size_t c = 0; c < grad.cols; ++c) {
                t_grad(b, 0, c) = grad(b, c);
            }
        }
        ring0::Tensor3D t_in_grad = attention_blocks[i].backward(t_grad);
        for (size_t b = 0; b < grad.rows; ++b) {
            for (size_t c = 0; c < grad.cols; ++c) {
                grad(b, c) = t_in_grad(b, 0, c);
            }
        }
    }

    // 5. Un-flatten gradient to 4D tensor matching conv backbone output
    if (!conv_layers.empty() && last_conv_final_feature_map.channels > 0) {
        Tensor4D grad_conv = Tensor4D::from_matrix(grad,
                                                   last_conv_final_feature_map.channels,
                                                   last_conv_final_feature_map.height,
                                                   last_conv_final_feature_map.width);

        int pool_idx = static_cast<int>(pool_layers.size()) - 1;
        for (int i = static_cast<int>(conv_layers.size()) - 1; i >= 0; --i) {
            if (has_pooling[i] && pool_idx >= 0) {
                grad_conv = pool_layers[pool_idx].backward(grad_conv, relevancy);
                pool_idx--;
            }
            grad_conv = conv_layers[i].backward(grad_conv, relevancy);
        }
    }
}

void CNN::reset_gradients() {
    for (auto& conv : conv_layers) {
        conv.reset_gradients();
    }
    for (auto& attn : attention_blocks) {
        attn.reset_gradients();
    }
    for (auto& tb : transformer_blocks) {
        tb.reset_gradients();
    }
    for (auto& rec : recursive_layers) {
        rec.reset_gradients();
    }
    if (use_neural_net_head) {
        dense_head.reset_gradients();
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
    } else {
        size_t fc, fh, fw;
        get_flattened_dim(fc, fh, fw);
        size_t new_in_dim = fc * fh * fw;
        if (!dense_layers.empty()) {
            size_t current_in = dense_layers[0].in_features;
            if (new_in_dim > current_in) {
                dense_layers[0].expand_input_dim(new_in_dim - current_in);
            }
        } else if (use_neural_net_head && !dense_head.layers.empty()) {
            size_t current_in = dense_head.layers[0].in_features;
            if (new_in_dim > current_in) {
                dense_head.layers[0].expand_input_dim(new_in_dim - current_in);
            }
        }
    }
    return true;
}

bool CNN::expand_dense_neurons(size_t layer_idx, size_t additional_neurons) {
    if (use_neural_net_head) {
        return dense_head.expand_hidden_layer(layer_idx, additional_neurons);
    }
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

    if (use_neural_net_head) {
        for (size_t j = 0; j + 1 < dense_head.layers.size(); ++j) {
            size_t added = std::max<size_t>(2, static_cast<size_t>(dense_head.layers[j].out_features * (growth_factor - 1.0f)));
            expand_dense_neurons(j, added);
        }
    } else {
        for (size_t j = 0; j + 1 < dense_layers.size(); ++j) {
            size_t added = std::max<size_t>(2, static_cast<size_t>(dense_layers[j].out_features * (growth_factor - 1.0f)));
            expand_dense_neurons(j, added);
        }
    }
}

size_t CNN::get_total_parameters() const {
    size_t total = 0;
    for (const auto& conv : conv_layers) {
        total += conv.get_parameter_count();
    }
    for (const auto& attn : attention_blocks) {
        total += attn.W_q.data.size() + attn.b_q.data.size();
        total += attn.W_k.data.size() + attn.b_k.data.size();
        total += attn.W_v.data.size() + attn.b_v.data.size();
        total += attn.W_o.data.size() + attn.b_o.data.size();
    }
    for (const auto& tb : transformer_blocks) {
        total += tb.attention.W_q.data.size() + tb.attention.b_q.data.size();
        total += tb.attention.W_k.data.size() + tb.attention.b_k.data.size();
        total += tb.attention.W_v.data.size() + tb.attention.b_v.data.size();
        total += tb.attention.W_o.data.size() + tb.attention.b_o.data.size();
        total += tb.ln1_gamma.data.size() + tb.ln1_beta.data.size();
        total += tb.ln2_gamma.data.size() + tb.ln2_beta.data.size();
        total += tb.W_gate.data.size() + tb.b_gate.data.size();
        total += tb.W_up.data.size() + tb.b_up.data.size();
        total += tb.W_down.data.size() + tb.b_down.data.size();
    }
    for (const auto& rec : recursive_layers) {
        total += rec.W_think.data.size() + rec.b_think.data.size() + rec.W_context.data.size();
    }
    if (use_neural_net_head) {
        total += dense_head.get_total_parameters();
    } else {
        for (const auto& dense : dense_layers) {
            total += dense.weights.rows * dense.weights.cols + dense.biases.cols;
        }
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

    for (size_t a = 0; a < attention_blocks.size(); ++a) {
        std::cout << " Attn    [GQA RoPE]  : " << attention_blocks[a].embed_dim
                  << " (heads=" << attention_blocks[a].num_heads << ")\n";
    }

    for (size_t t = 0; t < transformer_blocks.size(); ++t) {
        std::cout << " Xformer [Decoder]   : " << transformer_blocks[t].embed_dim
                  << " (SwiGLU FFN=" << transformer_blocks[t].ffn_dim << ")\n";
    }

    for (size_t r = 0; r < recursive_layers.size(); ++r) {
        std::cout << " Thought [Recursive] : " << recursive_layers[r].name
                  << " (" << recursive_layers[r].in_features << " -> " << recursive_layers[r].out_features
                  << ", depth=" << recursive_layers[r].thinking_depth << ")\n";
    }

    if (use_neural_net_head) {
        std::cout << " Head    [Ring2 Net] : " << dense_head.get_num_layers() << " dense layers\n";
        for (size_t j = 0; j < dense_head.layers.size(); ++j) {
            const auto& d = dense_head.layers[j];
            std::cout << "   - Head Dense " << j << ": " << d.in_features << " -> " << d.out_features << "\n";
        }
    } else {
        for (size_t j = 0; j < dense_layers.size(); ++j) {
            const auto& d = dense_layers[j];
            std::cout << " Dense " << j << "  [DenseFC]   : " << d.in_features << " -> " << d.out_features
                      << " | Params: " << (d.weights.rows * d.weights.cols + d.biases.cols) << "\n";
        }
    }

    std::cout << "---------------------------------------------------------\n";
    std::cout << "Total Parameters: " << get_total_parameters() << "\n";
    std::cout << "=========================================================\n\n";
}

} // namespace ring3
