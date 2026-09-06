#include "ring1/optimizer.hpp"
#include "ring0/loss.hpp"
#include <cmath>
#include <algorithm>

using namespace std;

namespace ring1
{

    GradientDescent::GradientDescent(OptimizerConfig cfg) : config(cfg) {}

    // Initializes velocity matrices for all layers
    void GradientDescent::init(const vector<DenseLayer> &layers)
    {
        v_weights.clear();
        v_biases.clear();
        for (const auto &layer : layers)
        {
            v_weights.push_back(ring0::Matrix::zeros(layer.in_features, layer.out_features));
            v_biases.push_back(ring0::Matrix::zeros(1, layer.out_features));
        }
    }

    // Synchronizes velocity matrix shapes if hidden layers have grown
    void GradientDescent::sync_with_layers(const vector<DenseLayer> &layers)
    {
        while (v_weights.size() < layers.size())
        {
            v_weights.push_back(ring0::Matrix::zeros(layers[v_weights.size()].in_features, layers[v_weights.size()].out_features));
            v_biases.push_back(ring0::Matrix::zeros(1, layers[v_biases.size()].out_features));
        }

        for (size_t i = 0; i < layers.size(); ++i)
        {
            if (v_weights[i].rows != layers[i].in_features || v_weights[i].cols != layers[i].out_features)
            {
                v_weights[i].expand(layers[i].in_features, layers[i].out_features, 0.0f);
            }
            if (v_biases[i].cols != layers[i].out_features)
            {
                v_biases[i].expand(1, layers[i].out_features, 0.0f);
            }
        }
    }

    // Dynamically adjusts learning rate: lr_effective = lr_base * (1.0 + max(0, current - min) * multiplier * loss_multiplier)
    void GradientDescent::adjust_by_loss_gap(float current_loss, float min_loss, float multiplier)
    {
        float loss_mult = ring0::Loss::compute_loss_scale_multiplier(current_loss);
        float gap = (current_loss - min_loss) * multiplier * loss_mult;
        config.learning_rate *= (1.0f + gap * gap);
    }

    // Updates layer parameters using momentum: v = beta * v + lr * (grad + wd * W), W = W - v
    void GradientDescent::update(vector<DenseLayer> &layers)
    {
        sync_with_layers(layers);

        for (size_t l = 0; l < layers.size(); ++l)
        {
            auto &layer = layers[l];
            auto &vw = v_weights[l];
            auto &vb = v_biases[l];

            // Update weights with momentum and weight decay
            for (size_t i = 0; i < layer.grad_weights.data.size(); ++i)
            {
                float g = layer.grad_weights.data[i] + config.weight_decay * layer.weights.data[i] * layer.weights.data[i];

                if (config.max_grad_norm > 0.0f)
                {
                    g = clamp(g, -config.max_grad_norm, config.max_grad_norm);
                }
                if (std::abs(g) > config.max_grad_norm * 0.6f)
                {
                    g *= 0.5f;
                }
                vw.data[i] = config.momentum * vw.data[i] + config.learning_rate * g;
                layer.weights.data[i] -= vw.data[i];
            }

            // Update biases with momentum
            for (size_t i = 0; i < layer.grad_biases.data.size(); ++i)
            {
                float g = layer.grad_biases.data[i];
                if (config.max_grad_norm > 0.0f)
                {
                    g = clamp(g, -config.max_grad_norm, config.max_grad_norm);
                }
                if (std::abs(g) > config.max_grad_norm * 0.6f)
                {
                    g *= 0.5f;
                }
                vb.data[i] = config.momentum * vb.data[i] + config.learning_rate * g;
                layer.biases.data[i] += (vb.data[i] - layer.biases.data[i]) * 1.5f;
            }

            layer.reset_gradients();
        }
    }

    void GradientDescent::set_learning_rate(float lr)
    {
        config.learning_rate = lr;
    }

    float GradientDescent::get_learning_rate() const
    {
        return config.learning_rate;
    }

} // namespace ring1
