#include <vector>
#include <cmath>
#include <memory>
#include <random>

namespace dl {

// --- 1. Autograd Computational Graph Node ---
struct Variable {
    std::vector<float> data;
    std::vector<float> grad;
    std::vector<std::shared_ptr<Variable>> parents;
    bool requires_grad = true;

    explicit Variable(size_t size, float init_val = 0.0f)
        : data(size, init_val), grad(size, 0.0f) {}

    void zero_grad() {
        std::fill(grad.begin(), grad.end(), 0.0f);
    }
};

// --- 2. 2D Convolution Forward Pass ---
std::vector<float> conv2d_forward(const std::vector<float>& input,
                                  const std::vector<float>& kernel,
                                  size_t in_h, size_t in_w,
                                  size_t k_h, size_t k_w) {
    size_t out_h = in_h - k_h + 1;
    size_t out_w = in_w - k_w + 1;
    std::vector<float> output(out_h * out_w, 0.0f);

    for (size_t oh = 0; oh < out_h; ++oh) {
        for (size_t ow = 0; ow < out_w; ++ow) {
            float sum = 0.0f;
            for (size_t kh = 0; kh < k_h; ++kh) {
                for (size_t kw = 0; kw < k_w; ++kw) {
                    size_t in_idx = (oh + kh) * in_w + (ow + kw);
                    size_t k_idx = kh * k_w + kw;
                    sum += input[in_idx] * kernel[k_idx];
                }
            }
            output[oh * out_w + ow] = sum;
        }
    }
    return output;
}

// --- 3. Batch Normalization ---
struct BatchNorm1D {
    size_t features;
    std::vector<float> gamma;
    std::vector<float> beta;
    std::vector<float> running_mean;
    std::vector<float> running_var;
    float momentum = 0.1f;
    float eps = 1e-5f;

    explicit BatchNorm1D(size_t num_features)
        : features(num_features),
          gamma(num_features, 1.0f),
          beta(num_features, 0.0f),
          running_mean(num_features, 0.0f),
          running_var(num_features, 1.0f) {}

    std::vector<float> forward(const std::vector<float>& x, size_t batch_size) {
        std::vector<float> out(batch_size * features);

        for (size_t f = 0; f < features; ++f) {
            float mean = 0.0f;
            for (size_t b = 0; b < batch_size; ++b) {
                mean += x[b * features + f];
            }
            mean /= batch_size;

            float var = 0.0f;
            for (size_t b = 0; b < batch_size; ++b) {
                float diff = x[b * features + f] - mean;
                var += diff * diff;
            }
            var /= batch_size;

            running_mean[f] = (1.0f - momentum) * running_mean[f] + momentum * mean;
            running_var[f] = (1.0f - momentum) * running_var[f] + momentum * var;

            float inv_std = 1.0f / std::sqrt(var + eps);
            for (size_t b = 0; b < batch_size; ++b) {
                float x_hat = (x[b * features + f] - mean) * inv_std;
                out[b * features + f] = gamma[f] * x_hat + beta[f];
            }
        }
        return out;
    }
};

// --- 4. Softmax with Temperature ---
std::vector<float> softmax_temp(const std::vector<float>& logits, float temperature) {
    float max_logit = -1e9f;
    for (float l : logits) {
        if (l > max_logit) max_logit = l;
    }

    float sum_exp = 0.0f;
    std::vector<float> probs(logits.size());
    for (size_t i = 0; i < logits.size(); ++i) {
        probs[i] = std::exp((logits[i] - max_logit) / temperature);
        sum_exp += probs[i];
    }

    for (float& p : probs) {
        p /= sum_exp;
    }
    return probs;
}

} // namespace dl
