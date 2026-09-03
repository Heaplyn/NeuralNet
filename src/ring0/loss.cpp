#include "ring0/loss.hpp"
#include <cmath>
#include <algorithm>

using namespace std;

namespace ring0 {

// Global definition of loss history storage
vector<float> Loss::loss_history;

void Loss::record_loss(float loss_val) {
    loss_history.push_back(loss_val);
}

float Loss::get_min_loss() {
    if (loss_history.empty()) return 0.0f;
    float min_val = loss_history[0];
    for (float v : loss_history) {
        if (v < min_val) min_val = v;
    }
    return min_val;
}

float Loss::get_latest_loss() {
    if (loss_history.empty()) return 0.0f;
    return loss_history.back();
}

float Loss::get_loss_gap(float multiplier) {
    if (loss_history.empty()) return 0.0f;
    float current = get_latest_loss();
    float min_l = get_min_loss();
    return max(0.0f, current - min_l) * multiplier;
}

float Loss::compute_loss_scale_multiplier(float loss) {
    if (loss >= 5.0f) {
        return 3.0f;
    } else if (loss >= 4.0f) {
        // Interpolate between [4.0 -> 2.2x, 5.0 -> 3.0x]
        float t = (loss - 4.0f) / (5.0f - 4.0f);
        return 2.2f + t * (3.0f - 2.2f);
    } else if (loss >= 3.0f) {
        // Interpolate between [3.0 -> 1.6x, 4.0 -> 2.2x]
        float t = (loss - 3.0f) / (4.0f - 3.0f);
        return 1.6f + t * (2.2f - 1.6f);
    } else if (loss >= 2.0f) {
        // Interpolate between [2.0 -> 1.0x, 3.0 -> 1.6x]
        float t = (loss - 2.0f) / (3.0f - 2.0f);
        return 1.0f + t * (1.6f - 1.0f);
    } else if (loss >= 1.0f) {
        // Interpolate between [1.0 -> 0.3x, 2.0 -> 1.0x]
        float t = (loss - 1.0f) / (2.0f - 1.0f);
        return 0.3f + t * (1.0f - 0.3f);
    } else {
        // Loss < 1.0 -> 0.3x
        return 0.3f;
    }
}

void Loss::clear_history() {
    loss_history.clear();
}

// Dispatches scalar loss computation
float Loss::compute(LossType type, const Matrix& predictions, const Matrix& targets) {
    float result = 0.0f;
    if (type == LossType::MSE) {
        result = mse(predictions, targets);
    } else {
        result = cross_entropy(predictions, targets);
    }
    return result;
}

// Dispatches loss gradient computation
Matrix Loss::gradient(LossType type, const Matrix& predictions, const Matrix& targets) {
    if (type == LossType::MSE) {
        return mse_gradient(predictions, targets);
    } else {
        return cross_entropy_gradient(predictions, targets);
    }
}

// Mean Squared Error across batch
float Loss::mse(const Matrix& predictions, const Matrix& targets) {
    float total = 0.0f;
    for (size_t i = 0; i < predictions.data.size(); ++i) {
        float diff = predictions.data[i] - targets.data[i];
        total += diff * diff;
    }
    return total / static_cast<float>(predictions.rows);
}

// Gradient of Mean Squared Error: dL/dPred = 2 * (Pred - Target) / (N * Dim)
Matrix Loss::mse_gradient(const Matrix& predictions, const Matrix& targets) {
    Matrix grad(predictions.rows, predictions.cols);
    float scale = 2.0f / static_cast<float>(predictions.rows * predictions.cols);
    for (size_t i = 0; i < predictions.data.size(); ++i) {
        grad.data[i] = (predictions.data[i] - targets.data[i]) * scale;
    }
    return grad;
}

// Categorical Cross-Entropy Loss: L = -sum(target * log(pred + eps)) / N
float Loss::cross_entropy(const Matrix& predictions, const Matrix& targets) {
    const float eps = 1e-7f;
    float total = 0.0f;
    for (size_t r = 0; r < predictions.rows; ++r) {
        for (size_t c = 0; c < predictions.cols; ++c) {
            if (targets(r, c) > 0.0f) {
                float p = clamp(predictions(r, c), eps, 1.0f - eps);
                total -= targets(r, c) * log(p);
            }
        }
    }
    return total / static_cast<float>(predictions.rows);
}

// Combined Softmax + Cross-Entropy Gradient: dL/dZ = (predictions - targets) / N
Matrix Loss::cross_entropy_gradient(const Matrix& predictions, const Matrix& targets) {
    Matrix grad(predictions.rows, predictions.cols);
    float scale = 1.0f / static_cast<float>(predictions.rows);
    for (size_t i = 0; i < predictions.data.size(); ++i) {
        grad.data[i] = (predictions.data[i] - targets.data[i]) * scale;
    }
    return grad;
}

// Builds the multi-order loss derivative pyramid across iterative layers
void LossDerivativePyramid::build(const vector<float>& raw_losses, size_t max_iterations, float clip_threshold, float magnitude_threshold) {
    layers.clear();
    stats.clear();

    if (raw_losses.empty()) return;

    // Layer 0: Raw token loss values
    layers.push_back(raw_losses);

    auto compute_layer_stats = [](const vector<float>& vec) -> DerivativeLayerStats {
        DerivativeLayerStats s;
        s.size = vec.size();
        if (vec.empty()) return s;

        s.min_val = vec[0];
        s.max_val = vec[0];
        double sum = 0.0;

        for (float v : vec) {
            if (v < s.min_val) s.min_val = v;
            if (v > s.max_val) s.max_val = v;
            sum += v;
        }

        s.mean_val = static_cast<float>(sum / vec.size());

        double var_sum = 0.0;
        for (float v : vec) {
            double diff = v - s.mean_val;
            var_sum += diff * diff;
        }
        s.variance = static_cast<float>(var_sum / vec.size());
        return s;
    };

    stats.push_back(compute_layer_stats(raw_losses));

    // Headroom between linear magnitude threshold and maximum saturation ceiling
    float theta = max(0.1f, magnitude_threshold);
    float ceiling = max(theta + 0.1f, clip_threshold);
    float headroom = ceiling - theta;

    // Iteratively build higher-order derivative layers
    for (size_t iter = 1; iter <= max_iterations; ++iter) {
        const auto& prev = layers.back();
        if (prev.size() < 2) break;

        vector<float> next_deriv(prev.size() - 1);
        for (size_t j = 0; j < prev.size() - 1; ++j) {
            float d = prev[j + 1] - prev[j];
            float abs_d = fabsf(d);

            float clamped_d;
            if (abs_d <= theta) {
                // Within normal range: pass through with 100% linear gradient fidelity (tanh is completely inactive)
                clamped_d = d;
            } else {
                // Beyond threshold: tanh gradually soft-saturates the excess with C^1 continuous boundary transition
                float excess = abs_d - theta;
                float sgn = (d >= 0.0f) ? 1.0f : -1.0f;
                clamped_d = sgn * (theta + headroom * tanhf(excess / headroom));
            }
            next_deriv[j] = clamped_d;
        }

        stats.push_back(compute_layer_stats(next_deriv));
        layers.push_back(move(next_deriv));
    }
}

// Curvature scale modulation factor in [0.65, 1.35]
float LossDerivativePyramid::compute_curvature_scale() const {
    if (stats.size() < 2) return 1.0f;

    // Check 1st and 2nd order derivative variance
    float first_deriv_var = stats[1].variance;
    float second_deriv_var = (stats.size() >= 3) ? stats[2].variance : 0.0f;

    // High curvature turbulence -> dampen step size to stabilize
    float total_turb = first_deriv_var + second_deriv_var * 0.5f;
    float scale = 1.0f / (1.0f + min(2.0f, total_turb * 0.5f));

    return clamp(scale * 1.1f, 0.65f, 1.35f);
}

} // namespace ring0
