#include "ring0/loss.hpp"
#include <cmath>
#include <algorithm>

using namespace std;

namespace ring0
{

    // Global definition of loss history storage
    vector<float> Loss::loss_history;

    void Loss::record_loss(float loss_val)
    {
        loss_history.push_back(loss_val);
    }

    float Loss::get_min_loss()
    {
        if (loss_history.empty())
            return 0.0f;
        return *min_element(loss_history.begin(), loss_history.end());
    }

    float Loss::get_latest_loss()
    {
        if (loss_history.empty())
            return 0.0f;
        return loss_history.back();
    }

    float Loss::get_loss_gap(float multiplier)
    {
        if (loss_history.empty())
            return 0.0f;
        float current = get_latest_loss();
        float min_l = get_min_loss();
        return max(0.0f, current - min_l) * multiplier;
    }

    // Division-based scaling: when loss is high or uncertain, damp gradient scale smoothly via division
    // to guarantee numerical stability and prevent gradient surges.
    float Loss::compute_loss_scale_multiplier(float loss)
    {
        if (std::isnan(loss) || std::isinf(loss) || loss > 15.0f)
        {
            return 0.5f;
        }
        // Smoothly divide when loss is elevated: 1.0 / (1.0 + 0.15 * max(0, loss - 2.5))
        float excess = std::max(0.0f, loss - 2.5f);
        float multiplier = 1.0f / (1.0f + 0.15f * excess);
        return std::clamp(multiplier, 0.45f, 1.0f);
    }

    void Loss::clear_history()
    {
        loss_history.clear();
    }

    float Loss::mse(const Matrix &predictions, const Matrix &targets)
    {
        if (predictions.rows == 0 || predictions.cols == 0)
            return 0.0f;

        double sum = 0.0;
        for (size_t i = 0; i < predictions.data.size(); ++i)
        {
            double delta = static_cast<double>(predictions.data[i] - targets.data[i]);
            sum += delta * delta;
        }
        return static_cast<float>(sum / static_cast<double>(predictions.data.size()));
    }

    Matrix Loss::mse_gradient(const Matrix &predictions, const Matrix &targets)
    {
        Matrix gradient(predictions.rows, predictions.cols);
        float scale = 2.0f / static_cast<float>(max<size_t>(1, predictions.data.size()));
        for (size_t i = 0; i < gradient.data.size(); ++i)
            gradient.data[i] = scale * (predictions.data[i] - targets.data[i]);
        return gradient;
    }

    float Loss::cross_entropy(const Matrix &predictions, const Matrix &targets)
    {
        if (predictions.rows == 0 || predictions.cols == 0)
            return 0.0f;

        double total = 0.0;
        for (size_t row = 0; row < predictions.rows; ++row)
        {
            float row_max = predictions(row, 0);
            for (size_t col = 1; col < predictions.cols; ++col)
                row_max = max(row_max, predictions(row, col));

            double exp_sum = 0.0;
            for (size_t col = 0; col < predictions.cols; ++col)
                exp_sum += exp(static_cast<double>(predictions(row, col) - row_max));

            for (size_t col = 0; col < predictions.cols; ++col)
            {
                float target = targets(row, col);
                if (target > 0.0f)
                {
                    double probability = exp(static_cast<double>(predictions(row, col) - row_max)) / exp_sum;
                    total -= static_cast<double>(target) * log(max(1e-12, probability));
                }
            }
        }
        return static_cast<float>(total / static_cast<double>(predictions.rows));
    }

    Matrix Loss::cross_entropy_gradient(const Matrix &predictions, const Matrix &targets)
    {
        Matrix gradient(predictions.rows, predictions.cols);
        for (size_t row = 0; row < predictions.rows; ++row)
        {
            float row_max = predictions(row, 0);
            for (size_t col = 1; col < predictions.cols; ++col)
                row_max = max(row_max, predictions(row, col));

            double exp_sum = 0.0;
            for (size_t col = 0; col < predictions.cols; ++col)
                exp_sum += exp(static_cast<double>(predictions(row, col) - row_max));

            for (size_t col = 0; col < predictions.cols; ++col)
            {
                float probability = static_cast<float>(exp(static_cast<double>(predictions(row, col) - row_max)) / exp_sum);
                gradient(row, col) = (probability - targets(row, col)) / static_cast<float>(predictions.rows);
            }
        }
        return gradient;
    }

    float Loss::compute(LossType type, const Matrix &predictions, const Matrix &targets)
    {
        return type == LossType::MSE ? mse(predictions, targets) : cross_entropy(predictions, targets);
    }

    Matrix Loss::gradient(LossType type, const Matrix &predictions, const Matrix &targets)
    {
        return type == LossType::MSE ? mse_gradient(predictions, targets) : cross_entropy_gradient(predictions, targets);
    }

    // Tuned pyramid: milder clamping + better curvature response
    void LossDerivativePyramid::build(const vector<float> &raw_losses,
                                      size_t max_iterations,
                                      float clip_threshold,
                                      float magnitude_threshold)
    {
        layers.clear();
        stats.clear();

        if (raw_losses.empty())
            return;

        layers.push_back(raw_losses);

        auto compute_layer_stats = [](const vector<float> &vec) -> DerivativeLayerStats
        {
            DerivativeLayerStats s;
            s.size = vec.size();
            if (vec.empty())
                return s;

            s.min_val = vec[0];
            s.max_val = vec[0];
            double sum = 0.0;

            for (float v : vec)
            {
                s.min_val = min(s.min_val, v);
                s.max_val = max(s.max_val, v);
                sum += v;
            }

            s.mean_val = static_cast<float>(sum / vec.size());

            double var_sum = 0.0;
            for (float v : vec)
            {
                double diff = v - s.mean_val;
                var_sum += diff * diff;
            }
            s.variance = static_cast<float>(var_sum / max(size_t(1), vec.size()));
            return s;
        };

        stats.push_back(compute_layer_stats(raw_losses));

        // Milder defaults
        float theta = max(0.08f, magnitude_threshold);      // slightly lower linear zone
        float ceiling = max(theta + 0.25f, clip_threshold); // more headroom
        float headroom = ceiling - theta;

        for (size_t iter = 1; iter <= max_iterations; ++iter)
        {
            const auto &prev = layers.back();
            if (prev.size() < 2)
                break;

            vector<float> next_deriv(prev.size() - 1);
            for (size_t j = 0; j < prev.size() - 1; ++j)
            {
                float d = prev[j + 1] - prev[j];
                float abs_d = fabsf(d);

                float clamped_d;
                if (abs_d <= theta)
                {
                    clamped_d = d; // pure linear
                }
                else
                {
                    // Softer saturation
                    float excess = abs_d - theta;
                    float sgn = (d >= 0.0f) ? 1.0f : -1.0f;
                    // Use a gentler curve (divide by larger value)
                    clamped_d = sgn * (theta + headroom * tanhf(excess / (headroom * 1.4f)));
                }
                next_deriv[j] = clamped_d;
            }

            stats.push_back(compute_layer_stats(next_deriv));
            layers.push_back(move(next_deriv));
        }
    }

    // Tuned curvature scale — less extreme range, smoother response
    float LossDerivativePyramid::compute_curvature_scale() const
    {
        if (stats.size() < 2)
            return 1.0f;

        float first_deriv_var = stats[1].variance;
        float second_deriv_var = (stats.size() >= 3) ? stats[2].variance : 0.0f;

        // Less aggressive turbulence penalty
        float total_turb = first_deriv_var + second_deriv_var * 0.35f;
        float scale = 1.0f / (1.0f + min(1.5f, total_turb * 0.35f));

        // Narrower, safer band
        return clamp(scale * 1.05f, 0.75f, 1.25f); // was [0.65, 1.35]
    }

} // namespace ring0