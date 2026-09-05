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

    // ... keep mse / cross_entropy / gradients exactly as they are ...

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