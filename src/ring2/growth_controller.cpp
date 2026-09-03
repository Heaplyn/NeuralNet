#include "ring2/growth_controller.hpp"
#include <numeric>
#include <cmath>
#include <sstream>
#include <iomanip>
#include <algorithm>

using namespace std;

namespace ring2 {

GrowthController::GrowthController(GrowthConfig cfg)
    : config(cfg),
      current_growth_rate(cfg.initial_growth_rate),
      cumulative_growth_pressure(0.0f),
      epochs_since_last_expansion(0) {}

// Resets internal loss history and pressure accumulators
void GrowthController::reset() {
    current_growth_rate = config.initial_growth_rate;
    cumulative_growth_pressure = 0.0f;
    epochs_since_last_expansion = 0;
    loss_history.clear();
    forecaster.reset();
}

void GrowthController::set_growth_rate(float rate) {
    current_growth_rate = clamp(rate, config.min_growth_rate, config.max_growth_rate);
}

// Analyzes epoch loss dynamics, adjusts growth rate, and triggers structural expansion if needed
GrowthReport GrowthController::record_epoch_loss(float loss, NeuralNet& net) {
    GrowthReport report;
    report.current_loss = loss;
    report.growth_rate = current_growth_rate;
    report.action = GrowthAction::None;

    loss_history.push_back(loss);
    epochs_since_last_expansion++;

    // --- Taylor loss-trajectory forecast (self-determination signal) ---
    // Cost: O(n*K) scalar math on the loss history; touches no weights.
    float forecast_net = 0.0f;      // predicted net loss change over the horizon
    float forecast_conf = 0.0f;     // forecast confidence in [0,1]
    bool  forecast_stagnation = false; // trajectory predicts little/no further improvement
    // Dynamic, forecast-determined width ceiling. Starts at the hard safety cap
    // and, when the forecast is confident that we are stalling, is allowed to
    // grow toward forecast_hard_ceiling ("very little limits").
    size_t dyn_ceiling = config.max_hidden_neurons;
    if (config.use_taylor_forecast && loss_history.size() >= 3) {
        vector<float> hist(loss_history.begin(), loss_history.end());
        ring0::TaylorTrajectory tj = forecaster.observe(hist);
        if (tj.valid) {
            forecast_net = tj.predicted[tj.horizon - 1] - loss;
            forecast_conf = tj.confidence;
            // Stagnation = predicted improvement is weak or loss predicted to rise.
            forecast_stagnation = (forecast_net > -config.stagnation_threshold * max(1.0f, loss))
                                   && (forecast_conf >= config.forecast_confidence_gate);
            if (config.forecast_uncaps_width && forecast_conf >= config.forecast_confidence_gate) {
                // Capacity pressure grows with how stuck we are predicted to be.
                // stall in [0,1]: 0 = strong descent ahead, 1 = flat/rising forecast.
                float stall = clamp(0.5f + 0.5f * tanhf(forecast_net * 2.0f), 0.0f, 1.0f);
                float expansion = 1.0f + forecast_conf * stall * (current_growth_rate + 1.0f);
                double target = static_cast<double>(config.max_hidden_neurons) * expansion;
                dyn_ceiling = static_cast<size_t>(min<double>(target, static_cast<double>(config.forecast_hard_ceiling)));
            }
        }
    }
    report.predicted_net_change = forecast_net;
    report.forecast_confidence = forecast_conf;
    report.dynamic_width_ceiling = dyn_ceiling;

    // Wait until enough historical loss points are recorded
    if (loss_history.size() < config.patience) {
        report.message = "Collecting baseline loss statistics...";
        return report;
    }

    if (loss_history.size() > config.patience * 2) {
        loss_history.pop_front();
    }

    // 1. Calculate average loss in older half vs newer half
    size_t half = loss_history.size() / 2;
    float old_avg = 0.0f;
    for (size_t i = 0; i < half; ++i) old_avg += loss_history[i];
    old_avg /= half;

    float new_avg = 0.0f;
    for (size_t i = half; i < loss_history.size(); ++i) new_avg += loss_history[i];
    new_avg /= (half);

    // 2. Relative improvement: delta = (old - new) / old
    float relative_improvement = (old_avg - new_avg) / (old_avg + 1e-6f) * .5f;
    report.loss_delta = relative_improvement;

    // Predictive trigger: the Taylor forecast can flag an oncoming plateau a few
    // steps before the realized half-average test would, letting capacity arrive
    // just-in-time instead of after the model is already stuck.
    bool plateau = (relative_improvement < config.stagnation_threshold) || forecast_stagnation;

    if (plateau) {
        // Stagnation / Plateau: Accelerate growth rate
        float stagnation_factor = (config.stagnation_threshold - relative_improvement) / config.stagnation_threshold;
        // Fold in the forecast: a confident flat/rising forecast adds pressure
        // even if the realized delta has not yet crossed the threshold.
        stagnation_factor = max(stagnation_factor, forecast_conf * clamp(0.5f + 0.5f * tanhf(forecast_net * 2.0f), 0.0f, 1.0f));
        float rate_increment = 0.05f * (1.0f + stagnation_factor * 5.0f) * (1.0f + relative_improvement);
        current_growth_rate = min(config.max_growth_rate, current_growth_rate + rate_increment);
        cumulative_growth_pressure += current_growth_rate;

        report.action = GrowthAction::RateIncreased;
        report.growth_rate = current_growth_rate;

        // Check if cumulative growth pressure warrants adding hidden units
        if (cumulative_growth_pressure >= 15.0f && epochs_since_last_expansion >= config.patience) {
            // Auto-determine how many neurons to add from the forecast: the more
            // confident and severe the predicted stall, the larger the step.
            float forecast_boost = 1.0f + forecast_conf * clamp(0.5f + 0.5f * tanhf(forecast_net * 2.0f), 0.0f, 1.0f) * 3.0f;
            size_t neurons_to_add = static_cast<size_t>(ceil(config.base_neurons_to_add * current_growth_rate * forecast_boost));

            bool expanded = false;
            for (size_t i = 0; i < net.layers.size() - 1; ++i) {
                if (net.layers[i].out_features + neurons_to_add <= dyn_ceiling) {
                    net.expand_hidden_layer(i, neurons_to_add);
                    expanded = true;
                }
            }

            if (expanded) {
                report.action = GrowthAction::StructuralExpansionTriggered;
                report.neurons_added = neurons_to_add;
                cumulative_growth_pressure = 0.0f;
                epochs_since_last_expansion = 0;

                ostringstream ss;
                ss << "Loss plateau detected (ΔL/L: " << fixed << setprecision(4) << relative_improvement
                   << "). Growth rate adjusted to " << current_growth_rate
                   << ". Added " << neurons_to_add << " neurons to hidden layer(s).";
                report.message = ss.str();
                return report;
            }
        }

        ostringstream ss;
        ss << "Loss stagnating (ΔL/L: " << fixed << setprecision(4) << relative_improvement
           << "). Increased growth rate to " << current_growth_rate;
        report.message = ss.str();
    } else {
        cout << "Current loss: " << fixed << setprecision(4) << loss << ". ";
        // Steady convergence: Relax growth rate
        current_growth_rate = max(config.min_growth_rate, current_growth_rate * 0.9f * (1.0f + loss * 10.0f));
        cumulative_growth_pressure = max(0.0f, cumulative_growth_pressure * 0.5f * (1.0f + loss * 6.0f));

        report.action = GrowthAction::RateDecreased;
        report.growth_rate = current_growth_rate;

        ostringstream ss;
        ss << "Loss improving steadily (ΔL/L: " << fixed << setprecision(4) << relative_improvement
           << "). Stabilized growth rate to " << current_growth_rate;
        report.message = ss.str();
    }

    return report;
}

} // namespace ring2
