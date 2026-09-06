#pragma once

/**
 * @file growth_controller.hpp
 * @brief Dynamic loss-guided network growth controller for Ring 2.
 */

#include "ring2/neural_net.hpp"
#include "ring0/taylor_predictor.hpp"
#include <deque>
#include <string>

using namespace std;

namespace ring2 {

/**
 * @struct GrowthConfig
 * @brief Parameters governing plateau detection and capacity expansion.
 */
struct GrowthConfig {
    float initial_growth_rate = 2.0f;       ///< Starting growth multiplier (gamma_0)
    float min_growth_rate = 0.01f;           ///< Minimum growth multiplier floor
    float max_growth_rate = 100.0f;           ///< Maximum growth multiplier ceiling
    float stagnation_threshold = 0.0000005f;    ///< Minimum expected relative improvement (0.5%)
    size_t patience = 15;                   ///< Window size for evaluating plateau
    size_t max_hidden_neurons = 1024;        ///< Hard safety cap on layer width (see forecast_uncaps_width)
    size_t base_neurons_to_add = 10;         ///< Base step size of neurons to add

    // --- Taylor-forecast-driven self-determination ---
    bool  use_taylor_forecast = true;        ///< Let the loss-trajectory forecast auto-size growth
    bool  forecast_uncaps_width = true;      ///< When confident, let the forecast raise width past max_hidden_neurons
    size_t forecast_hard_ceiling = 1000000;  ///< Absolute width ceiling (the "very little limit")
    float forecast_confidence_gate = 0.15f;  ///< Minimum forecast confidence to act predictively
};

/// Actions dispatched by the GrowthController
enum class GrowthAction {
    None,
    RateIncreased,
    RateDecreased,
    StructuralExpansionTriggered
};

/// Report generated each epoch/iteration
struct GrowthReport {
    GrowthAction action = GrowthAction::None;
    float current_loss = 0.0f;
    float loss_delta = 0.0f;
    float growth_rate = 1.0f;
    size_t neurons_added = 0;
    string message;

    // --- Taylor forecast telemetry ---
    float predicted_net_change = 0.0f;   ///< Forecast net loss change over the horizon
    float forecast_confidence = 0.0f;    ///< Confidence of the trajectory forecast [0,1]
    size_t dynamic_width_ceiling = 0;    ///< Forecast-determined active width ceiling this epoch
};

/**
 * @class GrowthController
 * @brief Monitors loss trajectory and dynamically tunes growth rates & triggers structural expansion.
 */
class GrowthController {
private:
    GrowthConfig config;
    float current_growth_rate;
    float cumulative_growth_pressure;
    deque<float> loss_history;
    size_t epochs_since_last_expansion;
    ring0::TaylorTrajectoryPredictor forecaster; ///< nth-order loss-trajectory predictor

public:
    explicit GrowthController(GrowthConfig cfg = {});

    /// Records loss for current epoch, analyzes dynamics, and optionally expands net
    GrowthReport record_epoch_loss(float loss, NeuralNet& net);

    float get_growth_rate() const { return current_growth_rate; }
    void set_growth_rate(float rate);
    void reset();
};

} // namespace ring2
