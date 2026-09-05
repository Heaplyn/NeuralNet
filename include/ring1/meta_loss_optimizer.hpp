#pragma once

/**
 * @file meta_loss_optimizer.hpp
 * @brief Meta-Neural Loss & Step Optimizer Network in Ring 1.
 *        Dynamically modulates loss landscapes, focal gamma, learning rate vectors,
 *        and curvature metrics using an online self-learning neural network.
 */

#include "ring0/tensor.hpp"
#include <vector>
#include <string>
#include <memory>

namespace ring1
{

    /**
     * @struct MetaLossTelemetry
     * @brief Observation state fed into the Meta-Neural Network.
     */
    struct MetaLossTelemetry
    {
        float current_loss = 5.0f;
        float delta_loss = 0.0f;
        float accel_loss = 0.0f;
        float d_loss_d_penalty = 0.0f;
        float gradient_variance = 0.01f;
        float layer_alignment = 0.85f;
        float token_entropy = 4.5f;
        float learning_rate = 0.02f;

        // --- Taylor loss-trajectory foresight (from ring0::TaylorTrajectoryPredictor) ---
        float predicted_delta = 0.0f;       ///< Predicted next-step loss change L_{t+1}-L_t
        float predicted_net = 0.0f;         ///< Predicted net change over the forecast horizon
        float trajectory_reward = 0.0f;     ///< Discounted foresight reward (positive = improvement ahead)
        float trajectory_confidence = 0.0f; ///< Forecast confidence in [0,1]
    };

    /**
     * @struct MetaOptimizationOutput
     * @brief Dynamic modulation signals computed by the Meta-Neural Network.
     */
    struct MetaOptimizationOutput
    {
        float loss_scale_multiplier = 1.0f; ///< Continuous loss scale multiplier [0.2, 4.0]
        float dynamic_focal_gamma = 1.0f;   ///< Dynamic focal loss exponent gamma [0.0, 3.0]
        float lr_step_modulator = 1.0f;     ///< Meta-predicted learning rate multiplier [0.5, 3.0]
        float curvature_scale = 1.0f;       ///< Curvature preconditioning factor [0.2, 2.5]
    };

    /**
     * @class MetaLossOptimizer
     * @brief Online Meta-Neural Network optimizing loss dynamics and convergence velocity.
     */
    class MetaLossOptimizer
    {
        /// Input feature dimension: 8 base telemetry + 4 Taylor foresight signals.
        static constexpr size_t META_INPUT_DIM = 12;

    private:
        // 3-Layer Meta MLP: 12 -> 32 -> 16 -> 4
        ring0::Matrix W1, b1;
        ring0::Matrix W2, b2;
        ring0::Matrix W3, b3;

        // Cache for backpropagation / policy updates
        std::vector<float> last_input;
        std::vector<float> last_h1;
        std::vector<float> last_h2;
        MetaOptimizationOutput last_output;

        float prev_loss = 0.0f;
        float prev_delta_loss = 0.0f;
        size_t step_count = 0;
        float meta_lr = 0.02f;

        /// Applies one policy-gradient-style update across ALL layers (W1,W2,W3) for a scalar reward.
        void apply_policy_gradient(float reward);

    public:
        explicit MetaLossOptimizer();

        /// Evaluates the meta-neural network with the current training telemetry
        MetaOptimizationOutput predict(const MetaLossTelemetry &telemetry);

        /// Updates the meta-neural network weights online based on convergence reward R = -(L_t - L_{t-1})
        void update_online(float current_loss);

        /**
         * @brief Foresight-augmented online update. Blends the realized reward
         *        -(L_t - L_{t-1}) with the Taylor-predicted trajectory reward so the
         *        meta-network is optimized for the whole predicted loss path, not just
         *        the last step. @param trajectory_reward from ring0::TaylorTrajectory.
         */
        void update_online(float current_loss, float trajectory_reward, float foresight_weight = 0.5f);

        /// Resets the meta-neural network internal state and caches
        void reset();

        /// Prints the current meta-neural telemetry and predicted modulation outputs
        void print_status() const;

        /// Retrieves the last computed outputs
        const MetaOptimizationOutput &get_last_output() const { return last_output; }
    };

} // namespace ring1
