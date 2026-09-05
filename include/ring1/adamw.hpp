#pragma once

/**
 * @file adamw.hpp
 * @brief AdamW optimizer with layer-wise directional weight shift credit attribution and loss-derivative feedback in Ring 1.
 */

#include "ring0/tensor.hpp"
#include "ring1/multi_formula_optimizer.hpp"
#include <vector>
#include <cmath>

using namespace std;

namespace ring1
{

    /**
     * @struct AdamWConfig
     * @brief Hyperparameters for the AdamW optimizer.
     */
    struct AdamWConfig
    {
        float lr = 0.001f;                ///< Base learning rate step size (alpha)
        float beta1 = 0.9f;               ///< Exponential decay rate for first moment estimates
        float beta2 = 0.99f;              ///< Exponential decay rate for second moment estimates (Transformer standard)
        float eps = 1e-8f;                ///< Epsilon term preventing division by zero
        float weight_decay = 0.01f;       ///< Decoupled weight decay coefficient (lambda)
        float loss_gap_factor = 0.6f;     ///< Multiplier for (currentLoss - minLoss) learning rate adjustment
        bool use_nesterov = true;         ///< Concept 1: Nesterov Accelerated Gradient lookahead in phase space
        bool use_natural_grad = true;     ///< Concept 3: Riemannian Natural Gradient preconditioning (Fisher metric)
        float curvature_scale = 1.0f;     ///< Concept 2: Rayleigh quotient curvature dampener/booster
        bool enable_multi_formula = true; ///< Dynamic 1-4 Formula Weight Physics
        float max_step = 0.35f;           ///< Per-element step trust region (|delta_w| cap); set per-step from loss
    };

    /**
     * @brief Loss-adaptive per-element step trust region.
     *
     * High loss (early / unstable) -> tight cap so no single update can detonate.
     * Low loss (converging)        -> looser cap, safe because gradients are small
     *                                 and it lets fine-tuning keep making progress.
     * Returns |delta_w| bound in [0.12, 0.6].
     */
    inline float trust_region_for_loss(float loss)
    {
        if (loss >= 6.0f)
            return 0.12f;
        if (loss >= 3.0f)
            return 0.12f + (0.35f - 0.12f) * (6.0f - loss) / 3.0f; // 6->0.12, 3->0.35
        if (loss >= 1.0f)
            return 0.35f + (0.60f - 0.35f) * (3.0f - loss) / 2.0f; // 3->0.35, 1->0.60
        return 0.60f;
    }

    /**
     * @class AdamW
     * @brief Adam optimizer with decoupled weight decay, Nesterov phase-space lookahead, Fisher metric, and layer attribution.
     */
    class AdamW
    {
    public:
        AdamWConfig config;
        size_t timestep; ///< Current optimization step index (t)

        vector<ring0::Matrix> m_list;      ///< 1st moment vectors (mean of gradients)
        vector<ring0::Matrix> v_list;      ///< 2nd moment vectors (uncentered variance of gradients)
        vector<ring0::Matrix> fisher_diag; ///< Empirical diagonal Fisher Information metric tensor F_ii
        vector<bool> decay_mask;           ///< Whether weight decay is applied to each registered parameter

        // --- Layer-Wise Weight Shift Credit Attribution & Directional Tracking ---
        vector<ring0::Matrix> last_shifts; ///< Recorded weight shifts Delta W from the previous step
        vector<float> layer_scales;        ///< Dynamic per-tensor learning rate multipliers in [0.5, 2.0]
        vector<float> layer_attributions;  ///< Historical attribution scores per layer
        vector<float> layer_directions;    ///< Direction (+1.0 scaling up, -1.0 scaling down) applied per layer
        vector<float> last_layer_loss_deltas; ///< Observed effect on loss (Delta L) for each layer operation

        // --- Penalization Derivative Tracking & Auto-Adjustment ---
        float penalty_factor = 0.09f;             ///< Active penalization multiplier
        float last_penalty_applied = 0.0f;        ///< Penalty magnitude at previous step
        float last_penalty_direction = 1.0f;      ///< Direction of last penalty shift (+1.0 increased, -1.0 decreased)
        float last_penalty_loss_delta = 0.0f;     ///< Observed effect on loss from last penalty operation
        float last_loss_observed = 0.0f;          ///< Previous loss observation
        float d_loss_d_penalty = 0.0f;            ///< Instantaneous empirical 1st derivative d(Loss) / d(Penalty)
        float last_d_loss_d_penalty = 0.0f;       ///< Previous 1st derivative for 2nd-order finite difference
        float ema_d_loss_d_penalty = 0.0f;        ///< Smoothed 1st derivative of penalization impact on loss
        float d2_loss_d_penalty2 = 0.0f;          ///< Empirical 2nd derivative d^2(Loss) / d(Penalty)^2
        float ema_d2_loss_d_penalty2 = 0.0f;      ///< Smoothed 2nd derivative (curvature)
        float taylor_penalty_prediction = 0.0f;   ///< Predicted Taylor optimal penalty update Delta pen_Taylor
        float taylor_penalty_confidence = 0.0f;   ///< Confidence score C in [0, 1] for Taylor prediction
        size_t penalty_observation_count = 0;     ///< Number of sequential observations for warmup gating

        // --- Multi-Formula Physics Statistics ---
        FormulaDistributionStats last_formula_stats;

        explicit AdamW(AdamWConfig cfg = {});

        /// Registers a new parameter tensor to track its moments, with optional weight decay toggle
        void register_param(const ring0::Matrix &param, bool apply_weight_decay = true);

        /// Updates a parameter tensor with incoming gradient matrix and records the shift
        void update_param(size_t index, ring0::Matrix &param, const ring0::Matrix &grad);

        /// Sets the base learning rate for the optimizer
        void set_learning_rate(float lr);

        /// Retrieves current active learning rate
        float get_learning_rate() const;

        /// Dynamically adjusts learning rate based on loss gap
        void adjust_by_loss_gap(float current_loss, float min_loss, float multiplier = 0.6f);

        /// Self-adjusts learning rate, moment momentum (beta1), and decay in real-time based on loss trajectory
        void self_adjust_by_loss(float current_loss, float ema_loss, float min_loss);

        /**
         * @brief Modulates layer-wise learning rates based on whether previous weight shifts reduced or raised loss.
         * @param loss_delta Delta L = L_current - L_previous (negative = improvement, positive = spike).
         */
        void update_attribution_feedback(float loss_delta);

        /**
         * @brief Records the derivative of penalization impact on loss d(Loss)/d(Penalty)
         *        and automatically adjusts the penalty factor based on sensitivity feedback.
         */
        void update_penalization_derivative(float current_loss, float loss_delta);

        /// Resets timestep and moment buffers
        void reset();
    };

} // namespace ring1
