#pragma once

/**
 * @file optimizer.hpp
 * @brief Gradient descent optimizer with momentum, weight decay, gradient clipping, and loss-gap adaptation in Ring 1.
 */

#include "ring0/tensor.hpp"
#include "ring1/layer.hpp"
#include <vector>

using namespace std;

namespace ring1
{

    /**
     * @struct OptimizerConfig
     * @brief Hyperparameters for Gradient Descent optimizer.
     */
    struct OptimizerConfig
    {
        float learning_rate = 0.002f; ///< Base step size
        float momentum = 0.8f;        ///< Momentum decay coefficient beta
        float weight_decay = 1e-4f;   ///< L2 weight regularization
        float max_grad_norm = 6.0f;   ///< Maximum gradient norm for clipping
        float loss_gap_factor = 0.4f; ///< Multiplier for (currentLoss - minLoss) adjustment
    };

    /**
     * @class GradientDescent
     * @brief Stochastic Gradient Descent (SGD) with momentum and adaptive loss-gap scaling.
     */
    class GradientDescent
    {
    public:
        OptimizerConfig config;

        vector<ring0::Matrix> v_weights; ///< Momentum velocity buffers for weights
        vector<ring0::Matrix> v_biases;  ///< Momentum velocity buffers for biases

        explicit GradientDescent(OptimizerConfig cfg = {});

        /// Initializes velocity buffers matching network layer shapes
        void init(const vector<DenseLayer> &layers);

        /// Updates layer parameters and resets gradients
        void update(vector<DenseLayer> &layers);

        void set_learning_rate(float lr);
        float get_learning_rate() const;

        /**
         * @brief Dynamically adjusts learning rate: lr_effective = lr_base * (1.0 + (current - min) * multiplier)
         */
        void adjust_by_loss_gap(float current_loss, float min_loss, float multiplier = 0.6f);

        /// Automatically resizes velocity buffers when layers dynamically expand
        void sync_with_layers(const vector<DenseLayer> &layers);
    };

} // namespace ring1
