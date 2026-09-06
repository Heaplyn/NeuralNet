#pragma once

/**
 * @file trainer.hpp
 * @brief Training engine for feedforward neural networks with dynamic growth scheduling in Ring 4.
 */

#include "ring0/loss.hpp"
#include "ring0/taylor_predictor.hpp"
#include "ring1/adamw.hpp"
#include "ring1/meta_loss_optimizer.hpp"
#include "ring1/optimizer.hpp"
#include "ring2/neural_net.hpp"
#include "ring2/growth_controller.hpp"
#include <functional>

using namespace std;

namespace ring4
{

    /**
     * @struct TrainingConfig
     * @brief Settings for feedforward network training epochs, batches, and loss.
     */
    struct TrainingConfig
    {
        size_t epochs = 2000;                                      ///< Total number of training epochs
        size_t batch_size = 32;                                    ///< Mini-batch size
        ring0::LossType loss_type = ring0::LossType::CrossEntropy; ///< Selected loss function
        bool enable_growth_controller = true;                      ///< Whether dynamic neuron addition is enabled
        size_t log_interval = 10;                                  ///< Logging frequency in epochs
        float learning_rate = 0.001f;                              ///< AdamW base learning rate
        float weight_decay = 0.01f;                                ///< AdamW decoupled weight decay
        float max_grad_norm = 1.0f;                                ///< Global gradient norm clip
        bool enable_meta_loss_opt = true;                          ///< Online meta-loss modulation
        bool enable_multi_formula_opt = true;                      ///< AdamW multi-formula updates
        bool enable_taylor_forecast = true;                        ///< Predictive loss-trajectory modulation
        float taylor_forecast_weight = 0.5f;                       ///< Blend weight for direct Taylor signals
    };

    /**
     * @struct EpochMetrics
     * @brief Evaluation stats recorded at the completion of an epoch.
     */
    struct EpochMetrics
    {
        size_t epoch;
        float loss;
        float accuracy;
        float growth_rate;
        ring2::GrowthReport growth_report;
    };

    /**
     * @class RingTrainer
     * @brief Executes mini-batch SGD training loops, calculates accuracy, and syncs with GrowthController.
     */
    class RingTrainer
    {
    public:
        ring2::NeuralNet &net;
        ring1::GradientDescent &optimizer;
        ring1::AdamW adaptive_optimizer;
        ring1::MetaLossOptimizer meta_loss_optimizer;
        ring0::TaylorTrajectoryPredictor loss_forecaster;
        ring2::GrowthController growth_controller;
        TrainingConfig config;
        float last_taylor_lr_scale = 1.0f;
        float last_taylor_curvature_scale = 1.0f;
        float last_meta_lr_scale = 1.0f;

    private:
        vector<float> loss_history;
        size_t registered_parameter_count = 0;
        float ema_loss = 0.0f;
        bool ema_initialized = false;

        void register_network_parameters(bool reset_optimizer_state);
        float compute_gradient_variance() const;
        float compute_gradient_norm() const;
        void apply_adaptive_update(float loss, float previous_loss);

    public:
        RingTrainer(ring2::NeuralNet &network,
                    ring1::GradientDescent &opt,
                    const ring2::GrowthConfig &growth_cfg = {},
                    TrainingConfig train_cfg = {});

        /// Trains the network on dataset (X, Y) with optional epoch callback
        void train(const ring0::Matrix &X,
                   const ring0::Matrix &Y,
                   const function<void(const EpochMetrics &)> &on_epoch_end = nullptr);

        /// Evaluates top-1 classification accuracy
        float evaluate_accuracy(const ring0::Matrix &X, const ring0::Matrix &Y);

        /// Returns predicted class index for a single input vector
        size_t predict_class(const ring0::Matrix &single_input);
    };

} // namespace ring4
