#pragma once

/**
 * @file trainer.hpp
 * @brief Training engine for feedforward neural networks with dynamic growth scheduling in Ring 3.
 */

#include "ring0/loss.hpp"
#include "ring1/optimizer.hpp"
#include "ring2/neural_net.hpp"
#include "ring2/growth_controller.hpp"
#include <functional>

using namespace std;

namespace ring3 {

/**
 * @struct TrainingConfig
 * @brief Settings for feedforward network training epochs, batches, and loss.
 */
struct TrainingConfig {
    size_t epochs = 100;                                      ///< Total number of training epochs
    size_t batch_size = 32;                                   ///< Mini-batch size
    ring0::LossType loss_type = ring0::LossType::CrossEntropy;///< Selected loss function
    bool enable_growth_controller = true;                     ///< Whether dynamic neuron addition is enabled
    size_t log_interval = 10;                                 ///< Logging frequency in epochs
};

/**
 * @struct EpochMetrics
 * @brief Evaluation stats recorded at the completion of an epoch.
 */
struct EpochMetrics {
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
class RingTrainer {
public:
    ring2::NeuralNet& net;
    ring1::GradientDescent& optimizer;
    ring2::GrowthController growth_controller;
    TrainingConfig config;

    RingTrainer(ring2::NeuralNet& network,
                ring1::GradientDescent& opt,
                const ring2::GrowthConfig& growth_cfg = {},
                TrainingConfig train_cfg = {});

    /// Trains the network on dataset (X, Y) with optional epoch callback
    void train(const ring0::Matrix& X,
               const ring0::Matrix& Y,
               const function<void(const EpochMetrics&)>& on_epoch_end = nullptr);

    /// Evaluates top-1 classification accuracy
    float evaluate_accuracy(const ring0::Matrix& X, const ring0::Matrix& Y);

    /// Returns predicted class index for a single input vector
    size_t predict_class(const ring0::Matrix& single_input);
};

} // namespace ring3
