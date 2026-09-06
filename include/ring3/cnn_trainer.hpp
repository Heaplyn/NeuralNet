#pragma once

/**
 * @file cnn_trainer.hpp
 * @brief Advanced training orchestrator for Convolutional Neural Networks in Ring 3.
 * 
 * Features:
 * 1. nth-order Taylor Loss-Trajectory Foresight (ring0::TaylorTrajectoryPredictor)
 * 2. Online Meta-Neural LR & Loss Multiplier Modulation (ring1::MetaLossOptimizer)
 * 3. Episodic Past Mistake Memory, Geometric Repulsion & Dynamic Capacity Sizing
 * 4. Automatic Gradient Energy Normalization & Adaptive L2 Scaling
 */

#include "ring3/cnn.hpp"
#include "ring0/loss.hpp"
#include "ring0/taylor_predictor.hpp"
#include "ring1/adamw.hpp"
#include "ring1/meta_loss_optimizer.hpp"
#include <vector>
#include <functional>
#include <deque>

namespace ring3 {

/**
 * @struct CNNTrainingConfig
 * @brief Hyperparameters and physics switches for CNN optimization.
 */
struct CNNTrainingConfig {
    size_t epochs = 30;
    size_t batch_size = 32;
    float learning_rate = 0.001f;
    float weight_decay = 0.001f;
    ring0::LossType loss_type = ring0::LossType::CrossEntropy;

    // --- 1. Taylor Foresight ---
    bool enable_taylor_forecast = true;
    float taylor_forecast_weight = 0.5f;

    // --- 2. Meta-Neural LR Modulation ---
    bool enable_meta_loss_opt = true;

    // --- 3. Mistake Checkpoint Memory & Sizing Repulsion ---
    bool enable_mistake_memory = true;
    size_t mistake_memory_capacity = 30;
    float mistake_repulsion_scale = 0.25f;
    float mistake_spike_threshold = 1.25f; ///< Spike above EMA triggering mistake recording
    size_t mistake_streak_growth_trigger = 3; ///< Streak of mistake alignments triggering capacity growth

    // --- 4. Auto Gradient Normalization ---
    bool enable_auto_grad_norm = true;
    float target_grad_norm = 1.0f;
    float max_grad_norm_ceiling = 3.0f;
    float auto_grad_ema_alpha = 0.05f;

    // --- Dynamic Growth Controls ---
    bool enable_dynamic_growth = true;
    size_t growth_patience = 8;
    float growth_multiplier = 1.30f;
};

/**
 * @struct CNNMistakeRecord
 * @brief Snapshot of a failed or unstable optimization trajectory.
 */
struct CNNMistakeRecord {
    float loss = 0.0f;
    float baseline_ema = 0.0f;
    float grad_norm = 0.0f;
    size_t model_params = 0;
    size_t step = 0;
    std::vector<float> grad_signature; ///< Normalized global gradient unit vector
};

/**
 * @struct CNNEpochMetrics
 * @brief Real-time metrics emitted at the conclusion of each CNN training epoch.
 */
struct CNNEpochMetrics {
    size_t epoch = 0;
    float loss = 0.0f;
    float accuracy = 0.0f;
    float raw_grad_norm = 0.0f;
    float normalized_grad_norm = 0.0f;
    float applied_lr = 0.001f;
    float meta_lr_scale = 1.0f;
    float taylor_lr_scale = 1.0f;
    float taylor_confidence = 0.0f;
    size_t stored_mistakes = 0;
    float mistake_similarity = 0.0f;
    size_t total_parameters = 0;
    bool capacity_expanded = false;
};

/**
 * @class CNNTrainer
 * @brief Orchestrates mini-batch training for CNN models with Taylor foresight, Meta-LR, past mistake memory, and auto gradient normalization.
 */
class CNNTrainer {
public:
    CNN& model;
    CNNTrainingConfig config;

    ring1::AdamW optimizer;
    ring1::MetaLossOptimizer meta_optimizer;
    ring0::TaylorTrajectoryPredictor taylor_forecaster;

    std::deque<CNNMistakeRecord> mistake_memory;

    float last_raw_grad_norm = 0.0f;
    float last_norm_scale = 1.0f;
    float running_grad_norm_ema = 1.0f;
    float last_taylor_lr_scale = 1.0f;
    float last_meta_lr_scale = 1.0f;
    float last_taylor_confidence = 0.0f;
    float last_mistake_similarity = 0.0f;

private:
    float ema_loss = 0.0f;
    bool ema_initialized = false;
    size_t step_counter = 0;
    size_t plateau_streak = 0;
    size_t mistake_alignment_streak = 0;
    float best_loss = 1e9f;

    std::vector<float> extract_global_gradient_unit_vector(float& out_norm);
    void apply_mistake_repulsion(std::vector<float>& global_grad_vector);
    void distribute_global_gradient_vector(const std::vector<float>& global_grad_vector);
    float perform_auto_gradient_normalization();
    void update_model_parameters(float effective_lr);
    void rebuild_optimizer_params();

public:
    CNNTrainer(CNN& net, CNNTrainingConfig cfg = {});

    /// Trains the CNN on training data with optional epoch callback
    void train(const ring0::Matrix& X_train,
               const ring0::Matrix& Y_train,
               const std::function<void(const CNNEpochMetrics&)>& on_epoch_end = nullptr);

    /// Evaluates top-1 classification accuracy on validation images
    float evaluate_accuracy(const ring0::Matrix& X_test, const ring0::Matrix& Y_test);

    /// Predicts class index for single sample
    size_t predict_class(const ring0::Matrix& single_image);
};

} // namespace ring3
