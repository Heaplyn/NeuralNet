#pragma once

/**
 * @file llm_trainer.hpp
 * @brief Training engine for Transformer Language Models with Multi-Order Loss Derivative Pyramid and Dynamic Context Extension in Ring 3.
 */

#include "ring0/loss.hpp"
#include "ring0/taylor_predictor.hpp"
#include "ring2/transformer_lm.hpp"
#include "ring1/adamw.hpp"
#include "ring1/meta_loss_optimizer.hpp"
#include "ring3/text_dataset.hpp"
#include <functional>

using namespace std;

namespace ring3 {

/**
 * @struct LLMTrainingConfig
 * @brief Configuration parameters for language model training runs.
 */
struct LLMTrainingConfig {
    size_t steps = 150;           ///< Total optimization iterations
    size_t batch_size = 8;        ///< Number of parallel context sequences (B)
    float learning_rate = 0.3f; ///< Peak / Base AdamW learning rate (max_lr)
    float min_learning_rate = 0.0000001f; ///< Minimum learning rate floor for cosine decay
    float warmup_ratio = 0.05f;   ///< Fraction of total steps devoted to linear warmup
    float weight_decay = 0.03f;   ///< Decoupled L2 regularization
    float max_grad_norm = 1.0f;   ///< Global parameter-gradient L2 clip (0 to disable)
    size_t log_interval = 15;     ///< Logging cadence

    // --- Loss-adaptive LR shrink & decay slowdown ---
    float loss_shrink_short_alpha = 0.05f; ///< EMA alpha for short-window loss
    float loss_shrink_long_alpha  = 0.01f; ///< EMA alpha for long-window loss
    float loss_shrink_floor       = 0.2f;  ///< Minimum LR multiplier
    size_t loss_shrink_warmup     = 30;    ///< Skip shrink until this many steps have run

    // --- Meta-Neural Loss & 4-Formula Optimization ---
    bool enable_meta_loss_opt = true;     ///< Online Meta-Neural Network loss & step optimizer
    bool enable_multi_formula_opt = true; ///< Dynamic 1-4 Formula Weight Physics

    // --- Progressive Context & Token Window Extension ---
    bool progressive_context_growth = true; ///< Automatically expand token window based on steps or loss
    bool step_based_context_growth = true;  ///< When true, automatically expands token window at step milestones
    size_t initial_seq_len = 32;            ///< Starting context sequence length (T_0)
    size_t max_seq_len = 2048;              ///< Maximum expanded context sequence length (T_max)

    // --- Progressive Depth Ramping ---
    bool progressive_depth_growth = true;   ///< Automatically ramps model depth (active blocks) as steps progress
    size_t initial_layers = 4;              ///< Starting active layer depth

    // --- Progressive Dataset Horizon Expansion (Loss-Adaptive Curriculum) ---
    bool progressive_dataset_growth = true; ///< Start on a small focused data slice and expand as loss drops
    float initial_dataset_ratio = 0.05f;    ///< Fraction of corpus to train on initially (e.g. 5%)

    // --- Advanced Calculus & Curvature Engine ---
    bool use_armijo_line_search = true;     ///< Concept 5: Armijo-Goldstein condition backtracking
    bool use_curvature_scaling = true;      ///< Concept 2: Second-order Rayleigh quotient curvature
    bool use_data_filter = true;            ///< Feature 6: Auto-learning information entropy filter

    // --- Periodic "text challenge" evaluation ---
    size_t eval_interval = 100;   ///< Fire the on_eval callback every N steps (0 disables)
};

/**
 * @struct LLMStepMetrics
 * @brief Performance statistics and accuracy metrics recorded per training step.
 */
struct LLMStepMetrics {
    size_t step;               ///< Current iteration index
    float loss;                ///< Cross-entropy / Focal loss across all sequence tokens
    float perplexity;          ///< Language model perplexity PPL = exp(loss)
    float top1_accuracy;       ///< Exact match percentage (Rank 1) [0% - 100%]
    float top20_accuracy;      ///< Top-20 candidate match percentage [0% - 100%]
    float rank_score_top20;    ///< Index-weakened rank accuracy score [0% - 100%]
    float learning_rate;       ///< Active dynamic learning rate at this step
    size_t active_seq_len;     ///< Current active context token window length
    float penalty_factor;      ///< Active penalization strength factor
    float d_loss_d_penalty;    ///< Derivative of penalization impact on loss d(Loss)/d(Penalty)
    float meta_loss_scale;     ///< Meta-predicted loss scaling multiplier
    float meta_focal_gamma;    ///< Meta-predicted dynamic focal loss exponent
};

/**
 * @struct BenchmarkTelemetry
 * @brief Comprehensive real-time training telemetry and throughput benchmark statistics.
 */
struct BenchmarkTelemetry {
    double elapsed_seconds = 0.0;
    double tokens_per_second = 0.0;
    double gflops_estimate = 0.0;
    double ms_per_step = 0.0;
    size_t total_tokens_processed = 0;
    float current_loss = 0.0f;
    float ema_loss = 0.0f;
    float perplexity = 0.0f;
    float top1_accuracy = 0.0f;
    float top20_accuracy = 0.0f;
    float rank_score = 0.0f;
    float learning_rate = 0.0f;
    size_t active_vocab_size = 0;
    size_t active_context_length = 0;
    size_t active_model_layers = 0;
    size_t total_parameters = 0;
    ring1::FormulaDistributionStats formula_stats;
};

/**
 * @class LLMTrainer
 * @brief Orchestrates mini-batching, dynamic LR scheduling, loss derivative pyramid, backward pass, AdamW optimization, and Real-Time Telemetry Dashboard.
 */
class LLMTrainer {
public:
    ring2::TransformerLM& model;
    ring1::AdamW optimizer;
    ring1::MetaLossOptimizer meta_loss_opt;
    LLMTrainingConfig config;

    // --- Hierarchical Loss Derivative Pyramid Engine ---
    ring0::LossDerivativePyramid last_pyramid;

    // --- Taylor nth-order loss-trajectory forecaster (step-to-step foresight) ---
    ring0::TaylorTrajectoryPredictor loss_forecaster;
    ring0::TaylorTrajectory last_forecast; ///< Most recent forward loss-trajectory forecast

    // --- Loss-adaptive state (updated inside train_step) ---
    float ema_loss_short = 0.0f;
    float ema_loss_long  = 0.0f;
    float initial_loss   = 0.0f;
    bool  ema_initialized = false;
    size_t observed_steps = 0;
    size_t current_seq_len = 32;
    float dynamic_lr_gain = 1.0f; ///< Uncapped dynamic learning rate surge multiplier (max = inf)
    size_t total_tokens_trained = 0;

    /// Callback fired whenever parameter expansion (neurogenesis) occurs
    function<void()> on_param_expansion = nullptr;

    LLMTrainer(ring2::TransformerLM& lm, LLMTrainingConfig cfg = {});

    /// Computes dynamic learning rate for step t using Linear Warmup + Loss-Decay-Slowdown Cosine Annealing
    float compute_scheduled_lr(size_t step, size_t total_steps) const;

    /// Multiplier in [floor, 1] applied on top of the scheduled LR when the short-window
    /// loss EMA rises above the long-window one.
    float compute_loss_shrink() const;

    /// Evaluates average loss over validation batches without parameter updates
    float evaluate_loss(const TextDataset& dataset, size_t eval_batches = 4);

    /// Executes single forward, accuracy scoring, loss derivative pyramid, backward pass, and AdamW update
    LLMStepMetrics train_step(const TextBatch& batch, bool compute_detailed_metrics = true);

    /// Prints a comprehensive, formatted visual Benchmark & Telemetry Dashboard to the console
    void print_benchmark_dashboard(const BenchmarkTelemetry& tel, size_t current_step, size_t total_steps) const;

    /// Runs full training loop across the dataset with dynamic learning rate and optional progress callback.
    void train(TextDataset& dataset,
               const function<void(const LLMStepMetrics&)>& on_step = nullptr,
               const function<void(size_t)>& on_eval = nullptr);
};

} // namespace ring3
