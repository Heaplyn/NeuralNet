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
#include <deque>

using namespace std;

namespace ring3
{

    /**
     * @struct MistakeCheckpoint
     * @brief Snapshot of system state and triple-signature (gradient, latent, weight) captured during failure steps.
     */
    struct MistakeCheckpoint
    {
        float loss = 0.0f;
        float ema_loss = 0.0f;
        float grad_norm = 0.0f;
        float penalty = 0.0f;
        float meta_scale = 0.0f;
        float gain = 1.0f;
        size_t step = 0;
        std::vector<float> fingerprint;      ///< Level C: Compact model weight fingerprint
        std::vector<float> grad_signature;   ///< Level A: Normalized gradient direction signature
        std::vector<float> latent_signature; ///< Level B: Normalized representation/logit distribution signature
    };

    /**
     * @struct LLMTrainingConfig
     * @brief Configuration parameters for language model training runs.
     */
    struct LLMTrainingConfig
    {
        size_t steps = 150;                 ///< Total optimization iterations
        size_t batch_size = 8;              ///< Number of parallel context sequences (B)
        float learning_rate = 0.001f;       ///< Peak / Base AdamW learning rate (max_lr, slashed to 0.001 for stability)
        float min_learning_rate = 0.00001f; ///< Minimum learning rate floor for cosine decay
        float warmup_ratio = 0.05f;         ///< Fraction of total steps devoted to linear warmup
        float weight_decay = 0.01f;         ///< Decoupled L2 regularization
        float max_grad_norm = .760f;        ///< Global parameter-gradient L2 clip (1.0 default)
        size_t log_interval = 15;           ///< Logging cadence

        // --- Loss-adaptive LR shrink & decay slowdown ---
        float loss_shrink_short_alpha = 0.05f; ///< EMA alpha for short-window loss
        float loss_shrink_long_alpha = 0.01f;  ///< EMA alpha for long-window loss
        float loss_shrink_floor = 0.34f;       ///< Minimum LR multiplier
        size_t loss_shrink_warmup = 30;        ///< Skip shrink until this many steps have run
        float max_lr_step_growth_ratio = 1.10f;///< Slew-rate limiter: max +10% LR increase per step

        // --- Tri-Level Mistake Checkpoint Repulsion Engine ---
        bool enable_mistake_repulsion = true;         ///< Repel optimization away from past mistake checkpoints
        float mistake_repel_lambda_grad = 0.20f;      ///< Level A: Gradient direction repulsion strength (every step)
        float mistake_repel_lambda_latent = 0.15f;    ///< Level B: Latent representation repulsion strength (every step)
        float mistake_repel_lambda_weight = 0.05f;    ///< Level C: Parameter space barrier repulsion strength
        size_t weight_repel_step_interval = 10;       ///< Level C evaluation cadence (every 10 steps)

        // --- Ablation baseline switch (Phase 0 of the stability plan) ---
        // When true, the trainer disables every experimental adaptive module so that
        // training runs on a clean, known-good stack: plain AdamW, fixed context,
        // no meta-network, no Taylor foresight nudges, no progressive depth/dataset
        // oscillation. Nothing is deleted -- flags are just overridden in the ctor
        // -- so ablation against the full engine is a single-flag flip.
        bool safe_mode = false;

        // --- Meta-Neural Loss & 4-Formula Optimization ---
        bool enable_meta_loss_opt = true;     ///< Online Meta-Neural Network loss & step optimizer
        bool enable_multi_formula_opt = true; ///< Dynamic 1-4 Formula Weight Physics

        // --- Progressive Context & Token Window Extension ---
        bool progressive_context_growth = true; ///< Automatically expand token window based on steps or loss
        bool step_based_context_growth = true;  ///< When true, automatically expands token window at step milestones
        size_t initial_seq_len = 32;            ///< Starting context sequence length (T_0)
        size_t max_seq_len = 2048;              ///< Maximum expanded context sequence length (T_max)

        // --- Progressive Depth Ramping ---
        bool progressive_depth_growth = true;  ///< Automatically ramps model depth (active blocks) as steps progress
        size_t initial_layers = 4;             ///< Starting active layer depth
        size_t max_layers = 5;                 ///< Maximum total model depth (L_max)
        size_t depth_ramp_step_interval = 250; ///< Every N steps, unlock 1 more layer

        // --- Progressive Dataset Horizon Growth ---
        bool progressive_dataset_growth = true; ///< Gradually expand dataset ratio
        float initial_dataset_ratio = 0.40f;    ///< Starting corpus slice (40%)
        float max_dataset_ratio = 1.00f;        ///< Final corpus slice (100%)
        float dataset_growth_rate = 0.002f;     ///< Linear increment of corpus ratio per step

        // --- Auxiliary Numerical Features ---
        bool use_curvature_scaling = true;  ///< Apply 2nd-order Rayleigh quotient curvature
        bool use_armijo_line_search = true; ///< Armijo-Goldstein step validation
        bool use_data_filter = false;       ///< Shannon information entropy batch filter

        // --- Calculus of Constructions (CoC) Formal Reasoning ---
        bool enable_coc_verification = true;  ///< Run CoC dependent-type & proof verification every few steps
        size_t coc_verification_interval = 5; ///< Interval in steps for formal CoC proof check (every 5 steps)

        // --- Periodic "text challenge" evaluation ---
        size_t eval_interval = 100; ///< Fire the on_eval callback every N steps (0 disables)
    };

    /**
     * @struct LLMStepMetrics
     * @brief Telemetry captured at each forward/backward optimization iteration.
     */
    struct LLMStepMetrics
    {
        size_t step;                    ///< Current iteration index
        float loss;                     ///< Cross-entropy / Focal loss across all sequence tokens
        float perplexity;               ///< Language model perplexity PPL = exp(loss)
        float top1_accuracy;            ///< Exact match percentage (Rank 1) [0% - 100%]
        float top20_accuracy;           ///< Top-20 candidate match percentage [0% - 100%]
        float rank_score_top20;         ///< Index-weakened rank accuracy score [0% - 100%]
        float learning_rate;            ///< Active dynamic learning rate at this step
        size_t active_seq_len;          ///< Current active context token window length
        float penalty_factor;           ///< Active penalization strength factor
        float d_loss_d_penalty;         ///< Derivative of penalization impact on loss d(Loss)/d(Penalty)
        float meta_loss_scale;          ///< Meta-predicted loss scaling multiplier
        float meta_focal_gamma;         ///< Meta-predicted dynamic focal loss exponent
        float taylor_penalty_conf;      ///< Confidence score C in [0, 1] for Taylor penalty prediction
        float taylor_penalty_pred;      ///< Taylor predicted optimal penalty step
        bool bad_batch_skipped = false; ///< True if loss > 15.0 caused batch update skip
        float coc_proof_score = 1.0f;   ///< CoC proof consistency score in [0, 1]
        bool coc_verified = true;       ///< True if CoC type check passed
        float mistake_similarity = 0.0f; ///< Max similarity to past bad mistake checkpoints in [0, 1]
        size_t mistake_count = 0;       ///< Number of recorded mistake checkpoints
    };

    /**
     * @struct BenchmarkTelemetry
     * @brief Comprehensive real-time training telemetry and throughput benchmark statistics.
     */
    struct BenchmarkTelemetry
    {
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
        float penalty_factor = 0.0f;
        float taylor_penalty_conf = 0.0f;
        float taylor_penalty_pred = 0.0f;
        float coc_proof_consistency = 1.0f;
        float mistake_similarity = 0.0f;
        size_t mistake_count = 0;
        size_t active_vocab_size = 0;
        size_t active_context_length = 0;
        size_t active_model_layers = 0;
        size_t total_parameters = 0;
        size_t chrono_ticks = 0;
        size_t background_streamed_tokens = 0;
        ring1::FormulaDistributionStats formula_stats;
    };

    class ChronoAsyncEngine;

    /**
     * @class LLMTrainer
     * @brief Orchestrates mini-batching, dynamic LR scheduling, loss derivative pyramid, backward pass, AdamW optimization, and Real-Time Telemetry Dashboard.
     */
    class LLMTrainer
    {
    public:
        ring2::TransformerLM &model;
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
        float ema_loss_long = 0.0f;
        float initial_loss = 0.0f;
        bool ema_initialized = false;
        size_t observed_steps = 0;
        size_t current_seq_len = 32;
        float dynamic_lr_gain = 1.0f; ///< Uncapped dynamic learning rate surge multiplier (max = inf)
        size_t total_tokens_trained = 0;
        float last_grad_norm = 0.0f; ///< Most recent pre-clip global L2 gradient norm (0 on spike steps)

        // --- Debug snapshot of previous step (for "what changed" deltas in the debug log) ---
        struct StepDebugSnapshot
        {
            bool have = false;
            float loss = 0.0f;
            float ema_short = 0.0f;
            float lr = 0.0f;
            float penalty = 0.0f;
            float grad_norm = 0.0f;
            float meta_scale = 0.0f;
            float focal_gamma = 0.0f;
            size_t active_layers = 0;
            size_t seq_len = 0;
            bool watchdog = false;
        } prev_debug_snapshot;

        // --- Stability watchdog state (Phase 2 of the stability plan) ---
        // Detects sustained loss RISE above recent EMA. When triggered, temporarily:
        //   * multiplies LR by watchdog_lr_penalty (default 0.25x) each affected step
        //   * disables meta-net + Taylor nudges + 4-formula routing
        // until loss recovers back to within watchdog_recover_gap of the pre-spike EMA.
        size_t watchdog_bad_streak = 0;          ///< consecutive steps of loss >> ema_loss_short
        size_t watchdog_recovery_left = 0;       ///< remaining steps to keep experimental modules frozen
        float watchdog_baseline_loss = 0.0f;     ///< EMA loss the moment the watchdog fired
        float watchdog_rise_gap = 1.5f;          ///< current > ema_short + gap counts as "bad"
        size_t watchdog_trigger_streak = 3;      ///< N consecutive bad steps to trigger
        size_t watchdog_min_recovery_steps = 25; ///< keep modules frozen at least this long
        float watchdog_recover_gap = 0.5f;       ///< current <= baseline + this = fully recovered
        float watchdog_lr_penalty = 0.25f;       ///< LR *= this while watchdog is active
        bool watchdog_active = false;

        // --- Mistake Checkpoint Memory & Fingerprinting ---
        static constexpr size_t MAX_MISTAKES = 30;
        std::deque<MistakeCheckpoint> mistake_memory;

        /// Computes similarity of current state to past mistake checkpoints in [0, 1]
        float compute_mistake_similarity(const MistakeCheckpoint &current_state) const;

        /// Records a bad / unstable state into mistake memory with gradient, latent, and weight signatures
        void record_mistake(float loss, float ema_loss, float grad_norm, float penalty, float meta_scale, float gain, size_t step,
                            const std::vector<float> &grad_sig = {}, const std::vector<float> &latent_sig = {});

        // --- Directional Effect & Damped Operation Reversal State ---
        float last_lr_direction = 1.0f;    ///< Direction (+1.0 surge, -1.0 damp) of last LR modulation
        float last_lr_loss_delta = 0.0f;   ///< Observed loss effect Delta L from last LR adjustment
        float last_curv_direction = 1.0f;  ///< Direction (+1.0 up, -1.0 down) of last curvature scaling
        float last_curv_loss_delta = 0.0f; ///< Observed loss effect Delta L from last curvature scaling

        /// Callback fired whenever parameter expansion (neurogenesis) occurs
        function<void()> on_param_expansion = nullptr;

        /// Optional pointer to background asynchronous data streamer
        class BackgroundDataStreamer* background_streamer = nullptr;

        /// Optional pointer to concurrent chrono asynchronous subsystem scheduler
        class ChronoAsyncEngine* chrono_engine = nullptr;

        LLMTrainer(ring2::TransformerLM &lm, LLMTrainingConfig cfg = {});

        /// Computes dynamic learning rate for step t using Linear Warmup + Loss-Decay-Slowdown Cosine Annealing
        float compute_scheduled_lr(size_t step, size_t total_steps) const;

        /// Multiplier in [floor, 1] applied on top of the scheduled LR when the short-window
        /// loss EMA rises above the long-window one.
        float compute_loss_shrink() const;

        /// Evaluates average loss over validation batches without parameter updates
        float evaluate_loss(const TextDataset &dataset, size_t eval_batches = 4);

        /// Executes single forward, accuracy scoring, loss derivative pyramid, backward pass, and AdamW update
        LLMStepMetrics train_step(const TextBatch &batch, bool compute_detailed_metrics = true);

        /// Prints a comprehensive, formatted visual Benchmark & Telemetry Dashboard to the console
        void print_benchmark_dashboard(const BenchmarkTelemetry &tel, size_t current_step, size_t total_steps) const;

        /// Runs full training loop across the dataset with dynamic learning rate and optional progress callback.
        void train(TextDataset &dataset,
                   const function<void(const LLMStepMetrics &)> &on_step = nullptr,
                   const function<void(size_t)> &on_eval = nullptr);
    };

} // namespace ring3
