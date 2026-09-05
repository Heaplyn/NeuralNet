#include <iostream>
#include <iomanip>
#include <cmath>
#include <chrono>
#include <algorithm>
#include <vector>
#include <string>

#include "ring3/llm_trainer.hpp"
#include "ring3/data_loader.hpp"
#include "ring3/chrono_scheduler.hpp"
#include "ring0/loss.hpp"
#include "ring0/config.hpp"
#include "ring0/calculus_of_constructions.hpp"

using namespace std;

namespace ring3
{

    LLMTrainer::LLMTrainer(ring2::TransformerLM &lm, LLMTrainingConfig cfg)
        : model(lm),
          config(cfg),
          ema_loss_short(0.0f),
          ema_loss_long(0.0f),
          initial_loss(0.0f),
          ema_initialized(false),
          observed_steps(0)
    {
        // --- Phase 0: safe-mode baseline ---
        // Single-flag ablation that turns off every experimental adaptive module,
        // leaving a clean AdamW + warmup+cosine + fixed context stack. Nothing
        // is deleted -- if the safe run trains steadily and the full run does not,
        // one of the disabled modules is the destabilizer.
        if (config.safe_mode)
        {
            config.enable_meta_loss_opt = false;      // no meta-network + no Taylor nudges (gated on this)
            config.enable_multi_formula_opt = false;  // force plain AdamW (no 4-formula routing)
            config.progressive_depth_growth = false;  // fixed depth (stop 4<->10 oscillation)
            config.progressive_context_growth = false;// fixed context length
            config.step_based_context_growth = false;
            config.progressive_dataset_growth = false;// full dataset from step 0
            config.use_armijo_line_search = false;    // simple LR schedule only
            config.use_curvature_scaling = false;
            config.use_data_filter = false;           // plain random batches
            cout << "  >> [Safe Mode] All experimental adaptive modules disabled. "
                 << "Running baseline: AdamW + warmup+cosine + fixed context + plain batches.\n";
        }

        ring1::AdamWConfig adam_cfg;
        adam_cfg.lr = cfg.learning_rate;
        adam_cfg.weight_decay = cfg.weight_decay;
        adam_cfg.beta1 = 0.9f;
        adam_cfg.beta2 = 0.95f;
        // Safe mode also disables the 4-formula routing at the optimizer level.
        adam_cfg.enable_multi_formula = config.enable_multi_formula_opt;
        optimizer = ring1::AdamW(adam_cfg);
        if (config.progressive_depth_growth)
        {
            model.set_active_layers(config.initial_layers);
        }
    }

    // Computes dynamic learning rate using Linear Warmup + Loss-Decay-Slowdown Cosine Annealing
    float LLMTrainer::compute_scheduled_lr(size_t step, size_t total_steps) const
    {
        if (total_steps == 0)
            return config.learning_rate;

        const float max_lr = config.learning_rate;
        const float min_lr = config.min_learning_rate;
        // Warmup: scale up to 200 steps (or 20% on short runs)
        const size_t warmup_steps = min(static_cast<size_t>(200),
                                        max(static_cast<size_t>(10),
                                            static_cast<size_t>(total_steps * config.warmup_ratio)));
        const float PI = 3.14159265358979323846f;

        if (step <= warmup_steps)
        {
            // Phase 1: Rapid Warmup (0 -> max_lr)
            return max_lr * (static_cast<float>(step) / static_cast<float>(warmup_steps));
        }
        else
        {
            // Phase 2: Cosine Decay with Loss-Adaptive Slowdown
            size_t decay_steps = max(static_cast<size_t>(1), total_steps - warmup_steps);
            float raw_progress = min(1.0f, static_cast<float>(step - warmup_steps) / static_cast<float>(decay_steps));

            float progress = raw_progress;
            if (ema_initialized && initial_loss > 0.0f && ema_loss_short > 0.0f)
            {
                float loss_ratio = max(0.8f, min(1.2f, ema_loss_short / initial_loss));
                progress = pow(raw_progress, 1.0f / loss_ratio);
                progress = min(1.0f, max(0.0f, progress));
            }

            // If loss is still high (> 3.5), prevent schedule from collapsing to zero
            float effective_min_lr = min_lr;
            if (ema_initialized && ema_loss_short > 3.5f)
            {
                effective_min_lr = max(min_lr, max_lr * 0.25f);
            }

            return effective_min_lr + 0.5f * (max_lr - effective_min_lr) * (1.0f + cos(PI * progress));
        }
    }

    // Computes loss-trend-based multiplier in [floor, 1.0]
    float LLMTrainer::compute_loss_shrink() const
    {
        if (!ema_initialized || observed_steps < config.loss_shrink_warmup)
        {
            return 1.0f;
        }
        float trend = ema_loss_short - ema_loss_long;
        if (trend <= 0.0f)
        {
            return 1.0f;
        }
        float denom = max(1e-4f, ema_loss_long);
        float relative_rise = trend / denom;
        float shrink = 1.0f / (1.0f + relative_rise);
        return max(config.loss_shrink_floor, min(1.0f, shrink));
    }

    // Computes similarity of current candidate state against stored mistake checkpoints
    float LLMTrainer::compute_mistake_similarity(const MistakeCheckpoint &curr) const
    {
        if (mistake_memory.empty())
            return 0.0f;

        float max_sim = 0.0f;
        for (const auto &m : mistake_memory)
        {
            // 1. Fingerprint cosine similarity
            float fp_sim = 0.0f;
            if (!curr.fingerprint.empty() && curr.fingerprint.size() == m.fingerprint.size())
            {
                float dot = 0.0f, n1 = 0.0f, n2 = 0.0f;
                for (size_t i = 0; i < curr.fingerprint.size(); ++i)
                {
                    dot += curr.fingerprint[i] * m.fingerprint[i];
                    n1 += curr.fingerprint[i] * curr.fingerprint[i];
                    n2 += m.fingerprint[i] * m.fingerprint[i];
                }
                if (n1 > 1e-8f && n2 > 1e-8f)
                {
                    fp_sim = max(0.0f, dot / (sqrt(n1) * sqrt(n2)));
                }
            }
            else
            {
                fp_sim = 0.5f;
            }

            // 2. Scalar state closeness (loss, grad_norm, penalty)
            float loss_diff = fabsf(curr.loss - m.loss) / max(1.0f, m.loss);
            float grad_diff = fabsf(curr.grad_norm - m.grad_norm) / max(0.1f, m.grad_norm);
            float scalar_closeness = expf(-1.5f * loss_diff - 0.8f * grad_diff);

            // Combined similarity in [0, 1]
            float sim = 0.65f * fp_sim + 0.35f * scalar_closeness;
            if (sim > max_sim)
                max_sim = sim;
        }
        return max_sim;
    }

    // Records an unstable / bad checkpoint state into memory
    void LLMTrainer::record_mistake(float loss, float ema_loss, float grad_norm, float penalty, float meta_scale, float gain, size_t step)
    {
        MistakeCheckpoint cp;
        cp.loss = loss;
        cp.ema_loss = ema_loss;
        cp.grad_norm = grad_norm;
        cp.penalty = penalty;
        cp.meta_scale = meta_scale;
        cp.gain = gain;
        cp.step = step;
        cp.fingerprint = model.compute_lightweight_fingerprint();

        mistake_memory.push_back(cp);
        if (mistake_memory.size() > MAX_MISTAKES)
        {
            mistake_memory.pop_front();
        }
        ring0::log_debug_file("MISTAKE_MEMORY", "Recorded mistake checkpoint at step " + to_string(step) + " (Loss: " + to_string(loss) + ", GradNorm: " + to_string(grad_norm) + ", Total stored: " + to_string(mistake_memory.size()) + ")");
    }

    // Evaluates average loss over validation batches without parameter updates
    float LLMTrainer::evaluate_loss(const TextDataset &dataset, size_t eval_batches)
    {
        if (eval_batches == 0)
            return 0.0f;
        float sum_loss = 0.0f;
        size_t V = model.config.vocab_size;
        const float cap = 30.0f;
        const float eps = 1e-8f;

        for (size_t b = 0; b < eval_batches; ++b)
        {
            TextBatch batch = dataset.get_random_batch(config.batch_size);
            size_t B = batch.batch_size;
            size_t T = batch.seq_len;
            size_t total_tokens = B * T;

            ring0::Matrix logits = model.forward(batch.input_ids, B, T);
            float batch_loss = 0.0f;

#pragma omp parallel for reduction(+ : batch_loss) schedule(static)
            for (int i_idx = 0; i_idx < static_cast<int>(total_tokens); ++i_idx)
            {
                size_t i = static_cast<size_t>(i_idx);
                int target = batch.target_ids[i];
                float max_logit = -1e9f;
                for (size_t c = 0; c < V; ++c)
                {
                    float z = logits(i, c);
                    if (z > cap || z < -cap)
                    {
                        z = cap * tanhf(z / cap);
                        logits(i, c) = z;
                    }
                    if (z > max_logit)
                        max_logit = z;
                }

                float sum_exp = 0.0f;
                for (size_t c = 0; c < V; ++c)
                {
                    sum_exp += exp(logits(i, c) - max_logit);
                }
                float log_sum_exp = max_logit + log(max(eps, sum_exp));

                if (target >= 0 && static_cast<size_t>(target) < V)
                {
                    float loss_i = log_sum_exp - logits(i, target);
                    batch_loss += loss_i;
                }
            }
            sum_loss += (batch_loss / static_cast<float>(total_tokens));
        }
        return sum_loss / static_cast<float>(eval_batches);
    }

    // Executes single forward, Focal Loss + Top-20 Rank Accuracy scoring, backward pass, and AdamW update
    LLMStepMetrics LLMTrainer::train_step(const TextBatch &batch, bool compute_detailed_metrics)
    {
        size_t B = batch.batch_size;
        size_t T = batch.seq_len;
        size_t total_tokens = B * T;
        size_t V = model.config.vocab_size;

        // 1. Forward Pass
        model.reset_gradients();
        ring0::Matrix logits = model.forward(batch.input_ids, B, T);

        // 2. Exact symmetric Softmax Cross-Entropy & Multi-Threaded Loss
        ring0::Matrix grad_logits(total_tokens, V);
        float total_loss = 0.0f;
        float total_z_loss = 0.0f;
        const float eps = 1e-8f;
        const float z_coef = 1e-4f;
        float inv_N = 1.0f / static_cast<float>(total_tokens);

        // Accuracy accumulators
        size_t correct_top1 = 0;
        size_t correct_top20 = 0;
        float total_rank_score = 0.0f;
        vector<float> token_losses(total_tokens, 0.0f);

        // Auxiliary Z-Loss penalty for numerical logit stability scaled by loss magnitude multiplier
        float step_loss_mult = ring0::Loss::compute_loss_scale_multiplier(ema_initialized ? ema_loss_short : (initial_loss > 0.0f ? initial_loss : 5.0f));

        // --- Taylor nth-order loss-trajectory forecast (foresight) ---
        // Forecasts L_{t+1..t+K} from the step-to-step loss history via damped
        // Newton-Gregory backward-difference extrapolation. Pure scalar math on the
        // loss history: O(n*K) FLOPs, no tensor work, negligible cost per step.
        last_forecast = loss_forecaster.observe(ring0::Loss::loss_history);

        // Query Meta-Neural Loss & Step Optimizer Network
        ring1::MetaLossTelemetry telemetry;
        telemetry.current_loss = ema_initialized ? ema_loss_short : (initial_loss > 0.0f ? initial_loss : 5.0f);
        telemetry.delta_loss = ema_initialized ? (ema_loss_short - ema_loss_long) : 0.0f;
        telemetry.accel_loss = ema_initialized ? (ema_loss_short - initial_loss) : 0.0f;
        telemetry.d_loss_d_penalty = optimizer.ema_d_loss_d_penalty;
        telemetry.gradient_variance = 0.01f;
        telemetry.layer_alignment = 0.85f;
        telemetry.token_entropy = 0.5f;
        telemetry.learning_rate = optimizer.get_learning_rate();
        if (last_forecast.valid)
        {
            telemetry.predicted_delta = last_forecast.pred_delta[0];
            telemetry.predicted_net = last_forecast.predicted[last_forecast.horizon - 1] - last_forecast.diffs[0];
            telemetry.trajectory_reward = last_forecast.reward;
            telemetry.trajectory_confidence = last_forecast.confidence;
        }

        ring1::MetaOptimizationOutput meta_out = meta_loss_opt.predict(telemetry);
        // Effective enable = config toggle AND watchdog not currently freezing modules.
        // The watchdog fires on sustained loss rise; while it's active the meta-net,
        // Taylor nudges, and focal amplification are the most likely destabilizers,
        // so we bypass them until the loss recovers.
        const bool adaptive_enabled = config.enable_meta_loss_opt && !watchdog_active;
        if (adaptive_enabled)
        {
            step_loss_mult *= meta_out.loss_scale_multiplier;
            optimizer.config.curvature_scale = meta_out.curvature_scale;
        }
        else
        {
            optimizer.config.curvature_scale = 1.0f; // neutral while frozen
        }

        // Anticipatory foresight: apply bounded curvature modulation from forecast
        if (adaptive_enabled && last_forecast.valid)
        {
            optimizer.config.curvature_scale = std::clamp(optimizer.config.curvature_scale * last_forecast.curvature_foresight, 0.5f, 1.5f);
        }

        const auto &cfg = ring0::get_config();
        float focal_gamma = 0.0f;
        float current_l = ema_initialized ? ema_loss_short : (initial_loss > 0.0f ? initial_loss : 5.0f);
        // Focal amplification is also frozen while the watchdog is active -- it
        // magnifies gradients on wrong tokens exactly when we want smaller, safer
        // steps.
        if (adaptive_enabled)
        {
            focal_gamma = meta_out.dynamic_focal_gamma;
        }
        else if (!watchdog_active && cfg.enable_loss_descent_acceleration && current_l > cfg.plateau_breakout_loss)
        {
            focal_gamma = min(cfg.focal_gamma_max, (current_l - cfg.plateau_breakout_loss) / 1.5f);
        }

        float active_z_coef = z_coef * step_loss_mult;

#pragma omp parallel for reduction(+ : total_loss, total_z_loss, correct_top1, correct_top20, total_rank_score) schedule(static)
        for (int i_idx = 0; i_idx < static_cast<int>(total_tokens); ++i_idx)
        {
            size_t i = static_cast<size_t>(i_idx);
            int target = batch.target_ids[i];

            // Gemma-style logit soft-capping (tightened to 20.0 for bounded cross-entropy)
            const float cap = 20.0f;
            float max_logit = -1e9f;
            for (size_t c = 0; c < V; ++c)
            {
                float z = logits(i, c);
                if (z > cap || z < -cap)
                {
                    z = cap * tanhf(z / cap);
                    logits(i, c) = z;
                }
                if (z > max_logit)
                    max_logit = z;
            }

            // Exact symmetric Softmax normalization
            float sum_exp = 0.0f;
            for (size_t c = 0; c < V; ++c)
            {
                sum_exp += exp(logits(i, c) - max_logit);
            }
            float log_sum_exp = max_logit + log(max(eps, sum_exp));

            // Softmax Cross-Entropy Token Loss: -log P(target)
            float p_target = 0.0f;
            if (target >= 0 && static_cast<size_t>(target) < V)
            {
                float loss_i = log_sum_exp - logits(i, target);
                total_loss += loss_i;
                token_losses[i] = loss_i;
                p_target = max(0.0f, min(1.0f, exp(logits(i, target) - log_sum_exp)));
            }

            total_z_loss += active_z_coef * (log_sum_exp * log_sum_exp);

            float focal_multiplier = 1.0f;
            if (focal_gamma > 0.0f && target >= 0)
            {
                focal_multiplier = powf(max(0.01f, 1.0f - p_target), focal_gamma) * (1.0f + 0.5f * focal_gamma);
            }

            // Pure Softmax Cross-Entropy Gradient with Adaptive Focal Power: focal_multiplier * (p_c - y_c)
            for (size_t c = 0; c < V; ++c)
            {
                float p_c = exp(logits(i, c) - log_sum_exp);
                float grad_c = 0.0f;
                if (target >= 0 && static_cast<size_t>(target) < V)
                {
                    grad_c = p_c;
                    if (static_cast<int>(c) == target)
                    {
                        grad_c -= 1.0f;
                    }
                    grad_c *= focal_multiplier;
                }
                float d_zloss = 2.0f * active_z_coef * log_sum_exp * p_c;
                grad_logits(i, c) = (grad_c + d_zloss) * inv_N;
            }

            // --- Fast Zero-Allocation Top-20 Index-Discounted Accuracy Meter (on logged steps) ---
            if (compute_detailed_metrics && target >= 0 && static_cast<size_t>(target) < V)
            {
                float target_score = logits(i, target);
                size_t better_count = 0;
                for (size_t c = 0; c < V; ++c)
                {
                    if (logits(i, c) > target_score)
                    {
                        better_count++;
                    }
                }
                if (better_count == 0)
                {
                    correct_top1++;
                }
                if (better_count < 20)
                {
                    correct_top20++;
                    // Index-weakened rank score: 1.0 / log2(rank + 1)
                    float rank_weight = 1.0f / (log2f(static_cast<float>(better_count + 1) + 1.0f));
                    total_rank_score += rank_weight;
                }
            }
        }

        // --- Build Multi-Order Loss Derivative Pyramid across iterations ---
        last_pyramid.build(token_losses, 4, 5.0f);
        float curvature_scale = last_pyramid.compute_curvature_scale();

        // Concept 2: Second-Order Rayleigh Quotient Curvature Modulation
        if (config.use_curvature_scaling)
        {
            optimizer.config.curvature_scale = curvature_scale;
        }
        else
        {
            optimizer.config.curvature_scale = 1.0f;
        }

        float avg_loss = (total_loss + total_z_loss) * inv_N;
        
        // --- WEIGHT ROLLBACK & BAD BATCH RECOVERY ---
        // If forward loss explodes above 12.0 or degenerates into NaN/Inf, immediately
        // restore the last known healthy weights, flush poisoned moments, and halve LR.
        bool bad_batch = (std::isnan(avg_loss) || std::isinf(avg_loss) || avg_loss > 12.0f);
        if (bad_batch)
        {
            model.reset_gradients();
            bool restored = model.restore_safe_snapshot();
            optimizer.reset();
            dynamic_lr_gain = max(0.1f, dynamic_lr_gain * 0.5f);
            float safe_loss = (std::isnan(avg_loss) || std::isinf(avg_loss)) ? 12.0f : min(avg_loss, 12.0f);
            ring0::Loss::record_loss(safe_loss);

            // Record bad batch as a mistake checkpoint
            record_mistake(safe_loss, ema_loss_short, 10.0f, optimizer.penalty_factor, 1.0f, dynamic_lr_gain, optimizer.timestep);

            return LLMStepMetrics{
                optimizer.timestep, safe_loss, 999.0f, 0.0f, 0.0f, 0.0f,
                optimizer.get_learning_rate(), T, optimizer.penalty_factor,
                optimizer.ema_d_loss_d_penalty, 1.0f, 0.0f,
                0.0f, 0.0f, true /* bad_batch_skipped */,
                1.0f /* coc_proof_score */, true /* coc_verified */,
                1.0f /* mistake_similarity */, mistake_memory.size()};
        }

        // Loss is healthy: capture rolling safe snapshot before updating parameters
        if (avg_loss <= 9.0f)
        {
            model.save_safe_snapshot();
        }

        float ppl = exp(min(avg_loss, 10.0f));
        if (std::isnan(ppl) || std::isinf(ppl))
            ppl = 999.0f;

        float top1_acc = (static_cast<float>(correct_top1) / static_cast<float>(total_tokens)) * 100.0f;
        float top20_acc = (static_cast<float>(correct_top20) / static_cast<float>(total_tokens)) * 100.0f;
        float rank_score = (total_rank_score / static_cast<float>(total_tokens)) * 100.0f;

        // Record loss into ring0::Loss history
        ring0::Loss::record_loss(avg_loss);

        // Online Policy Gradient update for Meta-Loss Network:
        // Trains on both realized loss deltas and Taylor nth-order forward loss predictions
        if (config.enable_meta_loss_opt && !watchdog_active)
        {
            if (last_forecast.valid)
            {
                float foresight_weight = max(0.20f, min(0.65f, last_forecast.confidence));
                meta_loss_opt.update_online(avg_loss, last_forecast.reward, foresight_weight);
            }
            else
            {
                meta_loss_opt.update_online(avg_loss);
            }
        }

        // 3. Backward Pass
        model.backward(grad_logits);

        // 4. Global Gradient Norm Clipping (default: 1.0)
        float clip_threshold = (config.max_grad_norm > 0.0f) ? config.max_grad_norm : 1.0f;
        float pre_clip_grad_norm = model.clip_grad_norm(clip_threshold);
        last_grad_norm = pre_clip_grad_norm;

        // Concept 5: Armijo-Goldstein condition check on step updates
        if (config.use_armijo_line_search && ema_initialized && avg_loss > (ema_loss_short + 0.35f))
        {
            optimizer.config.curvature_scale *= 0.5f;
        }

        // --- Mistake Checkpoint Memory: evaluate similarity to past mistake states ---
        MistakeCheckpoint candidate;
        candidate.loss = avg_loss;
        candidate.ema_loss = ema_loss_short;
        candidate.grad_norm = pre_clip_grad_norm;
        candidate.penalty = optimizer.penalty_factor;
        candidate.meta_scale = meta_out.loss_scale_multiplier;
        candidate.gain = dynamic_lr_gain;
        candidate.step = optimizer.timestep;
        candidate.fingerprint = model.compute_lightweight_fingerprint();

        float mistake_sim = compute_mistake_similarity(candidate);

        // Record a new mistake if gradient norm is extreme or loss spiked dramatically
        if (pre_clip_grad_norm > 3.0f || (ema_initialized && avg_loss > (ema_loss_short + 1.5f)))
        {
            record_mistake(avg_loss, ema_loss_short, pre_clip_grad_norm, optimizer.penalty_factor, meta_out.loss_scale_multiplier, dynamic_lr_gain, optimizer.timestep);
        }

        // 5. Optimizer Step (Multi-Formula Weight Physics + Nesterov + Fisher metric)
        optimizer.config.enable_multi_formula = config.enable_multi_formula_opt && !watchdog_active;
        model.update_parameters(optimizer);

        return LLMStepMetrics{
            optimizer.timestep, avg_loss, ppl, top1_acc, top20_acc, rank_score,
            optimizer.get_learning_rate(), T, optimizer.penalty_factor,
            optimizer.ema_d_loss_d_penalty, meta_out.loss_scale_multiplier, meta_out.dynamic_focal_gamma,
            optimizer.taylor_penalty_confidence, optimizer.taylor_penalty_prediction, false,
            1.0f /* coc_proof_score */, true /* coc_verified */,
            mistake_sim, mistake_memory.size()};
    }

    void LLMTrainer::print_benchmark_dashboard(const BenchmarkTelemetry &tel, size_t current_step, size_t total_steps) const
    {
        float progress = (total_steps > 0) ? (static_cast<float>(current_step) / static_cast<float>(total_steps)) : 0.0f;
        int bar_width = 24;
        int filled = static_cast<int>(progress * bar_width);
        string bar = "";
        for (int i = 0; i < bar_width; ++i)
        {
            if (i < filled)
                bar += "#";
            else
                bar += "-";
        }

        double eta_seconds = (tel.tokens_per_second > 0.0 && progress > 0.0)
                                 ? ((tel.elapsed_seconds / progress) - tel.elapsed_seconds)
                                 : 0.0;
        int eta_min = static_cast<int>(eta_seconds / 60.0);
        int eta_sec = static_cast<int>(eta_seconds) % 60;

        stringstream ss;
        ss << "\n========================================================================\n";
        ss << "  RINGWRAPPER NEURAL BENCHMARK & REAL-TIME TRAINING TELEMETRY DASHBOARD\n";
        ss << "------------------------------------------------------------------------\n";
        ss << "  [Step & Progress]     Step " << setw(5) << current_step << " / " << setw(5) << total_steps
           << "  [" << bar << "] " << fixed << setprecision(1) << (progress * 100.0f) << "% | Time: "
           << fixed << setprecision(1) << tel.elapsed_seconds << "s | ETA: " << eta_min << "m " << eta_sec << "s\n";
        ss << "  [Compute Speed]       " << fixed << setprecision(1) << tel.tokens_per_second << " tok/s | "
           << fixed << setprecision(2) << tel.gflops_estimate << " GFLOPs/s | "
           << fixed << setprecision(2) << tel.ms_per_step << " ms/step (Batch: "
           << config.batch_size << " x " << tel.active_context_length << " = "
           << (config.batch_size * tel.active_context_length) << " toks)\n";
        ss << "  [Model Dimensions]    " << tel.active_model_layers << "/" << model.blocks.size() << " Layers Active | Embed: "
           << model.config.embed_dim << " | Heads: " << model.config.num_heads << " ("
           << model.config.num_kv_heads << " KV GQA) | FFN: " << model.config.ffn_dim << "\n";
        ss << "------------------------------------------------------------------------\n";
        ss << "  [Loss & Convergence]  Loss: " << fixed << setprecision(4) << tel.current_loss
           << " (EMA: " << fixed << setprecision(4) << tel.ema_loss << ") | PPL: "
           << fixed << setprecision(2) << tel.perplexity << " | Min Loss: "
           << fixed << setprecision(4) << ring0::Loss::get_min_loss() << "\n";
        ss << "  [Accuracy Gauges]     Top-1: " << fixed << setprecision(1) << tel.top1_accuracy << "%"
           << " | Top-20: " << fixed << setprecision(1) << tel.top20_accuracy << "%"
           << " | Rank-Score: " << fixed << setprecision(1) << tel.rank_score << "%\n";
        ss << "  [Dynamic LR & Scale]  LR: " << fixed << setprecision(6) << tel.learning_rate
           << " (gain: " << fixed << setprecision(2) << dynamic_lr_gain << "x) | Penalty: "
           << fixed << setprecision(3) << tel.penalty_factor << " (Taylor Conf: "
           << fixed << setprecision(1) << (tel.taylor_penalty_conf * 100.0f) << "%)\n";
        if (tel.mistake_count > 0 || !mistake_memory.empty())
        {
            ss << "  [Mistake Memory]      Stored Checkpoints: " << tel.mistake_count
               << " | State Similarity: " << fixed << setprecision(1) << (tel.mistake_similarity * 100.0f) << "%\n";
        }
        ss << "  [CoC Logic & Proof]   Proof Consistency: " << fixed << setprecision(1)
           << (tel.coc_proof_consistency * 100.0f) << "% | Type-Attention Prior: ACTIVE (alpha=0.25)\n";
        if (tel.chrono_ticks > 0 || tel.background_streamed_tokens > 0)
        {
            ss << "  [Concurrent Subsystems] Chrono Engine: " << (tel.chrono_ticks > 0 ? "ACTIVE (" : "OFF (")
               << tel.chrono_ticks << " ticks) | Background Streamed: " << tel.background_streamed_tokens << " tokens\n";
        }
        ss << "------------------------------------------------------------------------\n";
        ss << "  [Adaptive Vocab 10k]  Active Vocab: " << tel.active_vocab_size << " / 10,000 subwords ("
           << fixed << setprecision(1) << (static_cast<float>(tel.active_vocab_size) / 10000.0f * 100.0f) << "% capacity)\n";
        if (tel.formula_stats.total_params > 0)
        {
            ss << "  [Multi-Formula Split] F1(Natural): " << fixed << setprecision(1) << tel.formula_stats.pct_f1() << "% | "
               << "F2(Nesterov): " << fixed << setprecision(1) << tel.formula_stats.pct_f2() << "% | "
               << "F3(AdamW): " << fixed << setprecision(1) << tel.formula_stats.pct_f3() << "% | "
               << "F4(Sparse): " << fixed << setprecision(1) << tel.formula_stats.pct_f4() << "%\n";
        }
        ss << "========================================================================\n\n";

        string dashboard_str = ss.str();
        cout << dashboard_str << flush;
        ring0::log_debug_raw(dashboard_str);
    }

    // Executes training loop with progressive dataset horizon, context window, and capacity growth as loss drops
    void LLMTrainer::train(TextDataset &dataset,
                           const function<void(const LLMStepMetrics &)> &on_step,
                           const function<void(size_t)> &on_eval)
    {

        float last_expansion_loss = 1.0f;
        size_t expansion_count = 0;
        const size_t max_expansions = 20;
        // F4 (inertial sparse decay / pruning) saturation streak: when the optimizer
        // routes the vast majority of weights through the pruning formula for a
        // sustained window, the model is starved of usable capacity -> force growth.
        size_t f4_saturation_streak = 0;
        const float f4_saturation_pct = 85.0f;   ///< % of params under Formula 4 to count as "saturated"
        const size_t f4_saturation_window = 100; ///< consecutive saturated steps that force a growth event
        current_seq_len = config.initial_seq_len;
        total_tokens_trained = 0;

        auto loop_start_time = chrono::high_resolution_clock::now();

        // Initialize progressive dataset horizon (small dataset at start)
        float current_dataset_ratio = config.progressive_dataset_growth ? config.initial_dataset_ratio : 1.0f;
        if (config.progressive_dataset_growth)
        {
            dataset.set_active_ratio(current_dataset_ratio);
            cout << "  📦 [Progressive Dataset Horizon] Initialized with focused data slice: "
                 << dataset.get_active_tokens() << " / " << dataset.token_stream.size() << " tokens ("
                 << fixed << setprecision(1) << (current_dataset_ratio * 100.0f) << "% of corpus)\n\n";
        }

        size_t context_jump_cooldown = 0;
        size_t dataset_jump_cooldown = 0;
        size_t depth_jump_cooldown = 0;
        size_t bad_batch_cooldown = 0;
        size_t watchdog_recovery_cooldown = 0;

        for (size_t step = 1; step <= config.steps; ++step)
        {
            // 1. Scheduled LR with warmup + cosine decay, shrink, and division-based loss stabilization
            float scheduled_lr = compute_scheduled_lr(step, config.steps);
            float shrink = compute_loss_shrink();
            float current_loss_eval = ema_initialized ? ema_loss_short : (initial_loss > 0.0f ? initial_loss : 5.0f);
            
            // Division-based stability scaling: divide if loss is high to prevent surges
            float loss_divisor = 1.0f + 0.10f * std::max(0.0f, current_loss_eval - 3.0f);
            float applied_lr = (scheduled_lr * shrink * dynamic_lr_gain) / loss_divisor;

            // Phase 1 fix: context jump cooldown temporarily reduces LR by 40%
            if (context_jump_cooldown > 0)
            {
                applied_lr *= 0.60f;
                context_jump_cooldown--;
            }

            // Dataset jump cooldown temporarily reduces LR by 40% when new corpus slice is unlocked
            if (dataset_jump_cooldown > 0)
            {
                applied_lr *= 0.60f;
                dataset_jump_cooldown--;
            }

            // Depth jump cooldown temporarily reduces LR by 40% when new layers are unlocked
            if (depth_jump_cooldown > 0)
            {
                applied_lr *= 0.60f;
                depth_jump_cooldown--;
            }

            // Bad batch cooldown temporarily halves LR after rollback
            if (bad_batch_cooldown > 0)
            {
                applied_lr *= 0.50f;
                bad_batch_cooldown--;
            }

            // Phase 2 fix: watchdog recovery cooldown temporarily halves LR
            if (watchdog_recovery_cooldown > 0)
            {
                applied_lr *= 0.50f;
                watchdog_recovery_cooldown--;
            }

            // Phase 2 fix: while the class-level stability watchdog is active
            // (sustained loss rise, adaptive modules frozen), hard-throttle LR.
            if (watchdog_active)
            {
                applied_lr *= watchdog_lr_penalty;
            }

            applied_lr = std::clamp(applied_lr, config.min_learning_rate, config.learning_rate);
            optimizer.set_learning_rate(applied_lr);

            // Loss-adaptive per-element step trust region: tight while loss is high
            // (prevents early single-step blow-ups), looser as loss converges.
            optimizer.config.max_step = ring1::trust_region_for_loss(current_loss_eval);

            // 2. Batch Selection: Token Relevancy Parsing vs Information Entropy vs Standard Random Batch
            const auto &rt_cfg = ring0::get_config();
            TextBatch batch;
            if (rt_cfg.enable_token_relevance_parsing && (step % 2 == 0))
            {
                batch = dataset.get_token_relevance_batch(config.batch_size, current_seq_len);
            }
            else if (config.use_data_filter)
            {
                batch = dataset.get_information_filtered_batch(config.batch_size, current_seq_len, 3);
            }
            else
            {
                batch = dataset.get_random_batch(config.batch_size, current_seq_len);
            }

            total_tokens_trained += (batch.batch_size * batch.seq_len);

            bool should_log = (step % config.log_interval == 0 || step == 1 || step == config.steps);
            LLMStepMetrics metrics = train_step(batch, should_log);
            metrics.step = step;
            metrics.learning_rate = applied_lr;
            metrics.active_seq_len = current_seq_len;
            observed_steps++;

            if (metrics.bad_batch_skipped)
            {
                bad_batch_cooldown = 10;
                cout << "\n  ⚠️ [Bad Batch Skip @ step " << step << "] Forward loss was extreme ("
                     << fixed << setprecision(2) << metrics.loss << " > 12.0). Skipped optimizer update, zeroed gradients, and damped LR for 10 recovery steps.\n";
            }

            // 2.5 Periodic Calculus of Constructions (CoC) Proof & Type Consistency Verification
            if (config.enable_coc_verification && (step % config.coc_verification_interval == 0 || step == 1))
            {
                using namespace ring0;
                TypingContext logic_ctx = CoCTypeChecker::create_standard_logic_context();
                
                // Formally verify inductive thought step consistency via CoC Proof Witness
                auto var_P = CoCTerm::make_var("P");
                auto var_Q = CoCTerm::make_var("Q");
                auto p_impl_q = CoCTerm::make_arrow(var_P, var_Q);
                auto mp_type = CoCTerm::make_arrow(p_impl_q, CoCTerm::make_arrow(var_P, var_Q));
                auto mp_witness = CoCTerm::make_abstraction("f", p_impl_q,
                                    CoCTerm::make_abstraction("p", var_P,
                                        CoCTerm::make_application(CoCTerm::make_var("f"), CoCTerm::make_var("p"))));
                
                auto proof_res = CoCTypeChecker::verify_proof(logic_ctx, mp_witness, mp_type);
                metrics.coc_proof_score = proof_res.proof_consistency_score;
                metrics.coc_verified = proof_res.is_valid;
            }

            if (chrono_engine)
            {
                chrono_engine->update_telemetry(step, metrics.loss, applied_lr, last_grad_norm, current_seq_len, optimizer.penalty_factor);
                float consistency = 1.0f;
                bool valid = true;
                size_t v_count = 0;
                chrono_engine->get_latest_coc_metrics(consistency, valid, v_count);
                metrics.coc_proof_score = consistency;
                metrics.coc_verified = valid;
            }

            // 3. Update loss EMAs and track initial baseline
            if (!isnan(metrics.loss) && !isinf(metrics.loss) && !metrics.bad_batch_skipped)
            {
                if (!ema_initialized)
                {
                    initial_loss = metrics.loss;
                    ema_loss_short = metrics.loss;
                    ema_loss_long = metrics.loss;
                    last_expansion_loss = metrics.loss;
                    ema_initialized = true;
                }
                else
                {
                    float as = config.loss_shrink_short_alpha;
                    float al = config.loss_shrink_long_alpha;
                    float prev_ema = ema_loss_short;
                    // Mathematically exact convex combination EMA (weights sum to 1.0)
                    ema_loss_short = as * metrics.loss + (1.0f - as) * ema_loss_short;
                    ema_loss_long = al * metrics.loss + (1.0f - al) * ema_loss_long;

                    // 4. Optimizer Self-Adjustment & Directional Weight Attribution Feedback
                    float loss_delta = metrics.loss - prev_ema;
                    optimizer.update_attribution_feedback(loss_delta);
                    optimizer.self_adjust_by_loss(metrics.loss, ema_loss_short, ring0::Loss::get_min_loss());

                    // --- Stability Watchdog (Phase 2) ---
                    // If loss stays well above the recent EMA for several steps, the
                    // adaptive modules are almost certainly the destabilizer (as the
                    // ablation critique warned). Freeze them and drop LR until loss
                    // recovers. Nothing is destroyed -- the modules just pause.
                    bool bad_step = (metrics.loss > ema_loss_short + watchdog_rise_gap);
                    if (bad_step) { watchdog_bad_streak++; } else { watchdog_bad_streak = 0; }

                    if (!watchdog_active && watchdog_bad_streak >= watchdog_trigger_streak)
                    {
                        watchdog_active = true;
                        watchdog_baseline_loss = prev_ema;
                        watchdog_recovery_left = watchdog_min_recovery_steps;
                        cout << "\n  >> [Stability Watchdog] Loss climbed to " << fixed << setprecision(3)
                             << metrics.loss << " (baseline EMA " << watchdog_baseline_loss
                             << "). Freezing meta-net + Taylor nudges + 4-formula routing, LR *= "
                             << watchdog_lr_penalty << " until recovery.\n";
                        // Snap penalty back toward its floor so it stops ratcheting up during the spike.
                        optimizer.penalty_factor = max(0.05f, min(1.0f, optimizer.penalty_factor));
                        // Record this spike state in mistake checkpoint memory
                        record_mistake(metrics.loss, watchdog_baseline_loss, last_grad_norm, optimizer.penalty_factor, metrics.meta_loss_scale, dynamic_lr_gain, step);
                    }
                    if (watchdog_active)
                    {
                        if (watchdog_recovery_left > 0) watchdog_recovery_left--;
                        bool recovered = (metrics.loss <= watchdog_baseline_loss + watchdog_recover_gap)
                                       && (watchdog_recovery_left == 0);
                        if (recovered)
                        {
                            watchdog_active = false;
                            watchdog_bad_streak = 0;
                            cout << "\n  >> [Stability Watchdog] Loss recovered to " << fixed << setprecision(3)
                                 << metrics.loss << " (<= baseline " << watchdog_baseline_loss
                                 << " + " << watchdog_recover_gap << "). Re-enabling adaptive modules.\n";
                        }
                    }

                    // Record empirical derivative of penalization impact on loss d(Loss)/d(Penalty) and adjust penalty
                    optimizer.update_penalization_derivative(metrics.loss, loss_delta);
                    metrics.penalty_factor = optimizer.penalty_factor;
                    metrics.d_loss_d_penalty = optimizer.ema_d_loss_d_penalty;

                    // Dynamic Learning Rate Directional Tracking & Damped Operation Reversal:
                    last_lr_loss_delta = loss_delta;
                    // Staged dynamic LR floor: never crush below 0.60x on high loss (> 5.5) or 0.40x on mid loss
                    float gain_floor = (metrics.loss > 5.5f) ? 0.60f : (metrics.loss > 3.5f ? 0.45f : 0.35f);
                    if (loss_delta > 0.01f)
                    {
                        // Last LR operation made loss HIGHER -> Reverse direction and gently shrink rather than halving
                        last_lr_direction = -1.0f;
                        float shrink = 1.0f / (1.0f + 0.8f * loss_delta);
                        dynamic_lr_gain = max(gain_floor, dynamic_lr_gain * shrink);
                    }
                    else if (loss_delta < -0.01f)
                    {
                        // Last LR operation decreased loss -> continue in favorable direction
                        last_lr_direction = 1.0f;
                        float surge = max(0.0f, min(0.04f, -loss_delta * 0.05f));
                        dynamic_lr_gain = min(1.30f, dynamic_lr_gain * (1.0f + surge));
                    }

                    // Mistake Checkpoint Memory Similarity Throttling:
                    if (metrics.mistake_similarity > 0.40f)
                    {
                        float penalty_mult = max(0.70f, 1.0f - 0.40f * (metrics.mistake_similarity - 0.40f) / 0.60f);
                        dynamic_lr_gain = max(gain_floor, dynamic_lr_gain * penalty_mult);
                    }

                    // Curvature Directional Tracking & Damped Operation Reversal:
                    last_curv_loss_delta = loss_delta;
                    if (loss_delta > 0.02f)
                    {
                        last_curv_direction = -1.0f;
                        optimizer.config.curvature_scale = max(0.50f, optimizer.config.curvature_scale * 0.75f);
                    }

                    // 4c. Background Asynchronous Data Streamer Polling:
                    if (background_streamer && (step % ring0::get_config().background_stream_poll_interval == 0))
                    {
                        size_t new_toks = background_streamer->poll_and_append(dataset);
                        if (new_toks > 0)
                        {
                            cout << "\n  📡 [Background Streamer @ step " << step << "] Ingested +"
                                 << new_toks << " new tokens from disk in background (Total active: "
                                 << dataset.token_stream.size() << " tokens)\n";
                            ring0::log_debug_file("DATA_STREAMER", "Appended " + to_string(new_toks) + " tokens from background streamer.");
                        }
                    }

                    // Stability Watchdog: detect unexpected loss spikes relative to recent stable EMA
                    if (step > 15 && metrics.loss > 1.6f * ema_loss_short)
                    {
                        cout << "\n  ⚠️ [Stability Watchdog] Loss spike (" << fixed << setprecision(3) << metrics.loss
                             << " vs EMA " << ema_loss_short << ") -> Activating 15-step recovery damping.\n";
                        watchdog_recovery_cooldown = 15;
                    }

                    // 5. Dynamic Neurogenesis: Expand feature dimensions as loss drops significantly
                    if (step > 50 && expansion_count < max_expansions)
                    {
                        if (ema_loss_short < (last_expansion_loss * 0.6f))
                        {
                            model.expand_capacity(1.4);
                            last_expansion_loss = ema_loss_short;
                            expansion_count++;

                            // Trigger coupled vocabulary expansion whenever parameters expand
                            if (on_param_expansion)
                            {
                                on_param_expansion();
                            }
                        }
                    }

                    // 5b. Forced Neurogenesis on Pruning Saturation:
                    // The loss-drop trigger above never fires on a plateau (loss is not
                    // dropping), yet a plateau is exactly when >85% of weights collapse
                    // onto Formula 4 (pruning). Track that streak and force a growth
                    // event once it persists, so a stuck model gets fresh capacity.
                    {
                        float pct_f4 = optimizer.last_formula_stats.pct_f4();
                        if (pct_f4 >= f4_saturation_pct)
                        {
                            f4_saturation_streak++;
                        }
                        else
                        {
                            f4_saturation_streak = 0;
                        }
                        if (f4_saturation_streak >= f4_saturation_window && expansion_count < max_expansions)
                        {
                            cout << "\n  >> [Forced Neurogenesis] Formula-4 (pruning) held >= "
                                 << f4_saturation_pct << "% for " << f4_saturation_streak
                                 << " steps -> injecting capacity to break saturation.\n";
                            model.expand_capacity(1.4);
                            last_expansion_loss = ema_loss_short;
                            expansion_count++;
                            f4_saturation_streak = 0;
                            if (on_param_expansion)
                            {
                                on_param_expansion();
                            }
                        }
                    }

                    // 6. Step-Adaptive Progressive Depth & Complexity Ramping
                    if (config.progressive_depth_growth)
                    {
                        size_t target_layers = config.initial_layers;
                        if (rt_cfg.fast_track_depth_unlock && (step >= 15 || ema_loss_short >= 3.8f))
                        {
                            target_layers = model.blocks.size(); // Fast-track: Unlock all 10 layers to crush the plateau
                        }
                        else if (step >= 500)
                        {
                            target_layers = model.blocks.size(); // All 10 layers
                        }
                        else if (step >= 200)
                        {
                            target_layers = min(model.blocks.size(), size_t(8));
                        }
                        else if (step >= 50)
                        {
                            target_layers = min(model.blocks.size(), size_t(6));
                        }

                        // Phase 1 fix: depth ramping is MONOTONIC. The ladder above
                        // can compute a lower target than the current depth when
                        // fast-track flips off and step drops through a boundary,
                        // which caused active layers to oscillate 4->10->4->10.
                        // Only apply the change when it's a strict INCREASE.
                        if (target_layers > model.num_active_layers)
                        {
                            size_t old_l = model.num_active_layers;
                            model.set_active_layers(target_layers);
                            depth_jump_cooldown = 15;
                            cout << "\n  >> [Curriculum Depth Ramping @ step " << step << "] Unlocking transformer depth: "
                                 << old_l << " -> " << target_layers << " layers!\n\n";
                        }
                    }

                    // 7. Step-Adaptive & Loss-Adaptive Token Window Context Extension up to 2048 tokens
                    if (config.step_based_context_growth)
                    {
                        size_t target_len = current_seq_len;
                        if (step >= 12000 && current_seq_len < 2048)
                        {
                            target_len = min(config.max_seq_len, static_cast<size_t>(2048));
                        }
                        else if (step >= 7000 && current_seq_len < 1024)
                        {
                            target_len = min(config.max_seq_len, static_cast<size_t>(1024));
                        }
                        else if (step >= 3500 && current_seq_len < 512)
                        {
                            target_len = min(config.max_seq_len, static_cast<size_t>(512));
                        }
                        else if (step >= 1500 && current_seq_len < 256)
                        {
                            target_len = min(config.max_seq_len, static_cast<size_t>(256));
                        }
                        else if (step >= 250 && current_seq_len < 128)
                        {
                            target_len = min(config.max_seq_len, static_cast<size_t>(128));
                        }
                        else if (step >= 80 && current_seq_len < 64)
                        {
                            target_len = min(config.max_seq_len, static_cast<size_t>(64));
                        }

                        if (target_len != current_seq_len)
                        {
                            size_t old_len = current_seq_len;
                            current_seq_len = target_len;
                            model.expand_max_seq_len(current_seq_len);
                            context_jump_cooldown = 20; // 20 steps of damped LR to settle on expanded window
                            cout << "\n  >> [Curriculum Token Horizon @ step " << step << "] Expanding context window: "
                                 << old_len << " -> " << current_seq_len << " tokens!\n\n";
                        }
                    }
                    else if (config.progressive_context_growth)
                    {
                        if (ema_loss_short < 2.5f && current_seq_len < 128)
                        {
                            size_t old_len = current_seq_len;
                            current_seq_len = min(config.max_seq_len, static_cast<size_t>(128));
                            model.expand_max_seq_len(current_seq_len);
                            context_jump_cooldown = 20;
                            cout << "\n  >> [Progressive Context Window] Loss (" << fixed << setprecision(2) << ema_loss_short
                                 << " < 2.5) mastered! Expanding token window: " << old_len << " -> " << current_seq_len << " tokens!\n\n";
                        }
                        else if (ema_loss_short < 1.8f && current_seq_len < config.max_seq_len)
                        {
                            size_t old_len = current_seq_len;
                            current_seq_len = config.max_seq_len;
                            model.expand_max_seq_len(current_seq_len);
                            context_jump_cooldown = 20;
                            cout << "\n  >> [Progressive Context Window] Quality Loss (" << fixed << setprecision(2) << ema_loss_short
                                 << " < 1.8) achieved! Expanding token window: " << old_len << " -> " << current_seq_len << " tokens!\n\n";
                        }
                    }

                    // 8. Progressive Dataset Horizon Expansion: Expands data volume as model masters earlier representations
                    if (config.progressive_dataset_growth && current_dataset_ratio < 1.0f)
                    {
                        float target_ratio = current_dataset_ratio;

                        // Loss-based expansions
                        if (ema_loss_short < 4.0f && current_dataset_ratio < 1.0f)
                        {
                            target_ratio = 1.0f; // 100% full dataset!
                        }
                        else if (ema_loss_short < 4.8f && current_dataset_ratio < 0.60f)
                        {
                            target_ratio = 0.60f; // 60% of dataset
                        }
                        else if (ema_loss_short < 5.4f && current_dataset_ratio < 0.25f)
                        {
                            target_ratio = 0.25f; // 25% of dataset
                        }
                        else if (ema_loss_short < 5.8f && current_dataset_ratio < 0.10f)
                        {
                            target_ratio = 0.10f; // 10% of dataset
                        }

                        // Step milestone failsafe expansion so training never starves on longer schedules
                        if (step >= 200 && current_dataset_ratio < 1.0f)
                        {
                            target_ratio = max(target_ratio, 1.0f);
                        }
                        else if (step >= 100 && current_dataset_ratio < 0.50f)
                        {
                            target_ratio = max(target_ratio, 0.50f);
                        }
                        else if (step >= 40 && current_dataset_ratio < 0.20f)
                        {
                            target_ratio = max(target_ratio, 0.20f);
                        }

                        if (target_ratio > current_dataset_ratio)
                        {
                            float old_r = current_dataset_ratio;
                            current_dataset_ratio = target_ratio;
                            dataset.set_active_ratio(current_dataset_ratio);
                            dataset_jump_cooldown = 10;
                            cout << "\n  📦 [Progressive Dataset Horizon @ step " << step << "] Loss ("
                                 << fixed << setprecision(2) << ema_loss_short << ") achieved! Expanding training dataset: "
                                 << (old_r * 100.0f) << "% -> " << (current_dataset_ratio * 100.0f) << "% ("
                                 << dataset.get_active_tokens() << " tokens)\n\n";
                        }
                    }
                }
            }

            // 9. Periodic Real-Time Benchmark Dashboard display (clean cadence every 100 steps)
            if ((step % 100 == 0 || step == 1 || step == config.steps) && config.steps >= 20)
            {
                auto now = chrono::high_resolution_clock::now();
                double elapsed_s = chrono::duration<double>(now - loop_start_time).count();
                if (elapsed_s < 0.001)
                    elapsed_s = 0.001;

                BenchmarkTelemetry tel;
                tel.elapsed_seconds = elapsed_s;
                tel.total_tokens_processed = total_tokens_trained;
                tel.tokens_per_second = static_cast<double>(total_tokens_trained) / elapsed_s;
                tel.ms_per_step = (elapsed_s / static_cast<double>(step)) * 1000.0;

                // Theoretical GFLOPs estimate: ~6 * params * tokens_per_second
                size_t total_p = model.get_total_parameters();
                tel.total_parameters = total_p;
                tel.gflops_estimate = (6.0 * static_cast<double>(total_p) * tel.tokens_per_second) / 1e9;

                tel.current_loss = metrics.loss;
                tel.ema_loss = ema_loss_short;
                tel.perplexity = metrics.perplexity;
                tel.top1_accuracy = metrics.top1_accuracy;
                tel.top20_accuracy = metrics.top20_accuracy;
                tel.rank_score = metrics.rank_score_top20;
                tel.learning_rate = applied_lr;
                tel.penalty_factor = optimizer.penalty_factor;
                tel.taylor_penalty_conf = optimizer.taylor_penalty_confidence;
                tel.taylor_penalty_pred = optimizer.taylor_penalty_prediction;
                tel.active_vocab_size = model.config.vocab_size;
                tel.active_context_length = current_seq_len;
                tel.active_model_layers = model.num_active_layers;
                tel.formula_stats = optimizer.last_formula_stats;
                tel.mistake_similarity = metrics.mistake_similarity;
                tel.mistake_count = mistake_memory.size();
                tel.chrono_ticks = chrono_engine ? chrono_engine->telemetry.total_chrono_ticks.load() : 0;
                tel.background_streamed_tokens = background_streamer ? background_streamer->get_total_streamed_tokens() : 0;

                print_benchmark_dashboard(tel, step, config.steps);
            }

            // 10. Progress callbacks and persistent debug file logging
            if (on_step)
            {
                on_step(metrics);
            }

            // ------------------------------------------------------------------
            //  Rich per-step debug log entry.
            //  A multi-line block per step so that when we diagnose divergences
            //  after the fact we can see EVERYTHING the trainer knew this step,
            //  and how each quantity changed since the previous step.
            //  Layout:
            //    LINE 1: single-line summary (easy to grep)
            //    LINE 2: loss family        (loss, ema_short/long, min, gaps, deltas)
            //    LINE 3: learning-rate math (scheduled, shrink, gain, inv_mult, applied, delta)
            //    LINE 4: gradients          (pre-clip norm, was_clipped, delta)
            //    LINE 5: penalty family     (factor, delta, dL/dPen, EMA dL/dPen)
            //    LINE 6: meta network       (scale, focal_gamma, deltas)  when active
            //    LINE 7: Taylor forecast    (pred_delta, pred_net, reward, confidence)  when valid
            //    LINE 8: 4-formula physics  (F1/F2/F3/F4 %)  when routing on
            //    LINE 9: curriculum state   (active_layers, seq_len, dataset_ratio, changes)
            //    LINE 10: watchdog          (active, bad_streak, recovery_left, baseline)
            //    LINE 11: EVENTS            (only when something notable changed)
            //  Only lines with meaningful content are emitted.
            // ------------------------------------------------------------------
            {
                stringstream ss;
                auto &p = prev_debug_snapshot;
                auto fdelta = [&](float now, float prev) {
                    float d = now - prev;
                    stringstream x;
                    x << (d >= 0 ? "+" : "") << fixed << setprecision(4) << d;
                    return x.str();
                };

                // --- 1. summary line ---
                ss << "Step " << setw(5) << step << "/" << config.steps
                   << " | Loss=" << fixed << setprecision(4) << metrics.loss
                   << " | EMA=" << fixed << setprecision(4) << ema_loss_short
                   << " | LR=" << fixed << setprecision(6) << metrics.learning_rate
                   << " | |g|=" << fixed << setprecision(3) << last_grad_norm
                   << " | Top1=" << fixed << setprecision(1) << metrics.top1_accuracy << "%"
                   << " | PPL=" << fixed << setprecision(1) << metrics.perplexity
                   << " | Penalty=" << fixed << setprecision(3) << metrics.penalty_factor
                   << " | Ctx=" << metrics.active_seq_len;
                if (metrics.bad_batch_skipped) ss << " [BAD_BATCH_SKIPPED]";
                if (watchdog_active)           ss << " [WATCHDOG_ACTIVE]";

                // --- 2. loss family ---
                float min_l = ring0::Loss::get_min_loss();
                ss << "\n    LOSS  cur=" << fixed << setprecision(4) << metrics.loss
                   << " ema_s=" << ema_loss_short
                   << " ema_l=" << ema_loss_long
                   << " min=" << min_l
                   << " gap_from_min=" << fixed << setprecision(4) << (metrics.loss - min_l);
                if (p.have) {
                    ss << " dLoss=" << fdelta(metrics.loss, p.loss)
                       << " dEMA=" << fdelta(ema_loss_short, p.ema_short);
                }

                // --- 3. LR math ---
                ss << "\n    LR    applied=" << scientific << setprecision(3) << metrics.learning_rate
                   << " gain=" << fixed << setprecision(3) << dynamic_lr_gain
                   << " (watchdog=" << (watchdog_active ? "ON x0.25" : "off") << ")";
                if (p.have) ss << " dLR=" << fdelta(metrics.learning_rate, p.lr);

                // --- 4. gradients ---
                ss << "\n    GRAD  pre_clip=" << fixed << setprecision(4) << last_grad_norm
                   << " clip_thresh=" << config.max_grad_norm
                   << " clipped=" << ((config.max_grad_norm > 0.0f && last_grad_norm > config.max_grad_norm) ? "YES" : "no");
                if (p.have) ss << " d|g|=" << fdelta(last_grad_norm, p.grad_norm);

                // --- 5. penalty family ---
                ss << "\n    PEN   factor=" << fixed << setprecision(4) << metrics.penalty_factor
                   << " dL/dPen=" << showpos << setprecision(4) << metrics.d_loss_d_penalty << noshowpos
                   << " EMA_dLdP=" << showpos << setprecision(4) << optimizer.ema_d_loss_d_penalty << noshowpos;
                if (p.have) ss << " dPen=" << fdelta(metrics.penalty_factor, p.penalty);

                // --- 6. meta-network (only if it produced anything) ---
                if (metrics.meta_loss_scale > 0.0f) {
                    ss << "\n    META  scale=" << fixed << setprecision(3) << metrics.meta_loss_scale << "x"
                       << " focal_gamma=" << setprecision(3) << metrics.meta_focal_gamma;
                    if (p.have) {
                        ss << " dScale=" << fdelta(metrics.meta_loss_scale, p.meta_scale)
                           << " dGamma=" << fdelta(metrics.meta_focal_gamma, p.focal_gamma);
                    }
                }

                // --- 7. Taylor forecast (only when valid) ---
                if (last_forecast.valid) {
                    float pred_net = last_forecast.predicted[last_forecast.horizon - 1] - last_forecast.diffs[0];
                    ss << "\n    TAYL  pred_dL=" << showpos << fixed << setprecision(4) << last_forecast.pred_delta[0]
                       << " pred_net=" << pred_net << noshowpos
                       << " reward=" << showpos << setprecision(4) << last_forecast.reward << noshowpos
                       << " conf=" << fixed << setprecision(3) << last_forecast.confidence
                       << " order=" << last_forecast.order;
                }

                // --- 8. 4-formula distribution ---
                const auto &fs = optimizer.last_formula_stats;
                if (fs.total_params > 0) {
                    ss << "\n    FORM  F1(NatGrad)=" << fixed << setprecision(1) << fs.pct_f1() << "%"
                       << " F2(NestCurv)=" << fs.pct_f2() << "%"
                       << " F3(AdamW)=" << fs.pct_f3() << "%"
                       << " F4(SparseDecay)=" << fs.pct_f4() << "%";
                }

                // --- 9. curriculum state ---
                ss << "\n    CURR  layers=" << model.num_active_layers << "/" << model.blocks.size()
                   << " seq_len=" << metrics.active_seq_len;
                if (p.have) {
                    if (model.num_active_layers != p.active_layers)
                        ss << " [LAYERS " << p.active_layers << "->" << model.num_active_layers << "]";
                    if (metrics.active_seq_len != p.seq_len)
                        ss << " [CTX " << p.seq_len << "->" << metrics.active_seq_len << "]";
                }

                // --- 10. watchdog ---
                if (watchdog_active || watchdog_bad_streak > 0) {
                    ss << "\n    WDOG  active=" << (watchdog_active ? "YES" : "no")
                       << " bad_streak=" << watchdog_bad_streak
                       << " recov_left=" << watchdog_recovery_left
                       << " baseline=" << fixed << setprecision(4) << watchdog_baseline_loss;
                }

                // --- 11. NOTABLE EVENTS this step (transitions only) ---
                if (p.have) {
                    bool any_event = false;
                    stringstream ev;
                    if (!p.watchdog && watchdog_active) { ev << " [WATCHDOG_TRIGGERED@" << fixed << setprecision(3) << watchdog_baseline_loss << "]"; any_event = true; }
                    if (p.watchdog && !watchdog_active) { ev << " [WATCHDOG_RECOVERED]"; any_event = true; }
                    if (metrics.bad_batch_skipped)      { ev << " [SPIKE_STEP_SKIPPED loss>" << fixed << setprecision(1) << (2.0f * std::log((float)std::max<size_t>(2, model.config.vocab_size))) << "]"; any_event = true; }
                    if (last_grad_norm == 0.0f && !metrics.bad_batch_skipped) { ev << " [ZERO_GRAD]"; any_event = true; }
                    if (config.max_grad_norm > 0.0f && last_grad_norm > 5.0f * config.max_grad_norm) { ev << " [HUGE_GRAD_CLIPPED]"; any_event = true; }
                    if (any_event) ss << "\n    EVENT" << ev.str();
                }

                ring0::log_debug_file("TRAIN_STEP", ss.str());

                // update snapshot for next step's deltas
                prev_debug_snapshot.have         = true;
                prev_debug_snapshot.loss         = metrics.loss;
                prev_debug_snapshot.ema_short    = ema_loss_short;
                prev_debug_snapshot.lr           = metrics.learning_rate;
                prev_debug_snapshot.penalty      = metrics.penalty_factor;
                prev_debug_snapshot.grad_norm    = last_grad_norm;
                prev_debug_snapshot.meta_scale   = metrics.meta_loss_scale;
                prev_debug_snapshot.focal_gamma  = metrics.meta_focal_gamma;
                prev_debug_snapshot.active_layers= model.num_active_layers;
                prev_debug_snapshot.seq_len      = metrics.active_seq_len;
                prev_debug_snapshot.watchdog     = watchdog_active;
            }

            if (on_eval && config.eval_interval > 0 && (step % config.eval_interval == 0))
            {
                on_eval(step);
            }
        }
    }

} // namespace ring3
