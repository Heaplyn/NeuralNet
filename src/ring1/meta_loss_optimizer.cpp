#include "ring1/meta_loss_optimizer.hpp"
#include "ring0/config.hpp"
#include <cmath>
#include <iostream>
#include <iomanip>
#include <random>
#include <algorithm>

namespace ring1
{
    // =====================================================================================
    //  META-NEURAL LOSS & STEP OPTIMIZER  —  a tiny neural network that tunes the BIG one.
    // -------------------------------------------------------------------------------------
    //  WHAT IT IS
    //    A small 3-layer MLP (12 -> 32 -> 16 -> 4) that runs once per training step. It
    //    watches how the main transformer's training is going (loss, its velocity and
    //    acceleration, gradient statistics, a Taylor forecast of where loss is heading,
    //    etc.) and outputs four "control knobs" that reshape the optimization on the fly:
    //        (1) loss_scale_multiplier - how hard to push cross-entropy gradients
    //        (2) dynamic_focal_gamma   - how much to focus on hard/unlearned tokens
    //        (3) lr_step_modulator     - a learning-rate multiplier
    //        (4) curvature_scale       - step preconditioning based on landscape shape
    //
    //  HOW IT LEARNS (no backprop through the transformer!)
    //    It uses a crude but cheap ONLINE POLICY GRADIENT. Each step it gets a scalar
    //    "reward" = how much the loss dropped (optionally blended with the Taylor forecast
    //    of future drops). If the last set of knobs was followed by a loss drop, it nudges
    //    its weights to produce more of that; if loss rose, it nudges the other way. This
    //    is a REINFORCE-style update, not exact gradients — good enough to steer knobs.
    //
    //  WHY IT EXISTS
    //    Hand-coded schedules (cosine LR, fixed focal gamma, etc.) can't tell whether the
    //    model is in a steep valley, a flat plateau, or a chaotic saddle. This network
    //    adapts the knobs to the *actual* live training dynamics.
    // =====================================================================================

    // ---- Activation helpers (all operate on a single scalar) --------------------------

    // GELU (Gaussian Error Linear Unit), tanh approximation. Smooth, near-linear for
    // large +x, softly gates toward 0 for -x. Used on the two hidden layers.
    static float meta_gelu(float x)
    {
        return 0.5f * x * (1.0f + std::tanh(0.79788456f * (x + 0.044715f * x * x * x)));
    }

    // Analytic derivative of meta_gelu. Kept for a future exact-gradient update path;
    // the current online update is policy-gradient style and does not call it.
    static float meta_gelu_deriv(float x)
    {
        float c = 0.79788456f;
        float arg = c * (x + 0.044715f * x * x * x);
        float t = std::tanh(arg);
        float dt = 1.0f - t * t;
        float darg = c * (1.0f + 3.0f * 0.044715f * x * x);
        return 0.5f * (1.0f + t) + 0.5f * x * dt * darg;
    }

    // Logistic sigmoid, squashes any real number into (0,1). Used to map the 4 raw
    // output logits into bounded [0,1] fractions, which are then rescaled to each
    // knob's physical range (see predict()).
    static float meta_sigmoid(float x)
    {
        return 1.0f / (1.0f + std::exp(-x));
    }

    MetaLossOptimizer::MetaLossOptimizer()
    {
        reset();
    }

    // Re-initializes all weights and caches. Called from the constructor and any time
    // the meta-network should start fresh. Weights use He/Kaiming-style init
    // (std = sqrt(2/fan_in)) so activations neither vanish nor explode through the GELU
    // layers. A FIXED rng seed (42) makes runs reproducible.
    void MetaLossOptimizer::reset()
    {
        std::mt19937 rng(42);
        std::normal_distribution<float> dist(0.0f, 0.1f);

        // Layer 1: input (12) -> hidden (32). 12 inputs = 8 base telemetry signals
        // + 4 Taylor loss-trajectory foresight signals. fan_in = 12.
        W1 = ring0::Matrix(META_INPUT_DIM, 32);
        b1 = ring0::Matrix::zeros(1, 32);
        for (float &w : W1.data)
            w = dist(rng) * std::sqrt(2.0f / static_cast<float>(META_INPUT_DIM));

        // Layer 2: hidden (32) -> hidden (16). fan_in = 32.
        W2 = ring0::Matrix(32, 16);
        b2 = ring0::Matrix::zeros(1, 16);
        for (float &w : W2.data)
            w = dist(rng) * std::sqrt(2.0f / 32.0f);

        // Layer 3: hidden (16) -> 4 output heads (the four control knobs). fan_in = 16.
        W3 = ring0::Matrix(16, 4);
        b3 = ring0::Matrix::zeros(1, 4);
        for (float &w : W3.data)
            w = dist(rng) * std::sqrt(2.0f / 16.0f);

        // Caches of the last forward pass (needed by the online policy-gradient update,
        // which reuses these activations instead of recomputing them).
        last_input.assign(META_INPUT_DIM, 0.0f);                      // zero-init
        last_h1.assign(32, 0.0f);                                     // cached hidden-layer-1 activations
        last_h2.assign(16, 0.0f);                                     // cached hidden-layer-2 activations
        last_output = MetaOptimizationOutput{1.0f, 0.0f, 1.0f, 1.0f}; // neutral init
        recent_loss_scales.clear();

        prev_loss = 0.0f;
        prev_delta_loss = 0.0f;
        prev_delta_loss_for_meta = 0.0f;
        meta_step_scale = 1.0f;
        step_count = 0;
        update_stride = 4; // only update policy every 4 steps to avoid noise buildup
        meta_lr = 0.005f;  // slashed learning rate for smooth meta-policy updates
    }

    float MetaLossOptimizer::compute_output_variance() const
    {
        if (recent_loss_scales.size() < 5)
            return 0.0f;
        float sum = 0.0f;
        for (float v : recent_loss_scales)
            sum += v;
        float mean = sum / static_cast<float>(recent_loss_scales.size());
        float sq_diff = 0.0f;
        for (float v : recent_loss_scales)
            sq_diff += (v - mean) * (v - mean);
        return sq_diff / static_cast<float>(recent_loss_scales.size());
    }

    bool MetaLossOptimizer::is_healthy() const
    {
        float var = compute_output_variance();
        // Stricter than before
        return var < 0.008f && meta_step_scale > 0.35f;
    }

    MetaOptimizationOutput MetaLossOptimizer::predict(const MetaLossTelemetry &telemetry)
    {
        // 1. Prepare normalized input vector (dim: 12) with strict NaN guards
        last_input.resize(META_INPUT_DIM);
        float cur_l = (std::isnan(telemetry.current_loss) || std::isinf(telemetry.current_loss)) ? 4.5f : telemetry.current_loss;
        float del_l = (std::isnan(telemetry.delta_loss) || std::isinf(telemetry.delta_loss)) ? 0.0f : telemetry.delta_loss;
        float acc_l = (std::isnan(telemetry.accel_loss) || std::isinf(telemetry.accel_loss)) ? 0.0f : telemetry.accel_loss;
        float d_pen = (std::isnan(telemetry.d_loss_d_penalty) || std::isinf(telemetry.d_loss_d_penalty)) ? 0.0f : telemetry.d_loss_d_penalty;
        float g_var = (std::isnan(telemetry.gradient_variance) || std::isinf(telemetry.gradient_variance)) ? 0.01f : telemetry.gradient_variance;
        float l_alg = (std::isnan(telemetry.layer_alignment) || std::isinf(telemetry.layer_alignment)) ? 0.85f : telemetry.layer_alignment;
        float t_ent = (std::isnan(telemetry.token_entropy) || std::isinf(telemetry.token_entropy)) ? 3.5f : telemetry.token_entropy;
        float l_rat = (std::isnan(telemetry.learning_rate) || std::isinf(telemetry.learning_rate)) ? 0.001f : telemetry.learning_rate;

        // --- Base telemetry features [0..7] ---
        last_input[0] = std::tanh((cur_l - 3.0f) * 0.5f); // current loss, centered at 3.0
        last_input[1] = std::tanh(del_l * 2.0f);          // loss velocity dL/dt (falling<0, rising>0)
        last_input[2] = std::tanh(acc_l * 5.0f);          // loss acceleration d^2L/dt^2 (curvature)
        last_input[3] = std::tanh(d_pen);                 // sensitivity of loss to the penalty factor
        last_input[4] = std::tanh(g_var * 10.0f);         // gradient variance (noisiness of the step)
        last_input[5] = std::clamp(l_alg, 0.0f, 1.0f);    // layer-shift alignment (0=chaotic .. 1=coherent)
        last_input[6] = std::tanh((t_ent - 3.5f) * 0.5f); // token prediction entropy, centered at 3.5
        last_input[7] = std::tanh(l_rat * 100.0f);        // current learning rate (scaled up since LR is tiny)

        // Taylor loss-trajectory foresight features (guarded)
        float p_del = (std::isnan(telemetry.predicted_delta) || std::isinf(telemetry.predicted_delta)) ? 0.0f : telemetry.predicted_delta;
        float p_net = (std::isnan(telemetry.predicted_net) || std::isinf(telemetry.predicted_net)) ? 0.0f : telemetry.predicted_net;
        float t_rew = (std::isnan(telemetry.trajectory_reward) || std::isinf(telemetry.trajectory_reward)) ? 0.0f : telemetry.trajectory_reward;
        float t_cnf = (std::isnan(telemetry.trajectory_confidence) || std::isinf(telemetry.trajectory_confidence)) ? 0.0f : telemetry.trajectory_confidence;
        last_input[8] = std::tanh(p_del * 2.0f);        // predicted next-step loss change
        last_input[9] = std::tanh(p_net * 1.5f);        // predicted net change over the whole horizon
        last_input[10] = std::tanh(t_rew * 1.5f);       // discounted trajectory reward (higher = better path ahead)
        last_input[11] = std::clamp(t_cnf, 0.0f, 1.0f); // forecast confidence (how much to trust [8..10])

        // 2. Layer 1: 12 -> 32 + GELU
        last_h1.assign(32, 0.0f);
        for (size_t j = 0; j < 32; ++j)
        {
            float sum = b1.data[j];
            for (size_t i = 0; i < META_INPUT_DIM; ++i)
            {
                sum += last_input[i] * W1.data[i * 32 + j];
            }
            if (std::isnan(sum) || std::isinf(sum))
                sum = 0.0f;
            sum = std::clamp(sum, -30.0f, 30.0f);
            last_h1[j] = meta_gelu(sum);
        }

        // 3. Layer 2: 32 -> 16 + GELU
        last_h2.assign(16, 0.0f);
        for (size_t j = 0; j < 16; ++j)
        {
            float sum = b2.data[j];
            for (size_t i = 0; i < 32; ++i)
            {
                sum += last_h1[i] * W2.data[i * 16 + j];
            }
            if (std::isnan(sum) || std::isinf(sum))
                sum = 0.0f;
            sum = std::clamp(sum, -30.0f, 30.0f);
            last_h2[j] = meta_gelu(sum);
        }

        // 4. Layer 3: 16 -> 4 Output Heads
        std::vector<float> logits(4, 0.0f);
        for (size_t j = 0; j < 4; ++j)
        {
            float sum = b3.data[j];
            for (size_t i = 0; i < 16; ++i)
            {
                sum += last_h2[i] * W3.data[i * 4 + j];
            }
            if (std::isnan(sum) || std::isinf(sum))
                sum = 0.0f;
            logits[j] = std::clamp(sum, -30.0f, 30.0f);
        }

        // Exploration noise on the output logits (very small, strictly bounded)
        {
            static thread_local std::mt19937 expl_rng(1337);
            std::normal_distribution<float> jitter(0.0f, 1.0f);
            float anneal = 1.0f / (1.0f + 0.002f * static_cast<float>(step_count));
            float noise_amp = 0.03f * anneal + 0.01f;
            for (size_t j = 0; j < 4; ++j)
            {
                logits[j] = std::clamp(logits[j] + noise_amp * jitter(expl_rng), -30.0f, 30.0f);
            }
        }
        float scale_lo = 0.88f;
        float scale_hi = 1.18f;

        if (telemetry.current_loss < 6.0f || meta_step_scale < 0.5f)
        {
            scale_lo = 0.92f;
            scale_hi = 1.12f;
        }

        float raw_loss_scale = std::clamp(scale_lo + (scale_hi - scale_lo) * meta_sigmoid(logits[0]), scale_lo, scale_hi);
        // Hard-clamped tight ranges: prevent runaway multipliers from destabilizing training
        // float raw_loss_scale = std::clamp(0.85f + 0.40f * meta_sigmoid(logits[0]), 0.85f, 1.25f);
        float raw_focal = std::clamp(0.35f * meta_sigmoid(logits[1]), 0.0f, 0.35f);
        float raw_lr_mod = std::clamp(0.85f + 0.35f * meta_sigmoid(logits[2]), 0.85f, 1.20f);
        float raw_curv = std::clamp(0.80f + 0.40f * meta_sigmoid(logits[3]), 0.80f, 1.20f);

        // OUTPUT SMOOTHING (EMA): heavier smoothing when loss is lower or model is deeper
        float alpha = (telemetry.current_loss < 6.0f) ? 0.05f : 0.12f;
        last_output.loss_scale_multiplier = (1.0f - alpha) * last_output.loss_scale_multiplier + alpha * raw_loss_scale;
        last_output.dynamic_focal_gamma = (1.0f - alpha) * last_output.dynamic_focal_gamma + alpha * raw_focal;
        last_output.lr_step_modulator = (1.0f - alpha) * last_output.lr_step_modulator + alpha * raw_lr_mod;
        last_output.curvature_scale = (1.0f - alpha) * last_output.curvature_scale + alpha * raw_curv;

        // Track rolling variance for health monitoring
        recent_loss_scales.push_back(last_output.loss_scale_multiplier);
        if (recent_loss_scales.size() > ROLLING_VARIANCE_WINDOW)
        {
            recent_loss_scales.pop_front();
        }

        // Meta Health Monitor: if output variance spikes, pull outputs back toward neutral (1.0 / 0.0)
        if (!is_healthy())
        {
            last_output.loss_scale_multiplier = 0.82f * last_output.loss_scale_multiplier + 0.18f * 1.0f;
            last_output.dynamic_focal_gamma = 0.82f * last_output.dynamic_focal_gamma + 0.18f * 0.0f;
            last_output.lr_step_modulator = 0.82f * last_output.lr_step_modulator + 0.18f * 1.0f;
            last_output.curvature_scale = 0.82f * last_output.curvature_scale + 0.18f * 1.0f;

            meta_step_scale *= 0.7f; // also decelerate learning
        }

        return last_output;
    }

    void MetaLossOptimizer::apply_policy_gradient(float reward)
    {
        // Clip reward to [-1.5, 1.5]
        float grad_scale = std::max(-1.5f, std::min<float>(1.5f, reward)) * meta_lr * meta_step_scale;

        // Output heads W3 / b3  (16 -> 4): weight_{i,j} += grad_scale * h2_i
        for (size_t j = 0; j < 4; ++j)
        {
            b3.data[j] = std::clamp(b3.data[j] + grad_scale * 0.08f, -6.0f, 6.0f);
            for (size_t i = 0; i < 16; ++i)
            {
                float new_w = W3.data[i * 4 + j] + grad_scale * last_h2[i];
                W3.data[i * 4 + j] = std::clamp(new_w, -6.0f, 6.0f);
            }
        }

        // Hidden layer W2 / b2  (32 -> 16)
        for (size_t j = 0; j < 16; ++j)
        {
            b2.data[j] = std::clamp(b2.data[j] + grad_scale * 0.04f, -6.0f, 6.0f);
            for (size_t i = 0; i < 32; ++i)
            {
                float new_w = W2.data[i * 16 + j] + grad_scale * last_h1[i] * 0.5f;
                W2.data[i * 16 + j] = std::clamp(new_w, -6.0f, 6.0f);
            }
        }

        // Input layer W1 / b1  (12 -> 32)
        for (size_t j = 0; j < 32; ++j)
        {
            b1.data[j] = std::clamp(b1.data[j] + grad_scale * 0.02f, -6.0f, 6.0f);
            for (size_t i = 0; i < META_INPUT_DIM; ++i)
            {
                float new_w = W1.data[i * 32 + j] + grad_scale * last_input[i] * 0.25f;
                W1.data[i * 32 + j] = std::clamp(new_w, -6.0f, 6.0f);
            }
        }
    }

    void MetaLossOptimizer::update_online(float current_loss)
    {
        update_online(current_loss, 0.0f, 0.0f);
    }

    void MetaLossOptimizer::update_online(float current_loss, float trajectory_reward, float foresight_weight)
    {
        if (std::isnan(current_loss) || std::isinf(current_loss) || current_loss <= 0.0f)
        {
            return;
        }
        if (prev_loss == 0.0f)
        {
            prev_loss = current_loss;
            return;
        }

        float delta_loss = current_loss - prev_loss;
        if (std::isnan(delta_loss) || std::isinf(delta_loss))
        {
            prev_loss = current_loss;
            return;
        }
        float current_delta = current_loss - prev_loss;
        float predicted_future_delta = current_delta + 0.5f * (current_delta - prev_delta_loss_for_meta);
        // How did the loss velocity change since last meta update?
        float meta_effect = current_delta - prev_delta_loss_for_meta;
        if (meta_effect > 0.02f)
        {
            meta_step_scale *= 0.65f; // strong deceleration
        }
        else if (meta_effect < -0.02f)
        {
            meta_step_scale = std::min<float>(1.0f, meta_step_scale * 1.08f); // slow recovery
        }
        else
        {
            meta_step_scale = std::min<float>(1.0f, meta_step_scale * 1.02f); // mild recovery
        }

        meta_step_scale = std::clamp(meta_step_scale, 0.12f, 1.0f);
        prev_delta_loss_for_meta = current_delta;
        // Self-lowering upon bad loss: if loss rose, lower meta_lr and pull outputs towards neutral
        if (delta_loss > 0.01f)
        {
            meta_lr = std::max(0.001f, meta_lr * 0.85f);
            last_output.loss_scale_multiplier = std::max(0.85f, last_output.loss_scale_multiplier * 0.95f);
            last_output.dynamic_focal_gamma = std::max(0.0f, last_output.dynamic_focal_gamma * 0.80f);
            last_output.lr_step_modulator = std::max(0.85f, last_output.lr_step_modulator * 0.95f);
            last_output.curvature_scale = std::max(0.80f, last_output.curvature_scale * 0.95f);
        }
        else if (delta_loss < -0.01f)
        {
            meta_lr = std::min(0.008f, meta_lr * 1.02f);
        }

        // Blend realized reward with Taylor-predicted trajectory reward:
        // the network is trained on both recent delta and the Taylor foresight horizon
        float realized_reward = -delta_loss;
        float fw = std::clamp(foresight_weight, 0.0f, 1.0f);
        float traj = (std::isnan(trajectory_reward) || std::isinf(trajectory_reward)) ? 0.0f : trajectory_reward;
        float reward = (1.0f - fw) * realized_reward + fw * traj;

        step_count++;

        // Strided updates: only update policy gradient every update_stride steps (every 4 steps)
        if (step_count % update_stride == 0)
        {
            apply_policy_gradient(reward);
        }

        prev_loss = current_loss;
    }

    void MetaLossOptimizer::print_status() const
    {
        std::cout << "  🧠 [Meta-Neural Loss Engine] Step: " << step_count
                  << " | LossScale: " << std::fixed << std::setprecision(2) << last_output.loss_scale_multiplier << "x"
                  << " | FocalGamma: " << last_output.dynamic_focal_gamma
                  << " | LR-Mod: " << last_output.lr_step_modulator << "x"
                  << " | Curvature: " << last_output.curvature_scale << "x\n";
    }

} // namespace ring1
