#include "ring1/meta_loss_optimizer.hpp"
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
        last_input.assign(META_INPUT_DIM, 0.0f); // zero-init (reset() runs from the ctor when last_input is still empty)
        last_h1.assign(32, 0.0f);                // cached hidden-layer-1 activations
        last_h2.assign(16, 0.0f);                // cached hidden-layer-2 activations
        last_output = MetaOptimizationOutput{};  // last knob values produced

        prev_loss *= .3f; // decay previous loss to avoid large initial reward spike
        prev_delta_loss *= 0.2f;
        step_count = 0;
        meta_lr = 0.2f; // raised from 0.005: the reward signal near a plateau is tiny,
                        // so a larger meta step is needed for the policy to actually move.
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
        // Each is squashed into roughly [-1,1] (via tanh) or [0,1] so no single raw
        // metric dominates the first layer. The constants center/scale each signal
        // around its typical operating range.
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
        // --- Taylor foresight features [8..11] (from ring0::TaylorTrajectoryPredictor) ---
        // These let the network react to where loss is HEADING, not just where it is.
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

        // Exploration noise on the output logits: breaks the "stuck at sigmoid(0)"
        // symmetry so the heads actually explore instead of pinning at their init
        // values (gamma=1.50, scale=2.10x forever). Amplitude anneals slowly with
        // step_count but never fully vanishes, keeping the meta-policy alive.
        {
            static thread_local std::mt19937 expl_rng(1337);
            std::normal_distribution<float> jitter(0.0f, 1.0f);
            float anneal = 1.0f / (1.0f + 0.002f * static_cast<float>(step_count)); // 1.0 -> ~0.33 over ~1000 steps
            float noise_amp = 0.35f * anneal + 0.05f;                               // floor of 0.05 so it never dies
            for (size_t j = 0; j < 4; ++j)
            {
                logits[j] = std::clamp(logits[j] + noise_amp * jitter(expl_rng), -30.0f, 30.0f);
            }
        }

        // Map each raw logit through sigmoid -> (0,1), then rescale to that knob's
        // physical range and hard-clamp. sigmoid(0)=0.5, so an untrained network sits
        // at the MIDDLE of every range (the exploration noise above nudges it off-center).
        last_output.loss_scale_multiplier = std::clamp(0.2f + 3.8f * meta_sigmoid(logits[0]), 0.2f, 4.0f); // gradient push [0.2, 4.0]
        last_output.dynamic_focal_gamma = std::clamp(3.0f * meta_sigmoid(logits[1]), 0.0f, 3.0f);          // hard-token focus [0.0, 3.0]
        last_output.lr_step_modulator = std::clamp(0.5f + 2.5f * meta_sigmoid(logits[2]), 0.5f, 3.0f);     // LR multiplier [0.5, 3.0]
        last_output.curvature_scale = std::clamp(0.2f + 2.3f * meta_sigmoid(logits[3]), 0.2f, 2.5f);       // step preconditioning [0.2, 2.5]

        return last_output;
    }

    // Basic online update: reward = how much loss dropped since last step.
    // (Overload below adds Taylor-forecast foresight to the reward.)
    void MetaLossOptimizer::update_online(float current_loss)
    {
        // Guard against corrupt loss values (NaN/Inf/non-positive) poisoning the policy.
        if (std::isnan(current_loss) || std::isinf(current_loss) || current_loss <= 0.0f)
        {
            return; // Ignore corrupt loss steps
        }

        // First real observation: just record it, no reward yet (need two points for a delta).
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

        // Reward is the NEGATIVE loss change: loss dropped -> reward>0 (reinforce the
        // knobs we just used); loss rose -> reward<0 (push knobs the other way).
        float reward = -delta_loss;
        apply_policy_gradient(reward);

        prev_loss = current_loss; // remember for next step's delta
        step_count++;
    }

    // One policy-gradient-style step across every layer, INCLUDING W1 so the input
    // features (base telemetry + Taylor foresight) actually shape the hidden code.
    // Previously W1 was frozen, which is a big reason the outputs stayed pinned at
    // their initialization (focal gamma=1.50, loss-scale=2.10x for all steps).
    // One REINFORCE-style policy-gradient step across EVERY layer (W1, W2, W3).
    //
    // Intuition: we don't have a true gradient of "future loss" w.r.t. these weights,
    // so we approximate it. Each weight is nudged in proportion to (a) the reward and
    // (b) the activation that fed into it last step. A positive reward reinforces
    // whatever pattern of activations produced the last knobs; a negative reward
    // reverses it. The per-layer 0.1 / 0.05 / 0.025 scales are small "learning-rate
    // fractions" that shrink toward the input side so early layers move more gently.
    // Everything is clamped to +/-10 to keep the tiny network from ever diverging.
    void MetaLossOptimizer::apply_policy_gradient(float reward)
    {
        // Clip reward to [-2,2] so one wild loss jump can't produce a giant weight step,
        // then scale by the meta learning rate. This is the common (reward * activation)
        // magnitude applied to every weight below.
        float grad_scale = std::max(-2.0f, std::min(2.0f, reward)) * meta_lr;

        // Output heads W3 / b3  (16 -> 4): weight_{i,j} += grad_scale * h2_i
        for (size_t j = 0; j < 4; ++j)
        {
            b3.data[j] = std::clamp(b3.data[j] + grad_scale * 0.1f, -10.0f, 10.0f);
            for (size_t i = 0; i < 16; ++i)
            {
                float new_w = W3.data[i * 4 + j] + grad_scale * last_h2[i];
                W3.data[i * 4 + j] = std::clamp(new_w, -10.0f, 10.0f);
            }
        }

        // Hidden layer W2 / b2  (32 -> 16): scaled 0.5x vs the head so mid-layers move less.
        for (size_t j = 0; j < 16; ++j)
        {
            b2.data[j] = std::clamp(b2.data[j] + grad_scale * 0.05f, -10.0f, 10.0f);
            for (size_t i = 0; i < 32; ++i)
            {
                float new_w = W2.data[i * 16 + j] + grad_scale * last_h1[i] * 0.5f;
                W2.data[i * 16 + j] = std::clamp(new_w, -10.0f, 10.0f);
            }
        }

        // Input layer W1 / b1  (12 -> 32): scaled 0.25x. CRITICAL — this layer used to be
        // frozen, which meant the 12 input features (telemetry + Taylor foresight) had no
        // learnable path and the outputs stayed pinned at their init values forever
        // (focal gamma=1.50, loss-scale=2.10x for every step). Training it is what makes
        // the meta-network actually adapt to the inputs.
        for (size_t j = 0; j < 32; ++j)
        {
            b1.data[j] = std::clamp(b1.data[j] + grad_scale * 0.025f, -10.0f, 10.0f);
            for (size_t i = 0; i < META_INPUT_DIM; ++i)
            {
                float new_w = W1.data[i * 32 + j] + grad_scale * last_input[i] * 0.25f;
                W1.data[i * 32 + j] = std::clamp(new_w, -10.0f, 10.0f);
            }
        }
    }

    // Foresight-augmented online update. Same as the basic version, but the reward is a
    // blend of the REALIZED drop (what actually happened last step) and the Taylor
    // PREDICTED trajectory reward (what's forecast to happen next). foresight_weight in
    // [0,1] sets the mix (0.5 = equal). This rewards the policy for setting up a good
    // future path, not just a good last step.
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

        // Blend realized reward with Taylor-predicted trajectory reward: the network
        // is optimized for the whole forecast path, not just the last observed step.
        float realized_reward = -delta_loss;
        float fw = std::clamp(foresight_weight, 0.0f, 1.0f);
        float traj = (std::isnan(trajectory_reward) || std::isinf(trajectory_reward)) ? 0.0f : trajectory_reward;
        float reward = (1.0f - fw) * realized_reward + fw * traj;
        apply_policy_gradient(reward);

        prev_loss = current_loss;
        step_count++;
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
