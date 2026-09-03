#include "ring1/meta_loss_optimizer.hpp"
#include <cmath>
#include <iostream>
#include <iomanip>
#include <random>
#include <algorithm>

namespace ring1
{

    static float meta_gelu(float x)
    {
        return 0.5f * x * (1.0f + std::tanh(0.79788456f * (x + 0.044715f * x * x * x)));
    }

    static float meta_gelu_deriv(float x)
    {
        float c = 0.79788456f;
        float arg = c * (x + 0.044715f * x * x * x);
        float t = std::tanh(arg);
        float dt = 1.0f - t * t;
        float darg = c * (1.0f + 3.0f * 0.044715f * x * x);
        return 0.5f * (1.0f + t) + 0.5f * x * dt * darg;
    }

    static float meta_sigmoid(float x)
    {
        return 1.0f / (1.0f + std::exp(-x));
    }

    MetaLossOptimizer::MetaLossOptimizer()
    {
        reset();
    }

    void MetaLossOptimizer::reset()
    {
        std::mt19937 rng(42);
        std::normal_distribution<float> dist(0.0f, 0.1f);

        // Initialize W1: 12 -> 32  (8 base telemetry + 4 Taylor foresight features)
        W1 = ring0::Matrix(META_INPUT_DIM, 32);
        b1 = ring0::Matrix::zeros(1, 32);
        for (float &w : W1.data)
            w = dist(rng) * std::sqrt(2.0f / static_cast<float>(META_INPUT_DIM));

        // Initialize W2: 32 -> 16
        W2 = ring0::Matrix(32, 16);
        b2 = ring0::Matrix::zeros(1, 16);
        for (float &w : W2.data)
            w = dist(rng) * std::sqrt(2.0f / 32.0f);

        // Initialize W3: 16 -> 4
        W3 = ring0::Matrix(16, 4);
        b3 = ring0::Matrix::zeros(1, 4);
        for (float &w : W3.data)
            w = dist(rng) * std::sqrt(2.0f / 16.0f);

        last_input.assign(META_INPUT_DIM, 0.0f); // zero-init (reset() runs from the ctor when last_input is still empty)
        last_h1.assign(32, 0.0f);
        last_h2.assign(16, 0.0f);
        last_output = MetaOptimizationOutput{};

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

        last_input[0] = std::tanh((cur_l - 3.0f) * 0.5f);
        last_input[1] = std::tanh(del_l * 2.0f);
        last_input[2] = std::tanh(acc_l * 5.0f);
        last_input[3] = std::tanh(d_pen);
        last_input[4] = std::tanh(g_var * 10.0f);
        last_input[5] = std::clamp(l_alg, 0.0f, 1.0f);
        last_input[6] = std::tanh((t_ent - 3.5f) * 0.5f);
        last_input[7] = std::tanh(l_rat * 100.0f);

        // Taylor loss-trajectory foresight features (guarded)
        float p_del = (std::isnan(telemetry.predicted_delta) || std::isinf(telemetry.predicted_delta)) ? 0.0f : telemetry.predicted_delta;
        float p_net = (std::isnan(telemetry.predicted_net) || std::isinf(telemetry.predicted_net)) ? 0.0f : telemetry.predicted_net;
        float t_rew = (std::isnan(telemetry.trajectory_reward) || std::isinf(telemetry.trajectory_reward)) ? 0.0f : telemetry.trajectory_reward;
        float t_cnf = (std::isnan(telemetry.trajectory_confidence) || std::isinf(telemetry.trajectory_confidence)) ? 0.0f : telemetry.trajectory_confidence;
        last_input[8] = std::tanh(p_del * 2.0f);
        last_input[9] = std::tanh(p_net * 1.5f);
        last_input[10] = std::tanh(t_rew * 1.5f);
        last_input[11] = std::clamp(t_cnf, 0.0f, 1.0f);

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

        // Map outputs to domain ranges with strict clamp:
        last_output.loss_scale_multiplier = std::clamp(0.2f + 3.8f * meta_sigmoid(logits[0]), 0.2f, 4.0f);
        last_output.dynamic_focal_gamma = std::clamp(3.0f * meta_sigmoid(logits[1]), 0.0f, 3.0f);
        last_output.lr_step_modulator = std::clamp(0.5f + 2.5f * meta_sigmoid(logits[2]), 0.5f, 3.0f);
        last_output.curvature_scale = std::clamp(0.2f + 2.3f * meta_sigmoid(logits[3]), 0.2f, 2.5f);

        return last_output;
    }

    void MetaLossOptimizer::update_online(float current_loss)
    {
        if (std::isnan(current_loss) || std::isinf(current_loss) || current_loss <= 0.0f)
        {
            return; // Ignore corrupt loss steps
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

        // Reward: Loss reduction is positive reward, loss explosion is negative penalty
        float reward = -delta_loss;
        apply_policy_gradient(reward);

        prev_loss = current_loss;
        step_count++;
    }

    // One policy-gradient-style step across every layer, INCLUDING W1 so the input
    // features (base telemetry + Taylor foresight) actually shape the hidden code.
    // Previously W1 was frozen, which is a big reason the outputs stayed pinned at
    // their initialization (focal gamma=1.50, loss-scale=2.10x for all steps).
    void MetaLossOptimizer::apply_policy_gradient(float reward)
    {
        float grad_scale = std::max(-2.0f, std::min(2.0f, reward)) * meta_lr;

        // Output heads W3 / b3
        for (size_t j = 0; j < 4; ++j)
        {
            b3.data[j] = std::clamp(b3.data[j] + grad_scale * 0.1f, -10.0f, 10.0f);
            for (size_t i = 0; i < 16; ++i)
            {
                float new_w = W3.data[i * 4 + j] + grad_scale * last_h2[i];
                W3.data[i * 4 + j] = std::clamp(new_w, -10.0f, 10.0f);
            }
        }

        // Hidden layer W2 / b2
        for (size_t j = 0; j < 16; ++j)
        {
            b2.data[j] = std::clamp(b2.data[j] + grad_scale * 0.05f, -10.0f, 10.0f);
            for (size_t i = 0; i < 32; ++i)
            {
                float new_w = W2.data[i * 16 + j] + grad_scale * last_h1[i] * 0.5f;
                W2.data[i * 16 + j] = std::clamp(new_w, -10.0f, 10.0f);
            }
        }

        // Input layer W1 / b1 (was previously never trained). Uses the cached input
        // activations so the telemetry + foresight features gain a learnable path.
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
