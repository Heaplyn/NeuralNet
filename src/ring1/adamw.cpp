#include "ring1/adamw.hpp"
#include "ring0/loss.hpp"
#include <algorithm>
#include <cmath>

using namespace std;

namespace ring1
{

    // Constructor
    AdamW::AdamW(AdamWConfig cfg) : config(cfg), timestep(0) {}

    // Registers parameter tensor and initializes corresponding zero-filled moment buffers
    void AdamW::register_param(const ring0::Matrix &param, bool apply_weight_decay)
    {
        m_list.push_back(ring0::Matrix::zeros(param.rows, param.cols));
        v_list.push_back(ring0::Matrix::zeros(param.rows, param.cols));
        fisher_diag.push_back(ring0::Matrix::zeros(param.rows, param.cols));
        decay_mask.push_back(apply_weight_decay);
        last_shifts.push_back(ring0::Matrix::zeros(param.rows, param.cols));
        layer_scales.push_back(1.0f);
        layer_attributions.push_back(0.0f);
    }

    // Resets timestep counter and clears all moment buffers
    void AdamW::reset()
    {
        timestep = 0;
        for (auto &m : m_list)
            m = ring0::Matrix::zeros(m.rows, m.cols);
        for (auto &v : v_list)
            v = ring0::Matrix::zeros(v.rows, v.cols);
        for (auto &f : fisher_diag)
            f = ring0::Matrix::zeros(f.rows, f.cols);
        for (auto &s : last_shifts)
            s = ring0::Matrix::zeros(s.rows, s.cols);
        fill(layer_scales.begin(), layer_scales.end(), 1.0f);
        fill(layer_attributions.begin(), layer_attributions.end(), 0.0f);
        decay_mask.clear();
    }

    void AdamW::set_learning_rate(float lr)
    {
        config.lr = lr;
    }

    float AdamW::get_learning_rate() const
    {
        return config.lr;
    }

    // Dynamically adjusts learning rate based on loss gap with loss magnitude multiplier
    void AdamW::adjust_by_loss_gap(float current_loss, float min_loss, float multiplier)
    {
        float loss_mult = ring0::Loss::compute_loss_scale_multiplier(current_loss);
        float gap = max(0.0f, current_loss - min_loss) * multiplier * loss_mult;
        config.lr *= (1.0f + gap);
    }

    // Self-adjusts optimizer hyperparameters (lr, momentum beta1, decay) with stochastic exploration jitter
    void AdamW::self_adjust_by_loss(float current_loss, float ema_loss, float min_loss)
    {
        if (ema_loss <= 0.0f || current_loss <= 0.0f)
            return;

        float delta = (current_loss - ema_loss) * .7f;
        float loss_mult = ring0::Loss::compute_loss_scale_multiplier(current_loss);

        // Slight stochastic exploration discrepancy (+/- 2.5%) to prevent getting stuck in local saddle points / local maxima
        float stochastic_jitter = 1.0f + (((static_cast<float>(rand() % 1000) / 1000.0f) - 0.5f) * 0.05f);

        if (delta < -0.01f)
        {
            float effective_penalty_factor = penalty_factor * loss_mult;
            float penalty = min(0.90f, (current_loss - ema_loss) * effective_penalty_factor);

            // Loss is actively dropping: aggressive descent surge with no upper limit
            float gain = max(0.0f, (ema_loss - current_loss) * 1.5f);
            config.lr *= (1.0f + gain) * stochastic_jitter * penalty * 2.0f;
            config.beta1 = 0.90f * penalty * 2.0f;
        }
        else if (delta > 0.15f)
        {
            // Clear macro loss spike: stabilize using derivative-tuned penalty_factor and loss scaling multiplier
            float effective_penalty_factor = penalty_factor * loss_mult;
            float penalty = min(0.90f, (current_loss - ema_loss) * effective_penalty_factor);
            config.lr *= (0.93f / (1.0f + penalty));
            config.beta1 = 0.95f / (1.0f + penalty);
            last_penalty_applied = penalty;
        }
        else
        {
            // Mild plateau: apply exploration jitter
            config.lr *= stochastic_jitter;
        }

        // Safeguard LR: minimum floor to prevent freeze, maximum ceiling to prevent numerical explosion
        config.lr = max(0.000001f, min(20.0f, config.lr));

        // Relax weight decay when approaching global minimum loss to preserve fine details
        if (min_loss > 0.0f)
        {
            float closeness = max(0.2f, min(1.0f, current_loss / (min_loss + 1e-4f)));
            config.weight_decay = 0.01f * closeness;
        }
    }

    // Modulates layer-wise learning rates based on whether previous weight shifts reduced or raised loss
    void AdamW::update_attribution_feedback(float loss_delta)
    {
        float loss_mult = ring0::Loss::compute_loss_scale_multiplier(last_loss_observed > 0.0f ? last_loss_observed : 5.0f);
        float effective_penalty_factor = penalty_factor * loss_mult;

        for (size_t i = 0; i < layer_scales.size(); ++i)
        {
            // Natural gentle mean-reversion towards 1.0 (prevents noise throttling)
            layer_scales[i] = 0.99f * layer_scales[i] + 0.01f * 1.0f;

            if (loss_delta < -0.05f)
            {
                // Uncapped layer attribution boost (max = inf)
                layer_scales[i] *= 1.05f;
                layer_attributions[i] += 1.0f;
            }
            else if (loss_delta > 0.15f)
            {
                // Damped by dynamically tuned attribution penalty
                float layer_penalty = min(0.90f, loss_delta * effective_penalty_factor * 5.0f);
                layer_scales[i] = max(0.2f, layer_scales[i] * (1.0f - layer_penalty));
                layer_attributions[i] -= 1.0f;
            }
        }
    }

    // Records the empirical derivative of penalization impact on loss d(Loss)/d(Penalty) and dynamically shifts penalty_factor
    void AdamW::update_penalization_derivative(float current_loss, float loss_delta)
    {
        if (last_loss_observed > 0.0f)
        {
            float delta_loss = current_loss - last_loss_observed;
            float delta_penalty = penalty_factor - last_penalty_applied;

            if (fabsf(delta_penalty) > 1e-4f)
            {
                // Exact empirical derivative: d(Loss) / d(Penalty)
                d_loss_d_penalty = delta_loss / delta_penalty / (1.0f + fabsf(delta_loss)) * .25f; // scale to [-3, 3] range
            }
            else
            {
                // If delta penalty is small, estimate sensitivity direction from loss delta
                d_loss_d_penalty = delta_loss * (penalty_factor > 0.5f ? 1.0f : -1.0f);
            }

            // Exponential moving average filter to reduce stochastic batch noise
            const float alpha = 0.85f;
            ema_d_loss_d_penalty = (1.0f - alpha) * ema_d_loss_d_penalty + alpha * d_loss_d_penalty;

            // Adaptive Sensitivity Feedback Rule: P_{t+1} = P_t - eta * (d Loss / d Penalty)
            // If d(Loss)/d(Penalty) > 0: Increasing penalty caused loss to rise -> REDUCE penalty!
            // If d(Loss)/d(Penalty) < 0: Increasing penalty caused loss to fall -> BOOST penalty!
            float eta_penalty = 0.15f;
            float penalty_step = -eta_penalty * tanhf(ema_d_loss_d_penalty * 0.5f);

            // Immediate direct responsiveness to loss spikes/drops
            if (loss_delta > 0.10f)
            {
                penalty_step += 0.05f; // Loss spiking: increase regularization penalty
            }
            else if (loss_delta < -0.05f)
            {
                penalty_step -= 0.03f; // Loss descending: relax penalty
            }

            // Dynamically update active penalty factor
            penalty_factor = max(0.00005f, min(30.0f, penalty_factor + penalty_step * max(1.0f / loss_delta, 0.08f)));
        }

        last_penalty_applied = penalty_factor;
        last_loss_observed = current_loss;
    }

    // Applies AdamW update with Nesterov accelerated momentum, Fisher metric preconditioning, and layer-wise attribution
    void AdamW::update_param(size_t index, ring0::Matrix &param, const ring0::Matrix &grad)
    {
        if (index >= m_list.size())
            return;

        auto &m = m_list[index];
        auto &v = v_list[index];
        if (index >= fisher_diag.size())
        {
            fisher_diag.resize(m_list.size());
        }
        auto &f = fisher_diag[index];

        // Resize moments if parameter shape changed
        if (m.rows != param.rows || m.cols != param.cols)
        {
            m = ring0::Matrix::zeros(param.rows, param.cols);
            v = ring0::Matrix::zeros(param.rows, param.cols);
            f = ring0::Matrix::zeros(param.rows, param.cols);
            if (index < last_shifts.size())
            {
                last_shifts[index] = ring0::Matrix::zeros(param.rows, param.cols);
            }
        }

        float t = static_cast<float>(timestep + 1);
        // Bias correction factors
        float beta1_corr = 1.0f - pow(config.beta1, t);
        float beta2_corr = 1.0f - pow(config.beta2, t);

        // Check if decoupled weight decay is enabled for this parameter
        bool apply_decay = (index < decay_mask.size()) ? decay_mask[index] : true;
        float wd = apply_decay ? config.weight_decay : 0.0f;

        // Layer-wise credit attribution scale and curvature modulation
        float layer_scale = (index < layer_scales.size()) ? layer_scales[index] : 1.0f;
        float effective_lr = config.lr * layer_scale * max(0.2f, min(3.0f, config.curvature_scale));

        // Dimension-aware step damping: the higher a parameter's dimension, the
        // less each element is moved. This mirrors fan-in init scaling (1/sqrt(dim))
        // and, crucially, tames the largest tensor in the model -- the tied
        // vocab x embed token-weight matrix (e.g. 10000 x 128), which is used as
        // BOTH the input embedding and the output head, so an un-damped step on it
        // perturbs the model twice and was the source of the loss blow-ups.
        // Reference dim = 128 -> factor 1.0; 256 -> 0.71; 10000 -> 0.11.
        const float ref_dim = 128.0f;
        float big_dim = static_cast<float>(std::max(param.rows, param.cols));
        float dim_damp = std::sqrt(ref_dim / std::max(ref_dim, big_dim));
        effective_lr *= dim_damp;

        const float beta1 = config.beta1;
        const float beta2 = config.beta2;
        const float eps = config.eps;
        const float effective_wd = wd * max(0.05f, penalty_factor);
        const float curvature_scale = config.curvature_scale;
        const bool use_multi_formula = config.enable_multi_formula;

        // Precompute tensor L2 norms for salience calculation
        float norm_g_sq = 0.0f;
        float norm_w_sq = 0.0f;
        if (use_multi_formula)
        {
            for (size_t i = 0; i < param.data.size(); ++i)
            {
                norm_g_sq += grad.data[i] * grad.data[i];
                norm_w_sq += param.data[i] * param.data[i];
            }
        }
        const float norm_g = std::sqrt(norm_g_sq);
        const float norm_w = std::sqrt(norm_w_sq);

        if (index == 0)
        {
            last_formula_stats = FormulaDistributionStats{};
        }

        size_t local_f1 = 0, local_f2 = 0, local_f3 = 0, local_f4 = 0;
        size_t num_elems = param.data.size();

#pragma omp parallel for schedule(static) if (param.data.size() > 1024) reduction(+ : local_f1, local_f2, local_f3, local_f4)
        for (int i_idx = 0; i_idx < static_cast<int>(param.data.size()); ++i_idx)
        {
            size_t i = static_cast<size_t>(i_idx);
            float g = grad.data[i];  // this element's gradient
            float w = param.data[i]; // this element's current value

            // Defensive: a NaN/Inf gradient or weight would poison the moments forever.
            if (std::isnan(g) || std::isinf(g))
                g = 0.0f;
            if (std::isnan(w) || std::isinf(w))
                w = 0.0f;

            // --- Adam's two running averages (per weight) ---
            // 1. First moment m = EMA of the gradient  ->  "which direction, smoothed"
            float new_m = beta1 * m.data[i] + (1.0f - beta1) * g;
            m.data[i] = (std::isnan(new_m) || std::isinf(new_m)) ? 0.0f : new_m;

            // 2. Second moment v = EMA of gradient^2   ->  "how big/noisy this weight's
            //    gradient has been"; used to shrink steps for high-variance weights.
            float new_v = beta2 * v.data[i] + (1.0f - beta2) * g * g;
            v.data[i] = (std::isnan(new_v) || std::isinf(new_v) || new_v < 0.0f) ? 0.0f : new_v;

            // 3. Diagonal Fisher information F = EMA of gradient^2 (slower decay). Estimates
            //    the local curvature of the loss (natural-gradient preconditioner). Similar
            //    to v but with its own horizon so the two can be blended independently.
            float new_f = 0.95f * f.data[i] + 0.05f * (g * g);
            f.data[i] = (std::isnan(new_f) || std::isinf(new_f) || new_f < 0.0f) ? 0.0f : new_f;

            // 4. Bias correction: early in training the EMAs start at 0 and are biased low,
            //    so divide by (1 - beta^t) to de-bias them (standard Adam correction).
            float m_hat = (beta1_corr > 1e-7f) ? (m.data[i] / beta1_corr) : m.data[i];
            float v_hat = (beta2_corr > 1e-7f) ? (v.data[i] / beta2_corr) : v.data[i];
            float f_hat = (beta2_corr > 1e-7f) ? (f.data[i] / beta2_corr) : f.data[i];

            float delta_w = 0.0f;

            if (use_multi_formula)
            {
                // 4-FORMULA WEIGHT PHYSICS: not every weight deserves the same update rule.
                // Score this weight's "importance" in [0,1] (from its gradient salience and
                // Fisher curvature relative to the tensor's norms), then route it to one of
                // four specialized update formulas by importance band. High-importance
                // weights get careful natural-gradient steps; low-importance ones get cheap
                // decay/pruning. (See multi_formula_optimizer.cpp for each formula.)
                float importance = MultiFormulaKernel::compute_importance(w, g, f_hat, norm_g, norm_w, num_elems);

                WeightFormulaType formula;
                if (importance > 0.65f)
                {
                    formula = WeightFormulaType::FORMULA_1_GEODESIC_NATURAL_GRAD;
                    local_f1++;
                }
                else if (importance > 0.45f)
                {
                    formula = WeightFormulaType::FORMULA_2_CURVATURE_NESTEROV;
                    local_f2++;
                }
                else if (importance > 0.20f)
                {
                    formula = WeightFormulaType::FORMULA_3_VARIANCE_BOUNDED_ADAMW;
                    local_f3++;
                }
                else
                {
                    formula = WeightFormulaType::FORMULA_4_INERTIAL_SPARSE_DECAY;
                    local_f4++;
                }

                delta_w = MultiFormulaKernel::execute_update_formula(
                    formula, w, g, m_hat, v_hat, f_hat,
                    effective_lr, beta1, beta1_corr, eps, effective_wd, curvature_scale);
            }
            else
            {
                // Standard baseline AdamW path
                if (effective_wd > 0.0f)
                {
                    w -= effective_lr * effective_wd * w;
                }
                float g_corr = (beta1_corr > 1e-7f) ? (g / beta1_corr) : g;
                float step_m = config.use_nesterov ? (beta1 * m_hat + (1.0f - beta1) * g_corr) : m_hat;
                float preconditioner = std::max(1e-7f, std::sqrt(std::max(0.0f, v_hat)) + eps);
                if (config.use_natural_grad)
                {
                    float fisher_val = std::sqrt(std::max(0.0f, f_hat)) + eps;
                    preconditioner = 0.7f * preconditioner + 0.3f * fisher_val;
                }
                delta_w = (effective_lr / preconditioner) * step_m;
            }

            if (std::isnan(delta_w) || std::isinf(delta_w))
            {
                delta_w = 0.0f;
            }
            else
            {
                // Per-element step trust region, loss-adaptive (config.max_step,
                // set each step via trust_region_for_loss). The old fixed ±2.0
                // was 20-100x the ~0.02-0.1 init scale, so once real gradients
                // flowed a single update could detonate the loss (8.7 -> 47 in
                // one step). A tight cap at high loss prevents that; it loosens
                // as loss falls so fine-tuning stays fast.
                const float step_cap = std::max(0.01f, config.max_step);
                delta_w = std::clamp(delta_w, -step_cap, step_cap);
            }

            // 5. Parameter update & shift recording
            float updated_w = w - delta_w;
            if (std::isnan(updated_w) || std::isinf(updated_w))
            {
                updated_w = 0.0f;
                m.data[i] = 0.0f;
                v.data[i] = 0.0f;
            }
            else
            {
                updated_w = std::clamp(updated_w, -30.0f, 30.0f);
            }
            param.data[i] = updated_w;

            if (index < last_shifts.size() && i < last_shifts[index].data.size())
            {
                last_shifts[index].data[i] = -delta_w;
            }
        }

        if (use_multi_formula)
        {
            last_formula_stats.count_f1 += local_f1;
            last_formula_stats.count_f2 += local_f2;
            last_formula_stats.count_f3 += local_f3;
            last_formula_stats.count_f4 += local_f4;
            last_formula_stats.total_params += param.data.size();
        }
    }

} // namespace ring1
