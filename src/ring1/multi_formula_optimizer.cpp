#include "ring1/multi_formula_optimizer.hpp"
#include <cmath>
#include <algorithm>

namespace ring1
{

    float MultiFormulaKernel::compute_importance(
        float w,
        float g,
        float fisher,
        float norm_g,
        float norm_w,
        size_t num_elements)
    {
        float n_f = static_cast<float>(std::max<size_t>(1, num_elements));

        // 1. Normalized Taylor salience relative to mean element magnitude in tensor
        float mean_energy = (norm_g * norm_w) / n_f + 1e-8f;
        float relative_salience = std::abs(g * w) / mean_energy;

        // 2. Normalized Fisher curvature relative to mean gradient variance
        float mean_fisher = (norm_g * norm_g) / n_f + 1e-8f;
        float relative_fisher = std::sqrt(std::max(0.0f, fisher) / mean_fisher);

        // 3. Dynamic non-linear blend
        float raw_score = 0.6f * relative_salience + 0.4f * relative_fisher;
        if (std::isnan(raw_score) || std::isinf(raw_score) || raw_score < 0.0f)
        {
            return 0.5f;
        }

        // 4. Smooth sigmoidal importance mapping into [0.0, 1.0]
        float importance = raw_score / (1.0f + raw_score);
        if (std::isnan(importance) || std::isinf(importance))
        {
            return 0.5f;
        }
        return std::clamp(importance, 0.0f, 1.0f);
    }

    float MultiFormulaKernel::execute_update_formula(
        WeightFormulaType formula,
        float w,
        float g,
        float m_hat,
        float v_hat,
        float f_hat,
        float effective_lr,
        float beta1,
        float beta1_corr,
        float eps,
        float effective_wd,
        float curvature_scale)
    {
        if (std::isnan(w) || std::isinf(w))
            w = 0.0f;
        if (std::isnan(g) || std::isinf(g))
            g = 0.0f;
        if (std::isnan(m_hat) || std::isinf(m_hat))
            m_hat = 0.0f;
        if (std::isnan(v_hat) || std::isinf(v_hat) || v_hat < 0.0f)
            v_hat = 0.0f;
        if (std::isnan(f_hat) || std::isinf(f_hat) || f_hat < 0.0f)
            f_hat = 0.0f;

        float delta_w = 0.0f;

        switch (formula)
        {
        // =========================================================================
        // FORMULA 1: Riemannian Geodesic Natural Gradient (Ultra-High Importance)
        // Highest precision: Fisher metric preconditioning + Nesterov momentum
        // =========================================================================
        case WeightFormulaType::FORMULA_1_GEODESIC_NATURAL_GRAD:
        {
            float g_corr = (beta1_corr > 1e-7f) ? (g / beta1_corr) : g;
            float nesterov_m = beta1 * m_hat + (1.0f - beta1) * g_corr;
            float fisher_val = std::sqrt(std::max(0.0f, f_hat)) + eps;
            float v_val = std::sqrt(std::max(0.0f, v_hat)) + eps;
            // Riemannian metric blend
            float natural_precond = std::max(1e-7f, 0.5f * v_val + 0.5f * fisher_val);
            delta_w = (effective_lr / natural_precond) * nesterov_m;
            if (effective_wd > 0.0f)
            {
                delta_w += effective_lr * (effective_wd * 0.5f) * w; // Soft decay on high importance
            }
            break;
        }

        // =========================================================================
        // FORMULA 2: Curvature-Scaled Nesterov Acceleration (High Importance)
        // Rayleigh quotient curvature modulation + Phase space lookahead
        // =========================================================================
        case WeightFormulaType::FORMULA_2_CURVATURE_NESTEROV:
        {
            float g_corr = (beta1_corr > 1e-7f) ? (g / beta1_corr) : g;
            float nesterov_m = beta1 * m_hat + (1.0f - beta1) * g_corr;
            float precond = std::max(1e-7f, (std::sqrt(std::max(0.0f, v_hat)) + eps) * std::max(0.2f, curvature_scale));
            delta_w = (effective_lr / precond) * nesterov_m;
            if (effective_wd > 0.0f)
            {
                delta_w += effective_lr * effective_wd * w;
            }
            break;
        }

        // =========================================================================
        // FORMULA 3: Variance-Bounded Decoupled AdamW (Medium Importance)
        // =========================================================================
        // FORMULA 3: Variance-Bounded Decoupled AdamW (Medium Importance)
        // Standard first-order AdamW with second moment variance clipping
        // =========================================================================
        case WeightFormulaType::FORMULA_3_VARIANCE_BOUNDED_ADAMW:
        {
            float bounded_v = std::max(1e-8f, std::min(100.0f, v_hat));
            float precond = std::max(1e-7f, std::sqrt(bounded_v) + eps);
            delta_w = (effective_lr / precond) * m_hat;
            if (effective_wd > 0.0f)
            {
                delta_w += effective_lr * effective_wd * w;
            }
            break;
        }

        // =========================================================================
        // FORMULA 4: Inertial Sparse Decay & Representation Compression (Low Importance)
        // Prunes noisy activations and accelerates sparse representation
        // =========================================================================
        case WeightFormulaType::FORMULA_4_INERTIAL_SPARSE_DECAY:
        {
            float precond = std::max(1e-7f, std::sqrt(std::max(0.0f, v_hat)) + eps);
            float standard_step = (effective_lr / precond) * m_hat;
            // Enhanced decoupled decay to compress near-zero noisy weights
            float decay_step = effective_lr * (effective_wd * 2.0f) * w;
            delta_w = 0.5f * standard_step + decay_step;
            break;
        }
        }

        if (std::isnan(delta_w) || std::isinf(delta_w))
        {
            delta_w = 0.0f;
        }
        else
        {
            delta_w = std::clamp(delta_w, -0.05f, 0.05f);
        }

        return delta_w;
    }

} // namespace ring1
