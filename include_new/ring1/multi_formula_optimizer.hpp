#pragma once

/**
 * @file multi_formula_optimizer.hpp
 * @brief Dynamic Multi-Formula Weight Physics Optimizer in Ring 1.
 *        Dispatches 1 to 4 specialized mathematical formulas per weight element
 *        based on first-order Taylor Salience, Fisher Information, and Relevancy.
 */

#include "ring0/tensor.hpp"
#include <vector>
#include <string>
#include <array>

namespace ring1 {

/**
 * @enum WeightFormulaType
 * @brief The 4 specialized mathematical update equations.
 */
enum class WeightFormulaType {
    FORMULA_1_GEODESIC_NATURAL_GRAD = 0, ///< Riemannian Fisher Natural Gradient (Ultra-High Importance)
    FORMULA_2_CURVATURE_NESTEROV = 1,    ///< Rayleigh Curvature Nesterov Lookahead (High Importance)
    FORMULA_3_VARIANCE_BOUNDED_ADAMW = 2,///< Variance-Bounded Decoupled AdamW (Medium Importance)
    FORMULA_4_INERTIAL_SPARSE_DECAY = 3  ///< Inertial Soft Decay & Pruning (Low Importance)
};

/**
 * @struct FormulaDistributionStats
 * @brief Percentage and count breakdown of parameters updated under each formula.
 */
struct FormulaDistributionStats {
    size_t count_f1 = 0; ///< Count under Formula 1
    size_t count_f2 = 0; ///< Count under Formula 2
    size_t count_f3 = 0; ///< Count under Formula 3
    size_t count_f4 = 0; ///< Count under Formula 4
    size_t total_params = 0;

    float pct_f1() const { return total_params > 0 ? (100.0f * count_f1 / total_params) : 0.0f; }
    float pct_f2() const { return total_params > 0 ? (100.0f * count_f2 / total_params) : 0.0f; }
    float pct_f3() const { return total_params > 0 ? (100.0f * count_f3 / total_params) : 0.0f; }
    float pct_f4() const { return total_params > 0 ? (100.0f * count_f4 / total_params) : 0.0f; }
};

/**
 * @class MultiFormulaKernel
 * @brief Dynamic Multi-Formula Weight Physics Engine.
 */
class MultiFormulaKernel {
public:
    /**
     * @brief Computes the dynamic importance factor I(w_i) for a parameter element.
     * @param w Weight parameter value
     * @param g Incoming gradient value
     * @param fisher Empirical diagonal Fisher information value
     * @param norm_g L2 norm of the full gradient tensor
     * @param norm_w L2 norm of the full parameter tensor
     * @param num_elements Total parameter count in tensor (N) for relative scale normalization
     * @return Normalized importance score in [0.0, 1.0]
     */
    static float compute_importance(float w, float g, float fisher, float norm_g, float norm_w, size_t num_elements);

    /**
     * @brief Dispatches the optimal update formula for a weight element.
     */
    static float execute_update_formula(
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
        float curvature_scale
    );
};

} // namespace ring1
