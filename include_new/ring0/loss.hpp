#pragma once

/**
 * @file loss.hpp
 * @brief Loss functions, multi-order hierarchical loss derivative pyramid, and loss history tracking for Ring 0.
 */

#include "ring0/tensor.hpp"
#include <vector>

using namespace std;

namespace ring0 {

/// Enum of supported loss functions
enum class LossType {
    MSE,
    CrossEntropy
};

/**
 * @struct DerivativeLayerStats
 * @brief Statistical metrics (min, max, mean, variance) for a single derivative layer in the hierarchy.
 */
struct DerivativeLayerStats {
    float min_val = 0.0f;
    float max_val = 0.0f;
    float mean_val = 0.0f;
    float variance = 0.0f;
    size_t size = 0;
};

/**
 * @class LossDerivativePyramid
 * @brief Gathers token losses and computes hierarchical discrete finite-difference derivatives across multiple iterations.
 * Uses tanh soft-saturation and bounded min-max tracking to prevent numerical explosions.
 */
class LossDerivativePyramid {
public:
    vector<vector<float>> layers;       ///< Hierarchical derivative layers: layers[0]=losses, layers[1]=1st deriv, ...
    vector<DerivativeLayerStats> stats; ///< Min, max, mean, variance per derivative layer

    /**
     * @brief Builds the hierarchical multi-order derivative pyramid across sequential iterations.
     * @param raw_losses Vector of individual token loss contributions.
     * @param max_iterations Number of derivative layers to build (default: 3).
     * @param clip_threshold Anti-explosion soft-saturation asymptotic ceiling (default: 5.0f).
     * @param magnitude_threshold Linear range threshold below which values pass through unattenuated (default: 2.0f).
     */
    void build(const vector<float>& raw_losses, size_t max_iterations = 3, float clip_threshold = 5.0f, float magnitude_threshold = 2.0f);

    /**
     * @brief Computes a curvature-guided gradient modulation multiplier in [0.65, 1.35].
     */
    float compute_curvature_scale() const;
};

/**
 * @class Loss
 * @brief Computes scalar loss values, parameter gradients, and tracks loss progression history.
 */
class Loss {
public:
    /// Global historical record of calculated losses across training iterations
    static vector<float> loss_history;

    /// Appends a new scalar loss value to the history vector
    static void record_loss(float loss_val);

    /// Retrieves the minimum recorded loss encountered so far
    static float get_min_loss();

    /// Retrieves the most recently recorded loss
    static float get_latest_loss();

    /**
     * @brief Computes the loss gap factor: (currentLoss - minLoss) * multiplier
     */
    static float get_loss_gap(float multiplier = 0.6f);

    /**
     * @brief Computes dynamic scaling multiplier based on loss magnitude:
     *        - Loss >= 5.0: 3.0x
     *        - Loss == 4.0: 2.2x
     *        - Loss == 3.0: 1.6x
     *        - Loss == 2.0: 1.0x
     *        - Loss <= 1.0: 0.3x
     *        Continuous piecewise linear interpolation is applied between milestones.
     */
    static float compute_loss_scale_multiplier(float loss);

    /// Clears the recorded loss history
    static void clear_history();

    /// Computes scalar loss based on selected LossType
    static float compute(LossType type, const Matrix& predictions, const Matrix& targets);

    /// Computes gradient matrix (dL/dPred) based on selected LossType
    static Matrix gradient(LossType type, const Matrix& predictions, const Matrix& targets);

    // --- Explicit Loss Functions ---

    /// Mean Squared Error: L = (1/N) * sum((y_hat - y)^2)
    static float mse(const Matrix& predictions, const Matrix& targets);
    /// MSE Gradient: dL/dy_hat = (2 / (N * D)) * (y_hat - y)
    static Matrix mse_gradient(const Matrix& predictions, const Matrix& targets);

    /// Cross-Entropy: L = -(1/N) * sum(y * log(y_hat + eps))
    static float cross_entropy(const Matrix& predictions, const Matrix& targets);
    /// Cross-Entropy Gradient with Softmax: dL/dZ = (y_hat - y) / N
    static Matrix cross_entropy_gradient(const Matrix& predictions, const Matrix& targets);
};

} // namespace ring0
