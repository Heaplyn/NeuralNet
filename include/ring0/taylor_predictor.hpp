#pragma once

/**
 * @file taylor_predictor.hpp
 * @brief Taylor / Newton-Gregory loss-trajectory predictor in Ring 0.
 *
 * Operates purely on the scalar step-to-step loss history (NOT on weight
 * tensors), so its full cost is O(n * K) scalar FLOPs per training step
 * (n = derivative order <= 5, K = forecast horizon <= 5). This is a few
 * dozen operations per step and allocates nothing in the hot path, so it
 * adds effectively zero overhead to the O(params) optimizer update.
 *
 * Core idea (discrete Taylor series):
 *   1. Build backward finite differences up to order n at the newest point:
 *        d^1 L_t = L_t - L_{t-1}
 *        d^2 L_t = d^1 L_t - d^1 L_{t-1}
 *        ... (this is exactly the Delta ladder)
 *   2. Extrapolate future losses via Newton-Gregory backward differences,
 *      which is the discrete Taylor expansion with step h = 1:
 *        L_{t+k} ~= sum_{j=0..n} C(k+j-1, j) * lambda_j * d^j L_t
 *      where lambda_j is a per-order recursive TRUST/damping factor that
 *      keeps high-order extrapolation from exploding (Runge phenomenon),
 *      in the same anti-explosion spirit as LossDerivativePyramid's tanh
 *      soft-clamp.
 *   3. Turn the predicted trajectory into a discounted reward used by the
 *      meta-optimizer, the penalty controller, and the LR/curvature scalers.
 *
 * Recursive parent/child coupling:
 *   Each derivative order is a node in a chain. Order j is the CHILD of
 *   order j-1 and the PARENT of order j+1. After each real step the node
 *   measures how well it predicted the next difference; that error flows
 *   UP the chain (a child corrects its parent's trust), while the aggregate
 *   trajectory confidence flows DOWN (a parent scales each child's
 *   contribution). The result is a small self-tuning meta-system where the
 *   orders interact rather than acting independently.
 */

#include <array>
#include <vector>
#include <cstddef>

using namespace std;

namespace ring0 {

/// Maximum supported derivative order (n) and forecast horizon (K).
constexpr size_t TAYLOR_MAX_ORDER = 5;
constexpr size_t TAYLOR_MAX_HORIZON = 5;

/**
 * @struct TaylorTrajectory
 * @brief Output bundle of a single trajectory prediction.
 */
struct TaylorTrajectory {
    size_t order = 0;                          ///< Effective derivative order actually used (<= n)
    size_t horizon = 0;                        ///< Number of future steps forecast (<= K)

    array<float, TAYLOR_MAX_ORDER + 1> diffs{}; ///< Backward differences: diffs[0]=L_t, diffs[j]=d^j L_t
    array<float, TAYLOR_MAX_HORIZON> predicted{}; ///< Predicted L_{t+1..t+K}
    array<float, TAYLOR_MAX_HORIZON> pred_delta{}; ///< Predicted step deltas L_{t+k}-L_{t+k-1}

    float reward = 0.0f;          ///< Discounted trajectory reward (positive = predicted improvement)
    float confidence = 0.0f;      ///< Aggregate forecast confidence in [0,1]

    // --- Actionable modulation signals derived from the forecast ---
    float penalty_foresight = 0.0f;  ///< Anticipatory penalty push: >0 raise penalty (predicted spike), <0 relax
    float lr_foresight_scale = 1.0f; ///< Predictive LR multiplier in [0.5, 1.6]
    float curvature_foresight = 1.0f;///< Predictive curvature scale in [0.5, 1.5]

    bool valid = false;           ///< False until enough history exists to extrapolate
};

/**
 * @class TaylorTrajectoryPredictor
 * @brief Recursive nth-order loss forecaster with self-tuning per-order trust.
 */
class TaylorTrajectoryPredictor {
public:
    /**
     * @brief Configuration for the predictor.
     */
    struct Config {
        size_t order = 4;               ///< Highest finite-difference order n (1..TAYLOR_MAX_ORDER)
        size_t horizon = 5;             ///< Forecast horizon K (1..TAYLOR_MAX_HORIZON)
        float base_damping = 0.6f;      ///< Base geometric per-order trust lambda_j = base_damping^j
        float reward_discount = 0.8f;   ///< Geometric discount over the horizon
        float immediate_weight = 1.0f;  ///< alpha: weight on the observed 1st-order term
        float clip_band = 5.0f;         ///< tanh soft-clamp band around L_t for predictions
        float trust_lr = 0.1f;          ///< EMA rate for self-tuning per-order trust
    };

    TaylorTrajectoryPredictor();
    explicit TaylorTrajectoryPredictor(Config cfg);

    /**
     * @brief Feeds the newest scalar loss and produces a fresh trajectory forecast.
     *
     * Also performs the recursive parent/child trust update using the PREVIOUS
     * forecast compared against the loss that actually arrived this step.
     *
     * @param loss_history Chronological loss values; the last element is L_t.
     * @return Trajectory forecast (valid==false if history is too short).
     */
    TaylorTrajectory observe(const vector<float>& loss_history);

    /// Most recent trajectory forecast.
    const TaylorTrajectory& last() const { return last_traj_; }

    /// Per-order trust factors (parent/child coupled), exposed for telemetry.
    const array<float, TAYLOR_MAX_ORDER + 1>& order_trust() const { return trust_; }

    /// Resets all internal recursive state.
    void reset();

    Config config;

private:
    array<float, TAYLOR_MAX_ORDER + 1> trust_{};       ///< Self-tuned per-order trust lambda_j
    array<float, TAYLOR_MAX_ORDER + 1> ema_order_err_{};///< EMA of each order's prediction error
    TaylorTrajectory last_traj_;
    float prev_predicted_next_ = 0.0f;  ///< L_{t} that the previous step forecast for "next"
    bool  have_prev_pred_ = false;

    /// Recomputes trust_ from ema_order_err_ with parent<-child, parent->child coupling.
    void recouple_trust();
};

} // namespace ring0
