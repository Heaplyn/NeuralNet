#include "ring0/taylor_predictor.hpp"
#include <algorithm>
#include <cmath>

using namespace std;

namespace ring0
{

    // Small integer binomial coefficient C(n, k) for the tiny ranges used here
    // (n <= order+horizon <= 10). Computed in double then returned as float.
    static float binom(size_t n, size_t k)
    {
        if (k > n)
            return 0.0f;
        if (k == 0 || k == n)
            return 1.0f;
        k = min(k, n - k);
        double result = 1.0;
        for (size_t i = 0; i < k; ++i)
        {
            result = result * static_cast<double>(n - i) / static_cast<double>(i + 1);
        }
        return static_cast<float>(result);
    }

    // C^1-continuous soft-clamp: linear near zero, tanh-saturating past +/- band.
    // Mirrors the anti-explosion philosophy of LossDerivativePyramid::build.
    static float soft_clamp(float x, float band)
    {
        float b = max(0.1f, band);
        if (fabsf(x) <= b)
            return x;
        float sgn = (x >= 0.0f) ? 1.0f : -1.0f;
        float excess = fabsf(x) - b;
        return sgn * (b * .5f + b * tanhf(excess / b));
    }

    TaylorTrajectoryPredictor::TaylorTrajectoryPredictor() : config()
    {
        reset();
    }

    TaylorTrajectoryPredictor::TaylorTrajectoryPredictor(Config cfg) : config(cfg)
    {
        reset();
    }

    void TaylorTrajectoryPredictor::reset()
    {
        config.order = clamp<size_t>(config.order, 1, TAYLOR_MAX_ORDER);
        config.horizon = clamp<size_t>(config.horizon, 1, TAYLOR_MAX_HORIZON);
        for (size_t j = 0; j <= TAYLOR_MAX_ORDER; ++j)
        {
            // Initial trust: geometric decay by order, full trust on order 0/1.
            trust_[j] = powf(config.base_damping, static_cast<float>(j));
            ema_order_err_[j] = 0.0f;
        }
        last_traj_ = TaylorTrajectory{};
        prev_predicted_next_ = 0.0f;
        have_prev_pred_ = false;
    }

    // Rebuilds per-order trust from accumulated prediction error, with recursive
    // parent<->child coupling so the orders interact instead of acting in isolation.
    void TaylorTrajectoryPredictor::recouple_trust()
    {
        const size_t n = config.order;

        // 1. Base trust from each order's own reliability (child self-assessment):
        //    lower EMA error => higher trust. Geometric prior keeps high orders
        //    conservative until they earn trust.
        array<float, TAYLOR_MAX_ORDER + 1> raw{};
        for (size_t j = 0; j <= n; ++j)
        {
            float prior = powf(config.base_damping, static_cast<float>(j));
            float reliability = 1.0f / (1.0f + ema_order_err_[j]); // in (0,1]
            raw[j] = prior * reliability;
        }

        // 2. Parent -> child gating: a child (order j+1) may only be trusted as far
        //    as its parent (order j) is. If the parent is unreliable, the noisier
        //    child is throttled proportionally. This is the top-down signal.
        for (size_t j = 1; j <= n; ++j)
        {
            float parent = raw[j - 1];
            raw[j] *= (0.5f + 0.5f * parent); // parent in [0,1] -> gate in [0.5,1.0]
        }

        // 3. Child -> parent correction: a highly reliable child implies the local
        //    curvature is real, so its parent's trust is nudged up; an erratic child
        //    warns the parent to be cautious. This is the bottom-up signal.
        for (size_t j = 0; j < n; ++j)
        {
            float child_reliab = 1.3f / (1.0f + ema_order_err_[j + 1] / (1.0f + ema_order_err_[j])); // in (0,1]
            float adjust = 0.3f + 1.2f * child_reliab;                                               // in [0.9, 1.1]
            raw[j] *= adjust;
        }

        // 4. Commit with clamping to a sane band.
        for (size_t j = 0; j <= n; ++j)
        {
            trust_[j] = clamp(raw[j], 0.0f, 1.25f);
        }
        for (size_t j = n + 1; j <= TAYLOR_MAX_ORDER; ++j)
        {
            trust_[j] = 0.0f;
        }
    }

    TaylorTrajectory TaylorTrajectoryPredictor::observe(const vector<float> &loss_history)
    {
        TaylorTrajectory traj;
        const size_t n_cfg = config.order;
        const size_t K = config.horizon;

        // Need at least (order + 1) points to form the full difference ladder.
        // Gracefully reduce the effective order if history is short.
        size_t hist = loss_history.size();
        if (hist < 2)
        {
            last_traj_ = traj; // invalid
            return traj;
        }
        size_t n = min(n_cfg, hist - 1);
        traj.order = n;
        traj.horizon = K;

        float L_t = loss_history[hist - 1];

        // --- Recursive parent/child trust update from the PREVIOUS forecast ---
        // Compare what we predicted for "this" step against what actually arrived,
        // attributing the error across orders by their marginal contribution.
        if (have_prev_pred_)
        {
            float actual_delta = L_t - loss_history[hist - 2];
            float pred_delta = prev_predicted_next_ - loss_history[hist - 2];
            float traj_err = fabsf(pred_delta - actual_delta);

            // Distribute the trajectory error onto each order's EMA reliability.
            // Higher orders are held more responsible for the curvature portion.
            for (size_t j = 0; j <= config.order; ++j)
            {
                float order_weight = (j == 0) ? 0.2f : powf(0.7f, static_cast<float>(j - 1));
                float e = traj_err * order_weight;
                ema_order_err_[j] = (1.0f - config.trust_lr) * ema_order_err_[j] + config.trust_lr * e;
            }
            recouple_trust();
        }
        else
        {
            recouple_trust();
        }

        // --- 1. Backward finite-difference ladder d^0..d^n at the newest point ---
        // Build a working buffer of the last (n+1) losses, then difference in place.
        array<float, TAYLOR_MAX_ORDER + 1> window{};
        for (size_t j = 0; j <= n; ++j)
        {
            window[j] = loss_history[hist - 1 - (n - j)]; // oldest..newest
        }
        // diffs[0] = L_t; diffs[j] = d^j L_t (backward difference at newest point)
        traj.diffs[0] = L_t;
        // Compute successive forward differences of the window; the LAST entry of
        // each pass equals the backward difference of that order at the newest point.
        array<float, TAYLOR_MAX_ORDER + 1> work = window;
        size_t len = n + 1;
        for (size_t order = 1; order <= n; ++order)
        {
            for (size_t i = 0; i + 1 < len; ++i)
            {
                work[i] = work[i + 1] - work[i];
            }
            --len;
            traj.diffs[order] = work[len - 1]; // newest backward difference of this order
        }

        // --- 2. Newton-Gregory backward extrapolation with per-order trust ---
        // L_{t+k} = sum_{j=0..n} C(k+j-1, j) * trust_j * d^j L_t   (soft-clamped)
        float prev_level = L_t;
        for (size_t k = 1; k <= K; ++k)
        {
            float acc = L_t; // j = 0 term
            for (size_t j = 1; j <= n; ++j)
            {
                float coeff = binom(k + j - 1, j) * trust_[j];
                acc += coeff * traj.diffs[j];
            }
            // Keep the forecast physically sane: soft-clamp deviation from current
            // loss, and forbid predicting negative loss.
            float dev = soft_clamp(acc - L_t, config.clip_band);
            float level = max(0.0f, L_t + dev);
            traj.predicted[k - 1] = level;
            traj.pred_delta[k - 1] = level - prev_level;
            prev_level = level;
        }

        // --- 3. Discounted trajectory reward ---
        // R = -alpha * d^1L_t  -  sum_k disc^k * predicted_delta_k
        // (a predicted DROP is a negative delta -> positive reward)
        float reward = -config.immediate_weight * traj.diffs[1];
        float disc = 1.0f;
        float disc_sum = 0.0f;
        for (size_t k = 1; k <= K; ++k)
        {
            disc *= config.reward_discount;
            reward += -disc * traj.pred_delta[k - 1];
            disc_sum += disc;
        }
        traj.reward = reward;

        // --- Confidence: average trust of the active orders, damped by recent error ---
        float trust_sum = 0.0f;
        for (size_t j = 1; j <= n; ++j)
            trust_sum += trust_[j];
        float mean_trust = (n > 0) ? (trust_sum / static_cast<float>(n)) : 0.0f;
        float recent_err = ema_order_err_[min<size_t>(1, n)];
        traj.confidence = clamp(mean_trust / (1.0f + recent_err), 0.0f, 1.0f);

        // --- Actionable modulation signals ---
        // Net predicted change over the horizon (positive => loss rising => bad).
        float net_pred = traj.predicted[K - 1] - L_t;

        // Anticipatory penalty: if the trajectory predicts a rise, push penalty up
        // BEFORE it happens; if it predicts a strong drop, relax. Scaled by trust.
        traj.penalty_foresight = traj.confidence * tanhf(net_pred * 0.75f);

        // Predictive LR: bolder step when confident descent is ahead, damped when a
        // spike is forecast (a forward-looking Armijo).
        float lr_sig = -tanhf(net_pred * 0.75f) * traj.confidence; // +ve if descent ahead
        traj.lr_foresight_scale = clamp(1.0f + 0.6f * lr_sig, 0.5f, 1.6f);

        // Predictive curvature: rising/oscillating forecast => shrink steps.
        float osc = 0.0f;
        for (size_t k = 1; k < K; ++k)
        {
            // sign flips in predicted deltas indicate oscillation / turbulence
            if (traj.pred_delta[k] * traj.pred_delta[k - 1] < 0.0f)
                osc += 1.0f;
        }
        float osc_frac = (K > 1) ? (osc / static_cast<float>(K - 1)) : 0.0f;
        traj.curvature_foresight = clamp(1.0f - 0.5f * osc_frac - 0.3f * max(0.0f, tanhf(net_pred)), 0.5f, 1.5f);

        traj.valid = true;

        // Remember the immediate one-step forecast for next step's trust update.
        prev_predicted_next_ = traj.predicted[0];
        have_prev_pred_ = true;

        last_traj_ = traj;
        return traj;
    }

} // namespace ring0
