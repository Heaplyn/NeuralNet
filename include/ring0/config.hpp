#pragma once

/**
 * @file config.hpp
 * @brief Central runtime configuration, debug mode switches, and thought chain options for Ring 0.
 */

#include <iostream>
#include <string>
#include <vector>
#include <iomanip>

using namespace std;

namespace ring0 {

/**
 * @enum LogLevel
 * @brief Granularity of runtime diagnostics and telemetry.
 */
enum class LogLevel {
    QUIET = 0,
    STANDARD = 1,
    VERBOSE = 2,
    DEBUG = 3,
    TRACE = 4
};

/**
 * @struct RuntimeConfig
 * @brief Global and component configuration options for debugging, thought chains, and logging.
 */
struct RuntimeConfig {
    // --- Debug Mode Switches ---
    bool debug_mode = true;                     ///< Master debug switch (prints verbose diagnostics)
    LogLevel log_level = LogLevel::TRACE;    ///< Active logging detail level
    bool verbose_thought_chains = true;         ///< When in debug mode, print detailed thought chain steps
    bool verbose_loss_scaling = false;          ///< Print loss multiplier adjustments
    bool verbose_gradient_flow = false;         ///< Print gradient norm and layer shift metrics
    bool verbose_kv_cache = false;              ///< Print KV-cache allocation and hit telemetry

    // --- Thought Chain & Reasoning Options ---
    bool enable_thought_chain_looping = true;   ///< Allow recursive layers to loop through and refine thought chains
    size_t default_thought_loops = 3;           ///< Number of recursive reasoning loops per layer
    size_t max_chain_reflection_cycles = 5;     ///< Multi-pass reflection cycles through the entire thought tree
    float thought_convergence_tol = 1e-4f;      ///< Convergence threshold for early exit in thought loops
    float thought_damping = 0.85f;              ///< Residual momentum damping across thought chain loops
    bool record_thought_history = true;         ///< Maintain internal trace of thought vectors for auditing

    // --- Rapid Loss Descent Acceleration (5.0 -> 2.0) ---
    bool enable_loss_descent_acceleration = true; ///< Enables adaptive focal modulation & gravity surge
    float descent_boost_ceiling = 3.5f;           ///< Max descent multiplier in high-loss zone
    float focal_gamma_max = 2.0f;                 ///< Peak focal loss gamma at loss >= 5.0
    float plateau_breakout_loss = 2.0f;           ///< Transition loss floor below which fine-tuning begins
    bool fast_track_depth_unlock = true;          ///< Unlocks all transformer layers early to boost capacity

    // --- Token Relevancy & Interpolated Context Parsing ---
    bool enable_token_relevance_parsing = true;   ///< Algorithm based on token relevance with dynamic parsed window
    size_t min_relevance_window = 8;              ///< Min window of parsed tokens around low-relevance tokens
    size_t max_relevance_window = 64;             ///< Max window of parsed tokens around high-relevance tokens
    float relevance_interpolated_alpha = 0.75f;   ///< Exponent for non-linear relevancy interpolation

    // --- Singleton / Global Instance Access ---
    static RuntimeConfig& get_instance() {
        static RuntimeConfig instance;
        return instance;
    }

    /// Sets debug mode and adjusts related verbosity switches
    void set_debug(bool enabled) {
        debug_mode = enabled;
        if (enabled) {
            log_level = LogLevel::DEBUG;
            verbose_thought_chains = true;
            verbose_loss_scaling = true;
            verbose_gradient_flow = true;
            verbose_kv_cache = true;
        } else {
            log_level = LogLevel::TRACE;
            verbose_loss_scaling = false;
            verbose_gradient_flow = false;
            verbose_kv_cache = false;
        }
    }

    /// Helper to print debug messages with formatted prefix
    static void log_debug(const string& msg) {
        if (get_instance().debug_mode) {
            cout << "  🔍 [DEBUG] " << msg << "\n";
        }
    }
};

/// Convenience global getter for debug mode status
inline bool is_debug_mode() {
    return RuntimeConfig::get_instance().debug_mode;
}

inline RuntimeConfig& get_config() {
    return RuntimeConfig::get_instance();
}

} // namespace ring0
