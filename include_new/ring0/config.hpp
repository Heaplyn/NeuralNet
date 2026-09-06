#pragma once

/**
 * @file config.hpp
 * @brief Central runtime configuration, debug mode switches, and thought chain options for Ring 0.
 *
 * @version 1.1 - Stability hotfix applied:
 * - bad_batch_loss_threshold reduced from 7.5 -> 2.5 (catches loss spikes at 3.0)
 * - watchdog_rise_gap tightened from 0.40 -> 0.30 (more sensitive to sharp rises)
 * - global_gradient_clip_norm reduced from 0.50 -> 0.35 (prevents weight explosion)
 */

#include <iostream>
#include <string>
#include <vector>
#include <iomanip>
#include <fstream>
#include <chrono>
#include <filesystem>
#include <ctime>

using namespace std;

namespace ring0
{

    /**
     * @enum LogLevel
     * @brief Granularity of runtime diagnostics and telemetry.
     */
    enum class LogLevel
    {
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
    struct RuntimeConfig
    {
        // --- Debug Mode Switches ---
        bool debug_mode = true;               ///< Master debug switch (prints verbose diagnostics)
        LogLevel log_level = LogLevel::TRACE; ///< Active logging detail level
        bool verbose_thought_chains = true;   ///< When in debug mode, print detailed thought chain steps
        bool verbose_loss_scaling = false;    ///< Print loss multiplier adjustments
        bool verbose_gradient_flow = false;   ///< Print gradient norm and layer shift metrics
        bool verbose_kv_cache = false;        ///< Print KV-cache allocation and hit telemetry

        // --- Thought Chain & Reasoning Options ---
        bool enable_thought_chain_looping = true; ///< Allow recursive layers to loop through and refine thought chains
        size_t default_thought_loops = 3;         ///< Number of recursive reasoning loops per layer
        size_t max_chain_reflection_cycles = 5;   ///< Multi-pass reflection cycles through the entire thought tree
        float thought_convergence_tol = 1e-6f;    ///< Convergence threshold for early exit in thought loops
        float thought_damping = 0.88f;            ///< Residual momentum damping across thought chain loops
        bool record_thought_history = true;       ///< Maintain internal trace of thought vectors for auditing

        // --- Calculus of Constructions (CoC) & Dependent Types ---
        bool enable_coc_verification = true;            ///< Periodic CoC type checking and proof verification
        size_t coc_verification_interval = 5;           ///< Cadence in steps to run formal proof checking
        float coc_type_guidance_alpha = 0.25f;          ///< Weight of dependent type compatibility prior in attention
        bool enable_coc_universe_stratification = true; ///< Enforce Prop < Type_0 < Type_1 universe hierarchy
        size_t max_beta_reduction_steps = 1000;         ///< Bound on normalization steps to guarantee termination
        float coc_proof_consistency_threshold = 0.85f;  ///< Minimum proof score required for sound thought step

        // --- Optimizer & Directional Physics ---
        bool enable_damped_operation_reversal = true;
        float reversal_shrink_factor = 0.20f;
        float reversal_loss_sensitivity = 0.65f;

        float global_gradient_clip_norm = 0.35f; // 🔧 [FIXED] Tighter clamp (was 0.50) to prevent spikes

        float logit_soft_cap = 12.0f;

        float adamw_beta1 = 0.90f;
        float adamw_beta2 = 0.95f;
        float adamw_eps = 1e-8f;

        float base_weight_decay = 0.01f;

        float max_trust_region_step = 0.20f;
        float min_trust_region_step = 0.05f;

        // --- Stability Watchdog & Weight Rollback Recovery ---
        bool enable_weight_rollback_recovery = true;
        float bad_batch_loss_threshold = 2.5f; // 🔧 [FIXED] Catches spikes at 2.7-3.4 (was 7.5, way too high)
        float watchdog_rise_gap = 0.30f;       // 🔧 [FIXED] More sensitive to sharp rises (was 0.40)
        size_t watchdog_trigger_streak = 2;
        float watchdog_lr_penalty = 0.28f;
        size_t watchdog_min_recovery_steps = 30;
        float watchdog_recover_gap = 0.25f;
        size_t dataset_cooldown_steps = 15;
        size_t context_cooldown_steps = 20;

        // --- Multi-Formula Weight Physics Routing ---
        bool enable_multi_formula_routing = true;
        float f1_natural_gradient_threshold = 0.52f;
        float f2_nesterov_threshold = 0.30f;
        float f3_adamw_threshold = 0.16f;
        float f4_sparse_decay_rate = 0.035f;

        // --- Taylor Trajectory Predictor & Curvature ---
        bool enable_taylor_prediction = true;
        float taylor_step_damping = -0.38f;
        float taylor_min_confidence = 0.19f;
        bool enable_rayleigh_curvature = true;
        float curvature_scale_floor = 0.0004f;
        float curvature_scale_ceiling = 2.50f;

        // --- Dense Recognition Benchmarks ---
        size_t recognition_letter_epochs = 120;
        size_t recognition_image_epochs = 20000;
        size_t recognition_letter_batch_size = 728;
        size_t recognition_image_batch_size = 10000;
        size_t recognition_image_train_limit = 70000;
        size_t recognition_image_test_limit = 70000;
        float recognition_learning_rate = 0.01f;
        float recognition_weight_decay = 0.0005f;
        float recognition_max_grad_norm = 1.0f;
        float recognition_taylor_weight = 0.5f;
        bool recognition_enable_growth = true;
        bool recognition_enable_meta_loss = true;
        bool recognition_enable_multi_formula = true;
        bool recognition_enable_taylor = true;

        // --- Generation & Sampling Hyperparameters ---
        float default_temperature = 0.80f;
        size_t default_top_k = 50;
        float default_top_p = 0.95f;
        float default_min_p = 0.00005f;
        float default_repetition_penalty = 1.05f;
        size_t default_lookback_window = 128;

        // --- Hardware & Parallel Execution ---
        size_t num_threads = 0;
        bool enable_avx2_acceleration = true;
        bool enable_cuda_backend = true;

        // --- Rapid Loss Descent Acceleration (5.0 -> 2.0) ---
        bool enable_loss_descent_acceleration = true;
        float descent_boost_ceiling = 3.5f;
        float focal_gamma_max = 2.0f;
        float plateau_breakout_loss = 2.0f;
        float plateau_breakout_loss_threshold = 2.0f;
        bool fast_track_depth_unlock = false;

        // --- Token Relevancy & Interpolated Context Parsing ---
        bool enable_token_relevance_parsing = true;
        size_t min_relevance_window = 8;
        size_t max_relevance_window = 128;
        float relevance_interpolated_alpha = 0.65f;

        // --- Asynchronous Background Data Streaming ---
        bool enable_background_data_streaming = true;
        size_t background_stream_poll_interval = 5;
        size_t initial_bootstrap_data_files = 1;

        // --- Singleton / Global Instance Access ---
        static RuntimeConfig &get_instance()
        {
            static RuntimeConfig instance;
            return instance;
        }

        /// Sets debug mode and adjusts related verbosity switches
        void set_debug(bool enabled)
        {
            debug_mode = enabled;
            if (enabled)
            {
                log_level = LogLevel::DEBUG;
                verbose_thought_chains = true;
                verbose_loss_scaling = true;
                verbose_gradient_flow = true;
                verbose_kv_cache = true;
            }
            else
            {
                log_level = LogLevel::TRACE;
                verbose_loss_scaling = false;
                verbose_gradient_flow = false;
                verbose_kv_cache = false;
            }
        }

        /// Helper to print formatted configuration summary
        void print_config_summary() const
        {
            std::cout << "\n========================================================================\n";
            std::cout << "  RINGWRAPPER RUNTIME CONFIGURATION MANIFEST\n";
            std::cout << "========================================================================\n";
            std::cout << "  * Calculus of Constructions: Enabled=" << (enable_coc_verification ? "YES" : "NO")
                      << " | Interval=Every " << coc_verification_interval << " steps | Alpha=" << std::fixed << std::setprecision(2) << coc_type_guidance_alpha << "\n";
            std::cout << "  * Directional Physics:       Damped Reversal=" << (enable_damped_operation_reversal ? "YES" : "NO")
                      << " | Reversal Factor=" << reversal_shrink_factor << " | Clip=" << global_gradient_clip_norm << "\n";
            std::cout << "  * Stability & Rollback:      Rollback=" << (enable_weight_rollback_recovery ? "YES" : "NO")
                      << " | Explosion Cap=" << bad_batch_loss_threshold << " | Soft-Cap=" << logit_soft_cap << "\n";
            std::cout << "  * Multi-Formula Routing:     Physics=" << (enable_multi_formula_routing ? "DYNAMIC 4-WAY" : "ADAMW")
                      << " | F1=" << f1_natural_gradient_threshold << " | F2=" << f2_nesterov_threshold << " | F3=" << f3_adamw_threshold << "\n";
            std::cout << "  * Taylor & Curvature:        Taylor Forecast=" << (enable_taylor_prediction ? "ACTIVE" : "OFF")
                      << " | Rayleigh Curvature=" << (enable_rayleigh_curvature ? "ACTIVE" : "OFF") << "\n";
            std::cout << "  * Generation Sampling:       Temp=" << default_temperature << " | Top-P=" << default_top_p
                      << " | Min-P=" << default_min_p << " | Rep-Pen=" << default_repetition_penalty << "\n";
            std::cout << "========================================================================\n\n";
        }

        /// Helper to print debug messages with formatted prefix
        static void log_debug(const std::string &msg)
        {
            if (get_instance().debug_mode)
            {
                std::cout << "  [DEBUG] " << msg << "\n";
            }
        }
    };

    /**
     * @class RunLogger
     * @brief Manages dedicated timestamped text log files in logs/ directory for full run diagnostics.
     */
    class RunLogger
    {
    private:
        std::ofstream log_file;
        std::string log_path;
        bool is_open = false;

    public:
        static RunLogger &get_instance()
        {
            static RunLogger instance;
            return instance;
        }

        void initialize(const std::string &log_dir = "logs")
        {
            try
            {
                std::filesystem::create_directories(log_dir);
                auto now = std::chrono::system_clock::now();
                auto in_time_t = std::chrono::system_clock::to_time_t(now);
                std::tm time_info;
#if defined(_WIN32) || defined(_WIN64)
                localtime_s(&time_info, &in_time_t);
#else
                localtime_r(&in_time_t, &time_info);
#endif
                char buf[64];
                std::strftime(buf, sizeof(buf), "%Y%m%d_%H%M%S", &time_info);
                log_path = log_dir + "/debug_run_" + std::string(buf) + ".txt";
                log_file.open(log_path, std::ios::out | std::ios::trunc);
                if (log_file.is_open())
                {
                    is_open = true;
                    log_file << "========================================================================\n";
                    log_file << " RINGWRAPPER NEURALNET SYSTEM EXECUTION & TRAINING DEBUG LOG\n";
                    log_file << " Session Timestamp: " << buf << "\n";
                    log_file << " Log File Path:     " << log_path << "\n";
                    log_file << "========================================================================\n\n";
                    log_file.flush();
                }
            }
            catch (...)
            {
            }
        }

        void log(const std::string &category, const std::string &msg)
        {
            if (!is_open)
                initialize();
            if (is_open)
            {
                log_file << "[" << category << "] " << msg << "\n";
                log_file.flush();
            }
        }

        void log_raw(const std::string &text)
        {
            if (!is_open)
                initialize();
            if (is_open)
            {
                log_file << text;
                log_file.flush();
            }
        }

        const std::string &get_log_path() const { return log_path; }
    };

    inline void log_debug_file(const std::string &category, const std::string &msg)
    {
        RunLogger::get_instance().log(category, msg);
    }

    inline void log_debug_raw(const std::string &text)
    {
        RunLogger::get_instance().log_raw(text);
    }

    /// Convenience global getter for debug mode status
    inline bool is_debug_mode()
    {
        return RuntimeConfig::get_instance().debug_mode;
    }

    inline RuntimeConfig &get_config()
    {
        return RuntimeConfig::get_instance();
    }

} // namespace ring0