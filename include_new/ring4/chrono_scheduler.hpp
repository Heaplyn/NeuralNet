#pragma once

/**
 * @file chrono_scheduler.hpp
 * @brief Concurrent Multi-Part Chrono Asynchronous Subsystem Scheduler in Ring 4.
 *        Executes multiple AI subsystems concurrently on independent std::chrono timers
 *        (Meta-Optimizer, Taylor Forecaster, CoC Proof Verifier, Vocab Cluster Miner, Watchdog Auditor).
 */

#include "ring0/config.hpp"
#include "ring0/taylor_predictor.hpp"
#include "ring0/calculus_of_constructions.hpp"
#include "ring1/meta_loss_optimizer.hpp"
#include "ring2/vocab_manager.hpp"
#include "ring2/transformer_lm.hpp"

#include <thread>
#include <mutex>
#include <atomic>
#include <chrono>
#include <vector>
#include <memory>
#include <string>

namespace ring4 {

/**
 * @struct SubsystemTelemetrySnapshot
 * @brief Thread-safe telemetry snapshot shared between main training loop and concurrent chrono threads.
 */
struct SubsystemTelemetrySnapshot {
    std::atomic<float> current_loss{5.0f};
    std::atomic<float> ema_loss_short{5.0f};
    std::atomic<float> ema_loss_long{5.0f};
    std::atomic<float> current_lr{0.001f};
    std::atomic<float> current_grad_norm{0.5f};
    std::atomic<size_t> current_step{0};
    std::atomic<size_t> current_seq_len{32};
    std::atomic<size_t> current_vocab_size{256};
    std::atomic<float> current_penalty{0.20f};
    std::atomic<bool> is_training_active{false};

    // Outputs updated by concurrent chrono threads:
    std::atomic<float> meta_loss_scale{1.0f};
    std::atomic<float> meta_focal_gamma{1.0f};
    std::atomic<float> meta_lr_modulator{1.0f};
    std::atomic<float> meta_curvature_scale{1.0f};

    std::atomic<float> taylor_reward{0.0f};
    std::atomic<float> taylor_confidence{0.0f};
    std::atomic<float> taylor_pred_delta{0.0f};

    std::atomic<float> coc_proof_consistency{1.0f};
    std::atomic<bool> coc_proof_valid{true};
    std::atomic<size_t> coc_verified_proofs_count{0};

    std::atomic<size_t> vocab_clusters_mined{0};
    std::atomic<float> vocab_cluster_loss{0.0f};

    std::atomic<bool> watchdog_alert{false};
    std::atomic<size_t> total_chrono_ticks{0};
};

/**
 * @class ChronoAsyncEngine
 * @brief High-performance concurrent engine running independent AI subsystems on dedicated std::chrono timers.
 */
class ChronoAsyncEngine {
private:
    std::atomic<bool> is_running{false};

    // Dedicated worker threads for concurrent subsystems
    std::thread meta_opt_thread;
    std::thread taylor_thread;
    std::thread coc_verifier_thread;
    std::thread vocab_miner_thread;
    std::thread watchdog_thread;

    // Mutex for protecting heavy data structures during async introspection
    mutable std::mutex meta_mutex;
    mutable std::mutex taylor_mutex;
    mutable std::mutex coc_mutex;
    mutable std::mutex vocab_mutex;

    // Subsystem instances owned or referenced
    ring1::MetaLossOptimizer meta_optimizer;
    ring0::TaylorTrajectoryPredictor taylor_predictor;
    ring0::TypingContext coc_logic_context;
    ring2::VocabManager* vocab_manager_ref = nullptr;
    ring2::TransformerLM* model_ref = nullptr;

    // Cadence configuration in chrono intervals
    std::chrono::milliseconds meta_opt_interval{120};     ///< Meta-Optimizer cycle (120ms)
    std::chrono::milliseconds taylor_interval{180};       ///< Taylor loss trajectory forecast (180ms)
    std::chrono::milliseconds coc_interval{250};          ///< CoC formal proof check (250ms)
    std::chrono::milliseconds vocab_interval{400};        ///< Vocab clustering & lexicon mining (400ms)
    std::chrono::milliseconds watchdog_interval{60};      ///< Stability watchdog & telemetry audit (60ms)

public:
    SubsystemTelemetrySnapshot telemetry;

    ChronoAsyncEngine();
    ~ChronoAsyncEngine();

    /**
     * @brief Attaches optional references to the model and vocabulary manager.
     */
    void attach_components(ring2::TransformerLM* model, ring2::VocabManager* vocab_mgr);

    /**
     * @brief Launches all concurrent subsystem threads on their respective chrono timers.
     */
    void start();

    /**
     * @brief Gracefully terminates all concurrent subsystem threads and joins them.
     */
    void stop();

    /**
     * @brief Updates training state snapshot from the main training loop.
     */
    void update_telemetry(size_t step, float loss, float lr, float grad_norm, size_t seq_len, float penalty);

    /**
     * @brief Fetches latest meta-optimization dynamic modulation values.
     */
    ring1::MetaOptimizationOutput get_latest_meta_output() const;

    /**
     * @brief Fetches latest Taylor forecast metrics.
     */
    void get_latest_taylor_metrics(float& out_reward, float& out_conf, float& out_pred_delta) const;

    /**
     * @brief Fetches latest CoC proof verification metrics.
     */
    void get_latest_coc_metrics(float& out_consistency, bool& out_valid, size_t& out_verified_count) const;

    bool is_active() const { return is_running.load(); }
};

} // namespace ring4
