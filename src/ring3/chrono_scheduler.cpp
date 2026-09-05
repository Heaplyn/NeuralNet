#include "ring3/chrono_scheduler.hpp"
#include <iostream>
#include <cmath>

namespace ring3 {

ChronoAsyncEngine::ChronoAsyncEngine()
    : is_running(false) {
    coc_logic_context = ring0::CoCTypeChecker::create_standard_logic_context();
}

ChronoAsyncEngine::~ChronoAsyncEngine() {
    stop();
}

void ChronoAsyncEngine::attach_components(ring2::TransformerLM* model, ring2::VocabManager* vocab_mgr) {
    model_ref = model;
    vocab_manager_ref = vocab_mgr;
}

void ChronoAsyncEngine::update_telemetry(size_t step, float loss, float lr, float grad_norm, size_t seq_len, float penalty) {
    telemetry.current_step.store(step, std::memory_order_relaxed);
    telemetry.current_loss.store(loss, std::memory_order_relaxed);
    telemetry.current_lr.store(lr, std::memory_order_relaxed);
    telemetry.current_grad_norm.store(grad_norm, std::memory_order_relaxed);
    telemetry.current_seq_len.store(seq_len, std::memory_order_relaxed);
    telemetry.current_penalty.store(penalty, std::memory_order_relaxed);

    float prev_ema_s = telemetry.ema_loss_short.load(std::memory_order_relaxed);
    float new_ema_s = 0.05f * loss + 0.95f * prev_ema_s;
    telemetry.ema_loss_short.store(new_ema_s, std::memory_order_relaxed);

    float prev_ema_l = telemetry.ema_loss_long.load(std::memory_order_relaxed);
    float new_ema_l = 0.01f * loss + 0.99f * prev_ema_l;
    telemetry.ema_loss_long.store(new_ema_l, std::memory_order_relaxed);
}

void ChronoAsyncEngine::start() {
    if (is_running.load()) return;
    is_running.store(true);
    telemetry.is_training_active.store(true);

    // 1. Thread 1: Concurrent Meta-Loss Optimizer
    meta_opt_thread = std::thread([this]() {
        while (is_running.load()) {
            std::this_thread::sleep_for(meta_opt_interval);
            if (!is_running.load()) break;

            float l = telemetry.current_loss.load(std::memory_order_relaxed);
            float ema_s = telemetry.ema_loss_short.load(std::memory_order_relaxed);
            float lr = telemetry.current_lr.load(std::memory_order_relaxed);
            float gn = telemetry.current_grad_norm.load(std::memory_order_relaxed);
            float tr = telemetry.taylor_reward.load(std::memory_order_relaxed);
            float tc = telemetry.taylor_confidence.load(std::memory_order_relaxed);

            if (std::isnan(l) || std::isinf(l)) continue;

            ring1::MetaLossTelemetry met_tel;
            met_tel.current_loss = l;
            met_tel.delta_loss = l - ema_s;
            met_tel.accel_loss = 0.0f;
            met_tel.gradient_variance = gn * gn * 0.1f;
            met_tel.learning_rate = lr;
            met_tel.trajectory_reward = tr;
            met_tel.trajectory_confidence = tc;

            ring1::MetaOptimizationOutput out;
            {
                std::lock_guard<std::mutex> lock(meta_mutex);
                out = meta_optimizer.predict(met_tel);
                meta_optimizer.update_online(l, tr, 0.5f);
            }

            telemetry.meta_loss_scale.store(out.loss_scale_multiplier, std::memory_order_relaxed);
            telemetry.meta_focal_gamma.store(out.dynamic_focal_gamma, std::memory_order_relaxed);
            telemetry.meta_lr_modulator.store(out.lr_step_modulator, std::memory_order_relaxed);
            telemetry.meta_curvature_scale.store(out.curvature_scale, std::memory_order_relaxed);
            telemetry.total_chrono_ticks.fetch_add(1, std::memory_order_relaxed);
        }
    });

    // 2. Thread 2: Concurrent Taylor Loss-Trajectory Forecaster & Curvature Prober
    taylor_thread = std::thread([this]() {
        std::vector<float> loss_hist;
        while (is_running.load()) {
            std::this_thread::sleep_for(taylor_interval);
            if (!is_running.load()) break;

            float l = telemetry.current_loss.load(std::memory_order_relaxed);
            if (std::isnan(l) || std::isinf(l)) continue;

            loss_hist.push_back(l);
            if (loss_hist.size() > 30) {
                loss_hist.erase(loss_hist.begin());
            }

            ring0::TaylorTrajectory forecast;
            {
                std::lock_guard<std::mutex> lock(taylor_mutex);
                forecast = taylor_predictor.observe(loss_hist);
            }

            if (forecast.valid) {
                telemetry.taylor_reward.store(forecast.reward, std::memory_order_relaxed);
                telemetry.taylor_confidence.store(forecast.confidence, std::memory_order_relaxed);
                telemetry.taylor_pred_delta.store(forecast.pred_delta[0], std::memory_order_relaxed);
            }
            telemetry.total_chrono_ticks.fetch_add(1, std::memory_order_relaxed);
        }
    });

    // 3. Thread 3: Concurrent Calculus of Constructions (CoC) Formal Proof Verification Engine
    coc_verifier_thread = std::thread([this]() {
        while (is_running.load()) {
            std::this_thread::sleep_for(coc_interval);
            if (!is_running.load()) break;

            // Formally check inductive Modus Ponens proof witness consistency in CoC
            using namespace ring0;
            auto var_P = CoCTerm::make_var("P");
            auto var_Q = CoCTerm::make_var("Q");
            auto p_impl_q = CoCTerm::make_arrow(var_P, var_Q);
            auto mp_type = CoCTerm::make_arrow(p_impl_q, CoCTerm::make_arrow(var_P, var_Q));
            auto mp_witness = CoCTerm::make_abstraction("f", p_impl_q,
                                CoCTerm::make_abstraction("p", var_P,
                                    CoCTerm::make_application(CoCTerm::make_var("f"), CoCTerm::make_var("p"))));

            ProofValidationResult result;
            {
                std::lock_guard<std::mutex> lock(coc_mutex);
                result = CoCTypeChecker::verify_proof(coc_logic_context, mp_witness, mp_type);
            }

            telemetry.coc_proof_consistency.store(result.proof_consistency_score, std::memory_order_relaxed);
            telemetry.coc_proof_valid.store(result.is_valid, std::memory_order_relaxed);
            telemetry.coc_verified_proofs_count.fetch_add(1, std::memory_order_relaxed);
            telemetry.total_chrono_ticks.fetch_add(1, std::memory_order_relaxed);
        }
    });

    // 4. Thread 4: Concurrent Vocab Manager & Semantic Clustering Engine
    vocab_miner_thread = std::thread([this]() {
        while (is_running.load()) {
            std::this_thread::sleep_for(vocab_interval);
            if (!is_running.load()) break;

            if (vocab_manager_ref && model_ref) {
                std::lock_guard<std::mutex> lock(vocab_mutex);
                auto v_tel = vocab_manager_ref->evaluate_vocab_layers(*model_ref);
                telemetry.vocab_clusters_mined.store(v_tel.layer2_categories, std::memory_order_relaxed);
                telemetry.vocab_cluster_loss.store(v_tel.category_clustering_loss, std::memory_order_relaxed);
            }
            telemetry.total_chrono_ticks.fetch_add(1, std::memory_order_relaxed);
        }
    });

    // 5. Thread 5: Concurrent Watchdog Stability & Telemetry Auditor
    watchdog_thread = std::thread([this]() {
        while (is_running.load()) {
            std::this_thread::sleep_for(watchdog_interval);
            if (!is_running.load()) break;

            float l = telemetry.current_loss.load(std::memory_order_relaxed);
            float ema_s = telemetry.ema_loss_short.load(std::memory_order_relaxed);

            bool alert = (!std::isnan(l) && !std::isinf(l) && l > ema_s + 1.2f);
            telemetry.watchdog_alert.store(alert, std::memory_order_relaxed);
            telemetry.total_chrono_ticks.fetch_add(1, std::memory_order_relaxed);
        }
    });
}

void ChronoAsyncEngine::stop() {
    if (!is_running.load()) return;
    is_running.store(false);
    telemetry.is_training_active.store(false);

    if (meta_opt_thread.joinable()) meta_opt_thread.join();
    if (taylor_thread.joinable()) taylor_thread.join();
    if (coc_verifier_thread.joinable()) coc_verifier_thread.join();
    if (vocab_miner_thread.joinable()) vocab_miner_thread.join();
    if (watchdog_thread.joinable()) watchdog_thread.join();
}

ring1::MetaOptimizationOutput ChronoAsyncEngine::get_latest_meta_output() const {
    ring1::MetaOptimizationOutput out;
    out.loss_scale_multiplier = telemetry.meta_loss_scale.load(std::memory_order_relaxed);
    out.dynamic_focal_gamma = telemetry.meta_focal_gamma.load(std::memory_order_relaxed);
    out.lr_step_modulator = telemetry.meta_lr_modulator.load(std::memory_order_relaxed);
    out.curvature_scale = telemetry.meta_curvature_scale.load(std::memory_order_relaxed);
    return out;
}

void ChronoAsyncEngine::get_latest_taylor_metrics(float& out_reward, float& out_conf, float& out_pred_delta) const {
    out_reward = telemetry.taylor_reward.load(std::memory_order_relaxed);
    out_conf = telemetry.taylor_confidence.load(std::memory_order_relaxed);
    out_pred_delta = telemetry.taylor_pred_delta.load(std::memory_order_relaxed);
}

void ChronoAsyncEngine::get_latest_coc_metrics(float& out_consistency, bool& out_valid, size_t& out_verified_count) const {
    out_consistency = telemetry.coc_proof_consistency.load(std::memory_order_relaxed);
    out_valid = telemetry.coc_proof_valid.load(std::memory_order_relaxed);
    out_verified_count = telemetry.coc_verified_proofs_count.load(std::memory_order_relaxed);
}

} // namespace ring3
