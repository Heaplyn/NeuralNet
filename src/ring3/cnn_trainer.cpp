#include "ring3/cnn_trainer.hpp"
#include <iostream>
#include <numeric>
#include <algorithm>
#include <random>
#include <cmath>

namespace ring3 {

CNNTrainer::CNNTrainer(CNN& net, CNNTrainingConfig cfg)
    : model(net), config(cfg) {
    optimizer.config.lr = cfg.learning_rate;
    optimizer.config.weight_decay = cfg.weight_decay;
    running_grad_norm_ema = cfg.target_grad_norm;
}

std::vector<float> CNNTrainer::extract_global_gradient_unit_vector(float& out_norm) {
    size_t total_p = model.get_total_parameters();
    std::vector<float> grad_vec;
    grad_vec.reserve(total_p);

    // Collect conv gradients
    for (const auto& conv : model.conv_layers) {
        for (float g : conv.grad_weights.data) grad_vec.push_back(g);
        for (float b : conv.grad_biases) grad_vec.push_back(b);
    }

    // Collect dense gradients
    for (const auto& dense : model.dense_layers) {
        for (float g : dense.grad_weights.data) grad_vec.push_back(g);
        for (float b : dense.grad_biases.data) grad_vec.push_back(b);
    }

    float norm_sq = 0.0f;
    for (float g : grad_vec) norm_sq += g * g;
    out_norm = std::sqrt(norm_sq);

    float inv_norm = 1.0f / (out_norm + 1e-9f);
    std::vector<float> unit_vec(grad_vec.size());
    for (size_t i = 0; i < grad_vec.size(); ++i) {
        unit_vec[i] = grad_vec[i] * inv_norm;
    }
    return unit_vec;
}

void CNNTrainer::apply_mistake_repulsion(std::vector<float>& global_grad_vector) {
    if (mistake_memory.empty() || !config.enable_mistake_memory) {
        last_mistake_similarity = 0.0f;
        return;
    }

    float max_sim = 0.0f;
    for (const auto& record : mistake_memory) {
        if (record.grad_signature.size() != global_grad_vector.size()) continue;

        float dot = 0.0f;
        for (size_t i = 0; i < global_grad_vector.size(); ++i) {
            dot += global_grad_vector[i] * record.grad_signature[i];
        }

        if (dot > 0.0f) {
            max_sim = std::max(max_sim, dot);
            float repulsion = config.mistake_repulsion_scale * std::sqrt(dot);
            for (size_t i = 0; i < global_grad_vector.size(); ++i) {
                global_grad_vector[i] -= repulsion * record.grad_signature[i];
            }
        }
    }
    last_mistake_similarity = max_sim;
}

void CNNTrainer::distribute_global_gradient_vector(const std::vector<float>& global_grad_vector) {
    size_t idx = 0;
    for (auto& conv : model.conv_layers) {
        for (float& g : conv.grad_weights.data) {
            if (idx < global_grad_vector.size()) g = global_grad_vector[idx++];
        }
        for (float& b : conv.grad_biases) {
            if (idx < global_grad_vector.size()) b = global_grad_vector[idx++];
        }
    }
    for (auto& dense : model.dense_layers) {
        for (float& g : dense.grad_weights.data) {
            if (idx < global_grad_vector.size()) g = global_grad_vector[idx++];
        }
        for (float& b : dense.grad_biases.data) {
            if (idx < global_grad_vector.size()) b = global_grad_vector[idx++];
        }
    }
}

float CNNTrainer::perform_auto_gradient_normalization() {
    float raw_norm = 0.0f;
    auto unit_grad = extract_global_gradient_unit_vector(raw_norm);
    last_raw_grad_norm = raw_norm;

    if (!config.enable_auto_grad_norm || raw_norm < 1e-9f) {
        last_norm_scale = 1.0f;
        return raw_norm;
    }

    // 1. Update running exponential moving average of gradient energy
    running_grad_norm_ema = (1.0f - config.auto_grad_ema_alpha) * running_grad_norm_ema +
                            config.auto_grad_ema_alpha * raw_norm;

    // 2. Apply Mistake Memory Repulsion to gradient trajectory
    apply_mistake_repulsion(unit_grad);

    // 3. Compute Auto Normalization Scale
    float target = config.target_grad_norm;
    float scale = target / std::max(1e-6f, raw_norm);
    if (scale * raw_norm > config.max_grad_norm_ceiling) {
        scale = config.max_grad_norm_ceiling / raw_norm;
    }
    last_norm_scale = scale;

    // 4. Re-scale unit gradient vector by target energy scale
    std::vector<float> final_grad(unit_grad.size());
    for (size_t i = 0; i < unit_grad.size(); ++i) {
        final_grad[i] = unit_grad[i] * (raw_norm * scale);
    }

    // 5. Write normalized gradients back to network parameter buffers
    distribute_global_gradient_vector(final_grad);
    return raw_norm * scale;
}

void CNNTrainer::update_model_parameters(float effective_lr) {
    float wd = config.weight_decay;

    // 1. Update Conv2D parameters
    for (auto& conv : model.conv_layers) {
        for (size_t i = 0; i < conv.weights.data.size(); ++i) {
            float g = conv.grad_weights.data[i];
            float w = conv.weights.data[i];
            w -= effective_lr * (g + wd * w);
            conv.weights.data[i] = w;
        }
        for (size_t i = 0; i < conv.biases.size(); ++i) {
            float g = conv.grad_biases[i];
            conv.biases[i] -= effective_lr * g;
        }
    }

    // 2. Update DenseLayer parameters
    for (auto& dense : model.dense_layers) {
        for (size_t i = 0; i < dense.weights.data.size(); ++i) {
            float g = dense.grad_weights.data[i];
            float w = dense.weights.data[i];
            w -= effective_lr * (g + wd * w);
            dense.weights.data[i] = w;
        }
        for (size_t i = 0; i < dense.biases.data.size(); ++i) {
            float g = dense.grad_biases.data[i];
            dense.biases.data[i] -= effective_lr * g;
        }
    }
}

void CNNTrainer::train(const ring0::Matrix& X_train,
                      const ring0::Matrix& Y_train,
                      const std::function<void(const CNNEpochMetrics&)>& on_epoch_end) {
    size_t num_samples = X_train.rows;
    size_t batch_size = config.batch_size;
    std::vector<size_t> indices(num_samples);
    std::iota(indices.begin(), indices.end(), 0);

    std::random_device rd;
    std::mt19937 g_rand(rd());

    for (size_t epoch = 1; epoch <= config.epochs; ++epoch) {
        float epoch_loss_sum = 0.0f;
        size_t total_batches = 0;

        // Shuffle indices per epoch using C++17 std::shuffle
        std::shuffle(indices.begin(), indices.end(), g_rand);

        for (size_t start = 0; start < num_samples; start += batch_size) {
            size_t current_bs = std::min(batch_size, num_samples - start);

            // Assemble mini-batch
            ring0::Matrix batch_x(current_bs, X_train.cols);
            ring0::Matrix batch_y(current_bs, Y_train.cols);
            for (size_t b = 0; b < current_bs; ++b) {
                size_t src_idx = indices[start + b];
                for (size_t c = 0; c < X_train.cols; ++c) batch_x(b, c) = X_train(src_idx, c);
                for (size_t c = 0; c < Y_train.cols; ++c) batch_y(b, c) = Y_train(src_idx, c);
            }

            model.reset_gradients();

            // 1. Forward pass
            ring0::Matrix logits = model.forward_flat(batch_x);

            // 2. Softmax probabilities and loss evaluation
            ring0::Matrix probs = logits.softmax_rows();
            float batch_loss = 0.0f;
            ring0::Matrix grad_logits(current_bs, Y_train.cols, 0.0f);

            for (size_t b = 0; b < current_bs; ++b) {
                for (size_t c = 0; c < Y_train.cols; ++c) {
                    float target = batch_y(b, c);
                    float prob = std::max(1e-12f, std::min(1.0f, probs(b, c)));
                    if (target > 0.5f) {
                        batch_loss -= std::log(prob);
                    }
                    grad_logits(b, c) = (probs(b, c) - target) / static_cast<float>(current_bs);
                }
            }
            batch_loss /= static_cast<float>(current_bs);
            epoch_loss_sum += batch_loss;
            total_batches++;

            // 3. Backward Pass
            model.backward(grad_logits);

            // 4. Auto Gradient Normalization & Mistake Repulsion
            perform_auto_gradient_normalization();

            // 5. Taylor Loss-Trajectory Foresight
            float taylor_scale = 1.0f;
            if (config.enable_taylor_forecast) {
                ring0::Loss::record_loss(batch_loss);
                auto forecast = taylor_forecaster.observe(ring0::Loss::loss_history);
                taylor_scale = forecast.lr_foresight_scale;
                last_taylor_confidence = forecast.confidence;
                last_taylor_lr_scale = taylor_scale;
            }

            // 6. Meta-Neural LR Modulation
            float meta_scale = 1.0f;
            if (config.enable_meta_loss_opt) {
                ring1::MetaLossTelemetry telem;
                telem.current_loss = ema_initialized ? ema_loss : batch_loss;
                telem.delta_loss = ema_initialized ? (batch_loss - ema_loss) : 0.0f;
                telem.gradient_variance = last_raw_grad_norm;
                telem.learning_rate = config.learning_rate;
                auto meta_out = meta_optimizer.predict(telem);
                meta_scale = meta_out.lr_step_modulator;
                last_meta_lr_scale = meta_scale;
                meta_optimizer.update_online(batch_loss);
            }

            // 7. Combined Applied Learning Rate & Parameter Update
            float applied_lr = config.learning_rate * taylor_scale * meta_scale;
            update_model_parameters(applied_lr);

            // 8. Update Loss EMA & Check for Mistake Spikes
            if (!ema_initialized) {
                ema_loss = batch_loss;
                ema_initialized = true;
            } else {
                if (batch_loss > ema_loss * config.mistake_spike_threshold && config.enable_mistake_memory) {
                    // Record past mistake checkpoint
                    float dummy_norm = 0.0f;
                    CNNMistakeRecord record;
                    record.loss = batch_loss;
                    record.baseline_ema = ema_loss;
                    record.grad_norm = last_raw_grad_norm;
                    record.model_params = model.get_total_parameters();
                    record.step = step_counter;
                    record.grad_signature = extract_global_gradient_unit_vector(dummy_norm);

                    if (mistake_memory.size() >= config.mistake_memory_capacity) {
                        mistake_memory.pop_front();
                    }
                    mistake_memory.push_back(record);
                }
                ema_loss = 0.95f * ema_loss + 0.05f * batch_loss;
            }
            step_counter++;
        }

        float avg_epoch_loss = epoch_loss_sum / std::max<size_t>(1, total_batches);
        float epoch_acc = evaluate_accuracy(X_train, Y_train);

        // Dynamic Capacity Sizing: Plateau or Mistake Alignment Check
        bool grew = false;
        if (config.enable_dynamic_growth) {
            if (avg_epoch_loss < best_loss - 0.005f) {
                best_loss = avg_epoch_loss;
                plateau_streak = 0;
            } else {
                plateau_streak++;
            }

            if (last_mistake_similarity > 0.40f) {
                mistake_alignment_streak++;
            } else {
                mistake_alignment_streak = 0;
            }

            if (plateau_streak >= config.growth_patience ||
                mistake_alignment_streak >= config.mistake_streak_growth_trigger) {
                model.expand_capacity(config.growth_multiplier);
                grew = true;
                plateau_streak = 0;
                mistake_alignment_streak = 0;
            }
        }

        if (on_epoch_end) {
            CNNEpochMetrics metrics;
            metrics.epoch = epoch;
            metrics.loss = avg_epoch_loss;
            metrics.accuracy = epoch_acc;
            metrics.raw_grad_norm = last_raw_grad_norm;
            metrics.normalized_grad_norm = last_raw_grad_norm * last_norm_scale;
            metrics.applied_lr = config.learning_rate * last_taylor_lr_scale * last_meta_lr_scale;
            metrics.meta_lr_scale = last_meta_lr_scale;
            metrics.taylor_lr_scale = last_taylor_lr_scale;
            metrics.taylor_confidence = last_taylor_confidence;
            metrics.stored_mistakes = mistake_memory.size();
            metrics.mistake_similarity = last_mistake_similarity;
            metrics.total_parameters = model.get_total_parameters();
            metrics.capacity_expanded = grew;
            on_epoch_end(metrics);
        }
    }
}

float CNNTrainer::evaluate_accuracy(const ring0::Matrix& X_test, const ring0::Matrix& Y_test) {
    if (X_test.rows == 0) return 0.0f;
    ring0::Matrix logits = model.forward_flat(X_test);
    size_t correct = 0;

    for (size_t b = 0; b < X_test.rows; ++b) {
        size_t pred_class = 0;
        float max_logit = logits(b, 0);
        for (size_t c = 1; c < logits.cols; ++c) {
            if (logits(b, c) > max_logit) {
                max_logit = logits(b, c);
                pred_class = c;
            }
        }

        size_t true_class = 0;
        float max_target = Y_test(b, 0);
        for (size_t c = 1; c < Y_test.cols; ++c) {
            if (Y_test(b, c) > max_target) {
                max_target = Y_test(b, c);
                true_class = c;
            }
        }

        if (pred_class == true_class) {
            correct++;
        }
    }
    return static_cast<float>(correct) / static_cast<float>(X_test.rows);
}

size_t CNNTrainer::predict_class(const ring0::Matrix& single_image) {
    ring0::Matrix logits = model.forward_flat(single_image);
    size_t pred = 0;
    float max_l = logits(0, 0);
    for (size_t c = 1; c < logits.cols; ++c) {
        if (logits(0, c) > max_l) {
            max_l = logits(0, c);
            pred = c;
        }
    }
    return pred;
}

} // namespace ring3
