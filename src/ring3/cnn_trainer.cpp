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
    rebuild_optimizer_params();
}

void CNNTrainer::rebuild_optimizer_params() {
    optimizer.m_list.clear();
    optimizer.v_list.clear();
    optimizer.fisher_diag.clear();
    optimizer.decay_mask.clear();
    optimizer.last_shifts.clear();
    optimizer.layer_scales.clear();
    optimizer.layer_attributions.clear();

    for (const auto& conv : model.conv_layers) {
        size_t fan_in = conv.in_channels * conv.kernel_h * conv.kernel_w;
        ring0::Matrix w_mat(conv.out_channels, fan_in);
        ring0::Matrix b_mat(1, conv.biases.size());
        optimizer.register_param(w_mat, true);
        optimizer.register_param(b_mat, false);
    }
    for (const auto& attn : model.attention_blocks) {
        optimizer.register_param(attn.W_q, true);
        optimizer.register_param(attn.b_q, false);
        optimizer.register_param(attn.W_k, true);
        optimizer.register_param(attn.b_k, false);
        optimizer.register_param(attn.W_v, true);
        optimizer.register_param(attn.b_v, false);
        optimizer.register_param(attn.W_o, true);
        optimizer.register_param(attn.b_o, false);
    }
    for (const auto& tb : model.transformer_blocks) {
        optimizer.register_param(tb.attention.W_q, true);
        optimizer.register_param(tb.attention.b_q, false);
        optimizer.register_param(tb.attention.W_k, true);
        optimizer.register_param(tb.attention.b_k, false);
        optimizer.register_param(tb.attention.W_v, true);
        optimizer.register_param(tb.attention.b_v, false);
        optimizer.register_param(tb.attention.W_o, true);
        optimizer.register_param(tb.attention.b_o, false);
        optimizer.register_param(tb.ln1_gamma, true);
        optimizer.register_param(tb.ln1_beta, false);
        optimizer.register_param(tb.ln2_gamma, true);
        optimizer.register_param(tb.ln2_beta, false);
        optimizer.register_param(tb.W_gate, true);
        optimizer.register_param(tb.b_gate, false);
        optimizer.register_param(tb.W_up, true);
        optimizer.register_param(tb.b_up, false);
        optimizer.register_param(tb.W_down, true);
        optimizer.register_param(tb.b_down, false);
    }
    for (const auto& rec : model.recursive_layers) {
        optimizer.register_param(rec.W_think, true);
        optimizer.register_param(rec.b_think, false);
        optimizer.register_param(rec.W_context, true);
    }
    if (model.use_neural_net_head) {
        for (const auto& dense : model.dense_head.layers) {
            optimizer.register_param(dense.weights, true);
            optimizer.register_param(dense.biases, false);
        }
    } else {
        for (const auto& dense : model.dense_layers) {
            optimizer.register_param(dense.weights, true);
            optimizer.register_param(dense.biases, false);
        }
    }
}

std::vector<float> CNNTrainer::extract_global_gradient_unit_vector(float& out_norm) {
    size_t total_p = model.get_total_parameters();
    std::vector<float> grad_vec;
    grad_vec.reserve(total_p);

    // 1. Conv gradients
    for (const auto& conv : model.conv_layers) {
        for (float g : conv.grad_weights.data) grad_vec.push_back(g);
        for (float b : conv.grad_biases) grad_vec.push_back(b);
    }

    // 2. Attention gradients
    for (const auto& attn : model.attention_blocks) {
        for (float g : attn.grad_W_q.data) grad_vec.push_back(g);
        for (float g : attn.grad_b_q.data) grad_vec.push_back(g);
        for (float g : attn.grad_W_k.data) grad_vec.push_back(g);
        for (float g : attn.grad_b_k.data) grad_vec.push_back(g);
        for (float g : attn.grad_W_v.data) grad_vec.push_back(g);
        for (float g : attn.grad_b_v.data) grad_vec.push_back(g);
        for (float g : attn.grad_W_o.data) grad_vec.push_back(g);
        for (float g : attn.grad_b_o.data) grad_vec.push_back(g);
    }

    // 3. Transformer gradients
    for (const auto& tb : model.transformer_blocks) {
        for (float g : tb.attention.grad_W_q.data) grad_vec.push_back(g);
        for (float g : tb.attention.grad_b_q.data) grad_vec.push_back(g);
        for (float g : tb.attention.grad_W_k.data) grad_vec.push_back(g);
        for (float g : tb.attention.grad_b_k.data) grad_vec.push_back(g);
        for (float g : tb.attention.grad_W_v.data) grad_vec.push_back(g);
        for (float g : tb.attention.grad_b_v.data) grad_vec.push_back(g);
        for (float g : tb.attention.grad_W_o.data) grad_vec.push_back(g);
        for (float g : tb.attention.grad_b_o.data) grad_vec.push_back(g);
        for (float g : tb.grad_ln1_gamma.data) grad_vec.push_back(g);
        for (float g : tb.grad_ln1_beta.data) grad_vec.push_back(g);
        for (float g : tb.grad_ln2_gamma.data) grad_vec.push_back(g);
        for (float g : tb.grad_ln2_beta.data) grad_vec.push_back(g);
        for (float g : tb.grad_W_gate.data) grad_vec.push_back(g);
        for (float g : tb.grad_b_gate.data) grad_vec.push_back(g);
        for (float g : tb.grad_W_up.data) grad_vec.push_back(g);
        for (float g : tb.grad_b_up.data) grad_vec.push_back(g);
        for (float g : tb.grad_W_down.data) grad_vec.push_back(g);
        for (float g : tb.grad_b_down.data) grad_vec.push_back(g);
    }

    // 4. Recursive gradients
    for (const auto& rec : model.recursive_layers) {
        for (float g : rec.grad_W_think.data) grad_vec.push_back(g);
        for (float g : rec.grad_b_think.data) grad_vec.push_back(g);
        for (float g : rec.grad_W_context.data) grad_vec.push_back(g);
    }

    // 5. Dense / NeuralNet gradients
    if (model.use_neural_net_head) {
        for (const auto& dense : model.dense_head.layers) {
            for (float g : dense.grad_weights.data) grad_vec.push_back(g);
            for (float b : dense.grad_biases.data) grad_vec.push_back(b);
        }
    } else {
        for (const auto& dense : model.dense_layers) {
            for (float g : dense.grad_weights.data) grad_vec.push_back(g);
            for (float b : dense.grad_biases.data) grad_vec.push_back(b);
        }
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
    for (auto& attn : model.attention_blocks) {
        for (float& g : attn.grad_W_q.data) if (idx < global_grad_vector.size()) g = global_grad_vector[idx++];
        for (float& g : attn.grad_b_q.data) if (idx < global_grad_vector.size()) g = global_grad_vector[idx++];
        for (float& g : attn.grad_W_k.data) if (idx < global_grad_vector.size()) g = global_grad_vector[idx++];
        for (float& g : attn.grad_b_k.data) if (idx < global_grad_vector.size()) g = global_grad_vector[idx++];
        for (float& g : attn.grad_W_v.data) if (idx < global_grad_vector.size()) g = global_grad_vector[idx++];
        for (float& g : attn.grad_b_v.data) if (idx < global_grad_vector.size()) g = global_grad_vector[idx++];
        for (float& g : attn.grad_W_o.data) if (idx < global_grad_vector.size()) g = global_grad_vector[idx++];
        for (float& g : attn.grad_b_o.data) if (idx < global_grad_vector.size()) g = global_grad_vector[idx++];
    }
    for (auto& tb : model.transformer_blocks) {
        for (float& g : tb.attention.grad_W_q.data) if (idx < global_grad_vector.size()) g = global_grad_vector[idx++];
        for (float& g : tb.attention.grad_b_q.data) if (idx < global_grad_vector.size()) g = global_grad_vector[idx++];
        for (float& g : tb.attention.grad_W_k.data) if (idx < global_grad_vector.size()) g = global_grad_vector[idx++];
        for (float& g : tb.attention.grad_b_k.data) if (idx < global_grad_vector.size()) g = global_grad_vector[idx++];
        for (float& g : tb.attention.grad_W_v.data) if (idx < global_grad_vector.size()) g = global_grad_vector[idx++];
        for (float& g : tb.attention.grad_b_v.data) if (idx < global_grad_vector.size()) g = global_grad_vector[idx++];
        for (float& g : tb.attention.grad_W_o.data) if (idx < global_grad_vector.size()) g = global_grad_vector[idx++];
        for (float& g : tb.attention.grad_b_o.data) if (idx < global_grad_vector.size()) g = global_grad_vector[idx++];
        for (float& g : tb.grad_ln1_gamma.data) if (idx < global_grad_vector.size()) g = global_grad_vector[idx++];
        for (float& g : tb.grad_ln1_beta.data) if (idx < global_grad_vector.size()) g = global_grad_vector[idx++];
        for (float& g : tb.grad_ln2_gamma.data) if (idx < global_grad_vector.size()) g = global_grad_vector[idx++];
        for (float& g : tb.grad_ln2_beta.data) if (idx < global_grad_vector.size()) g = global_grad_vector[idx++];
        for (float& g : tb.grad_W_gate.data) if (idx < global_grad_vector.size()) g = global_grad_vector[idx++];
        for (float& g : tb.grad_b_gate.data) if (idx < global_grad_vector.size()) g = global_grad_vector[idx++];
        for (float& g : tb.grad_W_up.data) if (idx < global_grad_vector.size()) g = global_grad_vector[idx++];
        for (float& g : tb.grad_b_up.data) if (idx < global_grad_vector.size()) g = global_grad_vector[idx++];
        for (float& g : tb.grad_W_down.data) if (idx < global_grad_vector.size()) g = global_grad_vector[idx++];
        for (float& g : tb.grad_b_down.data) if (idx < global_grad_vector.size()) g = global_grad_vector[idx++];
    }
    for (auto& rec : model.recursive_layers) {
        for (float& g : rec.grad_W_think.data) if (idx < global_grad_vector.size()) g = global_grad_vector[idx++];
        for (float& g : rec.grad_b_think.data) if (idx < global_grad_vector.size()) g = global_grad_vector[idx++];
        for (float& g : rec.grad_W_context.data) if (idx < global_grad_vector.size()) g = global_grad_vector[idx++];
    }
    if (model.use_neural_net_head) {
        for (auto& dense : model.dense_head.layers) {
            for (float& g : dense.grad_weights.data) {
                if (idx < global_grad_vector.size()) g = global_grad_vector[idx++];
            }
            for (float& b : dense.grad_biases.data) {
                if (idx < global_grad_vector.size()) b = global_grad_vector[idx++];
            }
        }
    } else {
        for (auto& dense : model.dense_layers) {
            for (float& g : dense.grad_weights.data) {
                if (idx < global_grad_vector.size()) g = global_grad_vector[idx++];
            }
            for (float& b : dense.grad_biases.data) {
                if (idx < global_grad_vector.size()) b = global_grad_vector[idx++];
            }
        }
    }
}

float CNNTrainer::perform_auto_gradient_normalization() {
    // 1. Extract unit vector and Frobenius L2 norm
    float raw_norm = 0.0f;
    std::vector<float> unit_grad = extract_global_gradient_unit_vector(raw_norm);
    last_raw_grad_norm = raw_norm;

    if (!config.enable_auto_grad_norm || raw_norm < 1e-7f) {
        last_norm_scale = 1.0f;
        return raw_norm;
    }

    // 2. Apply Past Mistake Repulsion
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
    optimizer.set_learning_rate(effective_lr);
    size_t p_idx = 0;

    for (auto& conv : model.conv_layers) {
        size_t fan_in = conv.in_channels * conv.kernel_h * conv.kernel_w;
        ring0::Matrix w_mat(conv.out_channels, fan_in);
        w_mat.data = conv.weights.data;
        ring0::Matrix gw_mat(conv.out_channels, fan_in);
        gw_mat.data = conv.grad_weights.data;
        optimizer.update_param(p_idx++, w_mat, gw_mat);
        conv.weights.data = std::move(w_mat.data);

        ring0::Matrix b_mat(1, conv.biases.size());
        b_mat.data = conv.biases;
        ring0::Matrix gb_mat(1, conv.grad_biases.size());
        gb_mat.data = conv.grad_biases;
        optimizer.update_param(p_idx++, b_mat, gb_mat);
        conv.biases = std::move(b_mat.data);
    }

    for (auto& attn : model.attention_blocks) {
        optimizer.update_param(p_idx++, attn.W_q, attn.grad_W_q);
        optimizer.update_param(p_idx++, attn.b_q, attn.grad_b_q);
        optimizer.update_param(p_idx++, attn.W_k, attn.grad_W_k);
        optimizer.update_param(p_idx++, attn.b_k, attn.grad_b_k);
        optimizer.update_param(p_idx++, attn.W_v, attn.grad_W_v);
        optimizer.update_param(p_idx++, attn.b_v, attn.grad_b_v);
        optimizer.update_param(p_idx++, attn.W_o, attn.grad_W_o);
        optimizer.update_param(p_idx++, attn.b_o, attn.grad_b_o);
    }

    for (auto& tb : model.transformer_blocks) {
        optimizer.update_param(p_idx++, tb.attention.W_q, tb.attention.grad_W_q);
        optimizer.update_param(p_idx++, tb.attention.b_q, tb.attention.grad_b_q);
        optimizer.update_param(p_idx++, tb.attention.W_k, tb.attention.grad_W_k);
        optimizer.update_param(p_idx++, tb.attention.b_k, tb.attention.grad_b_k);
        optimizer.update_param(p_idx++, tb.attention.W_v, tb.attention.grad_W_v);
        optimizer.update_param(p_idx++, tb.attention.b_v, tb.attention.grad_b_v);
        optimizer.update_param(p_idx++, tb.attention.W_o, tb.attention.grad_W_o);
        optimizer.update_param(p_idx++, tb.attention.b_o, tb.attention.grad_b_o);
        optimizer.update_param(p_idx++, tb.ln1_gamma, tb.grad_ln1_gamma);
        optimizer.update_param(p_idx++, tb.ln1_beta, tb.grad_ln1_beta);
        optimizer.update_param(p_idx++, tb.ln2_gamma, tb.grad_ln2_gamma);
        optimizer.update_param(p_idx++, tb.ln2_beta, tb.grad_ln2_beta);
        optimizer.update_param(p_idx++, tb.W_gate, tb.grad_W_gate);
        optimizer.update_param(p_idx++, tb.b_gate, tb.grad_b_gate);
        optimizer.update_param(p_idx++, tb.W_up, tb.grad_W_up);
        optimizer.update_param(p_idx++, tb.b_up, tb.grad_b_up);
        optimizer.update_param(p_idx++, tb.W_down, tb.grad_W_down);
        optimizer.update_param(p_idx++, tb.b_down, tb.grad_b_down);
    }

    for (auto& rec : model.recursive_layers) {
        optimizer.update_param(p_idx++, rec.W_think, rec.grad_W_think);
        optimizer.update_param(p_idx++, rec.b_think, rec.grad_b_think);
        optimizer.update_param(p_idx++, rec.W_context, rec.grad_W_context);
    }

    if (model.use_neural_net_head) {
        for (auto& dense : model.dense_head.layers) {
            optimizer.update_param(p_idx++, dense.weights, dense.grad_weights);
            optimizer.update_param(p_idx++, dense.biases, dense.grad_biases);
        }
    } else {
        for (auto& dense : model.dense_layers) {
            optimizer.update_param(p_idx++, dense.weights, dense.grad_weights);
            optimizer.update_param(p_idx++, dense.biases, dense.grad_biases);
        }
    }
    optimizer.timestep++;
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
                rebuild_optimizer_params();
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
