#include <iostream>
#include <vector>
#include <numeric>
#include <random>
#include <algorithm>
#include <cmath>

#include "ring4/trainer.hpp"
#include "ring0/loss.hpp"

using namespace std;

namespace ring4
{

    // Constructor: Initializes trainer, growth controller, and momentum optimizer
    RingTrainer::RingTrainer(ring2::NeuralNet &network,
                             ring1::GradientDescent &opt,
                             const ring2::GrowthConfig &growth_cfg,
                             TrainingConfig train_cfg)
        : net(network), optimizer(opt), growth_controller(growth_cfg), config(train_cfg)
    {
        ring0::Loss::clear_history();
        optimizer.init(net.layers);

        ring1::AdamWConfig adaptive_config;
        adaptive_config.lr = config.learning_rate;
        adaptive_config.weight_decay = config.weight_decay;
        adaptive_config.enable_multi_formula = config.enable_multi_formula_opt;
        adaptive_config.max_step = 0.20f;
        adaptive_optimizer = ring1::AdamW(adaptive_config);
        register_network_parameters(true);
    }

    void RingTrainer::register_network_parameters(bool reset_optimizer_state)
    {
        if (reset_optimizer_state)
        {
            ring1::AdamWConfig adaptive_config = adaptive_optimizer.config;
            adaptive_optimizer = ring1::AdamW(adaptive_config);
        }

        for (auto &layer : net.layers)
        {
            adaptive_optimizer.register_param(layer.weights, true);
            adaptive_optimizer.register_param(layer.biases, false);
        }
        registered_parameter_count = net.layers.size() * 2;
    }

    float RingTrainer::compute_gradient_norm() const
    {
        float norm_sq = 0.0f;
        for (const auto &layer : net.layers)
        {
            for (float value : layer.grad_weights.data)
                norm_sq += value * value;
            for (float value : layer.grad_biases.data)
                norm_sq += value * value;
        }
        return sqrtf(norm_sq);
    }

    float RingTrainer::compute_gradient_variance() const
    {
        size_t count = 0;
        float sum = 0.0f;
        for (const auto &layer : net.layers)
        {
            for (float value : layer.grad_weights.data)
            {
                sum += value;
                count++;
            }
            for (float value : layer.grad_biases.data)
            {
                sum += value;
                count++;
            }
        }
        if (count == 0)
            return 0.0f;

        float mean = sum / static_cast<float>(count);
        float variance = 0.0f;
        for (const auto &layer : net.layers)
        {
            for (float value : layer.grad_weights.data)
            {
                float delta = value - mean;
                variance += delta * delta;
            }
            for (float value : layer.grad_biases.data)
            {
                float delta = value - mean;
                variance += delta * delta;
            }
        }
        return variance / static_cast<float>(count);
    }

    void RingTrainer::apply_adaptive_update(float loss, float previous_loss)
    {
        loss_history.push_back(loss);
        const float delta_loss = loss - previous_loss;
        if (!ema_initialized)
        {
            ema_loss = loss;
            ema_initialized = true;
        }
        else
        {
            ema_loss = 0.05f * loss + 0.95f * ema_loss;
        }

        ring0::Loss::record_loss(loss);
        adaptive_optimizer.update_penalization_derivative(loss, delta_loss);
        ring0::TaylorTrajectory forecast;
        if (config.enable_taylor_forecast)
            forecast = loss_forecaster.observe(loss_history);

        ring1::MetaOptimizationOutput meta_output;
        if (config.enable_meta_loss_opt)
        {
            ring1::MetaLossTelemetry telemetry;
            telemetry.current_loss = loss;
            telemetry.delta_loss = delta_loss;
            telemetry.accel_loss = loss_history.size() > 2
                                       ? delta_loss - (loss_history[loss_history.size() - 2] - loss_history[loss_history.size() - 3])
                                       : 0.0f;
            telemetry.d_loss_d_penalty = adaptive_optimizer.ema_d_loss_d_penalty;
            telemetry.layer_alignment = adaptive_optimizer.penalty_factor;
            telemetry.gradient_variance = compute_gradient_variance();
            telemetry.learning_rate = adaptive_optimizer.get_learning_rate();
            telemetry.predicted_delta = forecast.valid ? forecast.pred_delta[0] : 0.0f;
            telemetry.predicted_net = forecast.valid ? forecast.predicted[forecast.horizon - 1] - loss : 0.0f;
            telemetry.trajectory_reward = forecast.reward;
            telemetry.trajectory_confidence = forecast.confidence;
            meta_output = meta_loss_optimizer.predict(telemetry);
        }

        float gradient_norm = compute_gradient_norm();
        float clip_scale = 1.0f;
        if (config.max_grad_norm > 0.0f && gradient_norm > config.max_grad_norm)
            clip_scale = config.max_grad_norm / gradient_norm;
        float loss_scale = config.enable_meta_loss_opt ? meta_output.loss_scale_multiplier : 1.0f;
        float gradient_scale = clip_scale * loss_scale;

        adaptive_optimizer.config.enable_multi_formula = config.enable_multi_formula_opt;
        float taylor_weight = std::clamp(config.taylor_forecast_weight, 0.0f, 1.0f);
        float taylor_lr_scale = forecast.valid ? forecast.lr_foresight_scale : 1.0f;
        float taylor_curvature_scale = forecast.valid ? forecast.curvature_foresight : 1.0f;
        float meta_lr_scale = config.enable_meta_loss_opt ? meta_output.lr_step_modulator : 1.0f;
        float lr_scale = meta_lr_scale * ((1.0f - taylor_weight) + taylor_weight * taylor_lr_scale);
        float curvature_scale = (config.enable_meta_loss_opt ? meta_output.curvature_scale : 1.0f) *
                                ((1.0f - taylor_weight) + taylor_weight * taylor_curvature_scale);
        adaptive_optimizer.config.curvature_scale = std::clamp(curvature_scale, 0.5f, 1.5f);
        adaptive_optimizer.set_learning_rate(config.learning_rate * std::clamp(lr_scale, 0.5f, 1.5f));
        last_taylor_lr_scale = taylor_lr_scale;
        last_taylor_curvature_scale = taylor_curvature_scale;
        last_meta_lr_scale = meta_lr_scale;

        size_t parameter_index = 0;
        for (auto &layer : net.layers)
        {
            ring0::Matrix weight_gradient = layer.grad_weights;
            ring0::Matrix bias_gradient = layer.grad_biases;
            for (float &value : weight_gradient.data)
                value *= gradient_scale;
            for (float &value : bias_gradient.data)
                value *= gradient_scale;

            adaptive_optimizer.update_param(parameter_index++, layer.weights, weight_gradient);
            adaptive_optimizer.update_param(parameter_index++, layer.biases, bias_gradient);
        }
        adaptive_optimizer.timestep++;
        adaptive_optimizer.update_attribution_feedback(delta_loss);
        adaptive_optimizer.self_adjust_by_loss(loss, ema_loss, ring0::Loss::get_min_loss());
        // The legacy self-adjuster can propose transformer-scale momentum values;
        // dense recognition keeps AdamW beta parameters in their valid range.
        adaptive_optimizer.config.beta1 = std::clamp(adaptive_optimizer.config.beta1, 0.50f, 0.99f);
        adaptive_optimizer.config.beta2 = std::clamp(adaptive_optimizer.config.beta2, 0.90f, 0.999f);

        if (config.enable_meta_loss_opt)
        {
            if (forecast.valid)
                meta_loss_optimizer.update_online(loss, forecast.reward, 0.5f);
            else
                meta_loss_optimizer.update_online(loss);
        }
    }

    // Executes mini-batch training loop with epoch shuffling and adaptive growth monitoring
    void RingTrainer::train(const ring0::Matrix &X,
                            const ring0::Matrix &Y,
                            const function<void(const EpochMetrics &)> &on_epoch_end)
    {

        size_t num_samples = X.rows;
        vector<size_t> indices(num_samples);
        iota(indices.begin(), indices.end(), 0);
        mt19937 rng(42);

        for (size_t epoch = 1; epoch <= config.epochs; ++epoch)
        {
            // Shuffle dataset indices each epoch
            shuffle(indices.begin(), indices.end(), rng);

            float epoch_loss = 0.0f;
            size_t num_batches = (num_samples + config.batch_size - 1) / config.batch_size;

            for (size_t b = 0; b < num_batches; ++b)
            {
                size_t start_idx = b * config.batch_size;
                size_t end_idx = min(start_idx + config.batch_size, num_samples);
                size_t current_batch_sz = end_idx - start_idx;

                ring0::Matrix batch_X(current_batch_sz, X.cols);
                ring0::Matrix batch_Y(current_batch_sz, Y.cols);

                for (size_t r = 0; r < current_batch_sz; ++r)
                {
                    size_t src_idx = indices[start_idx + r];
                    for (size_t c = 0; c < X.cols; ++c)
                    {
                        batch_X(r, c) = X(src_idx, c);
                    }
                    for (size_t c = 0; c < Y.cols; ++c)
                    {
                        batch_Y(r, c) = Y(src_idx, c);
                    }
                }

                // Forward Pass
                ring0::Matrix preds = net.forward(batch_X);

                // Compute Loss
                float b_loss = ring0::Loss::compute(config.loss_type, preds, batch_Y);
                epoch_loss += b_loss * current_batch_sz;

                // Compute Gradients
                ring0::Matrix grad_output = ring0::Loss::gradient(config.loss_type, preds, batch_Y);

                // Backward Pass & Adaptive AdamW/Meta-Loss Parameter Update
                net.backward(grad_output);
                apply_adaptive_update(b_loss, loss_history.empty() ? b_loss : loss_history.back());
            }

            epoch_loss /= static_cast<float>(num_samples);

            // Record loss in ring0::Loss history & adapt learning rate by (currentLoss - minLoss) * 0.6
            ring0::Loss::record_loss(epoch_loss);
            float min_loss = ring0::Loss::get_min_loss();
            optimizer.adjust_by_loss_gap(epoch_loss, min_loss, 0.2f);

            // Record loss in Growth Controller and check for structural layer expansion
            ring2::GrowthReport report;
            if (config.enable_growth_controller)
            {
                report = growth_controller.record_epoch_loss(epoch_loss, net);
                if (report.action == ring2::GrowthAction::StructuralExpansionTriggered)
                {
                    // Re-register tensors after structural changes so AdamW moments match shapes.
                    optimizer.sync_with_layers(net.layers);
                    register_network_parameters(true);
                }
            }

            float accuracy = evaluate_accuracy(X, Y);

            EpochMetrics metrics{
                epoch,
                epoch_loss,
                accuracy,
                growth_controller.get_growth_rate(),
                report};

            if (on_epoch_end)
            {
                on_epoch_end(metrics);
            }
        }
    }

    // Computes fraction of correctly classified examples
    float RingTrainer::evaluate_accuracy(const ring0::Matrix &X, const ring0::Matrix &Y)
    {
        ring0::Matrix preds = net.forward(X);
        size_t correct = 0;

        for (size_t r = 0; r < preds.rows; ++r)
        {
            size_t best_pred = 0;
            float max_val = preds(r, 0);
            for (size_t c = 1; c < preds.cols; ++c)
            {
                if (preds(r, c) > max_val)
                {
                    max_val = preds(r, c);
                    best_pred = c;
                }
            }

            size_t true_label = 0;
            float true_max = Y(r, 0);
            for (size_t c = 1; c < Y.cols; ++c)
            {
                if (Y(r, c) > true_max)
                {
                    true_max = Y(r, c);
                    true_label = c;
                }
            }

            if (best_pred == true_label)
            {
                correct++;
            }
        }

        return static_cast<float>(correct) / static_cast<float>(preds.rows);
    }

    // Predicts discrete class label for single input
    size_t RingTrainer::predict_class(const ring0::Matrix &single_input)
    {
        ring0::Matrix out = net.forward(single_input);
        size_t best = 0;
        float max_v = out(0, 0);
        for (size_t c = 1; c < out.cols; ++c)
        {
            if (out(0, c) > max_v)
            {
                max_v = out(0, c);
                best = c;
            }
        }
        return best;
    }

} // namespace ring4
