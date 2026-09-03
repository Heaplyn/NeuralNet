#include "ring3/trainer.hpp"
#include "ring0/loss.hpp"
#include <iostream>
#include <vector>
#include <numeric>
#include <random>
#include <algorithm>

using namespace std;

namespace ring3 {

// Constructor: Initializes trainer, growth controller, and momentum optimizer
RingTrainer::RingTrainer(ring2::NeuralNet& network,
                         ring1::GradientDescent& opt,
                         const ring2::GrowthConfig& growth_cfg,
                         TrainingConfig train_cfg)
    : net(network), optimizer(opt), growth_controller(growth_cfg), config(train_cfg) {
    optimizer.init(net.layers);
}

// Executes mini-batch training loop with epoch shuffling and adaptive growth monitoring
void RingTrainer::train(const ring0::Matrix& X,
                        const ring0::Matrix& Y,
                        const function<void(const EpochMetrics&)>& on_epoch_end) {

    size_t num_samples = X.rows;
    vector<size_t> indices(num_samples);
    iota(indices.begin(), indices.end(), 0);
    mt19937 rng(42);

    for (size_t epoch = 1; epoch <= config.epochs; ++epoch) {
        // Shuffle dataset indices each epoch
        shuffle(indices.begin(), indices.end(), rng);

        float epoch_loss = 0.0f;
        size_t num_batches = (num_samples + config.batch_size - 1) / config.batch_size;

        for (size_t b = 0; b < num_batches; ++b) {
            size_t start_idx = b * config.batch_size;
            size_t end_idx = min(start_idx + config.batch_size, num_samples);
            size_t current_batch_sz = end_idx - start_idx;

            ring0::Matrix batch_X(current_batch_sz, X.cols);
            ring0::Matrix batch_Y(current_batch_sz, Y.cols);

            for (size_t r = 0; r < current_batch_sz; ++r) {
                size_t src_idx = indices[start_idx + r];
                for (size_t c = 0; c < X.cols; ++c) {
                    batch_X(r, c) = X(src_idx, c);
                }
                for (size_t c = 0; c < Y.cols; ++c) {
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

            // Backward Pass & Parameter Update
            net.backward(grad_output);
            optimizer.update(net.layers);
        }

        epoch_loss /= static_cast<float>(num_samples);

        // Record loss in ring0::Loss history & adapt learning rate by (currentLoss - minLoss) * 0.6
        ring0::Loss::record_loss(epoch_loss);
        float min_loss = ring0::Loss::get_min_loss();
        optimizer.adjust_by_loss_gap(epoch_loss, min_loss, 0.2f);

        // Record loss in Growth Controller and check for structural layer expansion
        ring2::GrowthReport report;
        if (config.enable_growth_controller) {
            report = growth_controller.record_epoch_loss(epoch_loss, net);
            if (report.action == ring2::GrowthAction::StructuralExpansionTriggered) {
                // Synchronize optimizer momentum buffers if layers expanded
                optimizer.sync_with_layers(net.layers);
            }
        }

        float accuracy = evaluate_accuracy(X, Y);

        EpochMetrics metrics{
            epoch,
            epoch_loss,
            accuracy,
            growth_controller.get_growth_rate(),
            report
        };

        if (on_epoch_end) {
            on_epoch_end(metrics);
        }
    }
}

// Computes fraction of correctly classified examples
float RingTrainer::evaluate_accuracy(const ring0::Matrix& X, const ring0::Matrix& Y) {
    ring0::Matrix preds = net.forward(X);
    size_t correct = 0;

    for (size_t r = 0; r < preds.rows; ++r) {
        size_t best_pred = 0;
        float max_val = preds(r, 0);
        for (size_t c = 1; c < preds.cols; ++c) {
            if (preds(r, c) > max_val) {
                max_val = preds(r, c);
                best_pred = c;
            }
        }

        size_t true_label = 0;
        float true_max = Y(r, 0);
        for (size_t c = 1; c < Y.cols; ++c) {
            if (Y(r, c) > true_max) {
                true_max = Y(r, c);
                true_label = c;
            }
        }

        if (best_pred == true_label) {
            correct++;
        }
    }

    return static_cast<float>(correct) / static_cast<float>(preds.rows);
}

// Predicts discrete class label for single input
size_t RingTrainer::predict_class(const ring0::Matrix& single_input) {
    ring0::Matrix out = net.forward(single_input);
    size_t best = 0;
    float max_v = out(0, 0);
    for (size_t c = 1; c < out.cols; ++c) {
        if (out(0, c) > max_v) {
            max_v = out(0, c);
            best = c;
        }
    }
    return best;
}

} // namespace ring3
