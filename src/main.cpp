#include "ringwrapper.hpp"
#include <cctype>
#include <chrono>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <string>

using namespace std;

static size_t parse_test_letter(int argc, char *argv[])
{
    for (int i = 1; i + 1 < argc; ++i)
    {
        if (string(argv[i]) == "--letter")
        {
            char requested = static_cast<char>(toupper(static_cast<unsigned char>(argv[i + 1][0])));
            if (requested >= 'A' && requested <= 'Z')
                return static_cast<size_t>(requested - 'A');
        }
    }
    return 0;
}

static ring3::TrainingConfig adaptive_config(size_t epochs, size_t batch_size, float learning_rate)
{
    ring3::TrainingConfig config;
    config.epochs = epochs;
    config.batch_size = batch_size;
    config.learning_rate = learning_rate;
    config.weight_decay = 0.01f;
    config.max_grad_norm = 1.0f;
    config.enable_growth_controller = true;
    config.enable_meta_loss_opt = true;
    config.enable_multi_formula_opt = true;
    return config;
}

static float train_letter_database(int argc, char *argv[])
{
    auto [train_x, train_y] = ring3::LetterDataset::generate_augmented_dataset(40, 0.15f, 0.10f, 1337);
    auto [hard_test_x, hard_test_y] = ring3::LetterDataset::generate_augmented_dataset(10, 0.30f, 0.18f, 7331);

    ring2::NeuralNet model;
    model.add_dense(ring3::LetterDataset::INPUT_DIM, 64, ring0::ActivationType::ReLU);
    model.add_dense(64, ring3::LetterDataset::NUM_CLASSES, ring0::ActivationType::None);

    ring1::GradientDescent compatibility_optimizer;
    ring3::TrainingConfig config = adaptive_config(120, 64, 0.001f);
    ring3::RingTrainer trainer(model, compatibility_optimizer, {}, config);

    auto train_start = chrono::steady_clock::now();
    cout << "[A-Z] Training on 40 samples per letter with moderate noise.\n";
    cout << "[A-Z] Testing on 10 unseen samples per letter with stronger noise.\n";
    trainer.train(train_x, train_y, [&](const ring3::EpochMetrics &metrics)
                  {
                      if (metrics.epoch == 1 || metrics.epoch % 20 == 0)
                      {
                          float accuracy = trainer.evaluate_accuracy(hard_test_x, hard_test_y);
                          cout << "  Epoch " << setw(3) << metrics.epoch
                               << " | Loss: " << fixed << setprecision(4) << metrics.loss
                               << " | Hard A-Z accuracy: " << setprecision(1)
                               << accuracy * 100.0f << "%\n";
                      } });

    float accuracy = trainer.evaluate_accuracy(hard_test_x, hard_test_y);
    auto train_end = chrono::steady_clock::now();
    double elapsed_seconds = chrono::duration<double>(train_end - train_start).count();
    double samples_per_second = static_cast<double>(train_x.rows * config.epochs) /
                                max(0.001, elapsed_seconds);
    size_t tested_letter = parse_test_letter(argc, argv);
    size_t tested_sample = tested_letter * 10 + 5;
    ring0::Matrix letter_input(1, ring3::LetterDataset::INPUT_DIM);
    for (size_t feature = 0; feature < ring3::LetterDataset::INPUT_DIM; ++feature)
        letter_input(0, feature) = hard_test_x(tested_sample, feature);

    size_t guessed_letter = trainer.predict_class(letter_input);
    char expected = ring3::LetterDataset::get_char(tested_letter);
    char guessed = ring3::LetterDataset::get_char(guessed_letter);

    cout << "\n[A-Z] Tested letter: " << expected << " (unseen noisy sample)\n";
    cout << ring3::LetterDataset::to_ascii_art(hard_test_x, tested_sample);
    cout << "[A-Z] Guessed letter: " << guessed << "\n";
    cout << "[A-Z] Match: " << (expected == guessed ? "PASS" : "FAIL") << "\n";
    cout << "[A-Z] Hard accuracy: " << fixed << setprecision(2)
         << accuracy * 100.0f << "% (260 labeled test samples)\n\n";
    cout << "[A-Z] Benchmark: " << fixed << setprecision(2) << elapsed_seconds
         << " s | " << setprecision(1) << samples_per_second
         << " training samples/s | " << model.get_total_parameters() << " parameters\n\n";
    return accuracy;
}

static void train_image_database(const string &dataset_name, const string &dataset_root)
{
    const string train_images = dataset_root + "/train-images.idx3-ubyte";
    const string train_labels = dataset_root + "/train-labels.idx1-ubyte";
    const string test_images = dataset_root + "/t10k-images.idx3-ubyte";
    const string test_labels = dataset_root + "/t10k-labels.idx1-ubyte";

    if (!filesystem::exists(train_images) || !filesystem::exists(train_labels) ||
        !filesystem::exists(test_images) || !filesystem::exists(test_labels))
    {
        cout << "[" << dataset_name << "] Dataset files not found under " << dataset_root
             << "; skipping image benchmark.\n";
        return;
    }

    auto [train_x, train_y] = ring3::MnistDataset::load_dataset(train_images, train_labels, 10000);
    auto [test_x, test_y] = ring3::MnistDataset::load_dataset(test_images, test_labels, 2000);

    ring2::NeuralNet model;
    model.add_dense(ring3::MnistDataset::INPUT_DIM, 128, ring0::ActivationType::ReLU);
    model.add_dense(128, ring3::MnistDataset::NUM_CLASSES, ring0::ActivationType::None);

    ring1::GradientDescent compatibility_optimizer;
    ring3::TrainingConfig config = adaptive_config(8, 64, 0.001f);
    ring3::RingTrainer trainer(model, compatibility_optimizer, {}, config);

    auto train_start = chrono::steady_clock::now();
    cout << "[" << dataset_name << "] Training on " << train_x.rows << " real images and evaluating on "
         << test_x.rows << " held-out images.\n";
    trainer.train(train_x, train_y, [&](const ring3::EpochMetrics &metrics)
                  {
                      if (metrics.epoch == 1 || metrics.epoch % 2 == 0)
                      {
                          cout << "  " << dataset_name << " epoch " << metrics.epoch
                               << " | Loss: " << fixed << setprecision(4) << metrics.loss
                               << " | Held-out accuracy: " << setprecision(1)
                               << trainer.evaluate_accuracy(test_x, test_y) * 100.0f << "%\n";
                      } });

    size_t predicted = trainer.predict_class(test_x);
    size_t expected = 0;
    while (expected + 1 < test_y.cols && test_y(0, expected) < test_y(0, expected + 1))
        ++expected;

    cout << "[" << dataset_name << "] Tested class: " << expected << "\n";
    cout << ring3::MnistDataset::to_ascii_art(test_x, 0);
    cout << "[" << dataset_name << "] Guessed class: " << predicted << "\n";
    cout << "[" << dataset_name << "] Match: " << (expected == predicted ? "PASS" : "FAIL") << "\n";
    cout << "[" << dataset_name << "] Held-out accuracy: " << fixed << setprecision(2)
         << trainer.evaluate_accuracy(test_x, test_y) * 100.0f << "%\n";
    auto train_end = chrono::steady_clock::now();
    double elapsed_seconds = chrono::duration<double>(train_end - train_start).count();
    double samples_per_second = static_cast<double>(train_x.rows * config.epochs) /
                                max(0.001, elapsed_seconds);
    cout << "[" << dataset_name << "] Benchmark: " << fixed << setprecision(2)
         << elapsed_seconds << " s | " << setprecision(1) << samples_per_second
         << " training samples/s | " << model.get_total_parameters() << " parameters\n\n";
}

int main(int argc, char *argv[])
{
    cout << "=========================================================\n";
    cout << "       RINGWRAPPER HARDER RECOGNITION BENCHMARK          \n";
    cout << "=========================================================\n";
    cout << "Adaptive stack: AdamW + Meta-Loss + Taylor + growth control\n\n";
    cout << left << setw(18) << "Dataset" << right << setw(12) << "Train rows"
         << setw(12) << "Test rows" << setw(14) << "Metric" << "\n";
    cout << string(58, '-') << "\n";

    float letter_accuracy = train_letter_database(argc, argv);
    train_image_database("MNIST", "data/mnist");
    train_image_database("Fashion-MNIST", "data/fashion-mnist");

    cout << "\nBenchmark complete.\n";
    return letter_accuracy > 0.0f ? 0 : 1;
}
