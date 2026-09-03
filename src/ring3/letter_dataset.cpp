#include "ring3/letter_dataset.hpp"
#include <cstdint>
#include <random>
#include <sstream>
#include <cmath>
#include <algorithm>

using namespace std;

namespace ring3 {

// 5x7 Font Bitmaps (5 bits per row: bits 4..0 representing columns 0..4)
const uint8_t LetterDataset::BITMAPS_5X7[26][7] = {
    // A
    { 0b01110, 0b10001, 0b10001, 0b11111, 0b10001, 0b10001, 0b10001 },
    // B
    { 0b11110, 0b10001, 0b10001, 0b11110, 0b10001, 0b10001, 0b11110 },
    // C
    { 0b01111, 0b10000, 0b10000, 0b10000, 0b10000, 0b10000, 0b01111 },
    // D
    { 0b11110, 0b10001, 0b10001, 0b10001, 0b10001, 0b10001, 0b11110 },
    // E
    { 0b11111, 0b10000, 0b10000, 0b11110, 0b10000, 0b10000, 0b11111 },
    // F
    { 0b11111, 0b10000, 0b10000, 0b11110, 0b10000, 0b10000, 0b10000 },
    // G
    { 0b01111, 0b10000, 0b10000, 0b10111, 0b10001, 0b10001, 0b01111 },
    // H
    { 0b10001, 0b10001, 0b10001, 0b11111, 0b10001, 0b10001, 0b10001 },
    // I
    { 0b11111, 0b00100, 0b00100, 0b00100, 0b00100, 0b00100, 0b11111 },
    // J
    { 0b00111, 0b00010, 0b00010, 0b00010, 0b00010, 0b10010, 0b01100 },
    // K
    { 0b10001, 0b10010, 0b10100, 0b11000, 0b10100, 0b10010, 0b10001 },
    // L
    { 0b10000, 0b10000, 0b10000, 0b10000, 0b10000, 0b10000, 0b11111 },
    // M
    { 0b10001, 0b11011, 0b10101, 0b10001, 0b10001, 0b10001, 0b10001 },
    // N
    { 0b10001, 0b11001, 0b10101, 0b10011, 0b10001, 0b10001, 0b10001 },
    // O
    { 0b01110, 0b10001, 0b10001, 0b10001, 0b10001, 0b10001, 0b01110 },
    // P
    { 0b11110, 0b10001, 0b10001, 0b11110, 0b10000, 0b10000, 0b10000 },
    // Q
    { 0b01110, 0b10001, 0b10001, 0b10001, 0b10101, 0b10010, 0b01101 },
    // R
    { 0b11110, 0b10001, 0b10001, 0b11110, 0b10100, 0b10010, 0b10001 },
    // S
    { 0b01111, 0b10000, 0b10000, 0b01110, 0b00001, 0b00001, 0b11110 },
    // T
    { 0b11111, 0b00100, 0b00100, 0b00100, 0b00100, 0b00100, 0b00100 },
    // U
    { 0b10001, 0b10001, 0b10001, 0b10001, 0b10001, 0b10001, 0b01110 },
    // V
    { 0b10001, 0b10001, 0b10001, 0b10001, 0b10001, 0b01010, 0b00100 },
    // W
    { 0b10001, 0b10001, 0b10001, 0b10101, 0b10101, 0b11011, 0b10001 },
    // X
    { 0b10001, 0b10001, 0b01010, 0b00100, 0b01010, 0b10001, 0b10001 },
    // Y
    { 0b10001, 0b10001, 0b01010, 0b00100, 0b00100, 0b00100, 0b00100 },
    // Z
    { 0b11111, 0b00001, 0b00010, 0b00100, 0b01000, 0b10000, 0b11111 }
};

// Extracts 35 binary floats from a letter's 5x7 bitmap
static vector<float> extract_clean_vector(size_t class_idx) {
    vector<float> vec(LetterDataset::INPUT_DIM, 0.0f);
    for (size_t r = 0; r < LetterDataset::GRID_HEIGHT; ++r) {
        uint8_t row_bits = LetterDataset::BITMAPS_5X7[class_idx][r];
        for (size_t c = 0; c < LetterDataset::GRID_WIDTH; ++c) {
            bool set = (row_bits & (1 << (4 - c))) != 0;
            vec[r * LetterDataset::GRID_WIDTH + c] = set ? 1.0f : 0.0f;
        }
    }
    return vec;
}

// Builds the clean dataset of 26 letters (one-hot targets)
pair<ring0::Matrix, ring0::Matrix> LetterDataset::get_clean_dataset() {
    ring0::Matrix X(NUM_CLASSES, INPUT_DIM);
    ring0::Matrix Y = ring0::Matrix::zeros(NUM_CLASSES, NUM_CLASSES);

    for (size_t i = 0; i < NUM_CLASSES; ++i) {
        vector<float> vec = extract_clean_vector(i);
        for (size_t j = 0; j < INPUT_DIM; ++j) {
            X(i, j) = vec[j];
        }
        Y(i, i) = 1.0f;
    }

    return {X, Y};
}

// Generates augmented letters with flipped bits and Gaussian perturbations
pair<ring0::Matrix, ring0::Matrix> LetterDataset::generate_augmented_dataset(
    size_t samples_per_class,
    float noise_prob,
    float gaussian_stddev) {

    size_t total_samples = NUM_CLASSES * samples_per_class;
    ring0::Matrix X(total_samples, INPUT_DIM);
    ring0::Matrix Y = ring0::Matrix::zeros(total_samples, NUM_CLASSES);

    mt19937 rng(1337);
    uniform_real_distribution<float> prob_dist(0.0f, 1.0f);
    normal_distribution<float> gauss_dist(0.0f, gaussian_stddev);

    size_t sample_idx = 0;
    for (size_t c = 0; c < NUM_CLASSES; ++c) {
        vector<float> clean_vec = extract_clean_vector(c);

        for (size_t s = 0; s < samples_per_class; ++s) {
            for (size_t j = 0; j < INPUT_DIM; ++j) {
                float val = clean_vec[j];
                // Salt and pepper noise
                if (s > 0 && prob_dist(rng) < noise_prob) {
                    val = (val > 0.5f) ? 0.0f : 1.0f;
                }
                // Gaussian noise
                if (s > 0) {
                    val += gauss_dist(rng);
                }
                X(sample_idx, j) = clamp(val, 0.0f, 1.0f);
            }
            Y(sample_idx, c) = 1.0f;
            sample_idx++;
        }
    }

    return {X, Y};
}

// Formats 35 float pixel values as a 5x7 ASCII character grid
string LetterDataset::to_ascii_art(const vector<float>& vec, float threshold) {
    ostringstream ss;
    for (size_t r = 0; r < GRID_HEIGHT; ++r) {
        for (size_t c = 0; c < GRID_WIDTH; ++c) {
            float val = vec[r * GRID_WIDTH + c];
            if (val >= threshold) {
                ss << "##";
            } else if (val >= threshold * 0.5f) {
                ss << "::";
            } else {
                ss << "  ";
            }
        }
        ss << "\n";
    }
    return ss.str();
}

string LetterDataset::to_ascii_art(const ring0::Matrix& m, size_t row_idx, float threshold) {
    vector<float> vec(INPUT_DIM);
    for (size_t c = 0; c < INPUT_DIM; ++c) {
        vec[c] = m(row_idx, c);
    }
    return to_ascii_art(vec, threshold);
}

char LetterDataset::get_char(size_t class_idx) {
    if (class_idx < 26) {
        return static_cast<char>('A' + class_idx);
    }
    return '?';
}

} // namespace ring3
