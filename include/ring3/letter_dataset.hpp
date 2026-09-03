#pragma once

/**
 * @file letter_dataset.hpp
 * @brief 5x7 bitmap font representations of English alphabet (A-Z) with noise augmentation in Ring 3.
 */

#include "ring0/tensor.hpp"
#include <cstdint>
#include <vector>
#include <string>

using namespace std;

namespace ring3 {

/**
 * @struct LetterSample
 * @brief Represents a single letter sample with 35 float pixel inputs and label index.
 */
struct LetterSample {
    char letter;         ///< Character glyph ('A' through 'Z')
    size_t label;        ///< Integer class index (0 to 25)
    vector<float> data;  ///< 35 float values (5 width x 7 height)
};

/**
 * @class LetterDataset
 * @brief Generates clean and synthetically noisy 5x7 font bitmaps for alphabet classification.
 */
class LetterDataset {
public:
    static constexpr size_t GRID_WIDTH = 5;                        ///< Font width in pixels
    static constexpr size_t GRID_HEIGHT = 7;                       ///< Font height in pixels
    static constexpr size_t INPUT_DIM = GRID_WIDTH * GRID_HEIGHT;  ///< 35 total input pixels
    static constexpr size_t NUM_CLASSES = 26;                      ///< 26 alphabet letters (A-Z)

    /// Binary bitmask representations for all 26 uppercase letters
    static const uint8_t BITMAPS_5X7[26][7];

    /// Returns clean baseline dataset (1 sample per letter)
    static pair<ring0::Matrix, ring0::Matrix> get_clean_dataset();

    /// Generates augmented dataset with random salt-and-pepper and Gaussian noise
    static pair<ring0::Matrix, ring0::Matrix> generate_augmented_dataset(
        size_t samples_per_class = 20,
        float noise_prob = 0.08f,
        float gaussian_stddev = 0.05f
    );

    /// Renders 35-dim vector as ASCII block art
    static string to_ascii_art(const vector<float>& vec, float threshold = 0.5f);
    static string to_ascii_art(const ring0::Matrix& m, size_t row_idx, float threshold = 0.5f);

    /// Maps class index (0..25) to character ('A'..'Z')
    static char get_char(size_t class_idx);
};

} // namespace ring3
