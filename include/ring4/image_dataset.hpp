#pragma once

/**
 * @file image_dataset.hpp
 * @brief Physical object / Fashion-MNIST image dataset parser and ASCII visualizer in Ring 4.
 */

#include "ring0/tensor.hpp"
#include <string>
#include <vector>
#include <cstdint>

using namespace std;

namespace ring4 {

/**
 * @class ImageDataset
 * @brief Parses 28x28 physical object images (clothing, shoes, bags) from IDX files.
 */
class ImageDataset {
public:
    static constexpr size_t IMAGE_ROWS = 28;                       ///< Image height
    static constexpr size_t IMAGE_COLS = 28;                       ///< Image width
    static constexpr size_t INPUT_DIM = IMAGE_ROWS * IMAGE_COLS;   ///< 784 total input pixels
    static constexpr size_t NUM_CLASSES = 10;                      ///< 10 clothing categories

    /// Names of all 10 fashion object categories
    static const char* const CLASS_NAMES[10];

    /// Loads binary IDX3 images into float Matrix (N x 784)
    static ring0::Matrix load_images(const string& filepath, size_t max_samples = 0);

    /// Loads binary IDX1 labels into one-hot Matrix (N x 10)
    static ring0::Matrix load_labels(const string& filepath, size_t max_samples = 0);

    /// Loads both images and labels pair
    static pair<ring0::Matrix, ring0::Matrix> load_dataset(
        const string& images_path,
        const string& labels_path,
        size_t max_samples = 0
    );

    /// Renders 28x28 image sample as high-contrast ASCII art
    static string to_ascii_art(const ring0::Matrix& m, size_t row_idx, float threshold = 0.25f);

    /// Returns human-readable class name string (e.g. "Ankle Boot", "Pullover")
    static string get_class_name(size_t class_idx);
};

} // namespace ring4
