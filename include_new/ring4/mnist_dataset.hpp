#pragma once

/**
 * @file mnist_dataset.hpp
 * @brief IDX binary parser for the standard MNIST handwritten digits dataset in Ring 4.
 */

#include "ring0/tensor.hpp"
#include <string>
#include <vector>
#include <cstdint>

using namespace std;

namespace ring4 {

/**
 * @class MnistDataset
 * @brief Reads big-endian IDX3 image files and IDX1 label files into normalized float Matrices.
 */
class MnistDataset {
public:
    static constexpr size_t IMAGE_ROWS = 28;                       ///< Image height
    static constexpr size_t IMAGE_COLS = 28;                       ///< Image width
    static constexpr size_t INPUT_DIM = IMAGE_ROWS * IMAGE_COLS;   ///< 784 total input pixels
    static constexpr size_t NUM_CLASSES = 10;                      ///< 10 digit classes (0-9)

    /// Loads binary IDX3 images, normalizing pixel bytes [0, 255] to float [0.0, 1.0]
    static ring0::Matrix load_images(const string& filepath, size_t max_samples = 0);

    /// Loads binary IDX1 labels into one-hot encoded matrix (N x 10)
    static ring0::Matrix load_labels(const string& filepath, size_t max_samples = 0);

    /// Loads both image and label files as an (X, Y) pair
    static pair<ring0::Matrix, ring0::Matrix> load_dataset(
        const string& images_path,
        const string& labels_path,
        size_t max_samples = 0
    );

    /// Renders 28x28 grayscale sample in ASCII characters
    static string to_ascii_art(const ring0::Matrix& m, size_t row_idx, float threshold = 0.3f);
};

} // namespace ring4
