#include "ring3/mnist_dataset.hpp"
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <algorithm>

using namespace std;

namespace ring3 {

// Reads 32-bit big-endian integer from binary stream
static uint32_t read_uint32_be(ifstream& file) {
    uint8_t bytes[4];
    file.read(reinterpret_cast<char*>(bytes), 4);
    if (!file) {
        throw runtime_error("Unexpected end of file while reading IDX header");
    }
    return (static_cast<uint32_t>(bytes[0]) << 24) |
           (static_cast<uint32_t>(bytes[1]) << 16) |
           (static_cast<uint32_t>(bytes[2]) << 8) |
           (static_cast<uint32_t>(bytes[3]));
}

// Parses IDX3 image file into normalized float matrix (N x 784)
ring0::Matrix MnistDataset::load_images(const string& filepath, size_t max_samples) {
    ifstream file(filepath, ios::binary);
    if (!file.is_open()) {
        throw runtime_error("Failed to open MNIST images file: " + filepath);
    }

    uint32_t magic = read_uint32_be(file);
    if (magic != 2051) {
        throw runtime_error("Invalid MNIST image file magic number: " + to_string(magic) + " (expected 2051)");
    }

    uint32_t num_images = read_uint32_be(file);
    uint32_t rows = read_uint32_be(file);
    uint32_t cols = read_uint32_be(file);

    if (rows != IMAGE_ROWS || cols != IMAGE_COLS) {
        throw runtime_error("Unexpected image dimensions: " + to_string(rows) + "x" + to_string(cols));
    }

    size_t count = num_images;
    if (max_samples > 0 && max_samples < count) {
        count = max_samples;
    }

    size_t pixels_per_image = rows * cols;
    ring0::Matrix matrix(count, pixels_per_image);

    vector<uint8_t> buffer(count * pixels_per_image);
    file.read(reinterpret_cast<char*>(buffer.data()), buffer.size());
    if (!file && file.gcount() < static_cast<streamsize>(buffer.size())) {
        throw runtime_error("Failed to read expected number of image bytes");
    }

    for (size_t i = 0; i < buffer.size(); ++i) {
        matrix.data[i] = static_cast<float>(buffer[i]) / 255.0f;
    }

    return matrix;
}

// Parses IDX1 label file into one-hot float matrix (N x 10)
ring0::Matrix MnistDataset::load_labels(const string& filepath, size_t max_samples) {
    ifstream file(filepath, ios::binary);
    if (!file.is_open()) {
        throw runtime_error("Failed to open MNIST labels file: " + filepath);
    }

    uint32_t magic = read_uint32_be(file);
    if (magic != 2049) {
        throw runtime_error("Invalid MNIST label file magic number: " + to_string(magic) + " (expected 2049)");
    }

    uint32_t num_labels = read_uint32_be(file);
    size_t count = num_labels;
    if (max_samples > 0 && max_samples < count) {
        count = max_samples;
    }

    ring0::Matrix matrix = ring0::Matrix::zeros(count, NUM_CLASSES);

    vector<uint8_t> buffer(count);
    file.read(reinterpret_cast<char*>(buffer.data()), buffer.size());
    if (!file && file.gcount() < static_cast<streamsize>(buffer.size())) {
        throw runtime_error("Failed to read expected number of label bytes");
    }

    for (size_t i = 0; i < count; ++i) {
        uint8_t lbl = buffer[i];
        if (lbl < NUM_CLASSES) {
            matrix(i, lbl) = 1.0f;
        }
    }

    return matrix;
}

// Loads both images and labels pair
pair<ring0::Matrix, ring0::Matrix> MnistDataset::load_dataset(
    const string& images_path,
    const string& labels_path,
    size_t max_samples) {

    ring0::Matrix X = load_images(images_path, max_samples);
    ring0::Matrix Y = load_labels(labels_path, max_samples);

    if (X.rows != Y.rows) {
        throw runtime_error("Mismatch between image count (" + to_string(X.rows) +
                            ") and label count (" + to_string(Y.rows) + ")");
    }

    return {X, Y};
}

// Formats 28x28 normalized grayscale pixels as high-contrast ASCII art
string MnistDataset::to_ascii_art(const ring0::Matrix& m, size_t row_idx, float threshold) {
    ostringstream ss;
    for (size_t r = 0; r < IMAGE_ROWS; ++r) {
        for (size_t c = 0; c < IMAGE_COLS; ++c) {
            float val = m(row_idx, r * IMAGE_COLS + c);
            if (val >= 0.75f) {
                ss << "@@";
            } else if (val >= threshold) {
                ss << "##";
            } else if (val >= threshold * 0.5f) {
                ss << "..";
            } else {
                ss << "  ";
            }
        }
        ss << "\n";
    }
    return ss.str();
}

} // namespace ring3
