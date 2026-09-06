#include "ring4/image_dataset.hpp"
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <algorithm>

using namespace std;

namespace ring4 {

// Canonical labels for Fashion-MNIST physical objects
const char* const ImageDataset::CLASS_NAMES[10] = {
    "T-shirt/Top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle Boot"
};

// Big-endian 32-bit integer parser
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
ring0::Matrix ImageDataset::load_images(const string& filepath, size_t max_samples) {
    ifstream file(filepath, ios::binary);
    if (!file.is_open()) {
        throw runtime_error("Failed to open image file: " + filepath);
    }

    uint32_t magic = read_uint32_be(file);
    if (magic != 2051) {
        throw runtime_error("Invalid image IDX magic: " + to_string(magic));
    }

    uint32_t num_images = read_uint32_be(file);
    uint32_t rows = read_uint32_be(file);
    uint32_t cols = read_uint32_be(file);

    if (rows != IMAGE_ROWS || cols != IMAGE_COLS) {
        throw runtime_error("Image dimension mismatch: " + to_string(rows) + "x" + to_string(cols));
    }

    size_t count = num_images;
    if (max_samples > 0 && max_samples < count) {
        count = max_samples;
    }

    size_t pixels = rows * cols;
    ring0::Matrix matrix(count, pixels);

    vector<uint8_t> buffer(count * pixels);
    file.read(reinterpret_cast<char*>(buffer.data()), buffer.size());

    for (size_t i = 0; i < buffer.size(); ++i) {
        matrix.data[i] = static_cast<float>(buffer[i]) / 255.0f;
    }

    return matrix;
}

// Parses IDX1 label file into one-hot float matrix (N x 10)
ring0::Matrix ImageDataset::load_labels(const string& filepath, size_t max_samples) {
    ifstream file(filepath, ios::binary);
    if (!file.is_open()) {
        throw runtime_error("Failed to open label file: " + filepath);
    }

    uint32_t magic = read_uint32_be(file);
    if (magic != 2049) {
        throw runtime_error("Invalid label IDX magic: " + to_string(magic));
    }

    uint32_t num_labels = read_uint32_be(file);
    size_t count = num_labels;
    if (max_samples > 0 && max_samples < count) {
        count = max_samples;
    }

    ring0::Matrix matrix = ring0::Matrix::zeros(count, NUM_CLASSES);
    vector<uint8_t> buffer(count);
    file.read(reinterpret_cast<char*>(buffer.data()), buffer.size());

    for (size_t i = 0; i < count; ++i) {
        uint8_t lbl = buffer[i];
        if (lbl < NUM_CLASSES) {
            matrix(i, lbl) = 1.0f;
        }
    }

    return matrix;
}

// Loads both image and label tensors
pair<ring0::Matrix, ring0::Matrix> ImageDataset::load_dataset(
    const string& images_path,
    const string& labels_path,
    size_t max_samples) {

    ring0::Matrix X = load_images(images_path, max_samples);
    ring0::Matrix Y = load_labels(labels_path, max_samples);
    return {X, Y};
}

// Formats 28x28 normalized grayscale pixels as high-contrast ASCII art
string ImageDataset::to_ascii_art(const ring0::Matrix& m, size_t row_idx, float threshold) {
    ostringstream ss;
    for (size_t r = 0; r < IMAGE_ROWS; ++r) {
        for (size_t c = 0; c < IMAGE_COLS; ++c) {
            float val = m(row_idx, r * IMAGE_COLS + c);
            if (val >= 0.75f) {
                ss << "@@";
            } else if (val >= threshold) {
                ss << "##";
            } else if (val >= threshold * 0.4f) {
                ss << "::";
            } else {
                ss << "  ";
            }
        }
        ss << "\n";
    }
    return ss.str();
}

// Returns label name
string ImageDataset::get_class_name(size_t class_idx) {
    if (class_idx < NUM_CLASSES) {
        return CLASS_NAMES[class_idx];
    }
    return "Unknown";
}

} // namespace ring4
