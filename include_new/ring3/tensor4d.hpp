#pragma once

/**
 * @file tensor4d.hpp
 * @brief High-performance 4D Tensor container for computer vision and CNN feature maps in Ring 3.
 */

#include "ring0/tensor.hpp"
#include <vector>
#include <iostream>
#include <cmath>
#include <string>
#include <functional>

namespace ring3 {

/**
 * @class Tensor4D
 * @brief 4D Tensor representing (Batch x Channels x Height x Width) for convolution activations and gradients.
 */
class Tensor4D {
public:
    size_t batch_size = 0; ///< Batch dimension (B)
    size_t channels = 0;   ///< Channel / Feature map dimension (C)
    size_t height = 0;     ///< Spatial height (H)
    size_t width = 0;      ///< Spatial width (W)
    std::vector<float> data; ///< Flat contiguous buffer of size B * C * H * W

    Tensor4D() = default;
    Tensor4D(size_t b, size_t c, size_t h, size_t w, float init_val = 0.0f);
    Tensor4D(size_t b, size_t c, size_t h, size_t w, const std::vector<float>& d);

    // --- Element Access ---
    inline float& at(size_t b, size_t c, size_t h, size_t w) {
        return data[((b * channels + c) * height + h) * width + w];
    }

    inline const float& at(size_t b, size_t c, size_t h, size_t w) const {
        return data[((b * channels + c) * height + h) * width + w];
    }

    inline float& operator()(size_t b, size_t c, size_t h, size_t w) {
        return data[((b * channels + c) * height + h) * width + w];
    }

    inline const float& operator()(size_t b, size_t c, size_t h, size_t w) const {
        return data[((b * channels + c) * height + h) * width + w];
    }

    size_t size() const { return data.size(); }
    void fill(float val);
    void zero() { fill(0.0f); }

    // --- Initializers ---
    static Tensor4D zeros(size_t b, size_t c, size_t h, size_t w);
    static Tensor4D ones(size_t b, size_t c, size_t h, size_t w);
    static Tensor4D random_normal(size_t b, size_t c, size_t h, size_t w, float mean = 0.0f, float stddev = 1.0f);
    static Tensor4D random_uniform(size_t b, size_t c, size_t h, size_t w, float min_val = -1.0f, float max_val = 1.0f);
    static Tensor4D he(size_t out_c, size_t in_c, size_t k_h, size_t k_w);
    static Tensor4D xavier(size_t out_c, size_t in_c, size_t k_h, size_t k_w);

    // --- Transformations & Conversions ---
    Tensor4D pad(size_t pad_h, size_t pad_w) const;
    ring0::Matrix to_matrix() const; ///< Flattens (B, C, H, W) -> (B, C * H * W)
    static Tensor4D from_matrix(const ring0::Matrix& m, size_t c, size_t h, size_t w);

    // --- Arithmetic & Reductions ---
    Tensor4D operator+(const Tensor4D& other) const;
    Tensor4D operator-(const Tensor4D& other) const;
    Tensor4D operator*(float scalar) const;
    Tensor4D operator/(float scalar) const;

    Tensor4D& operator+=(const Tensor4D& other);
    Tensor4D& operator-=(const Tensor4D& other);
    Tensor4D& operator*=(float scalar);

    float norm() const;
    float norm_squared() const;

    // --- Numerical Stability ---
    bool has_nan_or_inf() const;
    void sanitize_nan_inf(float replace_val = 0.0f, float clamp_min = -50.0f, float clamp_max = 50.0f);

    void print(const std::string& name = "") const;
};

} // namespace ring3
