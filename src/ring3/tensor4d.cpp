#include "ring3/tensor4d.hpp"
#include <random>
#include <iomanip>

namespace ring3 {

Tensor4D::Tensor4D(size_t b, size_t c, size_t h, size_t w, float init_val)
    : batch_size(b), channels(c), height(h), width(w), data(b * c * h * w, init_val) {}

Tensor4D::Tensor4D(size_t b, size_t c, size_t h, size_t w, const std::vector<float>& d)
    : batch_size(b), channels(c), height(h), width(w), data(d) {
    if (data.size() != b * c * h * w) {
        data.resize(b * c * h * w, 0.0f);
    }
}

void Tensor4D::fill(float val) {
    std::fill(data.begin(), data.end(), val);
}

Tensor4D Tensor4D::zeros(size_t b, size_t c, size_t h, size_t w) {
    return Tensor4D(b, c, h, w, 0.0f);
}

Tensor4D Tensor4D::ones(size_t b, size_t c, size_t h, size_t w) {
    return Tensor4D(b, c, h, w, 1.0f);
}

Tensor4D Tensor4D::random_normal(size_t b, size_t c, size_t h, size_t w, float mean, float stddev) {
    Tensor4D res(b, c, h, w);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<float> dist(mean, stddev);
    for (float& v : res.data) {
        v = dist(gen);
    }
    return res;
}

Tensor4D Tensor4D::random_uniform(size_t b, size_t c, size_t h, size_t w, float min_val, float max_val) {
    Tensor4D res(b, c, h, w);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist(min_val, max_val);
    for (float& v : res.data) {
        v = dist(gen);
    }
    return res;
}

Tensor4D Tensor4D::he(size_t out_c, size_t in_c, size_t k_h, size_t k_w) {
    float fan_in = static_cast<float>(in_c * k_h * k_w);
    float stddev = std::sqrt(2.0f / std::max(1.0f, fan_in));
    return random_normal(out_c, in_c, k_h, k_w, 0.0f, stddev);
}

Tensor4D Tensor4D::xavier(size_t out_c, size_t in_c, size_t k_h, size_t k_w) {
    float fan_in = static_cast<float>(in_c * k_h * k_w);
    float fan_out = static_cast<float>(out_c * k_h * k_w);
    float limit = std::sqrt(6.0f / std::max(1.0f, fan_in + fan_out));
    return random_uniform(out_c, in_c, k_h, k_w, -limit, limit);
}

Tensor4D Tensor4D::pad(size_t pad_h, size_t pad_w) const {
    if (pad_h == 0 && pad_w == 0) return *this;
    size_t new_h = height + 2 * pad_h;
    size_t new_w = width + 2 * pad_w;
    Tensor4D res(batch_size, channels, new_h, new_w, 0.0f);

    for (size_t b = 0; b < batch_size; ++b) {
        for (size_t c = 0; c < channels; ++c) {
            for (size_t h = 0; h < height; ++h) {
                for (size_t w = 0; w < width; ++w) {
                    res(b, c, h + pad_h, w + pad_w) = (*this)(b, c, h, w);
                }
            }
        }
    }
    return res;
}

ring0::Matrix Tensor4D::to_matrix() const {
    size_t spatial = channels * height * width;
    ring0::Matrix m(batch_size, spatial);
    m.data = data;
    return m;
}

Tensor4D Tensor4D::from_matrix(const ring0::Matrix& m, size_t c, size_t h, size_t w) {
    size_t b = m.rows;
    Tensor4D res(b, c, h, w);
    if (m.data.size() == b * c * h * w) {
        res.data = m.data;
    }
    return res;
}

Tensor4D Tensor4D::operator+(const Tensor4D& other) const {
    Tensor4D res(batch_size, channels, height, width);
    for (size_t i = 0; i < data.size(); ++i) {
        res.data[i] = data[i] + other.data[i];
    }
    return res;
}

Tensor4D Tensor4D::operator-(const Tensor4D& other) const {
    Tensor4D res(batch_size, channels, height, width);
    for (size_t i = 0; i < data.size(); ++i) {
        res.data[i] = data[i] - other.data[i];
    }
    return res;
}

Tensor4D Tensor4D::operator*(float scalar) const {
    Tensor4D res(batch_size, channels, height, width);
    for (size_t i = 0; i < data.size(); ++i) {
        res.data[i] = data[i] * scalar;
    }
    return res;
}

Tensor4D Tensor4D::operator/(float scalar) const {
    Tensor4D res(batch_size, channels, height, width);
    float inv = 1.0f / (scalar + 1e-12f);
    for (size_t i = 0; i < data.size(); ++i) {
        res.data[i] = data[i] * inv;
    }
    return res;
}

Tensor4D& Tensor4D::operator+=(const Tensor4D& other) {
    for (size_t i = 0; i < data.size(); ++i) {
        data[i] += other.data[i];
    }
    return *this;
}

Tensor4D& Tensor4D::operator-=(const Tensor4D& other) {
    for (size_t i = 0; i < data.size(); ++i) {
        data[i] -= other.data[i];
    }
    return *this;
}

Tensor4D& Tensor4D::operator*=(float scalar) {
    for (size_t i = 0; i < data.size(); ++i) {
        data[i] *= scalar;
    }
    return *this;
}

float Tensor4D::norm() const {
    return std::sqrt(norm_squared());
}

float Tensor4D::norm_squared() const {
    float sum = 0.0f;
    for (float v : data) {
        sum += v * v;
    }
    return sum;
}

bool Tensor4D::has_nan_or_inf() const {
    for (float v : data) {
        if (std::isnan(v) || std::isinf(v)) return true;
    }
    return false;
}

void Tensor4D::sanitize_nan_inf(float replace_val, float clamp_min, float clamp_max) {
    for (float& v : data) {
        if (std::isnan(v) || std::isinf(v)) {
            v = replace_val;
        } else {
            v = std::clamp(v, clamp_min, clamp_max);
        }
    }
}

void Tensor4D::print(const std::string& name) const {
    std::cout << "[Tensor4D " << name << " shape=(" << batch_size << ", " << channels 
              << ", " << height << ", " << width << ") norm=" << norm() << "]\n";
}

} // namespace ring3
