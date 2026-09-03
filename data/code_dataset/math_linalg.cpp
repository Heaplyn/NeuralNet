#include <vector>
#include <cmath>
#include <stdexcept>
#include <complex>

namespace linalg {

// --- 1. 3D Vector Math ---
struct Vec3 {
    float x, y, z;

    Vec3(float x = 0, float y = 0, float z = 0) : x(x), y(y), z(z) {}

    Vec3 operator+(const Vec3& o) const { return {x + o.x, y + o.y, z + o.z}; }
    Vec3 operator-(const Vec3& o) const { return {x - o.x, y - o.y, z - o.z}; }
    Vec3 operator*(float s) const { return {x * s, y * s, z * s}; }

    float dot(const Vec3& o) const { return x * o.x + y * o.y + z * o.z; }
    
    Vec3 cross(const Vec3& o) const {
        return {y * o.z - z * o.y, z * o.x - x * o.z, x * o.y - y * o.x};
    }

    float length() const { return std::sqrt(dot(*this)); }

    Vec3 normalized() const {
        float l = length();
        return (l > 1e-6f) ? (*this * (1.0f / l)) : Vec3(0, 0, 0);
    }
};

// --- 2. Quaternion Rotation Math ---
struct Quaternion {
    float w, x, y, z;

    Quaternion(float w = 1, float x = 0, float y = 0, float z = 0) : w(w), x(x), y(y), z(z) {}

    static Quaternion from_axis_angle(const Vec3& axis, float angle_rad) {
        float half = angle_rad * 0.5f;
        float s = std::sin(half);
        Vec3 norm = axis.normalized();
        return {std::cos(half), norm.x * s, norm.y * s, norm.z * s};
    }

    Quaternion operator*(const Quaternion& q) const {
        return {
            w * q.w - x * q.x - y * q.y - z * q.z,
            w * q.x + x * q.w + y * q.z - z * q.y,
            w * q.y - x * q.z + y * q.w + z * q.x,
            w * q.z + x * q.y - y * q.x + z * q.w
        };
    }

    Vec3 rotate(const Vec3& v) const {
        Quaternion p(0, v.x, v.y, v.z);
        Quaternion q_conj(w, -x, -y, -z);
        Quaternion res = (*this) * p * q_conj;
        return {res.x, res.y, res.z};
    }
};

// --- 3. Fast Fourier Transform (Cooley-Tukey FFT) ---
using Complex = std::complex<double>;

void fft(std::vector<Complex>& a, bool invert) {
    size_t n = a.size();
    if (n <= 1) return;

    std::vector<Complex> a0(n / 2), a1(n / 2);
    for (size_t i = 0; 2 * i < n; ++i) {
        a0[i] = a[2 * i];
        a1[i] = a[2 * i + 1];
    }
    fft(a0, invert);
    fft(a1, invert);

    double ang = 2 * 3.14159265358979323846 / n * (invert ? -1 : 1);
    Complex w(1), wn(std::cos(ang), std::sin(ang));
    for (size_t i = 0; 2 * i < n; ++i) {
        a[i] = a0[i] + w * a1[i];
        a[i + n / 2] = a0[i] - w * a1[i];
        if (invert) {
            a[i] /= 2;
            a[i + n / 2] /= 2;
        }
        w *= wn;
    }
}

// --- 4. Matrix Inversion using Gauss-Jordan Elimination ---
std::vector<float> invert_matrix(const std::vector<float>& mat, size_t n) {
    std::vector<std::vector<float>> aug(n, std::vector<float>(2 * n, 0.0f));
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < n; ++j) {
            aug[i][j] = mat[i * n + j];
        }
        aug[i][i + n] = 1.0f;
    }

    for (size_t i = 0; i < n; ++i) {
        size_t pivot = i;
        for (size_t r = i + 1; r < n; ++r) {
            if (std::abs(aug[r][i]) > std::abs(aug[pivot][i])) pivot = r;
        }
        std::swap(aug[i], aug[pivot]);

        float divisor = aug[i][i];
        if (std::abs(divisor) < 1e-7f) throw std::runtime_error("Singular matrix cannot be inverted");

        for (size_t c = 0; c < 2 * n; ++c) aug[i][c] /= divisor;

        for (size_t r = 0; r < n; ++r) {
            if (r != i) {
                float factor = aug[r][i];
                for (size_t c = 0; c < 2 * n; ++c) {
                    aug[r][c] -= factor * aug[i][c];
                }
            }
        }
    }

    std::vector<float> inv(n * n);
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < n; ++j) {
            inv[i * n + j] = aug[i][j + n];
        }
    }
    return inv;
}

} // namespace linalg
