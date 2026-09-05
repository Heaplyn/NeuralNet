#include "ring0/tensor.hpp"
#include <random>
#include <cmath>
#include <iomanip>
#include <stdexcept>
#include <algorithm>
#if defined(__x86_64__) || defined(_M_X64)
#include <immintrin.h>
#endif

using namespace std;

namespace ring0 {

// Global deterministic RNG engine for reproducible tensor initializations
static mt19937& get_rng() {
    static mt19937 rng(42);
    return rng;
}

// Default constructor: empty 0x0 matrix
Matrix::Matrix() : rows(0), cols(0) {}

// Sized constructor: allocates rows * cols initialized to init_val
Matrix::Matrix(size_t r, size_t c, float init_val)
    : rows(r), cols(c), data(r * c, init_val) {}

// Sized constructor with existing flat data vector
Matrix::Matrix(size_t r, size_t c, const vector<float>& d)
    : rows(r), cols(c), data(d) {
    if (d.size() != r * c) {
        throw invalid_argument("Matrix dimension mismatch in data initialization");
    }
}

// Element access with bounds checking
float& Matrix::at(size_t r, size_t c) {
    return data[r * cols + c];
}

const float& Matrix::at(size_t r, size_t c) const {
    return data[r * cols + c];
}

// Element access operator (r, c)
float& Matrix::operator()(size_t r, size_t c) {
    return at(r, c);
}

const float& Matrix::operator()(size_t r, size_t c) const {
    return at(r, c);
}

// Factory: Zero-filled matrix
Matrix Matrix::zeros(size_t r, size_t c) {
    return Matrix(r, c, 0.0f);
}

// Factory: Ones-filled matrix
Matrix Matrix::ones(size_t r, size_t c) {
    return Matrix(r, c, 1.0f);
}

// Factory: Uniformly distributed random numbers in [min_val, max_val]
Matrix Matrix::random_uniform(size_t r, size_t c, float min_val, float max_val) {
    Matrix m(r, c);
    uniform_real_distribution<float> dist(min_val, max_val);
    for (auto& val : m.data) {
        val = dist(get_rng());
    }
    return m;
}

// Factory: Normally distributed random numbers N(mean, stddev)
Matrix Matrix::random_normal(size_t r, size_t c, float mean, float stddev) {
    Matrix m(r, c);
    normal_distribution<float> dist(mean, stddev);
    for (auto& val : m.data) {
        val = dist(get_rng());
    }
    return m;
}

// Factory: Xavier uniform initialization for Sigmoid/Tanh/Linear layers
Matrix Matrix::xavier(size_t in_dim, size_t out_dim) {
    float limit = sqrt(6.0f / static_cast<float>(in_dim + out_dim));
    return random_uniform(in_dim, out_dim, -limit, limit);
}

// Factory: He normal initialization for ReLU/GELU layers
Matrix Matrix::he(size_t in_dim, size_t out_dim) {
    float stddev = sqrt(1.0f / static_cast<float>(in_dim));
    return random_normal(in_dim, out_dim, 0.0f, stddev);
}

// Transpose: Swaps rows and columns: res(j, i) = at(i, j)
Matrix Matrix::transpose() const {
    Matrix res(cols, rows);
    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            res(j, i) = at(i, j);
        }
    }
    return res;
}

// Matrix multiplication: out = A * B_T where B_T is already transposed (out_cols x K)
void Matrix::matmul_transB_into(const Matrix& other_T, Matrix& out) const {
    if (cols != other_T.cols) {
        throw invalid_argument("Shape mismatch in matmul_transB: cols (" + to_string(cols) + 
                               ") != other_T.cols (" + to_string(other_T.cols) + ")");
    }
    if (out.rows != rows || out.cols != other_T.rows) {
        out = Matrix(rows, other_T.rows, 0.0f);
    }

    size_t out_cols = other_T.rows;
    size_t K = cols;

    #pragma omp parallel for schedule(static) if(rows * out_cols > 256)
    for (int i = 0; i < static_cast<int>(rows); ++i) {
        const float* a_row = &data[static_cast<size_t>(i) * K];
        float* res_row = &out.data[static_cast<size_t>(i) * out_cols];
        for (size_t j = 0; j < out_cols; ++j) {
            const float* b_row = &other_T.data[j * K];
            float dot = 0.0f;
            size_t k = 0;

#if defined(__AVX2__)
            __m256 acc = _mm256_setzero_ps();
            for (; k + 8 <= K; k += 8) {
                __m256 a_vec = _mm256_loadu_ps(a_row + k);
                __m256 b_vec = _mm256_loadu_ps(b_row + k);
#if defined(__FMA__)
                acc = _mm256_fmadd_ps(a_vec, b_vec, acc);
#else
                acc = _mm256_add_ps(acc, _mm256_mul_ps(a_vec, b_vec));
#endif
            }
            __m128 lo = _mm256_castps256_ps128(acc);
            __m128 hi = _mm256_extractf128_ps(acc, 1);
            __m128 sum4 = _mm_add_ps(lo, hi);
            sum4 = _mm_hadd_ps(sum4, sum4);
            sum4 = _mm_hadd_ps(sum4, sum4);
            dot = _mm_cvtss_f32(sum4);
#endif

            for (; k < K; ++k) {
                dot += a_row[k] * b_row[k];
            }
            res_row[j] = dot;
        }
    }
}

// Matrix multiplication: out = A * B using blocked cache-tiling
void Matrix::matmul_into(const Matrix& other, Matrix& out) const {
    if (cols != other.rows) {
        throw invalid_argument("Cannot multiply matrices: shape mismatch (" +
                               to_string(rows) + "x" + to_string(cols) + ") and (" +
                               to_string(other.rows) + "x" + to_string(other.cols) + ")");
    }

    if (out.rows != rows || out.cols != other.cols) {
        out = Matrix(rows, other.cols, 0.0f);
    } else {
        fill(out.data.begin(), out.data.end(), 0.0f);
    }

    size_t M = rows;
    size_t K = cols;
    size_t N = other.cols;

    // Cache-blocked GEMM: keep B rows contiguous in cache
    const size_t BLOCK_I = 64;
    const size_t BLOCK_K = 64;

    #pragma omp parallel for schedule(dynamic, 1) if(M * N > 512)
    for (int bi = 0; bi < static_cast<int>(M); bi += BLOCK_I) {
        size_t i_end = min(M, static_cast<size_t>(bi + BLOCK_I));
        for (size_t bk = 0; bk < K; bk += BLOCK_K) {
            size_t k_end = min(K, bk + BLOCK_K);

            for (size_t i = static_cast<size_t>(bi); i < i_end; ++i) {
                float* out_row = &out.data[i * N];
                const float* a_row = &data[i * K];

                for (size_t k = bk; k < k_end; ++k) {
                    float a_val = a_row[k];
                    if (fabsf(a_val) < 1e-12f) continue;

                    const float* b_row = &other.data[k * N];
                    size_t j = 0;

#if defined(__AVX2__)
                    __m256 a_vec = _mm256_set1_ps(a_val);
                    for (; j + 8 <= N; j += 8) {
                        __m256 b_vec = _mm256_loadu_ps(b_row + j);
                        __m256 c_vec = _mm256_loadu_ps(out_row + j);
#if defined(__FMA__)
                        c_vec = _mm256_fmadd_ps(a_vec, b_vec, c_vec);
#else
                        c_vec = _mm256_add_ps(c_vec, _mm256_mul_ps(a_vec, b_vec));
#endif
                        _mm256_storeu_ps(out_row + j, c_vec);
                    }
#endif
                    for (; j < N; ++j) {
                        out_row[j] += a_val * b_row[j];
                    }
                }
            }
        }
    }
}

// Matrix multiplication: C = A * B
Matrix Matrix::matmul(const Matrix& other) const {
    Matrix res(rows, other.cols);
    matmul_into(other, res);
    return res;
}

// Element-wise multiplication (Hadamard product)
Matrix Matrix::elementwise_mul(const Matrix& other) const {
    if (rows != other.rows || cols != other.cols) {
        throw invalid_argument("Elementwise multiplication dimension mismatch");
    }
    Matrix res(rows, cols);
    for (size_t i = 0; i < data.size(); ++i) {
        res.data[i] = data[i] * other.data[i];
    }
    return res;
}

// Element-wise addition
Matrix Matrix::operator+(const Matrix& other) const {
    if (rows != other.rows || cols != other.cols) {
        string msg = "Matrix addition dimension mismatch: (" +
                     to_string(rows) + "x" + to_string(cols) + ") vs (" +
                     to_string(other.rows) + "x" + to_string(other.cols) + ")";
        throw invalid_argument(msg);
    }
    Matrix res(rows, cols);
    for (size_t i = 0; i < data.size(); ++i) {
        res.data[i] = data[i] + other.data[i];
    }
    return res;
}

// Element-wise subtraction
Matrix Matrix::operator-(const Matrix& other) const {
    if (rows != other.rows || cols != other.cols) {
        string msg = "Matrix subtraction dimension mismatch: (" +
                     to_string(rows) + "x" + to_string(cols) + ") vs (" +
                     to_string(other.rows) + "x" + to_string(other.cols) + ")";
        throw invalid_argument(msg);
    }
    Matrix res(rows, cols);
    for (size_t i = 0; i < data.size(); ++i) {
        res.data[i] = data[i] - other.data[i];
    }
    return res;
}

// Scalar multiplication
Matrix Matrix::operator*(float scalar) const {
    Matrix res(rows, cols);
    for (size_t i = 0; i < data.size(); ++i) {
        res.data[i] = data[i] * scalar;
    }
    return res;
}

// Scalar division
Matrix Matrix::operator/(float scalar) const {
    Matrix res(rows, cols);
    for (size_t i = 0; i < data.size(); ++i) {
        res.data[i] = data[i] / scalar;
    }
    return res;
}

// In-place addition
Matrix& Matrix::operator+=(const Matrix& other) {
    if (rows != other.rows || cols != other.cols) {
        string msg = "Matrix in-place addition dimension mismatch: (" +
                     to_string(rows) + "x" + to_string(cols) + ") vs (" +
                     to_string(other.rows) + "x" + to_string(other.cols) + ")";
        throw invalid_argument(msg);
    }
    for (size_t i = 0; i < data.size(); ++i) {
        data[i] += other.data[i];
    }
    return *this;
}

// In-place subtraction
Matrix& Matrix::operator-=(const Matrix& other) {
    if (rows != other.rows || cols != other.cols) {
        throw invalid_argument("Matrix in-place subtraction dimension mismatch");
    }
    for (size_t i = 0; i < data.size(); ++i) {
        data[i] -= other.data[i];
    }
    return *this;
}

// In-place scalar multiplication
Matrix& Matrix::operator*=(float scalar) {
    for (size_t i = 0; i < data.size(); ++i) {
        data[i] *= scalar;
    }
    return *this;
}

// Broadcast addition: adds (1 x cols) bias row vector to each of the matrix's rows
Matrix Matrix::add_bias(const Matrix& bias) const {
    if (bias.cols != cols) {
        throw invalid_argument("Bias columns must match matrix columns");
    }
    Matrix res(rows, cols);
    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            res(i, j) = at(i, j) + bias(0, j);
        }
    }
    return res;
}

// Column reduction: Sums across rows to produce (1 x cols) row vector
Matrix Matrix::sum_rows() const {
    Matrix res(1, cols, 0.0f);
    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            res(0, j) += at(i, j);
        }
    }
    return res;
}

// Row reduction: Sums across columns to produce (rows x 1) column vector
Matrix Matrix::sum_cols() const {
    Matrix res(rows, 1, 0.0f);
    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            res(i, 0) += at(i, j);
        }
    }
    return res;
}

// Frobenius / L2 Euclidean norm squared
float Matrix::norm_squared() const {
    float sum_sq = 0.0f;
    for (float v : data) {
        if (!std::isnan(v) && !std::isinf(v)) {
            sum_sq += v * v;
        }
    }
    return sum_sq;
}

// Frobenius / L2 Euclidean norm
float Matrix::norm() const {
    return std::sqrt(norm_squared());
}

// Applies unary lambda/function to each element
Matrix Matrix::map(const function<float(float)>& func) const {
    Matrix res(rows, cols);
    for (size_t i = 0; i < data.size(); ++i) {
        res.data[i] = func(data[i]);
    }
    return res;
}

// Applies unary lambda/function to each element in-place
void Matrix::map_inplace(const function<float(float)>& func) {
    for (size_t i = 0; i < data.size(); ++i) {
        data[i] = func(data[i]);
    }
}

// Layer Normalization: Computes row mean & variance, normalizes, and scales with gamma and beta
Matrix Matrix::layer_norm(const Matrix& gamma, const Matrix& beta, float eps) const {
    Matrix res(rows, cols);
    for (size_t r = 0; r < rows; ++r) {
        // Compute mean of current row
        float mean = 0.0f;
        for (size_t c = 0; c < cols; ++c) {
            mean += at(r, c);
        }
        mean /= static_cast<float>(cols);

        // Compute variance of current row
        float var = 0.0f;
        for (size_t c = 0; c < cols; ++c) {
            float diff = at(r, c) - mean;
            var += diff * diff;
        }
        var /= static_cast<float>(cols);
        float stddev_inv = 1.0f / sqrt(var + eps);

        // Scale and shift: y = ((x - mean) * stddev_inv) * gamma + beta
        for (size_t c = 0; c < cols; ++c) {
            float norm_val = (at(r, c) - mean) * stddev_inv;
            float g = (gamma.cols == cols) ? gamma(0, c) : 1.0f;
            float b = (beta.cols == cols) ? beta(0, c) : 0.0f;
            res(r, c) = norm_val * g + b;
        }
    }
    return res;
}

// Layer Normalization: Computes Root Mean Square Layer Normalization (RMSNorm) with NaN guards
Matrix Matrix::rms_norm(const Matrix& gamma, float eps) const {
    Matrix res(rows, cols);
    for (size_t r = 0; r < rows; ++r) {
        // Step 1: Sum raw squares: x_i^2
        float sum_sq = 0.0f;
        for (size_t c = 0; c < cols; ++c) {
            float val = at(r, c);
            if (!std::isnan(val) && !std::isinf(val)) {
                sum_sq += val * val;
            }
        }
        float mean_sq = sum_sq / static_cast<float>(cols);
        if (std::isnan(mean_sq) || std::isinf(mean_sq) || mean_sq < 0.0f) {
            mean_sq = 0.0f;
        }

        // Step 2: Reciprocal of Root-Mean-Square
        float rms_inv = 1.0f / sqrt(mean_sq + eps);
        if (std::isnan(rms_inv) || std::isinf(rms_inv)) rms_inv = 1.0f;

        // Step 3: Scale raw x by rms_inv and gamma
        for (size_t c = 0; c < cols; ++c) {
            float g = (gamma.cols == cols) ? gamma(0, c) : 1.0f;
            if (std::isnan(g) || std::isinf(g)) g = 1.0f;
            float val = at(r, c);
            if (std::isnan(val) || std::isinf(val)) val = 0.0f;
            res(r, c) = (val * rms_inv) * g;
        }
    }
    return res;
}

// Analytical backward pass for RMSNorm with NaN guards
Matrix Matrix::rms_norm_backward(const Matrix& dY, const Matrix& gamma, Matrix& grad_gamma, float eps) const {
    Matrix dX(rows, cols);
    float C_float = static_cast<float>(cols);

    for (size_t r = 0; r < rows; ++r) {
        // 1. Recompute RMS statistics
        float sum_sq = 0.0f;
        for (size_t c = 0; c < cols; ++c) {
            float val = at(r, c);
            if (!std::isnan(val) && !std::isinf(val)) {
                sum_sq += val * val;
            }
        }
        float mean_sq = sum_sq / C_float;
        if (std::isnan(mean_sq) || std::isinf(mean_sq) || mean_sq < 0.0f) mean_sq = 0.0f;
        float rms = sqrt(mean_sq + eps);
        float inv_rms = (rms > 1e-12f) ? (1.0f / rms) : 1.0f;
        float inv_rms3 = inv_rms * inv_rms * inv_rms;

        // 2. Accumulate grad_gamma and compute inner dot product
        float dot_dy_gamma_x = 0.0f;
        for (size_t c = 0; c < cols; ++c) {
            float x_val = at(r, c);
            float g_val = (gamma.cols == cols) ? gamma(0, c) : 1.0f;
            float dy_val = dY(r, c);
            if (std::isnan(x_val) || std::isinf(x_val)) x_val = 0.0f;
            if (std::isnan(g_val) || std::isinf(g_val)) g_val = 1.0f;
            if (std::isnan(dy_val) || std::isinf(dy_val)) dy_val = 0.0f;

            // grad_gamma += dY * (x / rms)
            grad_gamma(0, c) += dy_val * (x_val * inv_rms);
            dot_dy_gamma_x += dy_val * g_val * x_val;
        }

        // 3. Compute dX: dL/dx = (gamma / rms) * dY - (x / (C * rms^3)) * sum(dY * gamma * x)
        for (size_t c = 0; c < cols; ++c) {
            float x_val = at(r, c);
            float g_val = (gamma.cols == cols) ? gamma(0, c) : 1.0f;
            float dy_val = dY(r, c);
            if (std::isnan(x_val) || std::isinf(x_val)) x_val = 0.0f;
            if (std::isnan(g_val) || std::isinf(g_val)) g_val = 1.0f;
            if (std::isnan(dy_val) || std::isinf(dy_val)) dy_val = 0.0f;

            float dx_val = (g_val * inv_rms) * dy_val - (x_val / C_float) * inv_rms3 * dot_dy_gamma_x;
            if (std::isnan(dx_val) || std::isinf(dx_val)) dx_val = 0.0f;
            dX(r, c) = dx_val;
        }
    }

    return dX;
}

// Numerically stabilized Softmax along rows (subtracting max logit per row to prevent overflow)
Matrix Matrix::softmax_rows() const {
    Matrix res(rows, cols);
    for (size_t r = 0; r < rows; ++r) {
        float max_val = -1e30f;
        for (size_t c = 0; c < cols; ++c) {
            float val = at(r, c);
            if (!std::isnan(val) && !std::isinf(val)) {
                if (val > max_val) max_val = val;
            }
        }
        if (max_val < -1e20f) max_val = 0.0f;

        float sum = 0.0f;
        for (size_t c = 0; c < cols; ++c) {
            float val = at(r, c);
            float diff = (std::isnan(val) || std::isinf(val)) ? -50.0f : (val - max_val);
            if (diff < -50.0f) diff = -50.0f;
            if (diff > 50.0f) diff = 50.0f;
            float exp_val = expf(diff);
            res(r, c) = exp_val;
            sum += exp_val;
        }

        float inv_sum = (sum > 1e-12f && !std::isnan(sum)) ? (1.0f / sum) : (1.0f / static_cast<float>(cols));
        for (size_t c = 0; c < cols; ++c) {
            float p = res(r, c) * inv_sum;
            if (std::isnan(p) || std::isinf(p) || p < 0.0f) {
                p = 1.0f / static_cast<float>(cols);
            }
            res(r, c) = p;
        }
    }
    return res;
}

bool Matrix::has_nan_or_inf() const {
    for (float v : data) {
        if (std::isnan(v) || std::isinf(v)) return true;
    }
    return false;
}

void Matrix::sanitize_nan_inf(float replace_val, float clamp_min, float clamp_max) {
    for (float& v : data) {
        if (std::isnan(v) || std::isinf(v)) {
            v = replace_val;
        } else {
            v = std::clamp(v, clamp_min, clamp_max);
        }
    }
}

bool Tensor3D::has_nan_or_inf() const {
    for (float v : data) {
        if (std::isnan(v) || std::isinf(v)) return true;
    }
    return false;
}

void Tensor3D::sanitize_nan_inf(float replace_val, float clamp_min, float clamp_max) {
    for (float& v : data) {
        if (std::isnan(v) || std::isinf(v)) {
            v = replace_val;
        } else {
            v = std::clamp(v, clamp_min, clamp_max);
        }
    }
}

// Expands row count, initializing new elements to init_val
void Matrix::expand_rows(size_t new_rows, float init_val) {
    if (new_rows <= rows) return;
    data.resize(new_rows * cols, init_val);
    rows = new_rows;
}

// Expands column count, preserving row alignment
void Matrix::expand_cols(size_t new_cols, float init_val) {
    if (new_cols <= cols) return;
    vector<float> new_data(rows * new_cols, init_val);
    for (size_t r = 0; r < rows; ++r) {
        for (size_t c = 0; c < cols; ++c) {
            new_data[r * new_cols + c] = at(r, c);
        }
    }
    data = move(new_data);
    cols = new_cols;
}

// Expands both rows and columns
void Matrix::expand(size_t new_rows, size_t new_cols, float init_val) {
    if (new_rows <= rows && new_cols <= cols) return;
    size_t target_r = max(rows, new_rows);
    size_t target_c = max(cols, new_cols);
    vector<float> new_data(target_r * target_c, init_val);
    for (size_t r = 0; r < rows; ++r) {
        for (size_t c = 0; c < cols; ++c) {
            new_data[r * target_c + c] = at(r, c);
        }
    }
    rows = target_r;
    cols = target_c;
    data = move(new_data);
}

// Prints formatted matrix elements
void Matrix::print(const string& name) const {
    if (!name.empty()) {
        cout << name << " (" << rows << "x" << cols << "):\n";
    }
    for (size_t i = 0; i < rows; ++i) {
        cout << "  [";
        for (size_t j = 0; j < cols; ++j) {
            cout << fixed << setprecision(4) << at(i, j);
            if (j + 1 < cols) cout << ", ";
        }
        cout << "]\n";
    }
}

// -------------------------------------------------------------
// Tensor3D Implementation
// -------------------------------------------------------------

Tensor3D::Tensor3D() : batch_size(0), seq_len(0), channels(0) {}

Tensor3D::Tensor3D(size_t b, size_t s, size_t c, float init_val)
    : batch_size(b), seq_len(s), channels(c), data(b * s * c, init_val) {}

// Element indexing: offset = ((b * seq_len + s) * channels) + c
float& Tensor3D::at(size_t b, size_t s, size_t c) {
    return data[(b * seq_len + s) * channels + c];
}

const float& Tensor3D::at(size_t b, size_t s, size_t c) const {
    return data[(b * seq_len + s) * channels + c];
}

float& Tensor3D::operator()(size_t b, size_t s, size_t c) {
    return at(b, s, c);
}

const float& Tensor3D::operator()(size_t b, size_t s, size_t c) const {
    return at(b, s, c);
}

// Flattens (B, T, C) into 2D Matrix of shape (B*T, C) for linear matrix multiplies
Matrix Tensor3D::to_matrix() const {
    Matrix m(batch_size * seq_len, channels);
    m.data = data;
    return m;
}

// Reshapes 2D Matrix of shape (B*T, C) back into 3D Tensor (B, T, C)
Tensor3D Tensor3D::from_matrix(const Matrix& m, size_t b, size_t s) {
    Tensor3D t(b, s, m.cols);
    t.data = m.data;
    return t;
}

// Extracts slice for single batch item b (shape: T x C)
Matrix Tensor3D::get_batch_matrix(size_t b) const {
    Matrix m(seq_len, channels);
    size_t offset = b * seq_len * channels;
    for (size_t i = 0; i < seq_len * channels; ++i) {
        m.data[i] = data[offset + i];
    }
    return m;
}

// Writes 2D matrix (T x C) into batch item b
void Tensor3D::set_batch_matrix(size_t b, const Matrix& m) {
    size_t offset = b * seq_len * channels;
    for (size_t i = 0; i < seq_len * channels; ++i) {
        data[offset + i] = m.data[i];
    }
}

// Residual addition: adds two 3D tensors element-wise
Tensor3D Tensor3D::operator+(const Tensor3D& other) const {
    Tensor3D res(batch_size, seq_len, channels);
    for (size_t i = 0; i < data.size(); ++i) {
        res.data[i] = data[i] + other.data[i];
    }
    return res;
}

// Applies Rotary Position Embeddings (RoPE) in-place to multi-head vectors
void Matrix::apply_rope(size_t num_heads, size_t head_dim, size_t pos_offset, float base_freq) {
    if (head_dim == 0 || num_heads == 0 || cols < num_heads * head_dim) return;

    for (size_t r = 0; r < rows; ++r) {
        float pos = static_cast<float>(pos_offset + r);

        for (size_t h = 0; h < num_heads; ++h) {
            size_t head_start = r * cols + h * head_dim;

            for (size_t k = 0; k < head_dim / 2; ++k) {
                float freq = pow(base_freq, -static_cast<float>(2 * k) / static_cast<float>(head_dim));
                float theta = pos * freq;
                float cos_theta = cos(theta);
                float sin_theta = sin(theta);

                size_t idx0 = head_start + 2 * k;
                size_t idx1 = head_start + 2 * k + 1;

                float x0 = data[idx0];
                float x1 = data[idx1];

                data[idx0] = x0 * cos_theta - x1 * sin_theta;
                data[idx1] = x0 * sin_theta + x1 * cos_theta;
            }
        }
    }
}

// Quantizes 32-bit float matrix to INT8 with per-row scaling
QuantizedMatrix QuantizedMatrix::quantize(const Matrix& m) {
    QuantizedMatrix q;
    q.rows = m.rows;
    q.cols = m.cols;
    q.qdata.resize(m.rows * m.cols);
    q.scales.resize(m.rows);

    for (size_t r = 0; r < m.rows; ++r) {
        float max_abs = 0.0f;
        for (size_t c = 0; c < m.cols; ++c) {
            float abs_val = fabsf(m.at(r, c));
            if (abs_val > max_abs) max_abs = abs_val;
        }

        float scale = max(1e-8f, max_abs / 127.0f);
        q.scales[r] = scale;
        float inv_scale = 1.0f / scale;

        for (size_t c = 0; c < m.cols; ++c) {
            float val = m.at(r, c) * inv_scale;
            val = max(-127.0f, min(127.0f, roundf(val)));
            q.qdata[r * m.cols + c] = static_cast<int8_t>(val);
        }
    }
    return q;
}

// Dequantizes INT8 matrix back to 32-bit float matrix
Matrix QuantizedMatrix::dequantize() const {
    Matrix m(rows, cols);
    for (size_t r = 0; r < rows; ++r) {
        float scale = scales[r];
        for (size_t c = 0; c < cols; ++c) {
            m(r, c) = static_cast<float>(qdata[r * cols + c]) * scale;
        }
    }
    return m;
}

// Fast quantized matrix multiply: C = Activation (M x K) * QuantizedWeights (K x N)
Matrix QuantizedMatrix::matmul_activation(const Matrix& activation) const {
    size_t M = activation.rows;
    size_t K = activation.cols;
    size_t N = cols;

    Matrix res(M, N, 0.0f);

    #pragma omp parallel for schedule(static) if(M * N > 512)
    for (int i = 0; i < static_cast<int>(M); ++i) {
        const float* act_row = &activation.data[static_cast<size_t>(i) * K];
        float* res_row = &res.data[static_cast<size_t>(i) * N];

        for (size_t j = 0; j < N; ++j) {
            float dot = 0.0f;
            for (size_t k = 0; k < K; ++k) {
                float w_val = static_cast<float>(qdata[k * N + j]) * scales[k];
                dot += act_row[k] * w_val;
            }
            res_row[j] = dot;
        }
    }
    return res;
}

} // namespace ring0
