#pragma once

/**
 * @file tensor.hpp
 * @brief Core 2D and 3D tensor math operations, RoPE, FlashAttention kernels, and INT8 Quantization for Ring 0.
 */

#include <vector>
#include <iostream>
#include <functional>
#include <string>
#include <cmath>
#include <cstdint>

using namespace std;

namespace ring0 {

class Matrix;

/**
 * @struct QuantizedMatrix
 * @brief INT8 Symmetric Quantized Matrix representation for 4x memory compression and cache efficiency.
 */
struct QuantizedMatrix {
    size_t rows = 0;
    size_t cols = 0;
    vector<int8_t> qdata;  ///< Quantized weights in [-127, 127]
    vector<float> scales;  ///< Per-row scale factors (scale = max(|row|) / 127.0)

    /// Quantizes full 32-bit float matrix to INT8
    static QuantizedMatrix quantize(const Matrix& m);

    /// Dequantizes INT8 matrix back to 32-bit float matrix
    Matrix dequantize() const;

    /// Fast quantized matrix multiplication: C = Activation (float) * QuantizedWeights (int8)
    Matrix matmul_activation(const Matrix& activation) const;
};

/**
 * @class Matrix
 * @brief 2D Matrix supporting continuous memory buffers, linear algebra, and layer operations.
 */
class Matrix {
public:
    size_t rows;        ///< Number of rows in the matrix
    size_t cols;        ///< Number of columns in the matrix
    vector<float> data; ///< Flat buffer storing matrix elements in row-major order

    // --- Constructors ---
    Matrix();                                                ///< Default constructor (0x0 matrix)
    Matrix(size_t r, size_t c, float init_val = 0.0f);       ///< Matrix filled with a constant value
    Matrix(size_t r, size_t c, const vector<float>& d);      ///< Matrix initialized from flat vector data

    // --- Element Access Operators ---
    float& at(size_t r, size_t c);             ///< Access element with bounds check
    const float& at(size_t r, size_t c) const;

    float& operator()(size_t r, size_t c);     ///< Fast access operator (r, c)
    const float& operator()(size_t r, size_t c) const;

    // --- Factory Initializers ---
    static Matrix zeros(size_t r, size_t c);   ///< All-zeros matrix
    static Matrix ones(size_t r, size_t c);    ///< All-ones matrix

    /// Uniform random numbers in [min_val, max_val]
    static Matrix random_uniform(size_t r, size_t c, float min_val = -1.0f, float max_val = 1.0f);

    /// Gaussian random numbers sampled from N(mean, stddev)
    static Matrix random_normal(size_t r, size_t c, float mean = 0.0f, float stddev = 1.0f);

    /// Xavier initialization: limit = sqrt(6 / (in_dim + out_dim))
    static Matrix xavier(size_t in_dim, size_t out_dim);

    /// He (Kaiming) initialization: stddev = sqrt(2 / in_dim)
    static Matrix he(size_t in_dim, size_t out_dim);

    // --- Linear Algebra ---
    Matrix transpose() const;                   ///< Transposes matrix (r x c) -> (c x r)
    Matrix matmul(const Matrix& other) const;   ///< Matrix multiplication C = A * B
    void matmul_into(const Matrix& other, Matrix& out) const; ///< Zero-alloc multiplication into pre-sized out
    void matmul_transB_into(const Matrix& other_T, Matrix& out) const; ///< Zero-alloc multiply with pre-transposed B
    Matrix elementwise_mul(const Matrix& other) const; ///< Hadamard product C = A .* B

    // --- Arithmetic Operators ---
    Matrix operator+(const Matrix& other) const;
    Matrix operator-(const Matrix& other) const;
    Matrix operator*(float scalar) const;
    Matrix operator/(float scalar) const;

    Matrix& operator+=(const Matrix& other);
    Matrix& operator-=(const Matrix& other);
    Matrix& operator*=(float scalar);

    // --- Broadcasting & Reductions ---
    Matrix add_bias(const Matrix& bias_row_or_col) const; ///< Broadcasts (1 x cols) bias to all rows
    Matrix sum_rows() const;                              ///< Sums across rows -> (1 x cols)
    Matrix sum_cols() const;                              ///< Sums across columns -> (rows x 1)

    // --- Functional Mapping & Normalization ---
    Matrix map(const function<float(float)>& func) const; ///< Element-wise function mapping
    void map_inplace(const function<float(float)>& func); ///< In-place element-wise mapping

    /**
     * @brief Normalizes each row to zero mean and unit variance, then scales and shifts.
     */
    Matrix layer_norm(const Matrix& gamma, const Matrix& beta, float eps = 1e-5f) const;

    /**
     * @brief Computes numerically stabilized row-wise Softmax probabilities.
     */
    Matrix softmax_rows() const;

    /**
     * @brief Computes Root Mean Square Layer Normalization (RMSNorm) for each row.
     */
    Matrix rms_norm(const Matrix& gamma, float eps = 1e-5f) const;

    /**
     * @brief Analytical backward pass for RMSNorm.
     */
    Matrix rms_norm_backward(const Matrix& dY, const Matrix& gamma, Matrix& grad_gamma, float eps = 1e-5f) const;

    /**
     * @brief Applies Rotary Position Embeddings (RoPE) in-place to multi-head vectors.
     * Rotates adjacent 2D feature coordinates by position angle theta_k = pos * base^(-2k/head_dim).
     */
    void apply_rope(size_t num_heads, size_t head_dim, size_t pos_offset = 0, float base_freq = 10000.0f);

    // --- Numerical Stability & NaN Prevention ---
    bool has_nan_or_inf() const;
    void sanitize_nan_inf(float replace_val = 0.0f, float clamp_min = -50.0f, float clamp_max = 50.0f);

    // --- Dynamic Resizing (For Growth Controller) ---
    void expand_rows(size_t new_rows, float init_val = 0.0f);
    void expand_cols(size_t new_cols, float init_val = 0.0f);
    void expand(size_t new_rows, size_t new_cols, float init_val = 0.0f);

    void print(const string& name = "") const; ///< Prints matrix shape and values
};

/**
 * @class Tensor3D
 * @brief 3D Tensor container representing (Batch x Sequence Length x Embedding Channels).
 */
class Tensor3D {
public:
    size_t batch_size; ///< Batch dimension (B)
    size_t seq_len;    ///< Sequence length in tokens (T)
    size_t channels;   ///< Embedding / Channel dimension (C)
    vector<float> data; ///< Flat buffer of size B * T * C

    Tensor3D();
    Tensor3D(size_t b, size_t s, size_t c, float init_val = 0.0f);

    float& at(size_t b, size_t s, size_t c);             ///< Access element [b, s, c]
    const float& at(size_t b, size_t s, size_t c) const;

    float& operator()(size_t b, size_t s, size_t c);     ///< Index operator [b, s, c]
    const float& operator()(size_t b, size_t s, size_t c) const;

    bool has_nan_or_inf() const;
    void sanitize_nan_inf(float replace_val = 0.0f, float clamp_min = -50.0f, float clamp_max = 50.0f);

    Matrix to_matrix() const;                                       ///< Flattens (B, T, C) -> (B*T x C)
    static Tensor3D from_matrix(const Matrix& m, size_t b, size_t s);///< Reshapes (B*T x C) -> (B, T, C)

    Matrix get_batch_matrix(size_t b) const;            ///< Extract 2D matrix (T x C) for single batch item
    void set_batch_matrix(size_t b, const Matrix& m);   ///< Write 2D matrix (T x C) into single batch item

    Tensor3D operator+(const Tensor3D& other) const;    ///< Residual element-wise addition
    Tensor3D& operator+=(const Tensor3D& other);
};

} // namespace ring0
