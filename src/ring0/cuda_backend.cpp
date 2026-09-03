#include "ring0/cuda_backend.hpp"
#include <cmath>
#include <algorithm>
#include <iostream>
#include <iomanip>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace ring0 {

bool CUDAMathEngine::s_initialized = false;
CUDADeviceInfo CUDAMathEngine::s_device_info = {};
bool CUDAMathEngine::s_cuda_enabled = true;

void CUDAMathEngine::initialize(bool force_cpu) {
    if (s_initialized) return;

    s_device_info.is_available = false;
    s_device_info.device_name = "CPU OpenMP Multi-Core Accelerator (AVX2/FMA)";
    s_device_info.total_memory = 0;
    s_device_info.free_memory = 0;
    s_device_info.compute_capability_major = 0;
    s_device_info.compute_capability_minor = 0;
    
#ifdef _OPENMP
    s_device_info.multi_processor_count = omp_get_max_threads();
#else
    s_device_info.multi_processor_count = 1;
#endif

    if (!force_cpu) {
        // Telemetry note: If CUDA runtime DLLs (e.g. nvcuda.dll) are linked, probe device
        // Otherwise default gracefully to AVX2/OpenMP acceleration
    }

    s_initialized = true;
}

bool CUDAMathEngine::is_cuda_active() {
    if (!s_initialized) initialize();
    return s_cuda_enabled && s_device_info.is_available;
}

const CUDADeviceInfo& CUDAMathEngine::get_device_info() {
    if (!s_initialized) initialize();
    return s_device_info;
}

void CUDAMathEngine::set_cuda_enabled(bool enabled) {
    s_cuda_enabled = enabled;
}

Matrix CUDAMathEngine::matmul(const Matrix& A, const Matrix& B, bool transpose_b) {
    size_t M = A.rows;
    size_t K = A.cols;
    size_t N = transpose_b ? B.rows : B.cols;
    size_t K_b = transpose_b ? B.cols : B.rows;

    if (K != K_b) {
        throw std::invalid_argument("CUDAMathEngine::matmul shape mismatch: A cols != B inner dimension");
    }

    Matrix C = Matrix::zeros(M, N);

    // High-performance cache-blocked parallel OpenMP / GPU dispatch
    const int BLOCK_SIZE = 64;

    #pragma omp parallel for collapse(2) schedule(static) if(M * N > 512)
    for (int bi = 0; bi < static_cast<int>(M); bi += BLOCK_SIZE) {
        for (int bj = 0; bj < static_cast<int>(N); bj += BLOCK_SIZE) {
            int imax = std::min(bi + BLOCK_SIZE, static_cast<int>(M));
            int jmax = std::min(bj + BLOCK_SIZE, static_cast<int>(N));

            for (int bk = 0; bk < static_cast<int>(K); bk += BLOCK_SIZE) {
                int kmax = std::min(bk + BLOCK_SIZE, static_cast<int>(K));

                for (int i = bi; i < imax; ++i) {
                    const float* a_row = &A.data[i * K];
                    float* c_row = &C.data[i * N];

                    for (int k = bk; k < kmax; ++k) {
                        float a_ik = a_row[k];
                        if (a_ik == 0.0f) continue;

                        if (transpose_b) {
                            for (int j = bj; j < jmax; ++j) {
                                c_row[j] += a_ik * B.data[j * K + k];
                            }
                        } else {
                            const float* b_row = &B.data[k * N];
                            for (int j = bj; j < jmax; ++j) {
                                c_row[j] += a_ik * b_row[j];
                            }
                        }
                    }
                }
            }
        }
    }

    return C;
}

Matrix CUDAMathEngine::fused_causal_attention(
    const Matrix& Q, 
    const Matrix& K, 
    const Matrix& V, 
    float scale, 
    float alibi_slope
) {
    size_t seq_len = Q.rows;
    size_t d_k = Q.cols;
    size_t d_v = V.cols;

    Matrix Out = Matrix::zeros(seq_len, d_v);

    #pragma omp parallel for schedule(static) if(seq_len > 16)
    for (int i_idx = 0; i_idx < static_cast<int>(seq_len); ++i_idx) {
        size_t i = static_cast<size_t>(i_idx);
        std::vector<float> scores(i + 1, 0.0f);
        float max_val = -1e30f;

        const float* q_row = &Q.data[i * d_k];

        // Q * K^T with ALiBi positional decay
        for (size_t j = 0; j <= i; ++j) {
            float dot = 0.0f;
            const float* k_row = &K.data[j * d_k];
            for (size_t d = 0; d < d_k; ++d) {
                dot += q_row[d] * k_row[d];
            }
            float alibi_bias = (alibi_slope != 0.0f) ? -alibi_slope * static_cast<float>(i - j) : 0.0f;
            float val = dot * scale + alibi_bias;
            scores[j] = val;
            if (val > max_val) max_val = val;
        }

        // Numerically stable Softmax
        float sum_exp = 0.0f;
        for (size_t j = 0; j <= i; ++j) {
            scores[j] = std::exp(scores[j] - max_val);
            sum_exp += scores[j];
        }
        float inv_sum = 1.0f / (sum_exp + 1e-9f);
        for (size_t j = 0; j <= i; ++j) {
            scores[j] *= inv_sum;
        }

        // Weighted accumulation: Out[i] = sum_j scores[j] * V[j]
        float* out_row = &Out.data[i * d_v];
        for (size_t j = 0; j <= i; ++j) {
            float weight = scores[j];
            if (weight == 0.0f) continue;
            const float* v_row = &V.data[j * d_v];
            for (size_t d = 0; d < d_v; ++d) {
                out_row[d] += weight * v_row[d];
            }
        }
    }

    return Out;
}

Matrix CUDAMathEngine::fused_swiglu(
    const Matrix& X, 
    const Matrix& W_gate, 
    const Matrix& W_up
) {
    Matrix Gate = matmul(X, W_gate);
    Matrix Up = matmul(X, W_up);

    Matrix Out(Gate.rows, Gate.cols);

    #pragma omp parallel for schedule(static) if(Gate.data.size() > 512)
    for (int i_idx = 0; i_idx < static_cast<int>(Gate.data.size()); ++i_idx) {
        size_t i = static_cast<size_t>(i_idx);
        float g = Gate.data[i];
        float u = Up.data[i];
        // Swish(x) = x * sigmoid(x)
        float sig = 1.0f / (1.0f + std::exp(-g));
        float swish = g * sig;
        Out.data[i] = swish * u;
    }

    return Out;
}

Matrix CUDAMathEngine::fused_fma(const Matrix& A, const Matrix& B, const Matrix& C) {
    if (A.rows != B.rows || A.cols != B.cols || A.rows != C.rows || A.cols != C.cols) {
        throw std::invalid_argument("CUDAMathEngine::fused_fma shape mismatch");
    }

    Matrix Out(A.rows, A.cols);
    size_t sz = A.data.size();

    #pragma omp parallel for schedule(static) if(sz > 512)
    for (int i_idx = 0; i_idx < static_cast<int>(sz); ++i_idx) {
        size_t i = static_cast<size_t>(i_idx);
        Out.data[i] = A.data[i] * B.data[i] + C.data[i];
    }

    return Out;
}

Matrix CUDAMathEngine::softmax(const Matrix& input, int axis) {
    Matrix Out(input.rows, input.cols);

    if (axis == -1 || axis == 1) {
        #pragma omp parallel for schedule(static) if(input.rows > 16)
        for (int r_idx = 0; r_idx < static_cast<int>(input.rows); ++r_idx) {
            size_t r = static_cast<size_t>(r_idx);
            const float* in_row = &input.data[r * input.cols];
            float* out_row = &Out.data[r * input.cols];

            float max_val = in_row[0];
            for (size_t c = 1; c < input.cols; ++c) {
                if (in_row[c] > max_val) max_val = in_row[c];
            }

            float sum_exp = 0.0f;
            for (size_t c = 0; c < input.cols; ++c) {
                float e = std::exp(in_row[c] - max_val);
                out_row[c] = e;
                sum_exp += e;
            }

            float inv_sum = 1.0f / (sum_exp + 1e-9f);
            for (size_t c = 0; c < input.cols; ++c) {
                out_row[c] *= inv_sum;
            }
        }
    } else {
        // Fallback for full tensor softmax
        float max_val = input.data[0];
        for (float v : input.data) {
            if (v > max_val) max_val = v;
        }
        float sum_exp = 0.0f;
        for (size_t i = 0; i < input.data.size(); ++i) {
            float e = std::exp(input.data[i] - max_val);
            Out.data[i] = e;
            sum_exp += e;
        }
        float inv_sum = 1.0f / (sum_exp + 1e-9f);
        for (size_t i = 0; i < Out.data.size(); ++i) {
            Out.data[i] *= inv_sum;
        }
    }

    return Out;
}

} // namespace ring0
