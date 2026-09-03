#pragma once

/**
 * @file cuda_backend.hpp
 * @brief CUDA Math Backend & GPU Hardware Acceleration in Ring 0.
 *        Supports dynamic CUDA detection and execution with seamless OpenMP CPU fallback.
 */

#include "ring0/tensor.hpp"
#include <vector>
#include <string>
#include <memory>
#include <iostream>

namespace ring0 {

/**
 * @struct CUDADeviceInfo
 * @brief Telemetry and capabilities of the active GPU device.
 */
struct CUDADeviceInfo {
    bool is_available = false;
    std::string device_name = "None (CPU Fallback)";
    size_t total_memory = 0;
    size_t free_memory = 0;
    int compute_capability_major = 0;
    int compute_capability_minor = 0;
    int multi_processor_count = 0;
};

/**
 * @class CUDAMathEngine
 * @brief Unified GPU Acceleration engine with automatic OpenMP multi-core fallback.
 */
class CUDAMathEngine {
private:
    static bool s_initialized;
    static CUDADeviceInfo s_device_info;
    static bool s_cuda_enabled;

public:
    /// Initializes CUDA backend, detects GPU hardware, or configures CPU fallback
    static void initialize(bool force_cpu = false);

    /// Checks if CUDA GPU hardware acceleration is active
    static bool is_cuda_active();

    /// Returns device telemetry and capability details
    static const CUDADeviceInfo& get_device_info();

    /// Sets whether CUDA execution is enabled (if hardware supports it)
    static void set_cuda_enabled(bool enabled);

    // --- Fused Mathematical Tensor Operations ---

    /**
     * @brief Matrix Multiplication: C = A * B (with optional transpose on B)
     *        Executes on GPU CUDA cores or parallel OpenMP CPU.
     */
    static Matrix matmul(const Matrix& A, const Matrix& B, bool transpose_b = false);

    /**
     * @brief Fused ALiBi Scaled Causal Attention:
     *        Computes Output = Softmax((Q * K^T) / sqrt(d) + ALiBi) * V
     */
    static Matrix fused_causal_attention(
        const Matrix& Q, 
        const Matrix& K, 
        const Matrix& V, 
        float scale, 
        float alibi_slope = 0.0f
    );

    /**
     * @brief Fused SwiGLU Gated Activation Forward Pass:
     *        Output = (X * W_gate * sigmoid(X * W_gate)) * (X * W_up)
     */
    static Matrix fused_swiglu(
        const Matrix& X, 
        const Matrix& W_gate, 
        const Matrix& W_up
    );

    /**
     * @brief Parallel Elementwise Fused Multiply-Add: Out = A * B + C
     */
    static Matrix fused_fma(const Matrix& A, const Matrix& B, const Matrix& C);

    /**
     * @brief Parallel Softmax with numerical stability
     */
    static Matrix softmax(const Matrix& input, int axis = -1);
};

} // namespace ring0
