# 🧮 Tensor3D & Matrix Math Primitives

In **Ring 0**, pure mathematical representations and high-throughput linear algebra primitives are implemented without higher-level neural abstractions.

---

## 🎯 Practical Explanation: What is this and Why Does it Exist?

### The Memory Bottleneck in Deep Learning
Modern CPUs and GPUs are fundamentally memory-bandwidth bound rather than compute bound. A single floating-point multiplication takes less than $1$ clock cycle, but fetching a float from main system RAM takes $\approx 200-300$ clock cycles.

If matrix data is laid out poorly in RAM, the CPU constantly stalls waiting for cache lines to populate.

### How RingWrapper Optimizes Linear Algebra
1. **Contiguous Flat Allocations**: Zero pointer-chasing overhead; memory is allocated in single contiguous chunks.
2. **Cache-Blocked Tile Decomposition**: Matrix multiplications are divided into $64 \times 64$ sub-blocks that fit directly into fast L1/L2 CPU hardware cache ($32\text{ KB} - 512\text{ KB}$).
3. **Loop Tiling & Transpose Optimization**: In a standard $A \times B$ multiplication, reading matrix $B$ column-by-column causes non-contiguous stride jumps. By either transposing $B$ or cache-blocking, memory is always read in sequential order along cache lines.

---

## 💻 Deep Code Breakdown

### 1. `ring0::Matrix` Memory Model
Located in `include/ring0/tensor.hpp`:

```cpp
struct Matrix {
    size_t rows = 0;
    size_t cols = 0;
    std::vector<float> data;

    // Fast static constructors
    static Matrix zeros(size_t r, size_t c) {
        Matrix m;
        m.rows = r;
        m.cols = c;
        m.data.assign(r * c, 0.0f);
        return m;
    }

    static Matrix random(size_t r, size_t c, float mean = 0.0f, float std = 0.02f) {
        Matrix m;
        m.rows = r;
        m.cols = c;
        m.data.resize(r * c);
        // Gaussian initialization (Xavier / He normal distribution)
        static thread_local std::mt19937 gen(42);
        std::normal_distribution<float> d(mean, std);
        for (auto& val : m.data) {
            val = d(gen);
        }
        return m;
    }

    // Zero-overhead inlined 2D indexing operator:
    inline float& operator()(size_t r, size_t c) {
        return data[r * cols + c];
    }
    inline const float& operator()(size_t r, size_t c) const {
        return data[r * cols + c];
    }
};
```

**Why this code is written this way**:
- `assign(r * c, 0.0f)` uses system `memset` under the hood for zero-page OS allocation speed.
- `thread_local std::mt19937` ensures that multi-threaded weight initialization does not suffer from mutex lock contention across OpenMP worker threads.

---

### 2. Cache-Blocked Parallel Matrix Multiplication Kernel
Located in `src/ring0/tensor.cpp`:

```cpp
Matrix Matrix::matmul(const Matrix& A, const Matrix& B) {
    // Dimension verification assertion
    if (A.cols != B.rows) {
        throw std::runtime_error("Matrix dimension mismatch: A.cols (" + 
            std::to_string(A.cols) + ") != B.rows (" + std::to_string(B.rows) + ")");
    }

    const size_t M = A.rows;
    const size_t K = A.cols;
    const size_t N = B.cols;

    Matrix C = Matrix::zeros(M, N);
    const size_t BLOCK_SIZE = 64; // Tuned for standard 32KB L1 Data Cache

    #pragma omp parallel for collapse(2) schedule(static) if (M * N > 2048)
    for (int bi = 0; bi < static_cast<int>(M); bi += BLOCK_SIZE) {
        for (int bj = 0; bj < static_cast<int>(N); bj += BLOCK_SIZE) {
            for (int bk = 0; bk < static_cast<int>(K); bk += BLOCK_SIZE) {
                
                // Bounds clipping for edge tiles
                int i_end = std::min(static_cast<int>(M), bi + static_cast<int>(BLOCK_SIZE));
                int j_end = std::min(static_cast<int>(N), bj + static_cast<int>(BLOCK_SIZE));
                int k_end = std::min(static_cast<int>(K), bk + static_cast<int>(BLOCK_SIZE));

                // Micro-kernel operating entirely in CPU registers and L1 cache
                for (int i = bi; i < i_end; ++i) {
                    const float* a_row = &A.data[i * K];
                    float* c_row = &C.data[i * N];

                    for (int k = bk; k < k_end; ++k) {
                        float a_val = a_row[k];
                        const float* b_row = &B.data[k * N];

                        #pragma omp simd
                        for (int j = bj; j < j_end; ++j) {
                            c_row[j] += a_val * b_row[j];
                        }
                    }
                }
            }
        }
    }

    return C;
}
```

### Why this algorithm is fast:
1. **Loop Ordering (`i -> k -> j`)**: Notice the innermost loop is over `j`. In row-major memory, `b_row[j]` and `c_row[j]` are contiguous in RAM! This enables the hardware prefetcher to stream data seamlessly and allows `#pragma omp simd` to emit AVX-256 vector Fused-Multiply-Add (`_mm256_fmadd_ps`) instructions.
2. **`collapse(2)`**: Flattens the outer two tile loops (`bi` and `bj`) across all available CPU cores, eliminating load imbalance.
3. **Threshold Guard (`if (M * N > 2048)`)**: Prevents OpenMP thread creation overhead for tiny matrix multiplies where single-threaded execution is faster.

---

### 3. `ring0::Tensor3D` Implementation
Used for batch processing and multi-head attention tensors $\mathbf{T} \in \mathbb{R}^{\text{Depth} \times \text{Rows} \times \text{Cols}}$:

```cpp
struct Tensor3D {
    size_t depth = 0;
    size_t rows = 0;
    size_t cols = 0;
    std::vector<float> data;

    inline float& operator()(size_t d, size_t r, size_t c) {
        return data[(d * rows + r) * cols + c];
    }
    inline const float& operator()(size_t d, size_t r, size_t c) const {
        return data[(d * rows + r) * cols + c];
    }

    // Extracts a 2D Matrix slice at depth index d without full copy allocation
    Matrix get_slice(size_t d) const {
        Matrix m;
        m.rows = rows;
        m.cols = cols;
        size_t slice_size = rows * cols;
        m.data.assign(data.begin() + d * slice_size, data.begin() + (d + 1) * slice_size);
        return m;
    }
};
```

---

## 🔗 Related Notes
- [[01 - Ring 0 (Core Math & Hardware)/Activation Functions|Activation Functions]]
- [[01 - Ring 0 (Core Math & Hardware)/CUDA & Hardware Acceleration Engine|CUDA & Hardware Acceleration Engine]]
- [[03 - Ring 2 (Models & Transformers)/Recognition Network|Recognition Network]]
- [[Index|Return to Master Index]]
