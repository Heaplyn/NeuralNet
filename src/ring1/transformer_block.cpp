#include "ring1/transformer_block.hpp"
#include "ring0/activations.hpp"
#include <cmath>

using namespace std;

namespace ring1 {

// Constructor: Initializes Attention (with GQA), RMSNorm, and SwiGLU Gated FFN weights
TransformerBlock::TransformerBlock(size_t dim, size_t heads, size_t ffn_d, size_t kv_heads)
    : embed_dim(dim),
      ffn_dim(ffn_d),
      num_heads(heads),
      attention(dim, heads, kv_heads),
      ln1_gamma(1, dim),
      ln1_beta(1, dim),
      grad_ln1_gamma(1, dim),
      grad_ln1_beta(1, dim),
      ln2_gamma(1, dim),
      ln2_beta(1, dim),
      grad_ln2_gamma(1, dim),
      grad_ln2_beta(1, dim),
      W_gate(dim, ffn_d),
      b_gate(1, ffn_d),
      W_up(dim, ffn_d),
      b_up(1, ffn_d),
      W_down(ffn_d, dim),
      b_down(1, dim),
      grad_W_gate(dim, ffn_d),
      grad_b_gate(1, ffn_d),
      grad_W_up(dim, ffn_d),
      grad_b_up(1, ffn_d),
      grad_W_down(ffn_d, dim),
      grad_b_down(1, dim) {

    float scale_in = sqrt(2.0f / static_cast<float>(dim));
    float scale_out = sqrt(2.0f / static_cast<float>(ffn_d));

    W_gate = ring0::Matrix::random_normal(dim, ffn_d, 0.0f, scale_in);
    b_gate = ring0::Matrix::zeros(1, ffn_d);

    W_up = ring0::Matrix::random_normal(dim, ffn_d, 0.0f, scale_in);
    b_up = ring0::Matrix::zeros(1, ffn_d);

    W_down = ring0::Matrix::random_normal(ffn_d, dim, 0.0f, scale_out);
    b_down = ring0::Matrix::zeros(1, dim);

    ln1_gamma = ring0::Matrix::ones(1, dim);
    ln1_beta = ring0::Matrix::zeros(1, dim);

    ln2_gamma = ring0::Matrix::ones(1, dim);
    ln2_beta = ring0::Matrix::zeros(1, dim);

    reset_gradients();
}

// Resets gradient accumulators across all sublayers to zero
void TransformerBlock::reset_gradients() {
    attention.reset_gradients();
    grad_ln1_gamma = ring0::Matrix::zeros(1, embed_dim);
    grad_ln1_beta = ring0::Matrix::zeros(1, embed_dim);
    grad_ln2_gamma = ring0::Matrix::zeros(1, embed_dim);
    grad_ln2_beta = ring0::Matrix::zeros(1, embed_dim);

    grad_W_gate = ring0::Matrix::zeros(W_gate.rows, W_gate.cols);
    grad_b_gate = ring0::Matrix::zeros(1, b_gate.cols);

    grad_W_up = ring0::Matrix::zeros(W_up.rows, W_up.cols);
    grad_b_up = ring0::Matrix::zeros(1, b_up.cols);

    grad_W_down = ring0::Matrix::zeros(W_down.rows, W_down.cols);
    grad_b_down = ring0::Matrix::zeros(1, b_down.cols);
}

// Dynamically expands SwiGLU hidden feature dimension as loss reaches lower levels
void TransformerBlock::expand_ffn_dim(size_t additional_dim) {
    if (additional_dim == 0) return;

    size_t new_dim = ffn_dim * additional_dim;
    float scale = 0.01f;

    W_gate.expand_cols(new_dim, scale);
    b_gate.expand_cols(new_dim, 0.0f);

    W_up.expand_cols(new_dim, scale);
    b_up.expand_cols(new_dim, 0.0f);

    W_down.expand_rows(new_dim, scale);

    ffn_dim = new_dim;
    reset_gradients();
}

// Forward pass for full context (Pre-RMSNorm + SwiGLU Gated Architecture)
ring0::Tensor3D TransformerBlock::forward(const ring0::Tensor3D& input) {
    last_input = input;
    size_t B = input.batch_size;
    size_t T = input.seq_len;

    // 1. Pre-RMSNorm 1
    ring0::Matrix in_mat = input.to_matrix();
    ring0::Matrix ln1_mat = in_mat.rms_norm(ln1_gamma);
    last_ln1 = ring0::Tensor3D::from_matrix(ln1_mat, B, T);

    // 2. Causal Grouped-Query Attention
    ring0::Tensor3D attn_out = attention.forward(last_ln1);

    // 3. First Residual Connection: H = input + Attention(RMSNorm1(input))
    ring0::Tensor3D H = input + attn_out;
    last_attn_res = H;

    // 4. Pre-RMSNorm 2
    ring0::Matrix H_mat = H.to_matrix();
    ring0::Matrix ln2_mat = H_mat.rms_norm(ln2_gamma);
    last_ln2 = ring0::Tensor3D::from_matrix(ln2_mat, B, T);

    // 5. SwiGLU Gated MLP: (SiLU(LN2 * W_gate) .* (LN2 * W_up)) * W_down
    last_gate_linear = ln2_mat.matmul(W_gate).add_bias(b_gate);
    last_up_linear = ln2_mat.matmul(W_up).add_bias(b_up);
    last_silu_gate = ring0::Activations::silu(last_gate_linear);
    last_gated = last_silu_gate.elementwise_mul(last_up_linear);

    ring0::Matrix mlp_out_mat = last_gated.matmul(W_down).add_bias(b_down);
    ring0::Tensor3D mlp_out = ring0::Tensor3D::from_matrix(mlp_out_mat, B, T);

    // 6. Second Residual Connection: Out = H + SwiGLU(RMSNorm2(H))
    return H + mlp_out;
}

// O(1) single-step forward pass using KV-Cache for fast generation
ring0::Matrix TransformerBlock::forward_step(const ring0::Matrix& x_token, size_t pos_idx, LayerKVCache& cache) {
    // 1. Pre-RMSNorm 1
    ring0::Matrix ln1_mat = x_token.rms_norm(ln1_gamma);

    // 2. Causal Attention Step with Layer KV-Cache
    ring0::Matrix attn_out = attention.forward_step(ln1_mat, pos_idx, cache);

    // 3. First Residual Connection
    ring0::Matrix H = x_token + attn_out;

    // 4. Pre-RMSNorm 2
    ring0::Matrix ln2_mat = H.rms_norm(ln2_gamma);

    // 5. SwiGLU Gated MLP
    ring0::Matrix gate = ln2_mat.matmul(W_gate).add_bias(b_gate);
    ring0::Matrix up = ln2_mat.matmul(W_up).add_bias(b_up);
    ring0::Matrix silu_gate = ring0::Activations::silu(gate);
    ring0::Matrix gated = silu_gate.elementwise_mul(up);
    ring0::Matrix mlp_out = gated.matmul(W_down).add_bias(b_down);

    // 6. Second Residual Connection
    return H + mlp_out;
}

// Backward pass: Computes SwiGLU, RMSNorm2, Attention, and RMSNorm1 gradients
ring0::Tensor3D TransformerBlock::backward(const ring0::Tensor3D& grad_output) {
    size_t B = grad_output.batch_size;
    size_t T = grad_output.seq_len;

    ring0::Matrix dY_mat = grad_output.to_matrix();

    // 1. Gradient through Down projection: W_down
    grad_W_down += last_gated.transpose().matmul(dY_mat);
    grad_b_down += dY_mat.sum_rows();
    ring0::Matrix dGated(dY_mat.rows, W_down.rows, 0.0f);
    dY_mat.matmul_transB_into(W_down, dGated);

    // 2. Gradient through Bilinear Gate Elementwise Product: gated = silu_gate .* up
    ring0::Matrix dUp = dGated.elementwise_mul(last_silu_gate);
    ring0::Matrix dSiluGate = dGated.elementwise_mul(last_up_linear);

    // Up projection gradients
    ring0::Matrix ln2_flat = last_ln2.to_matrix();
    grad_W_up += ln2_flat.transpose().matmul(dUp);
    grad_b_up += dUp.sum_rows();

    // Gate projection gradients through SiLU derivative
    ring0::Matrix dGate = ring0::Activations::silu_derivative(last_gate_linear, dSiluGate);
    grad_W_gate += ln2_flat.transpose().matmul(dGate);
    grad_b_gate += dGate.sum_rows();

    // Gradient to RMSNorm 2 output with zero-alloc pre-transposed matmuls
    ring0::Matrix dLn2(dGate.rows, W_gate.rows, 0.0f);
    ring0::Matrix dUp_proj(dUp.rows, W_up.rows, 0.0f);
    dGate.matmul_transB_into(W_gate, dLn2);
    dUp.matmul_transB_into(W_up, dUp_proj);
    dLn2 += dUp_proj;

    // Exact analytical RMSNorm 2 backward
    ring0::Matrix dH_mlp = last_attn_res.to_matrix().rms_norm_backward(dLn2, ln2_gamma, grad_ln2_gamma);

    // Gradient accumulated at H (residual branch + MLP branch)
    ring0::Matrix dH_mat = dY_mat + dH_mlp;
    ring0::Tensor3D dH = ring0::Tensor3D::from_matrix(dH_mat, B, T);

    // 3. Backprop through Attention sublayer
    ring0::Tensor3D dAttn_in = attention.backward(dH);

    // Exact analytical RMSNorm 1 backward
    ring0::Matrix dLn1_in = last_input.to_matrix().rms_norm_backward(dAttn_in.to_matrix(), ln1_gamma, grad_ln1_gamma);

    // 4. Total gradient to block input (residual branch + attention branch)
    ring0::Tensor3D dInput = dH + ring0::Tensor3D::from_matrix(dLn1_in, B, T);
    return dInput;
}

} // namespace ring1
