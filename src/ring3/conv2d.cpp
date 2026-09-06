#include "ring3/conv2d.hpp"
#include <algorithm>
#include <cmath>

namespace ring3 {

static inline float apply_activation_scalar(float x, ring0::ActivationType act) {
    switch (act) {
        case ring0::ActivationType::ReLU: return x > 0.0f ? x : 0.0f;
        case ring0::ActivationType::LeakyReLU: return x > 0.0f ? x : 0.01f * x;
        case ring0::ActivationType::Sigmoid: return 1.0f / (1.0f + std::exp(-x));
        case ring0::ActivationType::Tanh: return std::tanh(x);
        case ring0::ActivationType::SiLU: return x / (1.0f + std::exp(-x));
        case ring0::ActivationType::GELU: {
            float k = 0.7978845608f; // sqrt(2 / pi)
            return 0.5f * x * (1.0f + std::tanh(k * (x + 0.044715f * x * x * x)));
        }
        case ring0::ActivationType::None:
        default: return x;
    }
}

static inline float derivative_activation_scalar(float x, ring0::ActivationType act) {
    switch (act) {
        case ring0::ActivationType::ReLU: return x > 0.0f ? 1.0f : 0.0f;
        case ring0::ActivationType::LeakyReLU: return x > 0.0f ? 1.0f : 0.01f;
        case ring0::ActivationType::Sigmoid: {
            float s = 1.0f / (1.0f + std::exp(-x));
            return s * (1.0f - s);
        }
        case ring0::ActivationType::Tanh: {
            float t = std::tanh(x);
            return 1.0f - t * t;
        }
        case ring0::ActivationType::SiLU: {
            float s = 1.0f / (1.0f + std::exp(-x));
            return s * (1.0f + x * (1.0f - s));
        }
        case ring0::ActivationType::GELU: {
            float k = 0.7978845608f;
            float u = k * (x + 0.044715f * x * x * x);
            float tanh_u = std::tanh(u);
            float sech2_u = 1.0f - tanh_u * tanh_u;
            float du_dx = k * (1.0f + 3.0f * 0.044715f * x * x);
            return 0.5f * (1.0f + tanh_u) + 0.5f * x * sech2_u * du_dx;
        }
        case ring0::ActivationType::None:
        default: return 1.0f;
    }
}

Conv2D::Conv2D(size_t in_c, size_t out_c, size_t k_h, size_t k_w,
               size_t s_h, size_t s_w, size_t p_h, size_t p_w,
               ring0::ActivationType act)
    : in_channels(in_c), out_channels(out_c),
      kernel_h(k_h), kernel_w(k_w),
      stride_h(s_h), stride_w(s_w),
      pad_h(p_h), pad_w(p_w),
      activation(act),
      weights(Tensor4D::he(out_c, in_c, k_h, k_w)),
      biases(out_c, 0.0f),
      grad_weights(out_c, in_c, k_h, k_w, 0.0f),
      grad_biases(out_c, 0.0f) {}

Tensor4D Conv2D::forward(const Tensor4D& input) {
    last_input = input;
    size_t B = input.batch_size;
    size_t H_in = input.height;
    size_t W_in = input.width;

    size_t H_out = (H_in + 2 * pad_h - kernel_h) / stride_h + 1;
    size_t W_out = (W_in + 2 * pad_w - kernel_w) / stride_w + 1;

    Tensor4D padded_input = input.pad(pad_h, pad_w);
    Tensor4D pre_act(B, out_channels, H_out, W_out, 0.0f);
    Tensor4D output(B, out_channels, H_out, W_out, 0.0f);

    #pragma omp parallel for schedule(static)
    for (int b_idx = 0; b_idx < static_cast<int>(B); ++b_idx) {
        for (int oc_idx = 0; oc_idx < static_cast<int>(out_channels); ++oc_idx) {
            size_t b = static_cast<size_t>(b_idx);
            size_t oc = static_cast<size_t>(oc_idx);
            float b_val = biases[oc];

            for (size_t oh = 0; oh < H_out; ++oh) {
                size_t ih_start = oh * stride_h;
                for (size_t ow = 0; ow < W_out; ++ow) {
                    size_t iw_start = ow * stride_w;
                    float sum = b_val;

                    for (size_t ic = 0; ic < in_channels; ++ic) {
                        for (size_t kh = 0; kh < kernel_h; ++kh) {
                            for (size_t kw = 0; kw < kernel_w; ++kw) {
                                float x_val = padded_input(b, ic, ih_start + kh, iw_start + kw);
                                float w_val = weights(oc, ic, kh, kw);
                                sum += x_val * w_val;
                            }
                        }
                    }
                    pre_act(b, oc, oh, ow) = sum;
                    output(b, oc, oh, ow) = apply_activation_scalar(sum, activation);
                }
            }
        }
    }

    last_pre_act = pre_act;
    last_output = output;
    return output;
}

Tensor4D Conv2D::backward(const Tensor4D& grad_output, float relevancy) {
    size_t B = last_input.batch_size;
    size_t H_in = last_input.height;
    size_t W_in = last_input.width;

    size_t H_out = last_output.height;
    size_t W_out = last_output.width;

    Tensor4D grad_pre(B, out_channels, H_out, W_out, 0.0f);
    for (size_t i = 0; i < grad_pre.data.size(); ++i) {
        float d_out = grad_output.data[i] * relevancy;
        float z = last_pre_act.data[i];
        float d_act = derivative_activation_scalar(z, activation);
        grad_pre.data[i] = d_out * d_act;
    }

    // 1. Compute bias gradients
    for (size_t oc = 0; oc < out_channels; ++oc) {
        float b_sum = 0.0f;
        for (size_t b = 0; b < B; ++b) {
            for (size_t oh = 0; oh < H_out; ++oh) {
                for (size_t ow = 0; ow < W_out; ++ow) {
                    b_sum += grad_pre(b, oc, oh, ow);
                }
            }
        }
        grad_biases[oc] += b_sum;
    }

    // 2. Compute weight gradients
    Tensor4D padded_input = last_input.pad(pad_h, pad_w);
    #pragma omp parallel for schedule(static)
    for (int oc_idx = 0; oc_idx < static_cast<int>(out_channels); ++oc_idx) {
        for (int ic_idx = 0; ic_idx < static_cast<int>(in_channels); ++ic_idx) {
            size_t oc = static_cast<size_t>(oc_idx);
            size_t ic = static_cast<size_t>(ic_idx);

            for (size_t kh = 0; kh < kernel_h; ++kh) {
                for (size_t kw = 0; kw < kernel_w; ++kw) {
                    float w_sum = 0.0f;
                    for (size_t b = 0; b < B; ++b) {
                        for (size_t oh = 0; oh < H_out; ++oh) {
                            size_t ih = oh * stride_h + kh;
                            for (size_t ow = 0; ow < W_out; ++ow) {
                                size_t iw = ow * stride_w + kw;
                                w_sum += grad_pre(b, oc, oh, ow) * padded_input(b, ic, ih, iw);
                            }
                        }
                    }
                    grad_weights(oc, ic, kh, kw) += w_sum;
                }
            }
        }
    }

    // 3. Compute input gradients dX
    size_t padded_h = H_in + 2 * pad_h;
    size_t padded_w = W_in + 2 * pad_w;
    Tensor4D grad_padded(B, in_channels, padded_h, padded_w, 0.0f);

    #pragma omp parallel for schedule(static)
    for (int b_idx = 0; b_idx < static_cast<int>(B); ++b_idx) {
        for (int ic_idx = 0; ic_idx < static_cast<int>(in_channels); ++ic_idx) {
            size_t b = static_cast<size_t>(b_idx);
            size_t ic = static_cast<size_t>(ic_idx);

            for (size_t oc = 0; oc < out_channels; ++oc) {
                for (size_t oh = 0; oh < H_out; ++oh) {
                    size_t ih_start = oh * stride_h;
                    for (size_t ow = 0; ow < W_out; ++ow) {
                        size_t iw_start = ow * stride_w;
                        float gp = grad_pre(b, oc, oh, ow);

                        for (size_t kh = 0; kh < kernel_h; ++kh) {
                            for (size_t kw = 0; kw < kernel_w; ++kw) {
                                grad_padded(b, ic, ih_start + kh, iw_start + kw) += gp * weights(oc, ic, kh, kw);
                            }
                        }
                    }
                }
            }
        }
    }

    // Un-pad input gradient
    Tensor4D grad_input(B, in_channels, H_in, W_in, 0.0f);
    for (size_t b = 0; b < B; ++b) {
        for (size_t ic = 0; ic < in_channels; ++ic) {
            for (size_t h = 0; h < H_in; ++h) {
                for (size_t w = 0; w < W_in; ++w) {
                    grad_input(b, ic, h, w) = grad_padded(b, ic, h + pad_h, w + pad_w);
                }
            }
        }
    }

    return grad_input;
}

void Conv2D::expand_filters(size_t additional_out_channels) {
    if (additional_out_channels == 0) return;
    size_t new_out_c = out_channels + additional_out_channels;

    Tensor4D new_weights(new_out_c, in_channels, kernel_h, kernel_w, 0.0f);
    Tensor4D new_grad_weights(new_out_c, in_channels, kernel_h, kernel_w, 0.0f);
    std::vector<float> new_biases(new_out_c, 0.0f);
    std::vector<float> new_grad_biases(new_out_c, 0.0f);

    // Copy existing weights and biases
    for (size_t oc = 0; oc < out_channels; ++oc) {
        new_biases[oc] = biases[oc];
        new_grad_biases[oc] = grad_biases[oc];
        for (size_t ic = 0; ic < in_channels; ++ic) {
            for (size_t kh = 0; kh < kernel_h; ++kh) {
                for (size_t kw = 0; kw < kernel_w; ++kw) {
                    new_weights(oc, ic, kh, kw) = weights(oc, ic, kh, kw);
                    new_grad_weights(oc, ic, kh, kw) = grad_weights(oc, ic, kh, kw);
                }
            }
        }
    }

    // Initialize newly added filters with He initialization
    Tensor4D added_w = Tensor4D::he(additional_out_channels, in_channels, kernel_h, kernel_w);
    for (size_t oc = 0; oc < additional_out_channels; ++oc) {
        for (size_t ic = 0; ic < in_channels; ++ic) {
            for (size_t kh = 0; kh < kernel_h; ++kh) {
                for (size_t kw = 0; kw < kernel_w; ++kw) {
                    new_weights(out_channels + oc, ic, kh, kw) = added_w(oc, ic, kh, kw);
                }
            }
        }
    }

    out_channels = new_out_c;
    weights = new_weights;
    biases = new_biases;
    grad_weights = new_grad_weights;
    grad_biases = new_grad_biases;
}

void Conv2D::expand_input_channels(size_t additional_in_channels) {
    if (additional_in_channels == 0) return;
    size_t new_in_c = in_channels + additional_in_channels;

    Tensor4D new_weights(out_channels, new_in_c, kernel_h, kernel_w, 0.0f);
    Tensor4D new_grad_weights(out_channels, new_in_c, kernel_h, kernel_w, 0.0f);

    for (size_t oc = 0; oc < out_channels; ++oc) {
        for (size_t ic = 0; ic < in_channels; ++ic) {
            for (size_t kh = 0; kh < kernel_h; ++kh) {
                for (size_t kw = 0; kw < kernel_w; ++kw) {
                    new_weights(oc, ic, kh, kw) = weights(oc, ic, kh, kw);
                    new_grad_weights(oc, ic, kh, kw) = grad_weights(oc, ic, kh, kw);
                }
            }
        }
    }

    // Initialize new input channels with He scaling
    Tensor4D added_w = Tensor4D::he(out_channels, additional_in_channels, kernel_h, kernel_w);
    for (size_t oc = 0; oc < out_channels; ++oc) {
        for (size_t ic = 0; ic < additional_in_channels; ++ic) {
            for (size_t kh = 0; kh < kernel_h; ++kh) {
                for (size_t kw = 0; kw < kernel_w; ++kw) {
                    new_weights(oc, in_channels + ic, kh, kw) = added_w(oc, ic, kh, kw);
                }
            }
        }
    }

    in_channels = new_in_c;
    weights = new_weights;
    grad_weights = new_grad_weights;
}

void Conv2D::reset_gradients() {
    grad_weights.zero();
    std::fill(grad_biases.begin(), grad_biases.end(), 0.0f);
}

} // namespace ring3
