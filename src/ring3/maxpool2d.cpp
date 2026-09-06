#include "ring3/maxpool2d.hpp"
#include <limits>
#include <algorithm>

namespace ring3 {

MaxPool2D::MaxPool2D(size_t k_h, size_t k_w, size_t s_h, size_t s_w, size_t p_h, size_t p_w)
    : kernel_h(k_h), kernel_w(k_w), stride_h(s_h), stride_w(s_w), pad_h(p_h), pad_w(p_w) {}

Tensor4D MaxPool2D::forward(const Tensor4D& input) {
    last_batch_size = input.batch_size;
    last_channels = input.channels;
    last_in_h = input.height;
    last_in_w = input.width;

    size_t H_out = (last_in_h + 2 * pad_h - kernel_h) / stride_h + 1;
    size_t W_out = (last_in_w + 2 * pad_w - kernel_w) / stride_w + 1;

    Tensor4D output(last_batch_size, last_channels, H_out, W_out, 0.0f);
    argmax_indices.assign(last_batch_size * last_channels * H_out * W_out, 0);

    #pragma omp parallel for collapse(2) schedule(static)
    for (int b_idx = 0; b_idx < static_cast<int>(last_batch_size); ++b_idx) {
        for (int c_idx = 0; c_idx < static_cast<int>(last_channels); ++c_idx) {
            size_t b = static_cast<size_t>(b_idx);
            size_t c = static_cast<size_t>(c_idx);

            for (size_t oh = 0; oh < H_out; ++oh) {
                size_t ih_start = oh * stride_h;
                for (size_t ow = 0; ow < W_out; ++ow) {
                    size_t iw_start = ow * stride_w;

                    float max_val = -std::numeric_limits<float>::infinity();
                    size_t best_ih = ih_start;
                    size_t best_iw = iw_start;

                    for (size_t kh = 0; kh < kernel_h; ++kh) {
                        size_t ih = ih_start + kh;
                        if (ih >= pad_h && ih < last_in_h + pad_h) {
                            size_t actual_ih = ih - pad_h;
                            for (size_t kw = 0; kw < kernel_w; ++kw) {
                                size_t iw = iw_start + kw;
                                if (iw >= pad_w && iw < last_in_w + pad_w) {
                                    size_t actual_iw = iw - pad_w;
                                    float val = input(b, c, actual_ih, actual_iw);
                                    if (val > max_val) {
                                        max_val = val;
                                        best_ih = actual_ih;
                                        best_iw = actual_iw;
                                    }
                                }
                            }
                        }
                    }

                    output(b, c, oh, ow) = max_val;
                    size_t out_idx = ((b * last_channels + c) * H_out + oh) * W_out + ow;
                    argmax_indices[out_idx] = best_ih * last_in_w + best_iw;
                }
            }
        }
    }

    return output;
}

Tensor4D MaxPool2D::backward(const Tensor4D& grad_output, float relevancy) {
    size_t H_out = grad_output.height;
    size_t W_out = grad_output.width;

    Tensor4D grad_input(last_batch_size, last_channels, last_in_h, last_in_w, 0.0f);

    #pragma omp parallel for collapse(2) schedule(static)
    for (int b_idx = 0; b_idx < static_cast<int>(last_batch_size); ++b_idx) {
        for (int c_idx = 0; c_idx < static_cast<int>(last_channels); ++c_idx) {
            size_t b = static_cast<size_t>(b_idx);
            size_t c = static_cast<size_t>(c_idx);

            for (size_t oh = 0; oh < H_out; ++oh) {
                for (size_t ow = 0; ow < W_out; ++ow) {
                    size_t out_idx = ((b * last_channels + c) * H_out + oh) * W_out + ow;
                    size_t in_flat = argmax_indices[out_idx];
                    size_t ih = in_flat / last_in_w;
                    size_t iw = in_flat % last_in_w;

                    float g = grad_output(b, c, oh, ow) * relevancy;
                    #pragma omp atomic
                    grad_input(b, c, ih, iw) += g;
                }
            }
        }
    }

    return grad_input;
}

AvgPool2D::AvgPool2D(size_t k_h, size_t k_w, size_t s_h, size_t s_w, size_t p_h, size_t p_w)
    : kernel_h(k_h), kernel_w(k_w), stride_h(s_h), stride_w(s_w), pad_h(p_h), pad_w(p_w) {}

Tensor4D AvgPool2D::forward(const Tensor4D& input) {
    last_batch_size = input.batch_size;
    last_channels = input.channels;
    last_in_h = input.height;
    last_in_w = input.width;

    size_t H_out = (last_in_h + 2 * pad_h - kernel_h) / stride_h + 1;
    size_t W_out = (last_in_w + 2 * pad_w - kernel_w) / stride_w + 1;
    float pool_area = static_cast<float>(kernel_h * kernel_w);

    Tensor4D output(last_batch_size, last_channels, H_out, W_out, 0.0f);

    #pragma omp parallel for collapse(2) schedule(static)
    for (int b_idx = 0; b_idx < static_cast<int>(last_batch_size); ++b_idx) {
        for (int c_idx = 0; c_idx < static_cast<int>(last_channels); ++c_idx) {
            size_t b = static_cast<size_t>(b_idx);
            size_t c = static_cast<size_t>(c_idx);

            for (size_t oh = 0; oh < H_out; ++oh) {
                size_t ih_start = oh * stride_h;
                for (size_t ow = 0; ow < W_out; ++ow) {
                    size_t iw_start = ow * stride_w;
                    float sum = 0.0f;

                    for (size_t kh = 0; kh < kernel_h; ++kh) {
                        size_t ih = ih_start + kh;
                        if (ih >= pad_h && ih < last_in_h + pad_h) {
                            size_t actual_ih = ih - pad_h;
                            for (size_t kw = 0; kw < kernel_w; ++kw) {
                                size_t iw = iw_start + kw;
                                if (iw >= pad_w && iw < last_in_w + pad_w) {
                                    sum += input(b, c, actual_ih, iw - pad_w);
                                }
                            }
                        }
                    }
                    output(b, c, oh, ow) = sum / pool_area;
                }
            }
        }
    }

    return output;
}

Tensor4D AvgPool2D::backward(const Tensor4D& grad_output, float relevancy) {
    size_t H_out = grad_output.height;
    size_t W_out = grad_output.width;
    float pool_area = static_cast<float>(kernel_h * kernel_w);
    float scale = relevancy / pool_area;

    Tensor4D grad_input(last_batch_size, last_channels, last_in_h, last_in_w, 0.0f);

    #pragma omp parallel for collapse(2) schedule(static)
    for (int b_idx = 0; b_idx < static_cast<int>(last_batch_size); ++b_idx) {
        for (int c_idx = 0; c_idx < static_cast<int>(last_channels); ++c_idx) {
            size_t b = static_cast<size_t>(b_idx);
            size_t c = static_cast<size_t>(c_idx);

            for (size_t oh = 0; oh < H_out; ++oh) {
                size_t ih_start = oh * stride_h;
                for (size_t ow = 0; ow < W_out; ++ow) {
                    size_t iw_start = ow * stride_w;
                    float g = grad_output(b, c, oh, ow) * scale;

                    for (size_t kh = 0; kh < kernel_h; ++kh) {
                        size_t ih = ih_start + kh;
                        if (ih >= pad_h && ih < last_in_h + pad_h) {
                            size_t actual_ih = ih - pad_h;
                            for (size_t kw = 0; kw < kernel_w; ++kw) {
                                size_t iw = iw_start + kw;
                                if (iw >= pad_w && iw < last_in_w + pad_w) {
                                    #pragma omp atomic
                                    grad_input(b, c, actual_ih, iw - pad_w) += g;
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    return grad_input;
}

} // namespace ring3
