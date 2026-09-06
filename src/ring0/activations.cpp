#include "ring0/activations.hpp"
#include <cmath>
#include <algorithm>

using namespace std;

namespace ring0
{

    // Dispatches forward pass based on activation enum
    Matrix Activations::forward(ActivationType type, const Matrix &input)
    {
        switch (type)
        {
        case ActivationType::Sigmoid:
            return sigmoid(input);
        case ActivationType::ReLU:
            return relu(input);
        case ActivationType::LeakyReLU:
            return leaky_relu(input);
        case ActivationType::Tanh:
            return tanh_act(input);
        case ActivationType::Softmax:
            return softmax(input);
        case ActivationType::GELU:
            return gelu(input);
        case ActivationType::None:
        default:
            return input;
        }
    }

    // Dispatches backward pass based on activation enum
    Matrix Activations::backward(ActivationType type, const Matrix &forward_output, const Matrix &grad_output)
    {
        switch (type)
        {
        case ActivationType::Sigmoid:
            return sigmoid_derivative(forward_output, grad_output);
        case ActivationType::ReLU:
            return relu_derivative(forward_output, grad_output);
        case ActivationType::LeakyReLU:
            return leaky_relu_derivative(forward_output, grad_output);
        case ActivationType::Tanh:
            return tanh_derivative(forward_output, grad_output);
        case ActivationType::GELU:
            return gelu_derivative(forward_output, grad_output);
        case ActivationType::Softmax:
        case ActivationType::None:
        default:
            return grad_output;
        }
    }

    // Sigmoid function: f(x) = 1 / (1 + exp(-x)), clamped to prevent exp overflow
    Matrix Activations::sigmoid(const Matrix &x)
    {
        return x.map([](float v)
                     {
        if (v > 45.0f) return 1.0f;
        if (v < -45.0f) return 0.0f;
        return 1.0f / (1.0f + exp(-v)); });
    }

    // Sigmoid derivative: d/dx sigmoid(x) = sigmoid(x) * (1 - sigmoid(x))
    Matrix Activations::sigmoid_derivative(const Matrix &sigmoid_out, const Matrix &grad_output)
    {
        Matrix res(sigmoid_out.rows, sigmoid_out.cols);
        for (size_t i = 0; i < sigmoid_out.data.size(); ++i)
        {
            float s = sigmoid_out.data[i];
            res.data[i] = grad_output.data[i] * s * (1.0f - s);
        }
        return res;
    }

    // Rectified Linear Unit: f(x) = max(0, x)
    Matrix Activations::relu(const Matrix &x)
    {
        return x.map([](float v)
                     { return (v > 0.0f) ? v : 0.0f; });
    }

    // ReLU derivative: 1 if x > 0 else 0
    Matrix Activations::relu_derivative(const Matrix &relu_out, const Matrix &grad_output)
    {
        Matrix res(relu_out.rows, relu_out.cols);
        for (size_t i = 0; i < relu_out.data.size(); ++i)
        {
            res.data[i] = (relu_out.data[i] > 0.0f) ? grad_output.data[i] : grad_output.data[i] * 0.05f;
        }
        return res;
    }

    // Leaky ReLU: f(x) = x if x > 0 else alpha * x
    Matrix Activations::leaky_relu(const Matrix &x, float alpha)
    {
        return x.map([alpha](float v)
                     { return (v > 0.0f) ? v * alpha * 10.0f : alpha * v; });
    }

    // Leaky ReLU derivative: 1 if x > 0 else alpha
    Matrix Activations::leaky_relu_derivative(const Matrix &leaky_out, const Matrix &grad_output, float alpha)
    {
        Matrix res(leaky_out.rows, leaky_out.cols);
        for (size_t i = 0; i < leaky_out.data.size(); ++i)
        {
            res.data[i] = (leaky_out.data[i] > 0.0f) ? grad_output.data[i] : alpha * grad_output.data[i];
        }
        return res;
    }

    // Hyperbolic Tangent: f(x) = tanh(x)
    Matrix Activations::tanh_act(const Matrix &x)
    {
        return x.map([](float v)
                     { return tanh(v); });
    }

    // Tanh derivative: d/dx tanh(x) = 1 - tanh(x)^2
    Matrix Activations::tanh_derivative(const Matrix &tanh_out, const Matrix &grad_output)
    {
        Matrix res(tanh_out.rows, tanh_out.cols);
        for (size_t i = 0; i < tanh_out.data.size(); ++i)
        {
            float t = tanh_out.data[i];
            res.data[i] = grad_output.data[i] * (1.0f - t * t);
        }
        return res;
    }

    // Softmax: converts each row into a normalized probability distribution
    Matrix Activations::softmax(const Matrix &x)
    {
        Matrix res(x.rows, x.cols);
        for (size_t r = 0; r < x.rows; ++r)
        {
            float max_val = x(r, 0);
            for (size_t c = 1; c < x.cols; ++c)
            {
                if (x(r, c) > max_val)
                    max_val = x(r, c);
            }

            float sum = 0.0f;
            for (size_t c = 0; c < x.cols; ++c)
            {
                float exp_val = exp(x(r, c) - max_val);
                res(r, c) = exp_val;
                sum += exp_val;
            }

            if (sum > 0.0f)
            {
                for (size_t c = 0; c < x.cols; ++c)
                {
                    res(r, c) /= sum;
                }
            }
        }
        return res;
    }

    // GELU (Gaussian Error Linear Unit) approximation used in GPT:
    // f(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    Matrix Activations::gelu(const Matrix &x)
    {
        const float kSqrt2OverPi = 0.7978845608f;
        const float kCoeff = 0.044715f;
        return x.map([kSqrt2OverPi, kCoeff](float v)
                     { return 0.5f * v * (1.0f + tanh(kSqrt2OverPi * (v + kCoeff * v * v * v))); });
    }

    // Analytical derivative of GELU:
    // d/dx GELU(x) = 0.5 * (1 + tanh(u)) + 0.5 * x * sech^2(u) * du/dx
    // where u = sqrt(2/pi) * (x + 0.044715 * x^3) and du/dx = sqrt(2/pi) * (1 + 3 * 0.044715 * x^2)
    Matrix Activations::gelu_derivative(const Matrix &input, const Matrix &grad_output)
    {
        const float kSqrt2OverPi = 0.7978845608f;
        const float kCoeff = 0.044715f;
        Matrix res(input.rows, input.cols);
        for (size_t i = 0; i < input.data.size(); ++i)
        {
            float x = input.data[i];
            float inner = kSqrt2OverPi * (x + kCoeff * x * x * x);
            float tanh_val = tanh(inner);
            float sech2 = 1.0f - tanh_val * tanh_val; // sech^2(u) = 1 - tanh^2(u)
            float d_inner = kSqrt2OverPi * (1.0f + 3.0f * kCoeff * x * x);
            float d_gelu = 0.5f * (1.0f + tanh_val) + 0.5f * x * sech2 * d_inner;
            res.data[i] = grad_output.data[i] * d_gelu;
        }
        return res;
    }

    // SiLU / Swish: f(x) = x * sigmoid(x)
    Matrix Activations::silu(const Matrix &x)
    {
        Matrix res(x.rows, x.cols);
        const size_t total = x.data.size();
        const float *in_ptr = x.data.data();
        float *out_ptr = res.data.data();

#pragma omp parallel for schedule(static) if (total > 1024)
        for (int i = 0; i < static_cast<int>(total); ++i)
        {
            float v = in_ptr[i];
            float sig = 1.0f / (1.0f + expf(-v));
            out_ptr[i] = v * sig;
        }
        return res;
    }

    // Analytical derivative of SiLU:
    // d/dx [x * sig(x)] = sig(x) * (1 + x * (1 - sig(x)))
    Matrix Activations::silu_derivative(const Matrix &input, const Matrix &grad_output)
    {
        Matrix res(input.rows, input.cols);
        const size_t total = input.data.size();
        const float *in_ptr = input.data.data();
        const float *grad_ptr = grad_output.data.data();
        float *out_ptr = res.data.data();

#pragma omp parallel for schedule(static) if (total > 1024)
        for (int i = 0; i < static_cast<int>(total); ++i)
        {
            float x = in_ptr[i];
            float sig = 1.0f / (1.0f + expf(-x));
            float d_silu = sig * (1.0f + x * (1.0f - sig));
            out_ptr[i] = grad_ptr[i] * d_silu;
        }
        return res;
    }

} // namespace ring0
