#include "nn/sigmoid.h"
#include <cmath>
#include <algorithm>

Tensor Sigmoid::forward(const Tensor& input) {
    output_cache_ = Tensor(input.rows, input.cols);
    for (size_t i = 0; i < input.size(); ++i) {
        // Clamp to [-88, 88] for numerical stability
        float x = std::clamp(input[i], -88.0f, 88.0f);
        output_cache_[i] = 1.0f / (1.0f + std::exp(-x));
    }
    return output_cache_;
}

Tensor Sigmoid::backward(const Tensor& grad_output) {
    Tensor grad_input(grad_output.rows, grad_output.cols);
    for (size_t i = 0; i < grad_output.size(); ++i) {
        float s = output_cache_[i];
        grad_input[i] = grad_output[i] * s * (1.0f - s);
    }
    return grad_input;
}
