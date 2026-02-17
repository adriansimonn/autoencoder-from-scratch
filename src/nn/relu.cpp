#include "nn/relu.h"

Tensor ReLU::forward(const Tensor& input) {
    input_cache_ = input;
    Tensor out(input.rows, input.cols);
    for (size_t i = 0; i < input.size(); ++i) {
        out[i] = input[i] > 0.0f ? input[i] : 0.0f;
    }
    return out;
}

Tensor ReLU::backward(const Tensor& grad_output) {
    Tensor grad_input(grad_output.rows, grad_output.cols);
    for (size_t i = 0; i < grad_output.size(); ++i) {
        grad_input[i] = input_cache_[i] > 0.0f ? grad_output[i] : 0.0f;
    }
    return grad_input;
}
