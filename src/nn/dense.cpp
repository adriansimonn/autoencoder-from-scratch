#include "nn/dense.h"
#include <cmath>

DenseLayer::DenseLayer(size_t in_features, size_t out_features, InitMethod init)
    : in_features_(in_features), out_features_(out_features),
      b_(1, out_features, 0.0f),
      dW_(in_features, out_features, 0.0f),
      db_(1, out_features, 0.0f) {

    float stddev;
    if (init == InitMethod::He) {
        // He initialization: stddev = sqrt(2 / fan_in)
        stddev = std::sqrt(2.0f / static_cast<float>(in_features));
    } else {
        // Xavier initialization: stddev = sqrt(2 / (fan_in + fan_out))
        stddev = std::sqrt(2.0f / static_cast<float>(in_features + out_features));
    }
    W_ = Tensor::randn(in_features, out_features, 0.0f, stddev);
}

Tensor DenseLayer::forward(const Tensor& input) {
    input_cache_ = input;
    // y = x * W + b
    auto out = Tensor::matmul(input, W_);
    return Tensor::add(out, b_);
}

Tensor DenseLayer::backward(const Tensor& grad_output) {
    // dW = x^T * grad_output
    dW_ = Tensor::matmul(Tensor::transpose(input_cache_), grad_output);

    // db = sum of grad_output over batch (for single sample, just grad_output)
    if (grad_output.rows == 1) {
        db_ = grad_output;
    } else {
        db_ = Tensor::zeros(1, out_features_);
        for (size_t i = 0; i < grad_output.rows; ++i) {
            for (size_t j = 0; j < grad_output.cols; ++j) {
                db_(0, j) += grad_output(i, j);
            }
        }
    }

    // dx = grad_output * W^T
    return Tensor::matmul(grad_output, Tensor::transpose(W_));
}

std::vector<Parameter> DenseLayer::parameters() {
    return {{&W_, &dW_}, {&b_, &db_}};
}
